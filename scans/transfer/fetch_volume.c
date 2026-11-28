#define _GNU_SOURCE
#include <curl/curl.h>
#include <errno.h>
#include <json-c/json.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "../../storage/brick_store.c"
#include "../chunks/bucket_listing.h"

typedef struct {
    unsigned char *data;
    size_t size;
    size_t capacity;
    long long first, last, total;
} Buffer;

typedef struct {
    int64_t z, y, x, start, end;
} Key;

typedef struct {
    HxBrickStore *store;
    const char *base;
    const Key *keys;
    size_t count;
    atomic_size_t next;
    atomic_size_t done;
    atomic_size_t bytes;
    atomic_int failures;
    uint64_t space_floor;
    atomic_int low_space;
    pthread_mutex_t write_lock;
} Transfer;

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

static size_t receive(void *data, size_t width, size_t count, void *context) {
    Buffer *buffer = context;
    if (width && count > SIZE_MAX / width)
        return 0;
    size_t bytes = width * count;
    if (bytes > SIZE_MAX - buffer->size)
        return 0;
    size_t need = buffer->size + bytes;
    if (need > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 65536;
        while (capacity < need && capacity <= SIZE_MAX / 2)
            capacity *= 2;
        if (capacity < need)
            capacity = need;
        unsigned char *larger = realloc(buffer->data, capacity);
        if (!larger)
            return 0;
        buffer->data = larger;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->size, data, bytes);
    buffer->size += bytes;
    return bytes;
}

static size_t response_header(char *data, size_t width, size_t count, void *context) {
    Buffer *buffer = context;
    size_t bytes = width * count;
    const char name[] = "Content-Range: bytes ";
    if (bytes >= 5 && !memcmp(data, "HTTP/", 5)) {
        buffer->total = 0;
    }
    if (bytes > sizeof(name) && bytes < 128 && !strncasecmp(data, name, sizeof(name) - 1)) {
        char line[128];
        size_t n = bytes - (sizeof(name) - 1);
        memcpy(line, data + sizeof(name) - 1, n);
        line[n] = 0;
        char *end;
        errno = 0;
        buffer->total = 0;
        buffer->first = strtoll(line, &end, 10);
        if (end != line && *end == '-') {
            char *next = end + 1;
            buffer->last = strtoll(next, &end, 10);
            if (end != next && *end == '/') {
                next = end + 1;
                long long total = strtoll(next, &end, 10);
                if (!errno && end != next && *end == '\r') {
                    buffer->total = total;
                }
            }
        }
    }
    return bytes;
}

static int get_url(CURL *curl, const char *url, const char *range, Buffer *buffer, long *status) {
    buffer->size = 0;
    buffer->total = 0;
    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 180L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, receive);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, buffer);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "HerculeX-native/1");
    if (range) {
        curl_easy_setopt(curl, CURLOPT_RANGE, range);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, response_header);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, buffer);
    }
    CURLcode result = curl_easy_perform(curl);
    if (result == CURLE_OK)
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, status);
    return result == CURLE_OK ? 0 : -1;
}

static int acceptable(const Key *key, const Buffer *body, long code, int64_t brick_size) {
    if (code == 403 || code == 404) {
        return 1;
    }
    if (key->end < 0) {
        return code == 200;
    }
    if (code == 200) {
        return body->size == (uint64_t)brick_size;
    }
    return code == 206 && body->first == key->start && body->last == key->end &&
        body->total == brick_size && body->size == (uint64_t)(key->end - key->start + 1);
}

static int room_to_fetch(Transfer *transfer) {
    struct statvfs info;
    if (!transfer->space_floor || fstatvfs(transfer->store->data_fd, &info)) {
        return 1;
    }
    double room = (double)info.f_frsize * info.f_bavail;
    if (room >= transfer->space_floor) {
        return 1;
    }
    fprintf(stdout, "%s: %.0f GiB free\n", transfer->store->folder, room / 0x1p30);
    atomic_store(&transfer->low_space, 1);
    return 0;
}

static void *fetch_worker(void *context) {
    Transfer *transfer = context;
    CURL *curl = curl_easy_init();
    Buffer body = {0};
    while (curl && !atomic_load(&transfer->low_space)) {
        size_t index = atomic_fetch_add(&transfer->next, 1);
        if (index >= transfer->count)
            break;
        Key key = transfer->keys[index];
        char *url = NULL;
        if (asprintf(&url, "%s/%lld/%lld/%lld", transfer->base, (long long)key.z, (long long)key.y,
                (long long)key.x) < 0) {
            atomic_fetch_add(&transfer->failures, 1);
            continue;
        }
        long status = 0;
        int fetched = -1;
        char range[64];
        if (key.end >= 0) {
            snprintf(range, sizeof(range), "%lld-%lld", (long long)key.start, (long long)key.end);
        }
        for (int attempt = 0; attempt < 4; attempt++) {
            if (!get_url(curl, url, key.end >= 0 ? range : NULL, &body, &status) &&
                acceptable(&key, &body, status, transfer->store->bytes)) {
                fetched = 0;
                break;
            }
        }
        int stored = -1;
        if (!fetched) {
            pthread_mutex_lock(&transfer->write_lock);
            int present = status == 200 || status == 206;
            if (!atomic_load(&transfer->low_space)) {
                stored = hx_brick_put_range(transfer->store, key.z, key.y, key.x,
                    present ? body.data : NULL, present ? body.size : 0,
                    status == 206 ? key.start : 0);
                if (!stored) {
                    size_t done = atomic_fetch_add(&transfer->done, 1) + 1;
                    if (present) {
                        atomic_fetch_add(&transfer->bytes, body.size);
                    }
                    if (done == 1 || done % 200 == 0) {
                        room_to_fetch(transfer);
                    }
                    if (done % 200 == 0) {
                        stored = hx_brick_save(transfer->store);
                        fprintf(stdout, "%zu/%zu bricks\n", done, transfer->count);
                        fflush(stdout);
                    }
                }
            } else {
                stored = 0;
            }
            pthread_mutex_unlock(&transfer->write_lock);
        }
        if (fetched || stored) {
            fprintf(stderr, "HerculeX: failed CT brick %lld/%lld/%lld%s%ld\n", (long long)key.z,
                (long long)key.y, (long long)key.x, status ? ": HTTP " : "", status);
            atomic_fetch_add(&transfer->failures, 1);
        }
        free(url);
    }
    free(body.data);
    if (curl)
        curl_easy_cleanup(curl);
    return NULL;
}

static int transfer_keys(Transfer *transfer, int connections) {
    if (transfer->count && !room_to_fetch(transfer)) {
        return 1;
    }
    int workers = connections < 1 ? 1 : connections > 64 ? 64 : connections;
    if ((size_t)workers > transfer->count) {
        workers = (int)transfer->count;
    }
    pthread_t threads[64];
    int started = 0;
    for (; started < workers; started++) {
        if (pthread_create(&threads[started], NULL, fetch_worker, transfer)) {
            break;
        }
    }
    for (int i = 0; i < started; i++) {
        pthread_join(threads[i], NULL);
    }
    int saved = hx_brick_save(transfer->store);
    if (!saved && atomic_load(&transfer->low_space)) {
        return 1;
    }
    return saved || started != workers || atomic_load(&transfer->done) != transfer->count ||
            atomic_load(&transfer->failures)
        ? -1
        : 0;
}

static json_object *array_metadata(const char *base) {
    char *url = NULL;
    if (asprintf(&url, "%s/.zarray", base) < 0)
        return NULL;
    CURL *curl = curl_easy_init();
    Buffer body = {0};
    long status = 0;
    int ok = curl && !get_url(curl, url, NULL, &body, &status) && status == 200 &&
        body.size <= INT32_MAX;
    json_tokener *parser = ok ? json_tokener_new() : NULL;
    json_object *root =
        parser ? json_tokener_parse_ex(parser, (const char *)body.data, (int)body.size) : NULL;
    if (parser && json_tokener_get_error(parser) != json_tokener_success) {
        json_object_put(root);
        root = NULL;
    }
    if (parser)
        json_tokener_free(parser);
    free(body.data);
    if (curl)
        curl_easy_cleanup(curl);
    free(url);
    return root;
}

static int supported(json_object *meta, int64_t *shape, int64_t *chunks) {
    json_object *dtype, *compressor;
    return meta && !json_vector(meta, "shape", shape) && !json_vector(meta, "chunks", chunks) &&
        json_object_object_get_ex(meta, "dtype", &dtype) &&
        json_object_is_type(dtype, json_type_string) &&
        !strcmp(json_object_get_string(dtype), "|u1") &&
        json_object_object_get_ex(meta, "compressor", &compressor) &&
        json_object_is_type(compressor, json_type_null);
}

static char *trimmed_path(const char *path) {
    while (*path == '/')
        path++;
    size_t length = strlen(path);
    while (length && path[length - 1] == '/')
        length--;
    return strndup(path, length);
}

const char hx_usage_fetch_volume[] =
    "Stream one OME-Zarr level of a scan into a native brick store.";
static const char bucket[] = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";

typedef struct {
    const HxBrickStore *store;
    const char *root, *prefix;
    int64_t first, cells;
    uint8_t *present;
    atomic_int failed;
} ListedBricks;

static void list_brick_row(void *context, size_t index) {
    ListedBricks *job = context;
    int64_t z = job->first + (int64_t)index;
    const int64_t *offsets = job->store->offsets + z * job->cells;
    int64_t i = 0;
    while (i < job->cells && offsets[i] != HX_BRICK_UNKNOWN) {
        i++;
    }
    if (i == job->cells || atomic_load(&job->failed)) {
        return;
    }
    char *prefix = NULL, error[128] = {0};
    CURL *curl = curl_easy_init();
    InventoryListing listing = {.keys = json_object_new_array()};
    int complete = curl && listing.keys &&
        asprintf(&prefix, "%s/%lld/", job->prefix, (long long)z) >= 0 &&
        inventory_listing(curl, job->root, prefix, 0, 64, &listing, error) && listing.complete;
    if (complete) {
        size_t width = strlen(prefix);
        int64_t ny = job->store->grid[1], nx = job->store->grid[2];
        uint8_t *row = job->present + index * job->cells;
        for (size_t j = 0; j < json_object_array_length(listing.keys); j++) {
            json_object *item = json_object_array_get_idx(listing.keys, j);
            const char *key = json_object_get_string(json_object_object_get(item, "key"));
            if (!key || strncmp(key, prefix, width) != 0) {
                continue;
            }
            const char *start = key + width;
            char *end;
            int64_t y = strtoll(start, &end, 10);
            if (*start < '0' || *start > '9' || *end != '/' || y < 0 || y >= ny) {
                continue;
            }
            start = end + 1;
            int64_t x = strtoll(start, &end, 10);
            if (*start >= '0' && *start <= '9' && !*end && x >= 0 && x < nx) {
                row[y * nx + x] = 1;
            }
        }
    } else {
        atomic_store(&job->failed, 1);
    }
    json_object_put(listing.keys);
    if (curl) {
        curl_easy_cleanup(curl);
    }
    free(prefix);
}

static int fetch_level(const char *path, int level, const char *out, const char *root,
    int64_t first, int64_t last, int connections, int list) {
    char *clean = trimmed_path(path), *base = NULL, *source = NULL;
    int made = clean && asprintf(&base, "%s/%s/%d", root, clean, level) >= 0 &&
        asprintf(&source, "%s/%d", path, level) >= 0;
    json_object *meta = made ? array_metadata(base) : NULL;
    int64_t shape[3], chunks[3];
    if (!supported(meta, shape, chunks)) {
        fprintf(
            stderr, "HerculeX: unsupported or unavailable OME-Zarr array %s\n", base ? base : path);
        if (meta)
            json_object_put(meta);
        free(source);
        free(base);
        free(clean);
        return 1;
    }
    HxBrickStore store;
    int failed = hx_brick_open(&store, out, shape, chunks, source);
    if (failed) {
        fprintf(stderr, "HerculeX: cannot open brick store %s: %s\n", out, strerror(errno));
        goto finish;
    }
    if (first < 0)
        first = 0;
    if (last < 0 || last > store.grid[0])
        last = store.grid[0];
    if (first > last)
        first = last;
    int64_t possible = (last - first) * store.grid[1] * store.grid[2];
    Key *keys = calloc((size_t)(possible ? possible : 1), sizeof(*keys));
    size_t count = 0;
    if (!keys) {
        failed = 1;
        goto close;
    }
    uint8_t *published = list && possible ? calloc((size_t)possible, 1) : NULL;
    int64_t cells = store.grid[1] * store.grid[2];
    if (published) {
        ListedBricks job = {&store, root, base + strlen(root) + 1, first, cells, published, 0};
        inventory_parallel(list_brick_row, &job, (size_t)(last - first), 12);
        if (atomic_load(&job.failed)) {
            puts("brick listing incomplete, checking individual bricks");
            free(published);
            published = NULL;
        }
    }
    size_t skipped = 0;
    for (int64_t z = first; z < last; z++) {
        for (int64_t y = 0; y < store.grid[1]; y++) {
            for (int64_t x = 0; x < store.grid[2]; x++) {
                int64_t cell;
                if (hx_brick_cell(&store, z, y, x, &cell) ||
                    store.offsets[cell] != HX_BRICK_UNKNOWN) {
                    continue;
                }
                if (published && !published[cell - first * cells]) {
                    store.offsets[cell] = HX_BRICK_ABSENT;
                    skipped++;
                } else {
                    keys[count++] = (Key){.z = z, .y = y, .x = x, .end = -1};
                }
            }
        }
    }
    if (published) {
        printf("%zu masked-out bricks skipped\n", skipped);
    }
    free(published);
    Transfer transfer = {.store = &store,
        .base = base,
        .keys = keys,
        .count = count,
        .write_lock = PTHREAD_MUTEX_INITIALIZER};
    double start = monotonic_seconds();
    failed = transfer_keys(&transfer, connections);
    fprintf(stdout, "%s level %d: %zu bricks, %.2f GB, %.0f s, %d failures\n", path, level,
        atomic_load(&transfer.done), atomic_load(&transfer.bytes) / 1e9,
        monotonic_seconds() - start, atomic_load(&transfer.failures));
    free(keys);
close:
    hx_brick_close(&store);
finish:
    if (meta)
        json_object_put(meta);
    free(source);
    free(base);
    free(clean);
    return failed ? 1 : 0;
}

static int fetch_volume_main(int argc, char **argv) {
    if (argc < 4) {
        puts("Usage: ./run.sh scans.transfer.fetch_volume PATH LEVEL OUT [--root URL] "
             "[--rows FIRST LAST] [--connections N] [--list]");
        return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
    }
    char *end;
    long level = strtol(argv[2], &end, 10);
    if (*end || level < 0 || level > INT32_MAX)
        return 2;
    const char *root = bucket;
    int64_t first = 0, last = -1;
    int connections = 32, list = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--root") && i + 1 < argc)
            root = argv[++i];
        else if (!strcmp(argv[i], "--rows") && i + 2 < argc) {
            first = strtoll(argv[++i], NULL, 10);
            last = strtoll(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--connections") && i + 1 < argc)
            connections = (int)strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--list"))
            list = 1;
        else
            return 2;
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT))
        return 1;
    int status = fetch_level(argv[1], (int)level, argv[3], root, first, last, connections, list);
    curl_global_cleanup();
    return status;
}

#ifndef HX_FETCH_VOLUME_LIBRARY
int main(int argc, char **argv) { return fetch_volume_main(argc, argv); }
#endif
