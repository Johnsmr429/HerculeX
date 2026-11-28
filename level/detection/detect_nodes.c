#define _GNU_SOURCE
#include <curl/curl.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <json-c/json.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "../../storage/brick_store.c"
#define HX_NODE_STORE_LOCAL
#include "../../storage/node_store.c"

#include "../../storage/data_roots.c"

enum { HX_CELL = 8, HX_MARGIN = 8, AIR_LEVEL = 40, EDGE = 2, PASSES = 3 };

typedef int (*NodeDetector)(const uint8_t *, int, int, int, int, int, int, int, int, int, void *,
    int32_t *, uint32_t, uint32_t *, double *);

int hx_gpu_open(const char *source, char *message, int length);
int hx_gpu_cells(const uint8_t *slab, int Z, int Y, int X, int cells_z, int cells_y, int cells_x,
    int first_z, int air, int slab_z0, void *nodes, int32_t *ids, uint32_t capacity,
    uint32_t *count, double *seconds);
#ifdef HX_UNITY
int hx_cpu_cells(const uint8_t *slab, int z, int y, int x, int cz, int cy, int cx, int first_z,
    int air, int slab_z0, void *raw, int32_t *ids, uint32_t capacity, uint32_t *count,
    double *seconds);
#endif
int hx_detect_volume(NodeDetector detect, const uint8_t *bricks, int64_t bytes,
    const int64_t *offsets, const int32_t *shape, int cell, int margin, int air, int first,
    int last, const char *node_path, const char *cell_path, int64_t *total, double *gpu_time);

typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} Buffer;

typedef struct {
    int64_t first, last, low, high;
} Unit;

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static int make_folders(const char *path) {
    char *copy = strdup(path);
    if (!copy)
        return 0;
    for (char *at = copy + 1; *at; at++) {
        if (*at != '/')
            continue;
        *at = 0;
        int ok = !mkdir(copy, 0777) || errno == EEXIST;
        *at = '/';
        if (!ok) {
            free(copy);
            return 0;
        }
    }
    int ok = !mkdir(copy, 0777) || errno == EEXIST;
    free(copy);
    return ok;
}

static int remove_tree(const char *path) {
    struct stat current;
    if (lstat(path, &current)) {
        return errno == ENOENT ? 0 : -1;
    }
    if (!S_ISDIR(current.st_mode)) {
        return unlink(path);
    }
    DIR *directory = opendir(path);
    if (!directory)
        return errno == ENOENT ? 0 : -1;
    struct dirent *entry;
    int status = 0;
    while (!status) {
        entry = readdir(directory);
        if (!entry)
            break;
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        char *child = NULL;
        if (asprintf(&child, "%s/%s", path, entry->d_name) < 0) {
            status = -1;
            break;
        }
        struct stat info;
        if (lstat(child, &info))
            status = -1;
        else if (S_ISDIR(info.st_mode) && !S_ISLNK(info.st_mode))
            status = remove_tree(child);
        else
            status = unlink(child);
        free(child);
    }
    int saved = errno;
    closedir(directory);
    if (!status)
        status = rmdir(path);
    errno = saved;
    return status;
}

static int regular(const char *path) {
    struct stat info;
    return !stat(path, &info) && S_ISREG(info.st_mode);
}

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END)) {
        fclose(file);
        return NULL;
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    char *data = malloc((size_t)length + 1);
    int ok = data && fread(data, 1, (size_t)length, file) == (size_t)length;
    fclose(file);
    if (!ok) {
        free(data);
        return NULL;
    }
    data[length] = 0;
    return data;
}

static uint64_t hash_file(uint64_t hash, const char *path) {
    FILE *file = fopen(path, "rb");
    unsigned char block[65536];
    size_t count;
    while (file) {
        count = fread(block, 1, sizeof(block), file);
        if (!count)
            break;
        for (size_t i = 0; i < count; i++)
            hash = (hash ^ block[i]) * UINT64_C(1099511628211);
    }
    if (file)
        fclose(file);
    return hash;
}

static void method_id(char output[13]) {
    const char *sources[] = {"level/detection/detect_nodes.c", "level/detection/node_stream.h",
#ifdef __APPLE__
        "level/detection/nodes_apple/host.m", "level/detection/nodes_apple/kernel.metal",
#else
        "level/detection/nodes_cuda/host.c", "level/detection/nodes_cuda/kernel.cu",
#endif
        "storage/brick_store.c", "storage/node_store.c"};
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < sizeof(sources) / sizeof(*sources); i++)
        hash = hash_file(hash, sources[i]);
    snprintf(output, 13, "%012llx", (unsigned long long)(hash & UINT64_C(0xffffffffffff)));
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
        size_t capacity = buffer->capacity ? buffer->capacity : 4096;
        while (capacity < need && capacity <= SIZE_MAX / 2)
            capacity *= 2;
        if (capacity < need)
            capacity = need;
        char *larger = realloc(buffer->data, capacity);
        if (!larger)
            return 0;
        buffer->data = larger;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->size, data, bytes);
    buffer->size += bytes;
    return bytes;
}

static json_object *url_json(const char *url) {
    CURL *curl = curl_easy_init();
    Buffer body = {0};
    long status = 0;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, receive);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    }
    int ok = curl && curl_easy_perform(curl) == CURLE_OK &&
        !curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status) && status == 200;
    json_tokener *parser = ok ? json_tokener_new() : NULL;
    json_object *root =
        parser ? json_tokener_parse_ex(parser, (const char *)body.data, (int)body.size) : NULL;
    if (parser && json_tokener_get_error(parser) != json_tokener_success) {
        json_object_put(root);
        root = NULL;
    }
    if (parser)
        json_tokener_free(parser);
    if (curl)
        curl_easy_cleanup(curl);
    free(body.data);
    return root;
}

static int wait_child(pid_t pid) {
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR)
            return 1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

static int run_child(char *const arguments[]) {
    pid_t child = fork();
    if (!child) {
        execv(arguments[0], arguments);
        _exit(127);
    }
    return child < 0 ? 1 : wait_child(child);
}

static char *capture_child(char *const arguments[]) {
    int ends[2];
    if (pipe(ends))
        return NULL;
    pid_t child = fork();
    if (!child) {
        close(ends[0]);
        dup2(ends[1], STDOUT_FILENO);
        close(ends[1]);
        execv(arguments[0], arguments);
        _exit(127);
    }
    close(ends[1]);
    Buffer output = {0};
    unsigned char block[4096];
    ssize_t n;
    while ((n = read(ends[0], block, sizeof(block))) > 0)
        if (receive(block, 1, (size_t)n, &output) != (size_t)n)
            break;
    close(ends[0]);
    if (child < 0 || wait_child(child)) {
        free(output.data);
        return NULL;
    }
    char *data = realloc(output.data, output.size + 1);
    if (!data) {
        free(output.data);
        return NULL;
    }
    data[output.size] = 0;
    return data;
}

static int gpu_open(void) {
#ifdef __APPLE__
    const char *kernel = "level/detection/nodes_apple/kernel.metal";
#else
    const char *kernel = "level/detection/nodes_cuda/kernel.cu";
#endif
    char *body = read_file(kernel);
    char *source = NULL;
    int made = body &&
        asprintf(&source, "#define CELL %d\n#define MARGIN %d\n%s", HX_CELL, HX_MARGIN, body) >= 0;
    char message[4096] = {0};
    int status = made ? hx_gpu_open(source, message, sizeof(message)) : 1;
    if (status)
        fprintf(stderr, "HerculeX: sheet-node GPU unavailable: %s\n",
            *message ? message : strerror(errno));
    else
        fprintf(stdout, "sheet-node device: %s\n", message);
    free(source);
    free(body);
    return status;
}

static int record_write(const char *folder, json_object *record) {
    char *target = NULL, *part = NULL;
    int ok = asprintf(&target, "%s/record.json", folder) >= 0 &&
        asprintf(&part, "%s/.record-%ld.part", folder, (long)getpid()) >= 0 &&
        json_object_to_file_ext(part, record, JSON_C_TO_STRING_PRETTY) == 0 &&
        !rename(part, target);
    if (!ok && part)
        unlink(part);
    free(part);
    free(target);
    return ok ? 0 : -1;
}

static int publish_nodes(const char *part, const char *out) {
    struct stat info;
    if (lstat(out, &info)) {
        return errno == ENOENT ? rename(part, out) : -1;
    }
    if (!S_ISDIR(info.st_mode)) {
        errno = ENOTDIR;
        return -1;
    }
#ifdef __APPLE__
    int status = renameatx_np(AT_FDCWD, part, AT_FDCWD, out, RENAME_SWAP);
#else
    int status = renameat2(AT_FDCWD, part, AT_FDCWD, out, RENAME_EXCHANGE);
#endif
    return status || remove_tree(part);
}

static int detect_store(const char *level, const char *out, int64_t first, int64_t last,
    json_object *record, const char *method, int replace, NodeDetector detect) {
    HxBrickStore store;
    if (hx_brick_open(&store, level, NULL, NULL, NULL)) {
        fprintf(stderr, "HerculeX: cannot open CT store %s: %s\n", level, strerror(errno));
        return 1;
    }
    int64_t grid_z = (store.shape[0] + HX_CELL - 1) / HX_CELL;
    int valid = first >= 0 && first < last && last <= grid_z;
    int32_t shape[6];
    for (int i = 0; valid && i < 3; i++) {
        valid = store.shape[i] <= INT32_MAX && store.brick[i] <= INT32_MAX;
        shape[i] = (int32_t)store.shape[i];
        shape[i + 3] = (int32_t)store.brick[i];
    }
    if (!valid) {
        fprintf(stderr, "HerculeX: invalid CT geometry or cell planes\n");
        hx_brick_close(&store);
        return 1;
    }
    const uint8_t *data = NULL;
    int64_t bytes = 0;
    if (hx_brick_map(&store, &data, &bytes)) {
        hx_brick_close(&store);
        return 1;
    }
    char *part = NULL, *nodes = NULL, *cells = NULL, *receipt = NULL;
    int made = asprintf(&part, "%s.building.%ld", out, (long)getpid()) >= 0 &&
        asprintf(&nodes, "%s/nodes.bin", part) >= 0 &&
        asprintf(&cells, "%s/cells.bin", part) >= 0 &&
        asprintf(&receipt, "%s/record.json", out) >= 0;
    int status = 1;
    if (!made)
        goto finish;
    if (!replace && regular(receipt)) {
        status = 0;
        goto finish;
    }
    if (remove_tree(part) || !make_folders(part))
        goto finish;
    int64_t total = 0;
    double gpu_seconds = 0, began = now();
    int detected = hx_detect_volume(detect, data, bytes, store.offsets, shape, HX_CELL, HX_MARGIN,
        AIR_LEVEL, (int)first, (int)last, nodes, cells, &total, &gpu_seconds);
    if (detected || hx_node_pack(part)) {
        fprintf(stderr, "HerculeX: sheet detection failed with status %d: %s\n", detected,
            strerror(errno));
        goto finish;
    }
    json_object *shape_json = json_object_new_array_ext(3);
    json_object *grid_json = json_object_new_array_ext(3);
    json_object *planes = json_object_new_array_ext(2);
    for (int i = 0; i < 3; i++) {
        json_object_array_add(shape_json, json_object_new_int64(store.shape[i]));
        json_object_array_add(
            grid_json, json_object_new_int64((store.shape[i] + HX_CELL - 1) / HX_CELL));
    }
    json_object_array_add(planes, json_object_new_int64(first));
    json_object_array_add(planes, json_object_new_int64(last));
    json_object_object_add(record, "shape", shape_json);
    json_object_object_add(record, "cell", json_object_new_int(HX_CELL));
    json_object_object_add(record, "grid", grid_json);
    json_object_object_add(record, "cell_planes", planes);
    json_object_object_add(record, "nodes", json_object_new_int64(total));
    json_object_object_add(
        record, "gpu_seconds", json_object_new_double(round(gpu_seconds * 10) / 10));
    json_object_object_add(
        record, "seconds", json_object_new_double(round((now() - began) * 10) / 10));
    json_object_object_add(record, "method", json_object_new_string(method));
    if (record_write(part, record) || publish_nodes(part, out))
        goto finish;
    fprintf(stdout, "%s %lld nodes\n", strrchr(out, '/') ? strrchr(out, '/') + 1 : out,
        (long long)total);
    status = 0;
finish:
    if (status && part) {
        remove_tree(part);
    }
    if (data && bytes)
        munmap((void *)data, (size_t)bytes);
    hx_brick_close(&store);
    free(receipt);
    free(cells);
    free(nodes);
    free(part);
    return status;
}

static int shape_of(json_object *row, int level, int64_t shape[3]) {
    json_object *products, *roots, *path_value, *root_value;
    if (!json_object_object_get_ex(row, "products", &products) ||
        !json_object_object_get_ex(row, "roots", &roots) ||
        !json_object_object_get_ex(products, "ome-zarr", &path_value) ||
        !json_object_object_get_ex(roots, "ome-zarr", &root_value))
        return 0;
    const char *path = json_object_get_string(path_value);
    while (*path == '/')
        path++;
    size_t length = strlen(path);
    while (length && path[length - 1] == '/')
        length--;
    char *url = NULL;
    int made = asprintf(&url, "%s/%.*s/%d/.zarray", json_object_get_string(root_value), (int)length,
                   path, level) >= 0;
    json_object *meta = made ? url_json(url) : NULL, *array;
    int ok = meta && json_object_object_get_ex(meta, "shape", &array) &&
        json_object_array_length(array) == 3;
    for (int i = 0; ok && i < 3; i++)
        shape[i] = json_object_get_int64(json_object_array_get_idx(array, (size_t)i));
    if (meta)
        json_object_put(meta);
    free(url);
    return ok;
}

static int fetch_chunk(json_object *row, int level, int64_t first, int64_t last, const char *work) {
    json_object *products, *roots, *path, *root;
    if (!json_object_object_get_ex(row, "products", &products) ||
        !json_object_object_get_ex(row, "roots", &roots) ||
        !json_object_object_get_ex(products, "ome-zarr", &path) ||
        !json_object_object_get_ex(roots, "ome-zarr", &root))
        return 1;
    char level_text[24], first_text[24], last_text[24];
    char *connections = getenv("HERCULEX_FETCH_CONNECTIONS");
    snprintf(level_text, sizeof(level_text), "%d", level);
    snprintf(first_text, sizeof(first_text), "%lld", (long long)first);
    snprintf(last_text, sizeof(last_text), "%lld", (long long)last);
    const char *list = getenv("HERCULEX_LIST_CT_BRICKS");
    char *arguments[] = {"./run.sh", "scans.transfer.fetch_volume",
        (char *)json_object_get_string(path), level_text, (char *)work, "--root",
        (char *)json_object_get_string(root), "--rows", first_text, last_text, "--connections",
        connections && *connections ? connections : "16",
        list && !strcmp(list, "1") ? "--list" : NULL, NULL};
    return run_child(arguments);
}

static int unit_done(const char *folder, const char *method) {
    char *path = path_join(folder, "record.json");
    json_object *record = path ? json_object_from_file(path) : NULL;
    free(path);
    const char *used = json_object_get_string(json_object_object_get(record, "method"));
    int done = used && !strcmp(used, method);
    FILE *aliases = !done && used ? fopen("level/detection/same_output_methods.txt", "r") : NULL;
    if (aliases) {
        char candidate[64];
#ifdef __APPLE__
        const char lane[] = "apple:";
#else
        const char lane[] = "cuda:";
#endif
        while (!done && fscanf(aliases, "%63s", candidate) == 1) {
            const char *name = strchr(candidate, ':');
            if (!name || !strncmp(candidate, lane, sizeof(lane) - 1)) {
                done = !strcmp(used, name ? name + 1 : candidate);
            }
        }
        fclose(aliases);
    }
    json_object_put(record);
    return done;
}

static int sized_units(const char *folder, int rows) {
    const char *patterns[] = {"rows-*", "band-*"};
    for (int mode = 0; mode < 2; mode++) {
        char *pattern = path_join(folder, patterns[mode]);
        glob_t list = {0};
        if (pattern && !glob(pattern, 0, NULL, &list)) {
            for (size_t i = 0; i < list.gl_pathc; i++) {
                if (strstr(list.gl_pathv[i], ".building")) {
                    continue;
                }
                char *path = path_join(list.gl_pathv[i], "record.json");
                json_object *record = path ? json_object_from_file(path) : NULL, *setting;
                free(path);
                int current = record && json_object_object_get_ex(record, "unit_rows", &setting)
                    ? json_object_get_int(setting)
                    : 12;
                json_object_put(record);
                if (record && current != rows) {
                    fprintf(stderr, "HerculeX: %s holds units not cut to HERCULEX_UNIT_ROWS=%d\n",
                        folder, rows);
                    globfree(&list);
                    free(pattern);
                    return 0;
                }
            }
        }
        globfree(&list);
        free(pattern);
    }
    return 1;
}

static double central_row(const char *sample, const char *scan, int level, double fallback) {
    char *library = local_root(library_root), *pattern = NULL;
    glob_t list = {0};
    if (library && asprintf(&pattern, "%s/raw/%s/atlas_*/L[35]", library, sample) >= 0 &&
        !glob(pattern, 0, NULL, &list)) {
        for (size_t i = 0; i < list.gl_pathc; i++) {
            HxBrickStore atlas;
            const char *path = list.gl_pathv[i];
            if (hx_brick_open(&atlas, path, NULL, NULL, NULL)) {
                continue;
            }
            if (atlas.source && strstr(atlas.source, scan)) {
                int64_t present = 0;
                for (int64_t j = 0; j < atlas.count; j++) {
                    present += atlas.offsets[j] >= 0;
                }
                if (present) {
                    int64_t plane = atlas.grid[1] * atlas.grid[2];
                    int64_t middle = present / 2 + present % 2, before = 0, z = 0;
                    for (; z < atlas.grid[0]; z++) {
                        for (int64_t j = z * plane; j < (z + 1) * plane; j++) {
                            before += atlas.offsets[j] >= 0;
                        }
                        if (before >= middle) {
                            break;
                        }
                    }
                    fallback = ldexp(z + .5, path[strlen(path) - 1] - '0' - level);
                    hx_brick_close(&atlas);
                    break;
                }
            }
            hx_brick_close(&atlas);
        }
    }
    globfree(&list);
    free(pattern);
    free(library);
    return fallback;
}

static int carry_row(const char *previous, const char *work, int64_t row) {
    HxBrickStore source, target;
    if (hx_brick_open(&source, previous, NULL, NULL, NULL)) {
        return errno == ENOENT ? 0 : -1;
    }
    int status = 0, ready = row < source.grid[0];
    int64_t width = source.grid[2], height = source.grid[1], first = row * height * width;
    for (int64_t i = 0; ready && i < height * width; i++) {
        ready = source.offsets[first + i] != HX_BRICK_UNKNOWN;
    }
    if (ready) {
        const uint8_t *data = NULL;
        int64_t bytes = 0;
        int mapped = !hx_brick_map(&source, &data, &bytes);
        int opened =
            mapped && !hx_brick_open(&target, work, source.shape, source.brick, source.source);
        status = !opened || memcmp(source.shape, target.shape, sizeof(source.shape)) != 0 ||
            memcmp(source.brick, target.brick, sizeof(source.brick)) != 0;
        for (int64_t i = 0; !status && i < height * width; i++) {
            int64_t offset = source.offsets[first + i];
            if (target.offsets[first + i] != HX_BRICK_UNKNOWN) {
                continue;
            }
            if (offset > bytes - source.bytes && offset >= 0) {
                status = 1;
            } else {
                status = hx_brick_put(&target, row, i / width, i % width,
                    offset >= 0 ? data + offset : NULL, source.bytes);
            }
        }
        if (opened) {
            int failed = status ? 0 : hx_brick_save(&target);
            status = status || failed;
            hx_brick_close(&target);
        }
        if (data) {
            munmap((void *)data, bytes);
        }
    }
    hx_brick_close(&source);
    return status ? -1 : 0;
}

static int scan_units(
    json_object *row, int band, const char *nodes_root, const char *work_root, const char *method) {
    json_object *sample_value, *volume_value, *voxel_value;
    if (!json_object_object_get_ex(row, "sample", &sample_value) ||
        !json_object_object_get_ex(row, "volume", &volume_value) ||
        !json_object_object_get_ex(row, "voxel_um", &voxel_value))
        return 1;
    const char *sample = json_object_get_string(sample_value);
    const char *volume = json_object_get_string(volume_value);
    double voxel = json_object_get_double(voxel_value);
    int level = (int)fmax(0, nearbyint(log2(18.7 / voxel)));
    int64_t shape[3];
    if (!shape_of(row, level, shape)) {
        fprintf(stderr, "HerculeX: cannot read %s %s level %d shape\n", sample, volume, level);
        return 1;
    }
    int64_t rows = (shape[0] + 127) / 128;
    int chunk_rows = 12;
    const char *setting = getenv("HERCULEX_UNIT_ROWS");
    if (setting && *setting)
        chunk_rows = (int)strtol(setting, NULL, 10);
    if (chunk_rows < 2)
        return 1;
    char *folder = NULL;
    int checked = asprintf(&folder, "%s/%s/%s/nodes", nodes_root, sample, volume) >= 0 &&
        make_folders(folder) && sized_units(folder, chunk_rows);
    free(folder);
    if (!checked) {
        return 1;
    }
    Unit *units = calloc((size_t)(rows + 1), sizeof(*units));
    if (!units) {
        return 1;
    }
    size_t count = 0;
    int per_row = 128 / HX_CELL, seam = per_row / 2;
    if (band) {
        double middle = central_row(sample, volume, level, rows / 2.0);
        int64_t first = (int64_t)fmax(0, floor(middle - chunk_rows / 2.0));
        first = first < rows ? first : rows - 1;
        int64_t last = first + chunk_rows < rows ? first + chunk_rows : rows;
        int64_t top = (shape[0] + HX_CELL - 1) / HX_CELL - EDGE;
        units[count++] = (Unit){first, last, first * per_row + EDGE,
            last * per_row - EDGE < top ? last * per_row - EDGE : top};
    } else {
        int64_t first = 0;
        for (;;) {
            int64_t last = first + chunk_rows < rows ? first + chunk_rows : rows;
            int64_t top = (shape[0] + HX_CELL - 1) / HX_CELL - EDGE;
            int64_t low = first ? first * per_row + seam : EDGE;
            int64_t high = last == rows ? top : (last - 1) * per_row + seam;
            if (high > low)
                units[count++] = (Unit){first, last, low, high};
            if (last == rows)
                break;
            first = last - 1;
        }
    }
    int failed = 0, previous_done = 0;
    char *previous = NULL;
    int64_t previous_end = -1;
    for (size_t i = 0; i < count; i++) {
        Unit unit = units[i];
        char *out = NULL, *work = NULL;
        int made = asprintf(&out, "%s/%s/%s/nodes/%s-%04lld-%04lld", nodes_root, sample, volume,
                       band ? "band" : "rows", (long long)unit.first, (long long)unit.last) >= 0 &&
            asprintf(&work, "%s/%s/%s/L%d-%04lld-%04lld", work_root, sample, volume, level,
                (long long)unit.first, (long long)unit.last) >= 0;
        if (!made || unit_done(out, method)) {
            failed |= !made;
            free(work);
            free(out);
            continue;
        }
        struct statvfs disk;
        if (statvfs(nodes_root, &disk) || (uint64_t)disk.f_bavail * disk.f_frsize < 30ULL << 30) {
            fprintf(stderr, "HerculeX: stopped under 30 GiB free on the data disk\n");
            failed = 3;
            free(work);
            free(out);
            break;
        }
        if (previous && previous_end - 1 == unit.first && carry_row(previous, work, unit.first)) {
            fprintf(stderr, "HerculeX: cannot carry shared row %lld, fetching it\n",
                (long long)unit.first);
        }
        if (previous_done) {
            remove_tree(previous);
        }
        free(previous);
        previous = work;
        previous_end = unit.last;
        previous_done = 0;
        if (fetch_chunk(row, level, unit.first, unit.last, work)) {
            failed = 1;
        } else {
            json_object *record = json_object_new_object();
            json_object *range = json_object_new_array_ext(2);
            json_object_array_add(range, json_object_new_int64(unit.first));
            json_object_array_add(range, json_object_new_int64(unit.last));
            json_object_object_add(record, "sample", json_object_new_string(sample));
            json_object_object_add(record, "volume", json_object_new_string(volume));
            json_object_object_add(record, "scan_voxel_um", json_object_new_double(voxel));
            json_object_object_add(record, "level", json_object_new_int(level));
            json_object_object_add(
                record, "voxel_um", json_object_new_double(voxel * (1 << level)));
            json_object_object_add(record, "rows", range);
            json_object_object_add(record, "unit_rows", json_object_new_int(chunk_rows));
            previous_done =
                !detect_store(work, out, unit.low, unit.high, record, method, 1, hx_gpu_cells);
            failed |= !previous_done;
            json_object_put(record);
        }
        free(out);
    }
    if (previous_done) {
        remove_tree(previous);
    }
    free(previous);
    free(units);
    return failed;
}

static json_object *selected_rows(char **samples, int count, const char *scan) {
    size_t capacity = (size_t)count + 8;
    char **arguments = calloc(capacity, sizeof(*arguments));
    if (!arguments)
        return NULL;
    size_t used = 0;
    arguments[used++] = "./run.sh";
    arguments[used++] = "scans.catalog.scan_choice";
    arguments[used++] = scan ? "volume" : "chosen";
    arguments[used++] = "--json";
    for (int i = 0; i < count; i++)
        arguments[used++] = samples[i];
    if (scan)
        arguments[used++] = (char *)scan;
    char *text = capture_child(arguments);
    free(arguments);
    json_object *rows = text ? json_tokener_parse(text) : NULL;
    free(text);
    return rows;
}

static int run_scans(char **samples, int count, int band, const char *scan) {
    json_object *rows = selected_rows(samples, count, scan);
    if (!rows || !json_object_is_type(rows, json_type_array)) {
        fprintf(stderr, "HerculeX: no native scan selection\n");
        if (rows)
            json_object_put(rows);
        return 1;
    }
    char *outputs = local_root(outputs_root);
    char *unwrap = outputs ? path_join(outputs, "unwrapping") : NULL;
    char *nodes = unwrap ? path_join(unwrap, "nodes") : NULL;
    char *work = unwrap ? path_join(unwrap, "detection-work") : NULL;
    int failed = !nodes || !work || !make_folders(nodes) || !make_folders(work);
    char *lock_path = failed ? NULL : path_join(nodes, "sheet-detector.lock");
    int lock = lock_path ? open(lock_path, O_CREAT | O_RDWR | O_CLOEXEC, 0666) : -1;
    free(lock_path);
    if (lock < 0 || flock(lock, LOCK_EX | LOCK_NB)) {
        fprintf(stderr, "HerculeX: cannot claim sheet detection in %s: %s\n",
            nodes ? nodes : "the data folder", strerror(errno));
        failed = 1;
    } else {
        char method[13];
        method_id(method);
        size_t length = json_object_array_length(rows);
        if (!length && count) {
            fprintf(stderr, "HerculeX: no matching scan in the local catalog\n");
            failed = 1;
        }
        for (int pass = 0; length && pass < PASSES; pass++) {
            failed = 0;
            for (size_t i = 0; i < length; i++) {
                int status =
                    scan_units(json_object_array_get_idx(rows, i), band, nodes, work, method);
                if (status == 3) {
                    failed = status;
                    break;
                }
                failed |= status != 0;
            }
            if (!failed || failed == 3) {
                break;
            }
        }
    }
    if (lock >= 0) {
        close(lock);
    }
    free(work);
    free(nodes);
    free(unwrap);
    free(outputs);
    json_object_put(rows);
    return failed == 3 ? 3 : failed != 0;
}

static void usage(FILE *out) {
    fputs("Usage: ./run.sh level.detection.detect_nodes run [SCROLL ...] [--band] "
          "[--scan ID]\n"
          "       ./run.sh level.detection.detect_nodes store CT_STORE OUT FIRST LAST\n",
        out);
}

#ifdef HX_UNITY
int hx_synthetic_detector(
    const char *store, const char *out, int64_t planes, json_object *record, int gpu) {
    char method[13];
    if (gpu && gpu_open()) {
        return 1;
    }
    method_id(method);
    return detect_store(
        store, out, 0, planes, record, method, 1, gpu ? hx_gpu_cells : hx_cpu_cells);
}
#endif

int hx_detection_main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(argc < 2 ? stderr : stdout);
        return argc < 2;
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT))
        return 1;
    int code = 0;
    if (!strcmp(argv[1], "store")) {
        if (argc != 6) {
            usage(stderr);
            code = 2;
        } else if (gpu_open()) {
            code = 1;
        } else {
            char method[13];
            method_id(method);
            json_object *record = json_object_new_object();
            code = detect_store(argv[2], argv[3], strtoll(argv[4], NULL, 10),
                strtoll(argv[5], NULL, 10), record, method, 0, hx_gpu_cells);
            json_object_put(record);
        }
    } else if (!strcmp(argv[1], "run") || !strcmp(argv[1], "band")) {
        int band = !strcmp(argv[1], "band"), count = 0;
        const char *scan = NULL;
        char **samples = calloc((size_t)argc, sizeof(*samples));
        for (int i = 2; samples && i < argc; i++) {
            if (!strcmp(argv[i], "--band"))
                band = 1;
            else if (!strcmp(argv[i], "--scan") && i + 1 < argc)
                scan = argv[++i];
            else if (!strcmp(argv[i], "--since") && i + 1 < argc)
                i++;
            else
                samples[count++] = argv[i];
        }
        if (!samples || (scan && count != 1))
            code = 2;
        else if (gpu_open())
            code = 1;
        else
            code = run_scans(samples, count, band, scan);
        free(samples);
    } else {
        usage(stderr);
        code = 2;
    }
    curl_global_cleanup();
    return code;
}

#if !defined(HX_UNITY)
int main(int argc, char **argv) { return hx_detection_main(argc, argv); }
#endif
