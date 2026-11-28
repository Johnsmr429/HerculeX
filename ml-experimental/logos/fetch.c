#include <blosc.h>
#include <curl/curl.h>
#include <json-c/json.h>
#include <time.h>

#include "fetch.h"

typedef struct {
    int64_t chunk[3];
    int tries;
    char *data;
    size_t size, capacity;
    CURL *handle;
} Request;

typedef struct {
    int64_t chunks[3], shape[3];
    Box box;
    uint8_t *out;
    const char *base;
    int blosc;
} Job;

static size_t receive(void *data, size_t width, size_t count, void *context) {
    Request *r = (Request *)context;
    size_t n = width * count;
    if (r->size + n > r->capacity) {
        size_t capacity = r->capacity ? r->capacity * 2 : 1 << 20;
        while (capacity < r->size + n)
            capacity *= 2;
        char *grown = (char *)realloc(r->data, capacity);
        if (!grown)
            return 0;
        r->data = grown;
        r->capacity = capacity;
    }
    memcpy(r->data + r->size, data, n);
    r->size += n;
    return n;
}

static json_object *fetch_json(const char *url) {
    CURL *curl = curl_easy_init();
    Request r = {{0, 0, 0}, 0, NULL, 0, 0, curl};
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, receive);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    CURLcode code = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    json_object *out = NULL;
    if (code == CURLE_OK && receive((void *)"", 1, 1, &r))
        out = json_tokener_parse(r.data);
    free(r.data);
    return out;
}

static int unpack(Request *r, size_t bytes) {
    char *out = (char *)malloc(bytes ? bytes : 1);
    if (!out || blosc_decompress_ctx(r->data, out, bytes, 1) != (int)bytes) {
        free(out);
        return 0;
    }
    free(r->data);
    r->data = out;
    r->size = bytes;
    r->capacity = bytes;
    return 1;
}

static void reuse(
    const Job *job, const Volume *prev, const Box *pb, const int64_t *lo, const int64_t *hi) {
    int64_t size1 = job->box.hi[1] - job->box.lo[1], size2 = job->box.hi[2] - job->box.lo[2];
    for (int64_t z = lo[0]; z < hi[0]; z++)
        for (int64_t y = lo[1]; y < hi[1]; y++)
            memcpy(job->out + ((z - job->box.lo[0]) * size1 + (y - job->box.lo[1])) * size2 +
                    (lo[2] - job->box.lo[2]),
                prev->data + ((z - pb->lo[0]) * prev->shape[1] + (y - pb->lo[1])) * prev->shape[2] +
                    (lo[2] - pb->lo[2]),
                (size_t)(hi[2] - lo[2]));
}

static void place(const Job *job, const Request *r) {
    int64_t lo[3], hi[3], size[3];
    for (int a = 0; a < 3; a++) {
        int64_t start = r->chunk[a] * job->chunks[a];
        lo[a] = start > job->box.lo[a] ? start : job->box.lo[a];
        hi[a] = start + job->chunks[a] < job->box.hi[a] ? start + job->chunks[a] : job->box.hi[a];
        size[a] = job->box.hi[a] - job->box.lo[a];
    }
    for (int64_t z = lo[0]; z < hi[0]; z++)
        for (int64_t y = lo[1]; y < hi[1]; y++) {
            const char *src = r->data +
                ((z - r->chunk[0] * job->chunks[0]) * job->chunks[1] +
                    (y - r->chunk[1] * job->chunks[1])) *
                    job->chunks[2] +
                (lo[2] - r->chunk[2] * job->chunks[2]);
            uint8_t *dst = job->out +
                ((z - job->box.lo[0]) * size[1] + (y - job->box.lo[1])) * size[2] +
                (lo[2] - job->box.lo[2]);
            memcpy(dst, src, (size_t)(hi[2] - lo[2]));
        }
}

static void start(CURLM *multi, Request *r, const Job *job) {
    char url[2048];
    snprintf(url, sizeof(url), "%s/%lld/%lld/%lld", job->base, (long long)r->chunk[0],
        (long long)r->chunk[1], (long long)r->chunk[2]);
    r->size = 0;
    r->handle = curl_easy_init();
    curl_easy_setopt(r->handle, CURLOPT_URL, url);
    curl_easy_setopt(r->handle, CURLOPT_WRITEFUNCTION, receive);
    curl_easy_setopt(r->handle, CURLOPT_WRITEDATA, r);
    curl_easy_setopt(r->handle, CURLOPT_PRIVATE, r);
    curl_easy_setopt(r->handle, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(r->handle, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_multi_add_handle(multi, r->handle);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fputs("Usage: fetch ZARR_LEVEL_URL z0:z1,y0:y1,x0:x1 OUT.npy [CONNECTIONS] "
              "[--reuse PREVIOUS.npy z0:z1,y0:y1,x0:x1]\n",
            stderr);
        return 2;
    }
    Job job;
    memset(&job, 0, sizeof(job));
    job.base = argv[1];
    if (box_parse(argv[2], &job.box)) {
        fputs("fetch: bad box\n", stderr);
        return 2;
    }
    int connections = argc > 4 && argv[4][0] != '-' ? atoi(argv[4]) : 64;
    Volume prev = {0};
    Box pb = {{0, 0, 0}, {0, 0, 0}};
    int reusing = 0;
    for (int i = 4; i + 2 < argc; i++)
        if (!strcmp(argv[i], "--reuse")) {
            if (npy_open(argv[i + 1], &prev) || box_parse(argv[i + 2], &pb)) {
                fprintf(stderr, "fetch: cannot reuse %s\n", argv[i + 1]);
                return 1;
            }
            reusing = 1;
        }
    curl_global_init(CURL_GLOBAL_DEFAULT);
    char url[2048];
    snprintf(url, sizeof(url), "%s/.zarray", job.base);
    json_object *meta = fetch_json(url), *v = NULL;
    if (!meta) {
        fprintf(stderr, "fetch: cannot read %s\n", url);
        return 1;
    }
    int ok = json_object_object_get_ex(meta, "chunks", &v) && json_object_array_length(v) == 3;
    for (int a = 0; ok && a < 3; a++)
        job.chunks[a] = json_object_get_int64(json_object_array_get_idx(v, (size_t)a));
    ok = ok && json_object_object_get_ex(meta, "shape", &v) && json_object_array_length(v) == 3;
    for (int a = 0; ok && a < 3; a++)
        job.shape[a] = json_object_get_int64(json_object_array_get_idx(v, (size_t)a));
    json_object *dtype = NULL, *compressor = NULL, *separator = NULL, *id = NULL;
    ok = ok && json_object_object_get_ex(meta, "dtype", &dtype) &&
        !strcmp(json_object_get_string(dtype), "|u1") &&
        json_object_object_get_ex(meta, "dimension_separator", &separator) &&
        !strcmp(json_object_get_string(separator), "/");
    if (ok && json_object_object_get_ex(meta, "compressor", &compressor) && compressor) {
        job.blosc = json_object_object_get_ex(compressor, "id", &id) &&
            !strcmp(json_object_get_string(id), "blosc");
        ok = job.blosc;
    }
    json_object_put(meta);
    if (!ok) {
        fputs("fetch: the level is not a uint8 zarr, raw or Blosc, with '/' chunk keys\n", stderr);
        return 1;
    }
    int64_t size[3], first[3], last[3], total = 1;
    for (int a = 0; a < 3; a++) {
        if (job.box.lo[a] < 0)
            job.box.lo[a] = 0;
        if (job.box.hi[a] > job.shape[a])
            job.box.hi[a] = job.shape[a];
        size[a] = job.box.hi[a] - job.box.lo[a];
        first[a] = job.box.lo[a] / job.chunks[a];
        last[a] = (job.box.hi[a] - 1) / job.chunks[a];
        total *= last[a] - first[a] + 1;
    }
    char *part = (char *)malloc(strlen(argv[3]) + 8);
    sprintf(part, "%s.part", argv[3]);
    FILE *file = fopen(part, "wb");
    if (!file || npy_header(file, "|u1", size, 3)) {
        fprintf(stderr, "fetch: cannot write %s\n", part);
        return 1;
    }
    long header = ftell(file);
    fclose(file);
    size_t bytes = (size_t)(size[0] * size[1] * size[2]);
    int fd = open(part, O_RDWR);
    if (fd < 0 || ftruncate(fd, (off_t)(header + (long)bytes))) {
        fprintf(stderr, "fetch: cannot size %s\n", part);
        return 1;
    }
    void *map = mmap(NULL, (size_t)header + bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fputs("fetch: cannot map the output\n", stderr);
        return 1;
    }
    job.out = (uint8_t *)map + header;
    for (int a = 0; reusing && a < 3; a++) {
        pb.lo[a] = pb.lo[a] < 0 ? 0 : pb.lo[a];
        pb.hi[a] = pb.hi[a] > job.shape[a] ? job.shape[a] : pb.hi[a];
        if (prev.shape[a] != pb.hi[a] - pb.lo[a]) {
            fputs("fetch: the reused volume is not its box's shape\n", stderr);
            return 1;
        }
    }
    Request *requests = (Request *)calloc((size_t)total, sizeof(Request));
    int64_t n = 0, reused = 0;
    for (int64_t z = first[0]; z <= last[0]; z++)
        for (int64_t y = first[1]; y <= last[1]; y++)
            for (int64_t x = first[2]; x <= last[2]; x++) {
                int64_t chunk[3] = {z, y, x}, lo[3], hi[3];
                int inside = reusing;
                for (int a = 0; a < 3; a++) {
                    lo[a] = chunk[a] * job.chunks[a] > job.box.lo[a] ? chunk[a] * job.chunks[a]
                                                                     : job.box.lo[a];
                    hi[a] = (chunk[a] + 1) * job.chunks[a] < job.box.hi[a]
                        ? (chunk[a] + 1) * job.chunks[a]
                        : job.box.hi[a];
                    inside &= lo[a] >= pb.lo[a] && hi[a] <= pb.hi[a];
                }
                if (inside) {
                    reuse(&job, &prev, &pb, lo, hi);
                    reused++;
                    continue;
                }
                requests[n].chunk[0] = z;
                requests[n].chunk[1] = y;
                requests[n].chunk[2] = x;
                n++;
            }
    total = n;
    if (reusing)
        npy_close(&prev);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    CURLM *multi = curl_multi_init();
    curl_multi_setopt(multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, (long)connections);
    int64_t next = 0, done = 0, present = 0, missing = 0, failed = 0, running = 0;
    int64_t *retry = (int64_t *)malloc((size_t)total * sizeof(int64_t));
    int64_t retry_count = 0;
    size_t chunk_bytes = (size_t)(job.chunks[0] * job.chunks[1] * job.chunks[2]);
    double received = 0;
    while (done < total) {
        while (running < connections && (retry_count || next < total)) {
            Request *r = retry_count ? &requests[retry[--retry_count]] : &requests[next++];
            start(multi, r, &job);
            running++;
        }
        int still = 0;
        curl_multi_perform(multi, &still);
        curl_multi_poll(multi, NULL, 0, 200, NULL);
        CURLMsg *msg;
        int left;
        while ((msg = curl_multi_info_read(multi, &left))) {
            if (msg->msg != CURLMSG_DONE)
                continue;
            Request *r = NULL;
            curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, (char **)&r);
            long status = 0;
            curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &status);
            CURLcode result = msg->data.result;
            curl_multi_remove_handle(multi, msg->easy_handle);
            curl_easy_cleanup(msg->easy_handle);
            running--;
            if (result == CURLE_OK && status == 200 &&
                (job.blosc ? unpack(r, chunk_bytes) : r->size == chunk_bytes)) {
                place(&job, r);
                received += (double)r->size;
                present++;
                done++;
            } else if (result == CURLE_OK && (status == 404 || status == 403)) {
                missing++;
                done++;
            } else if (++r->tries < 6) {
                retry[retry_count++] = r - requests;
            } else {
                failed++;
                done++;
            }
            free(r->data);
            r->data = NULL;
            r->capacity = 0;
            r->size = 0;
        }
    }
    curl_multi_cleanup(multi);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double seconds = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    msync(map, (size_t)header + bytes, MS_SYNC);
    munmap(map, (size_t)header + bytes);
    if (failed) {
        fprintf(
            stderr, "fetch: %lld chunks failed; %s kept for a rerun\n", (long long)failed, part);
        return 1;
    }
    if (rename(part, argv[3])) {
        fprintf(stderr, "fetch: cannot rename %s\n", part);
        return 1;
    }
    printf("fetch: %lld x %lld x %lld voxels, %lld chunks present, %lld absent, %lld reused, "
           "%.2f GB "
           "in %.1f s (%.0f MB/s)\n",
        (long long)size[0], (long long)size[1], (long long)size[2], (long long)present,
        (long long)missing, (long long)reused, received / 1e9, seconds,
        received / 1e6 / (seconds > 0 ? seconds : 1));
    free(requests);
    free(retry);
    free(part);
    curl_global_cleanup();
    return 0;
}
