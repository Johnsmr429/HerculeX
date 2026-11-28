#pragma once
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <unistd.h>

#ifdef __APPLE__
static int sparse_part(int fd, off_t first, off_t last, off_t *block) {
    if (!*block) {
        return 0;
    }
    off_t start = first + (*block - first % *block) % *block;
    off_t stop = last - last % *block;
    if (stop <= start) {
        return 0;
    }
    fpunchhole_t hole = {.fp_offset = start, .fp_length = stop - start};
    if (!fcntl(fd, F_PUNCHHOLE, &hole)) {
        return 0;
    }
    if (errno == ENOTSUP || errno == ENOTTY || errno == EINVAL) {
        *block = 0;
        return 0;
    }
    return -1;
}
#endif

static off_t append_brick(
    int fd, const char *data, size_t length, off_t start, off_t bytes, off_t *block) {
    if (flock(fd, LOCK_EX)) {
        return -1;
    }
    off_t end = lseek(fd, 0, SEEK_END);
    size_t done = 0;
    int failed = end < 0;
    if (!failed && end > INT64_MAX - bytes) {
        errno = EFBIG;
        failed = 1;
    }
    while (!failed && done < length) {
        ssize_t n = pwrite(fd, data + done, length - done, end + start + done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            if (!n) {
                errno = EIO;
            }
            failed = 1;
        } else {
            done += n;
        }
    }
    if (!failed && (!length || start + length < (size_t)bytes)) {
        failed = ftruncate(fd, end + bytes);
    }
#ifdef __APPLE__
    if (!failed && length < (size_t)bytes) {
        failed = sparse_part(fd, end, end + start, block) ||
            sparse_part(fd, end + start + length, end + bytes, block);
    }
#else
    (void)block;
#endif
    int error = errno;
    if (failed && end >= 0 && ftruncate(fd, end)) {
        error = errno;
    }
    flock(fd, LOCK_UN);
    errno = error;
    return failed ? -1 : end;
}

#include <json-c/json.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

enum { HX_BRICK_UNKNOWN = -1, HX_BRICK_ABSENT = -2 };

typedef struct {
    char *folder;
    char *path;
    char *source;
    int64_t shape[3];
    int64_t brick[3];
    int64_t grid[3];
    int64_t count;
    int64_t bytes;
    int64_t *offsets;
    off_t block;
    int data_fd;
} HxBrickStore;

void hx_brick_close(HxBrickStore *store);

static int make_directory(const char *path) {
    char *copy = strdup(path);
    if (!copy)
        return -1;
    for (char *at = copy + 1; *at; at++) {
        if (*at != '/')
            continue;
        *at = 0;
        if (mkdir(copy, 0777) && errno != EEXIST) {
            free(copy);
            return -1;
        }
        *at = '/';
    }
    int status = mkdir(copy, 0777) && errno != EEXIST ? -1 : 0;
    free(copy);
    return status;
}

static int full_transfer(int fd, void *buffer, size_t bytes, int writing) {
    char *at = buffer;
    while (bytes) {
        ssize_t n = writing ? write(fd, at, bytes) : read(fd, at, bytes);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0) {
            if (!n)
                errno = EIO;
            return -1;
        }
        at += n;
        bytes -= (size_t)n;
    }
    return 0;
}

static int json_vector(json_object *object, const char *name, int64_t *values) {
    json_object *array;
    if (!json_object_object_get_ex(object, name, &array) ||
        !json_object_is_type(array, json_type_array) || json_object_array_length(array) != 3) {
        return -1;
    }
    for (int i = 0; i < 3; i++) {
        json_object *value = json_object_array_get_idx(array, (size_t)i);
        if (!json_object_is_type(value, json_type_int)) {
            return -1;
        }
        values[i] = json_object_get_int64(value);
    }
    return 0;
}

static json_object *vector_json(const int64_t *values) {
    json_object *array = json_object_new_array_ext(3);
    for (int i = 0; array && i < 3; i++)
        json_object_array_add(array, json_object_new_int64(values[i]));
    return array;
}

static int save_meta(const HxBrickStore *store) {
    json_object *root = json_object_new_object();
    if (!root)
        return -1;
    json_object_object_add(root, "shape", vector_json(store->shape));
    json_object_object_add(root, "brick", vector_json(store->brick));
    json_object_object_add(root, "source",
        store->source ? json_object_new_string(store->source) : json_object_new_null());
    char *path = NULL, *part = NULL;
    int ok = asprintf(&path, "%s/meta.json", store->folder) >= 0 &&
        asprintf(&part, "%s/meta.%ld.part", store->folder, (long)getpid()) >= 0 &&
        json_object_to_file_ext(part, root, JSON_C_TO_STRING_PLAIN) == 0 && !rename(part, path);
    int saved = errno;
    if (!ok && part)
        unlink(part);
    free(part);
    free(path);
    json_object_put(root);
    errno = saved;
    return ok ? 0 : -1;
}

static int load_meta(HxBrickStore *store) {
    char *path = NULL;
    if (asprintf(&path, "%s/meta.json", store->folder) < 0)
        return -1;
    json_object *root = json_object_from_file(path);
    free(path);
    if (!root)
        return -1;
    json_object *source;
    int ok = !json_vector(root, "shape", store->shape) && !json_vector(root, "brick", store->brick);
    if (ok && json_object_object_get_ex(root, "source", &source) &&
        !json_object_is_type(source, json_type_null))
        store->source = strdup(json_object_get_string(source));
    json_object_put(root);
    return ok ? 0 : -1;
}

static int set_geometry(HxBrickStore *store) {
    store->count = 1;
    store->bytes = 1;
    for (int i = 0; i < 3; i++) {
        if (store->shape[i] < 1 || store->brick[i] < 1 ||
            store->shape[i] > INT64_MAX - store->brick[i] + 1 ||
            store->count >
                INT64_MAX / ((store->shape[i] + store->brick[i] - 1) / store->brick[i]) ||
            store->bytes > INT64_MAX / store->brick[i]) {
            errno = EOVERFLOW;
            return -1;
        }
        store->grid[i] = (store->shape[i] + store->brick[i] - 1) / store->brick[i];
        store->count *= store->grid[i];
        store->bytes *= store->brick[i];
    }
    if (store->count > (int64_t)(SIZE_MAX / sizeof(*store->offsets))) {
        errno = EOVERFLOW;
        return -1;
    }
    return 0;
}

static int load_offsets(HxBrickStore *store) {
    char *path = NULL;
    if (asprintf(&path, "%s/offsets.npy", store->folder) < 0)
        return -1;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    free(path);
    if (fd < 0)
        return -1;
    unsigned char prefix[12] = {0};
    int status = full_transfer(fd, prefix, 10, 0);
    if (status || memcmp(prefix, "\x93NUMPY", 6) != 0 || !prefix[6]) {
        status = -1;
        errno = EINVAL;
    } else if (prefix[6] > 1) {
        status = full_transfer(fd, prefix + 10, 2, 0);
    }
    size_t header = (size_t)prefix[8] | (size_t)prefix[9] << 8 | (size_t)prefix[10] << 16 |
        (size_t)prefix[11] << 24;
    char *text = !status && header < (1U << 20) ? malloc(header + 1) : NULL;
    if (!text && !status) {
        status = -1;
        errno = header >= (1U << 20) ? EINVAL : ENOMEM;
    }
    if (!status) {
        status = full_transfer(fd, text, header, 0);
        text[header] = 0;
    }
    char expected[128];
    snprintf(expected, sizeof(expected), "'shape': (%lld, %lld, %lld)", (long long)store->grid[0],
        (long long)store->grid[1], (long long)store->grid[2]);
    if (!status &&
        (!strstr(text, "'descr': '<i8'") || strstr(text, "True") || !strstr(text, expected))) {
        status = -1;
        errno = EINVAL;
    }
    store->offsets = !status ? malloc((size_t)store->count * sizeof(*store->offsets)) : NULL;
    if (!store->offsets && !status) {
        status = -1;
        errno = ENOMEM;
    }
    if (!status)
        status =
            full_transfer(fd, store->offsets, (size_t)store->count * sizeof(*store->offsets), 0);
    free(text);
    int saved = errno;
    close(fd);
    errno = saved;
    return status;
}

int hx_brick_save(HxBrickStore *store) {
    char header[256];
    int base = snprintf(header, sizeof(header),
        "{'descr': '<i8', 'fortran_order': False, 'shape': (%lld, %lld, %lld), }",
        (long long)store->grid[0], (long long)store->grid[1], (long long)store->grid[2]);
    if (base < 0 || (size_t)base + 2 > sizeof(header)) {
        errno = EOVERFLOW;
        return -1;
    }
    size_t length = (size_t)base + 1;
    while ((10 + length) % 64)
        length++;
    memset(header + base, ' ', length - (size_t)base);
    header[length - 1] = '\n';
    unsigned char prefix[10] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0,
        (unsigned char)(length & 255), (unsigned char)(length >> 8)};
    char *path = NULL, *part = NULL;
    int ok = asprintf(&path, "%s/offsets.npy", store->folder) >= 0 &&
        asprintf(&part, "%s/offsets.%ld.part", store->folder, (long)getpid()) >= 0;
    int fd = ok ? open(part, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666) : -1;
    ok = fd >= 0 && !full_transfer(fd, prefix, sizeof(prefix), 1) &&
        !full_transfer(fd, header, length, 1) &&
        !full_transfer(fd, store->offsets, (size_t)store->count * sizeof(*store->offsets), 1);
    int saved = errno;
    if (fd >= 0 && close(fd) && ok) {
        saved = errno;
        ok = 0;
    }
    if (ok && rename(part, path)) {
        saved = errno;
        ok = 0;
    }
    if (!ok && part)
        unlink(part);
    free(part);
    free(path);
    errno = saved;
    return ok ? 0 : -1;
}

int hx_brick_open(HxBrickStore *store, const char *folder, const int64_t *shape,
    const int64_t *brick, const char *source) {
    memset(store, 0, sizeof(*store));
    store->data_fd = -1;
    store->folder = strdup(folder);
    if (!store->folder || asprintf(&store->path, "%s/bricks.bin", folder) < 0)
        goto fail;
    char *meta = NULL;
    if (asprintf(&meta, "%s/meta.json", folder) < 0)
        goto fail;
    int exists = access(meta, F_OK) == 0;
    free(meta);
    if (!exists) {
        if (!shape || !brick) {
            errno = ENOENT;
            goto fail;
        }
        memcpy(store->shape, shape, sizeof(store->shape));
        memcpy(store->brick, brick, sizeof(store->brick));
        store->source = source ? strdup(source) : NULL;
        if (make_directory(folder) || set_geometry(store))
            goto fail;
        store->offsets = malloc((size_t)store->count * sizeof(*store->offsets));
        if (!store->offsets)
            goto fail;
        for (int64_t i = 0; i < store->count; i++)
            store->offsets[i] = HX_BRICK_UNKNOWN;
        if (save_meta(store) || hx_brick_save(store))
            goto fail;
    } else if (load_meta(store) || set_geometry(store) || load_offsets(store)) {
        goto fail;
    }
    store->data_fd = open(store->path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (store->data_fd < 0)
        goto fail;
#ifdef __APPLE__
    struct statvfs space;
    if (!fstatvfs(store->data_fd, &space)) {
        store->block = space.f_frsize;
    }
#endif
    return 0;
fail:
    hx_brick_close(store);
    return -1;
}

int hx_brick_cell(const HxBrickStore *store, int64_t z, int64_t y, int64_t x, int64_t *cell) {
    int64_t values[] = {z, y, x};
    int64_t index = 0;
    for (int i = 0; i < 3; i++) {
        if (values[i] < 0 || values[i] >= store->grid[i]) {
            errno = ERANGE;
            return -1;
        }
        index = index * store->grid[i] + values[i];
    }
    *cell = index;
    return 0;
}

int hx_brick_put_range(HxBrickStore *store, int64_t z, int64_t y, int64_t x, const void *data,
    size_t length, int64_t start) {
    int64_t cell;
    if (hx_brick_cell(store, z, y, x, &cell))
        return -1;
    if (!data) {
        store->offsets[cell] = HX_BRICK_ABSENT;
        return 0;
    }
    if (start < 0 || start > store->bytes || length > (uint64_t)(store->bytes - start)) {
        errno = EINVAL;
        return -1;
    }
    off_t offset = append_brick(store->data_fd, data, length, start, store->bytes, &store->block);
    if (offset < 0) {
        return -1;
    }
    store->offsets[cell] = offset;
    return 0;
}

int hx_brick_put(
    HxBrickStore *store, int64_t z, int64_t y, int64_t x, const void *data, size_t length) {
    return hx_brick_put_range(store, z, y, x, data, length, 0);
}

int hx_brick_map(HxBrickStore *store, const uint8_t **data, int64_t *length) {
    struct stat info;
    if (fstat(store->data_fd, &info) || info.st_size < 0)
        return -1;
    *length = info.st_size;
    if (!info.st_size) {
        *data = NULL;
        return 0;
    }
    void *mapping = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, store->data_fd, 0);
    if (mapping == MAP_FAILED)
        return -1;
    *data = mapping;
    return 0;
}

void hx_brick_close(HxBrickStore *store) {
    if (store->data_fd >= 0)
        close(store->data_fd);
    free(store->offsets);
    free(store->source);
    free(store->path);
    free(store->folder);
    memset(store, 0, sizeof(*store));
    store->data_fd = -1;
}
