#pragma once

static const char index_name[] = "tiles.index";
static const char magic[8] = {'H', 'X', 'T', 'I', 'L', 'E', 'S', '1'};

#include <errno.h>
#include <fcntl.h>
#include <json-c/json.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int transfer_at(int fd, void *data, size_t len, off_t offset, int writing) {
    uint8_t *bytes = data;
    while (len) {
        ssize_t done = writing ? pwrite(fd, bytes, len, offset) : pread(fd, bytes, len, offset);
        if (done < 0 && errno == EINTR) {
            continue;
        }
        if (done <= 0) {
            if (!done) {
                errno = EIO;
            }
            return 0;
        }
        bytes += done;
        len -= done;
        offset += done;
    }
    return 1;
}

static int int_pair(json_object *pair, int *first, int *second) {
    if (!json_object_is_type(pair, json_type_array) || json_object_array_length(pair) != 2) {
        return 0;
    }
    int *out[] = {first, second};
    for (size_t i = 0; i < 2; i++) {
        json_object *value = json_object_array_get_idx(pair, i);
        int64_t number = json_object_get_int64(value);
        if (!json_object_is_type(value, json_type_int) || number > INT32_MAX ||
            number < INT32_MIN) {
            return 0;
        }
        *out[i] = number;
    }
    return 1;
}

typedef struct {
    int x, y, w, h;
    size_t first;
} TileSpan;

static int tile_rectangle(json_object *metadata, size_t level, TileSpan *span) {
    json_object *grid = json_object_object_get(metadata, "grid");
    json_object *origin = json_object_object_get(metadata, "origin");
    span->x = span->y = 0;
    return json_object_is_type(grid, json_type_array) &&
        int_pair(json_object_array_get_idx(grid, level), &span->w, &span->h) && span->w >= 0 &&
        span->h >= 0 &&
        (!origin ||
            (json_object_is_type(origin, json_type_array) &&
                int_pair(json_object_array_get_idx(origin, level), &span->x, &span->y)));
}

static int same_tile(int fd, uint64_t offset, const uint8_t *bytes, size_t length) {
    uint8_t block[8192];
    while (length) {
        size_t n = length < sizeof(block) ? length : sizeof(block);
        if (!transfer_at(fd, block, n, offset, 0)) {
            return -1;
        }
        if (memcmp(block, bytes, n) != 0) {
            return 0;
        }
        offset += n;
        bytes += n;
        length -= n;
    }
    return 1;
}

typedef struct {
    const uint8_t *data;
    size_t bytes, start, cells;
    json_object *document, *layers;
    const char *name;
    int pack;
    uint64_t pack_bytes, stamp;
} TileIndex;

static void tile_index_close(TileIndex *index) {
    if (index->pack >= 0) {
        close(index->pack);
    }
    json_object_put(index->document);
    if (index->data) {
        munmap((void *)index->data, index->bytes);
    }
}

static int tile_index_open(TileIndex *index, int directory) {
    *index = (TileIndex){.pack = -1};
    int fd = openat(directory, index_name, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return errno == ENOENT ? 0 : -1;
    }
    struct stat info;
    size_t bytes = !fstat(fd, &info) && info.st_size >= 12 && (uint64_t)info.st_size <= SIZE_MAX
        ? (size_t)info.st_size
        : 0;
    uint64_t stamp = 0;
    if (bytes) {
#ifdef __APPLE__
        struct timespec modified = info.st_mtimespec;
#else
        struct timespec modified = info.st_mtim;
#endif
        stamp = (uint64_t)modified.tv_sec * 1000000000 + modified.tv_nsec;
    }
    const uint8_t *data = bytes ? mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    close(fd);
    if (data == MAP_FAILED) {
        return -1;
    }
    uint32_t length;
    memcpy(&length, data + 8, sizeof(length));
    if (length > INT32_MAX || memcmp(data, magic, sizeof(magic)) != 0) {
        munmap((void *)data, bytes);
        return -1;
    }
    size_t start = ((size_t)length + 19) & ~(size_t)7;
    json_tokener *tokener = start <= bytes ? json_tokener_new() : NULL;
    json_object *document =
        tokener ? json_tokener_parse_ex(tokener, (const char *)data + 12, length) : NULL;
    if (tokener) {
        json_tokener_free(tokener);
    }
    json_object *field = json_object_object_get(document, "pack");
    const char *name =
        json_object_is_type(field, json_type_string) ? json_object_get_string(field) : NULL;
    json_object *count = json_object_object_get(document, "cells");
    json_object *layers = json_object_object_get(document, "layers");
    int64_t n = json_object_get_int64(count);
    int pack =
        name && *name && !strchr(name, '/') && strcmp(name, ".") != 0 && strcmp(name, "..") != 0
        ? openat(directory, name, O_RDONLY | O_CLOEXEC)
        : -1;
    int ok = pack >= 0 && !fstat(pack, &info) && S_ISREG(info.st_mode) && info.st_size >= 0 &&
        json_object_is_type(layers, json_type_object) &&
        json_object_is_type(count, json_type_int) && n >= 0 && (uint64_t)n <= (bytes - start) / 12;
    *index = (TileIndex){.data = data,
        .bytes = bytes,
        .start = start,
        .document = document,
        .layers = layers,
        .name = name,
        .pack = pack,
        .stamp = stamp};
    if (!ok) {
        return -1;
    }
    index->cells = n;
    index->pack_bytes = info.st_size;
    return 1;
}

static int tile_index_span(
    const TileIndex *index, json_object *layer, size_t count, size_t *first) {
    json_object *field = json_object_object_get(layer, "first");
    int64_t start = json_object_get_int64(field);
    if (!json_object_is_type(field, json_type_int) || start < 0 || (uint64_t)start > index->cells ||
        count > index->cells - (size_t)start) {
        return 0;
    }
    *first = start;
    return 1;
}

static inline int tile_index_level(
    const TileIndex *index, json_object *layer, size_t level, TileSpan *span) {
    size_t cells = 0;
    for (size_t i = 0; i <= level; i++) {
        if (!tile_rectangle(layer, i, span) || (size_t)span->w * span->h > index->cells - cells) {
            return 0;
        }
        cells += (size_t)span->w * span->h;
    }
    if (!tile_index_span(index, layer, cells, &span->first)) {
        return 0;
    }
    span->first += cells - (size_t)span->w * span->h;
    return 1;
}

static int tile_index_blob(const TileIndex *index, size_t row, uint64_t *offset, uint32_t *size) {
    if (row >= index->cells) {
        return 0;
    }
    double position;
    const uint8_t *table = index->data + index->start;
    memcpy(&position, table + row * 8, 8);
    memcpy(size, table + index->cells * 8 + row * 4, 4);
    if (!*size) {
        *offset = 0;
        return 1;
    }
    if (!(position >= 0 && position <= 0x1p53) || (uint64_t)position + *size > index->pack_bytes) {
        return 0;
    }
    *offset = position;
    return 1;
}
