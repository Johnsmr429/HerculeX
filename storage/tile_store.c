#pragma once
#include <errno.h>
#include <fcntl.h>
#include <libdeflate.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "tile_index.h"

typedef struct {
    struct libdeflate_compressor *compressor;
    uint8_t *rows;
    size_t capacity;
} PngWriter;

static pthread_key_t png_key;
static pthread_once_t png_once = PTHREAD_ONCE_INIT;
static int png_key_error;

static void close_png(void *value) {
    PngWriter *w = value;
    if (w) {
        libdeflate_free_compressor(w->compressor);
        free(w->rows);
        free(w);
    }
}

static void open_png_key(void) { png_key_error = pthread_key_create(&png_key, close_png); }

static PngWriter *png_writer(size_t size) {
    if (pthread_once(&png_once, open_png_key) || png_key_error) {
        errno = EIO;
        return NULL;
    }
    PngWriter *w = pthread_getspecific(png_key);
    if (!w) {
        w = calloc(1, sizeof(*w));
        if (!w) {
            errno = ENOMEM;
            return NULL;
        }
        w->compressor = libdeflate_alloc_compressor(3);
        if (!w->compressor || pthread_setspecific(png_key, w)) {
            close_png(w);
            errno = ENOMEM;
            return NULL;
        }
    }
    if (w->capacity < size) {
        uint8_t *rows = realloc(w->rows, size);
        if (!rows) {
            errno = ENOMEM;
            return NULL;
        }
        w->rows = rows;
        w->capacity = size;
    }
    return w;
}

static void png_u32(uint8_t *out, uint32_t value) {
    out[0] = value >> 24;
    out[1] = value >> 16;
    out[2] = value >> 8;
    out[3] = value;
}

static uint8_t *png_chunk(uint8_t *out, const char *type, size_t size) {
    png_u32(out, size);
    memcpy(out + 4, type, 4);
    png_u32(out + 8 + size, libdeflate_crc32(0, out + 4, size + 4));
    return out + size + 12;
}

static int png_paeth(int a, int b, int c) {
    int x = abs(b - c), y = abs(a - c), z = abs(a + b - 2 * c);
    if (x <= y && x <= z) {
        return a;
    }
    return y <= z ? b : c;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static uint8x16_t png_paeth16(uint8x16_t left, uint8x16_t above, uint8x16_t diagonal) {
    uint8x16_t a = vabdq_u8(above, diagonal), b = vabdq_u8(left, diagonal);
    int16x8_t low = vreinterpretq_s16_u16(vsubq_u16(
        vaddl_u8(vget_low_u8(left), vget_low_u8(above)), vshll_n_u8(vget_low_u8(diagonal), 1)));
    int16x8_t high =
        vreinterpretq_s16_u16(vsubq_u16(vaddl_high_u8(left, above), vshll_high_n_u8(diagonal, 1)));
    uint8x16_t c = vqmovun_high_s16(vqmovun_s16(vabsq_s16(low)), vabsq_s16(high));
    uint8x16_t take_left = vandq_u8(vcleq_u8(a, b), vcleq_u8(a, c));
    return vbslq_u8(take_left, left, vbslq_u8(vcleq_u8(b, c), above, diagonal));
}
#endif

static void png_rows(uint8_t *dst, const uint8_t *pixels, int w, int h, ptrdiff_t stride) {
    *dst++ = 1;
    dst[0] = pixels[0];
    for (int x = 1; x < w; x++) {
        dst[x] = pixels[x] - pixels[x - 1];
    }
    dst += w;
    for (int y = 1; y < h; y++) {
        const uint8_t *above = pixels;
        pixels += stride;
        *dst++ = 4;
        dst[0] = pixels[0] - above[0];
        int x = 1;
#if defined(__aarch64__) && defined(__ARM_NEON)
        for (; x <= w - 16; x += 16) {
            uint8x16_t left = vld1q_u8(pixels + x - 1), up = vld1q_u8(above + x);
            uint8x16_t diagonal = vld1q_u8(above + x - 1), value = vld1q_u8(pixels + x);
            vst1q_u8(dst + x, vsubq_u8(value, png_paeth16(left, up, diagonal)));
        }
#endif
        for (; x < w; x++) {
            dst[x] = pixels[x] - png_paeth(pixels[x - 1], above[x], above[x - 1]);
        }
        dst += w;
    }
}

static size_t encode_grey(PngWriter *encoder, const uint8_t *pixels, int w, int h, ptrdiff_t stride,
    uint8_t *data, size_t bound) {
    size_t length = ((size_t)w + 1) * h;
    png_rows(encoder->rows, pixels, w, h, stride);
    size_t size =
        libdeflate_zlib_compress(encoder->compressor, encoder->rows, length, data + 41, bound);
    if (!size) {
        return 0;
    }
    const uint8_t signature[] = {137, 'P', 'N', 'G', 13, 10, 26, 10};
    memcpy(data, signature, sizeof(signature));
    png_u32(data + 16, w);
    png_u32(data + 20, h);
    data[24] = 8;
    memset(data + 25, 0, 4);
    uint8_t *next = png_chunk(data + 8, "IHDR", 13);
    next = png_chunk(next, "IDAT", size);
    png_chunk(next, "IEND", 0);
    return size + 57;
}

#include <inttypes.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    char *name;
    json_object *metadata;
    TileSpan *spans;
    size_t levels, count;
    double *offsets;
    uint32_t *sizes;
} TileLayer;

typedef struct {
    uint64_t hash, offset;
    uint32_t size;
} TileKey;

typedef struct {
    int directory, pack, published;
    int locking;
    pthread_mutex_t lock;
    char name[64];
    uint64_t end;
    TileLayer **layers;
    size_t used, capacity;
    TileKey *known;
    size_t keys, slots;
} TilePack;

static void tile_layer_free(TileLayer *layer) {
    if (layer) {
        free(layer->name);
        json_object_put(layer->metadata);
        free(layer->spans);
        free(layer->offsets);
        free(layer->sizes);
        free(layer);
    }
}

static void tile_pack_close(TilePack *store) {
    for (size_t i = 0; i < store->used; i++) {
        tile_layer_free(store->layers[i]);
    }
    free(store->layers);
    free(store->known);
    if (store->locking) {
        pthread_mutex_destroy(&store->lock);
    }
    if (store->pack >= 0) {
        close(store->pack);
        if (!store->published) {
            unlinkat(store->directory, store->name, 0);
        }
    }
    if (store->directory >= 0) {
        close(store->directory);
    }
}

static int tile_pack_open(TilePack *store, const char *folder) {
    *store = (TilePack){.directory = -1, .pack = -1};
    store->locking = pthread_mutex_init(&store->lock, NULL) == 0;
    if (!store->locking) {
        return 0;
    }
    store->directory = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (store->directory < 0) {
        return 0;
    }
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    for (uint64_t serial = (uint64_t)time.tv_sec * 1000000000 + time.tv_nsec;; serial++) {
        snprintf(store->name, sizeof(store->name), "tiles-%" PRIu64 ".bin", serial);
        store->pack =
            openat(store->directory, store->name, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (store->pack >= 0 || errno != EEXIST) {
            return store->pack >= 0;
        }
    }
}

static TileLayer *tile_layer_add(TilePack *store, const char *name, json_object *metadata) {
    json_object *grid = json_object_object_get(metadata, "grid");
    if (!json_object_is_type(grid, json_type_array)) {
        return NULL;
    }
    size_t levels = json_object_array_length(grid);
    if (levels > SIZE_MAX / sizeof(TileSpan)) {
        return NULL;
    }
    TileLayer *layer = calloc(1, sizeof(*layer));
    if (!layer) {
        return NULL;
    }
    layer->name = strdup(name);
    layer->metadata = json_object_get(metadata);
    layer->levels = levels;
    layer->spans = calloc(levels ? levels : 1, sizeof(*layer->spans));
    int ok = layer->name && layer->spans;
    for (size_t i = 0; ok && i < levels; i++) {
        TileSpan *span = layer->spans + i;
        span->first = layer->count;
        ok = tile_rectangle(metadata, i, span) &&
            (uint64_t)span->w * span->h <= SIZE_MAX / sizeof(double) - layer->count;
        if (ok) {
            layer->count += (size_t)span->w * span->h;
        }
    }
    if (ok && store->used == store->capacity) {
        size_t capacity = store->capacity ? 2 * store->capacity : 8;
        TileLayer **next = capacity > store->capacity && capacity <= SIZE_MAX / sizeof(*next)
            ? realloc(store->layers, capacity * sizeof(*next))
            : NULL;
        if (next) {
            store->layers = next;
            store->capacity = capacity;
        }
        ok = next != NULL;
    }
    if (ok) {
        size_t count = layer->count ? layer->count : 1;
        layer->offsets = calloc(count, sizeof(*layer->offsets));
        layer->sizes = calloc(count, sizeof(*layer->sizes));
        ok = layer->offsets && layer->sizes;
    }
    if (!ok) {
        tile_layer_free(layer);
        return NULL;
    }
    store->layers[store->used++] = layer;
    return layer;
}

static size_t tile_cell(const TileLayer *layer, size_t level, int x, int y) {
    if (level >= layer->levels) {
        return SIZE_MAX;
    }
    const TileSpan *span = layer->spans + level;
    int64_t col = (int64_t)x - span->x, row = (int64_t)y - span->y;
    return col >= 0 && row >= 0 && col < span->w && row < span->h
        ? span->first + (size_t)row * span->w + col
        : SIZE_MAX;
}

static size_t tile_key_slot(uint64_t hash, size_t capacity) {
    hash ^= hash >> 32;
    hash *= UINT64_C(0xd6e8feb86659fd93);
    return (hash ^ (hash >> 32)) & (capacity - 1);
}

static int tile_key_grow(TilePack *store) {
    size_t slots = store->slots ? 2 * store->slots : 1024;
    TileKey *keys = slots > store->slots ? calloc(slots, sizeof(*keys)) : NULL;
    if (!keys) {
        return 0;
    }
    for (size_t i = 0; i < store->slots; i++) {
        TileKey key = store->known[i];
        if (key.size) {
            size_t at = tile_key_slot(key.hash, slots);
            while (keys[at].size) {
                at = (at + 1) & (slots - 1);
            }
            keys[at] = key;
        }
    }
    free(store->known);
    store->known = keys;
    store->slots = slots;
    return 1;
}

static int tile_insert(
    TilePack *store, TileLayer *layer, size_t cell, uint8_t *data, size_t length, uint64_t hash) {
    if (store->end + length > (UINT64_C(1) << 53) ||
        (store->keys >= store->slots / 2 && !tile_key_grow(store))) {
        return 0;
    }
    size_t at = tile_key_slot(hash, store->slots);
    while (store->known[at].size) {
        TileKey key = store->known[at];
        if (key.hash == hash) {
            int same = same_tile(store->pack, key.offset, data, length);
            if (same) {
                if (same > 0) {
                    layer->offsets[cell] = key.offset;
                    layer->sizes[cell] = key.size;
                }
                return same > 0;
            }
        }
        at = (at + 1) & (store->slots - 1);
    }
    if (!transfer_at(store->pack, data, length, store->end, 1)) {
        return 0;
    }
    store->known[at] = (TileKey){.hash = hash, .offset = store->end, .size = length};
    store->keys++;
    layer->offsets[cell] = store->end;
    layer->sizes[cell] = length;
    store->end += length;
    return 1;
}

static int tile_blob(TilePack *store, TileLayer *layer, size_t cell, uint8_t *data, size_t length) {
    if (cell >= layer->count || !length || length > UINT32_MAX) {
        return 0;
    }
    uint64_t hash = ((uint64_t)libdeflate_crc32(0, data, length) << 32) | length;
    if (pthread_mutex_lock(&store->lock)) {
        return 0;
    }
    int ok = tile_insert(store, layer, cell, data, length, hash);
    pthread_mutex_unlock(&store->lock);
    return ok;
}

static int keep_tile_layers(TilePack *store) {
    TileIndex index;
    int opened = tile_index_open(&index, store->directory);
    if (opened <= 0) {
        tile_index_close(&index);
        return opened == 0;
    }
    int ok = 1;
    uint8_t *buffer = NULL;
    size_t capacity = 0;
    json_object_object_foreach(index.layers, layer_name, info_json) {
        int exists = 0;
        for (size_t i = 0; i < store->used; i++) {
            exists |= !strcmp(store->layers[i]->name, layer_name);
        }
        if (exists) {
            continue;
        }
        TileLayer *layer = tile_layer_add(store, layer_name, info_json);
        size_t first;
        if (!layer || !tile_index_span(&index, info_json, layer->count, &first)) {
            ok = 0;
            break;
        }
        for (size_t i = 0; ok && i < layer->count; i++) {
            uint64_t offset;
            uint32_t size;
            if (!tile_index_blob(&index, first + i, &offset, &size)) {
                ok = 0;
                break;
            }
            if (!size) {
                continue;
            }
            if (capacity < size) {
                uint8_t *next = realloc(buffer, size);
                if (!next) {
                    ok = 0;
                    break;
                }
                buffer = next;
                capacity = size;
            }
            ok = transfer_at(index.pack, buffer, size, (off_t)offset, 0) &&
                tile_blob(store, layer, i, buffer, size);
        }
        if (!ok) {
            break;
        }
    }
    free(buffer);
    tile_index_close(&index);
    return ok;
}

static int tile_index_write(TilePack *store, int fd) {
    json_object *document = json_object_new_object(), *layers = json_object_new_object();
    int ok = document && layers;
    size_t count = 0;
    for (size_t i = 0; ok && i < store->used; i++) {
        TileLayer *layer = store->layers[i];
        json_object_object_add(layer->metadata, "first", json_object_new_int64(count));
        ok = !json_object_object_add(layers, layer->name, json_object_get(layer->metadata)) &&
            layer->count <= INT64_MAX / 12 - count;
        count += layer->count;
    }
    if (ok) {
        json_object_object_add(document, "pack", json_object_new_string(store->name));
        json_object_object_add(document, "layers", json_object_get(layers));
        json_object_object_add(document, "cells", json_object_new_int64(count));
        const char *text = json_object_to_json_string_ext(document, JSON_C_TO_STRING_PLAIN);
        size_t len = strlen(text), start = (len + 19) & ~(size_t)7;
        uint32_t size = len;
        char zero[8] = {0};
        ok = len <= UINT32_MAX && transfer_at(fd, (void *)magic, 8, 0, 1) &&
            transfer_at(fd, &size, 4, 8, 1) && transfer_at(fd, (void *)text, len, 12, 1) &&
            transfer_at(fd, zero, start - 12 - len, len + 12, 1);
        size_t moved = 0;
        for (size_t i = 0; ok && i < store->used; i++) {
            TileLayer *layer = store->layers[i];
            ok = transfer_at(fd, layer->offsets, 8 * layer->count, start + 8 * moved, 1) &&
                transfer_at(fd, layer->sizes, 4 * layer->count, start + 8 * count + 4 * moved, 1);
            moved += layer->count;
        }
    }
    json_object_put(document);
    json_object_put(layers);
    return ok;
}

static int tile_pack_publish(TilePack *store) {
    if (flock(store->directory, LOCK_EX)) {
        return 0;
    }
    char part[80];
    snprintf(part, sizeof(part), ".%s.index.part", store->name);
    int fd = keep_tile_layers(store)
        ? openat(store->directory, part, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666)
        : -1;
    if (fd >= 0) {
        int ok = tile_index_write(store, fd);
        store->published =
            close(fd) == 0 && ok && !renameat(store->directory, part, store->directory, index_name);
        if (!store->published) {
            unlinkat(store->directory, part, 0);
        }
    }
    if (!store->published) {
        flock(store->directory, LOCK_UN);
    }
    return store->published;
}
