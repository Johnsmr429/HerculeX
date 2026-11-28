#ifndef ZARR_SHARD_INCLUDED
#define ZARR_SHARD_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#else
#include "arena.h"
#endif

#define ZARR_SHARD_MAX_CHUNKS 64

typedef struct {
    FILE *file;
    size_t count, chunk_bytes;
    uint64_t offsets[ZARR_SHARD_MAX_CHUNKS];
    uint64_t lengths[ZARR_SHARD_MAX_CHUNKS];
} ZarrShard;

typedef struct {
    size_t depth, height, width, tile, rows, cols, count;
} ZarrShardArray;

typedef struct {
    size_t count, chunk_bytes;
    uint64_t bytes, seen;
    uint8_t index[16 * ZARR_SHARD_MAX_CHUNKS];
    int closed, failed;
} ZarrShardBuilder;

int ZarrShard_begin(ZarrShardBuilder *builder, size_t chunk_bytes, size_t count);
int ZarrShard_append(ZarrShardBuilder *builder, const char *path, size_t index, const uint8_t *chunk);
int ZarrShard_finish(ZarrShardBuilder *builder, const char *path);

int ZarrShard_write(const char *path, const uint8_t *chunks, size_t chunk_bytes,
                    size_t supplied, size_t count, size_t *written);

int ZarrShard_open(ZarrShard *shard, const char *path, size_t chunk_bytes, size_t count);
int ZarrShard_read(ZarrShard *shard, size_t index, uint8_t *out);
void ZarrShard_close(ZarrShard *shard);

int ZarrShard_array_meta(const char *directory, size_t depth, size_t height,
                         size_t width, size_t tile, size_t count);
int ZarrShard_array_meta_grid(const char *directory, size_t depth, size_t height,
                              size_t width, size_t tile, size_t rows, size_t cols);
#ifndef __cplusplus
int ZarrShard_array_open(Arena_T arena, const char *directory, ZarrShardArray *array);
#endif
int ZarrShard_array_load(const char *directory, ZarrShardArray *array);
int ZarrShard_merge_columns(const char *source, const char *target, size_t col0, size_t *files_written);
int ZarrShard_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
