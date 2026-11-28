#ifndef ZARR_U8_INCLUDED
#define ZARR_U8_INCLUDED

#include <stddef.h>
#include <stdint.h>

#define ZARR_U8_CACHE_SLOTS 512

typedef struct {
    char    root[1024];
    long    shape[3];
    long    chunk[3];
    uint8_t *slot_data[ZARR_U8_CACHE_SLOTS];
    struct ZuSharedChunk *slot_entry[ZARR_U8_CACHE_SLOTS];
    struct ZuSharedCache *shared;
    long    slot_key[ZARR_U8_CACHE_SLOTS][3];
    int     slot_used[ZARR_U8_CACHE_SLOTS];
    size_t  next_evict;
    size_t  misses, hits, missing;
    size_t  loads, shared_hits;
    size_t  slot_limit;
    int     pow2, csh[3];
    long    cmask[3];
    long    last_key[3];
    const uint8_t *last_data;
    int last_valid;
} ZarrU8;

typedef struct ZarrU8CacheStats {
    size_t handles, chunks, peak_chunks, bytes, peak_bytes;
    size_t loads, shared_hits;
} ZarrU8CacheStats;

int  ZarrU8_open(ZarrU8 *z, const char *root);
int  ZarrU8_clone(ZarrU8 *dst, const ZarrU8 *src);
void ZarrU8_cache_stats(const ZarrU8 *z, ZarrU8CacheStats *out);
void ZarrU8_close(ZarrU8 *z);

uint8_t ZarrU8_get(ZarrU8 *z, long iz, long iy, long ix);
void ZarrU8_get8(ZarrU8 *z, long iz, long iy, long ix, uint8_t v[8]);

uint8_t ZarrU8_sample(ZarrU8 *z, double pz, double py, double px);

int ZarrU8_present(ZarrU8 *z, long iz, long iy, long ix);

int ZarrU8_selftest(void);

#endif
