#ifndef HX_CACHED_CT_H
#define HX_CACHED_CT_H

#include <stdint.h>
#include <sqlite3.h>

typedef struct {
    int64_t shape[3], chunks[3];
    uint8_t fill;
    int fd;
    sqlite3 *index;
} HxCachedCt;

int hx_cached_ct_open(HxCachedCt *ct, const char *chunks_root, const char *sample,
    const char *scan_or_url, int level);
int hx_cached_ct_read(HxCachedCt *ct, const int64_t box[6], uint8_t *pixels, int64_t *unknown);
void hx_cached_ct_close(HxCachedCt *ct);

int hx_chunk_resolve(const char *sample, const char *scan, int refresh, char **url, int *level);
int hx_chunk_fetch_box(
    const char *sample, const char *scan, int level, const int64_t box[6], int connections);

int hx_chunk_url_shape(const char *url, int level, int64_t shape[3]);
int hx_chunk_fetch_url_box(const char *url, int level, const int64_t box[6], int connections);

#endif
