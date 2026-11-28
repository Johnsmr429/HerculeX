#ifndef STREAM_WELD_FORMAT_INCLUDED
#define STREAM_WELD_FORMAT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "chart_zipper.h"

typedef struct {
    char cube_id[48];
    size_t nv, nf, n_charts;
    float *verts;
    int32_t *faces;
    int32_t *vertex_chart;
    int32_t *parent0, *parent1;
} StreamWeldShard;

enum {
    STREAM_WELD_CONFLICT_GEOMETRY = 1u
};

typedef struct {
    uint32_t transaction_a;
    uint32_t transaction_b;
    uint32_t reason_mask;
} StreamWeldConflict;

int StreamWeld_write_shard(const char *path, const char *cube_id,
                           const float *verts, size_t nv,
                           const int32_t *faces, size_t nf,
                           const int32_t *vertex_chart, size_t n_charts,
                           const int32_t *parent0, const int32_t *parent1);

int StreamWeld_read_shard(Arena_T arena, const char *path,
                          StreamWeldShard *out);

int StreamWeld_write_patch(const char *path,
                           const StreamWeldShard *a,
                           const StreamWeldShard *b,
                           const int32_t *faces, size_t nf,
                           const ChartZipperTransaction *tx, size_t ntx,
                           const StreamWeldConflict *conflicts,
                           size_t nconflicts);

int StreamWeldFormat_selftest(void);

#endif
