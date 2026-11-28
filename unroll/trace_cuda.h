#ifndef TRACE_CUDA_H
#define TRACE_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int Z, Y, X, GY, GX;
    int stride, tile_z, tile_yx, tile_pad, min_cell_pts;
    int64_t box[6];
    float s_min, curv_min, min_layer_gap;
    int r_sm, r_hs, r_hj;
    float k_sm[65], k_hs[65], k_hj[65];
} TraceCudaCrest;

typedef struct {
    int za, zb, ya, yb, xa, xb;
    int pz0, pz1, py0, py1, px0, px1;
} TraceCudaTile;

typedef struct {
    double x, y, z, nx, ny, nz;
    float score, cos_w, sin_w, spacing;
    int32_t cell, count;
} TraceCudaSurfel;

typedef struct {
    double upload, kernels, binning;
    int64_t slabs, tiles, bytes;
} TraceCudaStats;

int trace_cuda_open(void);
int trace_cuda_crest_begin(const TraceCudaCrest *setup);
int trace_cuda_crest_slab(const uint8_t *planes, int pz0, int tZ, const double *centres);
int trace_cuda_crest_tile(
    const TraceCudaTile *tile, const TraceCudaSurfel **surfels, int64_t *count, int64_t *crest);
void trace_cuda_crest_end(TraceCudaStats *stats);

#ifdef __cplusplus
}
#endif

#endif
