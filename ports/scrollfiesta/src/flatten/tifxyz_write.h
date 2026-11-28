#ifndef TIFXYZ_WRITE_INCLUDED
#define TIFXYZ_WRITE_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct {
    float px_per_vox;
    int   max_dim;
    float contested_tol_vox;
    int preserve_sampling;
} TifXYZOpts;

typedef struct {
    int    width, height;
    double scale_u, scale_v;
    size_t cells_filled;
    size_t cells_contested;
    double bbox_lo[3];
    double bbox_hi[3];
} TifXYZStats;

int TifXYZ_write(Arena_T arena, const char *out_dir, const char *uuid,
                 const float *verts, size_t nv,
                 const int32_t *faces, size_t nf,
                 const float *uv, const TifXYZOpts *opts,
                 TifXYZStats *out_stats);

int TifXYZ_write_double(Arena_T arena, const char *out_dir, const char *uuid,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const double *uv, const TifXYZOpts *opts,
                        TifXYZStats *out_stats);

int TifXYZ_selftest(void);

#endif
