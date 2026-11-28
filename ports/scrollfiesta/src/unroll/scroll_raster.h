#ifndef SCROLL_RASTER_INCLUDED
#define SCROLL_RASTER_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "../common/raw_sample.h"
#include "piece_set.h"

typedef struct {
    double du, dv;
    size_t band_cols;
    double range;
    int    nsteps;
    double stretch_ratio;
    double stretch_floor;
    double max_edge3d;
    double synth_max_edge3d;
    double pct_lo, pct_hi;
    int    write_diag;
    int    write_xyzmap;
    int    strip_w;
    CubeTable *ct;
    const uint8_t *face_keep;
    double lo_fixed, hi_fixed;
    size_t synth_f0;
} RasterOpts;

void RasterOpts_default(RasterOpts *o);

typedef struct {
    size_t W, H;
    double u0, v0;
    double lo, hi;
    double fill;
    size_t filled, multi;
    size_t filled_synth;
    size_t synth_nosample;
    size_t skip_uv, skip_3d;
    size_t skip_own;
    size_t n_bands;
    int    raw_loaded, raw_missing;
    double xyz_lo[3], xyz_hi[3];
} RasterStats;

int ScrollRaster_run(Arena_T arena, const PieceSet *ps,
                     const char *raw_dir, long raw_chunk,
                     const char *prefix, const RasterOpts *opts,
                     RasterStats *out);

int ScrollRaster_selftest(void);

#endif
