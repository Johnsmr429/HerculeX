#ifndef RAWTEX_LAYERS_H
#define RAWTEX_LAYERS_H

#include <stddef.h>
#include <stdint.h>

#include "rawtex_bake.h"
#include "../common/raw_sample.h"

typedef struct RawtexLayersOpts {
    int    depth;
    double step;
    double offset;
    int    tile_cols;
    int    geom_tile_rows;
    int    chunk;
    double stretch_ratio;
    double stretch_floor;
    int    seam_reach;
    double seam_tol;
    int    seam_passes;
    const char *geom_dir;
    int    apron_px;
    int    apron_fit_px;
    int    apron_cell_px;
    int    apron_step_px;
    double smooth_sigma_px;
    double smooth_cap_vox;
    double step_sigma_px;
    double step_spread_px;
    struct RawtexHandedness *hand;
    const RawtexWindow *window;
    float  *pos_sink;
    int8_t *sign_sink;
    size_t  sink_w, sink_h;
} RawtexLayersOpts;

typedef struct RawtexPieceHand {
    size_t col0, col1;
    size_t row0, row1;
    size_t px_team;
    size_t px_mirror;
} RawtexPieceHand;

typedef struct RawtexHandedness {
    RawtexPieceHand *pieces;
    size_t npieces;
} RawtexHandedness;

void RawtexHandedness_free(RawtexHandedness *h);

typedef struct RawtexLayersStats {
    size_t W, H;
    double umin, vmin;
    size_t painted_px;
    size_t filled_px;
    size_t complete_px;
    size_t multi_px;
    size_t skip_uv_faces;
    size_t apron_px;
    size_t apron_rejected;
    size_t apron_fits;
    size_t smooth_px;
    size_t smooth_capped_px;
    double smooth_shift_rms;
    double smooth_turn_mean;
    size_t step_px;
    double step_move_rms;
    size_t chunks_written;
    double seconds;
} RawtexLayersStats;

typedef struct RawtexOrientStats {
    size_t components;
    size_t flipped;
    size_t weak;
    double agree_area;
    double flipped_area;
} RawtexOrientStats;

int RawtexLayers_orient(float *normals, const float *verts, size_t nv,
                        const int32_t *faces, size_t nf, const float *inward,
                        int smooth_iters, RawtexOrientStats *st);

int RawtexLayers_normals(float **out, const float *verts, size_t nv,
                         const int32_t *faces, size_t nf, const float *inward,
                         int smooth_iters, RawtexOrientStats *st);

int RawtexLayers_write(const char *zarr_dir, const char *mask_path,
                       CubeTable *ct, const float *verts, RawtexUv uv,
                       size_t nv, const int32_t *faces, size_t nf,
                       const float *normals, const uint8_t *face_skip,
                       const RawtexLayersOpts *o, const char *attrs_json,
                       RawtexLayersStats *st);

int RawtexLayers_quadric_fit(const uint8_t *state, const float *pos, long W, long H, long cx, long cy,
                             int fit_px, int step_px, double coef[18]);

int RawtexLayers_smooth(const uint8_t *state, float *pos, float *nrm, size_t W, size_t H,
                        double sigma_px, double cap_vox, size_t c0, size_t c1,
                        size_t *npx, size_t *ncapped, double *shift2, double *turn);

int RawtexLayers_stepblend(const uint8_t *state, float *pos, float *nrm, size_t W, size_t H,
                           double sigma_px, double spread_px, size_t c0, size_t c1,
                           size_t *nmoved, double *move2);

int RawtexLayers_selftest(void);

#endif
