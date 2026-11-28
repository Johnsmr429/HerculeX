#ifndef TIFXYZ_RECTO_INCLUDED
#define TIFXYZ_RECTO_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../common/pipeline_constants.h"
#include "../common/raw_sample.h"

typedef struct TifxyzRectoOpts {
    int midline;
    int inset;
    int smooth;
    int verbose;
    size_t count_c0, count_c1;
} TifxyzRectoOpts;

typedef struct TifxyzRectoHist {
    size_t n;
    double p5, p25, p50, p75, p95;
    double mean_abs, max_abs;
    double share_over_1p5, share_over_3;
} TifxyzRectoHist;

typedef struct TifxyzRectoQ {
    size_t n;
    double p01, p05, p25, p50, p75, p95, p99;
} TifxyzRectoQ;

typedef struct TifxyzRectoPrior {
    size_t rays, rays_with_band, rays_with_recto, rays_with_next, rays_with_seam;
    double range_vox, stride;
    TifxyzRectoQ recto_offset;
    TifxyzRectoQ spacing;
    TifxyzRectoQ thickness;
    TifxyzRectoQ gap_width;
    TifxyzRectoQ gap_depth;
    TifxyzRectoQ seam_depth;
    TifxyzRectoQ slope;
    TifxyzRectoQ edge_width;
    double gate_error;
    TifxyzRectoQ trough_level, crest_level;
    double air_level, band_level, band_min_level, level_separability;
    TifxyzRectoQ gap_trough_width, seam_trough_width;
    double trough_width_split, trough_width_separability, trough_width_error;
    double search_vox, max_total_vox, min_contrast, min_edge, edge_full, sigma, thick_min, thick_max;
    int    crowded;
} TifxyzRectoPrior;

int TifxyzRecto_prior(Arena_T arena, CubeTable *ct, const float *pos, size_t W, size_t H, int stride,
                      TifxyzRectoPrior *out);

typedef struct TifxyzRectoStats {
    size_t n_nodes, n_valid, n_dir;
    size_t n_supported_first, n_in_band_first, n_in_gap_first;
    size_t n_supported_last, n_in_band_last, n_in_gap_last;
    size_t n_beyond_first;
    int    outer_iters;
    double mean_abs_step[TXR_OUTER_ITERS];
    int    pcg_iters[TXR_OUTER_ITERS];
    double pcg_resid[TXR_OUTER_ITERS];
    size_t n_capped_total;
    size_t n_filtered_first, n_filtered_last;
    TifxyzRectoHist target_first;
    TifxyzRectoHist target_last;
    TifxyzRectoHist target_first_med;
    TifxyzRectoHist target_last_med;
    TifxyzRectoHist disp;
    double slope_p99, slope_max;
    double profile_first[2 * TXR_PROFILE_HALF + 1];
    double profile_last[2 * TXR_PROFILE_HALF + 1];
    size_t profile_n;
    double prm_search, prm_max_total, prm_min_contrast, prm_min_edge, prm_edge_full, prm_sigma, prm_thick_min,
           prm_thick_max, prm_air_level, prm_band_min_level;
    int    prm_levels;
    double sec_measure, sec_solve, sec_total;
    int    windows;
} TifxyzRectoStats;

#define TXR_PROFILE_N (2 * TXR_PROFILE_HALF + 1)

int TifxyzRecto_run(Arena_T arena, CubeTable *ct, float *pos, size_t W, size_t H,
                    const TifxyzRectoOpts *opts, const TifxyzRectoPrior *prior, TifxyzRectoStats *st,
                    float *target_first, float *target_last, float *target_first_med,
                    float *target_last_med, float *disp);

typedef void (*TifxyzRectoWindowFn)(void *ctx, size_t k, size_t n_windows, size_t lc0, size_t ww, size_t H,
                                    size_t c0, size_t c1, const TifxyzRectoStats *st, const float *target_first,
                                    const float *target_last, const float *target_first_med,
                                    const float *target_last_med, const float *disp);

int TifxyzRecto_run_windows(Arena_T arena, CubeTable *ct, const float *pos_in, float *pos_out, size_t W, size_t H,
                            size_t col0, size_t col1, size_t window_cols, size_t halo,
                            const TifxyzRectoOpts *opts, const TifxyzRectoPrior *prior, TifxyzRectoStats *st,
                            TifxyzRectoWindowFn fn, void *fn_ctx);

int TifxyzRecto_selftest(void);

#endif
