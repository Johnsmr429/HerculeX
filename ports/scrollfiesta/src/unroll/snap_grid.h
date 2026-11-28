#ifndef SNAP_GRID_INCLUDED
#define SNAP_GRID_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "../common/raw_sample.h"
#include "piece_set.h"

typedef struct SnapGridOpts {
    float  axis_point[3], axis_dir[3];
    double reach, occ_thresh, local_r, gain_bias, step;
    double tensor_weight, tensor_radius;
    double band_frac, min_gain, anchor_frac;
    int    region_cap, min_region;
    double pct_lo, pct_hi;
    int    dark_thresh;
    double dark_lambda, dark_sigma;
    double normal_range;
    int    normal_samples;
    int    target_mode, depth_bins;
    double w_match, w_close, smooth_mu, smooth_tau;
    double alpha, w_anchor;
    int    cg_max_iter;
    double cg_tol, max_disp;
    int    recto_iters, recto_inner_iters;
    double recto_range, recto_max_total;
    int    threads;
    int    global_mode;
    size_t global_nv_cap;
    int    verbose;
} SnapGridOpts;

void SnapGridOpts_default(SnapGridOpts *o);

typedef struct SnapGridStats {
    size_t n_sampled, n_dark, n_fixable, n_crack, n_rejected;
    size_t nreg, n_reg_fixable, n_reg_crack, n_reg_anchorless, n_reg_mixed,
           n_reg_reject;
    size_t max_region_size;
    size_t n_moved, n_reverted;
    double mean_disp, max_disp;
    double mean_target_tensor;
    double mean_quilt_cost, mean_target_dist;
    double win_lo, win_hi;
    size_t n_cubes_run, n_cubes_gco_fallback;
    size_t n_quilt_fallback;
    size_t n_recto_supported, n_recto_moved, n_recto_reverted,
           n_recto_slope_limited;
    int global_mode_requested, global_mode_used;
    size_t global_nv_cap;
    int recto_iterations;
    double recto_mean_disp, recto_max_disp, recto_mean_gradient;
    double sec_occ, sec_cv, sec_cubes, sec_recto;
} SnapGridStats;

int SnapGrid_run(Arena_T arena, PieceSet *ps, CubeTable *ct,
                 const SnapGridOpts *opts, SnapGridStats *out);

int SnapGrid_selftest(void);

#endif
