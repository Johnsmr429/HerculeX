#ifndef SURFACE_SNAP_INCLUDED
#define SURFACE_SNAP_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

struct CubeTable;

typedef struct SnapOpts {
    float  axis_point[3];
    float  axis_dir[3];
    const char *raw_dir;
    long   chunk;
    double reach;
    double occ_thresh;
    double local_r;
    double gain_bias;
    double tensor_weight;
    double tensor_radius;
    double step;
    double band_frac;
    double min_gain;
    double anchor_frac;
    int    region_cap;
    int    min_region;
    double pct_lo, pct_hi;
    int    dark_thresh;
    double dark_lambda;
    double dark_sigma;
    double normal_range;
    int    normal_samples;

    int    dilate_rings;

    int    target_mode;
    int    depth_bins;
    double w_match;
    double w_close;
    double smooth_mu;
    double smooth_tau;

    int    verbose;
} SnapOpts;

void SnapOpts_default(SnapOpts *o);

enum { SNAP_TARGET_CLOSEST = 0, SNAP_TARGET_QUILT_ARGMIN = 1, SNAP_TARGET_QUILT_MRF = 2 };

enum { SNAP_GOOD = 0, SNAP_FIXABLE = 1, SNAP_CRACK = 2 };
enum { SNAPREG_FIXABLE = 0, SNAPREG_CRACK = 1, SNAPREG_ANCHORLESS = 2, SNAPREG_MIXED = 3,
       SNAPREG_REJECT = 4 };

typedef struct SnapResult {
    uint8_t *vclass;
    float   *voff;
    float   *vdir;
    float   *vgain;
    float   *vtensor;
    float   *vblock;
    float   *vcur;
    float   *vkdefect;
    int32_t *vregion;
    uint8_t *rclass;
    int32_t *rsize;
    size_t   nreg;
    size_t n_good, n_dark, n_fixable, n_crack;
    size_t n_bidir;
    size_t n_blocked;
    size_t n_reg_fixable, n_reg_crack, n_reg_anchorless, n_reg_mixed, n_reg_reject;
    size_t n_rejected;
    size_t n_overgrow;
    size_t max_region_size;
    double win_lo, win_hi;
    double quilt_cost_mean;
    double target_dist_mean;
    size_t n_quilt_fallback;
    struct CubeTable *raw_table;
} SnapResult;

int SnapDetect_run(Arena_T arena,
                   const float *verts, size_t nv,
                   const int32_t *faces, size_t nf,
                   const float *uv,
                   const SnapOpts *opts, SnapResult *out);

int Snap_selftest(void);

#endif
