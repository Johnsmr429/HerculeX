#ifndef SNAP_QUILT_INCLUDED
#define SNAP_QUILT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../common/csr.h"
#include "../common/kdtree.h"
#include "../common/raw_sample.h"

typedef struct SnapQuiltOpts {
    float axis_point[3];
    float axis_dir[3];
    double reach;
    double step;
    double occ_thresh;
    double local_r;
    double band;
    double min_gain;
    int target_mode;
    int depth_bins;
    double w_match;
    double w_close;
    double smooth_mu;
    double smooth_tau;
    int verbose;
} SnapQuiltOpts;

typedef struct SnapQuiltStats {
    size_t n_dark;
    size_t n_fixable;
    size_t n_crack;
    size_t n_bidir;
    size_t n_blocked;
    size_t n_gco_fallback;
    double mean_quilt_cost;
    double mean_distance;
} SnapQuiltStats;

int SnapQuilt_select(Arena_T arena,
                     CubeTable *ct,
                     KDTree_T occ,
                     const float *occ_verts,
                     size_t occ_index_base,
                     const float *verts,
                     const float *normals,
                     size_t nv,
                     CSR_T adj,
                     const double *cv,
                     const uint8_t *has,
                     const uint8_t *dark,
                     const SnapQuiltOpts *opts,
                     uint8_t *vclass,
                     float *voff,
                     float *vdir,
                     float *vgain,
                     float *vblock,
                     SnapQuiltStats *out);

#endif
