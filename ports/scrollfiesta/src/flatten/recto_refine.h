#ifndef RECTO_REFINE_INCLUDED
#define RECTO_REFINE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../common/raw_sample.h"

typedef struct RectoRefineOpts {
    float axis_point[3];
    float axis_dir[3];
    double window_lo;
    double window_hi;
    double band_frac;
    double range;
    double sample_step;
    double edge_half;
    double min_edge_frac;
    double distance_bias;
    double normal_blend;
    double data_weight;
    double anchor_weight;
    double smooth_mu;
    double max_slope;
    double damping;
    double max_step;
    double max_total;
    double occ_thresh;
    double local_r;
    int    midline;
    double midline_min_thick;
    double midline_max_thick;
    int outer_iters;
    int inner_iters;
    double tol;
    int threads;
    int verbose;
} RectoRefineOpts;

void RectoRefineOpts_default(RectoRefineOpts *o);

typedef struct RectoRefineStats {
    size_t n_supported;
    size_t n_moved;
    size_t n_reverted;
    size_t n_slope_limited;
    int iterations;
    double mean_disp;
    double max_disp;
    double mean_abs_step;
    double mean_recto_gradient;
    double sec_total;
} RectoRefineStats;

int RectoRefine_run(Arena_T arena,
                    CubeTable *ct,
                    float *verts,
                    size_t nv,
                    const int32_t *faces,
                    size_t nf,
                    const RectoRefineOpts *opts,
                    RectoRefineStats *out);

int RectoRefine_selftest(void);

#endif
