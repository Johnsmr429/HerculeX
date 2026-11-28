#ifndef RIBBON_RELAX_INCLUDED
#define RIBBON_RELAX_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "fiber_field.h"

typedef struct RibbonRelaxOpts {
    double lambda_align;
    double coh_gate;
    int    sweeps;
    double movement_tolerance;
    int    line_search;
    double max_disp;
    double qc_reject;
    double max_stretch_growth;
    double stretch_guard_floor;
    int    reference_metric;
    int    fix_boundary;
    int    convex_boundary;
    int    global_injective;
    const double *guide_uv;
    double guide_weight;
    const double *guide_weight_vertex;
    const double *reference_uv;
    const uint8_t *pin;
    int    verbose;
} RibbonRelaxOpts;

void RibbonRelax_defaults(RibbonRelaxOpts *o);

typedef struct RibbonRelaxStats {
    int    iterations, converged;
    double last_max_movement;
    double stretch_mean_before, stretch_mean_after;
    double stretch_max_before,  stretch_max_after;
    double axis_err_before, axis_err_after;
    double energy_before, energy_after;
    double mean_disp, max_disp;
    double guide_rms_before, guide_rms_after;
    int    flips_before, flips_after;
    size_t n_interior, n_moved, n_fiber_faces, n_reject;
    size_t n_stretch_guard_reject;
    size_t n_convex_reject;
    size_t n_boundary_edges, n_boundary_collision_reject;
    size_t boundary_intersections_before, boundary_intersections_after;
    int    reverted;
} RibbonRelaxStats;

int RibbonRelax_run(Arena_T arena,
                    const float *verts, size_t nv,
                    const int32_t *faces, size_t nf,
                    const float *uv_in, const uint8_t *face_skip,
                    const FiberField *fib, double du, double dv,
                    const RibbonRelaxOpts *opts,
                    float *uv_out, RibbonRelaxStats *stats);

int RibbonRelax_run_double(Arena_T arena,
                           const float *verts, size_t nv,
                           const int32_t *faces, size_t nf,
                           const double *uv_in, const uint8_t *face_skip,
                           const FiberField *fib, double du, double dv,
                           const RibbonRelaxOpts *opts,
                           double *uv_out, RibbonRelaxStats *stats);

int RibbonRelax_selftest(void);

#endif
