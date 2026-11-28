#ifndef SLIM_REFINE_INCLUDED
#define SLIM_REFINE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct SlimRefineDirection {
    double h[3], v[3], weight;
} SlimRefineDirection;

typedef struct SlimRefineOpts {
    int iterations;
    int homotopy_stages;
    int final_iterations;
    int pcg_iterations;
    double pcg_tolerance;
    double energy_tolerance;
    double sigma_min;
    double weight_max;
    double proximal;
    int line_search;
    int use_amg;
    void *linear_context;
    int (*linear_solve)(Arena_T, void *context, size_t nv, size_t ne, const double *diagonal,
                        const int32_t *edge_a, const int32_t *edge_b,
                        const double *blocks, const double *rhs, double *result);
    int local_steps;
    size_t local_step_min_faces;
    int local_line_search;
    int local_growth_sweeps;
    int local_relaxation;
    int local_relax_boundary;
    const SlimRefineDirection *directions;
    double direction_weight;
    int direction_rigid_gauge;
    const float *guide_uv;
    const float *metric_uv;
    int guide_warm_start;
    int guard_boundary;
    int strip_pack;
    double padding;
    int threads;
    int verbose;
    const double *scaffold_uv;
    size_t scaffold_first_face;
    double scaffold_weight;
    const uint8_t *fixed_vertices;
} SlimRefineOpts;

void SlimRefine_defaults(SlimRefineOpts *opts);

int SlimRefine_open_seam_seed(Arena_T arena, const float *verts, size_t nv,
                              const int32_t *faces, size_t nf,
                              const double *uv, const double *displacement,
                              double *uv_out, double *accepted_alpha);

int SlimRefine_face_proxy(const double jacobian[4], const SlimRefineOpts *opts,
                          double *energy, double gradient[4], double block[3]);

typedef struct SlimRefineStats {
    size_t components;
    size_t edges;
    size_t boundary_edges;
    size_t metric_uv_faces;
    size_t metric_fallback_faces;
    size_t iterations_run;
    size_t homotopy_stages_run;
    size_t accepted_steps;
    size_t component_steps_accepted;
    size_t component_steps_rejected;
    size_t line_search_trials;
    double accepted_alpha_min;
    double accepted_alpha_mean;
    double accepted_alpha_max;
    size_t local_component_steps_accepted;
    size_t growth_component_steps_accepted;
    size_t growth_vertex_steps;
    double growth_vertex_alpha_mean;
    double growth_vertex_alpha_max;
    size_t relaxation_vertex_steps;
    double local_vertex_alpha_min;
    double local_vertex_alpha_mean;
    double local_vertex_alpha_max;
    size_t guide_components_accepted;
    size_t guide_components_rejected;
    size_t guide_line_search_trials;
    double guide_alpha_min;
    double guide_alpha_mean;
    double guide_alpha_max;
    double guide_energy_after;
    size_t pcg_iterations_total;
    int pcg_iterations_max;
    double pcg_relative_residual_max;
    int amg_levels;
    size_t amg_coarse_vertices;
    double energy_before;
    double energy_after;
    double objective_before;
    double objective_after;
    size_t direction_faces;
    double direction_area_fraction;
    double direction_energy_before;
    double direction_energy_after;
    double direction_angle_before;
    double direction_angle_after;
    double direction_gauge_radians;
    int direction_gauge_accepted;
    double qc_mean_before;
    double qc_mean_after;
    double qc_max_before;
    double qc_max_after;
    double min_det_before;
    double min_det_after;
    size_t flips_before;
    size_t flips_after;
    size_t boundary_intersections_before;
    size_t boundary_intersections_after;
    double atlas_width;
    double atlas_height;
    double setup_seconds;
    double guide_seconds;
    double assembly_seconds;
    double amg_seconds;
    double pcg_seconds;
    double line_search_seconds;
    double finalize_seconds;
    double total_seconds;
    int converged;
    int reverted;
} SlimRefineStats;

int SlimRefine_run_double(Arena_T arena,
                          const float *verts, size_t nv,
                          const int32_t *faces, size_t nf,
                          const double *uv_in,
                          const SlimRefineOpts *opts,
                          double *uv_out,
                          SlimRefineStats *stats);

int SlimRefine_selftest(void);

#endif
