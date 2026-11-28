#ifndef CLOSEUP_ALIGN_INCLUDED
#define CLOSEUP_ALIGN_INCLUDED

#include "slim_refine.h"

enum {
    CLOSEUP_ALIGN_PROPOSED = 0,
    CLOSEUP_ALIGN_UNCHANGED = 1,
    CLOSEUP_ALIGN_NO_EVIDENCE = 2,
    CLOSEUP_ALIGN_INVALID = -1,
    CLOSEUP_ALIGN_INFEASIBLE = -2,
    CLOSEUP_ALIGN_SOLVER_FAILED = -3,
    CLOSEUP_ALIGN_TOPOLOGY = -4,
    CLOSEUP_ALIGN_MAX_TRIALS = 26
};

typedef struct CloseupAlignMetric {
    size_t faces, measured_faces, invalid_evidence_faces, inverted_faces;
    size_t protected_bad25_faces;
    double area, measured_area, confidence_mass;
    double fiber_rms_degrees;
    double sd_mean;
    double direction_energy;
    double min_det, min_sigma, max_sigma;
    double bad10_area, bad25_area;
    double protected_area, protected_bad10_area, protected_bad25_area;
} CloseupAlignMetric;

typedef struct CloseupAlignOpts {
    SlimRefineOpts slim;
    int field_initialize;
    double target_rms_degrees;
    int local_global_iterations;
    const double *local_global_bad10_limits;
    int (*projection_begin)(void *context);
    void (*projection_end)(void *context);
    void *candidate_context;
    int (*candidate_guard)(void *, size_t, const double *);
} CloseupAlignOpts;

typedef struct CloseupAlignTrial {
    int seed;
    double direction_weight;
    int solver_status, feasible, scale_nonregression, selected;
    int candidate_guard_pass;
    size_t iterations_run, accepted_steps;
    CloseupAlignMetric metric;
} CloseupAlignTrial;

typedef struct CloseupAlignStats {
    int status, input_feasible, input_manifold_disk;
    int input_topology_supported;
    size_t boundary_loops;
    int field_attempted, field_status, field_feasible;
    int selected_seed, target_met, scale_nonregression;
    size_t field_faces, trials;
    double selected_direction_weight;
    int local_global_attempted,local_global_status,local_global_termination,local_global_loss_converged;
    double local_global_initial_loss,local_global_final_loss;
    CloseupAlignMetric before, field, after;
    CloseupAlignTrial trial[CLOSEUP_ALIGN_MAX_TRIALS];
} CloseupAlignStats;

void CloseupAlign_defaults(CloseupAlignOpts *opts);

int CloseupAlign_measure(Arena_T arena, const float *xyz, size_t nv,
                         const int32_t *faces, size_t nf, const double *uv,
                         const SlimRefineDirection *directions,
                         const uint8_t *protected_faces, int orientation,
                         CloseupAlignMetric *metric);

int CloseupAlign_propose(Arena_T arena, const float *xyz, size_t nv,
                         const int32_t *faces, size_t nf, const double *uv_in,
                         const SlimRefineDirection *directions,
                         const uint8_t *protected_faces,
                         const CloseupAlignOpts *opts, double *uv_out,
                         CloseupAlignStats *stats);

int CloseupAlign_rotate(Arena_T arena, const float *xyz, size_t nv,
                        const int32_t *faces, size_t nf, const double *uv_in,
                        const SlimRefineDirection *directions,
                        const uint8_t *protected_faces,
                        const CloseupAlignOpts *opts, double *uv_out,
                        CloseupAlignStats *stats);

int CloseupAlign_selftest(void);
int CloseupAlign_rigid_selftest(void);

#endif
