#ifndef ATLAS_COLLISION_REGISTER_INCLUDED
#define ATLAS_COLLISION_REGISTER_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "atlas_register.h"

typedef struct {
    const int32_t *faces;
    size_t nfaces;
    size_t nvertices;
    const uint8_t *face_keep;
    const double *base_u;
    const double *v;
    const int32_t *vertex_chart;
    size_t ncharts;

    const float *xyz;
    const uint8_t *allowed_chart_pair;
    const double *chart_radius;
    double same_sheet_radius_tolerance;
    double same_sheet_xyz_tolerance;

    const size_t *chart_rank;

    const double *desired_shift;
    const double *desired_weight;

    const AtlasRegisterEdge *edges;
    size_t nedges;
    double topology_deviation_limit;

    double collision_margin;
    int max_outer_iterations;
} AtlasCollisionRegisterProblem;

typedef struct {
    int32_t chart_lo;
    int32_t chart_hi;
    int32_t face_lo;
    int32_t face_hi;
    double lower;
    double final_delta;
    double slack;
    uint32_t updates;
    int topology;
    int collision;
    int initial;
    int active;
} AtlasCollisionRegisterBound;

typedef struct {
    int32_t chart0;
    int32_t chart1;
    int32_t face0;
    int32_t face1;
    int allowed;
    int reason;
    double min_xyz;
    double radius_delta;
} AtlasCollisionRegisterResidual;

typedef struct {
    size_t kept_faces;
    size_t metric_edges;
    size_t rejected_edges;
    size_t total_bounds;
    size_t topology_bounds;
    size_t collision_bounds_rejected;
    size_t collision_bounds;
    size_t collision_bounds_added;
    size_t outer_iterations;
    size_t exact_pairs_before;
    size_t exact_pairs_after;
    size_t exact_cross_before;
    size_t exact_cross_after;
    size_t moved_charts;
    int qp_iterations;
    int qp_active_bounds;
    double qp_objective;
    double qp_stationarity;
    size_t exact_allowed_before;
    size_t exact_allowed_after;
    size_t exact_hard_before;
    size_t exact_hard_after;
    size_t exact_topology_limited_before;
    size_t exact_topology_limited_after;
    double qp_max_violation;
    double shift_rms;
    double shift_max;
    int solve_failed;
    int collision_failed;
    int topology_infeasible;
    const AtlasCollisionRegisterBound *bound;
    size_t nbound;
    const AtlasCollisionRegisterResidual *residual;
    size_t nresidual;
} AtlasCollisionRegisterStats;

int AtlasCollisionRegister_solve(
    Arena_T arena, const AtlasCollisionRegisterProblem *problem,
    double *out_shift, AtlasCollisionRegisterStats *stats);

int AtlasCollisionRegister_selftest(void);

#endif
