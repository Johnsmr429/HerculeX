#ifndef ATLAS_RIBBON_FIT_INCLUDED
#define ATLAS_RIBBON_FIT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../unroll/piece_set.h"
#include "../unroll/scaffold.h"
#include "atlas_solution.h"

typedef enum {
    ATLAS_RIBBON_OBSERVATIONS = 0,
    ATLAS_RIBBON_POSITION = 1,
    ATLAS_RIBBON_TANGENT = 2,
    ATLAS_RIBBON_REGISTER_ONLY = 3,
    ATLAS_RIBBON_REGISTER_FIELD_ONLY = 4,
    ATLAS_RIBBON_REGISTER_FIELD_SMOOTH_ONLY = 5,
    ATLAS_RIBBON_REGISTER_COLLISION_ONLY = 6,
    ATLAS_RIBBON_REGISTERED_RIBBON = 7
} AtlasRibbonFitMode;

typedef enum {
    ATLAS_RIBBON_NODE_INVALID = 0,
    ATLAS_RIBBON_NODE_OBSERVED = 1,
    ATLAS_RIBBON_NODE_CHART_INTERP = 2,
    ATLAS_RIBBON_NODE_SEAM_INTERP = 3,
    ATLAS_RIBBON_NODE_SPIRAL_FILL = 4
} AtlasRibbonNodeProvenance;

typedef enum {
    ATLAS_RIBBON_EDGE_NONE = 0,
    ATLAS_RIBBON_EDGE_NO_SUPPORT = 1,
    ATLAS_RIBBON_EDGE_SUBGRID = 2,
    ATLAS_RIBBON_EDGE_U_GAP = 3,
    ATLAS_RIBBON_EDGE_METRIC = 4,
    ATLAS_RIBBON_EDGE_TOPOLOGY = 5,
    ATLAS_RIBBON_EDGE_CROSSING = 6,
    ATLAS_RIBBON_EDGE_METRIC_FILL = 7
} AtlasRibbonEdgeReject;

typedef struct {
    double slice_spacing;
    double observation_u_spacing;
    double fit_u_spacing;
    double local_xyz_tolerance;
    double tangent_dot_min;
    double max_fill_u;
    double max_bridge_stretch;
    double bridge_slack;
    double lambda_position;
    double lambda_smooth;
    double lambda_tangent;
    double lambda_register_vertical;
    double lambda_ladder;
    int register_sweeps;
    double collision_relaxation;
    int collision_rounds;
    int collision_sweeps;
    int collision_polish_rounds;
    AtlasRibbonFitMode mode;
} AtlasRibbonFitOptions;

typedef struct {
    int32_t row;
    int32_t column;
    int32_t chart;
    int32_t face;
    double u;
    double v;
    double p[2];
    double tangent[2];
} AtlasRibbonObservation;

typedef struct {
    int32_t row;
    int32_t column;
    int32_t chart0;
    int32_t chart1;
    int32_t face0;
    int32_t face1;
    int32_t known_reason;
    uint32_t observations;
    uint32_t charts;
    uint32_t clusters;
    int accepted;
    double u;
    double v;
    double p[2];
    double tangent[2];
    double max_cluster_separation;
} AtlasRibbonTarget;
typedef struct {
    int32_t row;
    int32_t source_column;
    int32_t chart;
    int32_t face;
    int32_t group;
    int32_t winding;
    uint64_t rank;
    uint32_t charts;
    uint32_t cluster_index;
    uint32_t cluster_count;
    double source_u;
    double v;
    double p[2];
    double tangent[2];
} AtlasRibbonLayerSample;

typedef enum {
    ATLAS_RIBBON_REGISTER_REMOTE = 1,
    ATLAS_RIBBON_REGISTER_METRIC = 2,
    ATLAS_RIBBON_REGISTER_ORDER = 3
} AtlasRibbonRegisterKind;

typedef struct {
    int32_t chart_lo;
    int32_t chart_hi;
    int32_t row;
    int32_t kind;
    uint64_t rank_lo;
    uint64_t rank_hi;
    uint64_t sample_lo;
    uint64_t sample_hi;
    uint32_t updates;
    double lower_shift_delta;
    double final_shift_delta;
    double source_u_delta;
    double required_arc;
    double endpoint_distance;
} AtlasRibbonRegisterBound;

typedef struct {
    int32_t row;
    int32_t chart_lo;
    int32_t chart_hi;
    uint64_t rank_lo;
    uint64_t rank_hi;
    int32_t face_lo;
    int32_t face_hi;
    int32_t winding_lo;
    int32_t winding_hi;
    uint32_t face_pairs;
    double v;
    double min_xyz;
    double current_delta;
    double escape_delta;
    double lower_shift_delta;
} AtlasRibbonCollisionBound;

typedef struct {
    AtlasRibbonObservation *observation;
    size_t nobservation;
    AtlasRibbonTarget *target;
    size_t ntarget;
    AtlasRibbonLayerSample *layer_sample;
    size_t nlayer_samples;

    double axis[3];
    double basis0[3];
    double basis1[3];
    double axis_point[3];

    double spiral_a;
    double spiral_b;
    int    spiral_sense;

    double observation_u0;
    double observation_du;
    double v0;
    double dv;
    size_t nrows;

    double u_min;
    double u_max;
    double v_min;
    double v_max;

    size_t input_faces;
    size_t sliced_faces;
    size_t slice_segments;
    size_t degenerate_segments;
    size_t invalid_chart_faces;
    size_t accepted_targets;
    size_t duplicate_targets;
    size_t conflict_targets;
    size_t conflict_clusters;
    size_t known_topology_conflicts;
    size_t remote_layer_samples;
} AtlasRibbonObservationSet;

typedef struct {
    int32_t row;
    size_t accepted_targets;
    size_t fitted_nodes;
    size_t supported_nodes;
    size_t curve_segments;
    size_t crossings;
    size_t crossing_cuts;
    size_t u_gap_cuts;
    size_t metric_jump_cuts;
    size_t direct_spans;
    size_t spiral_spans;
    size_t topology_cuts;
    double u_min;
    double u_max;
    double speed_mean;
    double speed_rms_error;
    double speed_max_error;
    double phase_backstep_fraction;
    double phase_max_backstep;
    double winding_span;
    double radius_pitch_median;
    size_t radius_order_tests;
    size_t radius_order_violations;
} AtlasRibbonRowMetric;

typedef struct {
    double *p;
    uint8_t *valid;
    uint8_t *provenance;
    uint8_t *support;
    uint8_t *u_edge;
    uint8_t *v_edge;
    uint8_t *u_reject;
    uint8_t *v_reject;
    size_t nrows;
    size_t ncolumns;
    double u0;
    double du;
    double v0;
    double dv;

    AtlasRibbonRowMetric *row_metric;
    double *registered_u;
    double *chart_shift;
    double *row_chart_shift;

    size_t observed_nodes;
    size_t chart_interp_nodes;
    size_t seam_interp_nodes;
    size_t spiral_fill_nodes;
    size_t fitted_nodes;
    size_t supported_nodes;
    size_t curve_segments;
    size_t row_crossings;
    size_t crossing_cuts;
    size_t u_gap_cuts;
    size_t metric_jump_cuts;
    size_t direct_spans;
    size_t spiral_spans;
    size_t topology_cuts;
    size_t winding_rows;
    size_t winding_phase_backsteps;
    size_t winding_radius_order_tests;
    size_t winding_radius_order_violations;

    size_t register_iterations;
    size_t register_constraints;
    size_t register_metric_constraints;
    size_t register_order_constraints;
    size_t register_unresolved;
    size_t register_inversions_before;
    size_t register_inversions_after;
    size_t register_local_inversions;
    int register_winding_direction;
    size_t register_smoothing_sweeps;
    double register_smoothing_max_delta;
    double register_vertical_shift_rms_before;
    double register_vertical_shift_max_before;

    size_t register_collision_rounds;
    size_t register_collision_constraints;
    size_t register_collision_face_pairs;
    size_t register_qp_failures;
    size_t register_present_chart_rows;
    double register_vertical_shift_rms;
    double register_vertical_shift_max;
    double register_local_gap_mean;
    double register_local_gap_max;
    double register_row_width_ratio_mean;
    double register_width_before;
    double register_width_after;
    double register_shift_rms;
    double register_shift_max;
    double register_max_required_arc;

    AtlasRibbonRegisterBound *register_bound;
    size_t nregister_bounds;
    size_t parameter_samples;
    double parameter_mean;
    double parameter_rms;
    double parameter_p50;
    double parameter_p90;
    double parameter_p95;
    double parameter_p99;
    double parameter_max;

    size_t chamfer_samples;
    double chamfer_mean;
    double chamfer_rms;
    double chamfer_p50;
    double chamfer_p90;
    double chamfer_p95;
    double chamfer_p99;
    double directed_hausdorff;

    size_t speed_samples;
    double speed_mean;
    double speed_rms_error;
    double speed_p95_error;
    double speed_max_error;
} AtlasRibbonFitResult;

void AtlasRibbonFitOptions_default(AtlasRibbonFitOptions *opts);

int AtlasRibbonFit_build_observations(
    Arena_T arena,
    const PieceSet *ps,
    const AtlasSolution *solution,
    const ScaffoldCalib *cal,
    const AtlasRibbonFitOptions *opts,
    AtlasRibbonObservationSet *out);

int AtlasRibbonFit_solve(
    Arena_T arena,
    const PieceSet *ps,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitOptions *opts,
    AtlasRibbonFitResult *out);

int AtlasRibbonFit_selftest(void);

int AtlasRibbonFit_refine_u_collisions(
    Arena_T scratch,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitOptions *opts,
    const AtlasRibbonCollisionBound *bounds,
    size_t nbounds,
    AtlasRibbonFitResult *fit);

int AtlasRibbonFit_refine_u_deltas(
    Arena_T scratch,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitOptions *opts,
    const AtlasRibbonCollisionBound *bounds,
    size_t nbounds,
    const double *edge_delta,
    const double *edge_weight,
    const double *anchor_target,
    double anchor_weight,
    int sweeps,
    AtlasRibbonFitResult *fit);

int AtlasRibbonFit_refresh_u_field(
    Arena_T scratch,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitOptions *opts,
    AtlasRibbonFitResult *fit);

int AtlasRibbonFit_build_registered_ribbon(
    Arena_T arena,
    const PieceSet *ps,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitOptions *opts,
    AtlasRibbonFitResult *fit);
#endif
