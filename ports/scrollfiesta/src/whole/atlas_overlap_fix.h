#ifndef ATLAS_OVERLAP_FIX_INCLUDED
#define ATLAS_OVERLAP_FIX_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../unroll/piece_set.h"
#include "../unroll/scaffold.h"

typedef struct {
    double aspect_ratio_kill;
    size_t floater_max_faces;
    size_t floater_group_max_faces;
    double chart_qc_kill;
    double neighbour_distance;
    double neighbour_normal_dot;
    size_t neighbour_min_shared;
    int    max_shift_wraps;
    double overlap_cell_size;
    double park_margin;
    int    park_unplaceable;

    int    wind_correct;
    int    wind_only;
    double wind_residual_limit;
    int    wind_cap;
    int    phase_targets;
    int    graded_targets;
    int    phase_graded;
    int    phase_all_groups;
    int    phase_flat_lock;
    int    phase_lateral_targets;
    int    phase_regions;
    int    lock_graded;
    int    postgraded_refine;

    int    tabu;
    int    tabu_span;
    int    tabu_tenure;
    int    tabu_max_iters;
    int    tabu_stall;
    double tabu_cell;
    double tabu_mask_cell;
    const char *tabu_mask_png;
    int    tabu_no_tear_lateral;
    int    tabu_cross_group_only;
    int    tabu_exact_loop;
    int    tabu_ladder_override;
    double tabu_ladder_step;
    double tabu_ladder_u0;
    int    tabu_active_core;
    long   tabu_core_z_lo, tabu_core_z_hi;
    long   tabu_core_y_lo, tabu_core_y_hi;
    long   tabu_core_x_lo, tabu_core_x_hi;
    double lambda_ov;
    double lambda_nb;
    double lambda_rad;
    double lambda_gauge;
    double nb_unhappy_w;
    double lateral_w;
    double lateral_u_gap;
    double family_w;

    int    relayout;
    double relayout_seam_cap;
} AtlasOverlapFixOptions;

void AtlasOverlapFixOptions_default(AtlasOverlapFixOptions *opts);

typedef enum {
    ATLAS_OVERLAP_FIX_SETTLED = 0,
    ATLAS_OVERLAP_FIX_SHIFTED = 1,
    ATLAS_OVERLAP_FIX_PARKED  = 2,
    ATLAS_OVERLAP_FIX_STUCK   = 3
} AtlasOverlapFixAction;

typedef struct {
    int32_t group;
    size_t  ncharts;
    size_t  nfaces;
    size_t  nvertices;
    double  mean_radius;
    double  radius_spread;
    double  circumference;
    double  shift_u;
    double  shift_v;
    int32_t wraps;
    size_t  overlap_raw;
    size_t  overlap_before;
    size_t  overlap_after;
    size_t  intra_overlap;
    size_t  intra_wrongwind;
    int     action;
} AtlasOverlapFixGroup;

typedef struct {
    int32_t chart;
    int32_t group;
    int32_t neighbourhood;
    int32_t phase_flat;
    int32_t phase_region;
    int32_t region_offset;
    int32_t cube;
    size_t  nfaces;
    size_t  nvertices;
    double  qc_max;
    double  radius_med;
    double  u_med;
    double  w_phi0;
    double  w_rad;
    double  w_phase;
    int32_t phase_gauge;
    int32_t phase_bin;
    double  wind;
    int32_t k;
    int32_t gbin;
    double  shift_u;
    size_t  relayout_rank;
    double  relayout_metric_shift_u;
    double  relayout_shift_u;
    int64_t cells_initial;
    int64_t cells_final;
} AtlasOverlapFixChart;

typedef struct {
    int32_t iter;
    int32_t kind;
    int32_t id;
    int32_t from;
    int32_t to;
    double  d_overlap;
    double  d_neighbour;
    double  d_radial;
    double  energy_after;
} AtlasOverlapFixTabuMove;

typedef struct {
    size_t state;
    size_t exact_cross;
    size_t exact_intra;
    size_t layout_abs;
    size_t weld_jumps_25;
    size_t weld_jumps_100;
    double search_overlap;
    double neighbour;
    double radial;
    double energy;
    double exact_score;
    double weld_mean_abs;
    double weld_rms;
    double weld_max;
    int    winner;
} AtlasOverlapFixTabuCandidate;

typedef struct {
    int32_t edge;
    int32_t chart0;
    int32_t chart1;
    int32_t target;
    int32_t final_delta;
    int32_t lateral;
    int32_t family;
    double  weight;
    double  du;
    double  final_du;
    double  phase_turn;
    double  phase_residual;
} AtlasOverlapFixTabuEdge;

typedef struct {
    int32_t chart0;
    int32_t chart1;
    uint32_t face_pairs;
    int32_t edge;
    int32_t target;
    int32_t final_delta;
    int32_t lateral;
    double radius_delta;
    double min_xyz;
} AtlasOverlapFixTabuOverlap;

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
} AtlasOverlapFixRelayoutBound;

typedef struct {
    int32_t chart0;
    int32_t chart1;
    int32_t face0;
    int32_t face1;
    int32_t allowed;
    int32_t reason;
    double min_xyz;
    double radius_delta;
} AtlasOverlapFixRelayoutResidual;

typedef struct {
    AtlasOverlapFixChart *chart;
    size_t nchart;
    AtlasOverlapFixTabuMove *move;
    size_t nmove;
    const int8_t *state_k;
    size_t nstate;
    AtlasOverlapFixTabuCandidate *candidate;
    size_t ncandidate;
    AtlasOverlapFixTabuEdge *edge;
    size_t nedge;
    AtlasOverlapFixTabuOverlap *overlap;
    size_t noverlap;
    AtlasOverlapFixRelayoutBound *relayout_bound;
    size_t nrelayout_bound;
    AtlasOverlapFixRelayoutResidual *relayout_residual;
    size_t nrelayout_residual;
    const int32_t *vertex_chart;
    const double *u_post_tabu;
} AtlasOverlapFixTabuReport;

typedef struct {
    size_t charts_before_cull;
    size_t total_charts;
    size_t total_groups;
    size_t total_faces;
    size_t faces_killed;
    size_t groups_before_cull;
    size_t groups_killed_small;
    size_t faces_killed_groups;
    size_t group_size_hist[12];
    size_t faces_killed_stretch;
    size_t faces_killed_charts;
    size_t charts_killed_small;
    size_t charts_killed_qc;
    size_t chart_size_hist[10];
    size_t chart_qc_hist[9];
    size_t authentic_neighbour_edges;
    size_t largest_group_charts;

    size_t weld_residual_hist[5];
    size_t weld_edges_nonzero_wrap;
    size_t weld_phase_residual_hist[5];
    size_t weld_phase_edges_nonzero;
    size_t weld_edges_trusted;
    size_t weld_edges_untrusted;
    size_t weld_edges_oversized;
    size_t gauge_tree_edges;
    size_t gauge_cycle_edges;
    size_t gauge_cycle_consistent;
    size_t gauge_cycle_inconsistent;
    size_t gauge_islands;
    size_t charts_wind_corrected;
    int32_t wind_wraps_min;
    int32_t wind_wraps_max;

    size_t overlap_pairs_raw;
    size_t intra_overlap_pairs_raw;
    size_t overlap_pairs_before;
    size_t overlap_pairs_after;
    size_t intra_overlap_pairs;
    size_t intra_radial_hist[6];
    size_t groups_overlapping_before;
    size_t groups_overlapping_after;

    size_t settled_groups;
    size_t shifted_groups;
    size_t parked_groups;
    size_t stuck_groups;

    double park_v_base;

    size_t tabu_iterations;
    size_t tabu_moves_chart;
    size_t tabu_moves_group;
    size_t tabu_moves_nbhd;
    size_t tabu_moves_graded;
    size_t tabu_moves_region;
    size_t tabu_graded_groups;
    size_t tabu_charts_moved;
    size_t tabu_active_charts;
    size_t tabu_welds_torn;
    size_t tabu_edges_excluded;
    size_t tabu_best_iter;
    size_t tabu_hash_rebuilds;
    size_t tabu_exact_queries;
    size_t tabu_exact_cache_entries;
    size_t tabu_exact_cache_hits;
    size_t tabu_exact_cache_misses;
    size_t tabu_exact_aabb_rejects;
    size_t tabu_exact_sat_tests;
    size_t tabu_archive_states;
    size_t tabu_leaderboard;
    size_t tabu_winner;
    size_t tabu_winner_iter;
    double tabu_exact_score;
    double tabu_energy_initial;
    double tabu_energy_final;
    double tabu_ladder_step;
    double tabu_ladder_u0;
    int tabu_ladder_fixed;
    size_t tabu_ladder_pairs;
    size_t tabu_phase_gauge_islands;
    size_t tabu_phase_flat_groups;
    size_t tabu_phase_gauge_inconsistent;
    size_t tabu_lateral_edges;
    size_t tabu_phase_lateral_trusted;
    size_t tabu_phase_lateral_nonzero;
    size_t tabu_phase_regions;
    size_t tabu_phase_region_links;
    size_t tabu_lateral_torn;
    size_t tabu_family_edges;
    size_t tabu_family_torn;

    size_t relayout_edges_weld;
    size_t relayout_edges_lateral;
    size_t relayout_edges_radial;
    double relayout_weld_rms_before;
    double relayout_weld_rms_after;
    double relayout_radial_rms_before;
    double relayout_radial_rms_after;
    double relayout_metric_shift_rms;
    double relayout_metric_shift_max;
    double relayout_shift_rms;
    double relayout_shift_max;
    double relayout_atlas_width_before;
    double relayout_atlas_width_after;
    size_t relayout_moved_charts;
    size_t relayout_exact_pairs_before;
    size_t relayout_exact_pairs_after;
    size_t relayout_cross_pairs_before;
    size_t relayout_cross_pairs_after;
    size_t relayout_allowed_pairs_before;
    size_t relayout_allowed_pairs_after;
    size_t relayout_hard_pairs_before;
    size_t relayout_hard_pairs_after;
    size_t relayout_topology_limited_before;
    size_t relayout_topology_limited_after;
    size_t relayout_total_bounds;
    size_t relayout_topology_bounds;
    size_t relayout_collision_bounds_rejected;
    size_t relayout_collision_bounds;
    size_t relayout_collision_bounds_added;
    size_t relayout_outer_iterations;
    int    relayout_qp_iterations;
    int    relayout_qp_active_bounds;
    double relayout_qp_objective;
    double relayout_qp_stationarity;
    double relayout_qp_max_violation;
    int    relayout_topology_infeasible;
    int    relayout_collision_failed;
    int    relayout_failed;
    int    relayout_reverted;
    size_t intra_overlap_pairs_after;
    size_t intra_radial_hist_after[6];
} AtlasOverlapFixStats;

int AtlasOverlapFix_solve(Arena_T arena,
                          const PieceSet *ps,
                          const ScaffoldCalib *cal,
                          const float *registered_u,
                          const AtlasOverlapFixOptions *opts,
                          double *out_u,
                          double *out_v,
                          uint8_t *face_keep,
                          int32_t *out_vertex_group,
                          AtlasOverlapFixGroup **out_group,
                          size_t *out_ngroups,
                          AtlasOverlapFixTabuReport *tabu_report,
                          AtlasOverlapFixStats *stats);

int AtlasOverlapFix_selftest(void);

#endif
