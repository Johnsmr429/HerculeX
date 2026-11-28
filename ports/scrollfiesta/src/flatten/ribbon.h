#ifndef RIBBON_INCLUDED
#define RIBBON_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

#define RIBBON_MAX_PEEL_LAYERS 64

typedef struct {
    float  axis_point[3];
    float  axis_dir[3];
    float  slice_h;
    float  sample_h;
    float  match_r;
    float  match_ang_deg;
    int    scaffold_solve;
    int    relax_iters;
    int    final_iters;
    int    metric_iters;
    float  metric_weight;
    int    solve_threads;
    int    solve_amg;
    const char *solve_level_prefix;
    int    solve_level_write_obj;
    float  grid_u;
    int    fit_ribbon;
    int    direct_ribbon;
    int    stitch_solve_gauges;
    int    fit_cover_width;
    int    verify_fit_width;
    float  wrap_spacing;

    int    emit_global;
    int    winding_sense;
    int    pin_orient;
    double spiral_a, spiral_b;
    int reference_phi_authoritative;
    const float *reference_phi;
    const double *reference_u;
    const double *reference_v;
    const float *reference_u_confidence;
    int solve_reference_u;
    int coarse_chain_seed;
    int projective_grid;
    int coherent_claims;
    int metric_project_only;
    int preserve_input_rows;
    const double *reference_atlas_shift;
    const int32_t *reference_material_island;
    size_t reference_material_island_count;
    const int32_t *reference_island;
    size_t reference_island_count;
    const int32_t *reference_chart;
    size_t reference_chart_count;
    const int32_t *reference_owner_block;
    const int32_t *reference_relation_a;
    const int32_t *reference_relation_b;
    const double *reference_relation_support;
    size_t reference_relation_count;
    int    component_global;
    int    peel_layers;
    int    peel_layer;
    int    ownership_construction;
    int    discard_conflicting_claims;
    int    reference_anchor_gauges;
    int    v_strip_planes;
    int    radial_bridge_cut;
} RibbonOpts;

void RibbonOpts_default(RibbonOpts *opts);

enum {
    RIB_PITCH_FALLBACK  = 0,
    RIB_PITCH_ESTIMATED = 1,
    RIB_PITCH_PINNED    = 2
};

typedef struct {
    size_t first_col;
    size_t last_col;
    int32_t source_chart;
} RibbonChartRun;

enum RibbonCellCode {
    RIB_CELL_NONE = 0,
    RIB_CELL_ABC  = 1,
    RIB_CELL_BDC  = 2,
    RIB_CELL_MAIN = 3,
    RIB_CELL_ABD  = 4,
    RIB_CELL_ADC  = 8,
    RIB_CELL_ANTI = 12
};

typedef struct {
    float   *uv;
    uint8_t *uv_ok;
    float   *phi;
    int32_t *group;
    double   u_origin;

    float   *grid_pos;
    uint8_t *grid_present;
    uint8_t *grid_valid;
    float   *grid_phi;
    int32_t *grid_island;
    int32_t *grid_material;
    size_t  *grid_chart_row_offsets;
    RibbonChartRun *grid_chart_runs;
    size_t   grid_chart_run_count;
    int32_t *source_chart_island;
    size_t   source_chart_island_count;
    uint8_t *grid_quad;
    uint8_t *grid_cut_h;
    uint8_t *grid_cut_v;
    size_t   nu, nk;
    size_t   grid_layers;
    size_t   grid_layer_base;
    size_t   grid_source_layers;
    float    grid_du;
    float    grid_dv;
    double   grid_u_origin;
    double   grid_v_origin;
    int      grid_projective;

    int     n_slices;
    int     n_chains;
    int     n_closed;
    int     n_multi_slices;
    size_t  n_samples;
    size_t  n_pairs;
    size_t  n_cont_pairs;
    size_t  n_strip_runs;
    size_t  n_strip_members;
    size_t  n_strip_links;
    size_t  n_strip_links_pruned;
    double  strip_length_rms;
    double  strip_align_rms;
    size_t  strip_gauge_components;
    double  strip_gauge_span;
    double  strip_gauge_max_shift;
    size_t  bridge_cuts;
    size_t  mono_repairs;
    size_t  certificate_samples;
    size_t  certificate_clamps;
    double  certificate_drift_p95;
    double  certificate_drift_max;
    double  certificate_bound_p50;
    size_t  uv_filled;
    size_t  uv_fallback;
    size_t  uv_phase_rejects;
    size_t  uv_phase_blocked;
    size_t  uv_component_rejects;
    size_t  uv_component_blocked;
    size_t  uv_gauge_observations;
    size_t  uv_gauge_relations;
    double  uv_gauge_max_shift;
    size_t  uv_atlas_islands;
    size_t  uv_atlas_packed_islands;
    double  uv_atlas_gutter;
    double  uv_atlas_pack_saved;
    double  match_cover;
    int     w_groups;
    int     w_prior_groups;
    int     w_unreached;
    size_t  w_conflicts;
    double  pitch_used;
    int     pitch_source;
    int     n_qp_comps;
    double  reg_max_shift;
    double  spiral_a, spiral_b, spiral_r2;
    double  u_span, v_span;
    double  phi_span_turns;
    double  duds_err_mean;
    double  duds_err_max;
    long    duds_hist[6];
    size_t  grid_claim_conflicts;
    size_t  grid_claim_conflict_cells;
    size_t  grid_claim_discarded_cells;
    size_t  grid_claim_replaced;
    size_t  grid_claim_candidates;
    size_t  grid_claim_required_peels;
    size_t  grid_claim_overflow;
    size_t  grid_claim_stored;
    size_t  grid_subcell_claim_chains;
    size_t  grid_reconstruction_components;
    size_t  grid_branch_conflicts;
    size_t  grid_branch_splits;
    size_t  grid_branch_relation_cuts;
    double  grid_branch_relation_cut_support;
    size_t  grid_long_edges_rowfit;
    size_t  grid_claim_repairs;
    size_t  grid_long_edges_repaired;
    size_t  grid_row_wrap_splits;
    size_t  grid_row_wrap_same_mesh;
    size_t  grid_row_wrap_same_solve;
    size_t  grid_row_wrap_same_island;
    size_t  grid_merge_junctions;
    size_t  grid_merge_ambiguous;
    size_t  grid_merge_rejected_island;
    size_t  grid_merge_rejected_relation;
    size_t  grid_merge_rejected_du;
    size_t  grid_merge_rejected_gap;
    size_t  grid_merge_rejected_radial;
    size_t  grid_merge_rejected_phase;
    size_t  grid_fuse_same_logical;
    size_t  grid_fuse_alternate;
    size_t  grid_fuse_violations;
    double  grid_fuse_violation_max;
    size_t  grid_transition_gap;
    size_t  grid_transition_certified;
    size_t  grid_transition_uncertified;
    size_t  grid_long_edges_vfill;
    size_t  grid_local_outlier_repairs;
    size_t  grid_local_pair_repairs;
    size_t  grid_local_pair_slots;
    size_t  grid_local_run_repairs;
    size_t  grid_local_run_slots;
    size_t  grid_local_patch_repairs;
    size_t  grid_local_patch_slots;
    size_t  grid_local_patch_small_components;
    size_t  grid_local_patch_enclosed_components;
    size_t  grid_local_patch_claim_slots;
    size_t  grid_local_supported_replacements;
    size_t  grid_long_edges_local;
    size_t  grid_long_edges_smooth;
    size_t  grid_both_diagonals_long;
    size_t  grid_long_edges_diag_main;
    size_t  grid_long_edges_diag_anti;
    size_t  grid_cell_lane_hist[8];
    size_t  grid_vfill_runs;
    size_t  grid_vfill_rows;
    size_t  grid_lane_mrf_sites;
    size_t  grid_lane_mrf_edges;
    size_t  grid_lane_mrf_changed;
    size_t  grid_lane_mrf_moved;
    size_t  grid_lane_mrf_chains;
    double  grid_lane_mrf_energy_before;
    double  grid_lane_mrf_energy_after;
    size_t  grid_lane_pair_observations;
    size_t  grid_lane_pairs;
    size_t  grid_lane_pairs_dropped;
    size_t  grid_emitted_faces;
    size_t  grid_material_faces_cut;
    size_t  grid_emitted_long_edges;
    size_t  grid_cells_full;
    size_t  grid_cells_emit2;
    size_t  grid_cells_emit1;
    size_t  grid_cells_emit0;
    size_t  grid_cells_flipped;
    size_t  grid_edge_cuts_h;
    size_t  grid_edge_cuts_v;
    size_t  grid_ufill_reject_stretch;
    size_t  grid_ufill_reject_steps;
    size_t  grid_ufill_reject_crossing;
    size_t  grid_vfill_reject_rows;
    size_t  grid_vfill_reject_stretch;
    size_t  grid_vfill_reject_crossing;
    size_t  grid_vfill_run_hist[6];
    double  chain_gauge_pair_median_before;
    double  chain_gauge_pair_p95_before;
    double  chain_gauge_pair_median_after;
    double  chain_gauge_pair_p95_after;
    double  chain_gauge_shift_rms;
    double  chain_gauge_shift_p95;
    double  chain_gauge_shift_max;
} RibbonResult;

int Ribbon_run(Arena_T arena,
               const float *verts, size_t nv,
               const int32_t *faces, size_t nf,
               const RibbonOpts *opts, RibbonResult *out);

typedef struct {
    size_t vertices, faces;
    size_t supported_vertices, generated_vertices;
    size_t atlas_columns, atlas_runs, empty_columns_removed;
    size_t primary_peel_vertices, peel_promotion_threshold;
    size_t promoted_peel_runs, promoted_peel_vertices;
    size_t extra_peel_runs, extra_peel_vertices;
} RibbonWriteStats;

int Ribbon_write_obj(const char *path, const RibbonResult *result,
                     int write_obj, int v_rebase, RibbonWriteStats *stats);

int Ribbon_write_obj_promote_peels(const char *path,
                                   const RibbonResult *result,
                                   int write_obj, int v_rebase,
                                   double min_primary_share,
                                   size_t min_vertices,
                                   RibbonWriteStats *stats);

int RibbonGridIO_selftest(void);

int Ribbon_flag_bad_faces(const float *verts, size_t nv,
                          const int32_t *faces, size_t nf,
                          const float *uv, double ratio, double floor_vox,
                          double len_min, uint8_t *out_bad, size_t *out_n);

int Ribbon_map_uv(Arena_T arena,
                  const float *cverts, size_t cnv,
                  const int32_t *cfaces, size_t cnf,
                  const float *cuv,
                  const float *verts, size_t nv,
                  const int32_t *faces, size_t nf,
                  float **out_uv, uint8_t **out_ok, size_t *out_fallback);

int Ribbon_selftest(void);

#endif
