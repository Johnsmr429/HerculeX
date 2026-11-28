#ifndef ASM_PLACE_INCLUDED
#define ASM_PLACE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "asm_axis.h"
#include "asm_types.h"
#include "asm_conflict.h"

typedef struct AsmPlaceOpts {
    int    min_anchors;
    double max_rms;
    double rel_rms;
    int    max_layers;
    double layer_tol;
    int    min_turn_hits;
    double max_contra_ratio;
    double max_rot_refine;
    double max_rim_ratio;
    int crosswrap_anchor_veto;
    double stress_band;
    const char *diag_dir;
    const AsmAxis *axis;
} AsmPlaceOpts;

typedef struct AsmPlaceStats {
    int budget_exhausted;
    size_t components_in;
    size_t placed, unplaced;
    size_t mirrored;
    size_t oriented_mirrored;
    size_t sweeps;
    size_t anchors_used;
    double area_in, area_placed, area_primary;
    size_t reverted_by_audit;
    size_t hop_retries, hop_retry_placed;
    size_t v_pinned, v_refused;
    double v_shift_max;
    size_t anchors_radius;
    size_t never_attempted;
    double never_attempted_area;
    int    axis_armed;
    double axis_b;
    double v_model[3];
    double wobble_primary_p50, wobble_primary_p90;
    double placed_area_beyond30, placed_area_beyond100;
    double spine_step_p50, spine_step_p90;
    size_t spine_bins;
    double v_span;
    double outline_top_p2p, outline_bot_p2p;
    size_t outline_bins_top, outline_bins_bot;
    double outline_pred_p2p;
    double outline_r_mean;
    double axis_tilt_deg;
    size_t cuts, cut_rels_deferred;
    double rms_p50, rms_max;
    double relevel_theta;
    double growth_measured;
    size_t growth_charts;
    double sec;
    double t_loop, t_measure, t_growth, t_setup, t_count, t_mirror, t_stretch, t_collect, t_cluster, t_clusterloop, t_fit, t_audit, t_probe, t_rim, t_insert, t_cut;
    size_t remeasures, quadric_calls, quadric_cached;
    size_t seam_placed, hop_placed;
    size_t v_seam_kept;
    size_t singles; double singles_area;
    size_t singles_by_status[7]; double singles_area_by_status[7];
    size_t deferred;
    size_t deferred_comps; double deferred_area;
    size_t last_call_placed;
    size_t placed_rms30; double placed_rms30_area;
    double reverted_area;
    size_t evid_comps, evid_dropped, evid_weak, evid_short; double evid_area;
    size_t noevid_comps; double noevid_area;
} AsmPlaceStats;

void AsmPlace_default_opts(AsmPlaceOpts *o);

int AsmPlace_run(AsmRun *run, const AsmPlaceOpts *o, const AsmConflictOpts *copts, AsmPlaceStats *st);

int AsmPlace_retry_component(AsmRun *run, const uint8_t *core, const AsmPlaceOpts *o,
                            const AsmConflictOpts *copts, AsmPlaceStats *st);

int AsmPlace_selftest(void);

#endif
