#ifndef ASM_POSE_INCLUDED
#define ASM_POSE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "asm_axis.h"
#include "asm_types.h"

typedef struct AsmPoseOpts {
    int    gn_iters;
    double cauchy_c;
    double switch_off;
    double damping;
    int    verbose_prior;
    double axis_dir[3];
    const AsmAxis *axis;
} AsmPoseOpts;

typedef struct AsmPoseStats {
    size_t charts_active, rels_active;
    size_t components;
    size_t parity_conflicts;
    size_t switched_off;
    size_t z_prior_charts;
    size_t frames_from_anchor, frames_from_mean, frames_none;
    int    largest_frame;
    size_t vprior_charts;
    double vprior_fit[3];
    double vplane_p50_before, vplane_p90_before, vplane_p50_after, vplane_p90_after;
    size_t components_after_switch;
    size_t largest_component_charts;
    double largest_component_area;
    size_t degree_hist[6];
    double degree_area[6];
    double resid_p50, resid_p95, resid_max;
    double xy_p50, xy_p95;
    int    iters;
    double sec;
} AsmPoseStats;

void AsmPose_default_opts(AsmPoseOpts *o);

int AsmPose_solve(AsmRun *run, const AsmPoseOpts *o, AsmPoseStats *st);

typedef struct AsmPoseRefineStats {
    size_t charts_active, charts_free, charts_pinned;
    size_t rels, rels_parity_refused, switched;
    int    iters;
    double resid_p50, resid_p95, xy_p50, xy_p95;
    double sec;
} AsmPoseRefineStats;
int AsmPose_refine(AsmRun *run, const AsmPoseOpts *o, const int32_t *rel_idx, size_t n_rel, const uint8_t *fixed, AsmPoseRefineStats *st);

int AsmPose_selftest(void);

#endif
