#ifndef ARC_REG_INCLUDED
#define ARC_REG_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "placed_cube.h"
#include "weld_track.h"

typedef struct {
    int    n_knots;
    double lambda_smooth;
    double lambda_ridge;
    int    cg_iters;
    int    relax_iters;
    int    final_iters;
    double sigma_floor;
    double max_warp;
    int    min_node_corr;
    int    conf_min;
    double max_mismatch;
    int    verbose;
} ArcRegOpts;

void ArcRegOpts_default(ArcRegOpts *o);

typedef struct {
    int32_t cube, gid;
    int     n_knots;
    double  phi0, dphi;
    double  delta[8];
} ArcRegWarp;

#define ARC_REG_MAX_KNOTS 8

typedef struct {
    size_t n_corr_used;
    size_t n_gated;
    size_t n_downweighted;
    size_t n_nodes;
    size_t n_scalar_nodes;
    double med_before;
    double med_after;
    double rms_before, rms_after;
    double p95_before, p95_after;
    size_t monotone_clamps;
    double max_abs_warp;
    int    reweight_rounds;
    double seconds;
} ArcRegStats;

int ArcReg_solve(Arena_T arena,
                 const WeldTrack *wt, const int32_t *wt_to_local,
                 SkinVert *const *skins, const size_t *nskin,
                 const PlacedReg *regs, size_t n_cubes,
                 double spiral_a, double spiral_b,
                 const ArcRegOpts *opts,
                 ArcRegWarp *out_warps, size_t *out_n_warps,
                 ArcRegStats *stats);

double ArcReg_eval(const ArcRegWarp *w, double phi);

int ArcReg_selftest(void);

#endif
