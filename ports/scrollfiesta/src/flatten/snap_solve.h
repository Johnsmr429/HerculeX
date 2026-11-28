#ifndef SNAP_SOLVE_INCLUDED
#define SNAP_SOLVE_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "surface_snap.h"

typedef struct SnapSolveOpts {
    double alpha;
    double w_anchor;
    int    max_iter;
    double tol;
    double occ_thresh;
    double local_r;
    double max_disp;
    int    verbose;
} SnapSolveOpts;

void SnapSolveOpts_default(SnapSolveOpts *o);

typedef struct SnapSolveResult {
    size_t n_moved;
    size_t n_reverted;
    double mean_disp;
    double max_disp;
} SnapSolveResult;

int SnapSolve_run(Arena_T arena, float *verts, size_t nv,
                  const int32_t *faces, size_t nf,
                  const SnapResult *det, const SnapSolveOpts *opts,
                  SnapSolveResult *out);

int SnapSolve_selftest(void);

#endif
