#ifndef SNAP_CG_INCLUDED
#define SNAP_CG_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "arena.h"
#include "csr.h"

int SnapCG_solve(Arena_T arena, const CSR_T adj,
                 const float *w, const float *target,
                 float *verts, size_t nv,
                 int max_iter, double tol);

#endif
