#ifndef SPARSE_LOCAL_GLOBAL_INCLUDED
#define SPARSE_LOCAL_GLOBAL_INCLUDED

#include "../common/arena.h"
#include <stddef.h>

typedef struct SparseLocalRow {
    int index[6];
    double value[6], lower;
} SparseLocalRow;

typedef struct SparseLocalGlobalStats {
    size_t iterations, factors, solves, penalty_changes, local_refinements;
    int converged, warm;
    double primal_residual, stationarity_residual, complementarity;
    double objective, penalty, linear_residual;
} SparseLocalGlobalStats;

int Sparse_local_global(Arena_T arena, size_t n, size_t nt,
    const int *row, const int *col, const double *h, const double *b,
    size_t nr, const SparseLocalRow *constraints,
    const double *initial, const double *initial_multipliers, int limit,
    double *out, double *out_multipliers, SparseLocalGlobalStats *stats);

int Sparse_local_global_selftest(void);

#endif
