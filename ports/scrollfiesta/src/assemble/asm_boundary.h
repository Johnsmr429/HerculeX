#ifndef ASM_BOUNDARY_INCLUDED
#define ASM_BOUNDARY_INCLUDED
#include "asm_types.h"

typedef struct AsmBoundaryStats {
    size_t runs, supported_runs, ordered_samples, short_run_samples, duplicate_samples;
    double support_length;
} AsmBoundaryStats;

int AsmBoundary_order(Arena_T arena, const AsmChart *a, const AsmChart *b,
                      AsmCorr *samples, size_t count, AsmBoundaryStats *stats);
int AsmBoundary_selftest(void);
int AsmBoundary_weights(Arena_T arena, const AsmChart *a, const AsmChart *b,
                        const AsmCorr *samples, size_t count, double *weights);
typedef struct AsmBoundaryLoops AsmBoundaryLoops;
AsmBoundaryLoops *AsmBoundary_loops(Arena_T arena, const AsmChart *c);
int AsmBoundary_weights_loops(Arena_T arena, const AsmChart *a, const AsmBoundaryLoops *la,
                              const AsmChart *b, const AsmBoundaryLoops *lb,
                              const AsmCorr *samples, size_t count, double *weights);
#endif
