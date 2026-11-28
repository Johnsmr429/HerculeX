#ifndef ASM_LAYER_CUT_INCLUDED
#define ASM_LAYER_CUT_INCLUDED

#include <stddef.h>
#include <stdio.h>

#include "asm_types.h"
#include "asm_axis.h"

typedef struct AsmLayerCutOpts {
    const AsmAxis *axis;
    int    max_hops;
    double min_radius;
    int    min_hits;
    int    max_passes;
} AsmLayerCutOpts;

typedef struct AsmLayerCutStats {
    size_t pairs;
    size_t pairs_joined;
    size_t witnesses_first;
    size_t witnesses_last;
    size_t joins_first;
    size_t dropped;
    int    passes;
    double sec;
} AsmLayerCutStats;

void AsmLayerCut_defaults(AsmLayerCutOpts *o);

int AsmLayerCut_run(AsmRun *run, const AsmLayerCutOpts *o, AsmLayerCutStats *st, FILE *ledger);

int AsmLayerCut_selftest(void);

#endif
