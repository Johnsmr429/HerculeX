#ifndef LANE_TURN_MRF_INCLUDED
#define LANE_TURN_MRF_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct {
    int32_t a, b;
    int32_t target;
    double  weight;
} LaneTurnEdge;

typedef struct {
    size_t  sites, edges, labels, rounds;
    size_t  changed, abstained, moved;
    int32_t shift_min, shift_max;
    double  energy_before, energy_after;
    double  mean_confidence;
} LaneTurnStats;

#define LANE_TURN_MAX_SHIFT 2

int LaneTurn_solve(Arena_T arena,
                   const LaneTurnEdge *edges, size_t nedges,
                   const double *lane_support, size_t nlanes,
                   double trust_weight,
                   int32_t **out_shift,
                   LaneTurnStats *stats);

int LaneTurn_selftest(void);

#endif
