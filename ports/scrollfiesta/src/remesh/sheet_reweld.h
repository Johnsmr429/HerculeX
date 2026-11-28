#ifndef SHEET_REWELD_INCLUDED
#define SHEET_REWELD_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "ball_pivot.h"

typedef struct {
    double overlap_r;
    double vote_weight;
    double min_score;
    double margin_frac;
    float  rho_max;
    float  band;
} SheetReweldParams;

typedef struct {
    size_t n_sheets;
    size_t n_candidates;
    size_t n_pairs;
    size_t n_faces_added;
} SheetReweldStats;

void SheetReweld_default_params(SheetReweldParams *p);

int SheetReweld_label(Arena_T arena, const int32_t *faces, size_t nf, size_t nv,
                      int32_t **out_vert_sheet, size_t *out_n_sheets);

int SheetReweld_process(Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        size_t n_phase1_bridge,
                        const int32_t *vert_sheet, size_t n_sheets,
                        float cube_size,
                        const BpaBridgeGate *gate,
                        const SheetReweldParams *params,
                        int32_t **out_faces, size_t *out_nf,
                        SheetReweldStats *stats);

int SheetReweld_selftest(void);

#endif
