#ifndef PRED_REJECT_INCLUDED
#define PRED_REJECT_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"

typedef struct {
    size_t  n_vox;
    size_t  fg_count;
    size_t  largest_cc;
    double  fill_frac;

    size_t  interior_count;
    double  interior_frac;
    int     max_thickness;

    int     rect_axis;
    double  rect_frac;
    int     max_rect_run;

    int     verdict;
    int     empty;
    const char *reason;
} PredRejectStats;

int PredReject_is_garbage(Arena_T arena, const uint8_t *vol,
                          int D, int H, int W, PredRejectStats *out);

int PredReject_selftest(void);

#endif
