#ifndef STRIP_METRICS_INCLUDED
#define STRIP_METRICS_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "piece_set.h"

typedef struct StripMetrics {
    double fill;
    double multi_frac;
    double dark_frac;
    double seam_dcol_mean;
    double base_dcol_mean;
    double seam_ratio;
    size_t n_seam_cols;
    size_t n_pairs_used;
    double vseam_gap_fill;
    double vseam_ratio;
    size_t n_vseam_rows;
    size_t synth_px;
    double synth_frac;
    double fill_total;
} StripMetrics;

int StripMetrics_seam_cols(Arena_T arena, const PieceSet *ps,
                           double u0, double du, size_t W,
                           int32_t **out_cols, size_t *out_n);

int StripMetrics_compute(Arena_T arena, const char *rawtex_tif,
                         const char *diagclass_tif,
                         const int32_t *seam_cols, size_t ncols,
                         int dark_thresh, double **out_col_dcol,
                         StripMetrics *out);

int StripMetrics_vseams(Arena_T arena, const char *rawtex_tif,
                        const char *diagclass_tif,
                        double v0, long chunk, StripMetrics *inout);

int StripMetrics_selftest(void);

#endif
