#ifndef FIBER_FIELD_INCLUDED
#define FIBER_FIELD_INCLUDED

#include <stdint.h>
#include "../common/arena.h"

#define FIBER_NBINS 36

typedef struct FiberField {
    float   *theta;
    float   *coh;
    uint8_t *valid;
    int W, H;
    int rosy;
    double range_deg;

    double  mean_coh;
    double  frac_valid;
    double  hist[FIBER_NBINS];
    double  peak_deg;
    double  axis_frac;
    double  mean_axis_err_deg;
} FiberField;

int FiberField_compute(Arena_T arena, const uint8_t *img, int W, int H,
                       double grad_sigma, double tensor_sigma, double axis_tol_deg,
                       int rosy, FiberField *out);

int FiberField_selftest(void);

#endif
