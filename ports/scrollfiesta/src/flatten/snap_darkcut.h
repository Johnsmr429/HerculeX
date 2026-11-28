#ifndef SNAP_DARKCUT_INCLUDED
#define SNAP_DARKCUT_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/csr.h"

typedef struct SnapDarkOpts {
    double dk;
    double lambda;
    double sigma;
    double band_hi;
} SnapDarkOpts;

int SnapDark_segment(const double *cv, const uint8_t *has, size_t nv,
                     CSR_T adj, const SnapDarkOpts *opts, uint8_t *out_dark);

#endif
