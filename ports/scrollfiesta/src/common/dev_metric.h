#ifndef DEV_METRIC_INCLUDED
#define DEV_METRIC_INCLUDED

#include <stddef.h>
#include <stdint.h>

void DevMetric_compute(const float *verts, size_t nv,
                       const int32_t *faces, size_t nf,
                       double *defect, unsigned char *boundary,
                       unsigned char *used);

typedef struct {
    size_t n_interior;
    size_t n_interior_bad;
    size_t n_boundary;
    double max_abs_k;
    double sum_abs_k_interior;
    double frac_interior_bad;
} DevMetricStats;

void DevMetric_interior_stats(const float *verts, size_t nv,
                              const int32_t *faces, size_t nf,
                              double k_thresh, DevMetricStats *out);

#endif
