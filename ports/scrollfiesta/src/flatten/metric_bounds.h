#ifndef METRIC_BOUNDS_INCLUDED
#define METRIC_BOUNDS_INCLUDED

typedef struct MetricBoundsResult {
    unsigned violated;
    int exact_fallback;
} MetricBoundsResult;

#ifdef __cplusplus
extern "C" {
#endif
int MetricBounds_classify(const float xyz[9], const double uv[6],
    const double bounds[4], MetricBoundsResult *result);
int MetricBounds_selftest(void);
#ifdef __cplusplus
}
#endif
#endif
