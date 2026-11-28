#ifndef CHART_POOL_H
#define CHART_POOL_H

#include <stddef.h>

typedef struct ChartPoolStats {
    size_t cubes;
    size_t cubes_failed;
    size_t charts;
    size_t charts_skipped;
    size_t verts, faces;
    double seconds;
} ChartPoolStats;

int ChartPool_from_store(const char *store_dir, const char *out_dir, const long *box, ChartPoolStats *st);

int ChartPool_selftest(void);

#endif
