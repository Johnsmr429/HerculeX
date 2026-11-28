#ifndef CHART_LINEAGE_INCLUDED
#define CHART_LINEAGE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct {
    void *points;
    size_t n_points;
    size_t n_components;
} ChartLineage;

typedef struct {
    size_t current_components;
    size_t components_with_lineage;
    size_t ambiguous_components;
    size_t matched_vertices;
} ChartLineageStats;

int ChartLineage_capture(Arena_T arena,
                         const float *verts, size_t nv,
                         const int32_t *faces, size_t nf,
                         ChartLineage *out);

int ChartLineage_assign(Arena_T arena, const ChartLineage *lineage,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        int32_t **out_merge_group,
                        ChartLineageStats *stats);

int ChartLineage_selftest(void);

#endif
