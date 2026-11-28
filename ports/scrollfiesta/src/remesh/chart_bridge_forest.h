#ifndef CHART_BRIDGE_FOREST_INCLUDED
#define CHART_BRIDGE_FOREST_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct {
    size_t input_charts;
    size_t bridge_faces_in;
    size_t bridge_faces_kept;
    size_t patches;
    size_t accepted_patches;
    size_t accepted_join_patches;
    size_t accepted_growth_patches;
    size_t accepted_hole_patches;
    size_t promotion_rounds;
    size_t promoted_patches;
    size_t promoted_faces;
    size_t accepted_source_shoulder_faces;
    size_t rejected_self_or_mixed;
    size_t rejected_patch_topology;
    size_t rejected_attachment;
    size_t rejected_cycle;
    size_t rejected_edge_conflict;
    size_t rejected_orientation;
} ChartBridgeForestStats;

typedef struct {
    size_t candidate_patches;
    size_t constraints_used;
    size_t constraints_conflicted;
    size_t charts_flipped;
    size_t faces_flipped;
} ChartBridgeOrientationStats;

int ChartBridgeForest_orient_source(Arena_T arena,
                                    const float *verts, size_t nv,
                                    int32_t *faces, size_t nf,
                                    size_t input_nf,
                                    const int32_t *vertex_chart,
                                    size_t n_charts,
                                    ChartBridgeOrientationStats *stats);

int ChartBridgeForest_filter(Arena_T arena,
                             const float *verts, size_t nv,
                             const int32_t *faces, size_t nf,
                             size_t input_nf,
                             const int32_t *vertex_chart,
                             size_t n_charts,
                             int32_t **out_faces, size_t *out_nf,
                             ChartBridgeForestStats *stats);

int ChartBridgeForest_filter_with_support(
                             Arena_T arena,
                             const float *verts, size_t nv,
                             const int32_t *faces, size_t nf,
                             size_t input_nf,
                             const int32_t *vertex_chart,
                             size_t n_charts,
                             size_t support_bridge_begin,
                             size_t support_bridge_end,
                             int32_t **out_faces, size_t *out_nf,
                             ChartBridgeForestStats *stats);

int ChartBridgeForest_selftest(void);

#endif
