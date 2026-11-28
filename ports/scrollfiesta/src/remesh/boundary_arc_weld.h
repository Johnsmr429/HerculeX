#ifndef BOUNDARY_ARC_WELD_INCLUDED
#define BOUNDARY_ARC_WELD_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct {
    double normal_dot_min;
    double tangent_dot_max;
    double ambiguity_margin;
    double max_anchor_gap;
    double min_support_length;
    size_t min_anchors;
} BoundaryArcWeldParams;

typedef struct {
    size_t boundary_vertices;
    size_t spatial_candidates;
    size_t reciprocal_pairs;
    size_t ambiguous_vertices;
    size_t arc_links;
    size_t arc_clusters;
    size_t accepted_clusters;
    size_t local_link_pairs;
    size_t accepted_pairs;
    size_t filter_tests;
    size_t filter_accepts;
    size_t chart_clusters_considered;
    size_t chart_clusters_accepted;
    size_t chart_clusters_rejected_self;
    size_t chart_clusters_rejected_nondisk;
    size_t chart_clusters_rejected_nonpath;
    size_t chart_clusters_rejected_sparse;
    size_t chart_clusters_rejected_cycle;
} BoundaryArcWeldStats;

typedef struct {
    const int32_t *chart_component;
    const uint8_t *chart_is_disk;
    size_t n_chart_components;
    int cross_chart_forest;
    int require_edge_zipper;
} BoundaryArcWeldPolicy;

void BoundaryArcWeld_default_params(BoundaryArcWeldParams *p);

int BoundaryArcWeld_process(Arena_T arena,
                            const float *verts, size_t nv,
                            int32_t *faces, size_t nf,
                            float eps,
                            const int32_t *merge_group,
                            const BoundaryArcWeldParams *params,
                            float **out_verts, size_t *out_nv,
                            size_t *out_nf,
                            BoundaryArcWeldStats *stats);

int BoundaryArcWeld_process_with_policy(
                            Arena_T arena,
                            const float *verts, size_t nv,
                            int32_t *faces, size_t nf,
                            float eps,
                            const int32_t *merge_group,
                            const BoundaryArcWeldParams *params,
                            const BoundaryArcWeldPolicy *policy,
                            float **out_verts, size_t *out_nv,
                            size_t *out_nf,
                            BoundaryArcWeldStats *stats);

int BoundaryArcWeld_selftest(void);

#endif
