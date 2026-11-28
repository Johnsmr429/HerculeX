#ifndef CHART_ZIPPER_H
#define CHART_ZIPPER_H

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "ball_pivot.h"

typedef struct {
    float cube_size;
    float band;
    float max_cross_edge;
    float normal_dot_min;
    float min_coverage;
    float min_support_length;
    float ambiguity_ratio;
    float min_arc_ratio;
    float skinny_weight;
    float min_triangle_altitude;
    float conflict_gap;
    float conflict_parallel_angle_deg;
    size_t min_hits;
    size_t ball_results;
    int trace;
} ChartZipperParams;

typedef struct {
    size_t planes;
    size_t boundary_edges;
    size_t seam_boundary_edges;
    size_t chains;
    size_t open_chains;
    size_t closed_chains_opened;
    size_t closed_chains_rejected;
    size_t branched_chains_rejected;

    size_t candidates;
    size_t ambiguous_candidates;
    size_t duplicate_pair_candidates;
    size_t cycle_candidates;
    size_t overlap_candidates;
    size_t geometry_candidates;
    size_t conflict_transactions;
    size_t conflict_pairs;
    size_t postcompact_conflict_transactions;
    size_t postcompact_conflict_pairs;
    size_t balanced_candidates;
    size_t trimmed_candidates;
    size_t accepted_transactions;
    size_t bridge_faces;
    int embedded_certificate;
} ChartZipperStats;

typedef struct {
    int32_t chart_a;
    int32_t chart_b;
    size_t face_first;
    size_t face_count;
    int8_t orient_a;
    int8_t orient_b;
    uint16_t flags;
    double full_coverage;
    double span_coverage;
    double mean_gap;
    double support;
    double score;
} ChartZipperTransaction;

typedef struct {
    int32_t vertex_a;
    int32_t vertex_b;
} ChartZipperMatchSample;

typedef struct {
    int32_t chart_a;
    int32_t chart_b;
    int32_t port_a;
    int32_t port_b;
    size_t interval_a_first;
    size_t interval_a_last;
    size_t interval_b_first;
    size_t interval_b_last;
    size_t sample_first;
    size_t sample_count;
    double full_coverage;
    double span_coverage;
    double mean_gap;
    double support;
    double score;
    double order_coverage;
} ChartZipperMatch;

enum {
    CHART_ZIPPER_TRANSACTION_PHYSICAL = 1u
};

void ChartZipper_default_params(ChartZipperParams *params);

int ChartZipper_process(Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const int32_t *chart_component,
                        size_t n_chart_components,
                        const BpaBridgeGate *gate,
                        const ChartZipperParams *params,
                        int32_t **out_faces, size_t *out_nf,
                        size_t *out_n_bridge,
                        ChartZipperStats *stats);

int ChartZipper_process_with_transactions(
                        Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const int32_t *chart_component,
                        size_t n_chart_components,
                        const BpaBridgeGate *gate,
                        const ChartZipperParams *params,
                        int32_t **out_faces, size_t *out_nf,
                        size_t *out_n_bridge,
                        ChartZipperStats *stats,
                        ChartZipperTransaction **out_transactions,
                        size_t *out_n_transactions);

int ChartZipper_enumerate_transactions(
                        Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const int32_t *chart_component,
                        size_t n_chart_components,
                        const BpaBridgeGate *gate,
                        const ChartZipperParams *params,
                        int32_t **out_faces, size_t *out_nf,
                        size_t *out_n_bridge,
                        ChartZipperStats *stats,
                        ChartZipperTransaction **out_transactions,
                        size_t *out_n_transactions);

int ChartZipper_match(
                        Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const int32_t *chart_component,
                        size_t n_chart_components,
                        const BpaBridgeGate *gate,
                        const ChartZipperParams *params,
                        ChartZipperMatch **out_matches,
                        size_t *out_n_matches,
                        ChartZipperMatchSample **out_samples,
                        size_t *out_n_samples,
                        ChartZipperStats *stats);

int ChartZipper_selftest(void);

#endif
