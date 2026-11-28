#ifndef WINDING_MRF_INCLUDED
#define WINDING_MRF_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

typedef struct {
    double center;
    double sigma;
    double weight;
    int32_t initial_label;
    int fixed;
} WindingMRFSite;

typedef struct {
    int32_t a, b;
    int32_t target;
    double weight;
} WindingMRFEdge;

enum {
    WINDING_MRF_EQUAL = 0,
    WINDING_MRF_AT_LEAST = 1,
    WINDING_MRF_AT_MOST = 2
};

typedef struct {
    int32_t site;
    int32_t label;
    double weight;
} WindingMRFUnaryPenalty;

typedef struct {
    int32_t label_min, label_max;
    int max_iterations;
    double cost_scale;
    double unary_huber_delta;
    double confidence_temperature;
    double abstain_threshold;
} WindingMRFOptions;

typedef struct {
    size_t sites, input_edges, active_edges, unique_pairs, labels;
    size_t changed_labels, abstained_sites;
    int32_t solution_min, solution_max;
    int64_t quantized_energy_before, quantized_energy_after;
    double energy_before, energy_after;
    double mean_confidence, min_confidence;
    size_t proposed_changes, rejected_move_components, rejected_move_sites;
    size_t natural_refined_sites;
} WindingMRFStats;

void WindingMRF_default_options(WindingMRFOptions *options);

int WindingMRF_solve(
    Arena_T arena,
    const WindingMRFSite *sites, size_t nsites,
    const WindingMRFEdge *edges, size_t nedges,
    const WindingMRFOptions *options,
    int32_t **out_labels,
    float **out_confidence,
    WindingMRFStats *stats);

int WindingMRF_solve_evidence(
    Arena_T arena, const WindingMRFSite *sites, size_t nsites,
    const WindingMRFEdge *edges, const int *kinds, size_t nedges,
    const WindingMRFOptions *options,
    int32_t **out_labels, float **out_confidence, WindingMRFStats *stats);

int WindingMRF_solve_with_penalties(
    Arena_T arena,
    const WindingMRFSite *sites, size_t nsites,
    const WindingMRFEdge *edges, size_t nedges,
    const WindingMRFUnaryPenalty *penalties, size_t npenalties,
    const WindingMRFOptions *options,
    int32_t **out_labels,
    float **out_confidence,
    WindingMRFStats *stats);

int WindingMRF_selftest(void);

#endif
