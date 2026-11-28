#ifndef ATLAS_RIBBON_TEXTURE_INCLUDED
#define ATLAS_RIBBON_TEXTURE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../unroll/piece_set.h"
#include "atlas_ribbon_fit.h"
#include "atlas_solution.h"

typedef struct {
    int search_columns;
    size_t min_samples;
    double min_correlation;
    double min_margin;
    double normal_range;
    int normal_steps;
} AtlasRibbonTextureOptions;

typedef struct {
    int32_t row0;
    int32_t row1;
    int32_t chart;
    int32_t group;
    int32_t winding;
    int32_t lag_columns;
    uint32_t samples;
    int accepted;
    double correlation;
    double zero_correlation;
    double margin;
    double desired_shift_delta;
    double current_shift_delta;
    double residual;
} AtlasRibbonTexturePair;

typedef struct {
    AtlasRibbonTexturePair *pair;
    size_t npairs;
    size_t accepted_pairs;
    size_t sampled_layer_points;
    size_t missing_layer_points;
    int cubes_loaded;
    int cubes_missing;
    double accepted_correlation_mean;
    double accepted_margin_mean;
    double current_delta_rms;
    double desired_delta_rms;
    double residual_rms;
    double residual_p95;
    double residual_max;
} AtlasRibbonTextureResult;

void AtlasRibbonTextureOptions_default(AtlasRibbonTextureOptions *opts);

int AtlasRibbonTexture_measure(
    Arena_T arena,
    const PieceSet *ps,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitResult *fit,
    const char *raw_dir,
    const AtlasRibbonTextureOptions *opts,
    AtlasRibbonTextureResult *out);

int AtlasRibbonTexture_refresh(
    Arena_T scratch,
    const AtlasSolution *solution,
    const AtlasRibbonObservationSet *observations,
    const AtlasRibbonFitResult *fit,
    AtlasRibbonTextureResult *phase);

int AtlasRibbonTexture_selftest(void);

#endif
