#ifndef STUDIO_CUDA_H
#define STUDIO_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[128];
    int ndim;
    int64_t shape[6];
    int64_t elements;
    const float *data;
} StudioTensor;

typedef struct {
    void *map;
    size_t map_bytes;
    StudioTensor *tensors;
    int count;
} StudioWeights;

int studio_weights_open(const char *path, StudioWeights *out);
void studio_weights_close(StudioWeights *w);
const StudioTensor *studio_weights_find(const StudioWeights *w, const char *name);

typedef struct StudioEngine StudioEngine;

typedef struct {
    int batch_size;
    int use_f444;
    int profile;
} StudioConfig;

StudioEngine *studio_engine_create(const StudioWeights *weights, StudioConfig cfg);
void studio_engine_destroy(StudioEngine *eng);
void studio_engine_forward_tiles(
    StudioEngine *eng, const float *input_ncdhw, int batch, float *output_n1hw);
void studio_engine_predict(
    StudioEngine *eng, const uint8_t *volume_dhw, int depth, int height, int width, float *out_hw);
void studio_engine_print_profile(StudioEngine *eng);

#ifdef __cplusplus
}
#endif

#endif
