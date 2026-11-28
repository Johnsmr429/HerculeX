#ifndef FIBER_MAP_INCLUDED
#define FIBER_MAP_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"

typedef struct FiberMapDirection {
    double column, row, h[2], v[2], weight;
} FiberMapDirection;

typedef struct FiberMapDirections {
    size_t count;
    const FiberMapDirection *sample;
} FiberMapDirections;

int FiberMap_read_directions(Arena_T arena, const char *path, FiberMapDirections *out);

typedef struct FiberMapGrid {
    int rows, cols;
    double row0, col0, cell;
    const float *u, *v;
    double affine[6];
    double source_scale[2];
    const FiberMapDirections *directions;
    int max_iterations;
    double max_seconds;
    void *checkpoint_context;
    int (*checkpoint)(void *,const double *,size_t,int);
} FiberMapGrid;

typedef struct FiberMapResult {
    int rows, cols;
    double row0, col0, cell;
    double *uv;
} FiberMapResult;

typedef struct FiberMapStats {
    size_t source_nodes, extended_nodes, relaxed_nodes;
    size_t baseline_pinned_nodes;
    size_t unsafe_cells_before, unsafe_faces_before;
    size_t metric_faces_before, metric_faces_after, source_metric_bad, affine_metric_bad;
    int rigid_baseline;
    int passes;
    double node_correction_rms, node_correction_max;
    double seconds;
    int metric_iterations;
    double metric_fit_before, metric_fit_after;
    size_t direction_samples;
    int direction_fit;
    double direction_angle_before, direction_angle_after;
} FiberMapStats;

int FiberMap_fit(Arena_T arena, const FiberMapGrid *grid,
                  const float *xyz, const double *canvas_uv, size_t nv,
                 const int32_t *faces, size_t nf, double *output_uv,
                 FiberMapResult *result, FiberMapStats *stats);

int FiberMap_selftest(void);

#endif
