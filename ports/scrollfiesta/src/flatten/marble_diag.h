#ifndef MARBLE_DIAG_INCLUDED
#define MARBLE_DIAG_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "rawtex_bake.h"

typedef struct MarbleDiagOpts {
    int    stride_pixels;
    double depth_range;
    double depth_step;
    double tensor_radius;
    double score_threshold;
    int    minimum_region_cells;
} MarbleDiagOpts;

typedef struct MarbleDiagStats {
    size_t raster_width, raster_height;
    size_t audit_width, audit_height;
    size_t sampled_cells;
    size_t highlighted_cells;
    size_t highlighted_pixels;
    size_t highlighted_regions;
    double score_p50, score_p90, score_p95, score_p99;
} MarbleDiagStats;

void MarbleDiag_defaults(MarbleDiagOpts *opts);

int MarbleDiag_write(const char *prefix, const char *rawtex_path,
                     CubeTable *ct,
                     const float *verts, const float *uv, size_t nv,
                     const int32_t *faces, size_t nf,
                     const float *vertex_normals,
                     double raster_du, double raster_dv, size_t raster_max_px,
                     const RawtexWindow *window,
                     const MarbleDiagOpts *opts, MarbleDiagStats *stats);

int MarbleDiag_write_field(const char *prefix, const char *rawtex_path,
                     CubeTable *ct,
                     const float *verts, RawtexUv uv, size_t nv,
                     const int32_t *faces, size_t nf,
                     const float *vertex_normals,
                     double raster_du, double raster_dv, size_t raster_max_px,
                     const RawtexWindow *window,
                     const MarbleDiagOpts *opts, MarbleDiagStats *stats);

#endif
