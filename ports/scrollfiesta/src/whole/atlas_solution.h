#ifndef ATLAS_SOLUTION_INCLUDED
#define ATLAS_SOLUTION_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../unroll/piece_set.h"
#include "atlas_overlap_fix.h"

typedef struct {
    int32_t chart;
    int32_t group;
    int32_t winding;
    uint64_t rank;
    double qc_max;
    double wind;
} AtlasSolutionChart;

typedef struct {
    int32_t chart0;
    int32_t chart1;
    int32_t face0;
    int32_t face1;
    int32_t allowed;
    int32_t reason;
    double min_xyz;
    double radius_delta;
} AtlasSolutionResidual;

typedef struct {
    size_t nvertices;
    size_t nfaces;
    size_t ncharts;
    size_t nresiduals;
    uint64_t piece_fingerprint;

    double *u;
    double *v;
    uint8_t *face_keep;
    int32_t *vertex_chart;
    AtlasSolutionChart *chart;
    AtlasSolutionResidual *residual;
} AtlasSolution;

uint64_t AtlasSolution_piece_fingerprint(const PieceSet *ps);

int AtlasSolution_write(const char *path,
                        const PieceSet *ps,
                        const double *u,
                        const double *v,
                        const uint8_t *face_keep,
                        const AtlasOverlapFixTabuReport *report);

int AtlasSolution_read(Arena_T arena,
                       const char *path,
                       const PieceSet *ps,
                       AtlasSolution *out);

int AtlasSolution_rank_by_wind(Arena_T arena, AtlasSolution *solution);

int AtlasSolution_selftest(Arena_T arena, const char *path);

#endif
