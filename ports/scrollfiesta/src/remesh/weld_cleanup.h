#ifndef WELD_CLEANUP_INCLUDED
#define WELD_CLEANUP_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"
#include "seam_planes.h"

typedef struct {
    double sliver_min_alt;
    double degen_area;
    double max_collapse_len;
    int    flip_max_rounds;
    int    collapse_max_rounds;
} WeldCleanupParams;

typedef struct {
    size_t n_flips;
    size_t n_collapses;
    size_t faces_in;
    size_t faces_out;
    size_t targets_in;
    size_t targets_out;
} WeldCleanupStats;

void WeldCleanup_default_params(WeldCleanupParams *p);

int WeldCleanup_process(Arena_T arena, ComponentMesh *cm,
                        const WeldCleanupParams *params,
                        WeldCleanupStats *stats);

typedef struct {
    double band;
    double collapse_below;
    double max_collapse_len;
    int    max_rounds;
    int    flip_max_rounds;
} WeldRecoarsenParams;

typedef struct {
    size_t n_collapses;
    size_t n_flips;
    size_t faces_in;
    size_t faces_out;
} WeldRecoarsenStats;

void WeldCleanup_default_recoarsen_params(WeldRecoarsenParams *p);

int WeldCleanup_recoarsen_seam(Arena_T arena, ComponentMesh *cm,
                               const SeamPlane *planes, size_t np,
                               const WeldRecoarsenParams *params,
                               WeldRecoarsenStats *st);

size_t WeldCleanup_flip_rounds(Arena_T arena, const float *verts, size_t nv,
                               int32_t *faces, size_t nf, int max_rounds);

size_t WeldCleanup_flip_rounds_masked(
    Arena_T arena, const float *verts, size_t nv,
    int32_t *faces, size_t nf, int max_rounds,
    const uint8_t *face_frozen);

size_t WeldCleanup_flip_rounds_active_masked(
    Arena_T arena, const float *verts, size_t nv,
    int32_t *faces, size_t nf, int max_rounds,
    const uint8_t *face_frozen, const uint8_t *vertex_active);

#endif
