#ifndef PATCH_REPAIR_INCLUDED
#define PATCH_REPAIR_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"

typedef struct PatchRepairStats {
    size_t n_components;
    size_t n_tear_clusters;
    size_t n_patches;
    size_t n_repaired;
    size_t n_skip_gate;
    size_t n_skip_ring;
    size_t n_skip_project;
    size_t n_skip_tri;
    size_t faces_deleted;
    size_t faces_added;
} PatchRepairStats;

int PatchRepair_repair(Arena_T arena,
                       const float *verts, size_t nv,
                       const int32_t *faces, size_t nf,
                       int32_t **out_faces, size_t *out_nf,
                       PatchRepairStats *stats);

#endif
