#ifndef MANIFOLD_GUARD_INCLUDED
#define MANIFOLD_GUARD_INCLUDED

#include <stddef.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"

typedef struct {
    size_t nm_edges_resolved;
    size_t faces_deleted;
    size_t pinch_splits;
    size_t orient_flips;
} ManifoldGuardStats;

int ManifoldGuard_process(Arena_T arena, ComponentMesh *meshes, size_t n_meshes,
                          int reorient, ManifoldGuardStats *out);

int ManifoldGuard_process_ex(Arena_T arena,
                             ComponentMesh *meshes, size_t n_meshes,
                             int reorient, int allow_vertex_splits,
                             ManifoldGuardStats *out);

int ManifoldGuard_selftest(Arena_T arena);

#endif
