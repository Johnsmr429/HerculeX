#ifndef MESH_MANIFOLD_INCLUDED
#define MESH_MANIFOLD_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "arena.h"

typedef struct {
    size_t nv, nf;
    size_t boundary_edges;
    size_t manifold_edges;
    size_t same_dir_edges;
    size_t nm_edges;
    size_t nm_verts;
} MeshManifoldStats;

MeshManifoldStats MeshManifold_audit(Arena_T arena, size_t nv,
                                     const int32_t *faces, size_t nf);

size_t MeshManifold_mark_nonmanifold_vertices(Arena_T arena, size_t nv,
                                              const int32_t *faces, size_t nf,
                                              uint8_t *out_mask);

static inline int MeshManifold_ok(const MeshManifoldStats *s) {
    return s->nm_edges == 0 && s->nm_verts == 0;
}

int MeshManifold_selftest(Arena_T arena);

#endif
