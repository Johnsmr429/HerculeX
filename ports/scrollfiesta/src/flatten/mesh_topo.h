#ifndef MESH_TOPO_INCLUDED
#define MESH_TOPO_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct {
    long   n_verts, n_faces, n_edges;
    long   n_boundary_edges;
    long   n_manifold_edges;
    long   n_nonmanifold_edges;
    long   n_boundary_loops;
    long   euler;
    double genus;
    long   n_unref_verts;
    int    is_disk;
} MeshTopoInfo;

int MeshTopo_analyze(Arena_T arena, const float *verts, size_t nv,
                     const int32_t *faces, size_t nf, MeshTopoInfo *out);

int MeshTopo_selftest(void);

#endif
