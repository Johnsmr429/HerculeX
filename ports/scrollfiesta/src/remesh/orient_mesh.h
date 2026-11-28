#ifndef ORIENT_MESH_INCLUDED
#define ORIENT_MESH_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"

int OrientMesh_consistent(Arena_T arena,
                          const float *verts, size_t nv,
                          const float *normals,
                          int32_t *faces, size_t nf,
                          size_t *out_flips,
                          size_t *out_components,
                          size_t *out_residual_same_dir);

#endif
