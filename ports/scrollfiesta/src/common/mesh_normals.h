#ifndef MESH_NORMALS_INCLUDED
#define MESH_NORMALS_INCLUDED

#include <stddef.h>
#include <stdint.h>

float *MeshNormals_compute(const float *verts, size_t nv,
                           const int32_t *faces, size_t nf);

float *MeshNormals_compute_runs(const float *verts, size_t nv, const int32_t *faces,
                                const size_t *vofs, const size_t *fofs, size_t nrun);

void MeshNormals_run(const float *verts, const int32_t *faces, size_t f0, size_t f1, size_t v0, size_t v1,
                     float *normal);

int MeshNormals_selftest(void);

#endif
