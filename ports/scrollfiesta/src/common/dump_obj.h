#ifndef DUMP_OBJ_INCLUDED
#define DUMP_OBJ_INCLUDED

#include "arena.h"
#include "mesh_types.h"
#include <stddef.h>

void DumpObj_extract_cube_id(const char *input_path, char *out, size_t out_sz);

int DumpObj_ensure_dir(const char *path);

void DumpObj_write_meshes(Arena_T arena, const char *dir,
                          const char *cube_id, const char *step_tag,
                          const ComponentMesh *meshes, size_t count);

void DumpObj_write_mesh_ptrs(Arena_T arena, const char *dir,
                             const char *cube_id, const char *step_tag,
                             const ComponentMesh **meshes, size_t count);

void DumpObj_write_one_world(Arena_T arena, const char *path,
                             const char *cube_id,
                             const float *verts, size_t nv,
                             const int32_t *faces, size_t nf,
                             const float color[3]);

void DumpObj_write_points_world(Arena_T arena, const char *path,
                                const char *cube_id,
                                const float *verts, const float *normals,
                                size_t nv);

#endif
