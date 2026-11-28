#ifndef DEPTH_PEEL_INCLUDED
#define DEPTH_PEEL_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"

int DepthPeel_process(Arena_T arena, const ComponentMesh *mesh,
                      double min_gap, double gap_depth, size_t min_comp_verts,
                      ComponentMesh **out_meshes, size_t *out_count);

int DepthPeel_selftest(void);

#endif
