#ifndef BRIDGE_CUT_INCLUDED
#define BRIDGE_CUT_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"

int BridgeCut_process(Arena_T               arena,
                      const ComponentMesh   *mesh,
                      int                    grid_size,
                      double                 timeout_sec,
                      ComponentMesh        **out_meshes,
                      size_t                *out_count);

#endif
