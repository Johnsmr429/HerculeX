#ifndef ORACLE_INCLUDED
#define ORACLE_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"

int Oracle_count_sheets(Arena_T              arena,
                        const ComponentMesh *mesh,
                        int                  grid_size);

#endif
