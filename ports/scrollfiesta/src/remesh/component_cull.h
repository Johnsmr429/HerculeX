#ifndef COMPONENT_CULL_INCLUDED
#define COMPONENT_CULL_INCLUDED

#include <stddef.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"

int ComponentCull_by_area(Arena_T arena,
                          const ComponentMesh *in, size_t n_in,
                          float min_frac,
                          ComponentMesh **out, size_t *n_out,
                          size_t *n_culled);

#endif
