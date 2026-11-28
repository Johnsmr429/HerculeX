#ifndef NORMAL_ORIENT_INCLUDED
#define NORMAL_ORIENT_INCLUDED

#include <stddef.h>
#include "../common/arena.h"

int NormalOrient_consistent(Arena_T arena,
                            const float *verts,
                            float *normals,
                            size_t nv,
                            float radius,
                            size_t *out_flips);

int NormalOrient_selftest(void);

#endif
