#ifndef ORIENT_WELD_INCLUDED
#define ORIENT_WELD_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"

int OrientWeld_components(Arena_T arena,
                          const float *verts, size_t nv,
                          int32_t *faces, size_t nf,
                          float radius,
                          size_t *out_flipped);

int OrientWeld_components_axis(Arena_T arena,
                               const float *verts, size_t nv,
                               int32_t *faces, size_t nf,
                               float radius,
                               const float *axis_point,
                               const float *axis_dir,
                               size_t *out_flipped,
                               size_t *out_radial);

#endif
