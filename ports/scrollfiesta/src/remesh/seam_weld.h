#ifndef SEAM_WELD_INCLUDED
#define SEAM_WELD_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "ball_pivot.h"

int SeamWeld_bridge(Arena_T arena,
                    const float *verts, size_t nv,
                    const int32_t *faces, size_t nf,
                    float cube_size, float rho, float rho_max, float band,
                    const uint8_t *want_mask,
                    const BpaBridgeGate *gate,
                    int32_t **out_faces, size_t *out_nf,
                    size_t *out_n_bridge);

#endif
