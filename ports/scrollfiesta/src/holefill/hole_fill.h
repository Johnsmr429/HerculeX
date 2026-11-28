#ifndef HOLE_FILL_INCLUDED
#define HOLE_FILL_INCLUDED

#include "../common/arena.h"
#include <stdint.h>
#include <stddef.h>

#define HOLEFILL_CHART_PINCH_MAX_STEP_VOX 0.25f

#define HOLEFILL_CHART_MAX_CONDITION (1.25 / 0.75)

int HoleFill_process(Arena_T arena,
                     float **verts, int32_t **faces,
                     size_t *nv, size_t *nf,
                     const int cube_shape[3],
                     size_t *out_n_loops,
                     size_t *out_n_interior,
                     size_t *out_n_filled);

int HoleFill_triangle_selftest(void);

int HoleFill_process_ex(Arena_T arena,
                        float **verts, int32_t **faces,
                        size_t *nv, size_t *nf,
                        const int cube_shape[3],
                        int interior_only,
                        size_t *out_n_loops,
                        size_t *out_n_interior,
                        size_t *out_n_filled);

#endif
