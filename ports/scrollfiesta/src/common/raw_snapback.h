#ifndef RAW_SNAPBACK_INCLUDED
#define RAW_SNAPBACK_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"
#include "mesh_types.h"

int RawSnap_process(Arena_T arena,
                    ComponentMesh *meshes, size_t n_meshes,
                    const uint8_t *input_vol,
                    size_t D, size_t H, size_t W,
                    int iterations, float alpha, float dist_max);

#endif
