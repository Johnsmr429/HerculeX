#ifndef HALO_LOADER_INCLUDED
#define HALO_LOADER_INCLUDED

#include <stdint.h>
#include "arena.h"

int HaloLoader_load(Arena_T arena,
                    const char *pred_dir,
                    const char *cube_id,
                    int cube_size,
                    int halo_voxels,
                    uint8_t **out_vol,
                    int *out_p_size,
                    int64_t out_origin[3]);

#endif
