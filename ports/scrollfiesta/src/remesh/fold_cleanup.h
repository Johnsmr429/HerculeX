#ifndef FOLD_CLEANUP_INCLUDED
#define FOLD_CLEANUP_INCLUDED

#include <stddef.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"

int FoldCleanup_process(Arena_T arena,
                        ComponentMesh *meshes, size_t n_meshes,
                        int max_passes,
                        size_t *out_faces_removed);

#endif
