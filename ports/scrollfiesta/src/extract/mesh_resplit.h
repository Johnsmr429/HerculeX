#ifndef MESH_RESPLIT_INCLUDED
#define MESH_RESPLIT_INCLUDED

#include <stddef.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"
#include "mesh_extract.h"

typedef struct {
    const char *dir;
    const char *cube_id;
    const char *cc_stage;
    const char *mls_stage;
} MeshResplitDump;

int MeshResplit_run(Arena_T arena,
                    const ComponentMesh   *in, size_t n_in,
                    const MeshResplitCloud *clouds,
                    int                    n_threads,
                    const MeshResplitDump *dump,
                    ComponentMesh        **out, size_t *n_out,
                    MeshResplitCloud     **out_clouds);

int MeshResplit_remesh_pieces(Arena_T arena,
                              const MeshResplitCloud *cloud,
                              const ComponentMesh    *pieces, size_t n_pieces,
                              const char             *dump_tag,
                              const MeshResplitDump  *dump,
                              ComponentMesh         **out, size_t *n_out,
                              MeshResplitCloud      **out_clouds);

int MeshResplit_resurface_own(Arena_T arena, ComponentMesh *m,
                              const float cell_origin[3], int iters);

int MeshResplit_assign_selftest(void);

#endif
