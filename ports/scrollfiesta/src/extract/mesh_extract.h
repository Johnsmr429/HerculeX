#ifndef MESH_EXTRACT_INCLUDED
#define MESH_EXTRACT_INCLUDED

#include <stdint.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"
#include "pred_recto.h"

typedef struct {
    int32_t label;
    int32_t count;
} CompInfo;

typedef struct {
    const float *orig_pts;
    const float *lop_pts;
    float        cell_origin[3];
    size_t       n;
} MeshResplitCloud;

int cc_label_3d(Arena_T arena, const uint8_t *vol,
                int D, int H, int W,
                int32_t *labels,
                CompInfo **out_comps, int32_t *out_n_comps);

int MeshExtract_run(Arena_T          arena,
                    const char      *tiff_path,
                    const char      *pred_dir,
                    int              halo_voxels,
                    size_t           cube_D,
                    size_t           cube_H,
                    size_t           cube_W,
                    int              n_threads,
                    const char      *dump_cube_dir,
                    const char      *cube_id,
                    float            trim_inset,
                    const uint8_t   *vol_in,
                    int              p_size_in,
                    const int64_t   *cube_origin_in,
                    PredRectoField  *recto,
                    ComponentMesh  **out_meshes,
                    size_t          *out_n_meshes,
                    MeshResplitCloud **out_clouds);

#endif
