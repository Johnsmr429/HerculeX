#ifndef PIPELINE_CUBE_INCLUDED
#define PIPELINE_CUBE_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "../common/mesh_types.h"

typedef struct {
    const char *tiff_path;
    const char *pred_dir;
    const char *cube_id;
    int         halo_voxels;
    int         cube_D, cube_H, cube_W;
    int         n_threads;
    float       cvt_target_ratio;
    float       trim_inset;
    const char *dump_dir;
    int         skip_simplify;
    int         surface_f0_inner;
    int         full_dumps;
    int         cull_oracle_tangles;

    const uint8_t *vol_in;
    int            p_size_in;
    int64_t        cube_origin_zyx[3];
} PipelineInput;

typedef struct {
    ComponentMesh *meshes;
    size_t         n_meshes;
    ComponentMesh *trimmed;
    size_t         n_trimmed;
    double         t_extract;
    double         t_cvt;
    double         t_trim;
    double         t_dump;
    size_t         n_bad_sheets;
} PipelineOutput;

int pipeline_process_cube(Arena_T arena,
                          const PipelineInput *in,
                          PipelineOutput *out);

int pipeline_selftest(void);

#endif
