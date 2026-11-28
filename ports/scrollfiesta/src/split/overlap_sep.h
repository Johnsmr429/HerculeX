#ifndef OVERLAP_SEP_INCLUDED
#define OVERLAP_SEP_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"

typedef struct {
    int            enable_delamination_merge;
    const uint8_t *pred_vol;
    size_t         pred_D;
    size_t         pred_H;
    size_t         pred_W;
    float          pred_offset[3];
    float          pred_safety_distance;

    const char    *raw_dir;
    long           raw_chunk;
    float          world_offset[3];
    float          raw_snap_reach;
    float          raw_snap_alpha;
    float          raw_quilt_weight;
} OverlapSepOptions;

typedef struct {
    int32_t *face0;
    int32_t *face1;
    size_t count;
} OverlapSepPairSet;

int OverlapSep_detect_pairs(Arena_T arena, const ComponentMesh *mesh,
                            OverlapSepPairSet *out_pairs);

int OverlapSep_process(Arena_T              arena,
                       const ComponentMesh  *mesh,
                       int                   sheet_count,
                       int                   n_threads,
                       double                timeout_sec,
                       ComponentMesh       **out_meshes,
                       size_t               *out_count);

int OverlapSep_process_ex(Arena_T                 arena,
                          const ComponentMesh     *mesh,
                          int                      sheet_count,
                          int                      n_threads,
                          double                   timeout_sec,
                          const OverlapSepOptions *options,
                          ComponentMesh          **out_meshes,
                          size_t                  *out_count);

void OverlapSep_set_debug_dir(const char *dir);

#endif
