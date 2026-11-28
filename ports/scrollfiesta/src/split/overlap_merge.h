#ifndef OVERLAP_MERGE_INCLUDED
#define OVERLAP_MERGE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "../common/mesh_types.h"

typedef struct {
    int     min_rings;
    int     max_rings;
    int     local_search_rings;
    size_t  max_overlap_pairs;
    size_t  max_seed_faces;
    size_t  max_patch_faces;
    double  max_patch_fraction;
    double  max_seed_diameter;
    double  max_patch_diameter;
    double  max_plane_rms;
    double  contact_distance;
    double  merge_distance;
} OverlapMergeConfig;

typedef struct {
    const char *reason;
    int         rings;
    size_t      overlap_pairs;
    size_t      seed_faces;
    size_t      patch_faces_removed;
    size_t      patch_faces_added;
    size_t      patch_face_start;
    size_t      patch_vertices_coalesced;
    size_t      interface_edges;
    size_t      boundary_loops_before;
    size_t      boundary_loops_after;
    double      seed_diameter;
    double      patch_diameter;
    double      plane_rms;
    float       plane_normal[3];
    float       plane_center[3];
    const int32_t *exterior_source_faces;
    const uint8_t *vertex_role;
    const float   *chart_uv;
} OverlapMergeStats;

enum {
    OVERLAP_MERGE_VERTEX_EXTERIOR = 0,
    OVERLAP_MERGE_VERTEX_ANCHOR   = 1,
    OVERLAP_MERGE_VERTEX_REPAIR   = 2
};

void OverlapMerge_default_config(OverlapMergeConfig *config);

int OverlapMerge_try(Arena_T arena,
                     const ComponentMesh *mesh,
                     const int32_t *face_labels,
                     int32_t num_labels,
                     const int32_t *adj_fa,
                     const int32_t *adj_fb,
                     size_t n_adj,
                     const int32_t *ovl_fa,
                     const int32_t *ovl_fb,
                     size_t n_ovl,
                     const OverlapMergeConfig *config,
                     ComponentMesh *out,
                     OverlapMergeStats *stats);

#endif
