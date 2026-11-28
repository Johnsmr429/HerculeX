#ifndef EXPORT_ATLAS_INCLUDED
#define EXPORT_ATLAS_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "piece_set.h"
#include "scaffold.h"

typedef struct {
    int sign;
    int curved_axis;
    double offset;
    double coherence;
    size_t groups, ambiguous_groups;
    size_t cycle_conflicts, isolated_vertices;
} AtlasPolarStats;

typedef struct {
    int    piece_wraps;
    double slab_v;
    double du, dv;
    int    write_winding;
    double max_edge3d, stretch_ratio, stretch_floor, conflict_dist;
    int    relax;
    int    relax_sweeps;
    int    phase_u;
    int    polar_u;
    int    input_polar_u;
    AtlasPolarStats input_polar_stats;
    int    verbose;
} AtlasOpts;

void AtlasOpts_default(AtlasOpts *o);

int Atlas_lift_polar_u(Arena_T arena, const PieceSet *ps,
                       const ScaffoldCalib *c, float **out_uv,
                       AtlasPolarStats *stats);

typedef struct {
    size_t n_pieces;
    size_t n_written;
    size_t n_empty;
    size_t n_over_cap;
    size_t n_quarantine_faces;
    size_t n_phase_quarantine_faces;
    size_t n_u_outlier_faces;
    size_t total_valid_px, total_multi_px, total_conflict_px;
    size_t total_conflict_same_cube_px, total_conflict_cross_cube_px;
    size_t total_conflict_unknown_cube_px;
    size_t total_conflict_d_le4, total_conflict_d_le8;
    size_t total_conflict_d_le16, total_conflict_d_le32;
    size_t total_conflict_d_gt32;
    double total_conflict_d_sum, max_conflict_d;
    size_t total_conflict_turn_known, total_conflict_turn_le025;
    size_t total_conflict_turn_le05, total_conflict_turn_le1;
    size_t total_conflict_turn_gt1;
    size_t total_conflict_face_shared_edge;
    size_t total_conflict_face_shared_vertex, total_conflict_face_disjoint;
    size_t total_conflict_centroid_d_le8, total_conflict_centroid_d_gt8;
    size_t total_conflict_centroid_du_le2, total_conflict_centroid_du_le4;
    size_t total_conflict_centroid_du_le8, total_conflict_centroid_du_gt8;
    size_t total_conflict_centroid_dv_le2, total_conflict_centroid_dv_le4;
    size_t total_conflict_centroid_dv_le8, total_conflict_centroid_dv_gt8;
    double conflict_frac;
    int    polar_sign, polar_curved_axis;
    double polar_offset, polar_coherence;
    size_t polar_groups, polar_ambiguous_groups;
    size_t polar_cycle_conflicts, polar_isolated_vertices;
    double seconds;
} AtlasStats;

int ExportAtlas_run(Arena_T arena, const PieceSet *ps, const ScaffoldCalib *c,
                    const char *seg_root, const char *prefix,
                    const AtlasOpts *opts, AtlasStats *out);

int ExportAtlas_selftest(void);

#endif
