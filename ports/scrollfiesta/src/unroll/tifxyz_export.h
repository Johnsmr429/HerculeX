#ifndef TIFXYZ_EXPORT_INCLUDED
#define TIFXYZ_EXPORT_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "piece_set.h"

typedef struct {
    double du, dv;
    double stretch_ratio;
    double stretch_floor;
    double max_edge3d;
    double conflict_dist;
    int    flip_u;
    int    flip_v;
    int    write_winding;
    int    write_provenance;
    const uint8_t *face_keep;
    int    have_window;
    double win_u0, win_u1, win_v0, win_v1;
} TifxyzOpts;

void TifxyzOpts_default(TifxyzOpts *o);

typedef struct {
    size_t W, H;
    double u0, v0;
    size_t valid;
    size_t multi;
    size_t conflicts;
    size_t conflict_same_cube, conflict_cross_cube, conflict_unknown_cube;
    size_t conflict_d_le4, conflict_d_le8, conflict_d_le16;
    size_t conflict_d_le32, conflict_d_gt32;
    double conflict_d_sum, conflict_d_max;
    size_t conflict_turn_known;
    size_t conflict_turn_le025, conflict_turn_le05, conflict_turn_le1;
    size_t conflict_turn_gt1;
    size_t conflict_face_shared_edge, conflict_face_shared_vertex;
    size_t conflict_face_disjoint;
    size_t conflict_centroid_d_le8, conflict_centroid_d_gt8;
    size_t conflict_centroid_du_le2, conflict_centroid_du_le4;
    size_t conflict_centroid_du_le8, conflict_centroid_du_gt8;
    size_t conflict_centroid_dv_le2, conflict_centroid_dv_le4;
    size_t conflict_centroid_dv_le8, conflict_centroid_dv_gt8;
    size_t skip_uv, skip_3d, skip_own;
    double bbox_lo[3], bbox_hi[3];
    double area_vx2;
} TifxyzStats;

int TifxyzExport_run(Arena_T arena, const PieceSet *ps, const char *seg_dir,
                     const char *uuid, const TifxyzOpts *opts,
                     TifxyzStats *out);

int TifxyzExport_selftest(void);

#endif
