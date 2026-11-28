#ifndef SEAM_HOLE_FILL_H
#define SEAM_HOLE_FILL_H

#include "../common/arena.h"
#include "../common/mesh_types.h"

typedef struct SeamHoleFillParams {
    double cube;
    double band;
    double max_extent;
    int    max_loop;
    double near_vox;
    double umb_y, umb_x, pitch;
    double wind_tol_turns;
} SeamHoleFillParams;

typedef struct SeamHoleFillStats {
    size_t loops;
    size_t seam_candidates;
    size_t filled;
    size_t tris_added;
    size_t skip_phase;
    size_t skip_extent;
    size_t skip_geom;
    size_t skip_manifold;
    size_t skip_bubble;
} SeamHoleFillStats;

void SeamHoleFill_default_params(SeamHoleFillParams *p);

int SeamHoleFill_process(Arena_T arena, ComponentMesh *cm,
                         const SeamHoleFillParams *p, SeamHoleFillStats *st);

#endif
