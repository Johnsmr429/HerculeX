#ifndef UV_WELD_INCLUDED
#define UV_WELD_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "piece_set.h"

typedef struct UVWeldOpts {
    double r3d;
    double skin;
    double du_gate;
    double dv_gate;
    long   chunk;
    int    verbose;
} UVWeldOpts;

void UVWeldOpts_default(UVWeldOpts *o);

typedef struct UVWeldStats {
    size_t n_boundary;
    size_t n_pairs;
    size_t rej_cube, rej_gid, rej_uv;
    size_t n_clusters;
    size_t n_merged_verts;
    size_t n_degen_faces;
    double du_absmed;
} UVWeldStats;

int UVWeld_run(Arena_T arena, PieceSet *ps, const UVWeldOpts *opts,
               UVWeldStats *out);

int UVWeld_selftest(void);

#endif
