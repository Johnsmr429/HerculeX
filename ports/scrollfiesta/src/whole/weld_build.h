#ifndef WELD_BUILD_INCLUDED
#define WELD_BUILD_INCLUDED

#include <stddef.h>
#include "../common/arena.h"
#include "cube_schedule.h"
#include "placed_cube.h"
#include "weld_track.h"

typedef struct {
    double pair_gate;
    double radius_gate;
    double frac_gate;
    double margin_frac;
    int    verbose;
} WeldBuildOpts;

void WeldBuildOpts_default(WeldBuildOpts *o);

int WeldBuild_pass1(Arena_T arena, const CubeNode *cnodes, size_t n_cubes,
                    SkinVert *const *skins, const size_t *nskin,
                    const float axis_point[3],
                    double spiral_a, double spiral_b, double pitch,
                    const WeldBuildOpts *opts, WeldTrack *out);

int WeldBuild_selftest(void);

#endif
