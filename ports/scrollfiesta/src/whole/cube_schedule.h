#ifndef CUBE_SCHEDULE_INCLUDED
#define CUBE_SCHEDULE_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "axis_warp.h"

typedef struct {
    char     id[48];
    int64_t  oz, oy, ox;
    double   r, theta;
    double   w_phase;
    int32_t  nbr[6];
} CubeNode;

int CubeSched_build(Arena_T arena, const char *dump_dir, const char *leaf_stage,
                    int64_t chunk, const float axis_point[3], double pitch,
                    const char *seed_id,
                    CubeNode **out_nodes, size_t *out_n,
                    int32_t **out_order, int32_t **out_comp);

int CubeSched_build_axis_warp(
    Arena_T arena, const char *dump_dir, const char *leaf_stage,
    int64_t chunk, const float axis_point[3], const AxisWarp *axis_warp,
    double pitch, const char *seed_id,
    CubeNode **out_nodes, size_t *out_n,
    int32_t **out_order, int32_t **out_comp);

int CubeSched_link_and_order(Arena_T arena, CubeNode *nodes, size_t n,
                             int64_t chunk, const float axis_point[3],
                             double pitch, int32_t seed_idx,
                             int32_t **out_order, int32_t **out_comp);
int CubeSched_link_and_order_axis_warp(
    Arena_T arena, CubeNode *nodes, size_t n, int64_t chunk,
    const float axis_point[3], const AxisWarp *axis_warp,
    double pitch, int32_t seed_idx,
    int32_t **out_order, int32_t **out_comp);

int CubeSched_selftest(void);

#endif
