#ifndef PLACED_CUBE_INCLUDED
#define PLACED_CUBE_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "../flatten/ribbon.h"
#include "axis_warp.h"

typedef struct {
    float   p[3];
    float   u, v, phi;
    int32_t gid;
} SkinVert;

typedef struct {
    const int32_t *g_wk;
    const double  *g_du;
    int32_t        n_groups;
    int32_t        wk_cube;
    double         du_cube;
    const int32_t *g_warp_nk;
    const double  *g_warp_phi0;
    const double  *g_warp_dphi;
    const double  *g_warp_delta;
    int32_t        g_warp_stride;
} PlacedReg;

typedef struct {
    size_t nv, nf;
    size_t n_skin;
    size_t n_badface;
    size_t n_badlink_face;
    size_t n_unmapped_face;
    size_t n_geom_restored_faces;
    size_t n_geom_restored_charts;
    size_t n_group_vertices;
    long   loops_cut;
    double u_min, u_max, v_min, v_max, phi_min, phi_max;
    double spiral_r2;
    double spiral_a, spiral_b;
    int    w_groups;
    size_t uv_filled;
    size_t uv_fallback;
    double seconds;
    int    status;
} PlacedStats;

int PlacedCube_unwrap(Arena_T arena, const char *mesh_path, const char *out_dir,
                      const char *id, const RibbonOpts *ropts,
                      const AxisWarp *axis_warp, int sever,
                      const int64_t origin[3], int64_t chunk, double skin_dist,
                      double cut_ratio, double cut_floor, double cut_len,
                      int write_obj, PlacedStats *stats);

int PlacedCube_finalize(Arena_T arena, const char *out_dir, const char *id,
                        const PlacedReg *reg,
                        double spiral_a, double spiral_b, int write_obj,
                        double out_ranges[6]);

int PlacedCube_load_skin(Arena_T arena, const char *out_dir, const char *id,
                         int raw, SkinVert **out, size_t *out_n);

int PlacedCube_selftest(void);

#endif
