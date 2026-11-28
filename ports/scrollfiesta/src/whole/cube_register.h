#ifndef CUBE_REGISTER_INCLUDED
#define CUBE_REGISTER_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "placed_cube.h"

typedef struct {
    PlacedReg tab;
    size_t  n_pairs;
    double  dphi_mad;
    double  du_mad;
    int     low_conf;
    int     deferred;
    int32_t n_groups_direct;
} CubeReg;

static inline double CubeReg_deltaU(double a, double b, double phi, int32_t k)
{
    const double pi = 3.14159265358979323846;
    return 2.0 * pi * a * (double)k + b * (double)k * phi
         + pi * b * (double)k * (double)k;
}

static inline void CubeReg_pick(const PlacedReg *r, int32_t gid,
                                int32_t *out_k, double *out_du)
{
    if (r != NULL && r->g_wk != NULL && gid >= 0 && gid < r->n_groups) {
        *out_k = r->g_wk[gid];
        *out_du = r->g_du[gid];
    } else {
        *out_k = r != NULL ? r->wk_cube : 0;
        *out_du = r != NULL ? r->du_cube : 0.0;
    }
}

static inline SkinVert CubeReg_apply_vert(SkinVert s, const PlacedReg *r,
                                          double a, double b)
{
    const double pi = 3.14159265358979323846;
    int32_t k = 0;
    double du = 0.0;
    CubeReg_pick(r, s.gid, &k, &du);
    SkinVert o = s;
    o.u   = (float)((double)s.u + CubeReg_deltaU(a, b, (double)s.phi, k) + du);
    o.phi = (float)((double)s.phi + 2.0 * pi * (double)k);
    return o;
}

int CubeReg_solve(Arena_T scratch, Arena_T tab_arena,
                  const SkinVert *skin_n, size_t n_n,
                  const SkinVert *pooled_p, size_t n_p,
                  double spiral_a, double spiral_b,
                  double pair_gate, size_t min_pairs, size_t min_group_pairs,
                  CubeReg *out);

int CubeReg_selftest(void);

#endif
