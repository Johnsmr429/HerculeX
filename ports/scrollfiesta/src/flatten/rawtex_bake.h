#ifndef RAWTEX_BAKE_H
#define RAWTEX_BAKE_H

#include <stddef.h>
#include <stdint.h>

#include "../common/raw_sample.h"

typedef struct RawtexUv { const float *f32; const double *f64; } RawtexUv;
static inline double Rawtex_uv(RawtexUv uv, size_t value)
{ return uv.f64 ? uv.f64[value] : (double)uv.f32[value]; }
static inline int Rawtex_has_uv(RawtexUv uv)
{ return (uv.f32 != NULL) != (uv.f64 != NULL); }

typedef struct DiagOpts {
    const char *prefix;
    int         dark_thresh;
    const char *smear_obj;
} DiagOpts;

typedef struct RawtexPlan {
    double umin, umax, vmin, vmax;
    size_t W, H;
    double need_du, need_dv;
    size_t max_px;
    int    ok;
} RawtexPlan;

typedef struct RawtexWindow {
    double umin, umax, vmin, vmax;
} RawtexWindow;

int Rawtex_plan(const float *uv, size_t nv, double du, double dv,
                size_t max_px, RawtexPlan *out);

int Rawtex_plan_window(const float *uv, size_t nv, double du, double dv,
                       size_t max_px, const RawtexWindow *window,
                       RawtexPlan *out);

int Rawtex_write_tif(const char *path, CubeTable *ct,
                     const float *verts, const float *uv, size_t nv,
                     const int32_t *faces, size_t nf,
                     const float *normals,
                     const uint8_t *face_skip,
                     double range, int nsteps,
                     double du, double dv, double lo, double hi,
                     double stretch_ratio, double stretch_floor,
                     double max_edge3d, size_t max_px,
                     const DiagOpts *diag,
                     size_t *out_W, size_t *out_H,
                     double *out_fill, size_t *out_multi,
                     size_t *out_skip_uv, size_t *out_skip_3d);

int Rawtex_write_tif_window(const char *path, CubeTable *ct,
                            const float *verts, const float *uv, size_t nv,
                            const int32_t *faces, size_t nf,
                            const float *normals,
                            const uint8_t *face_skip,
                            double range, int nsteps,
                            double du, double dv, double lo, double hi,
                            double stretch_ratio, double stretch_floor,
                            double max_edge3d, size_t max_px,
                            const RawtexWindow *window,
                            const DiagOpts *diag,
                            size_t *out_W, size_t *out_H,
                            double *out_fill, size_t *out_multi,
                            size_t *out_skip_uv, size_t *out_skip_3d);

void Rawtex_stretch_window(const double *val, const uint8_t *has, size_t nv,
                           double pct_lo, double pct_hi,
                           double *out_lo, double *out_hi);

int Rawtex_plan_field(RawtexUv uv, size_t nv, double du, double dv,
                       size_t max_px, const RawtexWindow *window,
                       RawtexPlan *out);

int Rawtex_write_tif_field(const char *path, CubeTable *ct,
                            const float *verts, RawtexUv uv, size_t nv,
                            const int32_t *faces, size_t nf,
                            const float *normals,
                            const uint8_t *face_skip,
                            double range, int nsteps,
                            double du, double dv, double lo, double hi,
                            double stretch_ratio, double stretch_floor,
                            double max_edge3d, size_t max_px,
                            const RawtexWindow *window,
                            const DiagOpts *diag,
                            size_t *out_W, size_t *out_H,
                            double *out_fill, size_t *out_multi,
                            size_t *out_skip_uv, size_t *out_skip_3d);

int Rawtex_selftest(void);

int Rawtex_write_ct_tiled(const char *path, CubeTable *ct,
    const float *verts, RawtexUv uv, size_t nv, const int32_t *faces, size_t nf,
    const float *normals, double lo, double hi, const RawtexWindow *window,
    size_t max_px, size_t band_rows, size_t *out_W, size_t *out_H,
    double *out_fill, size_t *out_multi);
int Rawtex_tiled_selftest(void);

#endif
