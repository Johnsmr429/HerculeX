#ifndef STITCH_CUDA_H
#define STITCH_CUDA_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define STITCH_SHARED __host__ __device__ static inline
#else
#define STITCH_SHARED static inline
#endif

#define STITCH_PI 3.14159265358979323846
#define STITCH_BLOCK 32

typedef struct {
    float x, y, z;
    float nx, ny, nz;
} StitchSpot;

typedef struct {
    StitchSpot spot;
    float score, cos_w, sin_w, spacing, theta;
    int32_t cell, count, patch, winding;
} StitchSurfel;

typedef struct {
    int32_t at, hit, tie;
    float dist;
} StitchSeamHit;

typedef struct {
    int32_t at, hit, tie;
} StitchRayHit;

typedef struct StitchCuda StitchCuda;

STITCH_SHARED double stitch_phi(float theta, int32_t winding, int curl) {
    return (double)curl * (double)theta + 2.0 * STITCH_PI * (double)winding;
}

STITCH_SHARED int stitch_seam_near(
    const StitchSpot *sa, const StitchSpot *sb, float normal_reach, float *dist) {
    float ca = sa->nx * sb->nx + sa->ny * sb->ny + sa->nz * sb->nz;
    if (ca < 0.88f)
        return 0;
    float vx = sb->x - sa->x, vy = sb->y - sa->y, vz = sb->z - sa->z;
    float d = sqrtf(vx * vx + vy * vy + vz * vz);
    float dn = fabsf(vx * sa->nx + vy * sa->ny + vz * sa->nz);
    if (dn > normal_reach)
        return 0;
    *dist = d;
    return 1;
}

STITCH_SHARED int stitch_ray_near(
    const StitchSpot *sa, const StitchSpot *sb, double qx, double qy, double qz, double *reach) {
    double ex = sb->x - qx, ey = sb->y - qy, ez = sb->z - qz;
    double along = ex * sa->nx + ey * sa->ny + ez * sa->nz;
    double off2 = ex * ex + ey * ey + ez * ez - along * along;
    if (fabs(along) > 1.5 || off2 > 9.0)
        return 0;
    if (sa->nx * sb->nx + sa->ny * sb->ny + sa->nz * sb->nz < 0.7)
        return 0;
    *reach = along;
    return 1;
}

STITCH_SHARED int stitch_edge_near(const StitchSpot *su, const StitchSpot *sv, int stride) {
    float max_dt = 2.05f * stride;
    float max_dn = fminf(2.8f, 0.40f * stride);
    float ca = su->nx * sv->nx + su->ny * sv->ny + su->nz * sv->nz;
    if (ca < 0.82f)
        return 0;
    float vx = sv->x - su->x, vy = sv->y - su->y, vz = sv->z - su->z;
    float nax = su->nx + sv->nx, nay = su->ny + sv->ny, naz = su->nz + sv->nz;
    float nl = sqrtf(nax * nax + nay * nay + naz * naz) + 1e-6f;
    nax /= nl;
    nay /= nl;
    naz /= nl;
    float dn = fabsf(vx * nax + vy * nay + vz * naz);
    float d2 = vx * vx + vy * vy + vz * vz;
    float dt = sqrtf(fmaxf(0.0f, d2 - dn * dn));
    if (dt > max_dt || dt < 0.12f * stride)
        return 0;
    if (dn > max_dn || dn > 0.30f * dt + 0.8f)
        return 0;
    return 1;
}

STITCH_SHARED int stitch_bridge_near(
    const StitchSpot *su, const StitchSpot *sv, int stride, int reach) {
    float ca = su->nx * sv->nx + su->ny * sv->ny + su->nz * sv->nz;
    if (ca < 0.86f)
        return 0;
    float vx = sv->x - su->x, vy = sv->y - su->y, vz = sv->z - su->z;
    float nax = su->nx + sv->nx, nay = su->ny + sv->ny, naz = su->nz + sv->nz;
    float nl = sqrtf(nax * nax + nay * nay + naz * naz) + 1e-6f;
    nax /= nl;
    nay /= nl;
    naz /= nl;
    float dn = fabsf(vx * nax + vy * nay + vz * naz);
    float d3 = sqrtf(vx * vx + vy * vy + vz * vz);
    if (d3 > ((float)reach + 0.5f) * stride || dn > 2.8f)
        return 0;
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif

StitchCuda *stitch_cuda_open(void);
void stitch_cuda_close(StitchCuda *g);
const char *stitch_cuda_name(const StitchCuda *g);
void stitch_cuda_timed(StitchCuda *g);
void stitch_cuda_report(const StitchCuda *g);
int stitch_cuda_unit(
    StitchCuda *g, int32_t unit, const StitchSurfel *surfels, const int32_t *piece, int64_t count);
void stitch_cuda_units_drop(StitchCuda *g);
int64_t stitch_cuda_seam(StitchCuda *g, int32_t unit_a, int32_t unit_b, double lo, double hi,
    float reach, float normal_reach, StitchSeamHit **hits);
int64_t stitch_cuda_rays(
    StitchCuda *g, int32_t unit, int32_t every, double reach, StitchRayHit **hits);
int stitch_cuda_emit_open(StitchCuda *g, int64_t nodes, int32_t sheets, const int64_t *sheet_first,
    const StitchSurfel *const *sheet_nodes, const int32_t *order, int32_t stride, int32_t curl,
    int32_t bridge, double phi_gate, int32_t *label, int32_t **pairs, int64_t *pair_count);
int stitch_cuda_emit_bridges(StitchCuda *g, const int32_t *from, const int32_t *to, int64_t count);
int stitch_cuda_emit_flatten(StitchCuda *g, const int32_t *slot_node, int64_t slots,
    const int64_t *piece_first, int32_t pieces, int32_t steps, double tolerance, double held,
    float *u, float *v);
int stitch_cuda_emit_rows(
    StitchCuda *g, const int32_t *nodes, int64_t count, int64_t *first, int32_t **to);
void stitch_cuda_emit_close(StitchCuda *g);

#ifdef __cplusplus
}
#endif

#endif
