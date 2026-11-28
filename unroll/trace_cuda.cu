#include "trace_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    float x, y, z, nx, ny, nz, score, depth;
    uint32_t key;
} Pt;

typedef struct {
    int tZ, tY, tX, hZ, hY, hX;
    int gZ, gY, gX;
    int pz0, py0, px0;
    int cz0, cz1, cy0, cy1, cx0, cx1;
    int stride, lz, ly, lx, nz, ny, nx;
    int GY, GX;
    long long y0, x0;
    double bz, by, bx;
} Tile;

enum { FULL = 7, HALF = 11, CELL = 5, THREADS = 256 };

static struct {
    TraceCudaCrest s;
    int tZ, tY, tX, ncell;
    int slab_pz0, slab_tZ;
    uint8_t *slab;
    float *full[FULL], *half[HALF], *k;
    double *centre;
    int *cell;
    Pt *pts;
    TraceCudaSurfel *stmp, *sout, *host_out;
    int *host_cell;
    size_t pts_cap, stmp_cap, sout_cap, host_cap;
    TraceCudaStats stats;
} C;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

__device__ __forceinline__ int clampd(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void fused_kernel(const uint8_t *slab, size_t plane, int X, Tile t, float *out) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tpl = (unsigned)t.tY * t.tX;
    if (i >= (unsigned)t.tZ * tpl)
        return;
    int x = (int)(i % t.tX), y = (int)((i / t.tX) % t.tY), z = (int)(i / tpl);
    float s = slab[(size_t)z * plane + (size_t)(t.py0 + y) * X + t.px0 + x] * (1.0f / 255.0f);
    out[i] = s < 0.02f ? 0.0f : s;
}

__global__ void blur_x(const float *in, float *out, int Z, int Y, int X, const float *k, int r) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned)Z * Y * X)
        return;
    int x = (int)(i % X);
    const float *src = in + (i - x);
    float s = 0.0f;
    for (int d = -r; d <= r; d++)
        s += k[d + r] * src[clampd(x + d, 0, X - 1)];
    out[i] = s;
}

__global__ void blur_y(const float *in, float *out, int Z, int Y, int X, const float *k, int r) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned pl = (unsigned)Y * X;
    if (i >= (unsigned)Z * pl)
        return;
    int x = (int)(i % X), y = (int)((i / X) % Y);
    const float *plane = in + (i / pl) * pl;
    float acc = 0.0f;
    for (int d = -r; d <= r; d++)
        acc += k[d + r] * plane[(unsigned)clampd(y + d, 0, Y - 1) * X + x];
    out[i] = acc;
}

__global__ void blur_z(const float *in, float *out, int Z, int Y, int X, const float *k, int r) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned pl = (unsigned)Y * X;
    if (i >= (unsigned)Z * pl)
        return;
    int z = (int)(i / pl);
    unsigned in_plane = i - (unsigned)z * pl;
    float acc = 0.0f;
    for (int d = -r; d <= r; d++)
        acc += k[d + r] * in[(unsigned)clampd(z + d, 0, Z - 1) * pl + in_plane];
    out[i] = acc;
}

__global__ void half_accum(const float *s_sm, Tile t, float *hs, float *j0, float *j1, float *j2,
    float *j3, float *j4, float *j5) {
    unsigned hi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned hplane = (unsigned)t.hY * t.hX;
    if (hi >= (unsigned)t.hZ * hplane)
        return;
    int Z = t.tZ, Y = t.tY, X = t.tX;
    int hz = (int)(hi / hplane), hy = (int)((hi / t.hX) % t.hY), hx = (int)(hi % t.hX);
    unsigned plane = (unsigned)Y * X;
    int z_lo = 2 * hz, z_hi = z_lo + 2 < Z ? z_lo + 2 : Z;
    int y_lo = 2 * hy, y_hi = y_lo + 2 < Y ? y_lo + 2 : Y;
    int x_lo = 2 * hx, x_hi = x_lo + 2 < X ? x_lo + 2 : X;
    float a00 = 0, a11 = 0, a22 = 0, a01 = 0, a02 = 0, a12 = 0, sv = 0;
    int cnt = 0;
    for (int z = z_lo; z < z_hi; z++) {
        int zm = z > 0 ? z - 1 : 0, zp = z + 1 < Z ? z + 1 : Z - 1;
        float zsc = (z > 0 && z + 1 < Z) ? 0.5f : 1.0f;
        for (int y = y_lo; y < y_hi; y++) {
            int ym = y > 0 ? y - 1 : 0, yp = y + 1 < Y ? y + 1 : Y - 1;
            float ysc = (y > 0 && y + 1 < Y) ? 0.5f : 1.0f;
            for (int x = x_lo; x < x_hi; x++) {
                int xm = x > 0 ? x - 1 : 0, xp = x + 1 < X ? x + 1 : X - 1;
                float xsc = (x > 0 && x + 1 < X) ? 0.5f : 1.0f;
                unsigned i = (unsigned)z * plane + (unsigned)y * X + x;
                sv += s_sm[i];
                float gz = (s_sm[(unsigned)zp * plane + (unsigned)y * X + x] -
                               s_sm[(unsigned)zm * plane + (unsigned)y * X + x]) *
                    zsc;
                float gy = (s_sm[(unsigned)z * plane + (unsigned)yp * X + x] -
                               s_sm[(unsigned)z * plane + (unsigned)ym * X + x]) *
                    ysc;
                float gx = (s_sm[(unsigned)z * plane + (unsigned)y * X + xp] -
                               s_sm[(unsigned)z * plane + (unsigned)y * X + xm]) *
                    xsc;
                a00 += gx * gx;
                a11 += gy * gy;
                a22 += gz * gz;
                a01 += gx * gy;
                a02 += gx * gz;
                a12 += gy * gz;
                cnt++;
            }
        }
    }
    float inv = cnt > 0 ? 1.0f / (float)cnt : 0.0f;
    hs[hi] = sv * inv;
    j0[hi] = a00 * inv;
    j1[hi] = a11 * inv;
    j2[hi] = a22 * inv;
    j3[hi] = a01 * inv;
    j4[hi] = a02 * inv;
    j5[hi] = a12 * inv;
}

__device__ void jacobi3_top_d(
    double a00, double a11, double a22, double a01, double a02, double a12, float out[3]) {
    double m[3][3] = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sw = 0; sw < 16; sw++) {
        double off = fabs(m[0][1]) + fabs(m[0][2]) + fabs(m[1][2]);
        if (off < 1e-18)
            break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++) {
                if (fabs(m[p][q]) < 1e-20)
                    continue;
                double th = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {
                    double x = m[k][p], y = m[k][q];
                    m[k][p] = c * x - s * y;
                    m[k][q] = s * x + c * y;
                }
                for (int k = 0; k < 3; k++) {
                    double x = m[p][k], y = m[q][k];
                    m[p][k] = c * x - s * y;
                    m[q][k] = s * x + c * y;
                }
                for (int k = 0; k < 3; k++) {
                    double x = v[k][p], y = v[k][q];
                    v[k][p] = c * x - s * y;
                    v[k][q] = s * x + c * y;
                }
            }
    }
    int top = 0;
    if (m[1][1] > m[top][top])
        top = 1;
    if (m[2][2] > m[top][top])
        top = 2;
    double nx = v[0][top], ny = v[1][top], nz = v[2][top];
    double l = sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 1e-12) {
        out[0] = (float)(nx / l);
        out[1] = (float)(ny / l);
        out[2] = (float)(nz / l);
    } else {
        out[0] = 1.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
    }
}

__global__ void half_eig(Tile t, const float *j0, const float *j1, const float *j2, const float *j3,
    const float *j4, const float *j5, const double *cxh, const double *cyh, float *hnx, float *hny,
    float *hnz) {
    unsigned hi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned hplane = (unsigned)t.hY * t.hX;
    if (hi >= (unsigned)t.hZ * hplane)
        return;
    int hz = (int)(hi / hplane), hy = (int)((hi / t.hX) % t.hY), hx = (int)(hi % t.hX);
    double cx = cxh[hz], cy = cyh[hz];
    double ry = (double)(t.y0 + 2 * hy) + 0.5 - cy;
    float n[3];
    jacobi3_top_d(j0[hi], j1[hi], j2[hi], j3[hi], j4[hi], j5[hi], n);
    double rx = (double)(t.x0 + 2 * hx) + 0.5 - cx;
    if (n[0] * rx + n[1] * ry < 0.0) {
        n[0] = -n[0];
        n[1] = -n[1];
        n[2] = -n[2];
    }
    hnx[hi] = n[0];
    hny[hi] = n[1];
    hnz[hi] = n[2];
}

__global__ void upsample(Tile t, const float *s_sm, const float *hs, const float *hnx,
    const float *hny, const float *hnz, const double *cxf, const double *cyf, float *nx, float *ny,
    float *nz, float *s_crest) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned plane = (unsigned)t.tY * t.tX;
    if (i >= (unsigned)t.tZ * plane)
        return;
    int hZ = t.hZ, hY = t.hY, hX = t.hX;
    int z = (int)(i / plane), y = (int)((i / t.tX) % t.tY), x = (int)(i % t.tX);
    unsigned hplane = (unsigned)hY * hX;
    double cx = cxf[z], cy = cyf[z];
    float gz = 0.5f * ((float)z - 0.5f);
    if (gz < 0.0f)
        gz = 0.0f;
    else if (gz > hZ - 1.0001f)
        gz = hZ - 1.0001f;
    int iz0 = (int)gz, iz1 = iz0 + (iz0 + 1 < hZ);
    float fz = gz - iz0;
    double ry = (double)(t.y0 + y) - cy;
    float gy = 0.5f * ((float)y - 0.5f);
    if (gy < 0.0f)
        gy = 0.0f;
    else if (gy > hY - 1.0001f)
        gy = hY - 1.0001f;
    int iy0 = (int)gy, iy1 = iy0 + (iy0 + 1 < hY);
    float fy = gy - iy0;
    float gx = 0.5f * ((float)x - 0.5f);
    if (gx < 0.0f)
        gx = 0.0f;
    else if (gx > hX - 1.0001f)
        gx = hX - 1.0001f;
    int ix0 = (int)gx, ix1 = ix0 + (ix0 + 1 < hX);
    float fx = gx - ix0;
    unsigned ids[8] = {
        (unsigned)iz0 * hplane + (unsigned)iy0 * hX + ix0,
        (unsigned)iz0 * hplane + (unsigned)iy0 * hX + ix1,
        (unsigned)iz0 * hplane + (unsigned)iy1 * hX + ix0,
        (unsigned)iz0 * hplane + (unsigned)iy1 * hX + ix1,
        (unsigned)iz1 * hplane + (unsigned)iy0 * hX + ix0,
        (unsigned)iz1 * hplane + (unsigned)iy0 * hX + ix1,
        (unsigned)iz1 * hplane + (unsigned)iy1 * hX + ix0,
        (unsigned)iz1 * hplane + (unsigned)iy1 * hX + ix1,
    };
    float wts[8] = {
        (1.0f - fz) * (1.0f - fy) * (1.0f - fx),
        (1.0f - fz) * (1.0f - fy) * fx,
        (1.0f - fz) * fy * (1.0f - fx),
        (1.0f - fz) * fy * fx,
        fz * (1.0f - fy) * (1.0f - fx),
        fz * (1.0f - fy) * fx,
        fz * fy * (1.0f - fx),
        fz * fy * fx,
    };
    float rnx = hnx[ids[0]], rny = hny[ids[0]], rnz = hnz[ids[0]];
    float sx = 0, sy = 0, sz = 0, sw = 0;
    for (int k = 0; k < 8; k++) {
        float knx = hnx[ids[k]], kny = hny[ids[k]], knz = hnz[ids[k]];
        float sgn = (knx * rnx + kny * rny + knz * rnz) < 0.0f ? -1.0f : 1.0f;
        sx += wts[k] * sgn * knx;
        sy += wts[k] * sgn * kny;
        sz += wts[k] * sgn * knz;
        sw += wts[k] * hs[ids[k]];
    }
    float l = sqrtf(sx * sx + sy * sy + sz * sz) + 1e-6f;
    sx /= l;
    sy /= l;
    sz /= l;
    double rx = (double)(t.x0 + x) - cx;
    if (sx * rx + sy * ry < 0.0) {
        sx = -sx;
        sy = -sy;
        sz = -sz;
    }
    nx[i] = sx;
    ny[i] = sy;
    nz[i] = sz;
    s_crest[i] = 0.72f * s_sm[i] + 0.28f * sw;
}

__device__ float trilinear_d(const float *vol, int Z, int Y, int X, float z, float y, float x) {
    if (z < 0.0f)
        z = 0.0f;
    else if (z > Z - 1.0001f)
        z = Z - 1.0001f;
    if (y < 0.0f)
        y = 0.0f;
    else if (y > Y - 1.0001f)
        y = Y - 1.0001f;
    if (x < 0.0f)
        x = 0.0f;
    else if (x > X - 1.0001f)
        x = X - 1.0001f;
    int iz = (int)z, iy = (int)y, ix = (int)x;
    float fz = z - iz, fy = y - iy, fx = x - ix;
    int iz1 = iz + (iz + 1 < Z), iy1 = iy + (iy + 1 < Y), ix1 = ix + (ix + 1 < X);
    unsigned pl = (unsigned)Y * X;
    unsigned i000 = (unsigned)iz * pl + (unsigned)iy * X + ix;
    unsigned i001 = (unsigned)iz * pl + (unsigned)iy * X + ix1;
    unsigned i010 = (unsigned)iz * pl + (unsigned)iy1 * X + ix;
    unsigned i011 = (unsigned)iz * pl + (unsigned)iy1 * X + ix1;
    unsigned i100 = (unsigned)iz1 * pl + (unsigned)iy * X + ix;
    unsigned i101 = (unsigned)iz1 * pl + (unsigned)iy * X + ix1;
    unsigned i110 = (unsigned)iz1 * pl + (unsigned)iy1 * X + ix;
    unsigned i111 = (unsigned)iz1 * pl + (unsigned)iy1 * X + ix1;
    float c00 = vol[i000] * (1.0f - fx) + vol[i001] * fx;
    float c01 = vol[i010] * (1.0f - fx) + vol[i011] * fx;
    float c10 = vol[i100] * (1.0f - fx) + vol[i101] * fx;
    float c11 = vol[i110] * (1.0f - fx) + vol[i111] * fx;
    float c0 = c00 * (1.0f - fy) + c01 * fy;
    float c1 = c10 * (1.0f - fy) + c11 * fy;
    return c0 * (1.0f - fz) + c1 * fz;
}

__device__ __forceinline__ unsigned core_voxel(const Tile &t, unsigned i, int *z, int *y, int *x) {
    unsigned cy_n = (unsigned)(t.cy1 - t.cy0), cx_n = (unsigned)(t.cx1 - t.cx0);
    *z = t.cz0 + (int)(i / (cy_n * cx_n));
    *y = t.cy0 + (int)((i / cx_n) % cy_n);
    *x = t.cx0 + (int)(i % cx_n);
    return ((unsigned)*z * t.tY + (unsigned)*y) * t.tX + (unsigned)*x;
}

__device__ __forceinline__ Pt crest_point(const Tile &t, unsigned i, float delta, const float *s,
    const float *nxv, const float *nyv, const float *nzv) {
    int z, y, x;
    unsigned ti = core_voxel(t, i, &z, &y, &x);
    float nxi = nxv[ti], nyi = nyv[ti], nzi = nzv[ti];
    Pt p;
    p.x = (float)(t.px0 + x) + delta * nxi;
    p.y = (float)(t.py0 + y) + delta * nyi;
    p.z = (float)(t.pz0 + z) + delta * nzi;
    p.nx = nxi;
    p.ny = nyi;
    p.nz = nzi;
    p.score = s[ti];
    p.depth = 0.0f;
    p.key = i;
    return p;
}

__device__ __forceinline__ int cell_of(const Tile &t, const Pt &p) {
    int cz = clampd((int)(p.z / t.stride), t.lz, t.lz + t.nz - 1);
    int cy = clampd((int)(p.y / t.stride), t.ly, t.ly + t.ny - 1);
    int cx = clampd((int)(p.x / t.stride), t.lx, t.lx + t.nx - 1);
    return ((cz - t.lz) * t.ny + (cy - t.ly)) * t.nx + (cx - t.lx);
}

__global__ void crest_mark(Tile t, const float *s, const float *nxv, const float *nyv,
    const float *nzv, float s_min, float curv_min, float *mark, int *cnt) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned)(t.cz1 - t.cz0) * (unsigned)(t.cy1 - t.cy0) * (unsigned)(t.cx1 - t.cx0))
        return;
    mark[i] = 2.0f;
    int z, y, x;
    unsigned ti = core_voxel(t, i, &z, &y, &x);
    if (z <= 0 || z >= t.tZ - 1)
        return;
    int gz = t.pz0 + z;
    if (gz <= 0 || gz >= t.gZ - 1)
        return;
    if (y < 1 || y >= t.tY - 1 || x < 1 || x >= t.tX - 1)
        return;
    int gy = t.py0 + y;
    if (gy <= 0 || gy >= t.gY - 1)
        return;
    int gx = t.px0 + x;
    if (gx <= 0 || gx >= t.gX - 1)
        return;
    float v0 = s[ti];
    if (v0 < s_min)
        return;
    float nxi = nxv[ti], nyi = nyv[ti], nzi = nzv[ti];
    float vp1 = trilinear_d(s, t.tZ, t.tY, t.tX, z + nzi, y + nyi, x + nxi);
    float vm1 = trilinear_d(s, t.tZ, t.tY, t.tX, z - nzi, y - nyi, x - nxi);
    if (v0 < vp1 || v0 < vm1)
        return;
    float curv = 2.0f * v0 - vp1 - vm1;
    if (curv < curv_min)
        return;
    float denom = vm1 - 2.0f * v0 + vp1;
    float delta = fabsf(denom) > 1e-6f ? 0.5f * (vm1 - vp1) / denom : 0.0f;
    if (delta < -0.5f)
        delta = -0.5f;
    else if (delta > 0.5f)
        delta = 0.5f;
    mark[i] = delta;
    atomicAdd(cnt + cell_of(t, crest_point(t, i, delta, s, nxv, nyv, nzv)), 1);
}

__global__ void crest_fill(Tile t, const float *s, const float *nxv, const float *nyv,
    const float *nzv, const float *mark, const int *first, int *fill, Pt *pts) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned)(t.cz1 - t.cz0) * (unsigned)(t.cy1 - t.cy0) * (unsigned)(t.cx1 - t.cx0))
        return;
    float delta = mark[i];
    if (delta > 1.0f)
        return;
    Pt p = crest_point(t, i, delta, s, nxv, nyv, nzv);
    int lc = cell_of(t, p);
    pts[first[lc] + atomicAdd(fill + lc, 1)] = p;
}

__global__ void bin_cells(Tile t, int min_pts, float gap, Pt *pts, const int *first, const int *cnt,
    TraceCudaSurfel *out, int *nsurf) {
    int lc = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lc >= t.nz * t.ny * t.nx)
        return;
    int n = cnt[lc], made = 0;
    nsurf[lc] = 0;
    if (n < min_pts)
        return;
    Pt *p = pts + first[lc];
    for (int i = 1; i < n; i++) {
        Pt v = p[i];
        int j = i;
        while (j > 0 && p[j - 1].key > v.key) {
            p[j] = p[j - 1];
            j--;
        }
        p[j] = v;
    }
    double mx = 0, my = 0, mz = 0;
    float rnx = p[0].nx, rny = p[0].ny, rnz = p[0].nz;
    for (int i = 0; i < n; i++) {
        double sgn = (p[i].nx * rnx + p[i].ny * rny + p[i].nz * rnz) < 0 ? -1.0 : 1.0;
        mx += sgn * p[i].nx;
        my += sgn * p[i].ny;
        mz += sgn * p[i].nz;
    }
    double ml = sqrt(mx * mx + my * my + mz * mz) + 1e-9;
    mx /= ml;
    my /= ml;
    mz /= ml;
    for (int i = 0; i < n; i++)
        p[i].depth = (float)(p[i].x * mx + p[i].y * my + p[i].z * mz);
    for (int i = 1; i < n; i++) {
        Pt v = p[i];
        int j = i;
        while (j > 0 && p[j - 1].depth > v.depth) {
            p[j] = p[j - 1];
            j--;
        }
        p[j] = v;
    }
    int cz = t.lz + lc / (t.ny * t.nx), cy = t.ly + (lc / t.nx) % t.ny, cx = t.lx + lc % t.nx;
    int32_t cell = (int32_t)(((int64_t)cz * t.GY + cy) * t.GX + cx);
    TraceCudaSurfel *dst = out + first[lc] / min_pts;
    for (int c0 = 0; c0 < n;) {
        int c1 = c0 + 1;
        while (c1 < n && p[c1].depth - p[c1 - 1].depth <= gap)
            c1++;
        if (c1 - c0 >= min_pts) {
            double sx = 0, sy = 0, sz = 0, snx = 0, sny = 0, snz = 0, ssc = 0, scw = 0, ssw = 0,
                   ssp = 0;
            double wtot = 0;
            for (int k = c0; k < c1; k++) {
                double w = p[k].score;
                double sgn = (p[k].nx * mx + p[k].ny * my + p[k].nz * mz) < 0 ? -1.0 : 1.0;
                sx += w * p[k].x;
                sy += w * p[k].y;
                sz += w * p[k].z;
                snx += w * sgn * p[k].nx;
                sny += w * sgn * p[k].ny;
                snz += w * sgn * p[k].nz;
                ssc += p[k].score;
                scw += w * 1.0f;
                ssw += w * 0.0f;
                ssp += w * 12.0f;
                wtot += w;
            }
            double nl = sqrt(snx * snx + sny * sny + snz * snz) + 1e-9;
            TraceCudaSurfel r;
            r.x = t.bx + sx / wtot;
            r.y = t.by + sy / wtot;
            r.z = t.bz + sz / wtot;
            r.nx = snx / nl;
            r.ny = sny / nl;
            r.nz = snz / nl;
            r.score = (float)(ssc / (double)(c1 - c0));
            r.cos_w = (float)(scw / wtot);
            r.sin_w = (float)(ssw / wtot);
            r.spacing = (float)(ssp / wtot);
            r.cell = cell;
            r.count = c1 - c0;
            dst[made++] = r;
        }
        c0 = c1;
    }
    nsurf[lc] = made;
}

__global__ void compact_cells(int ncell, int min_pts, const TraceCudaSurfel *in, const int *first,
    const int *nsurf, const int *sfirst, TraceCudaSurfel *out) {
    int lc = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (lc >= ncell)
        return;
    const TraceCudaSurfel *src = in + first[lc] / min_pts;
    TraceCudaSurfel *dst = out + sfirst[lc];
    for (int k = 0; k < nsurf[lc]; k++)
        dst[k] = src[k];
}

static int bad(cudaError_t e, const char *what) {
    if (e == cudaSuccess)
        return 0;
    fprintf(stderr, "trace: CUDA %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

static unsigned grid(size_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

static void blur(float *in, float *out, float *tmp, int Z, int Y, int X, const float *k, int r) {
    size_t n = (size_t)Z * Y * X;
    blur_x<<<grid(n), THREADS>>>(in, tmp, Z, Y, X, k, r);
    blur_y<<<grid(n), THREADS>>>(tmp, out, Z, Y, X, k, r);
    blur_z<<<grid(n), THREADS>>>(out, tmp, Z, Y, X, k, r);
    cudaMemcpyAsync(out, tmp, n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
}

template <typename T> static int room(T **buffer, size_t *cap, size_t need, const char *what) {
    if (need <= *cap)
        return 0;
    cudaFree(*buffer);
    *buffer = NULL;
    *cap = 0;
    size_t want = need + need / 4;
    if (bad(cudaMalloc(buffer, want * sizeof(T)), what))
        return 1;
    *cap = want;
    return 0;
}

static void crest_free(void) {
    cudaFree(C.slab);
    for (int i = 0; i < FULL; i++)
        cudaFree(C.full[i]);
    for (int i = 0; i < HALF; i++)
        cudaFree(C.half[i]);
    cudaFree(C.k);
    cudaFree(C.centre);
    cudaFree(C.cell);
    cudaFree(C.pts);
    cudaFree(C.stmp);
    cudaFree(C.sout);
    free(C.host_out);
    free(C.host_cell);
    TraceCudaStats stats = C.stats;
    memset(&C, 0, sizeof(C));
    C.stats = stats;
}

extern "C" int trace_cuda_open(void) {
    int devices = 0;
    size_t available = 0, total = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1 ||
        cudaSetDevice(0) != cudaSuccess || cudaMemGetInfo(&available, &total) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return 0;
}

extern "C" int trace_cuda_crest_begin(const TraceCudaCrest *setup) {
    crest_free();
    memset(&C.stats, 0, sizeof(C.stats));
    C.s = *setup;
    const TraceCudaCrest *s = &C.s;
    int span_z = s->tile_z + 2 * s->tile_pad, span_yx = s->tile_yx + 2 * s->tile_pad;
    C.tZ = span_z < s->Z ? span_z : s->Z;
    C.tY = span_yx < s->Y ? span_yx : s->Y;
    C.tX = span_yx < s->X ? span_yx : s->X;
    size_t vox = (size_t)C.tZ * C.tY * C.tX;
    size_t hvox = (size_t)((C.tZ + 1) / 2) * ((C.tY + 1) / 2) * ((C.tX + 1) / 2);
    int cells_z = (s->tile_z + s->stride - 1) / s->stride;
    int cells_yx = (s->tile_yx + s->stride - 1) / s->stride;
    size_t ncell = (size_t)cells_z * cells_yx * cells_yx;
    if (vox >= (size_t)1 << 31 || ncell >= (size_t)1 << 28 || s->min_cell_pts < 1 || s->r_sm > 32 ||
        s->r_hs > 32 || s->r_hj > 32)
        return -1;
    C.ncell = (int)ncell;
    C.slab_tZ = 0;
    int failed = bad(cudaMalloc(&C.slab, (size_t)C.tZ * s->Y * s->X), "slab");
    for (int i = 0; i < FULL && !failed; i++)
        failed = bad(cudaMalloc(&C.full[i], vox * sizeof(float)), "tile");
    for (int i = 0; i < HALF && !failed; i++)
        failed = bad(cudaMalloc(&C.half[i], hvox * sizeof(float)), "tile");
    failed = failed || bad(cudaMalloc(&C.k, 3 * 65 * sizeof(float)), "kernels") ||
        bad(cudaMalloc(&C.centre, (size_t)4 * C.tZ * sizeof(double)), "centres") ||
        bad(cudaMalloc(&C.cell, CELL * ncell * sizeof(int)), "cells") ||
        bad(cudaMemcpy(C.k, s->k_sm, 65 * sizeof(float), cudaMemcpyHostToDevice), "kernels") ||
        bad(cudaMemcpy(C.k + 65, s->k_hs, 65 * sizeof(float), cudaMemcpyHostToDevice), "kernels") ||
        bad(cudaMemcpy(C.k + 130, s->k_hj, 65 * sizeof(float), cudaMemcpyHostToDevice), "kernels");
    C.host_cell = (int *)malloc(2 * ncell * sizeof(int));
    if (failed || !C.host_cell) {
        crest_free();
        return -1;
    }
    return 0;
}

extern "C" int trace_cuda_crest_slab(
    const uint8_t *planes, int pz0, int tZ, const double *centres) {
    if (!C.slab || tZ > C.tZ)
        return -1;
    double t0 = now();
    int hZ = (tZ + 1) / 2;
    size_t bytes = (size_t)tZ * C.s.Y * C.s.X;
    if (bad(cudaMemcpy(C.slab, planes, bytes, cudaMemcpyHostToDevice), "slab upload") ||
        bad(cudaMemcpy(C.centre, centres, (size_t)(2 * tZ + 2 * hZ) * sizeof(double),
                cudaMemcpyHostToDevice),
            "centre upload"))
        return -1;
    C.slab_pz0 = pz0;
    C.slab_tZ = tZ;
    C.stats.upload += now() - t0;
    C.stats.bytes += (int64_t)bytes;
    C.stats.slabs++;
    return 0;
}

extern "C" int trace_cuda_crest_tile(
    const TraceCudaTile *tile, const TraceCudaSurfel **surfels, int64_t *count, int64_t *crest) {
    const TraceCudaCrest *s = &C.s;
    Tile t;
    t.tZ = tile->pz1 - tile->pz0;
    t.tY = tile->py1 - tile->py0;
    t.tX = tile->px1 - tile->px0;
    if (!C.slab || tile->pz0 != C.slab_pz0 || t.tZ != C.slab_tZ || t.tY > C.tY || t.tX > C.tX)
        return -1;
    double t0 = now();
    t.hZ = (t.tZ + 1) / 2;
    t.hY = (t.tY + 1) / 2;
    t.hX = (t.tX + 1) / 2;
    t.gZ = s->Z;
    t.gY = s->Y;
    t.gX = s->X;
    t.pz0 = tile->pz0;
    t.py0 = tile->py0;
    t.px0 = tile->px0;
    t.cz0 = tile->za - tile->pz0;
    t.cz1 = tile->zb - tile->pz0;
    t.cy0 = tile->ya - tile->py0;
    t.cy1 = tile->yb - tile->py0;
    t.cx0 = tile->xa - tile->px0;
    t.cx1 = tile->xb - tile->px0;
    t.stride = s->stride;
    t.lz = tile->za / s->stride;
    t.ly = tile->ya / s->stride;
    t.lx = tile->xa / s->stride;
    t.nz = (tile->zb - 1) / s->stride - t.lz + 1;
    t.ny = (tile->yb - 1) / s->stride - t.ly + 1;
    t.nx = (tile->xb - 1) / s->stride - t.lx + 1;
    t.GY = s->GY;
    t.GX = s->GX;
    t.y0 = (long long)s->box[2] + tile->py0;
    t.x0 = (long long)s->box[4] + tile->px0;
    t.bz = (double)s->box[0];
    t.by = (double)s->box[2];
    t.bx = (double)s->box[4];
    int ncell = t.nz * t.ny * t.nx;
    if (ncell > C.ncell)
        return -1;
    size_t vox = (size_t)t.tZ * t.tY * t.tX, hvox = (size_t)t.hZ * t.hY * t.hX;
    size_t core = (size_t)(t.cz1 - t.cz0) * (t.cy1 - t.cy0) * (t.cx1 - t.cx0);
    float *raw = C.full[0], *sm = C.full[1], *tmp = C.full[2], *nx = C.full[3], *ny = C.full[4],
          *nz = C.full[5], *ridge = C.full[6];
    float *hs = C.half[0], **hj = C.half + 1, *htmp = C.half[7], *hnx = C.half[8], *hny = C.half[9],
          *hnz = C.half[10];
    const double *cxf = C.centre, *cyf = cxf + t.tZ, *cxh = cyf + t.tZ, *cyh = cxh + t.hZ;
    int *cnt = C.cell, *first = cnt + C.ncell, *fill = first + C.ncell, *nsurf = fill + C.ncell,
        *sfirst = nsurf + C.ncell;
    fused_kernel<<<grid(vox), THREADS>>>(C.slab, (size_t)s->Y * s->X, s->X, t, raw);
    blur(raw, sm, tmp, t.tZ, t.tY, t.tX, C.k, s->r_sm);
    half_accum<<<grid(hvox), THREADS>>>(sm, t, hs, hj[0], hj[1], hj[2], hj[3], hj[4], hj[5]);
    blur(hs, hs, htmp, t.hZ, t.hY, t.hX, C.k + 65, s->r_hs);
    for (int k = 0; k < 6; k++)
        blur(hj[k], hj[k], htmp, t.hZ, t.hY, t.hX, C.k + 130, s->r_hj);
    half_eig<<<grid(hvox), THREADS>>>(
        t, hj[0], hj[1], hj[2], hj[3], hj[4], hj[5], cxh, cyh, hnx, hny, hnz);
    upsample<<<grid(vox), THREADS>>>(t, sm, hs, hnx, hny, hnz, cxf, cyf, nx, ny, nz, ridge);
    cudaMemsetAsync(cnt, 0, (size_t)ncell * sizeof(int), 0);
    crest_mark<<<grid(core), THREADS>>>(t, ridge, nx, ny, nz, s->s_min, s->curv_min, raw, cnt);
    int *host_cnt = C.host_cell, *host_first = C.host_cell + C.ncell;
    if (bad(cudaMemcpy(host_cnt, cnt, (size_t)ncell * sizeof(int), cudaMemcpyDeviceToHost),
            "crest kernels"))
        return -1;
    double t1 = now();
    C.stats.kernels += t1 - t0;
    int64_t points = 0;
    for (int c = 0; c < ncell; c++) {
        host_first[c] = (int)points;
        points += host_cnt[c];
    }
    *surfels = NULL;
    *count = 0;
    *crest = points;
    C.stats.tiles++;
    if (!points)
        return 0;
    if (points >= (int64_t)1 << 31 || room(&C.pts, &C.pts_cap, (size_t)points, "points") ||
        room(&C.stmp, &C.stmp_cap, (size_t)points / s->min_cell_pts + 1, "surfels"))
        return -1;
    if (bad(cudaMemcpy(first, host_first, (size_t)ncell * sizeof(int), cudaMemcpyHostToDevice),
            "cell upload"))
        return -1;
    cudaMemsetAsync(fill, 0, (size_t)ncell * sizeof(int), 0);
    crest_fill<<<grid(core), THREADS>>>(t, ridge, nx, ny, nz, raw, first, fill, C.pts);
    bin_cells<<<grid(ncell), THREADS>>>(
        t, s->min_cell_pts, s->min_layer_gap, C.pts, first, cnt, C.stmp, nsurf);
    if (bad(cudaMemcpy(host_cnt, nsurf, (size_t)ncell * sizeof(int), cudaMemcpyDeviceToHost),
            "binning"))
        return -1;
    int64_t made = 0;
    for (int c = 0; c < ncell; c++) {
        host_first[c] = (int)made;
        made += host_cnt[c];
    }
    if (made) {
        if (room(&C.sout, &C.sout_cap, (size_t)made, "surfels"))
            return -1;
        if ((size_t)made > C.host_cap) {
            size_t want = (size_t)made + (size_t)made / 4;
            TraceCudaSurfel *grown =
                (TraceCudaSurfel *)realloc(C.host_out, want * sizeof(TraceCudaSurfel));
            if (!grown)
                return -1;
            C.host_out = grown;
            C.host_cap = want;
        }
        if (bad(cudaMemcpy(sfirst, host_first, (size_t)ncell * sizeof(int), cudaMemcpyHostToDevice),
                "cell upload"))
            return -1;
        compact_cells<<<grid(ncell), THREADS>>>(
            ncell, s->min_cell_pts, C.stmp, first, nsurf, sfirst, C.sout);
        if (bad(cudaMemcpy(C.host_out, C.sout, (size_t)made * sizeof(TraceCudaSurfel),
                    cudaMemcpyDeviceToHost),
                "surfel download"))
            return -1;
    }
    if (bad(cudaGetLastError(), "crest tile"))
        return -1;
    *surfels = C.host_out;
    *count = made;
    C.stats.binning += now() - t1;
    return 0;
}

extern "C" void trace_cuda_crest_end(TraceCudaStats *stats) {
    cudaDeviceSynchronize();
    crest_free();
    if (stats)
        *stats = C.stats;
}
