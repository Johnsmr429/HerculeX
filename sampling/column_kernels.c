
#include <math.h>
#include <stdint.h>

// One voxel of a brick store, zero where no brick was fetched; g holds grid z, y, x then brick z, y, x.
static inline float voxel_at(const uint8_t *data, const int64_t *offsets, const int32_t *g, int z, int y, int x) {
    if (z < 0 || y < 0 || x < 0) return 0;
    int iz = z / g[3], iy = y / g[4], ix = x / g[5];
    if (iz >= g[0] || iy >= g[1] || ix >= g[2]) return 0;
    int64_t o = offsets[((int64_t)iz * g[1] + iy) * g[2] + ix];
    if (o < 0) return 0;
    return data[o + ((int64_t)(z % g[3]) * g[4] + y % g[4]) * g[5] + x % g[5]];
}

static inline float trilinear_slow(const uint8_t *data, const int64_t *offsets, const int32_t *g, int iz, int iy,
                                   int ix, float fz, float fy, float fx) {
    float v = 0;
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            for (int e = 0; e < 2; e++) {
                float w = (a ? fz : 1 - fz) * (b ? fy : 1 - fy) * (e ? fx : 1 - fx);
                v += w * voxel_at(data, offsets, g, iz + a, iy + b, ix + e);
            }
    return v;
}

void hx_sample_columns(const uint8_t *data, const int64_t *offsets, const int32_t *geometry, const float *position,
                       const float *normal, int64_t columns, int layers, float first, uint8_t *out) {
    const int g0 = geometry[0], g1 = geometry[1], g2 = geometry[2], bz = geometry[3], by = geometry[4],
              bx = geometry[5];
    const int64_t row = bx, plane = (int64_t)by * bx;
    for (int64_t c = 0; c < columns; c++) {
        float px = position[c], py = position[columns + c], pz = position[2 * columns + c];
        if (px < 0) {
            for (int l = 0; l < layers; l++) out[l * columns + c] = 0;
            continue;
        }
        float nx = normal[c], ny = normal[columns + c], nz = normal[2 * columns + c];
        int kz = -1, ky = -1, kx = -1;
        const uint8_t *brick = 0;
        for (int l = 0; l < layers; l++) {
            float d = first + l;
            float x = px + d * nx, y = py + d * ny, z = pz + d * nz;
            int ix = (int)floorf(x), iy = (int)floorf(y), iz = (int)floorf(z);
            float fx = x - ix, fy = y - iy, fz = z - iz;
            float v;
            if (iz < 0 || iy < 0 || ix < 0) {
                v = trilinear_slow(data, offsets, geometry, iz, iy, ix, fz, fy, fx);
            } else {
                int sz = iz / bz, sy = iy / by, sx = ix / bx, rz = iz - sz * bz, ry = iy - sy * by, rx = ix - sx * bx;
                if (rz + 1 >= bz || ry + 1 >= by || rx + 1 >= bx || sz >= g0 || sy >= g1 || sx >= g2) {
                    v = trilinear_slow(data, offsets, geometry, iz, iy, ix, fz, fy, fx);
                } else {
                    if (sz != kz || sy != ky || sx != kx) {
                        int64_t o = offsets[((int64_t)sz * g1 + sy) * g2 + sx];
                        brick = o < 0 ? 0 : data + o;
                        kz = sz, ky = sy, kx = sx;
                    }
                    if (!brick) {
                        v = 0;
                    } else {
                        const uint8_t *p = brick + rz * plane + ry * row + rx;
                        float az = 1 - fz, ay = 1 - fy, ax = 1 - fx;
                        v = 0;
                        v += az * ay * ax * p[0];
                        v += az * ay * fx * p[1];
                        v += az * fy * ax * p[row];
                        v += az * fy * fx * p[row + 1];
                        v += fz * ay * ax * p[plane];
                        v += fz * ay * fx * p[plane + 1];
                        v += fz * fy * ax * p[plane + row];
                        v += fz * fy * fx * p[plane + row + 1];
                    }
                }
            }
            out[l * columns + c] = (uint8_t)lrintf(v);
        }
    }
}

// A grid plane at one column, bilinear between grid points, each product rounded as numpy rounds it.
static inline float grid_value(const float *plane, int64_t w, int a, int b, float fy, float fx) {
#pragma clang fp contract(off)
    const float *top = plane + (int64_t)a * w + b, *low = top + w;
    float ax = 1 - fx, ay = 1 - fy;
    float upper = top[0] * ax + top[1] * fx;
    float lower = low[0] * ax + low[1] * fx;
    return upper * ay + lower * fy;
}

static inline void column_at(const float *grid, int64_t h, int64_t w, int a, int b, float fy, float fx, float *position,
                             float *normal) {
#pragma clang fp contract(off)
    int64_t plane = h * w;
    if (!(grid_value(grid, w, a, b, fy, fx) > 0.999f)) {
        position[0] = -1;
        return;
    }
    for (int i = 0; i < 3; i++) position[i] = grid_value(grid + (1 + i) * plane, w, a, b, fy, fx);
    float n[3];
    for (int i = 0; i < 3; i++) n[i] = grid_value(grid + (4 + i) * plane, w, a, b, fy, fx);
    float length = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (length < 1e-9f) length = 1e-9f;
    for (int i = 0; i < 3; i++) normal[i] = n[i] / length;
}

int64_t hx_sample_grid(const uint8_t *data, const int64_t *offsets, const int32_t *geometry, const float *grid,
                       int64_t h, int64_t w, const int32_t *cell_y, const int32_t *cell_x, const float *frac_y,
                       const float *frac_x, int64_t height, int64_t width, int64_t y0, int64_t x0, int tile, int layers,
                       float first, float *position, float *normal, uint8_t *out) {
    int64_t kept = 0, columns = (int64_t)tile * tile;
    for (int ty = 0; ty < tile; ty++)
        for (int tx = 0; tx < tile; tx++) {
            int64_t c = (int64_t)ty * tile + tx, y = y0 + ty, x = x0 + tx;
            float p[3] = {-1, -1, -1}, n[3] = {0, 0, 0};
            if (y < height && x < width) column_at(grid, h, w, cell_y[y], cell_x[x], frac_y[y], frac_x[x], p, n);
            kept += p[0] >= 0;
            for (int i = 0; i < 3; i++) {
                position[i * columns + c] = p[i];
                normal[i * columns + c] = n[i];
            }
        }
    if (kept) hx_sample_columns(data, offsets, geometry, position, normal, columns, layers, first, out);
    return kept;
}
