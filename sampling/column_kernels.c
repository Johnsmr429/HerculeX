#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static inline float voxel_at(
    const uint8_t *data, const int64_t *offsets, const int32_t *g, int z, int y, int x) {
    if (z < 0 || y < 0 || x < 0)
        return 0;
    int iz = z / g[3], iy = y / g[4], ix = x / g[5];
    if (iz >= g[0] || iy >= g[1] || ix >= g[2])
        return 0;
    int64_t o = offsets[((int64_t)iz * g[1] + iy) * g[2] + ix];
    if (o < 0)
        return 0;
    return data[o + ((int64_t)(z % g[3]) * g[4] + y % g[4]) * g[5] + x % g[5]];
}

static inline float trilinear_slow(const uint8_t *data, const int64_t *offsets, const int32_t *g,
    int iz, int iy, int ix, float fz, float fy, float fx) {
    float v = 0;
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            for (int e = 0; e < 2; e++) {
                float w = (a ? fz : 1 - fz) * (b ? fy : 1 - fy) * (e ? fx : 1 - fx);
                v += w * voxel_at(data, offsets, g, iz + a, iy + b, ix + e);
            }
    return v;
}

static void sample_column(const uint8_t *data, const int64_t *offsets, const int32_t *geometry,
    const float *p, const float *n, int layers, float first, uint8_t *out, int64_t pitch) {
    if (p[0] < 0) {
        for (int l = 0; l < layers; l++)
            out[l * pitch] = 0;
        return;
    }
    int g0 = geometry[0], g1 = geometry[1], g2 = geometry[2];
    int bz = geometry[3], by = geometry[4], bx = geometry[5];
    int64_t row = bx, plane = (int64_t)by * bx;
    float px = p[0], py = p[1], pz = p[2], nx = n[0], ny = n[1], nz = n[2];
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
            int sz = iz / bz, sy = iy / by, sx = ix / bx, rz = iz - sz * bz, ry = iy - sy * by,
                rx = ix - sx * bx;
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
        out[l * pitch] = (uint8_t)lrintf(v);
    }
}

void hx_sample_columns(const uint8_t *data, const int64_t *offsets, const int32_t *geometry,
    const float *position, const float *normal, int64_t columns, int layers, float first,
    uint8_t *out) {
    for (int64_t c = 0; c < columns; c++) {
        float p[3] = {position[c], position[columns + c], position[2 * columns + c]};
        float n[3] = {normal[c], normal[columns + c], normal[2 * columns + c]};
        sample_column(data, offsets, geometry, p, n, layers, first, out + c, columns);
    }
}

static inline float grid_value(const float *plane, int64_t w, int a, int b, float fy, float fx) {
#pragma clang fp contract(off)
    const float *top = plane + (int64_t)a * w + b, *low = top + w;
    float ax = 1 - fx, ay = 1 - fy;
    float upper = top[0] * ax + top[1] * fx;
    float lower = low[0] * ax + low[1] * fx;
    return upper * ay + lower * fy;
}

static inline void column_at(const float *grid, int64_t h, int64_t w, int a, int b, float fy,
    float fx, float *position, float *normal) {
#pragma clang fp contract(off)
    int64_t plane = h * w;
    if (!(grid_value(grid, w, a, b, fy, fx) > 0.999f)) {
        position[0] = -1;
        return;
    }
    for (int i = 0; i < 3; i++)
        position[i] = grid_value(grid + (1 + i) * plane, w, a, b, fy, fx);
    float n[3];
    for (int i = 0; i < 3; i++)
        n[i] = grid_value(grid + (4 + i) * plane, w, a, b, fy, fx);
    float length = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (length < 1e-9f)
        length = 1e-9f;
    for (int i = 0; i < 3; i++)
        normal[i] = n[i] / length;
}

int64_t hx_sample_grid(const uint8_t *data, const int64_t *offsets, const int32_t *geometry,
    const float *grid, int64_t h, int64_t w, const int32_t *cell_y, const int32_t *cell_x,
    const float *frac_y, const float *frac_x, int64_t height, int64_t width, int64_t y0, int64_t x0,
    int tile, int layers, float first, float *position, float *normal, uint8_t *out) {
    int64_t kept = 0, columns = (int64_t)tile * tile;
    for (int ty = 0; ty < tile; ty++)
        for (int tx = 0; tx < tile; tx++) {
            int64_t c = (int64_t)ty * tile + tx, y = y0 + ty, x = x0 + tx;
            float p[3] = {-1, -1, -1}, n[3] = {0, 0, 0};
            if (y < height && x < width)
                column_at(grid, h, w, cell_y[y], cell_x[x], frac_y[y], frac_x[x], p, n);
            kept += p[0] >= 0;
            for (int i = 0; i < 3; i++) {
                position[i * columns + c] = p[i];
                normal[i * columns + c] = n[i];
            }
        }
    if (kept)
        hx_sample_columns(data, offsets, geometry, position, normal, columns, layers, first, out);
    return kept;
}

#define ROWS_PER_BATCH 8

typedef struct {
    const uint8_t *data;
    const int64_t *offsets;
    const int32_t *geometry;
    const float *grid;
    const int32_t *rows, *columns;
    const float *fy, *fx;
    int64_t h, w, height, width, next;
    int layers;
    uint8_t *out;
    float *rays;
} Surface;

static void *sample_rows(void *arg) {
    Surface *s = arg;
    int64_t pixels = s->height * s->width;
    int first = -(s->layers / 2);
    for (;;) {
        int64_t row = __atomic_fetch_add(&s->next, ROWS_PER_BATCH, __ATOMIC_RELAXED);
        if (row >= s->height)
            break;
        int64_t end = row + ROWS_PER_BATCH;
        for (int64_t y = row; y < end && y < s->height; y++) {
            for (int64_t x = 0; x < s->width; x++) {
                float p[3] = {-1, 0, 0}, n[3] = {0};
                column_at(s->grid, s->h, s->w, s->rows[y], s->columns[x], s->fy[y], s->fx[x], p, n);
                int64_t i = y * s->width + x;
                if (s->rays) {
                    for (int k = 0; k < 3; k++) {
                        s->rays[6 * i + k] = p[k];
                        s->rays[6 * i + 3 + k] = n[k];
                    }
                } else {
                    sample_column(s->data, s->offsets, s->geometry, p, n, s->layers, first,
                        s->out + i, pixels);
                }
            }
        }
    }
    return NULL;
}

void hx_flatten(const uint8_t *data, const int64_t *offsets, const int32_t *geometry,
    const float *grid, int64_t h, int64_t w, const int32_t *rows, const int32_t *columns,
    const float *fy, const float *fx, int64_t height, int64_t width, int layers, uint8_t *out,
    float *rays) {
    Surface s = {.data = data,
        .offsets = offsets,
        .geometry = geometry,
        .grid = grid,
        .rows = rows,
        .columns = columns,
        .fy = fy,
        .fx = fx,
        .h = h,
        .w = w,
        .height = height,
        .width = width,
        .layers = layers,
        .out = out,
        .rays = rays};
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    int max_batches = (height + ROWS_PER_BATCH - 1) / ROWS_PER_BATCH;
    int limit = online < 1 ? 1 : (int)online;
    if (limit > max_batches)
        limit = max_batches;
    pthread_t *threads = limit > 1 ? malloc((size_t)(limit - 1) * sizeof(*threads)) : NULL;
    int workers = 0;
    while (threads && workers < limit - 1 && (workers + 1) * ROWS_PER_BATCH < height) {
        if (pthread_create(&threads[workers], NULL, sample_rows, &s))
            break;
        workers++;
    }
    sample_rows(&s);
    for (int i = 0; i < workers; i++)
        pthread_join(threads[i], NULL);
    free(threads);
}

void hx_stack_ranges(
    const int32_t *geometry, const float *grid, int h, int w, int layers, int32_t *range) {
    int cells = h * w, radius = layers / 2;
    for (int y = 0; y < h - 1; y++) {
        for (int x = 0; x < w - 1; x++) {
            int origin = y * w + x;
            int corner[] = {origin, origin + 1, origin + w, origin + w + 1};
            if (!(grid[corner[0]] || grid[corner[1]] || grid[corner[2]] || grid[corner[3]]))
                continue;
            int lower[3], upper[3];
            for (int axis = 0; axis < 3; axis++) {
                const float *position = grid + (3 - axis) * cells;
                float lo = INFINITY, hi = -INFINITY;
                float all_lo = INFINITY, all_hi = -INFINITY;
                for (int k = 0; k < 4; k++) {
                    float value = position[corner[k]];
                    all_lo = fminf(all_lo, value);
                    all_hi = fmaxf(all_hi, value);
                    if (grid[corner[k]]) {
                        lo = fminf(lo, value);
                        hi = fmaxf(hi, value);
                    }
                }
                float reach = radius + 3 + 0x1p-18f * fmaxf(fabsf(all_lo), fabsf(all_hi));
                lo -= reach + (lo - all_lo) / 128;
                hi += reach + (all_hi - hi) / 128;
                float end = (float)geometry[axis] * geometry[axis + 3] - 1;
                lower[axis] = floorf(fmaxf(0, fminf(end, lo)));
                upper[axis] = floorf(fmaxf(0, fminf(end, hi)));
            }
            int bz = geometry[3], by = geometry[4], bx = geometry[5];
            for (int iz = lower[0] / bz; iz <= upper[0] / bz; iz++) {
                int start = lower[0] - iz * bz, end = upper[0] - iz * bz;
                start = start < 0 ? 0 : start;
                end = end >= bz ? bz - 1 : end;
                for (int iy = lower[1] / by; iy <= upper[1] / by; iy++) {
                    for (int ix = lower[2] / bx; ix <= upper[2] / bx; ix++) {
                        int64_t index = ((int64_t)iz * geometry[1] + iy) * geometry[2] + ix;
                        int32_t *span = range + 2 * index;
                        if (start < span[0])
                            span[0] = start;
                        if (end > span[1])
                            span[1] = end;
                    }
                }
            }
        }
    }
}

int hx_sample_fma(float a, float b, float c) { return a * b + c != 0; }
