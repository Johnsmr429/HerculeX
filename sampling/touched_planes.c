#pragma once
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include "column_kernels.c"

#define ROW_BLOCK 64

typedef struct {
    int32_t first, last;
} Range;

typedef struct {
    Range *ranges;
    int32_t *ids;
    int32_t count;
} Touched;

static inline void include(Touched *touched, int64_t index, int32_t first, int32_t last) {
    Range *range = touched->ranges + index;
    int32_t old = __atomic_load_n(&range->first, __ATOMIC_RELAXED);
    while (first < old &&
        !__atomic_compare_exchange_n(
            &range->first, &old, first, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
    old = __atomic_load_n(&range->last, __ATOMIC_RELAXED);
    while (last > old) {
        if (__atomic_compare_exchange_n(
                &range->last, &old, last, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            if (old < 0 && touched->ids) {
                int32_t at = __atomic_fetch_add(&touched->count, 1, __ATOMIC_RELAXED);
                touched->ids[at] = (int32_t)index;
            }
            break;
        }
    }
}

static inline int brick_below(float value, int side, int reach) {
    return (int)floorf((floorf(value) - reach) / side);
}

static inline int brick_above(float value, int side, int reach) {
    return (int)floorf((floorf(value) + 1 + reach) / side);
}

static inline void mark_range(
    Touched *touched, const int32_t *g, int x0, int x1, int y0, int y1, int low, int high) {
    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    x1 = x1 >= g[2] ? g[2] - 1 : x1;
    y1 = y1 >= g[1] ? g[1] - 1 : y1;
    low = low < 0 ? 0 : low;
    high = high >= g[0] * g[3] ? g[0] * g[3] - 1 : high;
    for (int z = low; z <= high;) {
        int iz = z / g[3], start = z - iz * g[3];
        int stop = high - iz * g[3] + 1;
        if (stop > g[3])
            stop = g[3];
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                int64_t index = ((int64_t)iz * g[1] + y) * g[2] + x;
                include(touched, index, start, stop - 1);
            }
        z += stop - start;
    }
}

static inline void mark(Touched *touched, const int32_t *g, float x, float y, float z, int reach) {
    int plane = (int)floorf(z);
    mark_range(touched, g, brick_below(x, g[5], reach), brick_above(x, g[5], reach),
        brick_below(y, g[4], reach), brick_above(y, g[4], reach), plane - reach, plane + 1 + reach);
}

static inline int mark_line(Touched *touched, const int32_t *g, const float *p, const float *n,
    float first, float last, int reach) {
    float sx = p[0] + first * n[0], ex = p[0] + last * n[0];
    float sy = p[1] + first * n[1], ey = p[1] + last * n[1];
    int x0 = brick_below(sx, g[5], reach), x1 = brick_above(sx, g[5], reach);
    int y0 = brick_below(sy, g[4], reach), y1 = brick_above(sy, g[4], reach);
    if (x0 != brick_below(ex, g[5], reach) || x1 != brick_above(ex, g[5], reach) ||
        y0 != brick_below(ey, g[4], reach) || y1 != brick_above(ey, g[4], reach))
        return 0;
    float sz = p[2] + first * n[2], ez = p[2] + last * n[2];
    mark_range(touched, g, x0, x1, y0, y1, (int)floorf(fminf(sz, ez)) - reach,
        (int)floorf(fmaxf(sz, ez)) + 1 + reach);
    return 1;
}

static void touch_block(int64_t block, const int32_t *geometry, const float *grid, int64_t h,
    int64_t w, const int32_t *cell_y, const int32_t *cell_x, const float *frac_y,
    const float *frac_x, int64_t height, int64_t width, int layers, float first, float last_layer,
    int reach, Touched *touched) {
    int64_t last = (block + 1) * ROW_BLOCK < height ? (block + 1) * ROW_BLOCK : height;
    for (int64_t y = block * ROW_BLOCK; y < last; y++)
        for (int64_t x = 0; x < width; x++) {
            float p[3] = {-1, -1, -1}, n[3] = {0, 0, 0};
            column_at(grid, h, w, cell_y[y], cell_x[x], frac_y[y], frac_x[x], p, n);
            if (p[0] < 0 || mark_line(touched, geometry, p, n, first, last_layer, reach))
                continue;
            for (int l = 0; l < layers; l++) {
                float d = first + l;
                mark(touched, geometry, p[0] + d * n[0], p[1] + d * n[1], p[2] + d * n[2], reach);
            }
        }
}

static void touched_ranges(const int32_t *geometry, const float *grid, int64_t h, int64_t w,
    const int32_t *cell_y, const int32_t *cell_x, const float *frac_y, const float *frac_x,
    int64_t height, int64_t width, int layers, float first, int reach, Touched *touched) {
    if (layers <= 0)
        return;
    float last_layer = first + (layers - 1);
    int64_t blocks = (height + ROW_BLOCK - 1) / ROW_BLOCK;
#ifdef __APPLE__
    dispatch_apply(
        (size_t)blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
          touch_block((int64_t)block, geometry, grid, h, w, cell_y, cell_x, frac_y, frac_x, height,
              width, layers, first, last_layer, reach, touched);
        });
#else
#pragma omp parallel for num_threads(4) schedule(dynamic)
    for (int64_t block = 0; block < blocks; block++)
        touch_block(block, geometry, grid, h, w, cell_y, cell_x, frac_y, frac_x, height, width,
            layers, first, last_layer, reach, touched);
#endif
}

void hx_touched_ranges(const int32_t *geometry, const float *grid, int64_t h, int64_t w,
    const int32_t *cell_y, const int32_t *cell_x, const float *frac_y, const float *frac_x,
    int64_t height, int64_t width, int layers, float first, int reach, Range *ranges) {
    Touched touched = {.ranges = ranges};
    touched_ranges(geometry, grid, h, w, cell_y, cell_x, frac_y, frac_x, height, width, layers,
        first, reach, &touched);
}

static void fractions(int cells, int stride, int64_t length, int32_t *index, float *part) {
    for (int64_t i = 0; i < length; i++) {
        double point = fmin((double)i / stride, cells - 1.000001);
        index[i] = (int32_t)point;
        part[i] = point - index[i];
    }
}

static int mark_surface(Touched *touched, const int32_t *geom, const float *xyz,
    const float *normal, int h, int w, int stride, int layers) {
    int64_t height = (int64_t)(h - 1) * stride + 1, width = (int64_t)(w - 1) * stride + 1;
    int64_t cells = (int64_t)h * w;
    if (h < 2 || w < 2 || stride < 1 || layers < 1 ||
        (uint64_t)cells > SIZE_MAX / (7 * sizeof(float)) ||
        (uint64_t)height + width > SIZE_MAX / (sizeof(int32_t) + sizeof(float))) {
        errno = EINVAL;
        return -1;
    }
    for (int i = 0; i < 3; i++) {
        if (geom[i] < 1 || geom[i + 3] < 1 || geom[i] > INT32_MAX / geom[i + 3]) {
            errno = EOVERFLOW;
            return -1;
        }
    }
    float *grid = malloc((size_t)cells * 7 * sizeof(float));
    int32_t *indices = malloc((size_t)(height + width) * (sizeof(int32_t) + sizeof(float)));
    if (!indices || !grid) {
        free(indices);
        free(grid);
        errno = ENOMEM;
        return -1;
    }
    int32_t *cy = indices, *cx = cy + height;
    float *fy = (float *)(cx + width), *fx = fy + height;
    memcpy(grid + cells, xyz, (size_t)cells * 3 * sizeof(float));
    memcpy(grid + 4 * cells, normal, (size_t)cells * 3 * sizeof(float));
    for (int64_t i = 0; i < cells; i++) {
        grid[i] = xyz[i] > 0;
    }
    fractions(h, stride, height, cy, fy);
    fractions(w, stride, width, cx, fx);
    int first = -(layers / 2);
    touched_ranges(geom, grid, h, w, cy, cx, fy, fx, height, width, layers, first, 2, touched);
    free(indices);
    free(grid);
    return 0;
}
