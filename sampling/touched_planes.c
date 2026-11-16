
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include "column_kernels.c"

#define ROW_BLOCK 64

typedef struct { int32_t first, last; } Range;

static inline void include(Range *range, int32_t first, int32_t last) {
    int32_t old = __atomic_load_n(&range->first, __ATOMIC_RELAXED);
    while (first < old &&
           !__atomic_compare_exchange_n(&range->first, &old, first, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    old = __atomic_load_n(&range->last, __ATOMIC_RELAXED);
    while (last > old &&
           !__atomic_compare_exchange_n(&range->last, &old, last, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}


static inline int brick_below(float value, int side, int reach) { return (int)floorf((floorf(value) - reach) / side); }

static inline int brick_above(float value, int side, int reach) { return (int)floorf((floorf(value) + 1 + reach) / side); }

static inline void mark_range(Range *touched, const int32_t *g,
                              int x0, int x1, int y0, int y1, int low, int high) {
    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    x1 = x1 >= g[2] ? g[2] - 1 : x1;
    y1 = y1 >= g[1] ? g[1] - 1 : y1;
    low = low < 0 ? 0 : low;
    high = high >= g[0] * g[3] ? g[0] * g[3] - 1 : high;
    for (int z = low; z <= high;) {
        int iz = z / g[3], start = z - iz * g[3];
        int stop = high - iz * g[3] + 1;
        if (stop > g[3]) stop = g[3];
        for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
            Range *range = touched + ((int64_t)iz * g[1] + y) * g[2] + x;
            include(range, start, stop - 1);
        }
        z += stop - start;
    }
}

static inline void mark(Range *touched, const int32_t *g, float x, float y, float z, int reach) {
    int plane = (int)floorf(z);
    mark_range(touched, g, brick_below(x, g[5], reach), brick_above(x, g[5], reach),
               brick_below(y, g[4], reach), brick_above(y, g[4], reach), plane - reach, plane + 1 + reach);
}

static inline int mark_line(Range *touched, const int32_t *g, const float *p, const float *n,
                            float first, float last, int reach) {
    float sx = p[0] + first * n[0], ex = p[0] + last * n[0];
    float sy = p[1] + first * n[1], ey = p[1] + last * n[1];
    int x0 = brick_below(sx, g[5], reach), x1 = brick_above(sx, g[5], reach);
    int y0 = brick_below(sy, g[4], reach), y1 = brick_above(sy, g[4], reach);
    if (x0 != brick_below(ex, g[5], reach) || x1 != brick_above(ex, g[5], reach) ||
        y0 != brick_below(ey, g[4], reach) || y1 != brick_above(ey, g[4], reach)) return 0;
    // The column is monotone in each axis. Adjacent z-intervals overlap or touch because its normal has unit length.
    float sz = p[2] + first * n[2], ez = p[2] + last * n[2];
    mark_range(touched, g, x0, x1, y0, y1, (int)floorf(fminf(sz, ez)) - reach, (int)floorf(fmaxf(sz, ez)) + 1 + reach);
    return 1;
}

// One block of ROW_BLOCK rows of columns: the bricks each touches, merged into the shared ranges by atomic min/max.
static void touch_block(int64_t block, const int32_t *geometry, const float *grid, int64_t h, int64_t w, const int32_t *cell_y,
                        const int32_t *cell_x, const float *frac_y, const float *frac_x, int64_t height, int64_t width, int layers,
                        float first, float last_layer, int reach, Range *touched) {
    int64_t last = (block + 1) * ROW_BLOCK < height ? (block + 1) * ROW_BLOCK : height;
    for (int64_t y = block * ROW_BLOCK; y < last; y++) for (int64_t x = 0; x < width; x++) {
        float p[3] = {-1, -1, -1}, n[3] = {0, 0, 0};
        column_at(grid, h, w, cell_y[y], cell_x[x], frac_y[y], frac_x[x], p, n);
        if (p[0] < 0 || mark_line(touched, geometry, p, n, first, last_layer, reach)) continue;
        for (int l = 0; l < layers; l++) {
            float d = first + l;
            mark(touched, geometry, p[0] + d * n[0], p[1] + d * n[1], p[2] + d * n[2], reach);
        }
    }
}

void hx_touched_ranges(const int32_t *geometry, const float *grid, int64_t h, int64_t w, const int32_t *cell_y, const int32_t *cell_x, const float *frac_y,
                       const float *frac_x, int64_t height, int64_t width, int layers, float first, int reach, Range *touched) {
    if (layers <= 0) return;
    float last_layer = first + (layers - 1);
    int64_t blocks = (height + ROW_BLOCK - 1) / ROW_BLOCK;
#ifdef __APPLE__
    dispatch_apply((size_t)blocks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
        touch_block((int64_t)block, geometry, grid, h, w, cell_y, cell_x, frac_y, frac_x, height, width, layers, first, last_layer, reach, touched);
    });
#else  // elsewhere (Linux, in the submission's container) the blocks one after another: the merged ranges are the same
    for (int64_t block = 0; block < blocks; block++)
        touch_block(block, geometry, grid, h, w, cell_y, cell_x, frac_y, frac_x, height, width, layers, first, last_layer, reach, touched);
#endif
}
