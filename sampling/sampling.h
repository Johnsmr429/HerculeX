#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define LAYERS 31
#define SAMPLE_TILE 128
#define NORMAL_UM 225.0
#define SMOOTHING 1.5
#define BRIDGE 8
#define RELAXATIONS 40
#define FOLD_CELLS 3.0
#define MARGIN 2
#define SHEET_HALF_UM 28.0
#define SAFE_VOXELS 2
#define SAMPLING "mesh-bridged-8-smoothed-1.5"
#define FIBRE_TILE 320
#define SMALLEST_FIBRE_TILE 160
#define FIBRE_TILES 8
#define FEWEST_FIBRE_TILES 3
#define TRIES_PER_TILE 8
#define FACE_SEARCH 8
#define SHEET_INSET 12
#define TILE_INSET 20
#define FEWEST_PIXELS 8000

f32 pairwise(const f32 *a, i64 n) {
    if (n < 8) {
        f32 r = -0.0f;
        for (i64 i = 0; i < n; i++) {
            r += a[i];
        }
        return r;
    }
    if (n <= 128) {
        f32 r[8];
        for (int j = 0; j < 8; j++) {
            r[j] = a[j];
        }
        i64 i = 8;
        for (; i < n - n % 8; i += 8) {
            for (int j = 0; j < 8; j++) {
                r[j] += a[i + j];
            }
        }
        f32 s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            s += a[i];
        }
        return s;
    }
    i64 half = n / 2 - n / 2 % 8;
    return pairwise(a, half) + pairwise(a + half, n - half);
}

f64 pairwise64(const f64 *a, i64 n) {
    if (n < 8) {
        f64 r = -0.0;
        for (i64 i = 0; i < n; i++) {
            r += a[i];
        }
        return r;
    }
    if (n <= 128) {
        f64 r[8];
        for (int j = 0; j < 8; j++) {
            r[j] = a[j];
        }
        i64 i = 8;
        for (; i < n - n % 8; i += 8) {
            for (int j = 0; j < 8; j++) {
                r[j] += a[i + j];
            }
        }
        f64 s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            s += a[i];
        }
        return s;
    }
    i64 half = n / 2 - n / 2 % 8;
    return pairwise64(a, half) + pairwise64(a + half, n - half);
}

f32 mean_of(const f32 *a, i64 n) { return pairwise(a, n) / (f32)n; }

i64 reflected(i64 i, i64 n) {
    i64 m = i % (2 * n);
    m += m < 0 ? 2 * n : 0;
    return m < n ? m : 2 * n - 1 - m;
}

void gaussian_axis(const f32 *in, f32 *out, const i64 *dims, int rank, int axis, f64 sigma) {
    int radius = (int)floor(4.0 * sigma + 0.5);
    f64 weights[512], sum = 0;
    for (int k = -radius; k <= radius; k++) {
        weights[k + radius] = exp(-0.5 / (sigma * sigma) * (f64)(k * k));
        sum += weights[k + radius];
    }
    for (int k = 0; k <= 2 * radius; k++) {
        weights[k] /= sum;
    }
    int plain = radius >= 10 ? radius / 8 * 8 : 0;
    i64 outer = 1, inner = 1, length = dims[axis];
    for (int k = 0; k < axis; k++) {
        outer *= dims[k];
    }
    for (int k = axis + 1; k < rank; k++) {
        inner *= dims[k];
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (inner >= 8) {
        for (i64 o = 0; o < outer; o++) {
            const f32 *source = in + o * length * inner;
            for (i64 l = 0; l < length; l++) {
                const f32 *centre = source + l * inner;
                f32 *target = out + (o * length + l) * inner;
                const f32 *below[256], *above[256];
                for (int tap = 0; tap < radius; tap++) {
                    int j = radius - tap;
                    below[tap] = source + reflected(l - j, length) * inner;
                    above[tap] = source + reflected(l + j, length) * inner;
                }
                i64 i = 0;
                for (; i + 8 <= inner; i += 8) {
                    float64x2_t sums[4];
                    for (int k = 0; k < 4; k++) {
                        sums[k] = vmulq_n_f64(
                            vcvt_f64_f32(vld1_f32(centre + i + 2 * k)), weights[radius]);
                    }
                    for (int tap = 0; tap < radius; tap++) {
                        for (int k = 0; k < 4; k++) {
                            i64 start = i + 2 * k;
                            float64x2_t a = vcvt_f64_f32(vld1_f32(below[tap] + start));
                            float64x2_t b = vcvt_f64_f32(vld1_f32(above[tap] + start));
                            float64x2_t pair = vaddq_f64(a, b);
                            sums[k] = tap < plain
                                ? vaddq_f64(sums[k], vmulq_n_f64(pair, weights[tap]))
                                : vfmaq_n_f64(sums[k], pair, weights[tap]);
                        }
                    }
                    for (int k = 0; k < 4; k++) {
                        vst1_f32(target + i + 2 * k, vcvt_f32_f64(sums[k]));
                    }
                }
                for (; i < inner; i++) {
                    f64 total = centre[i] * weights[radius];
                    for (int tap = 0; tap < radius; tap++) {
                        f64 pair = (f64)below[tap][i] + above[tap][i];
                        total = tap < plain ? total + pair * weights[tap]
                                            : fma(pair, weights[tap], total);
                    }
                    target[i] = total;
                }
            }
        }
        return;
    }
#endif
    f64 *line = malloc((length + 2 * radius) * sizeof(f64));
    for (i64 o = 0; o < outer; o++) {
        for (i64 i = 0; i < inner; i++) {
            const f32 *source = in + o * length * inner + i;
            for (i64 k = -radius; k < length + radius; k++) {
                line[k + radius] = source[reflected(k, length) * inner];
            }
            f32 *target = out + o * length * inner + i;
            i64 l = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
            if (inner == 1) {
                for (; l + 8 <= length; l += 8) {
                    const f64 *centre = line + radius + l;
                    float64x2_t sums[4];
                    for (int k = 0; k < 4; k++) {
                        sums[k] = vmulq_n_f64(vld1q_f64(centre + 2 * k), weights[radius]);
                    }
                    for (int j = radius, tap = 0; j > 0; j--, tap++) {
                        for (int k = 0; k < 4; k++) {
                            const f64 *start = centre + 2 * k;
                            float64x2_t pair =
                                vaddq_f64(vld1q_f64(start - j), vld1q_f64(start + j));
                            sums[k] = tap < plain
                                ? vaddq_f64(sums[k], vmulq_n_f64(pair, weights[tap]))
                                : vfmaq_n_f64(sums[k], pair, weights[tap]);
                        }
                    }
                    for (int k = 0; k < 4; k++) {
                        vst1_f32(target + l + 2 * k, vcvt_f32_f64(sums[k]));
                    }
                }
            }
#endif
            for (; l < length; l++) {
                const f64 *centre = line + radius + l;
                f64 total = centre[0] * weights[radius];
                for (int j = radius, tap = 0; j > 0; j--, tap++) {
                    f64 pair = centre[-j] + centre[j];
                    total =
                        tap < plain ? total + pair * weights[tap] : fma(pair, weights[tap], total);
                }
                target[l * inner] = total;
            }
        }
    }
    free(line);
}

void gaussian(f32 *x, f32 *spare, const i64 *dims, int rank, const f64 *sigmas) {
    i64 n = 1;
    for (int d = 0; d < rank; d++) {
        n *= dims[d];
    }
    for (int axis = 0; axis < rank; axis++) {
        gaussian_axis(x, spare, dims, rank, axis, sigmas[axis]);
        memcpy(x, spare, n * sizeof(f32));
    }
}

void gaussian2(f32 *x, f32 *spare, int h, int w, f64 sigma) {
    i64 dims[2] = {h, w};
    f64 sigmas[2] = {sigma, sigma};
    gaussian(x, spare, dims, 2, sigmas);
}

void grow(u8 *m, u8 *t, int h, int w, int corners, int dilate) {
    static const int neighbours[][2] = {
        {0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    int taps = corners ? 9 : 5, target = dilate != 0;
    memcpy(t, m, (i64)h * w);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int found = 0;
            for (int k = 0; k < taps && !found; k++) {
                int yy = y + neighbours[k][0], xx = x + neighbours[k][1];
                int set = yy >= 0 && yy < h && xx >= 0 && xx < w && t[yy * w + xx];
                found = set == target;
            }
            m[y * w + x] = found ? target : !target;
        }
    }
}

static void fibre_inset(u8 *mask, u8 *distance, int side) {
    for (int y = 0; y < side; y++) {
        for (int x = 0; x < side; x++) {
            i64 i = (i64)y * side + x;
            if (!mask[i]) {
                distance[i] = 0;
                continue;
            }
            int above = y ? distance[i - side] : 0;
            int left = x ? distance[i - 1] : 0;
            int d = above < left ? above : left;
            distance[i] = d < SHEET_INSET ? d + 1 : SHEET_INSET + 1;
        }
    }
    for (int y = side - 1; y >= 0; y--) {
        for (int x = side - 1; x >= 0; x--) {
            i64 i = (i64)y * side + x;
            int below = y + 1 < side ? distance[i + side] : 0;
            int right = x + 1 < side ? distance[i + 1] : 0;
            int d = (below < right ? below : right) + 1;
            distance[i] = d < distance[i] ? d : distance[i];
            mask[i] = distance[i] > SHEET_INSET;
        }
    }
}

void fill_holes(u8 *m, int h, int w, i32 *queue) {
    u8 *outside = calloc((i64)h * w, 1);
    i64 head = 0, tail = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int edge = y == 0 || x == 0 || y == h - 1 || x == w - 1;
            if (edge && !m[y * w + x]) {
                outside[y * w + x] = 1;
                queue[tail++] = y * w + x;
            }
        }
    }
    while (head < tail) {
        i32 i = queue[head++];
        int y = i / w, x = i % w;
        int around[4][2] = {{y - 1, x}, {y + 1, x}, {y, x - 1}, {y, x + 1}};
        for (int k = 0; k < 4; k++) {
            int yy = around[k][0], xx = around[k][1];
            if (yy < 0 || yy >= h || xx < 0 || xx >= w || m[yy * w + xx] || outside[yy * w + xx]) {
                continue;
            }
            outside[yy * w + xx] = 1;
            queue[tail++] = yy * w + xx;
        }
    }
    for (i64 i = 0; i < (i64)h * w; i++) {
        m[i] = !outside[i];
    }
    free(outside);
}

void voronoi(i32 *fy, i32 *fx, int len, i64 step, int d, int other, i32 *g, i32 *ny, i32 *nx) {
    int l = -1;
    for (int i = 0; i < len; i++) {
        i64 at = i * step;
        if (fy[at] < 0) {
            continue;
        }
        f64 fd = d ? fx[at] : fy[at];
        f64 t = (d ? fy[at] : fx[at]) - other, wr = t * t;
        while (l >= 1) {
            i64 i1 = g[l] * step, i2 = g[l - 1] * step;
            f64 f1 = d ? fx[i1] : fy[i1], f2 = d ? fx[i2] : fy[i2];
            f64 a = f1 - f2, b = fd - f1, c = a + b;
            f64 tu = (d ? fy[i2] : fx[i2]) - other, tv = (d ? fy[i1] : fx[i1]) - other;
            if (c * (tv * tv) - b * (tu * tu) - a * wr - a * b * c <= 0.0) {
                break;
            }
            l--;
        }
        g[++l] = i;
    }
    int top = l;
    if (top < 0) {
        return;
    }
    l = 0;
    for (int i = 0; i < len; i++) {
        i64 a = g[l] * step;
        f64 t0 = d ? fy[a] - other : fy[a] - i, t1 = d ? fx[a] - i : fx[a] - other;
        f64 d1 = t0 * t0 + t1 * t1;
        while (l < top) {
            i64 b = g[l + 1] * step;
            f64 s0 = d ? fy[b] - other : fy[b] - i, s1 = d ? fx[b] - i : fx[b] - other;
            f64 d2 = s0 * s0 + s1 * s1;
            if (d1 <= d2) {
                break;
            }
            d1 = d2;
            l++;
        }
        ny[i] = fy[g[l] * step];
        nx[i] = fx[g[l] * step];
    }
    for (int i = 0; i < len; i++) {
        fy[i * step] = ny[i];
        fx[i * step] = nx[i];
    }
}

void nearest_good(const u8 *good, int h, int w, i32 *fy, i32 *fx) {
    int most = h > w ? h : w;
    i32 *g = malloc(most * 4), *ny = malloc(most * 4), *nx = malloc(most * 4);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            i64 i = (i64)y * w + x;
            fy[i] = good[i] ? y : -1;
            fx[i] = good[i] ? x : -1;
        }
        voronoi(fy + x, fx + x, h, w, 0, x, g, ny, nx);
    }
    for (int y = 0; y < h; y++) {
        voronoi(fy + (i64)y * w, fx + (i64)y * w, w, 1, 1, y, g, ny, nx);
    }
    free(g);
    free(ny);
    free(nx);
}

typedef struct {
    int h, w, stride;
    f64 voxel;
    f32 *xyz;
} Surface;

Surface load_surface(Arena *a, const char *folder) {
    Surface s = {0};
    int h, w;
    f32 *planes[3];
    for (int c = 0; c < 3; c++) {
        planes[c] = tiff_read(a, fmt(a, "%s/%c.tif", folder, "xyz"[c]), &w, &h);
    }
    s.h = h;
    s.w = w;
    s.xyz = push(a, 3 * (i64)h * w * 4);
    for (int c = 0; c < 3; c++) {
        memcpy(s.xyz + c * (i64)h * w, planes[c], (i64)h * w * 4);
    }
    json_object *meta = json_read(fmt(a, "%s/meta.json", folder));
    need(meta != 0, "%s has no meta.json", folder);
    s.stride = ji64(meta, "stride");
    s.voxel = jf64(meta, "voxel_um");
    json_object_put(meta);
    return s;
}

void bridged_mesh(Surface *s, u8 *good, f32 *spare, f32 *work, i32 *queue) {
    int h = s->h, w = s->w, pad = BRIDGE + 2, ph = h + 2 * pad, pw = w + 2 * pad;
    i64 n = (i64)h * w;
    i64 count = 0;
    for (i64 i = 0; i < n; i++) {
        good[i] = s->xyz[i] > 0;
        count += good[i];
    }
    if (count == n) {
        return;
    }
    u8 *closed = calloc((i64)ph * pw, 1), *scratch = malloc((i64)ph * pw), *gap = malloc(n);
    for (int y = 0; y < h; y++) {
        memcpy(closed + (i64)(y + pad) * pw + pad, good + (i64)y * w, w);
    }
    for (int k = 0; k < BRIDGE; k++) {
        grow(closed, scratch, ph, pw, 1, 1);
    }
    for (int k = 0; k < BRIDGE; k++) {
        grow(closed, scratch, ph, pw, 1, 0);
    }
    fill_holes(closed, ph, pw, queue);
    int any = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            i64 i = (i64)y * w + x;
            gap[i] = closed[(i64)(y + pad) * pw + pad + x] && !good[i];
            any |= gap[i];
        }
    }
    free(closed);
    free(scratch);
    if (!any) {
        free(gap);
        return;
    }
    f32 *weight = work, *grid = s->xyz;
    for (i64 i = 0; i < n; i++) {
        weight[i] = good[i];
    }
    gaussian2(weight, spare, h, w, BRIDGE / 2.0);
    for (i64 i = 0; i < n; i++) {
        weight[i] = weight[i] > 1e-6f ? weight[i] : 1e-6f;
    }
    f32 *plane = work + n;
    for (int c = 0; c < 3; c++) {
        f32 *p = grid + c * n;
        for (i64 i = 0; i < n; i++) {
            plane[i] = good[i] ? p[i] : 0;
        }
        gaussian2(plane, spare, h, w, BRIDGE / 2.0);
        for (i64 i = 0; i < n; i++) {
            p[i] = good[i] ? p[i] : plane[i] / weight[i];
        }
    }
    f32 *around = work;
    for (int r = 0; r < RELAXATIONS; r++) {
        for (int c = 0; c < 3; c++) {
            f32 *p = grid + c * n, *q = around + c * n;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    f32 up = p[(y > 0 ? y - 1 : 0) * w + x],
                        down = p[(y < h - 1 ? y + 1 : y) * w + x];
                    f32 left = p[y * w + (x > 0 ? x - 1 : 0)],
                        right = p[y * w + (x < w - 1 ? x + 1 : x)];
                    f32 total = up + down;
                    total = total + left;
                    total = total + right;
                    q[(i64)y * w + x] = 0.25f * total;
                }
            }
        }
        for (int c = 0; c < 3; c++) {
            for (i64 i = 0; i < n; i++) {
                grid[c * n + i] = gap[i] ? around[c * n + i] : grid[c * n + i];
            }
        }
    }
    u8 *fold = calloc(n, 1);
    f32 limit = FOLD_CELLS * s->stride;
    for (int axis = 0; axis < 2; axis++) {
        for (int y = 0; y < h - (axis == 0); y++) {
            for (int x = 0; x < w - (axis == 1); x++) {
                i64 i = (i64)y * w + x, j = axis == 0 ? i + w : i + 1;
                f32 total = 0;
                for (int c = 0; c < 3; c++) {
                    f32 d = grid[c * n + j] - grid[c * n + i], squared = d * d;
                    total = c ? total + squared : squared;
                }
                int known = (good[i] || gap[i]) && (good[j] || gap[j]);
                if (sqrtf(total) > limit && known) {
                    fold[i] = fold[j] = 1;
                }
            }
        }
    }
    for (i64 i = 0; i < n; i++) {
        int keep = (good[i] || gap[i]) && !(fold[i] && gap[i]);
        for (int c = 0; c < 3; c++) {
            grid[c * n + i] = keep ? grid[c * n + i] : -1;
        }
    }
    free(fold);
    free(gap);
}

void smoothed_mesh(Surface *s, u8 *good, f32 *spare, f32 *work) {
    int h = s->h, w = s->w;
    i64 n = (i64)h * w;
    f32 *weight = work, *plane = work + n;
    for (i64 i = 0; i < n; i++) {
        good[i] = s->xyz[i] > 0;
        weight[i] = good[i];
    }
    gaussian2(weight, spare, h, w, SMOOTHING);
    for (i64 i = 0; i < n; i++) {
        weight[i] = weight[i] > 1e-4f ? weight[i] : 1e-4f;
    }
    for (int c = 0; c < 3; c++) {
        f32 *p = s->xyz + c * n;
        for (i64 i = 0; i < n; i++) {
            plane[i] = good[i] ? p[i] : 0;
        }
        gaussian2(plane, spare, h, w, SMOOTHING);
        for (i64 i = 0; i < n; i++) {
            p[i] = good[i] ? plane[i] / weight[i] : -1;
        }
    }
}

Surface extended(Arena *a, Surface s) {
    int h = s.h + 2 * MARGIN, w = s.w + 2 * MARGIN;
    i64 n = (i64)h * w;
    Surface e = s;
    e.h = h;
    e.w = w;
    e.xyz = push(a, 3 * n * 4);
    f32 *before = malloc(3 * n * 4);
    for (i64 i = 0; i < 3 * n; i++) {
        e.xyz[i] = -1;
    }
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < s.h; y++) {
            memcpy(e.xyz + c * n + (i64)(y + MARGIN) * w + MARGIN,
                s.xyz + c * (i64)s.h * s.w + (i64)y * s.w, s.w * 4);
        }
    }
    int steps[4][2] = {{0, 1}, {0, -1}, {1, 1}, {1, -1}};
    for (int k = 0; k < MARGIN; k++) {
        memcpy(before, e.xyz, 3 * n * 4);
        for (int d = 0; d < 4; d++) {
            int axis = steps[d][0], step = steps[d][1];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int ny = axis ? y : ((y - step) % h + h) % h,
                        nx = axis ? ((x - step) % w + w) % w : x;
                    int fy = axis ? y : ((y - 2 * step) % h + h) % h;
                    int fx = axis ? ((x - 2 * step) % w + w) % w : x;
                    i64 i = (i64)y * w + x, near = (i64)ny * w + nx, far = (i64)fy * w + fx;
                    if (!(e.xyz[i] <= 0 && before[near] > 0 && before[far] > 0)) {
                        continue;
                    }
                    for (int c = 0; c < 3; c++) {
                        f32 twice = 2 * before[c * n + near];
                        e.xyz[c * n + i] = twice - before[c * n + far];
                    }
                }
            }
        }
    }
    free(before);
    return e;
}

int by_float(const void *a, const void *b) {
    f32 x = *(const f32 *)a, y = *(const f32 *)b;
    return (x > y) - (x < y);
}

f64 interpolated(f64 z, const f64 *line, int rows, int column) {
    if (z <= line[0]) {
        return line[column];
    }
    if (z >= line[(rows - 1) * 3]) {
        return line[(rows - 1) * 3 + column];
    }
    int k = 0;
    while (line[(k + 1) * 3] <= z) {
        k++;
    }
    f64 z0 = line[k * 3], z1 = line[(k + 1) * 3], v0 = line[k * 3 + column],
        v1 = line[(k + 1) * 3 + column];
    f64 slope = (v1 - v0) / (z1 - z0);
    return slope * (z - z0) + v0;
}

void grid_normals(Surface *e, f64 centre_x, f64 centre_y, f32 *normal, f32 *spare) {
    int h = e->h, w = e->w;
    i64 n = (i64)h * w;
    u8 *good = malloc(n);
    i32 *fy = malloc(n * 4), *fx = malloc(n * 4);
    for (i64 i = 0; i < n; i++) {
        good[i] = e->xyz[i] > 0;
    }
    nearest_good(good, h, w, fy, fx);
    f32 *smooth = malloc(3 * n * 4);
    f64 sigma = NORMAL_UM / (e->stride * e->voxel);
    for (int c = 0; c < 3; c++) {
        for (i64 i = 0; i < n; i++) {
            smooth[c * n + i] = e->xyz[c * n + (i64)fy[i] * w + fx[i]];
        }
        gaussian2(smooth + c * n, spare, h, w, sigma);
    }
    f32 *du = malloc(3 * n * 4), *dv = malloc(3 * n * 4);
    for (int c = 0; c < 3; c++) {
        const f32 *p = smooth + c * n;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                i64 i = (i64)y * w + x;
                du[c * n + i] = h < 2 ? 0
                    : y == 0          ? (p[i + w] - p[i]) / 1.0f
                    : y == h - 1      ? (p[i] - p[i - w]) / 1.0f
                                      : (p[i + w] - p[i - w]) / 2.0f;
                dv[c * n + i] = w < 2 ? 0
                    : x == 0          ? (p[i + 1] - p[i]) / 1.0f
                    : x == w - 1      ? (p[i] - p[i - 1]) / 1.0f
                                      : (p[i + 1] - p[i - 1]) / 2.0f;
            }
        }
    }
    f64 sign = 0;
    for (i64 i = 0; i < n; i++) {
        f32 a0 = du[i], a1 = du[n + i], a2 = du[2 * n + i];
        f32 b0 = dv[i], b1 = dv[n + i], b2 = dv[2 * n + i];
        f32 p0 = a1 * b2, q0 = a2 * b1, p1 = a2 * b0, q1 = a0 * b2, p2 = a0 * b1, q2 = a1 * b0;
        f32 c[3] = {p0 - q0, p1 - q1, p2 - q2};
        f32 s0 = c[0] * c[0], s1 = c[1] * c[1], s2 = c[2] * c[2];
        f32 total = s0 + s1;
        total = total + s2;
        f32 length = sqrtf(total);
        length = length > 1e-9f ? length : 1e-9f;
        for (int k = 0; k < 3; k++) {
            normal[k * n + i] = c[k] / length;
        }
        if (good[i]) {
            f64 ix = centre_x - smooth[i], iy = centre_y - smooth[n + i];
            sign += normal[i] * ix + normal[n + i] * iy;
        }
    }
    if (sign < 0) {
        for (i64 i = 0; i < 3 * n; i++) {
            normal[i] = -normal[i];
        }
    }
    free(good);
    free(fy);
    free(fx);
    free(smooth);
    free(du);
    free(dv);
}

typedef struct {
    Surface margined;
    f32 *normal;
} Readable;

Readable readable_cells(Arena *a, Surface s, const f64 *line, int rows, int smooth) {
    i64 n = (i64)s.h * s.w;
    u8 *good = malloc(n);
    f32 *spare = malloc(n * 4), *work = malloc(3 * n * 4);
    i32 *queue = malloc(((i64)(s.h + 2 * BRIDGE + 4) * (s.w + 2 * BRIDGE + 4)) * 4);
    bridged_mesh(&s, good, spare, work, queue);
    if (smooth) {
        smoothed_mesh(&s, good, spare, work);
    }
    f32 *z = malloc(n * 4);
    i64 count = 0;
    for (i64 i = 0; i < n; i++) {
        if (s.xyz[i] > 0) {
            z[count++] = s.xyz[2 * n + i];
        }
    }
    qsort(z, count, 4, by_float);
    f32 middle = count % 2 ? z[count / 2] : (z[count / 2 - 1] + z[count / 2]) / 2.0f;
    Readable r = {.margined = extended(a, s)};
    i64 m = (i64)r.margined.h * r.margined.w;
    r.normal = push(a, 3 * m * 4);
    f32 *wide = malloc(m * 4);
    grid_normals(&r.margined, interpolated(middle, line, rows, 1),
        interpolated(middle, line, rows, 2), r.normal, wide);
    free(good);
    free(spare);
    free(work);
    free(queue);
    free(z);
    free(wide);
    return r;
}

Readable readable(Arena *a, Surface s, const f64 *line, int rows) {
    return readable_cells(a, s, line, rows, 1);
}

typedef struct {
    i32 *cell;
    f32 *part;
    i64 count;
} Cells;

Cells column_cells(Arena *a, i64 cells, int stride) {
    Cells c = {.count = (cells - 1) * stride + 1};
    c.cell = push(a, c.count * 4);
    c.part = push(a, c.count * 4);
    for (i64 i = 0; i < c.count; i++) {
        f64 place = (f64)i / stride;
        place = place < cells - 1.000001 ? place : cells - 1.000001;
        c.cell[i] = (i32)place;
        c.part[i] = place - c.cell[i];
    }
    return c;
}

f32 voxel_at(const Bricks *b, i64 z, i64 y, i64 x) {
    if (z < 0 || y < 0 || x < 0) {
        return 0;
    }
    i64 iz = z / b->brick[0], iy = y / b->brick[1], ix = x / b->brick[2];
    if (iz >= b->grid[0] || iy >= b->grid[1] || ix >= b->grid[2]) {
        return 0;
    }
    i64 o = b->offsets[(iz * b->grid[1] + iy) * b->grid[2] + ix];
    if (o < 0) {
        return 0;
    }
    return b->data[o + ((z % b->brick[0]) * b->brick[1] + y % b->brick[1]) * b->brick[2] +
        x % b->brick[2]];
}

f32 trilinear(const Bricks *b, i64 iz, i64 iy, i64 ix, f32 fz, f32 fy, f32 fx) {
    f32 v = 0;
    for (int k = 0; k < 8; k++) {
        int a = k >> 2, c = k >> 1 & 1, e = k & 1;
        f32 w = (a ? fz : 1 - fz) * (c ? fy : 1 - fy);
        w = w * (e ? fx : 1 - fx);
        v = fmaf(w, voxel_at(b, iz + a, iy + c, ix + e), v);
    }
    return v;
}

f32 grid_value(const f32 *plane, i64 w, int a, int b, f32 fy, f32 fx) {
    const f32 *top = plane + (i64)a * w + b, *low = top + w;
    f32 ax = 1 - fx, ay = 1 - fy;
    f32 t0 = top[0] * ax, t1 = top[1] * fx, l0 = low[0] * ax, l1 = low[1] * fx;
    f32 upper = t0 + t1, lower = l0 + l1;
    f32 u = upper * ay, l = lower * fy;
    return u + l;
}

int column_at(const f32 *grid, i64 h, i64 w, int a, int b, f32 fy, f32 fx, f32 *p, f32 *nv) {
    i64 plane = h * w;
    if (!(grid_value(grid, w, a, b, fy, fx) > 0.999f)) {
        return 0;
    }
    f32 m[3];
    for (int i = 0; i < 3; i++) {
        p[i] = grid_value(grid + (1 + i) * plane, w, a, b, fy, fx);
        m[i] = grid_value(grid + (4 + i) * plane, w, a, b, fy, fx);
    }
    f32 s0 = m[0] * m[0], s1 = m[1] * m[1], s2 = m[2] * m[2];
    f32 total = s0 + s1;
    total = total + s2;
    f32 length = sqrtf(total);
    length = length < 1e-9f ? 1e-9f : length;
    for (int i = 0; i < 3; i++) {
        nv[i] = m[i] / length;
    }
    return 1;
}

u8 sample_voxel(const Bricks *b, f32 x, f32 y, f32 z, i64 *last, const u8 **brick) {
    i64 ix = floorf(x), iy = floorf(y), iz = floorf(z);
    f32 fx = x - ix, fy = y - iy, fz = z - iz, v;
    i64 sz = iz / b->brick[0], sy = iy / b->brick[1], sx = ix / b->brick[2];
    i64 rz = iz - sz * b->brick[0], ry = iy - sy * b->brick[1], rx = ix - sx * b->brick[2];
    int edge = iz < 0 || iy < 0 || ix < 0 || rz + 1 >= b->brick[0] || ry + 1 >= b->brick[1] ||
        rx + 1 >= b->brick[2] || sz >= b->grid[0] || sy >= b->grid[1] || sx >= b->grid[2];
    if (edge) {
        v = trilinear(b, iz, iy, ix, fz, fy, fx);
        return lrintf(v);
    }
    i64 key = (sz * b->grid[1] + sy) * b->grid[2] + sx;
    if (key != *last) {
        i64 o = b->offsets[key];
        *brick = o < 0 ? 0 : b->data + o;
        *last = key;
    }
    if (!*brick) {
        return 0;
    }
    i64 row = b->brick[2], plane = b->brick[1] * row;
    const u8 *p = *brick + rz * plane + ry * row + rx;
    f32 az = 1 - fz, ay = 1 - fy, ax = 1 - fx;
    f32 weights[8] = {az * ay * ax, az * ay * fx, az * fy * ax, az * fy * fx, fz * ay * ax,
        fz * ay * fx, fz * fy * ax, fz * fy * fx};
    i64 at[8] = {0, 1, row, row + 1, plane, plane + 1, plane + row, plane + row + 1};
    v = 0;
    for (int k = 0; k < 8; k++) {
        v = fmaf(weights[k], p[at[k]], v);
    }
    return lrintf(v);
}

typedef struct {
    const Bricks *store;
    const f32 *grid;
    i64 h, w, height, width;
    Cells rows, cols;
    int layers;
    u8 *stack;
} Flatten;

void flatten_tile(void *context, i64 tile) {
    Flatten *f = context;
    i64 across = (f->width + SAMPLE_TILE - 1) / SAMPLE_TILE;
    i64 y0 = tile / across * SAMPLE_TILE, x0 = tile % across * SAMPLE_TILE;
    int half = f->layers / 2;
    f32 first = -half;
    for (i64 y = y0; y < y0 + SAMPLE_TILE && y < f->height; y++) {
        for (i64 x = x0; x < x0 + SAMPLE_TILE && x < f->width; x++) {
            f32 p[3], n[3];
            int kept = column_at(f->grid, f->h, f->w, f->rows.cell[y], f->cols.cell[x],
                f->rows.part[y], f->cols.part[x], p, n);
            i64 last = -1;
            const u8 *brick = 0;
            for (int l = 0; l < f->layers && kept; l++) {
                f32 d = first + l;
                f->stack[((i64)l * f->height + y) * f->width + x] = sample_voxel(f->store,
                    fmaf(d, n[0], p[0]), fmaf(d, n[1], p[1]), fmaf(d, n[2], p[2]), &last, &brick);
            }
        }
    }
}

f32 *column_grid(Readable *r, int turn) {
    Surface *e = &r->margined;
    i64 n = (i64)e->h * e->w;
    f32 *grid = malloc(7 * n * 4);
    for (i64 i = 0; i < n; i++) {
        grid[i] = e->xyz[i] > 0;
    }
    memcpy(grid + n, e->xyz, 3 * n * 4);
    for (i64 i = 0; i < 3 * n; i++) {
        grid[4 * n + i] = r->normal[i] * turn;
    }
    return grid;
}

typedef struct {
    int layers, height, width;
    u8 *cube;
} Stack;

Stack flatten(Arena *a, Readable *r, int turn, const Bricks *store, int layers) {
    Surface *e = &r->margined;
    f32 *grid = column_grid(r, turn);
    Flatten f = {.store = store, .grid = grid, .h = e->h, .w = e->w, .layers = layers};
    f.rows = column_cells(a, e->h, e->stride);
    f.cols = column_cells(a, e->w, e->stride);
    f.height = f.rows.count;
    f.width = f.cols.count;
    f.stack = pushz(a, (i64)layers * f.height * f.width);
    i64 tiles =
        ((f.height + SAMPLE_TILE - 1) / SAMPLE_TILE) * ((f.width + SAMPLE_TILE - 1) / SAMPLE_TILE);
    parallel(tiles, cores(), flatten_tile, &f);
    free(grid);
    return (Stack){layers, f.height, f.width, f.stack};
}

void include(i32 *range, i32 first, i32 last) {
    range[0] = first < range[0] ? first : range[0];
    range[1] = last > range[1] ? last : range[1];
}

int brick_below(f32 value, i64 side, int reach) { return floorf((floorf(value) - reach) / side); }

int brick_above(f32 value, i64 side, int reach) {
    return floorf((floorf(value) + 1 + reach) / side);
}

void mark_range(i32 *touched, const Bricks *b, int x0, int x1, int y0, int y1, int low, int high) {
    x0 = x0 < 0 ? 0 : x0;
    y0 = y0 < 0 ? 0 : y0;
    x1 = x1 >= b->grid[2] ? b->grid[2] - 1 : x1;
    y1 = y1 >= b->grid[1] ? b->grid[1] - 1 : y1;
    low = low < 0 ? 0 : low;
    high = high >= b->grid[0] * b->brick[0] ? b->grid[0] * b->brick[0] - 1 : high;
    for (int z = low; z <= high;) {
        int iz = z / b->brick[0], start = z - iz * b->brick[0];
        int stop = high - iz * b->brick[0] + 1;
        stop = stop > b->brick[0] ? b->brick[0] : stop;
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                include(
                    touched + 2 * (((i64)iz * b->grid[1] + y) * b->grid[2] + x), start, stop - 1);
            }
        }
        z += stop - start;
    }
}

typedef struct {
    int x0, x1, y0, y1;
} BrickFootprint;

static BrickFootprint brick_footprint(const Bricks *b, const f32 *p, const f32 *n, int layer) {
    f32 x = fmaf(layer, n[0], p[0]), y = fmaf(layer, n[1], p[1]);
    return (BrickFootprint){brick_below(x, b->brick[2], SAFE_VOXELS),
        brick_above(x, b->brick[2], SAFE_VOXELS), brick_below(y, b->brick[1], SAFE_VOXELS),
        brick_above(y, b->brick[1], SAFE_VOXELS)};
}

static int same_footprint(BrickFootprint x, BrickFootprint y) {
    return x.x0 == y.x0 && x.x1 == y.x1 && x.y0 == y.y0 && x.y1 == y.y1;
}

static void mark_ray_span(i32 *touched, const Bricks *b, BrickFootprint r, f32 z0, f32 z1) {
    mark_range(touched, b, r.x0, r.x1, r.y0, r.y1, (int)floorf(fminf(z0, z1)) - SAFE_VOXELS,
        (int)floorf(fmaxf(z0, z1)) + 1 + SAFE_VOXELS);
}

static void mark_crossing_ray(i32 *touched, const Bricks *b, const f32 *p, const f32 *n, int first,
    int last, BrickFootprint start, BrickFootprint end, f32 z0, f32 z1) {
    if (!(last >= first && last - first <= (1 << 24) && fabsf(z0) <= 0x1p23f &&
            fabsf(z1) <= 0x1p23f && fabsf(n[2]) <= 2)) {
        for (int layer = first; layer <= last; layer++) {
            BrickFootprint r = brick_footprint(b, p, n, layer);
            f32 z = fmaf(layer, n[2], p[2]);
            mark_ray_span(touched, b, r, z, z);
        }
        return;
    }
    while (!same_footprint(start, end)) {
        int lo = first, hi = last;
        BrickFootprint following = end;
        while (hi - lo > 1) {
            int mid = lo + (hi - lo) / 2;
            BrickFootprint r = brick_footprint(b, p, n, mid);
            if (same_footprint(r, start)) {
                lo = mid;
            } else {
                hi = mid;
                following = r;
            }
        }
        mark_ray_span(touched, b, start, z0, fmaf(lo, n[2], p[2]));
        first = hi;
        start = following;
        z0 = fmaf(first, n[2], p[2]);
    }
    mark_ray_span(touched, b, start, z0, z1);
}

void mark_ranges_layers(i32 *touched, const Bricks *b, Readable *r, Arena *a, int layers) {
    Surface *e = &r->margined;
    f32 *grid = column_grid(r, 1);
    Cells rows = column_cells(a, e->h, e->stride), cols = column_cells(a, e->w, e->stride);
    int first = -(layers / 2), last = first + layers - 1;
    for (i64 y = 0; y < rows.count; y++) {
        for (i64 x = 0; x < cols.count; x++) {
            f32 p[3], n[3];
            if (!column_at(grid, e->h, e->w, rows.cell[y], cols.cell[x], rows.part[y], cols.part[x],
                    p, n)) {
                continue;
            }
            BrickFootprint start = brick_footprint(b, p, n, first);
            BrickFootprint end = brick_footprint(b, p, n, last);
            f32 z0 = fmaf(first, n[2], p[2]), z1 = fmaf(last, n[2], p[2]);
            if (same_footprint(start, end)) {
                mark_ray_span(touched, b, start, z0, z1);
            } else {
                mark_crossing_ray(touched, b, p, n, first, last, start, end, z0, z1);
            }
        }
    }
    free(grid);
}

void mark_ranges(i32 *touched, const Bricks *b, Readable *r, Arena *a) {
    mark_ranges_layers(touched, b, r, a, LAYERS);
}

int sheet_layers(f64 voxel) { return 2 * (int)ceil(SHEET_HALF_UM / voxel) + 1; }

void across_sheet(Stack s, f64 voxel, u8 *sheet, u8 *layer0) {
    int middle = s.layers / 2, reach = sheet_layers(voxel) / 2;
    i64 n = (i64)s.height * s.width;
    memcpy(layer0, s.cube + middle * n, n);
    memset(sheet, 0, n);
    for (int l = middle - reach; l <= middle + reach; l++) {
        const u8 *layer = s.cube + l * n;
        for (i64 i = 0; i < n; i++) {
            sheet[i] = layer[i] > sheet[i] ? layer[i] : sheet[i];
        }
    }
}

void face_depth(
    const f32 *smoothed, i64 layer_stride, i64 layers, i64 pixels, i64 start, f64 *depth) {
    i64 falls = layers - 1;
    f32 *least = malloc(pixels * 4);
    i32 *steepest = calloc(pixels, 4);
    for (i64 i = 0; i < pixels; i++) {
        least[i] = smoothed[layer_stride + i] - smoothed[i];
    }
    for (i64 z = 1; z < falls; z++) {
        const f32 *a = smoothed + z * layer_stride, *b = a + layer_stride;
        for (i64 i = 0; i < pixels; i++) {
            f32 fall = b[i] - a[i];
            if (fall < least[i]) {
                least[i] = fall;
                steepest[i] = z;
            }
        }
    }
    for (i64 i = 0; i < pixels; i++) {
        i64 s = steepest[i];
        s = s < 1 ? 1 : s > falls - 2 ? falls - 2 : s;
        const f32 *at_layer = smoothed + s * layer_stride + i;
        f32 before = at_layer[0] - at_layer[-layer_stride],
            at = at_layer[layer_stride] - at_layer[0];
        f32 after = at_layer[2 * layer_stride] - at_layer[layer_stride];
        f32 twice = 2 * at, curve = (before - twice) + after;
        f32 offset = fabsf(curve) > 1e-3f ? (0.5f * (before - after)) / curve : 0.0f;
        offset = offset < -1 ? -1 : offset > 1 ? 1 : offset;
        depth[i] = ((f64)(start + s) + 0.5) + (f64)offset;
    }
    free(least);
    free(steepest);
}

void layer_at(const u8 *block, i64 layer_stride, i64 layers, i64 height, i64 width, i64 row_stride,
    const f64 *depth, f64 offset, f32 *out) {
    i64 last = layers - 1;
    for (i64 y = 0; y < height; y++) {
        const u8 *row = block + y * row_stride;
        for (i64 x = 0; x < width; x++) {
            f64 d = depth[y * width + x] + offset, floored = floor(d);
            i64 whole = floored;
            i64 low = whole < 0 ? 0 : whole > last ? last : whole;
            i64 high = whole + 1 < 0 ? 0 : whole + 1 > last ? last : whole + 1;
            f32 part = d - floored;
            f32 lower = row[low * layer_stride + x], upper = row[high * layer_stride + x];
            f32 a = lower * (1.0f - part), b = upper * part;
            out[y * width + x] = a + b;
        }
    }
}

f32 grain(const f32 *image, const u8 *where, int side, f32 *work) {
    i64 n = (i64)side * side;
    f32 *smooth[4], *spare = work + 4 * n, *across = work + 5 * n, *along = work + 6 * n;
    f64 widths[4] = {1.5, 3.0, 3.0, 6.0};
    for (int k = 0; k < 4; k++) {
        smooth[k] = k == 2 ? smooth[1] : work + k * n;
        if (k != 2) {
            memcpy(smooth[k], image, n * 4);
            gaussian2(smooth[k], spare, side, side, widths[k]);
        }
    }
    f32 ratios[2];
    i64 count = 0;
    for (i64 i = 0; i < n; i++) {
        count += where[i];
    }
    for (int s = 0; s < 2; s++) {
        f32 *fine = smooth[2 * s], *coarse = smooth[2 * s + 1];
        f32 *band = work + 7 * n;
        for (i64 i = 0; i < n; i++) {
            band[i] = fine[i] - coarse[i];
        }
        i64 k = 0;
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                i64 i = (i64)y * side + x;
                if (!where[i]) {
                    continue;
                }
                f32 a = y == 0      ? (band[i + side] - band[i]) / 1.0f
                    : y == side - 1 ? (band[i] - band[i - side]) / 1.0f
                                    : (band[i + side] - band[i - side]) / 2.0f;
                f32 b = x == 0      ? (band[i + 1] - band[i]) / 1.0f
                    : x == side - 1 ? (band[i] - band[i - 1]) / 1.0f
                                    : (band[i + 1] - band[i - 1]) / 2.0f;
                across[k] = a * a;
                along[k++] = b * b;
            }
        }
        f32 top = mean_of(across, count), bottom = mean_of(along, count);
        top = top > 1e-9f ? top : (f32)1e-9;
        bottom = bottom > 1e-9f ? bottom : (f32)1e-9;
        ratios[s] = logf(top / bottom);
    }
    return (ratios[0] + ratios[1]) / 2.0f;
}

f64 outer_ply_grain(const u8 *block, i64 layer_stride, int layers, int side, const u8 *where,
    const f32 *smoothed, i64 smoothed_stride, f32 *work) {
    i64 n = (i64)side * side;
    f64 *depth = malloc(n * 8);
    face_depth(smoothed, smoothed_stride, 2 * FACE_SEARCH + 1, n, layers / 2 - FACE_SEARCH, depth);
    f32 *image = work + 8 * n;
    f64 grains[3];
    for (int k = 0; k < 3; k++) {
        layer_at(block, layer_stride, layers, side, side, side, depth, k - 3, image);
        grains[k] = grain(image, where, side, work);
    }
    free(depth);
    return (grains[0] + grains[1] + grains[2]) / 3;
}

typedef struct {
    const char *face;
    f64 polarity, error;
    int tiles;
} Recto;

Recto recto_face(Stack s) {
    int side = s.height < s.width ? s.height : s.width;
    side = side < FIBRE_TILE ? side : FIBRE_TILE;
    Recto r = {.face = "unknown"};
    if (side < SMALLEST_FIBRE_TILE) {
        return r;
    }
    i64 n = (i64)side * side;
    Pcg64 rng = pcg64((u64)s.height * 31 + s.width);
    u8 *block = malloc(s.layers * n), *where = malloc(n), *scratch = malloc(n);
    f32 *work = malloc(9 * n * 4), *smoothed = malloc((2 * FACE_SEARCH + 1) * n * 4);
    f32 *spare = malloc((2 * FACE_SEARCH + 1) * n * 4);
    f64 values[FIBRE_TILES];
    for (int t = 0; t < FIBRE_TILES * TRIES_PER_TILE && r.tiles < FIBRE_TILES; t++) {
        i64 y = draw(&rng, s.height - side + 1), x = draw(&rng, s.width - side + 1);
        for (int l = 0; l < s.layers; l++) {
            for (int row = 0; row < side; row++) {
                memcpy(block + l * n + (i64)row * side,
                    s.cube + ((i64)l * s.height + y + row) * s.width + x, side);
            }
        }
        i64 covered = 0, judged = 0;
        for (i64 i = 0; i < n; i++) {
            where[i] = block[s.layers / 2 * n + i] > 0;
            covered += where[i];
        }
        if (covered < 0.5 * n) {
            continue;
        }
        fibre_inset(where, scratch, side);
        for (int yy = 0; yy < side; yy++) {
            for (int xx = 0; xx < side; xx++) {
                int inside = yy >= TILE_INSET && yy < side - TILE_INSET && xx >= TILE_INSET &&
                    xx < side - TILE_INSET;
                where[(i64)yy * side + xx] &= inside;
                judged += where[(i64)yy * side + xx];
            }
        }
        if (judged < FEWEST_PIXELS) {
            continue;
        }
        int start = s.layers / 2 - FACE_SEARCH;
        for (i64 i = 0; i < (2 * FACE_SEARCH + 1) * n; i++) {
            smoothed[i] = block[start * n + i];
        }
        i64 dims[3] = {2 * FACE_SEARCH + 1, side, side};
        f64 sigmas[3] = {0.7, 2.5, 2.5};
        gaussian(smoothed, spare, dims, 3, sigmas);
        f64 near = outer_ply_grain(block, n, s.layers, side, where, smoothed, n, work);
        f64 far = outer_ply_grain(block + (s.layers - 1) * n, -n, s.layers, side, where,
            smoothed + 2 * FACE_SEARCH * n, -n, work);
        values[r.tiles++] = near - far;
    }
    free(block);
    free(where);
    free(scratch);
    free(work);
    free(smoothed);
    free(spare);
    if (r.tiles < FEWEST_FIBRE_TILES) {
        return r;
    }
    f64 mean = pairwise64(values, r.tiles) / r.tiles, squares[FIBRE_TILES];
    for (int k = 0; k < r.tiles; k++) {
        f64 d = values[k] - mean;
        squares[k] = d * d;
    }
    f64 error = sqrt(pairwise64(squares, r.tiles) / r.tiles) / sqrt((f64)r.tiles);
    f64 bar = 2.5 * (error > 0.03 ? error : 0.03);
    r.face = fabs(mean) >= bar ? (mean > 0 ? "near" : "far") : "unknown";
    char text[64];
    snprintf(text, sizeof(text), "%.4f", mean);
    r.polarity = strtod(text, 0);
    snprintf(text, sizeof(text), "%.4f", error);
    r.error = strtod(text, 0);
    return r;
}
