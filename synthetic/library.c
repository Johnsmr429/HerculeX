#include <glob.h>
#include "scans/chunks/cached_ct.h"

enum {
    SLAB_SIDE = 512,
    SLAB_DEPTH = 12,
    SLAB_SNAP = 4,
    SLAB_GAP_STEP = 4,
    SLAB_AIR_SIDE = 32,
    SLAB_AIR_REACH = 160
};

typedef struct {
    const char *package, *scroll, *folder, *out, *chunks, *scan, *id;
    Array points;
    double low, high, voxel;
    HxCachedCt ct;
} SlabLibrary;

static const char *slab_scan(const char *scroll) {
    const char *scans[][2] = {
        {"PHerc0139", "20250728140407-9.362um-1.2m-113keV-masked"},
        {"PHerc0172", "20241024131839-7.910um-53keV-masked"},
        {"PHerc1667", "20231117161658-7.910um-53keV-masked"},
        {"PHercParis4", "20230205180739-7.910um-54keV-masked"},
    };
    for (size_t i = 0; i < sizeof(scans) / sizeof(*scans); i++) {
        if (!strcmp(scroll, scans[i][0])) {
            return scans[i][1];
        }
    }
    die("no slab-library scan for %s", scroll);
    return NULL;
}

static void slab_find_cache(SlabLibrary *lib) {
    int status = hx_cached_ct_open(&lib->ct, lib->chunks, lib->scroll, lib->scan, 0);
    if (status == 1) {
        status = hx_cached_ct_open(&lib->ct, lib->chunks, lib->scroll, lib->id, 0);
    }
    need(status >= 0, "%s: cannot open level-0 CT cache for %s/%s", lib->chunks, lib->scroll,
        lib->scan);
}

static SlabLibrary slab_open(Arena *a, const char *package, const char *scan_scroll) {
    need(*package && !strchr(package, '/') && strcmp(package, ".") != 0 &&
            strcmp(package, "..") != 0,
        "invalid package name: %s", package);
    size_t length = strcspn(package, "-");
    const char *scroll = scan_scroll ? scan_scroll : fmt(a, "%.*s", (int)length, package);
    const char *folder = fmt(a, "%s/bench/controls/%s", OUTPUTS, package);
    const char *truth = fmt(a, "%s/truth.npz", folder);
    SlabLibrary lib = {
        .package = package,
        .scroll = scroll,
        .folder = folder,
        .out = fmt(a, "%s/synthetic/library/%s", OUTPUTS, scroll),
        .low = INFINITY,
        .high = -INFINITY,
    };
    lib.chunks = env("HERCULEX_CHUNKS", fmt(a, "%s/cache/chunks", OUTPUTS));
    lib.scan = slab_scan(scroll);
    lib.id = fmt(a, "%.14s", lib.scan);
    slab_find_cache(&lib);
    lib.points = npz_get(a, truth, "points");
    Array zones = npz_get(a, truth, "zone"), voxel = npz_get(a, truth, "voxel_um");
    need(lib.points.ndim == 2 && lib.points.shape[1] == 3 && lib.points.shape[0] > 0 &&
            !strcmp(lib.points.descr, "<f4") &&
            lib.points.size == (size_t)elements(lib.points) * 4 && zones.ndim == 1 &&
            zones.shape[0] == lib.points.shape[0] && !strcmp(zones.descr, "|u1") &&
            zones.size == (size_t)zones.shape[0] && !strcmp(voxel.descr, "<f8") &&
            voxel.size == sizeof(double),
        "%s: invalid slab truth arrays", truth);
    memcpy(&lib.voxel, voxel.data, sizeof(lib.voxel));
    const f32 *p = lib.points.data;
    for (i64 i = 0; i < lib.points.shape[0]; i++) {
        if (((const u8 *)zones.data)[i] == 5) {
            lib.low = fmin(lib.low, p[3 * i + 2] - 256.0);
            lib.high = fmax(lib.high, p[3 * i + 2] + 256.0);
        }
    }
    mkdirs(lib.out);
    return lib;
}

static u8 *slab_read(Arena *a, SlabLibrary *lib, const i64 *box, i64 *dims) {
    size_t count = 1;
    for (int i = 0; i < 3; i++) {
        dims[i] = box[2 * i + 1] - box[2 * i];
        need(dims[i] > 0 && (u64)dims[i] <= SIZE_MAX / count, "invalid slab CT box");
        count *= dims[i];
    }
    u8 *ct = push(a, count);
    i64 unknown = 0;
    if (lib->ct.index) {
        need(!hx_cached_ct_read(&lib->ct, box, ct, &unknown), "%s: cannot read cached level-0 CT",
            lib->package);
    }
    if (!lib->ct.index || unknown) {
        hx_cached_ct_close(&lib->ct);
        int status = hx_chunk_fetch_box(lib->scroll, lib->id, 0, box, 0);
        if (status == 130) {
            exit(130);
        }
        need(!status, "%s: cannot fetch CT box [%lld:%lld,%lld:%lld,%lld:%lld] for %s/%s level 0",
            lib->package, (long long)box[0], (long long)box[1], (long long)box[2],
            (long long)box[3], (long long)box[4], (long long)box[5], lib->scroll, lib->scan);
        slab_find_cache(lib);
        need(lib->ct.index && !hx_cached_ct_read(&lib->ct, box, ct, &unknown) && !unknown,
            "%s: fetched CT box is incomplete in the level-0 cache", lib->package);
    }
    return ct;
}

static int slab_order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double slab_quantile(const double *sorted, size_t n, double q) {
    double at = q * (n - 1);
    size_t i = (size_t)floor(at), j = (size_t)ceil(at);
    double t = at - i, d = sorted[j] - sorted[i];
    return t >= .5 ? sorted[j] - d * (1 - t) : sorted[i] + d * t;
}

static int slab_show(const char *path, const char *out) {
    Arena a = arena(GB);
    Array ct = npz_get(&a, path, "ct");
    need(ct.ndim == 3 && !strcmp(ct.descr, "|u1") && ct.shape[0] > 0 && ct.shape[1] > 0 &&
            ct.shape[2] > 0 && ct.shape[2] < INT_MAX / 6 && ct.shape[1] < INT_MAX &&
            ct.shape[0] < INT_MAX - 8 - ct.shape[2] * 6 && ct.size == (size_t)elements(ct),
        "%s: expected a uint8 CT slab", path);
    int h = (int)ct.shape[0], w = (int)ct.shape[1], d = (int)ct.shape[2], rows = h + 8 + 6 * d;
    u8 *image = pushz(&a, (size_t)rows * w);
    const u8 *p = ct.data;
    for (size_t i = 0; i < (size_t)h * w; i++) {
        image[i] = p[i * d + d / 2];
    }
    for (int k = 0; k < d; k++) {
        for (int x = 0; x < w; x++) {
            u8 value = p[((size_t)(h / 2) * w + x) * d + k];
            for (int r = 0; r < 6; r++) {
                image[((size_t)h + 8 + 6 * k + r) * w + x] = value;
            }
        }
    }
    size_t histogram[256] = {0}, n = 0;
    for (size_t i = 0; i < (size_t)rows * w; i++) {
        histogram[image[i]]++;
    }
    double *values = push(&a, (size_t)rows * w * sizeof(*values));
    for (int k = 1; k < 256; k++) {
        for (size_t i = 0; i < histogram[k]; i++) {
            values[n++] = k;
        }
    }
    need(n > 0, "%s: the slab preview has no nonzero CT", path);
    double low = slab_quantile(values, n, .01), high = slab_quantile(values, n, .995);
    for (size_t i = 0; i < (size_t)rows * w; i++) {
        image[i] = high > low ? (u8)fmax(0, fmin(255, (image[i] - low) / (high - low) * 255)) : 0;
    }
    png_write(out, image, w, rows);
    munmap(a.base, a.size);
    return 0;
}

static void slab_air(Arena *a, SlabLibrary *lib, int count, Pcg64 *rng) {
    const int width = 2 * SLAB_AIR_REACH, half = SLAB_AIR_SIDE / 2;
    const size_t pixels = (size_t)width * width,
                 cube = SLAB_AIR_SIDE * SLAB_AIR_SIDE * SLAB_AIR_SIDE;
    const f32 *p = lib->points.data;
    i64 *eligible = push(a, lib->points.shape[0] * sizeof(*eligible));
    i64 n = 0;
    for (i64 i = 0; i < lib->points.shape[0]; i++) {
        if (p[i * 3 + 2] < lib->low || p[i * 3 + 2] > lib->high) {
            eligible[n++] = i;
        }
    }
    need(n > 0 && n <= UINT32_MAX, "%s: no supported training-point range", lib->package);
    u8 *blocks = push(a, (size_t)count * cube);
    int *summed = push(a, (size_t)(width + 1) * (width + 1) * sizeof(*summed));
    int *spots = push(a, pixels * sizeof(*spots));
    size_t mark = a->used;
    int made = 0;
    i64 tries = 0;
    u64 sum = 0;
    while (made < count && tries < 200LL * count) {
        tries++;
        const f32 *v = p + 3 * eligible[draw(rng, n)];
        i64 x = (i64)nearbyint(v[0]), y = (i64)nearbyint(v[1]), z = (i64)nearbyint(v[2]);
        i64 box[] = {z - half, z + half, y - SLAB_AIR_REACH, y + SLAB_AIR_REACH, x - SLAB_AIR_REACH,
            x + SLAB_AIR_REACH},
            dims[3];
        u8 *ct = slab_read(a, lib, box, dims);
        size_t zero = 0;
        memset(summed, 0, (size_t)(width + 1) * sizeof(*summed));
        for (int row = 0; row < width; row++) {
            int bad_row = 0, *above = summed + row * (width + 1), *here = above + width + 1;
            here[0] = 0;
            for (int col = 0; col < width; col++) {
                int bad = 0;
                for (int k = 0; k < SLAB_AIR_SIDE; k++) {
                    u8 value = ct[k * pixels + row * width + col];
                    zero += value == 0;
                    bad |= value >= 70;
                }
                bad_row += bad;
                here[col + 1] = above[col + 1] + bad_row;
            }
        }
        int found = 0;
        if ((double)zero / (pixels * SLAB_AIR_SIDE) <= .001) {
            for (int row = 0; row + SLAB_AIR_SIDE <= width; row++) {
                const int *top = summed + row * (width + 1);
                const int *bottom = top + SLAB_AIR_SIDE * (width + 1);
                for (int col = 0; col + SLAB_AIR_SIDE <= width; col++) {
                    int end = col + SLAB_AIR_SIDE;
                    if (bottom[end] - bottom[col] == top[end] - top[col]) {
                        spots[found++] = row * width + col;
                    }
                }
            }
        }
        if (found) {
            int offset = spots[draw(rng, found)];
            u8 *dst = blocks + (size_t)made++ * cube;
            for (int k = 0; k < SLAB_AIR_SIDE; k++) {
                for (int row = 0; row < SLAB_AIR_SIDE; row++) {
                    const u8 *src = ct + k * pixels + offset + row * width;
                    memcpy(dst, src, SLAB_AIR_SIDE);
                    for (int col = 0; col < SLAB_AIR_SIDE; col++) {
                        sum += *dst++;
                    }
                }
            }
        }
        a->used = mark;
    }
    need(made, "%s: no between-sheet air cubes found", lib->package);
    double level = sum / (double)(made * cube);
    const char *names[] = {"blocks", "level"};
    Array arrays[] = {
        {.descr = "|u1",
            .ndim = 4,
            .shape = {made, SLAB_AIR_SIDE, SLAB_AIR_SIDE, SLAB_AIR_SIDE},
            .data = blocks,
            .size = made * cube},
        {.descr = "<f8", .data = &level, .size = sizeof(level)},
    };
    npz_write_codec(fmt(a, "%s/air.npz", lib->out), 2, names, arrays, ZIP_CM_DEFLATE);
    json_object *summary = json_object_new_object();
    json_object_object_add(summary, "package", json_object_new_string(lib->package));
    json_object_object_add(summary, "blocks", json_object_new_int(made));
    json_object_object_add(summary, "tries", json_object_new_int64(tries));
    json_object_object_add(summary, "level", jnum(nearbyint(level * 100) / 100));
    puts(json_object_to_json_string(summary));
    json_object_put(summary);
}

typedef struct {
    const char *name;
    f32 *grid[4];
    int w, h;
    double step;
} SlabMesh;

typedef struct {
    double v[3];
} SlabPoint;

typedef struct {
    i64 dims[3], stride[3];
    double *data;
} SlabSpline;

typedef struct {
    i64 index[4];
    double weight[4];
} SlabTaps;

static void slab_prefilter_line(double *line, i64 n, i64 stride) {
    if (n == 1) {
        return;
    }
    const double pole = -0.267949192431122706472553658494127633;
    const double gain = (1 - pole) * (1 - 1 / pole);
    i64 last = (n - 1) * stride;
    for (i64 i = 0; i <= last; i += stride) {
        line[i] *= gain;
    }
    double pn = pow(pole, n), power = pole, start = line[0];
    double first = start + pn * line[last];
    for (i64 i = stride; i < last; i += stride, power *= pole) {
        first += power * (line[i] + pn * line[last - i]);
    }
    first += power * (line[last] + pn * first);
    line[0] = first * (pole / (1 - pn * pn)) + start;
    for (i64 i = stride; i <= last; i += stride) {
        line[i] += pole * line[i - stride];
    }
    line[last] *= pole / (pole - 1);
    for (i64 i = last - stride; i >= 0; i -= stride) {
        line[i] = pole * (line[i + stride] - line[i]);
    }
}

static void slab_prefilter(SlabSpline spline) {
    i64 total = spline.dims[0] * spline.dims[1] * spline.dims[2];
    for (int axis = 0; axis < 3; axis++) {
        i64 n = spline.dims[axis], stride = spline.stride[axis];
        for (i64 block = 0; block < total; block += n * stride) {
            for (i64 inner = 0; inner < stride; inner++) {
                slab_prefilter_line(spline.data + block + inner, n, stride);
            }
        }
    }
}

static SlabSpline slab_spline(Arena *a, i64 z, i64 y, i64 x) {
    need(z > 0 && y > 0 && x > 0 && (u64)x <= SIZE_MAX / 8 / y / z, "invalid spline shape");
    return (SlabSpline){
        .dims = {z, y, x},
        .stride = {x * y, x, 1},
        .data = push(a, (size_t)z * y * x * sizeof(double)),
    };
}

static i64 slab_nearest(i64 i, i64 n) { return i < 0 ? 0 : i >= n ? n - 1 : i; }

static SlabTaps slab_taps(double v, i64 n, i64 stride) {
    double at = floor(v), x = v - at, y = 1 - x;
    SlabTaps taps;
    taps.weight[0] = y * y * y / 6;
    taps.weight[1] = (x * x * (x - 2) * 3 + 4) / 6;
    taps.weight[2] = (y * y * (y - 2) * 3 + 4) / 6;
    taps.weight[3] = 1;
    for (int i = 0; i < 3; i++) {
        taps.weight[3] -= taps.weight[i];
    }
    for (int i = 0; i < 4; i++) {
        taps.index[i] = slab_nearest((i64)at - 1 + i, n) * stride;
    }
    return taps;
}

static double slab_at(const SlabSpline *s, double z, double y, double x) {
    SlabTaps tz = slab_taps(z, s->dims[0], s->stride[0]);
    SlabTaps ty = slab_taps(y, s->dims[1], s->stride[1]);
    SlabTaps tx = slab_taps(x, s->dims[2], s->stride[2]);
    double total = 0;
    for (int k = 0; k < 4; k++) {
        for (int j = 0; j < 4; j++) {
            const double *row = s->data + tz.index[k] + ty.index[j];
            for (int i = 0; i < 4; i++) {
                total += ((row[tx.index[i]] * tz.weight[k]) * ty.weight[j]) * tx.weight[i];
            }
        }
    }
    return total;
}

static void slab_surface(Arena *a, const SlabMesh *mesh, int r, int c, SlabPoint *p) {
    size_t saved = a->used;
    int pad = 12, width = (int)(SLAB_SIDE / mesh->step + 3) + 2;
    for (int axis = 0; axis < 3; axis++) {
        int w = width + 2 * pad;
        SlabSpline s = slab_spline(a, 1, w, w);
        for (int y = 0; y < w; y++) {
            int sy = r - 1 + slab_nearest(y - pad, width);
            for (int x = 0; x < w; x++) {
                int sx = c - 1 + slab_nearest(x - pad, width);
                s.data[(size_t)y * w + x] = mesh->grid[axis][(size_t)sy * mesh->w + sx];
            }
        }
        slab_prefilter(s);
        const double *from = s.data;
        for (int y = 0; y < SLAB_SIDE; y++) {
            SlabTaps ty = slab_taps((r + y / mesh->step) - (r - 1) + pad, w, w);
            for (int x = 0; x < SLAB_SIDE; x++) {
                SlabTaps tx = slab_taps((c + x / mesh->step) - (c - 1) + pad, w, 1);
                double value = 0;
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        value += from[ty.index[j] + tx.index[i]] * ty.weight[j] * tx.weight[i];
                    }
                }
                p[(size_t)y * SLAB_SIDE + x].v[axis] = value;
            }
        }
        a->used = saved;
    }
}

static SlabPoint slab_axis(const SlabPoint *knots, int count, double z) {
    if (z <= knots[0].v[0]) {
        return knots[0];
    }
    if (z >= knots[count - 1].v[0]) {
        return knots[count - 1];
    }
    int low = 0, high = count - 1;
    while (high - low > 1) {
        int mid = low + (high - low) / 2;
        if (knots[mid].v[0] <= z) {
            low = mid;
        } else {
            high = mid;
        }
    }
    SlabPoint p = {{z, 0, 0}};
    for (int i = 1; i < 3; i++) {
        double slope = (knots[high].v[i] - knots[low].v[i]) / (knots[high].v[0] - knots[low].v[0]);
        p.v[i] = knots[low].v[i] + slope * (z - knots[low].v[0]);
    }
    return p;
}

static void slab_normals(SlabPoint *p, SlabPoint *normal, const SlabPoint *axis, int knots) {
    double direction = 0, height = 0;
    for (int y = 0; y < SLAB_SIDE; y++) {
        for (int x = 0; x < SLAB_SIDE; x++) {
            size_t at = (size_t)y * SLAB_SIDE + x;
            int l = x > 0 ? x - 1 : x, r = x + 1 < SLAB_SIDE ? x + 1 : x;
            int t = y > 0 ? y - 1 : y, b = y + 1 < SLAB_SIDE ? y + 1 : y;
            SlabPoint du, dv;
            for (int i = 0; i < 3; i++) {
                du.v[i] = (p[(size_t)y * SLAB_SIDE + r].v[i] - p[(size_t)y * SLAB_SIDE + l].v[i]) /
                    (r - l);
                dv.v[i] = (p[(size_t)b * SLAB_SIDE + x].v[i] - p[(size_t)t * SLAB_SIDE + x].v[i]) /
                    (b - t);
            }
            double nx = du.v[1] * dv.v[2] - du.v[2] * dv.v[1];
            double ny = du.v[2] * dv.v[0] - du.v[0] * dv.v[2];
            double nz = du.v[0] * dv.v[1] - du.v[1] * dv.v[0];
            double norm = fmax(1e-9, sqrt(nx * nx + ny * ny + nz * nz));
            normal[at] = (SlabPoint){{nx / norm, ny / norm, nz / norm}};
            SlabPoint centre = slab_axis(axis, knots, p[at].v[2]);
            direction += normal[at].v[0] * (centre.v[1] - p[at].v[0]) +
                normal[at].v[1] * (centre.v[2] - p[at].v[1]);
            height += dv.v[2];
        }
    }
    if (direction < 0) {
        for (size_t i = 0; i < (size_t)SLAB_SIDE * SLAB_SIDE; i++) {
            for (int axis = 0; axis < 3; axis++) {
                normal[i].v[axis] = -normal[i].v[axis];
            }
        }
    }
    if (height < 0) {
        for (int y = 0; y < SLAB_SIDE / 2; y++) {
            for (int x = 0; x < SLAB_SIDE; x++) {
                size_t lo = (size_t)y * SLAB_SIDE + x,
                       hi = (size_t)(SLAB_SIDE - 1 - y) * SLAB_SIDE + x;
                SlabPoint point = p[lo], n = normal[lo];
                p[lo] = p[hi];
                normal[lo] = normal[hi];
                p[hi] = point;
                normal[hi] = n;
            }
        }
    }
}

static int slab_sample(Arena *a, SlabLibrary *lib, const SlabPoint *p, const SlabPoint *normal,
    float *profile, double *z_range) {
    const int depth = SLAB_DEPTH + SLAB_SNAP;
    const size_t pixels = (size_t)SLAB_SIDE * SLAB_SIDE;
    const size_t mark = a->used;
    double low[] = {INFINITY, INFINITY, INFINITY}, high[] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < pixels; i++) {
        for (int axis = 0; axis < 3; axis++) {
            low[axis] = fmin(low[axis], p[i].v[axis]);
            high[axis] = fmax(high[axis], p[i].v[axis]);
        }
    }
    z_range[0] = low[2];
    z_range[1] = high[2];
    i64 box[6], dims[3];
    for (int k = 0; k < 3; k++) {
        box[2 * k] = (i64)fmax(0, floor(low[2 - k] - depth - 2));
        box[2 * k + 1] = (i64)ceil(high[2 - k] + depth + 3);
    }
    u8 *ct = slab_read(a, lib, box, dims);
    SlabSpline spline = slab_spline(a, dims[0], dims[1], dims[2]);
    for (i64 i = 0; i < dims[0] * dims[1] * dims[2]; i++) {
        spline.data[i] = ct[i];
    }
    slab_prefilter(spline);
    size_t zero = 0;
    for (size_t i = 0; i < pixels; i++) {
        for (int d = -depth; d <= depth; d++) {
            double x = p[i].v[0] + d * normal[i].v[0] - box[4];
            double y = p[i].v[1] + d * normal[i].v[1] - box[2];
            double z = p[i].v[2] + d * normal[i].v[2] - box[0];
            f32 value = (f32)slab_at(&spline, z, y, x);
            profile[i * (2 * depth + 1) + d + depth] = value;
            zero += value == 0;
        }
    }
    a->used = mark;
    return (double)zero / (pixels * (2 * depth + 1)) <= .01;
}

static void slab_smooth_shift(double *shift, double *spare) {
    double weights[13], sum = 0;
    const int side = SLAB_SIDE;
    for (int i = -12; i <= 12; i++) {
        double w = exp(-.5 / 9.0 * i * i);
        sum += w;
        weights[abs(i)] = w;
    }
    for (int i = 0; i <= 12; i++) {
        weights[i] /= sum;
    }
    for (int axis = 0; axis < 2; axis++) {
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                const double *from = axis ? spare : shift;
                double *to = axis ? shift : spare;
                int at = axis ? x : y, stride = axis ? 1 : side;
                const double *line = from + (axis ? y * side : x);
                double total = weights[0] * line[at * stride];
                for (int i = 12; i > 0; i--) {
                    total += weights[i] *
                        (line[reflected(at - i, side) * stride] +
                            line[reflected(at + i, side) * stride]);
                }
                to[y * side + x] = total;
            }
        }
    }
}

static float *slab_centre(Arena *a, const float *profile, double *shift) {
    const i64 pixels = (i64)SLAB_SIDE * SLAB_SIDE;
    enum { D = 2 * (SLAB_DEPTH + SLAB_SNAP) + 1, MID = D / 2, OUT = 2 * SLAB_DEPTH + 1 };
    float *slab = push(a, pixels * OUT * sizeof(*slab));
    size_t mark = a->used;
    float *smooth = push(a, pixels * D * sizeof(*smooth));
    float *spare = push(a, pixels * D * sizeof(*spare));
    const i64 dims[] = {SLAB_SIDE, SLAB_SIDE, D};
    memcpy(smooth, profile, pixels * D * sizeof(*smooth));
    gaussian(smooth, spare, dims, 3, (double[]){3, 3, 2.5});
    for (i64 i = 0; i < pixels; i++) {
        const float *scan = smooth + i * D + MID - SLAB_SNAP;
        int k = 0;
        for (int j = 1; j <= 2 * SLAB_SNAP; j++) {
            if (scan[j] > scan[k]) {
                k = j;
            }
        }
        k = k < 1 ? 1 : k > 2 * SLAB_SNAP - 1 ? 2 * SLAB_SNAP - 1 : k;
        float bend = scan[k - 1] - 2 * scan[k] + scan[k + 1];
        float pos = bend < 0 ? .5f * (scan[k - 1] - scan[k + 1]) / fminf(bend, -1e-6f) : 0;
        shift[i] = k + (double)fminf(fmaxf(pos, -.5f), .5f) - SLAB_SNAP;
    }
    a->used = mark;
    double *buf = push(a, pixels * sizeof(*buf));
    slab_smooth_shift(shift, buf);
    a->used = mark;
    SlabSpline spline = slab_spline(a, SLAB_SIDE, SLAB_SIDE, D);
    for (i64 i = 0; i < pixels * D; i++) {
        spline.data[i] = profile[i];
    }
    slab_prefilter(spline);
    for (int y = 0; y < SLAB_SIDE; y++) {
        for (int x = 0; x < SLAB_SIDE; x++) {
            i64 pos = (i64)y * SLAB_SIDE + x;
            for (int i = 0; i < OUT; i++) {
                slab[pos * OUT + i] =
                    (float)slab_at(&spline, y, x, MID + i - SLAB_DEPTH + shift[pos]);
            }
        }
    }
    a->used = mark;
    return slab;
}

static void slab_gaps(Arena *a, const float *slab, _Float16 *in, _Float16 *out, double *median) {
    enum { DEPTHS = SLAB_DEPTH * 2 + 1, SMALL = SLAB_SIDE / SLAB_GAP_STEP };
    size_t mark = a->used;
    i64 count = (i64)SLAB_SIDE * SLAB_SIDE * DEPTHS;
    float *smooth = push(a, count * sizeof(*smooth));
    float *spare = push(a, count * sizeof(*spare));
    memcpy(smooth, slab, count * sizeof(*smooth));
    gaussian(smooth, spare, (i64[]){SLAB_SIDE, SLAB_SIDE, DEPTHS}, 3, (double[]){2, 2, 1});
    int hist[2][SLAB_DEPTH + 1] = {{0}};
    for (int y = 0; y < SMALL; y++) {
        for (int x = 0; x < SMALL; x++) {
            int at = y * SMALL + x;
            const float *column =
                smooth + ((size_t)y * SLAB_SIDE + x) * SLAB_GAP_STEP * DEPTHS + SLAB_DEPTH;
            for (int k = 0; k < 2; k++) {
                int gap = SLAB_DEPTH, valley = 0, sign = k ? -1 : 1;
                float low = column[0];
                for (int d = 1; d <= SLAB_DEPTH; d++) {
                    float value = column[d * sign];
                    if (value < low) {
                        low = value;
                        valley = d;
                    }
                    if (d >= 3 && value - low > 4) {
                        gap = valley >= 3 ? valley : 3;
                        break;
                    }
                }
                (k ? out : in)[at] = (_Float16)gap;
                hist[k][gap]++;
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        int n = 0, middle = SMALL * SMALL / 2;
        median[i] = 0;
        for (int gap = 0; gap <= SLAB_DEPTH; gap++) {
            int end = n + hist[i][gap];
            if (n <= middle - 1 && end > middle - 1) {
                median[i] += gap * .5;
            }
            if (n <= middle && end > middle) {
                median[i] += gap * .5;
                break;
            }
            n = end;
        }
    }
    a->used = mark;
}

static int slab_mesh_order(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static SlabMesh *slab_meshes(Arena *a, const SlabLibrary *lib, int *count) {
    glob_t g = {0};
    int code = glob(fmt(a, "%s/meshes/*.tifxyz", lib->folder), 0, NULL, &g);
    need(!code && g.gl_pathc > 0 && g.gl_pathc <= INT_MAX, "%s: no truth meshes", lib->package);
    qsort(g.gl_pathv, g.gl_pathc, sizeof(*g.gl_pathv), slab_mesh_order);
    SlabMesh *meshes = pushz(a, g.gl_pathc * sizeof(*meshes));
    for (size_t i = 0; i < g.gl_pathc; i++) {
        const char *path = g.gl_pathv[i];
        SlabMesh *m = meshes + i;
        m->name = fmt(a, "%s", strrchr(path, '/') + 1);
        json_object *meta = json_read(fmt(a, "%s/meta.json", path));
        double scale = json_object_get_double(json_object_array_get_idx(jget(meta, "scale"), 0));
        need(scale > 0 && isfinite(scale) && scale * SLAB_SIDE <= INT_MAX - 8,
            "%s: invalid truth mesh step", path);
        m->step = 1.0 / scale;
        json_object_put(meta);
        for (int k = 0; k < 4; k++) {
            int w, h;
            m->grid[k] = tiff_read(a, fmt(a, "%s/%c.tif", path, "xyzw"[k]), &w, &h);
            need(!k || (w == m->w && h == m->h), "%s: inconsistent truth grids", path);
            m->w = w;
            m->h = h;
        }
    }
    *count = (int)g.gl_pathc;
    globfree(&g);
    return meshes;
}

static SlabPoint *slab_axis_read(Arena *a, const SlabLibrary *lib, int *count) {
    json_object *axis = json_read(fmt(a, "%s/axis.json", lib->folder));
    json_object *knots = jget(axis, "knots_z_x_y");
    size_t n = json_object_array_length(knots);
    need(n > 0 && n <= INT_MAX, "%s: invalid scroll axis", lib->package);
    SlabPoint *out = push(a, n * sizeof(*out));
    for (size_t i = 0; i < n; i++) {
        json_object *point = json_object_array_get_idx(knots, i);
        need(json_object_array_length(point) == 3, "%s: invalid axis knot", lib->package);
        for (int k = 0; k < 3; k++) {
            out[i].v[k] = json_object_get_double(json_object_array_get_idx(point, k));
        }
    }
    json_object_put(axis);
    *count = (int)n;
    return out;
}

static int slab_mesh_window(Arena *a, const SlabMesh *mesh, const SlabLibrary *lib, int r, int c,
    int cells, double *winding) {
    size_t mark = a->used;
    double low = INFINITY, high = -INFINITY;
    int side = cells + 2, count = 0;
    double *windings = push(a, (size_t)side * side * sizeof(*windings));
    for (int y = r - 1; y < r + cells + 1; y++) {
        for (int x = c - 1; x < c + cells + 1; x++) {
            size_t i = (size_t)y * mesh->w + x;
            if (!(mesh->grid[0][i] > 0)) {
                a->used = mark;
                return 0;
            }
            low = fmin(low, mesh->grid[2][i]);
            high = fmax(high, mesh->grid[2][i]);
            if (mesh->grid[3][i] > 0) {
                windings[count++] = mesh->grid[3][i];
            }
        }
    }
    qsort(windings, count, sizeof(*windings), slab_order);
    *winding = count ? slab_quantile(windings, count, .5) : NAN;
    a->used = mark;
    return high < lib->low || low > lib->high;
}

static json_object *slab_save(Arena *a, SlabLibrary *lib, const SlabMesh *mesh, int row, int col,
    u64 seed, int index, double winding, const SlabPoint *axis, int knots) {
    enum { D = SLAB_DEPTH * 2 + 1, SMALL = SLAB_SIDE / SLAB_GAP_STEP };
    const size_t pixels = (size_t)SLAB_SIDE * SLAB_SIDE, reduced = (size_t)SMALL * SMALL;
    SlabPoint *p = push(a, pixels * sizeof(*p)), *normals = push(a, pixels * sizeof(*normals));
    float *profile = push(a, pixels * (D + 2 * SLAB_SNAP) * sizeof(*profile));
    slab_surface(a, mesh, row, col, p);
    slab_normals(p, normals, axis, knots);
    double z[2];
    if (!slab_sample(a, lib, p, normals, profile, z)) {
        return NULL;
    }
    double *shift = push(a, pixels * sizeof(*shift));
    float *slab = slab_centre(a, profile, shift);
    u8 *raw = push(a, pixels * D);
    for (size_t i = 0; i < pixels * D; i++) {
        raw[i] = (u8)fminf(255, fmaxf(0, nearbyintf(slab[i])));
    }
    _Float16 *in = push(a, reduced * sizeof(*in)), *out = push(a, reduced * sizeof(*out));
    double med[2];
    slab_gaps(a, slab, in, out, med);
    _Float16 *ridge = push(a, pixels * sizeof(*ridge));
    f32 *points = push(a, reduced * 3 * sizeof(*points));
    f32 *normal = push(a, reduced * 3 * sizeof(*normal));
    for (size_t i = 0; i < pixels; i++) {
        ridge[i] = (_Float16)shift[i];
    }
    for (int y = 0; y < SMALL; y++) {
        for (int x = 0; x < SMALL; x++) {
            size_t from = ((size_t)y * SLAB_SIDE + x) * SLAB_GAP_STEP, to = (size_t)y * SMALL + x;
            for (int k = 0; k < 3; k++) {
                points[to * 3 + k] = (f32)p[from].v[k];
                normal[to * 3 + k] = (f32)normals[from].v[k];
            }
        }
    }
    const char *name = fmt(a, "slab-%03llu-%03d", (unsigned long long)seed, index);
    const char *names[] = {"ct", "half_in", "half_out", "points", "normals", "ridge"};
    Array arrays[] = {
        {.descr = "|u1",
            .ndim = 3,
            .shape = {SLAB_SIDE, SLAB_SIDE, D},
            .data = raw,
            .size = pixels * D},
        {.descr = "<f2", .ndim = 2, .shape = {SMALL, SMALL}, .data = in, .size = reduced * 2},
        {.descr = "<f2", .ndim = 2, .shape = {SMALL, SMALL}, .data = out, .size = reduced * 2},
        {.descr = "<f4",
            .ndim = 3,
            .shape = {SMALL, SMALL, 3},
            .data = points,
            .size = reduced * 12},
        {.descr = "<f4",
            .ndim = 3,
            .shape = {SMALL, SMALL, 3},
            .data = normal,
            .size = reduced * 12},
        {.descr = "<f2",
            .ndim = 2,
            .shape = {SLAB_SIDE, SLAB_SIDE},
            .data = ridge,
            .size = pixels * 2},
    };
    npz_write_codec(fmt(a, "%s/%s.npz", lib->out, name), 6, names, arrays, ZIP_CM_DEFLATE);
    json_object *entry = json_object_new_object(), *heights = json_object_new_array();
    json_object *gaps = json_object_new_array();
    json_object_object_add(entry, "slab", json_object_new_string(name));
    json_object_object_add(entry, "package", json_object_new_string(lib->package));
    json_object_object_add(entry, "mesh", json_object_new_string(mesh->name));
    json_object_object_add(entry, "row", json_object_new_int(row));
    json_object_object_add(entry, "col", json_object_new_int(col));
    json_object_object_add(entry, "winding",
        isfinite(winding) ? jnum(nearbyint(winding * 1000) / 1000) : json_object_new_double(NAN));
    for (int k = 0; k < 2; k++) {
        json_object_array_add(heights, jnum(nearbyint(z[k] * 10) / 10));
        json_object_array_add(gaps, jnum(med[k]));
    }
    json_object_object_add(entry, "z", heights);
    json_object_object_add(entry, "half_gap_median", gaps);
    return entry;
}

static void slab_build(Arena *a, SlabLibrary *lib, int count, u64 seed, Pcg64 *rng) {
    int meshes, knots;
    SlabMesh *grid = slab_meshes(a, lib, &meshes);
    SlabPoint *axis = slab_axis_read(a, lib, &knots);
    size_t mark = a->used;
    int made = 0;
    time_t start = time(NULL);
    json_object *slabs = json_object_new_array();
    for (i64 tried = 1; made < count && tried <= 40LL * count; tried++) {
        SlabMesh *m = grid + draw(rng, meshes);
        int cells = (int)ceil(SLAB_SIDE / m->step) + 4;
        if (m->w <= cells + 2 || m->h <= cells + 2) {
            continue;
        }
        int row = 1 + (int)draw(rng, m->h - cells - 2);
        int col = 1 + (int)draw(rng, m->w - cells - 2);
        double winding;
        if (!slab_mesh_window(a, m, lib, row, col, cells, &winding)) {
            continue;
        }
        json_object *entry = slab_save(a, lib, m, row, col, seed, made, winding, axis, knots);
        a->used = mark;
        if (entry) {
            json_object_array_add(slabs, entry);
            printf("%s: %d/%d slabs, %lld tries, %lld s\n", lib->package, ++made, count,
                (long long)tried, (long long)(time(NULL) - start));
            fflush(stdout);
        }
    }
    json_object *summary = json_object_new_object();
    json_object_object_add(summary, "package", json_object_new_string(lib->package));
    json_object_object_add(summary, "scroll", json_object_new_string(lib->scroll));
    json_object_object_add(summary, "voxel_um", jnum(lib->voxel));
    json_object_object_add(summary, "size", json_object_new_int(SLAB_SIDE));
    json_object_object_add(summary, "depth", json_object_new_int(SLAB_DEPTH));
    json_object_object_add(summary, "seconds", json_object_new_int64(time(NULL) - start));
    json_object_object_add(summary, "slabs", slabs);
    json_write(
        fmt(a, "%s/library-%s-%03llu.json", lib->out, lib->package, (unsigned long long)seed),
        summary);
    json_object_object_del(summary, "slabs");
    puts(json_object_to_json_string(summary));
    json_object_put(summary);
}

static int slab_library_command(int argc, char **argv) {
    const char *usage = "synthetic.library build PACKAGE [--count N] [--seed N]\n"
                        "synthetic.library air PACKAGE [--count N] [--seed N]\n"
                        "synthetic.library show SLAB.npz OUT.png";
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts(usage);
        return 0;
    }
    if (!strcmp(argv[0], "show")) {
        need(argc == 3, "%s", usage);
        return slab_show(argv[1], argv[2]);
    }
    int air = !strcmp(argv[0], "air"), count = air ? 64 : 32;
    need(argc >= 2 && (air || !strcmp(argv[0], "build")), "%s", usage);
    u64 seed = 0;
    for (int i = 2; i < argc; i++) {
        char split[32];
        const char *value;
        const char *key = slices_cli_option(argv[i], split, &value);
        if (!value) {
            need(i + 1 < argc, "%s needs a value", key);
            value = argv[++i];
        }
        if (!strcmp(key, "--count")) {
            count = slices_positive(value, key);
        } else if (!strcmp(key, "--seed")) {
            seed = slices_unsigned(value, key);
        } else {
            die("unknown slab-library option: %s", key);
        }
    }
    Arena a = arena(4 * GB);
    SlabLibrary lib = slab_open(&a, argv[1], NULL);
    Pcg64 rng = pcg64(seed);
    if (air) {
        slab_air(&a, &lib, count, &rng);
    } else {
        slab_build(&a, &lib, count, seed, &rng);
    }
    hx_cached_ct_close(&lib.ct);
    munmap(a.base, a.size);
    return 0;
}
