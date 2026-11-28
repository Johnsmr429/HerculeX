#define _GNU_SOURCE
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tiffio.h>

static TIFF *open_axis(const char *folder, int axis) {
    char *file = NULL;
    if (asprintf(&file, "%s/%c.tif", folder, "xyz"[axis]) < 0)
        return NULL;
    TIFF *t = file ? TIFFOpen(file, "r") : NULL;
    free(file);
    return t;
}

static double sample_value(const void *data, size_t i, uint16_t bits, uint16_t format) {
    if (format == SAMPLEFORMAT_IEEEFP)
        return bits == 32 ? ((const float *)data)[i] : ((const double *)data)[i];
    if (format == SAMPLEFORMAT_INT) {
        if (bits == 8)
            return ((const int8_t *)data)[i];
        if (bits == 16)
            return ((const int16_t *)data)[i];
        if (bits == 32)
            return ((const int32_t *)data)[i];
        return (double)((const int64_t *)data)[i];
    }
    if (bits == 8)
        return ((const uint8_t *)data)[i];
    if (bits == 16)
        return ((const uint16_t *)data)[i];
    if (bits == 32)
        return ((const uint32_t *)data)[i];
    return (double)((const uint64_t *)data)[i];
}

static int read_axis(const char *folder, uint32_t expected_w, uint32_t expected_h, int axis,
    float *out, double *precise, uint8_t *valid, uint16_t *sample_bits, uint16_t *sample_format) {
    TIFF *t = open_axis(folder, axis);
    if (!t)
        return 0;
    uint32_t width = 0, height = 0;
    uint16_t bits = 0, format = 0, samples = 0;
    TIFFGetFieldDefaulted(t, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLEFORMAT, &format);
    TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLESPERPIXEL, &samples);
    int ok = TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &width) &&
        TIFFGetField(t, TIFFTAG_IMAGELENGTH, &height) && width == expected_w &&
        height == expected_h && samples == 1 &&
        ((format == SAMPLEFORMAT_IEEEFP && (bits == 32 || bits == 64)) ||
            ((format == SAMPLEFORMAT_UINT || format == SAMPLEFORMAT_INT) &&
                (bits == 8 || bits == 16 || bits == 32 || bits == 64)));
    int tiled = TIFFIsTiled(t);
    uint32_t bw = width, bh = 1;
    if (tiled)
        ok = ok && TIFFGetField(t, TIFFTAG_TILEWIDTH, &bw) &&
            TIFFGetField(t, TIFFTAG_TILELENGTH, &bh);
    uint64_t bytes = tiled ? TIFFTileSize64(t) : TIFFScanlineSize64(t);
    if (!bw || !bh || bytes > SIZE_MAX || (uint64_t)bh * bw > bytes / (bits >= 8 ? bits / 8 : 1))
        ok = 0;
    void *block = ok ? malloc((size_t)bytes) : NULL;
    if (!block)
        ok = 0;
    for (uint64_t y = 0; ok && y < height; y += bh) {
        for (uint64_t x = 0; ok && x < width; x += bw) {
            int64_t got = tiled ? TIFFReadTile(t, block, (uint32_t)x, (uint32_t)y, 0, 0)
                                : TIFFReadScanline(t, block, (uint32_t)y, 0);
            if (got < 0 || (tiled && (uint64_t)got < bytes)) {
                ok = 0;
                break;
            }
            uint32_t rows = (uint32_t)fmin(height - y, bh);
            uint32_t cols = (uint32_t)fmin(width - x, bw);
            for (uint32_t r = 0; r < rows; r++) {
                size_t at = (size_t)(y + r) * width + x;
                for (uint32_t c = 0; c < cols; c++) {
                    double value = sample_value(block, (size_t)r * bw + c, bits, format);
                    if (out)
                        out[at + c] = (float)value;
                    if (precise)
                        precise[at + c] = value;
                    if (valid)
                        valid[at + c] = value >= 0;
                }
            }
        }
    }
    if (!ok)
        fprintf(stderr, "cannot read %c coordinates in %s\n", "xyz"[axis], folder);
    free(block);
    if (sample_bits)
        *sample_bits = bits;
    if (sample_format)
        *sample_format = format;
    TIFFClose(t);
    return ok;
}

static double *read_raw_mesh(const char *folder, uint32_t *w, uint32_t *h, int *type) {
    TIFF *file = open_axis(folder, 0);
    if (!file)
        return NULL;
    int valid = TIFFGetField(file, TIFFTAG_IMAGEWIDTH, w) &&
        TIFFGetField(file, TIFFTAG_IMAGELENGTH, h) && *w && *h;
    TIFFClose(file);
    if (!valid || *w > INT_MAX || *h > INT_MAX ||
        (uint64_t)*w * *h > SIZE_MAX / (3 * sizeof(double)))
        return NULL;
    size_t count = (size_t)*w * *h;
    double *mesh = malloc(3 * count * sizeof(*mesh));
    if (!mesh)
        return NULL;
    *type = 32;
    for (int axis = 0; axis < 3; axis++) {
        uint16_t bits = 0, format = 0;
        if (!read_axis(folder, *w, *h, axis, NULL, mesh + axis * count, NULL, &bits, &format)) {
            free(mesh);
            return NULL;
        }
        if (format != SAMPLEFORMAT_IEEEFP)
            *type = 0;
        else if (*type && bits > *type)
            *type = bits;
    }
    return mesh;
}

#ifdef HX_MESH_BRIDGE
#include "fibre_kernels.c"
#define HX_MESH_RELAX_ONLY
#include "../level/level/sheet_flattening.c"

enum { BRIDGE = 8, GAUSS_RADIUS = 16, RELAXATIONS = 40 };

typedef struct {
    uint16_t bits, format;
} MeshType;

static MeshType common_type(MeshType a, MeshType b) {
    if (a.format == b.format)
        return (MeshType){.bits = a.bits > b.bits ? a.bits : b.bits, .format = a.format};
    if (a.format == SAMPLEFORMAT_IEEEFP || b.format == SAMPLEFORMAT_IEEEFP) {
        MeshType real = a.format == SAMPLEFORMAT_IEEEFP ? a : b;
        MeshType integer = a.format == SAMPLEFORMAT_IEEEFP ? b : a;
        real.bits = real.bits == 64 || integer.bits > 16 ? 64 : 32;
        return real;
    }
    MeshType signed_type = a.format == SAMPLEFORMAT_INT ? a : b;
    MeshType positive = a.format == SAMPLEFORMAT_UINT ? a : b;
    if (signed_type.bits > positive.bits)
        return signed_type;
    if (positive.bits == 64)
        return (MeshType){.bits = 64, .format = SAMPLEFORMAT_IEEEFP};
    signed_type.bits = 2 * positive.bits;
    return signed_type;
}

static const double bridge_weights[] = {0x1.18ac19141d621p-15, 0x1.71bc883f2eb6cp-14,
    0x1.c98da230e44c3p-13, 0x1.09f5f6e2ac01cp-11, 0x1.2274b595fa0dep-10, 0x1.29fd07d6ae2c8p-9,
    0x1.1f319f5fd1d9ep-8, 0x1.04053ad72a045p-7, 0x1.ba4f4530f2e69p-7, 0x1.616787004c73fp-6,
    0x1.0942ec97f52edp-5, 0x1.76141cc98066cp-5, 0x1.ef92d62bd84abp-5, 0x1.34602a221842cp-4,
    0x1.6887143b624f8p-4, 0x1.8bf65056bb619p-4, 0x1.988806a9e3a2ep-4, 0x1.8bf65056bb619p-4,
    0x1.6887143b624f8p-4, 0x1.34602a221842cp-4, 0x1.ef92d62bd84abp-5, 0x1.76141cc98066cp-5,
    0x1.0942ec97f52edp-5, 0x1.616787004c73fp-6, 0x1.ba4f4530f2e69p-7, 0x1.04053ad72a045p-7,
    0x1.1f319f5fd1d9ep-8, 0x1.29fd07d6ae2c8p-9, 0x1.2274b595fa0dep-10, 0x1.09f5f6e2ac01cp-11,
    0x1.c98da230e44c3p-13, 0x1.71bc883f2eb6cp-14, 0x1.18ac19141d621p-15};

typedef struct {
    int radius, plain;
    const double *weights;
} Blur;

static const Blur default_blur = {
    .radius = GAUSS_RADIUS, .plain = GAUSS_RADIUS, .weights = bridge_weights};

static int gaussian_mesh(const float *in, float *temp, float *out, int h, int w, const Blur *blur) {
    long dims[] = {h, w};
    return !gaussian_axis(in, temp, dims, 2, 0, blur->weights, blur->radius, blur->plain) &&
        !gaussian_axis(temp, out, dims, 2, 1, blur->weights, blur->radius, blur->plain);
}

static void precise_axis(
    const double *in, double *out, int h, int w, int horizontal, MeshType type, const Blur *blur) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t at = (size_t)y * w + x;
            double sum = in[at] * blur->weights[blur->radius];
            for (int j = blur->radius, tap = 0; j > 0; j--, tap++) {
                size_t a = horizontal ? (size_t)y * w + reflected(x - j, w)
                                      : (size_t)reflected(y - j, h) * w + x;
                size_t b = horizontal ? (size_t)y * w + reflected(x + j, w)
                                      : (size_t)reflected(y + j, h) * w + x;
                double pair = in[a] + in[b];
                sum = tap < blur->plain ? sum + pair * blur->weights[tap]
                                        : fma(pair, blur->weights[tap], sum);
            }
            out[at] = type.format == SAMPLEFORMAT_IEEEFP ? sum : trunc(sum);
        }
    }
}

static void integral(const uint8_t *pixels, int h, int w, uint64_t *sum) {
    size_t pitch = (size_t)w + 1;
    memset(sum, 0, pitch * sizeof(*sum));
    for (int y = 0; y < h; y++) {
        size_t row = (size_t)y * pitch;
        uint64_t across = 0;
        sum[row + pitch] = 0;
        for (int x = 0; x < w; x++) {
            across += pixels[(size_t)y * w + x];
            sum[row + pitch + x + 1] = sum[row + x + 1] + across;
        }
    }
}

static uint64_t rectangle(
    const uint64_t *sum, size_t pitch, size_t y, size_t x, size_t bottom, size_t right) {
    return sum[bottom * pitch + right] - sum[y * pitch + right] - sum[bottom * pitch + x] +
        sum[y * pitch + x];
}

static void visit(uint8_t *mask, size_t *queue, size_t *tail, size_t i) {
    if (mask[i] == 0) {
        mask[i] = 2;
        queue[(*tail)++] = i;
    }
}

static int gap_mask(const uint8_t *good, int h, int w, uint8_t *gap, size_t *count, int reach) {
    int ph = h + 2 * reach, pw = w + 2 * reach;
    size_t n = (size_t)h * w, expanded = (size_t)ph * pw;
    uint64_t *sum = calloc(((size_t)ph + 1) * ((size_t)pw + 1), sizeof(*sum));
    uint8_t *grown = calloc(expanded, sizeof(*grown));
    size_t *queue = malloc(n * sizeof(*queue));
    if (!sum || !grown || !queue) {
        free(sum);
        free(grown);
        free(queue);
        return 0;
    }
    integral(good, h, w, sum);
    for (int y = 0; y < ph; y++) {
        int top = y > 2 * reach ? y - 2 * reach : 0;
        int end = y + 1 < h ? y + 1 : h;
        for (int x = 0; x < pw; x++) {
            int left = x > 2 * reach ? x - 2 * reach : 0;
            int right = x + 1 < w ? x + 1 : w;
            grown[(size_t)y * pw + x] = rectangle(sum, (size_t)w + 1, top, left, end, right) != 0;
        }
    }
    integral(grown, ph, pw, sum);
    free(grown);
    int side = 2 * reach + 1;
    size_t tail = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            gap[i] =
                rectangle(sum, (size_t)pw + 1, y, x, y + side, x + side) == (unsigned)(side * side);
            if (!y || !x || y == h - 1 || x == w - 1)
                visit(gap, queue, &tail, i);
        }
    }
    free(sum);
    for (size_t head = 0; head < tail; head++) {
        size_t i = queue[head];
        if (i >= (size_t)w)
            visit(gap, queue, &tail, i - w);
        if (i + w < n)
            visit(gap, queue, &tail, i + w);
        if (i % w)
            visit(gap, queue, &tail, i - 1);
        if (i % w + 1 < (unsigned)w)
            visit(gap, queue, &tail, i + 1);
    }
    free(queue);
    *count = 0;
    for (size_t i = 0; i < n; i++) {
        gap[i] = gap[i] != 2 && !good[i];
        *count += gap[i];
    }
    return 1;
}

static int reject_folds(float *out, int h, int w, double stride, const uint8_t *good, uint8_t *gap,
    size_t count, int rounds, double fold_cells) {
    size_t n = (size_t)h * w;
    if (hx_relax_mesh(out, 1, 3, h, w, NULL, gap, (int64_t)count, rounds))
        return 0;
    uint8_t *fold = calloc(n, sizeof(*fold));
    if (!fold)
        return 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            if (!(good[i] || gap[i]))
                continue;
            for (int axis = 0; axis < 2; axis++) {
                if ((axis == 0 && y == h - 1) || (axis == 1 && x == w - 1))
                    continue;
                size_t j = i + (axis == 0 ? w : 1);
                if (!(good[j] || gap[j]))
                    continue;
                float total = 0;
                for (int a = 0; a < 3; a++) {
                    float d = out[a * n + j] - out[a * n + i];
                    total += d * d;
                }
                if (sqrtf(total) > (float)(fold_cells * stride))
                    fold[i] = fold[j] = 1;
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (!good[i] && (!gap[i] || fold[i])) {
            for (int a = 0; a < 3; a++)
                out[a * n + i] = -1;
        }
    }
    free(fold);
    return 1;
}

static int bridge_grid(const double *xyz, MeshType type, int h, int w, double stride,
    const uint8_t *good, uint8_t *gap, size_t count, float *out, const Blur *blur, int rounds,
    double fold_cells) {
    size_t n = (size_t)h * w;
    float *scratch = malloc(4 * n * sizeof(*scratch));
    int narrow = type.bits == 32 && type.format == SAMPLEFORMAT_IEEEFP;
    double *precise = narrow ? NULL : malloc(2 * n * sizeof(*precise));
    if (!scratch || (!narrow && !precise)) {
        free(scratch);
        free(precise);
        return 0;
    }
    float *temp = scratch + n, *weight = temp + n, *smooth = weight + n;
    for (size_t i = 0; i < n; i++)
        scratch[i] = good[i];
    int ok = gaussian_mesh(scratch, temp, weight, h, w, blur);
    for (size_t i = 0; i < n; i++)
        weight[i] = fmaxf(weight[i], 1e-6f);
    for (int axis = 0; ok && axis < 3; axis++) {
        const double *source = xyz + axis * n;
        float *result = out + axis * n;
        if (narrow) {
            for (size_t i = 0; i < n; i++)
                scratch[i] = good[i] ? (float)source[i] : 0;
            ok = gaussian_mesh(scratch, temp, smooth, h, w, blur);
            if (ok) {
                for (size_t i = 0; i < n; i++)
                    result[i] = good[i] ? (float)source[i] : smooth[i] / weight[i];
            }
        } else {
            for (size_t i = 0; i < n; i++)
                precise[i] = good[i] ? source[i] : 0;
            precise_axis(precise, precise + n, h, w, 0, type, blur);
            precise_axis(precise + n, precise, h, w, 1, type, blur);
            for (size_t i = 0; i < n; i++) {
                double filled =
                    type.bits <= 16 ? (float)precise[i] / weight[i] : precise[i] / weight[i];
                result[i] = good[i] ? (float)source[i] : (float)filled;
            }
        }
    }
    free(scratch);
    free(precise);
    return ok && reject_folds(out, h, w, stride, good, gap, count, rounds, fold_cells);
}

static int smooth_grid(const double *xyz, MeshType type, const uint8_t *good, int h, int w,
    const Blur *blur, float *out) {
    size_t n = (size_t)h * w;
    int narrow = type.bits == 32 && type.format == SAMPLEFORMAT_IEEEFP;
    float *buf = malloc(4 * n * sizeof(*buf));
    double *wide = narrow ? NULL : malloc(2 * n * sizeof(*wide));
    if (!buf || (!narrow && !wide)) {
        free(buf);
        free(wide);
        return 0;
    }
    float *weight = buf + n, *temp = weight + n, *filtered = temp + n;
    for (size_t i = 0; i < n; i++) {
        buf[i] = good[i];
    }
    int ok = gaussian_mesh(buf, temp, weight, h, w, blur);
    for (size_t i = 0; ok && i < n; i++) {
        weight[i] = fmaxf(weight[i], 1e-4f);
    }
    for (int axis = 0; ok && axis < 3; axis++) {
        const double *source = xyz + axis * n;
        float *dest = out + axis * n;
        if (narrow) {
            for (size_t i = 0; i < n; i++) {
                buf[i] = good[i] ? (float)source[i] : 0;
            }
            ok = gaussian_mesh(buf, temp, filtered, h, w, blur);
            for (size_t i = 0; ok && i < n; i++) {
                dest[i] = good[i] ? filtered[i] / weight[i] : -1;
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                wide[i] = good[i] ? source[i] : 0;
            }
            precise_axis(wide, wide + n, h, w, 0, type, blur);
            precise_axis(wide + n, wide, h, w, 1, type, blur);
            for (size_t i = 0; i < n; i++) {
                double v = type.bits <= 16 ? (float)wide[i] / weight[i] : wide[i] / weight[i];
                dest[i] = good[i] ? (float)v : -1;
            }
        }
    }
    free(buf);
    free(wide);
    return ok;
}

static void float_unit_normals(const float *points, int64_t h, int64_t w, float *out) {
    int64_t size = h * w;
    for (int64_t y = 0; y < h; y++) {
        for (int64_t x = 0; x < w; x++) {
            int64_t i = y * w + x, up = y ? i - w : i, down = y + 1 < h ? i + w : i;
            int64_t left = x ? i - 1 : i, right = x + 1 < w ? i + 1 : i;
            float fy = y && y + 1 < h ? .5f : 1, fx = x && x + 1 < w ? .5f : 1;
            float a[3], b[3], n[3], magnitude = 0;
            for (int k = 0; k < 3; k++) {
                const float *p = points + k * size;
                a[k] = (p[down] - p[up]) * fy;
                b[k] = (p[right] - p[left]) * fx;
            }
            for (int k = 0; k < 3; k++) {
                int u = (k + 1) % 3, v = (k + 2) % 3;
                n[k] = a[u] * b[v] - a[v] * b[u];
                magnitude += n[k] * n[k];
            }
            float length = sqrtf(magnitude);
            if (length < 1e-9f) {
                length = 1e-9f;
            }
            for (int k = 0; k < 3; k++) {
                out[k * size + i] = n[k] / length;
            }
        }
    }
}

#ifdef HX_COLUMN_NATIVE
static const double smooth_weights[] = {0x1.7638649a0fcd1p-14, 0x1.0d891ab53680ap-10,
    0x1.f1e741f80c9b8p-8, 0x1.26dd9e7a284e1p-5, 0x1.bfdc88289323cp-4, 0x1.b42851d1c8a96p-3,
    0x1.1058ef29a59d7p-2, 0x1.b42851d1c8a96p-3, 0x1.bfdc88289323cp-4, 0x1.26dd9e7a284e1p-5,
    0x1.f1e741f80c9b8p-8, 0x1.0d891ab53680ap-10, 0x1.7638649a0fcd1p-14};

static int plain_taps(int radius) {
#ifdef __aarch64__
    return radius >= 10 ? radius / 8 * 8 : 0;
#else
    return radius;
#endif
}

static void extend_surface(float *points, float *before, int h, int w) {
    size_t n = (size_t)h * w;
    for (int pass = 0; pass < 2; pass++) {
        memcpy(before, points, 3 * n * sizeof(*points));
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                size_t i = (size_t)y * w + x;
                for (int d = 0; points[i] <= 0 && d < 4; d++) {
                    int step = d % 2 ? 1 : -1;
                    size_t j = d < 2 ? (size_t)((y + step + (int64_t)h) % h) * w + x
                                     : (size_t)y * w + (x + step + (int64_t)w) % w;
                    size_t k = d < 2 ? (size_t)((y + 2 * step + (int64_t)h) % h) * w + x
                                     : (size_t)y * w + (x + 2 * step + (int64_t)w) % w;
                    if (before[j] > 0 && before[k] > 0) {
                        for (int a = 0; a < 3; a++) {
                            points[a * n + i] = 2.f * before[a * n + j] - before[a * n + k];
                        }
                    }
                }
            }
        }
    }
}

static double row_split(const int *nearest, int row, int left, int right) {
    int64_t a = row - nearest[left], b = row - nearest[right];
    return ((double)(right * (int64_t)right - left * (int64_t)left) + b * b - a * a) /
        (2. * (right - left));
}

static void fill_nearest(
    const float *points, float *filled, int *nearest, int h, int w, int *site, double *edge) {
    size_t n = (size_t)h * w;
    for (int x = 0; x < w; x++) {
        int above = -1, below = -1;
        for (int y = 0; y < h; y++) {
            size_t i = (size_t)y * w + x;
            if (points[i] > 0) {
                above = y;
            }
            nearest[i] = above;
        }
        for (int y = h - 1; y >= 0; y--) {
            size_t i = (size_t)y * w + x;
            if (points[i] > 0) {
                below = y;
            }
            if (below >= 0 && (nearest[i] < 0 || below - y < y - nearest[i])) {
                nearest[i] = below;
            }
        }
    }
    for (int y = 0; y < h; y++) {
        const int *row = nearest + (size_t)y * w;
        int count = 0;
        for (int x = 0; x < w; x++) {
            if (row[x] < 0) {
                continue;
            }
            double split = -INFINITY;
            while (count) {
                split = row_split(row, y, site[count - 1], x);
                if (split > edge[count - 1]) {
                    break;
                }
                count--;
            }
            edge[count] = count ? split : -INFINITY;
            site[count++] = x;
        }
        if (!count) {
            continue;
        }
        for (int x = 0, at = 0; x < w; x++) {
            while (at + 1 < count && edge[at + 1] < x) {
                at++;
            }
            int col = site[at];
            size_t src = (size_t)row[col] * w + col, dest = (size_t)y * w + x;
            for (int a = 0; a < 3; a++) {
                filled[a * n + dest] = points[a * n + src];
            }
        }
    }
}

static int surface_normals(float *surface, int h, int w, double sigma, float *work) {
    if (!(sigma > 0 && sigma < (INT_MAX - 1.) / 8)) {
        return 0;
    }
    int radius = (int)(4 * sigma + .5);
    size_t n = (size_t)h * w;
    float *filled = work, *temp = filled + 3 * n;
    float *normal = surface + 3 * n;
    int *nearest = malloc(n * sizeof(*nearest));
    int *site = malloc((size_t)w * sizeof(*site));
    double *edge = malloc((size_t)w * sizeof(*edge));
    double *weights = malloc((2 * (size_t)radius + 1) * sizeof(*weights));
    int ok = nearest && site && edge && weights;
    if (ok) {
        fill_nearest(surface, filled, nearest, h, w, site, edge);
        double factor = -.5 / (sigma * sigma), total = 0;
        for (int i = -radius; i <= radius; i++) {
            double power = i;
            weights[i + radius] = radius ? exp(factor * (power * power)) : 1;
            total += weights[i + radius];
        }
        for (int i = 0; i <= 2 * radius; i++) {
            weights[i] /= total;
        }
        Blur blur = {.radius = radius, .plain = plain_taps(radius), .weights = weights};
        for (int a = 0; ok && a < 3; a++) {
            ok = gaussian_mesh(filled + a * n, temp, filled + a * n, h, w, &blur);
        }
        if (ok) {
            float_unit_normals(filled, h, w, normal);
        }
    }
    free(nearest);
    free(site);
    free(edge);
    free(weights);
    return ok;
}

static int layout_normals_raw(
    const float *mesh, int h, int w, double x, double y, double spacing, float *result) {
    int rows = h + 4, columns = w + 4;
    size_t cells = (size_t)h * w, expanded = (size_t)rows * columns;
    float *surface = malloc(6 * expanded * sizeof(*surface));
    float *work = malloc(4 * expanded * sizeof(*work));
    if (!surface || !work) {
        free(work);
        free(surface);
        return 0;
    }
    for (size_t i = 0; i < 3 * expanded; i++)
        surface[i] = -1;
    size_t known = 0;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            size_t i = (size_t)row * w + col;
            if (mesh[i] > 0) {
                size_t at = (size_t)(row + 2) * columns + col + 2;
                for (int a = 0; a < 3; a++)
                    surface[a * expanded + at] = mesh[a * cells + i];
                known++;
            }
        }
    }
    int ok = 1;
    if (known) {
        extend_surface(surface, work, rows, columns);
        ok = surface_normals(surface, rows, columns, 225. / spacing, work);
        float *n = surface + 3 * expanded;
        double vote = 0;
        for (size_t i = 0; ok && i < expanded; i++)
            if (surface[i] > 0)
                vote += n[i] * (x - work[i]) + n[i + expanded] * (y - work[i + expanded]);
        for (size_t i = 0; ok && i < 3 * expanded; i++)
            result[i] = vote < 0 ? -n[i] : n[i];
    } else {
        memset(result, 0, 3 * expanded * sizeof(*result));
    }
    free(work);
    free(surface);
    return ok;
}

float *hx_readable_surface(const char *folder, int stride, double voxel, int *height, int *width) {
    TIFF *first = open_axis(folder, 0);
    if (!first) {
        return NULL;
    }
    uint32_t w = 0, h = 0;
    TIFFGetField(first, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(first, TIFFTAG_IMAGELENGTH, &h);
    TIFFClose(first);
    if (!h || !w || h > INT_MAX - 2 * BRIDGE - 1 || w > INT_MAX - 2 * BRIDGE - 1 ||
        (uint64_t)(h + 2 * BRIDGE + 1) * (w + 2 * BRIDGE + 1) > SIZE_MAX / (8 * sizeof(double))) {
        return NULL;
    }
    size_t n = (size_t)h * w, expanded = (size_t)(h + 4) * (w + 4);
    double *xyz = malloc(3 * n * sizeof(*xyz));
    uint8_t *good = malloc(2 * n);
    float *surface = malloc(6 * expanded * sizeof(*surface));
    float *work = malloc(4 * expanded * sizeof(*work));
    int ok = xyz && good && surface && work;
    MeshType type = {0};
    for (int a = 0; ok && a < 3; a++) {
        MeshType t = {0};
        ok = read_axis(folder, w, h, a, NULL, xyz + a * n, NULL, &t.bits, &t.format);
        type = a ? common_type(type, t) : t;
    }
    size_t known = 0, holes = 0;
    for (size_t i = 0; ok && i < n; i++) {
        good[i] = xyz[i] > 0;
        known += good[i];
    }
    uint8_t *gap = good ? good + n : NULL;
    if (ok && known && known < n) {
        ok = gap_mask(good, h, w, gap, &holes, BRIDGE);
    }
    if (ok && holes) {
        ok = bridge_grid(
            xyz, type, h, w, stride, good, gap, holes, work, &default_blur, RELAXATIONS, 3);
        type = (MeshType){.bits = 32, .format = SAMPLEFORMAT_IEEEFP};
        for (size_t i = 0; ok && i < 3 * n; i++) {
            xyz[i] = work[i];
        }
        for (size_t i = 0; ok && i < n; i++) {
            good[i] = xyz[i] > 0;
        }
    }
    Blur blur = {.radius = 6, .plain = plain_taps(6), .weights = smooth_weights};
    if (ok && known) {
        ok = smooth_grid(xyz, type, good, h, w, &blur, work + n);
    }
    for (size_t i = 0; ok && i < 3 * expanded; i++) {
        surface[i] = -1;
    }
    if (ok && known) {
        for (int a = 0; a < 3; a++) {
            for (uint32_t y = 0; y < h; y++) {
                memcpy(surface + a * expanded + (size_t)(y + 2) * (w + 4) + 2,
                    work + (a + 1) * n + (size_t)y * w, w * sizeof(*work));
            }
        }
        extend_surface(surface, work, h + 4, w + 4);
        ok = surface_normals(surface, h + 4, w + 4, 225. / (stride * voxel), work);
    } else if (ok) {
        memset(surface + 3 * expanded, 0, 3 * expanded * sizeof(*surface));
    }
    free(xyz);
    free(good);
    free(work);
    if (!ok) {
        free(surface);
        return NULL;
    }
    *height = h + 4;
    *width = w + 4;
    return surface;
}
#endif
#endif
