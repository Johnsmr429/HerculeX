#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIDE 384

typedef struct {
    float z, y, x;
    int32_t set, kind;
    float w;
} Point;

typedef struct {
    int d, h, w;
    float *ct, *sh, *u;
    int n;
    Point *pts;
    uint8_t *lab;
} Slab;

static const float WIND[6][3] = {{142, 108, 255}, {255, 176, 32}, {47, 184, 198}, {255, 107, 107},
    {78, 139, 255}, {123, 203, 76}};
static const float RAMP[4][3] = {{18, 16, 28}, {86, 46, 8}, {253, 174, 0}, {255, 239, 204}};
static const float CREAM[3] = {255, 239, 204};

static void die(const char *what, const char *path) {
    fprintf(stderr, "tiles: %s %s\n", what, path);
    exit(1);
}

static void *must(void *p) {
    if (!p)
        die("out of memory", "");
    return p;
}

static Slab read_slab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot open", path);
    int32_t hdr[4];
    Slab s = {0};
    if (fread(hdr, 4, 4, f) != 4)
        die("short header", path);
    s.d = hdr[0];
    s.h = hdr[1];
    s.w = hdr[2];
    size_t v = (size_t)s.d * s.h * s.w;
    float **fields[3] = {&s.ct, &s.sh, &s.u};
    for (int k = 0; k < 3; k++) {
        *fields[k] = must(malloc(v * 4));
        if (fread(*fields[k], 4, v, f) != v)
            die("short field", path);
    }
    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1)
        die("short points", path);
    s.n = n;
    s.pts = must(malloc((size_t)(n ? n : 1) * sizeof(Point)));
    if ((int)fread(s.pts, sizeof(Point), n, f) != n)
        die("short points", path);
    s.lab = must(malloc(v));
    if (fread(s.lab, 1, v, f) != v) {
        free(s.lab);
        s.lab = NULL;
    }
    fclose(f);
    return s;
}

static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static float percentile(const float *a, int n, float q) {
    float *c = must(malloc((size_t)n * 4));
    memcpy(c, a, (size_t)n * 4);
    qsort(c, n, 4, cmpf);
    float pos = q / 100.f * (n - 1);
    int i = (int)pos;
    float r = i + 1 < n ? c[i] + (c[i + 1] - c[i]) * (pos - i) : c[n - 1];
    free(c);
    return r;
}

static float cubic(float x) {
    x = fabsf(x);
    if (x < 1)
        return (1.5f * x - 2.5f) * x * x + 1;
    if (x < 2)
        return ((-0.5f * x + 2.5f) * x - 4) * x + 2;
    return 0;
}

static float *upsample(const float *a, int n, int out) {
    float *tmp = must(malloc((size_t)n * out * 4)), *r = must(malloc((size_t)out * out * 4));
    float scale = (float)n / out;
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < out; i++) {
            float c = (i + 0.5f) * scale - 0.5f;
            int c0 = (int)floorf(c);
            float wts[4], sum = 0;
            for (int k = 0; k < 4; k++)
                sum += wts[k] = cubic(c - (c0 - 1 + k));
            for (int j = 0; j < (pass ? out : n); j++) {
                float acc = 0;
                for (int k = 0; k < 4; k++) {
                    int q = c0 - 1 + k;
                    q = q < 0 ? 0 : q >= n ? n - 1 : q;
                    acc += wts[k] * (pass ? tmp[(size_t)q * out + j] : a[(size_t)j * n + q]);
                }
                if (pass)
                    r[(size_t)i * out + j] = acc / sum;
                else
                    tmp[(size_t)j * out + i] = acc / sum;
            }
        }
    free(tmp);
    return r;
}

static void write_bmp(const char *path, const float *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f)
        die("cannot write", path);
    int row = (w * 3 + 3) & ~3;
    uint32_t size = 54 + (uint32_t)row * h;
    uint8_t hd[54] = {'B', 'M'};
    uint32_t v32[] = {size, 0, 54, 40};
    memcpy(hd + 2, &v32[0], 4);
    memcpy(hd + 10, &v32[2], 4);
    memcpy(hd + 14, &v32[3], 4);
    int32_t dims[2] = {w, h};
    memcpy(hd + 18, dims, 8);
    uint16_t planes[2] = {1, 24};
    memcpy(hd + 26, planes, 4);
    fwrite(hd, 1, 54, f);
    uint8_t *line = must(calloc(row, 1));
    for (int y = h - 1; y >= 0; y--) {
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                float v = rgb[((size_t)y * w + x) * 3 + c];
                line[x * 3 + 2 - c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f);
            }
        fwrite(line, 1, row, f);
    }
    free(line);
    fclose(f);
}

static void ramp(float t, float *o) {
    t = (t < 0 ? 0 : t > 1 ? 1 : t) * 3;
    int i = t >= 3 ? 2 : (int)t;
    float f = t - i;
    for (int c = 0; c < 3; c++)
        o[c] = RAMP[i][c] * (1 - f) + RAMP[i + 1][c] * f;
}

static void wind(float u, float *o) {
    float t = fmodf(u, 6.f);
    if (t < 0)
        t += 6;
    int i = (int)floorf(t);
    float f = t - i;
    f = f * f * (3 - 2 * f);
    for (int c = 0; c < 3; c++)
        o[c] = WIND[i % 6][c] * (1 - f) + WIND[(i + 1) % 6][c] * f;
}

static float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

static void tile_path(char *buf, size_t n, const char *out, const char *name, const char *kind) {
    snprintf(buf, n, "%s/%s-%s.bmp", out, name, kind);
}

static void slab_tiles(FILE *js, const char *name, const char *path, int zpick, const char *out) {
    Slab s = read_slab(path);
    int hw = s.h * s.w, n = SIDE * SIDE;
    int z = s.d / 2;
    if (zpick >= 0)
        z = zpick;
    else if (s.n) {
        float *zs = must(malloc((size_t)s.n * 4));
        for (int i = 0; i < s.n; i++)
            zs[i] = s.pts[i].z;
        z = (int)lroundf(percentile(zs, s.n, 50));
        free(zs);
        z = z < 0 ? 0 : z >= s.d ? s.d - 1 : z;
    }
    const float *ct = s.ct + (size_t)z * hw, *sh = s.sh + (size_t)z * hw, *u = s.u + (size_t)z * hw;
    float lo = percentile(ct, hw, 1), hi = percentile(ct, hw, 99);
    float *g0 = must(malloc((size_t)hw * 4));
    for (int i = 0; i < hw; i++)
        g0[i] = clamp01((ct[i] - lo) / (hi - lo + 1e-6f));
    float *g = upsample(g0, s.w, SIDE), *p = upsample(sh, s.w, SIDE), *w = upsample(u, s.w, SIDE);
    float *rgb = must(malloc((size_t)n * 12));
    char file[1024];
    for (int i = 0; i < n; i++)
        for (int c = 0; c < 3; c++)
            rgb[i * 3 + c] = g[i] * 255;
    tile_path(file, sizeof file, out, name, "ct");
    write_bmp(file, rgb, SIDE, SIDE);
    for (int i = 0; i < n; i++)
        ramp(clamp01(p[i]), rgb + i * 3);
    tile_path(file, sizeof file, out, name, "sheet");
    write_bmp(file, rgb, SIDE, SIDE);
    for (int i = 0; i < n; i++) {
        float on = clamp01((clamp01(p[i]) - 0.42f) / 0.16f), col[3], b = g[i] * 255;
        wind(w[i], col);
        for (int c = 0; c < 3; c++)
            rgb[i * 3 + c] = b * 0.32f * (1 - on) + (0.12f * b + 0.88f * col[c]) * on;
    }
    tile_path(file, sizeof file, out, name, "winding");
    write_bmp(file, rgb, SIDE, SIDE);
    int labelled = 0;
    if (s.lab)
        for (int i = 0; i < hw; i++)
            labelled |= s.lab[(size_t)z * hw + i] != 255;
    if (labelled) {
        for (int i = 0; i < hw; i++) {
            uint8_t v = s.lab[(size_t)z * hw + i];
            g0[i] = v == 255 ? 0.5f : v;
        }
        float *l = upsample(g0, s.w, SIDE);
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 3; c++)
                rgb[i * 3 + c] = clamp01(l[i]) * CREAM[c];
        free(l);
        tile_path(file, sizeof file, out, name, "label");
        write_bmp(file, rgb, SIDE, SIDE);
    }
    uint8_t hidden[256];
    int order[256];
    for (int t = 0; t < 256; t++)
        order[t] = t, hidden[t] = 0;
    uint64_t r = 7;
    for (int t = 255; t > 0; t--) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        int k = (int)((r >> 33) % (uint64_t)(t + 1)), tmp = order[t];
        order[t] = order[k];
        order[k] = tmp;
    }
    for (int t = 0; t < 192; t++)
        hidden[order[t]] = 1;
    int cell = SIDE / 16;
    for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
            int i = y * SIDE + x;
            if (!hidden[(y / cell) * 16 + x / cell])
                for (int c = 0; c < 3; c++)
                    rgb[i * 3 + c] = g[i] * 255;
            else {
                int edge = y % cell == 0 || x % cell == 0;
                const float m[3] = {226, 222, 236}, e[3] = {206, 200, 222};
                for (int c = 0; c < 3; c++)
                    rgb[i * 3 + c] = edge ? e[c] : m[c];
            }
        }
    tile_path(file, sizeof file, out, name, "masked");
    write_bmp(file, rgb, SIDE, SIDE);
    float k = (float)SIDE / s.h;
    fprintf(js, "\"%s\":{\"z\":%d,\"points\":[", name, z);
    int first = 1;
    for (int i = 0; i < s.n; i++)
        if (fabsf(s.pts[i].z - z) <= 2) {
            fprintf(js, "%s[%.1f,%.1f]", first ? "" : ",", s.pts[i].x * k, s.pts[i].y * k);
            first = 0;
        }
    fprintf(js, "],\"turns\":[");
    float wmin = w[0], wmax = w[0];
    for (int i = 1; i < n; i++)
        wmin = w[i] < wmin ? w[i] : wmin, wmax = w[i] > wmax ? w[i] : wmax;
    float *ys = must(malloc((size_t)n * 4)), *xs = must(malloc((size_t)n * 4));
    first = 1;
    for (int t = (int)floorf(wmin); t <= (int)ceilf(wmax); t++) {
        int m = 0;
        for (int i = 0; i < n; i++)
            if (p[i] > 0.5f && fabsf(w[i] - t) < 0.2f)
                ys[m] = (float)(i / SIDE), xs[m] = (float)(i % SIDE), m++;
        if (m <= 400)
            continue;
        float my = percentile(ys, m, 50), mx = percentile(xs, m, 50), best = 1e30f;
        int bi = 0;
        for (int j = 0; j < m; j++) {
            float dd = (ys[j] - my) * (ys[j] - my) + (xs[j] - mx) * (xs[j] - mx);
            if (dd < best)
                best = dd, bi = j;
        }
        fprintf(js, "%s[%d,%d,%d]", first ? "" : ",", t, (int)xs[bi], (int)ys[bi]);
        first = 0;
    }
    fprintf(js, "]}");
    free(ys);
    free(xs);
    free(rgb);
    free(g0);
    free(g);
    free(p);
    free(w);
    free(s.ct);
    free(s.sh);
    free(s.u);
    free(s.pts);
    free(s.lab);
}

static void context_tile(const char *npy, const char *out) {
    FILE *f = fopen(npy, "rb");
    if (!f)
        die("cannot open", npy);
    uint8_t pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6))
        die("not an npy file", npy);
    int hl = pre[8] | pre[9] << 8;
    char *hdr = must(calloc(hl + 1, 1));
    if ((int)fread(hdr, 1, hl, f) != hl || !strstr(hdr, "'|u1'"))
        die("expected a uint8 array", npy);
    long rows = 0, cols = 0;
    char *sp = strstr(hdr, "'shape': (");
    if (!sp || sscanf(sp, "'shape': (%ld, %ld)", &rows, &cols) != 2)
        die("expected a 2D array", npy);
    free(hdr);
    uint8_t *a = must(malloc((size_t)rows * cols));
    if (fread(a, 1, (size_t)rows * cols, f) != (size_t)rows * cols)
        die("short array", npy);
    fclose(f);
    long side = rows > cols ? rows : cols, oy = (side - rows) / 2, ox = (side - cols) / 2;
    double step = side / 256.0;
    float *c = must(calloc(256 * 256, 4)), *vals = must(malloc(256 * 256 * 4));
    double *rowacc = must(malloc(256 * sizeof(double)));
    for (int ty = 0; ty < 256; ty++) {
        memset(rowacc, 0, 256 * sizeof(double));
        double y0 = ty * step, y1 = (ty + 1) * step;
        for (long sy = (long)y0; sy < (long)ceil(y1); sy++) {
            double wy = fmin(sy + 1, y1) - fmax(sy, y0);
            long ry = sy - oy;
            if (ry < 0 || ry >= rows || wy <= 0)
                continue;
            for (int tx = 0; tx < 256; tx++) {
                double x0 = tx * step, x1 = (tx + 1) * step, acc = 0;
                for (long sx = (long)x0; sx < (long)ceil(x1); sx++) {
                    double wx = fmin(sx + 1, x1) - fmax(sx, x0);
                    long rx = sx - ox;
                    if (rx >= 0 && rx < cols && wx > 0)
                        acc += wx * a[ry * cols + rx];
                }
                rowacc[tx] += wy * acc;
            }
        }
        for (int tx = 0; tx < 256; tx++)
            c[ty * 256 + tx] = (float)(rowacc[tx] / (step * step));
    }
    int m = 0;
    for (int i = 0; i < 256 * 256; i++)
        if (c[i] > 0)
            vals[m++] = c[i];
    float lo = percentile(vals, m, 2), hi = percentile(vals, m, 99.5f);
    for (int i = 0; i < 256 * 256; i++)
        c[i] = clamp01((c[i] - lo) / (hi - lo));
    float *up = upsample(c, 256, 512), *rgb = must(malloc(512 * 512 * 12));
    for (int i = 0; i < 512 * 512; i++)
        for (int k = 0; k < 3; k++)
            rgb[i * 3 + k] = clamp01(up[i]) * 255;
    char file[1024];
    snprintf(file, sizeof file, "%s/context.bmp", out);
    write_bmp(file, rgb, 512, 512);
    free(a);
    free(c);
    free(vals);
    free(rowacc);
    free(up);
    free(rgb);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: tiles DUMP_DIR CONTEXT.npy OUT_DIR\n");
        return 2;
    }
    const struct {
        const char *name, *file;
        int z;
    } picks[] = {{"hero", "slab-0011-source1.bin", 32}, {"points", "slab-0022-source3.bin", -1},
        {"lines", "slab-0031-source5.bin", -1}, {"kaggle", "slab-0044-source7.bin", 32}};
    char path[1024];
    snprintf(path, sizeof path, "%s/tiles.js", argv[3]);
    FILE *js = fopen(path, "w");
    if (!js)
        die("cannot write", path);
    fprintf(js, "window.TILES = {");
    for (size_t i = 0; i < sizeof picks / sizeof picks[0]; i++) {
        snprintf(path, sizeof path, "%s/%s", argv[1], picks[i].file);
        if (i)
            fprintf(js, ",");
        slab_tiles(js, picks[i].name, path, picks[i].z, argv[3]);
    }
    fprintf(js, "};\n");
    fclose(js);
    context_tile(argv[2], argv[3]);
    return 0;
}
