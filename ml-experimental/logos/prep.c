#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <json-c/json.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <zlib.h>

enum { SD = 64, SH = 128, SW = 128, CTX = 256, CTXN = 4, MAXPTS = 4096 };
enum { LAYOUT_LABELLED = 0, LAYOUT_UNLABELLED = 1, LAYOUT_SURFACE = 2 };

typedef struct {
    float z, y, x;
} Knot;

typedef struct {
    Knot *k;
    int n;
} Axis;

typedef struct {
    uint8_t *data;
    int64_t shape[3];
} Vol8;

typedef struct {
    uint16_t *data;
    int64_t shape[3];
} Vol16;

typedef struct {
    float z, y, x;
    int32_t set, kind;
    float wind;
} Point;

typedef struct {
    FILE *f;
    int layout;
    uint64_t count;
} Shard;

typedef struct {
    uint32_t source, split;
    int32_t curl;
    uint32_t ct_bits;
    float voxel_um, confidence;
    int32_t oz, oy, ox;
    float cy0, cx0, cy1, cx1, cscale;
} Meta;

static void die(const char *fmt, const char *arg) {
    fprintf(stderr, "prep: ");
    fprintf(stderr, fmt, arg);
    fprintf(stderr, "\n");
    exit(1);
}

static void *grab(size_t bytes) {
    void *p = calloc(1, bytes ? bytes : 1);
    if (!p)
        die("out of memory%s", "");
    return p;
}

static const char *opt(int argc, char **argv, const char *key, const char *fallback) {
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], key))
            return argv[i + 1];
    return fallback;
}

static double num(const char *s) {
    char *end = NULL;
    double v = s ? strtod(s, &end) : 0;
    if (!s || end == s || *end)
        die("not a number: %s", s ? s : "(missing)");
    return v;
}

static int cmp_knot(const void *a, const void *b) {
    float x = ((const Knot *)a)->z, y = ((const Knot *)b)->z;
    return (x > y) - (x < y);
}

static void axis_push(Axis *a, float z, float y, float x) {
    a->k = realloc(a->k, sizeof(Knot) * (size_t)(a->n + 1));
    a->k[a->n++] = (Knot){z, y, x};
}

static Axis axis_load(const char *path, const char *format, double factor) {
    Axis a = {0};
    if (!strcmp(format, "zyx")) {
        FILE *f = fopen(path, "r");
        if (!f)
            die("cannot read %s", path);
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            for (char *p = line; *p; p++)
                if (*p == ',' || *p == '[' || *p == ']')
                    *p = ' ';
            double z, y, x;
            if (sscanf(line, "%lf %lf %lf", &z, &y, &x) == 3)
                axis_push(&a, (float)(z / factor), (float)(y / factor), (float)(x / factor));
        }
        fclose(f);
    } else {
        json_object *root = json_object_from_file(path), *list = root, *v;
        if (!root)
            die("cannot read %s", path);
        if (json_object_is_type(root, json_type_object)) {
            if (json_object_object_get_ex(root, "control_points", &v) ||
                json_object_object_get_ex(root, "points", &v))
                list = v;
            else if (json_object_object_get_ex(root, "knots_z_x_y", &v)) {
                for (size_t i = 0; i < json_object_array_length(v); i++) {
                    json_object *r = json_object_array_get_idx(v, i);
                    axis_push(&a,
                        (float)(json_object_get_double(json_object_array_get_idx(r, 0)) / factor),
                        (float)(json_object_get_double(json_object_array_get_idx(r, 2)) / factor),
                        (float)(json_object_get_double(json_object_array_get_idx(r, 1)) / factor));
                }
                list = NULL;
            }
        }
        for (size_t i = 0; list && i < json_object_array_length(list); i++) {
            json_object *r = json_object_array_get_idx(list, i), *x, *y, *z;
            if (json_object_object_get_ex(r, "x", &x) && json_object_object_get_ex(r, "y", &y) &&
                json_object_object_get_ex(r, "z", &z))
                axis_push(&a, (float)(json_object_get_double(z) / factor),
                    (float)(json_object_get_double(y) / factor),
                    (float)(json_object_get_double(x) / factor));
        }
        json_object_put(root);
    }
    if (a.n < 2)
        die("too few centre-line points in %s", path);
    qsort(a.k, (size_t)a.n, sizeof(Knot), cmp_knot);
    return a;
}

static void axis_at(const Axis *a, double z, float *cy, float *cx) {
    if (z <= a->k[0].z) {
        *cy = a->k[0].y;
        *cx = a->k[0].x;
        return;
    }
    if (z >= a->k[a->n - 1].z) {
        *cy = a->k[a->n - 1].y;
        *cx = a->k[a->n - 1].x;
        return;
    }
    int lo = 0, hi = a->n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (a->k[mid].z <= z)
            lo = mid;
        else
            hi = mid;
    }
    double t = (z - a->k[lo].z) / (a->k[hi].z - a->k[lo].z);
    *cy = (float)(a->k[lo].y + t * (a->k[hi].y - a->k[lo].y));
    *cx = (float)(a->k[lo].x + t * (a->k[hi].x - a->k[lo].x));
}

static Vol8 npy_load(const char *path) {
    Vol8 v = {0};
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot read %s", path);
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0)
        die("not an npy file: %s", path);
    size_t hlen = pre[6] == 1 ? (size_t)(pre[8] | pre[9] << 8) : 0;
    if (pre[6] != 1) {
        unsigned char more[2];
        if (fread(more, 1, 2, f) != 2)
            die("bad npy header: %s", path);
        hlen = (size_t)(pre[8] | pre[9] << 8 | more[0] << 16 | more[1] << 24);
    }
    char *h = grab(hlen + 1);
    if (fread(h, 1, hlen, f) != hlen)
        die("bad npy header: %s", path);
    if (!strstr(h, "|u1") || strstr(h, "'fortran_order': True"))
        die("expected a C-order uint8 npy: %s", path);
    const char *s = strchr(strstr(h, "'shape'"), '(');
    if (sscanf(s, "(%lld, %lld, %lld)", (long long *)&v.shape[0], (long long *)&v.shape[1],
            (long long *)&v.shape[2]) != 3)
        die("expected a 3D npy: %s", path);
    free(h);
    size_t n = (size_t)(v.shape[0] * v.shape[1] * v.shape[2]);
    v.data = grab(n);
    if (fread(v.data, 1, n, f) != n)
        die("short npy: %s", path);
    fclose(f);
    return v;
}

static unsigned char *gunzip_all(const unsigned char *in, size_t n, size_t want) {
    unsigned char *out = grab(want);
    z_stream zs = {0};
    zs.next_in = (unsigned char *)in;
    zs.avail_in = (uInt)n;
    zs.next_out = out;
    zs.avail_out = (uInt)want;
    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK)
        die("zlib init failed%s", "");
    int rc = inflate(&zs, Z_FINISH);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || zs.total_out != want)
        die("gzip data has the wrong size%s", "");
    return out;
}

static Vol16 nrrd_load(const char *path) {
    Vol16 v = {0};
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot read %s", path);
    char line[1024];
    int gz = 0, bytes = 2;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || (line[0] == '\r' && line[1] == '\n'))
            break;
        if (!strncmp(line, "type:", 5))
            bytes = strstr(line, "uint8") || strstr(line, "unsigned char") ? 1 : 2;
        if (!strncmp(line, "encoding:", 9))
            gz = strstr(line, "gzip") != NULL;
        if (!strncmp(line, "endian:", 7) && strstr(line, "big"))
            die("big-endian nrrd: %s", path);
        if (!strncmp(line, "sizes:", 6) &&
            sscanf(line + 6, "%lld %lld %lld", (long long *)&v.shape[2], (long long *)&v.shape[1],
                (long long *)&v.shape[0]) != 3)
            die("expected a 3D nrrd: %s", path);
    }
    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    size_t rest = (size_t)(ftell(f) - start);
    fseek(f, start, SEEK_SET);
    unsigned char *raw = grab(rest);
    if (fread(raw, 1, rest, f) != rest)
        die("short nrrd: %s", path);
    fclose(f);
    size_t n = (size_t)(v.shape[0] * v.shape[1] * v.shape[2]);
    unsigned char *payload = gz ? gunzip_all(raw, rest, n * (size_t)bytes) : raw;
    if (!gz && rest < n * (size_t)bytes)
        die("short nrrd payload: %s", path);
    v.data = grab(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++)
        v.data[i] = bytes == 1 ? payload[i] : (uint16_t)(payload[2 * i] | payload[2 * i + 1] << 8);
    if (payload != raw)
        free(payload);
    free(raw);
    return v;
}

static Vol16 tiff_load(const char *path) {
    Vol16 v = {0};
    TIFF *t = TIFFOpen(path, "r");
    if (!t)
        die("cannot read %s", path);
    uint32_t w = 0, h = 0;
    uint16_t bits = 8;
    int pages = 0;
    do
        pages++;
    while (TIFFReadDirectory(t));
    TIFFSetDirectory(t, 0);
    TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(t, TIFFTAG_BITSPERSAMPLE, &bits);
    v.shape[0] = pages;
    v.shape[1] = h;
    v.shape[2] = w;
    v.data = grab((size_t)pages * h * w * sizeof(uint16_t));
    unsigned char *row = grab((size_t)TIFFScanlineSize(t));
    for (int p = 0; p < pages; p++) {
        TIFFSetDirectory(t, (uint16_t)p);
        for (uint32_t y = 0; y < h; y++) {
            if (TIFFReadScanline(t, row, y, 0) < 0)
                die("cannot read a TIFF row in %s", path);
            uint16_t *out = v.data + ((size_t)p * h + y) * w;
            for (uint32_t x = 0; x < w; x++)
                out[x] = bits == 16 ? ((uint16_t *)row)[x] : row[x];
        }
    }
    free(row);
    TIFFClose(t);
    return v;
}

static Shard shard_open(const char *path, int layout) {
    Shard s = {fopen(path, "wb"), layout, 0};
    if (!s.f)
        die("cannot write %s", path);
    unsigned char head[64] = {'L', 'G', 'S', 'F'};
    memcpy(head + 4, &(uint32_t){(uint32_t)layout}, 4);
    fwrite(head, 1, 64, s.f);
    return s;
}

static void shard_close(Shard *s) {
    fseek(s->f, 8, SEEK_SET);
    fwrite(&s->count, 8, 1, s->f);
    if (fclose(s->f))
        die("cannot finish a shard%s", "");
    fprintf(stderr, "prep: %llu records\n", (unsigned long long)s->count);
}

static void write_header(Shard *s, const Meta *m, uint32_t npoints) {
    unsigned char h[128] = {'L', 'G', 'S', '1'};
    uint32_t u[] = {1, m->source, m->split, (uint32_t)m->curl, m->ct_bits};
    memcpy(h + 4, u, sizeof(u));
    float f[] = {m->voxel_um, m->confidence};
    memcpy(h + 24, f, sizeof(f));
    int32_t o[] = {m->oz, m->oy, m->ox};
    memcpy(h + 32, o, sizeof(o));
    float c[] = {m->cy0, m->cx0, m->cy1, m->cx1, m->cscale};
    memcpy(h + 44, c, sizeof(c));
    memcpy(h + 64, &npoints, 4);
    fwrite(h, 1, 128, s->f);
}

static void context_slab(const Vol8 *ctx, double factor, const Meta *m, uint8_t *out, Meta *mo) {
    memset(out, 0, CTXN * CTX * CTX);
    int step = 1;
    while (ctx->shape[1] / step > CTX || ctx->shape[2] / step > CTX)
        step++;
    double scale = factor * step;
    const int offs[CTXN] = {-24, -8, 8, 24};
    for (int s = 0; s < CTXN; s++) {
        int64_t cz = (int64_t)floor((m->oz + SD / 2 + offs[s]) / factor);
        cz = cz < 0 ? 0 : cz >= ctx->shape[0] ? ctx->shape[0] - 1 : cz;
        for (int64_t y = 0; y < CTX && y * step < ctx->shape[1]; y++)
            for (int64_t x = 0; x < CTX && x * step < ctx->shape[2]; x++)
                out[((size_t)s * CTX + y) * CTX + x] =
                    ctx->data[(cz * ctx->shape[1] + y * step) * ctx->shape[2] + x * step];
    }
    mo->cy0 = (float)(m->oy / scale);
    mo->cx0 = (float)(m->ox / scale);
    mo->cy1 = (float)((m->oy + SH) / scale);
    mo->cx1 = (float)((m->ox + SW) / scale);
    mo->cscale = (float)scale;
}

static void centre_slab(const Axis *a, const Meta *m, float *out) {
    for (int z = 0; z < SD; z++) {
        float cy, cx;
        axis_at(a, m->oz + z, &cy, &cx);
        out[2 * z] = cy - (float)m->oy;
        out[2 * z + 1] = cx - (float)m->ox;
    }
}

static void write_labelled(Shard *s, Meta m, const Vol8 *ctx, double factor, const Axis *a,
    const uint16_t *ct, const uint8_t *sheet, const uint16_t *inst, const Point *pts, uint32_t np) {
    static uint8_t context[CTXN * CTX * CTX];
    static float centre[SD * 2];
    static unsigned char ptbuf[24 * MAXPTS];
    context_slab(ctx, factor, &m, context, &m);
    centre_slab(a, &m, centre);
    if (np > MAXPTS)
        np = MAXPTS;
    write_header(s, &m, np);
    size_t n = (size_t)SD * SH * SW;
    fwrite(ct, 2, n, s->f);
    fwrite(sheet, 1, n, s->f);
    fwrite(inst, 2, n, s->f);
    fwrite(context, 1, sizeof(context), s->f);
    fwrite(centre, 4, SD * 2, s->f);
    memset(ptbuf, 0, sizeof(ptbuf));
    for (uint32_t i = 0; i < np; i++) {
        float f3[3] = {pts[i].z, pts[i].y, pts[i].x};
        memcpy(ptbuf + 24 * i, f3, 12);
        memcpy(ptbuf + 24 * i + 12, &pts[i].set, 4);
        memcpy(ptbuf + 24 * i + 16, &pts[i].kind, 4);
        memcpy(ptbuf + 24 * i + 20, &pts[i].wind, 4);
    }
    fwrite(ptbuf, 1, sizeof(ptbuf), s->f);
    s->count++;
}

static void write_unlabelled(
    Shard *s, Meta m, const Vol8 *ctx, double factor, const Axis *a, const uint8_t *ct) {
    static uint8_t context[CTXN * CTX * CTX];
    static float centre[SD * 2];
    context_slab(ctx, factor, &m, context, &m);
    centre_slab(a, &m, centre);
    m.ct_bits = 8;
    write_header(s, &m, 0);
    fwrite(ct, 1, (size_t)SD * SH * SW, s->f);
    fwrite(context, 1, sizeof(context), s->f);
    fwrite(centre, 4, SD * 2, s->f);
    s->count++;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static char **list_dir(const char *dir, const char *must, int *n) {
    DIR *d = opendir(dir);
    if (!d)
        die("cannot list %s", dir);
    char **out = NULL;
    *n = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.' || (must && !strstr(e->d_name, must)))
            continue;
        out = realloc(out, sizeof(char *) * (size_t)(*n + 1));
        out[(*n)++] = strdup(e->d_name);
    }
    closedir(d);
    qsort(out, (size_t)*n, sizeof(char *), cmp_str);
    return out;
}

static void common_meta(int argc, char **argv, Meta *m) {
    m->curl = (int32_t)num(opt(argc, argv, "--curl", NULL));
    m->voxel_um = (float)num(opt(argc, argv, "--voxel", NULL));
    m->source = (uint32_t)num(opt(argc, argv, "--source", NULL));
    m->confidence = (float)num(opt(argc, argv, "--confidence", "1"));
    if (m->curl != 1 && m->curl != -1)
        die("--curl must be 1 or -1%s", "");
}

static double orientation_contrast(
    const uint16_t *ct, const uint16_t *mask, int64_t n, int transpose) {
    double in = 0, out = 0;
    int64_t ni = 0, no = 0;
    for (int64_t z = 0; z < n; z += 2)
        for (int64_t y = 0; y < n; y += 2)
            for (int64_t x = 0; x < n; x += 2) {
                uint16_t c = transpose ? ct[(x * n + y) * n + z] : ct[(z * n + y) * n + x];
                if (mask[(z * n + y) * n + x]) {
                    in += c;
                    ni++;
                } else {
                    out += c;
                    no++;
                }
            }
    return ni && no ? in / ni - out / no : 0;
}

static int cmd_instance(int argc, char **argv) {
    const char *dir = argv[2], *outp = argv[3];
    Meta base = {0};
    common_meta(argc, argv, &base);
    base.ct_bits = 16;
    double cf = num(opt(argc, argv, "--context-factor", NULL));
    Vol8 ctx = npy_load(opt(argc, argv, "--context", NULL));
    Axis a =
        axis_load(opt(argc, argv, "--umbilicus", NULL), opt(argc, argv, "--umbilicus-format", "vc"),
            num(opt(argc, argv, "--umbilicus-factor", "1")));
    int every = (int)num(opt(argc, argv, "--holdout-every", "10"));
    int n = 0;
    char **cubes = list_dir(dir, NULL, &n);
    Shard s = shard_open(outp, LAYOUT_LABELLED);
    size_t sn = (size_t)SD * SH * SW;
    uint16_t *ct = grab(sn * 2), *inst = grab(sn * 2);
    uint8_t *sheet = grab(sn);
    for (int c = 0; c < n; c++) {
        long long cz, cy, cx;
        if (sscanf(cubes[c], "%lld_%lld_%lld", &cz, &cy, &cx) != 3)
            continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s/%s_mask.nrrd", dir, cubes[c], cubes[c]);
        Vol16 mask = nrrd_load(path);
        snprintf(path, sizeof(path), "%s/%s/%s_volume.nrrd", dir, cubes[c], cubes[c]);
        Vol16 vol = nrrd_load(path);
        int64_t e = mask.shape[0];
        if (mask.shape[1] != e || mask.shape[2] != e || vol.shape[0] != e)
            die("instance cube %s is not a cube", cubes[c]);
        double keep = orientation_contrast(vol.data, mask.data, e, 0);
        double flip = orientation_contrast(vol.data, mask.data, e, 1);
        int transpose = flip > keep;
        fprintf(stderr, "prep: %s contrast as stored %.1f, transposed %.1f -> %s\n", cubes[c], keep,
            flip, transpose ? "transposed" : "as stored");
        Meta m = base;
        m.split = (uint32_t)(c % every == every - 1);
        for (int64_t bz = 0; bz + SD <= e; bz += SD)
            for (int64_t by = 0; by + SH <= e; by += SH)
                for (int64_t bx = 0; bx + SW <= e; bx += SW) {
                    for (int64_t z = 0; z < SD; z++)
                        for (int64_t y = 0; y < SH; y++)
                            for (int64_t x = 0; x < SW; x++) {
                                int64_t gz = bz + z, gy = by + y, gx = bx + x;
                                size_t i = ((size_t)z * SH + y) * SW + x;
                                ct[i] = transpose ? vol.data[(gx * e + gy) * e + gz]
                                                  : vol.data[(gz * e + gy) * e + gx];
                                inst[i] = mask.data[(gz * e + gy) * e + gx];
                                sheet[i] = inst[i] ? 1 : 0;
                            }
                    m.oz = (int32_t)(cz + bz);
                    m.oy = (int32_t)(cy + by);
                    m.ox = (int32_t)(cx + bx);
                    write_labelled(&s, m, &ctx, cf, &a, ct, sheet, inst, NULL, 0);
                }
        free(mask.data);
        free(vol.data);
    }
    shard_close(&s);
    return 0;
}

static int cmd_fibre(int argc, char **argv) {
    const char *dir = argv[2], *outp = argv[3];
    Meta base = {0};
    common_meta(argc, argv, &base);
    base.ct_bits = 16;
    double cf = num(opt(argc, argv, "--context-factor", NULL));
    Vol8 ctx = npy_load(opt(argc, argv, "--context", NULL));
    Axis a =
        axis_load(opt(argc, argv, "--umbilicus", NULL), opt(argc, argv, "--umbilicus-format", "vc"),
            num(opt(argc, argv, "--umbilicus-factor", "1")));
    int every = (int)num(opt(argc, argv, "--holdout-every", "10"));
    char images[4096], labels[4096];
    snprintf(images, sizeof(images), "%s/imagesTr", dir);
    snprintf(labels, sizeof(labels), "%s/labelsTr", dir);
    int n = 0;
    const char *prefix = opt(argc, argv, "--prefix", "");
    char **all = list_dir(labels, ".tif", &n);
    char **files = grab(sizeof(char *) * (size_t)(n ? n : 1));
    int kept = 0;
    for (int i = 0; i < n; i++)
        if (!strncmp(all[i], prefix, strlen(prefix)))
            files[kept++] = all[i];
    n = kept;
    Shard s = shard_open(outp, LAYOUT_LABELLED);
    size_t sn = (size_t)SD * SH * SW;
    uint16_t *ct = grab(sn * 2), *inst = grab(sn * 2);
    uint8_t *sheet = grab(sn);
    for (int c = 0; c < n; c++) {
        char stem[1024], path[8192];
        snprintf(stem, sizeof(stem), "%s", files[c]);
        *strstr(stem, ".tif") = 0;
        long long cz, cy, cx, size;
        const char *p = strchr(stem, '_');
        if (!p || sscanf(p + 1, "%lld_%lld_%lld_%lld", &cz, &cy, &cx, &size) != 4) {
            fprintf(stderr, "prep: skipping %s (no position in its name)\n", files[c]);
            continue;
        }
        snprintf(path, sizeof(path), "%s/%s", labels, files[c]);
        Vol16 lab = tiff_load(path);
        snprintf(path, sizeof(path), "%s/%s_0000.tif", images, stem);
        Vol16 img = tiff_load(path);
        if (memcmp(lab.shape, img.shape, sizeof(lab.shape)))
            die("fibre image and label differ in shape: %s", files[c]);
        Meta m = base;
        m.split = (uint32_t)(c % every == every - 1);
        m.ct_bits = 16;
        for (int64_t bz = 0; bz + SD <= img.shape[0]; bz += SD)
            for (int64_t by = 0; by + SH <= img.shape[1]; by += SH)
                for (int64_t bx = 0; bx + SW <= img.shape[2]; bx += SW) {
                    int64_t fibres = 0;
                    for (int64_t z = 0; z < SD; z++)
                        for (int64_t y = 0; y < SH; y++)
                            for (int64_t x = 0; x < SW; x++) {
                                size_t i = ((size_t)z * SH + y) * SW + x;
                                size_t g =
                                    ((size_t)(bz + z) * img.shape[1] + (by + y)) * img.shape[2] +
                                    (bx + x);
                                ct[i] = img.data[g];
                                inst[i] = 0;
                                sheet[i] = lab.data[g] ? 1 : 255;
                                fibres += lab.data[g] != 0;
                            }
                    if (fibres < 64)
                        continue;
                    m.oz = (int32_t)(cz + bz);
                    m.oy = (int32_t)(cy + by);
                    m.ox = (int32_t)(cx + bx);
                    write_labelled(&s, m, &ctx, cf, &a, ct, sheet, inst, NULL, 0);
                }
        free(lab.data);
        free(img.data);
    }
    shard_close(&s);
    return 0;
}

typedef struct {
    Point *p;
    int n;
    int split;
} PointSet;

static int load_sets(
    const char *path, int kind, int base, double factor, PointSet **sets, int *nsets) {
    json_object *root = json_object_from_file(path), *cols;
    if (!root || !json_object_object_get_ex(root, "collections", &cols))
        die("cannot read point collections in %s", path);
    json_object_object_foreach(cols, key, col) {
        json_object *pts;
        if (!json_object_object_get_ex(col, "points", &pts) || json_object_object_length(pts) == 0)
            continue;
        PointSet ps = {0};
        int sid = base + atoi(key);
        ps.split = (sid * 2654435761u >> 16) % 10 == 9;
        json_object_object_foreach(pts, pk, pv) {
            (void)pk;
            json_object *p, *w;
            if (!json_object_object_get_ex(pv, "p", &p) || json_object_array_length(p) < 3)
                continue;
            float wind = NAN;
            if (json_object_object_get_ex(pv, "wind_a", &w) &&
                !json_object_is_type(w, json_type_null))
                wind = (float)json_object_get_double(w);
            if (kind != 0 && isnan(wind))
                continue;
            ps.p = realloc(ps.p, sizeof(Point) * (size_t)(ps.n + 1));
            ps.p[ps.n++] =
                (Point){(float)(json_object_get_double(json_object_array_get_idx(p, 2)) / factor),
                    (float)(json_object_get_double(json_object_array_get_idx(p, 1)) / factor),
                    (float)(json_object_get_double(json_object_array_get_idx(p, 0)) / factor), sid,
                    kind, wind};
        }
        if (ps.n) {
            *sets = realloc(*sets, sizeof(PointSet) * (size_t)(*nsets + 1));
            (*sets)[(*nsets)++] = ps;
        }
    }
    json_object_put(root);
    return 0;
}

static int load_line(const char *path, int sid, double factor, double spacing, PointSet *out) {
    json_object *root = json_object_from_file(path), *lp;
    if (!root)
        return -1;
    if (!json_object_object_get_ex(root, "line_points", &lp) || json_object_array_length(lp) < 2) {
        json_object_put(root);
        return -1;
    }
    PointSet ps = {0};
    ps.split = 1;
    double carry = 0, pz = 0, py = 0, px = 0;
    size_t n = json_object_array_length(lp);
    for (size_t i = 0; i < n; i++) {
        json_object *q = json_object_array_get_idx(lp, i);
        double x = json_object_get_double(json_object_array_get_idx(q, 0)) / factor;
        double y = json_object_get_double(json_object_array_get_idx(q, 1)) / factor;
        double z = json_object_get_double(json_object_array_get_idx(q, 2)) / factor;
        if (i) {
            carry += sqrt((z - pz) * (z - pz) + (y - py) * (y - py) + (x - px) * (x - px));
            if (carry < spacing) {
                pz = z, py = y, px = x;
                continue;
            }
            carry = 0;
        }
        ps.p = realloc(ps.p, sizeof(Point) * (size_t)(ps.n + 1));
        ps.p[ps.n++] = (Point){(float)z, (float)y, (float)x, sid, 0, NAN};
        pz = z, py = y, px = x;
    }
    json_object_put(root);
    *out = ps;
    return 0;
}

static int inside(const Point *p, int64_t oz, int64_t oy, int64_t ox) {
    return p->z >= oz && p->z <= oz + SD - 1 && p->y >= oy && p->y <= oy + SH - 1 && p->x >= ox &&
        p->x <= ox + SW - 1;
}

static void plan_boxes(PointSet *sets, int nsets, int max_per_set, FILE *out) {
    for (int s = 0; s < nsets; s++) {
        unsigned char *used = grab((size_t)sets[s].n);
        int boxes = 0;
        for (int i = 0; i < sets[s].n && boxes < max_per_set; i++) {
            if (used[i])
                continue;
            int64_t oz = (int64_t)floor(sets[s].p[i].z) - SD / 2,
                    oy = (int64_t)floor(sets[s].p[i].y) - SH / 2,
                    ox = (int64_t)floor(sets[s].p[i].x) - SW / 2;
            oz = oz < 0 ? 0 : oz;
            oy = oy < 0 ? 0 : oy;
            ox = ox < 0 ? 0 : ox;
            int hits = 0;
            for (int j = i; j < sets[s].n; j++)
                if (!used[j] && inside(&sets[s].p[j], oz, oy, ox)) {
                    used[j] = 1;
                    hits++;
                }
            fprintf(out, "%lld %lld %lld %d %d\n", (long long)oz, (long long)oy, (long long)ox,
                sets[s].split, hits);
            boxes++;
        }
        free(used);
    }
}

static void gather_sets(int argc, char **argv, PointSet **sets, int *nsets) {
    double factor = num(opt(argc, argv, "--point-factor", "1"));
    double spacing = num(opt(argc, argv, "--spacing", "4"));
    const char *names[] = {"--same", "--relative", "--absolute"};
    for (int k = 0; k < 3; k++) {
        const char *f = opt(argc, argv, names[k], NULL);
        if (f)
            load_sets(f, k, (k + 1) * 1000000, factor, sets, nsets);
    }
    const char *lines = opt(argc, argv, "--lines", NULL);
    if (lines) {
        int n = 0, base = (int)num(opt(argc, argv, "--line-base", "5000000"));
        char **files = list_dir(lines, ".json", &n);
        for (int i = 0; i < n; i++) {
            char path[8192];
            snprintf(path, sizeof(path), "%s/%s", lines, files[i]);
            PointSet ps;
            if (load_line(path, base + i, factor, spacing, &ps) == 0 && ps.n) {
                *sets = realloc(*sets, sizeof(PointSet) * (size_t)(*nsets + 1));
                (*sets)[(*nsets)++] = ps;
            }
        }
    }
}

static int cmd_plan_points(int argc, char **argv) {
    PointSet *sets = NULL;
    int nsets = 0;
    gather_sets(argc, argv, &sets, &nsets);
    FILE *out = fopen(argv[2], "w");
    if (!out)
        die("cannot write %s", argv[2]);
    plan_boxes(sets, nsets, (int)num(opt(argc, argv, "--max-boxes", "1000000")), out);
    fclose(out);
    int total = 0;
    for (int s = 0; s < nsets; s++)
        total += sets[s].n;
    fprintf(stderr, "prep: %d sets, %d points\n", nsets, total);
    return 0;
}

static int cmd_points(int argc, char **argv) {
    const char *plan = argv[2], *fetched = argv[3], *outp = argv[4];
    Meta base = {0};
    common_meta(argc, argv, &base);
    base.ct_bits = 16;
    double cf = num(opt(argc, argv, "--context-factor", NULL));
    Vol8 ctx = npy_load(opt(argc, argv, "--context", NULL));
    Axis a =
        axis_load(opt(argc, argv, "--umbilicus", NULL), opt(argc, argv, "--umbilicus-format", "vc"),
            num(opt(argc, argv, "--umbilicus-factor", "1")));
    PointSet *sets = NULL;
    int nsets = 0;
    gather_sets(argc, argv, &sets, &nsets);
    FILE *pf = fopen(plan, "r");
    if (!pf)
        die("cannot read %s", plan);
    Shard s = shard_open(outp, LAYOUT_LABELLED);
    size_t sn = (size_t)SD * SH * SW;
    uint16_t *ct = grab(sn * 2), *inst = grab(sn * 2);
    uint8_t *sheet = grab(sn);
    memset(sheet, 255, sn);
    Point *pts = grab(sizeof(Point) * MAXPTS);
    long long oz, oy, ox;
    int split, hits, row = 0;
    while (fscanf(pf, "%lld %lld %lld %d %d", &oz, &oy, &ox, &split, &hits) == 5) {
        char path[8192];
        snprintf(path, sizeof(path), "%s/%06d.npy", fetched, row++);
        struct stat st;
        if (stat(path, &st)) {
            fprintf(stderr, "prep: missing %s\n", path);
            continue;
        }
        Vol8 box = npy_load(path);
        if (box.shape[0] != SD || box.shape[1] != SH || box.shape[2] != SW) {
            free(box.data);
            continue;
        }
        for (size_t i = 0; i < sn; i++)
            ct[i] = box.data[i];
        free(box.data);
        uint32_t np = 0;
        for (int q = 0; q < nsets && np < MAXPTS; q++) {
            if (sets[q].split != split)
                continue;
            for (int i = 0; i < sets[q].n && np < MAXPTS; i++)
                if (inside(&sets[q].p[i], oz, oy, ox)) {
                    pts[np] = sets[q].p[i];
                    pts[np].z -= (float)oz;
                    pts[np].y -= (float)oy;
                    pts[np].x -= (float)ox;
                    np++;
                }
        }
        if (np == 0)
            continue;
        Meta m = base;
        m.split = (uint32_t)split;
        m.ct_bits = 8;
        m.oz = (int32_t)oz;
        m.oy = (int32_t)oy;
        m.ox = (int32_t)ox;
        write_labelled(&s, m, &ctx, cf, &a, ct, sheet, inst, pts, np);
    }
    fclose(pf);
    shard_close(&s);
    return 0;
}

static int cmd_plan_unlabelled(int argc, char **argv) {
    Vol8 ctx = npy_load(argv[2]);
    double cf = num(opt(argc, argv, "--context-factor", NULL));
    int count = (int)num(opt(argc, argv, "--count", "8"));
    int64_t bz = (int64_t)num(opt(argc, argv, "--box-z", "512")),
            byx = (int64_t)num(opt(argc, argv, "--box-yx", "1024"));
    unsigned seed = (unsigned)num(opt(argc, argv, "--seed", "1"));
    int64_t ez0 = (int64_t)num(opt(argc, argv, "--exclude-z0", "-1")),
            ez1 = (int64_t)num(opt(argc, argv, "--exclude-z1", "-1"));
    double min_fill = num(opt(argc, argv, "--min-fill", "0.5"));
    int64_t zmax = (int64_t)(ctx.shape[0] * cf), ymax = (int64_t)(ctx.shape[1] * cf),
            xmax = (int64_t)(ctx.shape[2] * cf);
    int made = 0;
    int64_t *taken = grab(sizeof(int64_t) * 3 * (size_t)(count + 1));
    for (int tries = 0; tries < count * 400 && made < count; tries++) {
        seed = seed * 1664525u + 1013904223u;
        int64_t oz = (int64_t)((seed >> 8) % (uint64_t)(zmax > bz ? zmax - bz : 1));
        seed = seed * 1664525u + 1013904223u;
        int64_t oy = (int64_t)((seed >> 8) % (uint64_t)(ymax > byx ? ymax - byx : 1));
        seed = seed * 1664525u + 1013904223u;
        int64_t ox = (int64_t)((seed >> 8) % (uint64_t)(xmax > byx ? xmax - byx : 1));
        oz = oz / SD * SD;
        oy = oy / SH * SH;
        ox = ox / SW * SW;
        if (ez0 >= 0 && oz < ez1 && oz + bz > ez0)
            continue;
        int seen = 0;
        for (int k = 0; k < made && !seen; k++)
            seen = taken[3 * k] == oz && taken[3 * k + 1] == oy && taken[3 * k + 2] == ox;
        if (seen)
            continue;
        int64_t filled = 0, total = 0;
        for (int64_t z = (int64_t)(oz / cf); z < (int64_t)((oz + bz) / cf) && z < ctx.shape[0]; z++)
            for (int64_t y = (int64_t)(oy / cf); y < (int64_t)((oy + byx) / cf) && y < ctx.shape[1];
                y++)
                for (int64_t x = (int64_t)(ox / cf);
                    x < (int64_t)((ox + byx) / cf) && x < ctx.shape[2]; x++) {
                    filled += ctx.data[(z * ctx.shape[1] + y) * ctx.shape[2] + x] >= 16;
                    total++;
                }
        if (!total || (double)filled < min_fill * (double)total)
            continue;
        printf("%lld:%lld,%lld:%lld,%lld:%lld\n", (long long)oz, (long long)(oz + bz),
            (long long)oy, (long long)(oy + byx), (long long)ox, (long long)(ox + byx));
        taken[3 * made] = oz;
        taken[3 * made + 1] = oy;
        taken[3 * made + 2] = ox;
        made++;
    }
    fprintf(stderr, "prep: %d boxes\n", made);
    return 0;
}

static int cmd_unlabelled(int argc, char **argv) {
    Vol8 big = npy_load(argv[2]);
    const char *outp = argv[3];
    long long oz, oy, ox;
    if (sscanf(opt(argc, argv, "--origin", ""), "%lld,%lld,%lld", &oz, &oy, &ox) != 3)
        die("--origin z,y,x is required%s", "");
    Meta base = {0};
    common_meta(argc, argv, &base);
    base.ct_bits = 8;
    double cf = num(opt(argc, argv, "--context-factor", NULL));
    Vol8 ctx = npy_load(opt(argc, argv, "--context", NULL));
    Axis a =
        axis_load(opt(argc, argv, "--umbilicus", NULL), opt(argc, argv, "--umbilicus-format", "vc"),
            num(opt(argc, argv, "--umbilicus-factor", "1")));
    double min_fill = num(opt(argc, argv, "--min-fill", "0.3"));
    base.split = (uint32_t)num(opt(argc, argv, "--split", "0"));
    Shard s = shard_open(outp, LAYOUT_UNLABELLED);
    size_t sn = (size_t)SD * SH * SW;
    uint8_t *ct = grab(sn);
    for (int64_t bz = 0; bz + SD <= big.shape[0]; bz += SD)
        for (int64_t by = 0; by + SH <= big.shape[1]; by += SH)
            for (int64_t bx = 0; bx + SW <= big.shape[2]; bx += SW) {
                int64_t filled = 0;
                for (int64_t z = 0; z < SD; z++)
                    for (int64_t y = 0; y < SH; y++) {
                        const uint8_t *src = big.data +
                            ((size_t)(bz + z) * big.shape[1] + (by + y)) * big.shape[2] + bx;
                        memcpy(ct + ((size_t)z * SH + y) * SW, src, SW);
                        for (int64_t x = 0; x < SW; x++)
                            filled += src[x] >= 16;
                    }
                if ((double)filled < min_fill * (double)sn)
                    continue;
                Meta m = base;
                m.oz = (int32_t)(oz + bz);
                m.oy = (int32_t)(oy + by);
                m.ox = (int32_t)(ox + bx);
                write_unlabelled(&s, m, &ctx, cf, &a, ct);
            }
    shard_close(&s);
    return 0;
}

static int cmd_join(int argc, char **argv) {
    Shard out = {0};
    int layout = -1;
    static unsigned char buf[1 << 20];
    for (int i = 3; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        unsigned char head[64];
        if (!f || fread(head, 1, 64, f) != 64 || memcmp(head, "LGSF", 4))
            die("not a shard: %s", argv[i]);
        uint32_t lay;
        memcpy(&lay, head + 4, 4);
        uint64_t count;
        memcpy(&count, head + 8, 8);
        if (layout < 0) {
            layout = (int)lay;
            out = shard_open(argv[2], layout);
        } else if ((int)lay != layout)
            die("shards of different layouts: %s", argv[i]);
        size_t got;
        while ((got = fread(buf, 1, sizeof(buf), f)) > 0)
            fwrite(buf, 1, got, out.f);
        fclose(f);
        out.count += count;
    }
    if (layout < 0)
        out = shard_open(argv[2], LAYOUT_UNLABELLED);
    shard_close(&out);
    return 0;
}

static int cmp_u64(const void *a, const void *b) {
    unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b;
    return (x > y) - (x < y);
}

static int cmd_kaggle(int argc, char **argv) {
    const char *dir = argv[2], *outp = argv[3];
    FILE *f = fopen(opt(argc, argv, "--csv", NULL), "r");
    if (!f)
        die("cannot read %s", "--csv");
    int every = (int)num(opt(argc, argv, "--holdout-every", "10"));
    unsigned long long *rows = NULL;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        unsigned long long id, scroll;
        if (sscanf(line, "%llu,%llu", &id, &scroll) != 2)
            continue;
        rows = realloc(rows, sizeof(unsigned long long) * 2 * (size_t)(n + 1));
        rows[2 * n] = id;
        rows[2 * n + 1] = scroll;
        ++n;
    }
    fclose(f);
    qsort(rows, (size_t)n, 2 * sizeof(unsigned long long), cmp_u64);
    Shard s = shard_open(outp, LAYOUT_SURFACE);
    size_t sn = (size_t)SD * SH * SW;
    uint8_t *ct = grab(sn), *lab = grab(sn);
    long kept = 0, skipped = 0;
    for (int c = 0; c < n; ++c) {
        char path[8192];
        snprintf(path, sizeof(path), "%s/train_images/%llu.tif", dir, rows[2 * c]);
        struct stat st;
        if (stat(path, &st)) {
            fprintf(stderr, "prep: missing %s\n", path);
            continue;
        }
        Vol16 img = tiff_load(path);
        snprintf(path, sizeof(path), "%s/train_labels/%llu.tif", dir, rows[2 * c]);
        Vol16 lb = tiff_load(path);
        if (memcmp(img.shape, lb.shape, sizeof(img.shape)))
            die("Kaggle image and label differ in shape: %s", path);
        int minv = 65535, maxv = 0;
        size_t total_vox = (size_t)(img.shape[0] * img.shape[1] * img.shape[2]);
        for (size_t i = 0; i < total_vox; ++i) {
            if (img.data[i] < minv)
                minv = img.data[i];
            if (img.data[i] > maxv)
                maxv = img.data[i];
        }
        int span = maxv > minv ? maxv - minv : 1;
        int rescale = maxv > 255;
        int64_t ny = img.shape[1] / SH, nx = img.shape[2] / SW;
        int64_t y0s = (img.shape[1] - ny * SH) / 2, x0s = (img.shape[2] - nx * SW) / 2;
        Meta m = {0};
        m.source = 7;
        m.split = (uint32_t)(c % every == every - 1);
        m.curl = 1;
        m.ct_bits = 8;
        m.voxel_um = 8.f;
        m.confidence = (float)num(opt(argc, argv, "--confidence", "1"));
        m.oz = (int32_t)rows[2 * c + 1];
        m.oy = (int32_t)(rows[2 * c] & 0x7fffffff);
        m.cscale = 1.f;
        for (int64_t bz = 0; bz + SD <= img.shape[0]; bz += SD)
            for (int64_t by = 0; by < ny; ++by)
                for (int64_t bx = 0; bx < nx; ++bx) {
                    long known = 0;
                    for (int64_t z = 0; z < SD; ++z)
                        for (int64_t y = 0; y < SH; ++y)
                            for (int64_t x = 0; x < SW; ++x) {
                                size_t g = ((size_t)(bz + z) * img.shape[1] + y0s + by * SH + y) *
                                        img.shape[2] +
                                    x0s + bx * SW + x;
                                size_t i = ((size_t)z * SH + y) * SW + x;
                                ct[i] = rescale ? (uint8_t)(((uint32_t)(img.data[g] - minv) * 255u +
                                                                (uint32_t)(span / 2)) /
                                                      (uint32_t)span)
                                                : (uint8_t)img.data[g];
                                uint16_t v = lb.data[g];
                                lab[i] = v == 0 ? 0 : v == 1 ? 1 : 255;
                                known += v <= 1;
                            }
                    if (known < 4096) {
                        ++skipped;
                        continue;
                    }
                    m.ox = (int32_t)(bz * 10000 + by * 100 + bx);
                    write_header(&s, &m, 0);
                    fwrite(ct, 1, sn, s.f);
                    fwrite(lab, 1, sn, s.f);
                    s.count++;
                    ++kept;
                }
        free(img.data);
        free(lb.data);
        const char *del = opt(argc, argv, "--delete-raw", NULL);
        if (del && num(del) != 0) {
            snprintf(path, sizeof(path), "%s/train_images/%llu.tif", dir, rows[2 * c]);
            remove(path);
        }
    }
    shard_close(&s);
    fprintf(
        stderr, "prep: %d volumes, %ld slabs kept, %ld without labels skipped\n", n, kept, skipped);
    return 0;
}

static int usage(void) {
    fprintf(stderr,
        "usage: prep instance DIR OUT.shard --context CTX.npy --context-factor F --umbilicus FILE "
        "[--umbilicus-format vc|zyx] [--umbilicus-factor F] --curl C --voxel UM --source ID "
        "[--confidence C] "
        "[--holdout-every 10]\n"
        "       prep fibre DATASET_DIR OUT.shard (same options) [--prefix s1_]\n"
        "       prep plan-points PLAN.txt [--same F] [--relative F] [--absolute F] [--lines DIR "
        "--line-base N] "
        "[--point-factor F] [--spacing 4] [--max-boxes N]\n"
        "       prep points PLAN.txt FETCHED_DIR OUT.shard (the same point options and the "
        "instance options)\n"
        "       prep plan-unlabelled CTX.npy --context-factor F [--count 8] [--box-z 512] "
        "[--box-yx 1024] "
        "[--seed S] [--exclude-z0 Z --exclude-z1 Z] [--min-fill 0.5]\n"
        "       prep unlabelled BOX.npy OUT.shard --origin z,y,x (the instance options) "
        "[--min-fill 0.3] [--split 0|1]\n"
        "       prep join OUT.shard IN.shard ...\n"
        "       prep kaggle KAGGLE_DIR OUT.shard --csv train.csv [--holdout-every 10] "
        "[--confidence 1] "
        "[--delete-raw 1]\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc < 3)
        return usage();
    if (!strcmp(argv[1], "instance") && argc >= 4)
        return cmd_instance(argc, argv);
    if (!strcmp(argv[1], "fibre") && argc >= 4)
        return cmd_fibre(argc, argv);
    if (!strcmp(argv[1], "plan-points"))
        return cmd_plan_points(argc, argv);
    if (!strcmp(argv[1], "points") && argc >= 5)
        return cmd_points(argc, argv);
    if (!strcmp(argv[1], "plan-unlabelled"))
        return cmd_plan_unlabelled(argc, argv);
    if (!strcmp(argv[1], "kaggle") && argc >= 4)
        return cmd_kaggle(argc, argv);
    if (!strcmp(argv[1], "join"))
        return cmd_join(argc, argv);
    if (!strcmp(argv[1], "unlabelled") && argc >= 4)
        return cmd_unlabelled(argc, argv);
    return usage();
}
