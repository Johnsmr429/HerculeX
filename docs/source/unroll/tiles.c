#include <fcntl.h>
#include <json.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <unistd.h>

#define ZA 4720
#define AXIAL 1024
#define ZOOM_VOX 448
#define ZOOM 896
#define SIDE_SCALE 4.0
#define SIDE_Z0 2704
#define SIDE_Z1 6160
#define BAND 4.0

static const float WIND[6][3] = {{142, 108, 255}, {255, 176, 32}, {47, 184, 198}, {255, 107, 107},
    {78, 139, 255}, {123, 203, 76}};
static const int SIDE_UNITS[6] = {2608, 3184, 3760, 4336, 4912, 5488};

typedef struct {
    float x, y, z, nx, ny, nz, score, cw, sw, spacing, theta;
    int32_t cell, count, comp, winding;
} Surfel;

typedef struct {
    double z, x, y;
} Knot;

typedef struct {
    float px, py, nx, ny;
    int t;
} Pt;

typedef struct {
    Pt *p;
    size_t n, cap;
} Pts;

static void push(Pts *a, Pt p) {
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 4096;
        a->p = realloc(a->p, a->cap * sizeof(Pt));
        if (!a->p) {
            fprintf(stderr, "tiles: out of memory\n");
            exit(1);
        }
    }
    a->p[a->n++] = p;
}

typedef struct {
    int w, h;
    float *rgb;
} Image;

typedef struct {
    int shape[3], brick;
    int64_t *off;
    int nb[3];
    const uint8_t *raw;
} Atlas;

static Knot *knots;
static int nknots;
static Atlas atlas;
static double theta0 = 0;

static void die(const char *what, const char *path) {
    fprintf(stderr, "tiles: %s %s\n", what, path);
    exit(1);
}

static void *must(void *p) {
    if (!p)
        die("out of memory", "");
    return p;
}

static void axis_at(double z, double *x, double *y) {
    int i = 0;
    while (i < nknots - 2 && knots[i + 1].z < z)
        i++;
    double t = (z - knots[i].z) / (knots[i + 1].z - knots[i].z);
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    *x = knots[i].x + t * (knots[i + 1].x - knots[i].x);
    *y = knots[i].y + t * (knots[i + 1].y - knots[i].y);
}

static void load_axis(const char *path) {
    json_object *root = json_object_from_file(path);
    if (!root || !json_object_is_type(root, json_type_array))
        die("cannot read axis", path);
    nknots = (int)json_object_array_length(root);
    knots = must(calloc(nknots, sizeof(Knot)));
    for (int i = 0; i < nknots; i++) {
        json_object *row = json_object_array_get_idx(root, i);
        knots[i].z = json_object_get_double(json_object_array_get_idx(row, 0));
        knots[i].x = json_object_get_double(json_object_array_get_idx(row, 1));
        knots[i].y = json_object_get_double(json_object_array_get_idx(row, 2));
    }
    json_object_put(root);
}

static void load_atlas(const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/meta.json", dir);
    json_object *meta = json_object_from_file(path), *v;
    if (!meta || !json_object_object_get_ex(meta, "shape", &v))
        die("cannot read atlas meta", path);
    for (int k = 0; k < 3; k++)
        atlas.shape[k] = json_object_get_int(json_object_array_get_idx(v, k));
    json_object_object_get_ex(meta, "brick", &v);
    atlas.brick = json_object_get_int(json_object_array_get_idx(v, 0));
    json_object_put(meta);
    snprintf(path, sizeof path, "%s/offsets.npy", dir);
    FILE *f = fopen(path, "rb");
    uint8_t pre[10];
    if (!f || fread(pre, 1, 10, f) != 10)
        die("cannot read", path);
    int hl = pre[8] | pre[9] << 8;
    char *hdr = must(calloc(hl + 1, 1));
    if ((int)fread(hdr, 1, hl, f) != hl || !strstr(hdr, "<i8"))
        die("expected int64 offsets", path);
    char *sp = strstr(hdr, "'shape': (");
    if (!sp || sscanf(sp, "'shape': (%d, %d, %d)", &atlas.nb[0], &atlas.nb[1], &atlas.nb[2]) != 3)
        die("expected 3D offsets", path);
    free(hdr);
    size_t n = (size_t)atlas.nb[0] * atlas.nb[1] * atlas.nb[2];
    atlas.off = must(malloc(n * 8));
    if (fread(atlas.off, 8, n, f) != n)
        die("short offsets", path);
    fclose(f);
    snprintf(path, sizeof path, "%s/bricks.bin", dir);
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st))
        die("cannot open", path);
    atlas.raw = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (atlas.raw == MAP_FAILED)
        die("cannot map", path);
    close(fd);
}

static float atlas_voxel(int z, int y, int x) {
    if (z < 0 || y < 0 || x < 0 || z >= atlas.shape[0] || y >= atlas.shape[1] ||
        x >= atlas.shape[2])
        return 0;
    int b = atlas.brick;
    int64_t o = atlas.off[((size_t)(z / b) * atlas.nb[1] + y / b) * atlas.nb[2] + x / b];
    if (o < 0)
        return 0;
    return atlas.raw[o + ((size_t)(z % b) * b + y % b) * b + x % b];
}

static float atlas_at(double x, double y, double z) {
    double ax = x / 8 - 0.5, ay = y / 8 - 0.5;
    int az = (int)lround(z / 8), x0 = (int)floor(ax), y0 = (int)floor(ay);
    double fx = ax - x0, fy = ay - y0;
    return (
        float)((1 - fy) * ((1 - fx) * atlas_voxel(az, y0, x0) + fx * atlas_voxel(az, y0, x0 + 1)) +
        fy * ((1 - fx) * atlas_voxel(az, y0 + 1, x0) + fx * atlas_voxel(az, y0 + 1, x0 + 1)));
}

static Image image(int w, int h) {
    Image im = {w, h, must(calloc((size_t)w * h * 3, sizeof(float)))};
    return im;
}

static const float RAMP[6][3] = {
    {94, 60, 200}, {78, 139, 255}, {47, 184, 198}, {123, 203, 76}, {255, 176, 32}, {255, 107, 107}};
static double tlo = 6, thi = 87;

static void turn_colour(int t, float *c) {
    double v = (t - tlo) / (thi - tlo);
    v = (v < 0 ? 0 : v > 1 ? 1 : v) * 5;
    int i = v >= 5 ? 4 : (int)v;
    double f = v - i;
    for (int k = 0; k < 3; k++) {
        c[k] = (float)(RAMP[i][k] * (1 - f) + RAMP[i + 1][k] * f);
        if (t & 1)
            c[k] = c[k] * 0.62f;
    }
}

static void dot(Image *im, double px, double py, double r, const float *c) {
    for (int y = (int)floor(py - r); y <= (int)ceil(py + r); y++)
        for (int x = (int)floor(px - r); x <= (int)ceil(px + r); x++) {
            if (x < 0 || y < 0 || x >= im->w || y >= im->h)
                continue;
            double d = hypot(x + 0.5 - px, y + 0.5 - py), a = r + 0.5 - d;
            a = a < 0 ? 0 : a > 1 ? 1 : a;
            float *p = im->rgb + ((size_t)y * im->w + x) * 3;
            for (int k = 0; k < 3; k++)
                p[k] = (float)(p[k] * (1 - a) + c[k] * a);
        }
}

static void line(Image *im, double x0, double y0, double x1, double y1, double w, const float *c) {
    double len = hypot(x1 - x0, y1 - y0);
    int n = (int)ceil(len * 2) + 1;
    for (int i = 0; i <= n; i++) {
        double t = (double)i / n;
        dot(im, x0 + t * (x1 - x0), y0 + t * (y1 - y0), w / 2, c);
    }
}

static void write_bmp(const char *dir, const char *name, const Image *im) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.bmp", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f)
        die("cannot write", path);
    int row = (im->w * 3 + 3) & ~3;
    uint8_t hd[54] = {'B', 'M'};
    uint32_t size = 54 + (uint32_t)row * im->h, off = 54, dib = 40;
    memcpy(hd + 2, &size, 4);
    memcpy(hd + 10, &off, 4);
    memcpy(hd + 14, &dib, 4);
    int32_t dims[2] = {im->w, im->h};
    memcpy(hd + 18, dims, 8);
    uint16_t planes[2] = {1, 24};
    memcpy(hd + 26, planes, 4);
    fwrite(hd, 1, 54, f);
    uint8_t *buf = must(calloc(row, 1));
    for (int y = im->h - 1; y >= 0; y--) {
        for (int x = 0; x < im->w; x++)
            for (int c = 0; c < 3; c++) {
                float v = im->rgb[((size_t)y * im->w + x) * 3 + c];
                buf[x * 3 + 2 - c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f);
            }
        fwrite(buf, 1, row, f);
    }
    free(buf);
    fclose(f);
}

typedef struct {
    double x0, y0, size;
} Box;

static void axial_background(Image *im, Box b, float gain) {
    double s = b.size / im->w;
    for (int y = 0; y < im->h; y++)
        for (int x = 0; x < im->w; x++) {
            float v = atlas_at(b.x0 + (x + 0.5) * s, b.y0 + (y + 0.5) * s, ZA) * gain;
            float *p = im->rgb + ((size_t)y * im->w + x) * 3;
            p[0] = p[1] = p[2] = v;
        }
}

static void side_frame(double z, double *ax, double *ay, double *ux, double *uy) {
    axis_at(z, ax, ay);
    *ux = cos(theta0);
    *uy = sin(theta0);
}

static void side_background(Image *im, float gain) {
    for (int y = 0; y < im->h; y++) {
        double z = SIDE_Z0 + (y + 0.5) * SIDE_SCALE, ax, ay, ux, uy;
        side_frame(z, &ax, &ay, &ux, &uy);
        for (int x = 0; x < im->w; x++) {
            double s = (x + 0.5) * SIDE_SCALE;
            float v = atlas_at(ax + s * ux, ay + s * uy, z) * gain;
            float *p = im->rgb + ((size_t)y * im->w + x) * 3;
            p[0] = p[1] = p[2] = v;
        }
    }
}

static int side_place(double x, double y, double z, double *px, double *py) {
    if (z < SIDE_Z0 || z >= SIDE_Z1)
        return 0;
    double ax, ay, ux, uy;
    side_frame(z, &ax, &ay, &ux, &uy);
    double dx = x - ax, dy = y - ay, s = dx * ux + dy * uy, d = -dx * uy + dy * ux;
    if (s <= 0 || fabs(d) > BAND)
        return 0;
    *px = s / SIDE_SCALE;
    *py = (z - SIDE_Z0) / SIDE_SCALE;
    return 1;
}

typedef void (*SurfelFn)(const Surfel *s, void *ctx);

static void each_surfel(const char *work, int z0, SurfelFn fn, void *ctx) {
    char path[1024];
    snprintf(path, sizeof path, "%s/graphs/unit_z%05d_%05d/surfels.bin", work, z0, z0 + 768);
    FILE *f = fopen(path, "rb");
    if (!f)
        die("missing unit graph", path);
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2)
        die("short graph", path);
    const size_t chunk = 1 << 20;
    Surfel *buf = must(malloc(chunk * sizeof(Surfel)));
    for (int64_t left = hdr[0]; left > 0;) {
        size_t n = left < (int64_t)chunk ? (size_t)left : chunk;
        if (fread(buf, sizeof(Surfel), n, f) != n)
            die("short graph", path);
        for (size_t i = 0; i < n; i++)
            fn(buf + i, ctx);
        left -= (int64_t)n;
    }
    free(buf);
    fclose(f);
}

typedef struct {
    Pts *axial, *zoom;
    Box ab, zb;
    long *density;
    int grid;
} UnitAxial;

static void unit_axial(const Surfel *s, void *ctx) {
    UnitAxial *u = ctx;
    if (fabsf(s->z - ZA) > BAND)
        return;
    double k = AXIAL / u->ab.size;
    push(u->axial,
        (Pt){(float)((s->x - u->ab.x0) * k), (float)((s->y - u->ab.y0) * k), 0, 0, s->winding});
    if (u->density) {
        int gx = (int)((s->x - u->ab.x0) / 64), gy = (int)((s->y - u->ab.y0) / 64);
        if (gx >= 0 && gy >= 0 && gx < u->grid && gy < u->grid)
            u->density[gy * u->grid + gx]++;
        return;
    }
    double zk = ZOOM / u->zb.size, px = (s->x - u->zb.x0) * zk, py = (s->y - u->zb.y0) * zk;
    double nl = hypot(s->nx, s->ny);
    if (px < -40 || py < -40 || px > ZOOM + 40 || py > ZOOM + 40 || nl < 0.3)
        return;
    push(u->zoom, (Pt){(float)px, (float)py, (float)(s->nx / nl), (float)(s->ny / nl), s->winding});
}

typedef struct {
    Pts *side;
    int lo, hi;
} UnitSide;

static void unit_side(const Surfel *s, void *ctx) {
    UnitSide *u = ctx;
    if (s->z < u->lo || s->z >= u->hi)
        return;
    double px, py;
    uint32_t h = ((uint32_t)s->comp * 2654435761u) ^ ((uint32_t)u->lo * 40503u);
    if (side_place(s->x, s->y, s->z, &px, &py))
        push(u->side, (Pt){(float)px, (float)py, 0, 0, (int)(h >> 4)});
}

static void draw_dots(Image *im, const Pts *a, double r, int categorical) {
    for (size_t i = 0; i < a->n; i++) {
        float c[3];
        if (categorical) {
            int t = a->p[i].t;
            for (int k = 0; k < 3; k++)
                c[k] = WIND[t % 6][k] * ((t / 6) % 3 == 0 ? 0.6f : 1.0f);
        } else
            turn_colour(a->p[i].t, c);
        dot(im, a->p[i].px, a->p[i].py, r, c);
    }
}

static int cmpi(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void turn_range(const Pts *a, double *lo, double *hi) {
    int *t = must(malloc(a->n * sizeof(int)));
    for (size_t i = 0; i < a->n; i++)
        t[i] = a->p[i].t;
    qsort(t, a->n, sizeof(int), cmpi);
    *lo = t[a->n / 100];
    *hi = t[a->n - 1 - a->n / 100];
    free(t);
}

static void draw_zoom(Image *im, const Pts *a) {
    const float white[3] = {245, 245, 247};
    double zk = ZOOM / (double)ZOOM_VOX;
    for (size_t i = 0; i < a->n; i++) {
        const Pt *p = a->p + i;
        float c[3];
        turn_colour(p->t, c);
        double tx = -p->ny, ty = p->nx, h = 4.6 * zk;
        line(im, p->px - tx * h, p->py - ty * h, p->px + tx * h, p->py + ty * h, 2.8, c);
    }
    for (size_t i = 0; i < a->n; i += 9) {
        const Pt *p = a->p + i;
        double ex = p->px + p->nx * 8 * zk, ey = p->py + p->ny * 8 * zk;
        line(im, p->px, p->py, ex, ey, 1.2, white);
        dot(im, ex, ey, 1.7, white);
    }
}

static float *read_tif(const char *path, uint32_t *w, uint32_t *h) {
    TIFF *t = TIFFOpen(path, "r");
    if (!t)
        die("cannot open", path);
    TIFFGetField(t, TIFFTAG_IMAGEWIDTH, w);
    TIFFGetField(t, TIFFTAG_IMAGELENGTH, h);
    float *a = must(malloc((size_t)*w * *h * 4));
    for (uint32_t r = 0; r < *h; r++)
        if (TIFFReadScanline(t, a + (size_t)r * *w, r, 0) < 0)
            die("cannot read", path);
    TIFFClose(t);
    return a;
}

static double bbox(json_object *p, int corner, int axis) {
    json_object *b;
    json_object_object_get_ex(p, "bbox", &b);
    return json_object_get_double(
        json_object_array_get_idx(json_object_array_get_idx(b, corner), axis));
}

static int band_hits_box(double x0, double y0, double x1, double y1, double margin) {
    double ax, ay, ux = cos(theta0), uy = sin(theta0);
    axis_at((SIDE_Z0 + SIDE_Z1) / 2.0, &ax, &ay);
    double cx[4] = {x0, x1, x0, x1}, cy[4] = {y0, y0, y1, y1}, dmin = 1e30, dmax = -1e30,
           smax = -1e30;
    for (int k = 0; k < 4; k++) {
        double dx = cx[k] - ax, dy = cy[k] - ay, d = -dx * uy + dy * ux, s = dx * ux + dy * uy;
        dmin = d < dmin ? d : dmin;
        dmax = d > dmax ? d : dmax;
        smax = s > smax ? s : smax;
    }
    return dmin <= margin && dmax >= -margin && smax > -margin;
}

static void pieces(const char *runs, Pts *axial, Box ab, Pts *side) {
    char path[1024];
    snprintf(path, sizeof path, "%s/pieces/record.json", runs);
    json_object *rec = json_object_from_file(path), *list;
    if (!rec || !json_object_object_get_ex(rec, "pieces", &list))
        die("cannot read", path);
    size_t n = json_object_array_length(list);
    double k = AXIAL / ab.size;
    for (size_t i = 0; i < n; i++) {
        json_object *p = json_object_array_get_idx(list, i), *v;
        double zlo = bbox(p, 0, 2), zhi = bbox(p, 1, 2);
        int for_axial = zlo <= ZA + BAND && zhi >= ZA - BAND;
        int for_side = zlo < SIDE_Z1 && zhi >= SIDE_Z0 &&
            band_hits_box(bbox(p, 0, 0), bbox(p, 0, 1), bbox(p, 1, 0), bbox(p, 1, 1), 160);
        if (!for_axial && !for_side)
            continue;
        json_object_object_get_ex(p, "uuid", &v);
        const char *uuid = json_object_get_string(v);
        json_object_object_get_ex(p, "winding", &v);
        int turn = json_object_get_int(v);
        uint32_t w, h;
        float *g[3];
        const char *names[3] = {"x", "y", "z"};
        for (int a = 0; a < 3; a++) {
            snprintf(path, sizeof path, "%s/pieces/%s/%s.tif", runs, uuid, names[a]);
            g[a] = read_tif(path, &w, &h);
        }
        for (size_t j = 0; j < (size_t)w * h; j++) {
            float x = g[0][j], y = g[1][j], z = g[2][j];
            if (x < 0 || y < 0 || z < 0)
                continue;
            if (for_axial && fabsf(z - ZA) <= BAND)
                push(axial, (Pt){(float)((x - ab.x0) * k), (float)((y - ab.y0) * k), 0, 0, turn});
            double px, py;
            if (for_side && side_place(x, y, z, &px, &py))
                push(side, (Pt){(float)px, (float)py, 0, 0, turn});
        }
        for (int a = 0; a < 3; a++)
            free(g[a]);
    }
    json_object_put(rec);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: tiles WORK RUNS ATLAS OUT\n");
        return 2;
    }
    const char *work = argv[1], *runs = argv[2], *out = argv[4];
    char path[1024];
    snprintf(path, sizeof path, "%s/axis.json", work);
    load_axis(path);
    load_atlas(argv[3]);
    double ax, ay;
    axis_at(ZA, &ax, &ay);
    Box ab = {ax - 3000, ay - 3000, 6000};
    int grid = (int)(ab.size / 64);
    long *density = must(calloc((size_t)grid * grid, sizeof(long)));
    Pts axial_before = {0}, axial_after = {0}, zoom_pts = {0}, side_before = {0}, side_after = {0};
    UnitAxial ua = {&axial_before, &zoom_pts, ab, {0, 0, ZOOM_VOX}, density, grid};
    each_surfel(work, 4336, unit_axial, &ua);
    int win = ZOOM_VOX / 64, bx = 0, by = 0;
    long best = -1;
    for (int gy = 0; gy + win <= grid; gy++)
        for (int gx = 0; gx + win <= grid; gx++) {
            double cx = ab.x0 + (gx + win / 2.0) * 64 - ax, cy = ab.y0 + (gy + win / 2.0) * 64 - ay;
            if (hypot(cx, cy) < 900)
                continue;
            long sum = 0;
            for (int y = gy; y < gy + win; y++)
                for (int x = gx; x < gx + win; x++)
                    sum += density[y * grid + x];
            if (sum > best)
                best = sum, bx = gx, by = gy;
        }
    Box zb = {ab.x0 + bx * 64, ab.y0 + by * 64, ZOOM_VOX};
    axial_before.n = 0;
    ua.zb = zb;
    ua.density = NULL;
    each_surfel(work, 4336, unit_axial, &ua);
    for (int u = 0; u < 6; u++) {
        UnitSide us = {&side_before, SIDE_UNITS[u] + 96, SIDE_UNITS[u] + 672};
        each_surfel(work, SIDE_UNITS[u], unit_side, &us);
    }
    pieces(runs, &axial_after, ab, &side_after);
    int sw = (int)(3600 / SIDE_SCALE), sh = (int)((SIDE_Z1 - SIDE_Z0) / SIDE_SCALE);
    Image axial = image(AXIAL, AXIAL), axial_unit = image(AXIAL, AXIAL), zoom = image(ZOOM, ZOOM);
    Image before = image(sw, sh), after = image(sw, sh);
    axial_background(&axial, ab, 0.42f);
    axial_background(&axial_unit, ab, 0.42f);
    for (int y = 0; y < zoom.h; y++)
        for (int x = 0; x < zoom.w; x++) {
            double s = zb.size / zoom.w;
            float v = atlas_at(zb.x0 + (x + 0.5) * s, zb.y0 + (y + 0.5) * s, ZA) * 0.34f;
            float *p = zoom.rgb + ((size_t)y * zoom.w + x) * 3;
            p[0] = p[1] = p[2] = v;
        }
    side_background(&before, 0.42f);
    side_background(&after, 0.42f);
    draw_dots(&axial, &axial_after, 1.1, 0);
    draw_dots(&after, &side_after, 1.0, 0);
    draw_dots(&before, &side_before, 1.0, 1);
    double glo = tlo, ghi = thi;
    turn_range(&axial_before, &tlo, &thi);
    double ulo = tlo, uhi = thi;
    draw_dots(&axial_unit, &axial_before, 1.1, 0);
    draw_zoom(&zoom, &zoom_pts);
    write_bmp(out, "axial-turns", &axial);
    write_bmp(out, "axial-unit", &axial_unit);
    write_bmp(out, "zoom", &zoom);
    write_bmp(out, "side-before", &before);
    write_bmp(out, "side-after", &after);
    snprintf(path, sizeof path, "%s/tiles.js", out);
    FILE *js = fopen(path, "w");
    if (!js)
        die("cannot write", path);
    fprintf(js,
        "window.TILES = "
        "{\"za\":%d,\"axial\":{\"x0\":%.1f,\"y0\":%.1f,\"size\":%.1f,\"axis\":[%.1f,%.1f]},"
        "\"zoom\":{\"x0\":%.1f,\"y0\":%.1f,\"size\":%.1f},\"side\":{\"z0\":%d,\"z1\":%d,\"scale\":%"
        ".1f,"
        "\"w\":%d,\"h\":%d,\"theta\":%.4f,\"units\":[",
        ZA, ab.x0, ab.y0, ab.size, ax, ay, zb.x0, zb.y0, zb.size, SIDE_Z0, SIDE_Z1, SIDE_SCALE, sw,
        sh, theta0);
    for (int u = 0; u < 6; u++)
        fprintf(js, "%s[%d,%d]", u ? "," : "", SIDE_UNITS[u] + 96, SIDE_UNITS[u] + 672);
    fprintf(js,
        "]},\"turns\":[%.0f,%.0f],\"unit_turns\":[%.0f,%.0f],\"counts\":{\"axial\":%zu,\"unit\":%"
        "zu,\"zoom\":%zu,\"before\":%zu,"
        "\"after\":%zu}};\n",
        glo, ghi, ulo, uhi, axial_after.n, axial_before.n, zoom_pts.n, side_before.n, side_after.n);
    fclose(js);
    return 0;
}
