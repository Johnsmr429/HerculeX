#define _GNU_SOURCE
#include "parallel_blocks.h"
#ifdef TRACE_CUDA
#include "trace_cuda.h"
#endif

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <json-c/json.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tiffio.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    double z, x, y;
} HxAxisKnot;

typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} SheetNode;

void hx_centre_at(const HxAxisKnot *knots, int64_t count, double z, double *x, double *y);
int64_t hx_level_minimize(int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *k, const int32_t *up, const int32_t *down, int32_t *x, int32_t first_step,
    int64_t *log);
#ifdef HERCULEX_CUDA_FLOW
typedef struct {
    void *(*open)(int32_t, int32_t, const int32_t *, const void *);
    int64_t (*run)(void *, int32_t *, int64_t *);
    void (*close)(void *);
} HxDeviceFlow;
void hx_level_device(HxDeviceFlow *api);
void *hx_flow_open(int32_t, int32_t, const int32_t *, const void *);
int64_t hx_flow_run(void *, int32_t *, int64_t *);
void hx_flow_close(void *);
#endif
void hx_level_flatten(const SheetNode *nodes, const int32_t *order, const int64_t *piece_first,
    int32_t pieces, const int32_t *ea, const int32_t *eb, const int64_t *edge_first, int32_t *local,
    int8_t *sign, float *u, float *v, int32_t steps, double tolerance, double held,
    int32_t threads);
int64_t hx_raster_cells(int64_t height, int64_t width, int64_t nodes, const double *xyz,
    const double *normal, double *u, double *v, double step, double fold, int64_t *flat,
    double *grid, uint8_t *locked, double *spread);
int hx_centre_cells(int64_t height, int64_t width, int64_t nodes, const int64_t *flat,
    const double *du, const double *dv, const double *xyz, const uint8_t *take,
    const uint8_t *source, double step, int32_t slope_nodes, double slope_spread, double slope_low,
    double slope_high, double *grid);
int hx_mesh_isometry(const double *grid, int64_t height, int64_t width, const uint8_t *known,
    const uint8_t *locked, double stride, double *values);
int hx_relax_mesh(void *grid, int as_float, int64_t depth, int64_t height, int64_t width,
    const uint8_t *known, const uint8_t *free_cells, int64_t free_count, int32_t rounds);

static double g_skip_cost = 0.35;

static double clock_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int make_dirs(const char *path) {
    char *copy = strdup(path);
    if (!copy)
        return -1;
    for (char *p = copy + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = 0;
        if (mkdir(copy, 0775) && errno != EEXIST) {
            free(copy);
            return -1;
        }
        *p = '/';
    }
    int err = mkdir(copy, 0775) && errno != EEXIST;
    free(copy);
    return err ? -1 : 0;
}

typedef struct {
    void *map_base;
    size_t map_size;
    const void *data;
    int ndim;
    int64_t shape[6];
    int itemsize;
    char dtype;
} NpyArray;

static int map_npy(const char *path, NpyArray *out) {
    memset(out, 0, sizeof(*out));
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        close(fd);
        return -1;
    }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED)
        return -1;
    const uint8_t *b = base;
    if (memcmp(b, "\x93NUMPY", 6) != 0) {
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    size_t hlen = 0, hdr_off = 10;
    if (b[6] == 1) {
        hlen = (size_t)b[8] | ((size_t)b[9] << 8);
    } else {
        hlen = (size_t)b[8] | ((size_t)b[9] << 8) | ((size_t)b[10] << 16) | ((size_t)b[11] << 24);
        hdr_off = 12;
    }
    if (hdr_off + hlen > (size_t)st.st_size) {
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    char *hdr = malloc(hlen + 1);
    if (!hdr) {
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    memcpy(hdr, b + hdr_off, hlen);
    hdr[hlen] = 0;
    char *dpos = strstr(hdr, "'descr'");
    char *spos = strstr(hdr, "'shape'");
    if (!dpos || !spos) {
        free(hdr);
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    char *q1 = strchr(dpos + 7, '\'');
    char *q2 = q1 ? strchr(q1 + 1, '\'') : NULL;
    if (!q1 || !q2 || q2 - q1 < 3) {
        free(hdr);
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    out->dtype = q1[2];
    out->itemsize = atoi(q1 + 3);
    char *p1 = strchr(spos + 7, '(');
    char *p2 = p1 ? strchr(p1 + 1, ')') : NULL;
    if (!p1 || !p2) {
        free(hdr);
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    int nd = 0;
    char *cur = p1 + 1;
    size_t total = 1;
    while (cur < p2 && nd < 6) {
        while (cur < p2 && (*cur == ' ' || *cur == ','))
            cur++;
        if (cur >= p2)
            break;
        char *end = NULL;
        long long v = strtoll(cur, &end, 10);
        if (end == cur)
            break;
        out->shape[nd++] = (int64_t)v;
        total *= (size_t)v;
        cur = end;
    }
    out->ndim = nd;
    free(hdr);
    size_t bytes = total * (size_t)out->itemsize;
    if (hdr_off + hlen + bytes > (size_t)st.st_size) {
        munmap(base, (size_t)st.st_size);
        return -1;
    }
    out->map_base = base;
    out->map_size = (size_t)st.st_size;
    out->data = b + hdr_off + hlen;
    return 0;
}

static void unmap_npy(NpyArray *a) {
    if (a->map_base)
        munmap(a->map_base, a->map_size);
    memset(a, 0, sizeof(*a));
}

static int load_axis_json(const char *path, HxAxisKnot **knots_out, int64_t *count_out) {
    json_object *root = json_object_from_file(path);
    if (!root)
        return -1;
    json_object *arr = root;
    if (json_object_is_type(root, json_type_object)) {
        if (!json_object_object_get_ex(root, "knots_z_x_y", &arr)) {
            json_object_put(root);
            return -1;
        }
    }
    if (!arr || !json_object_is_type(arr, json_type_array)) {
        json_object_put(root);
        return -1;
    }
    size_t n = json_object_array_length(arr);
    HxAxisKnot *knots = calloc(n ? n : 1, sizeof(*knots));
    if (!knots) {
        json_object_put(root);
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        json_object *row = json_object_array_get_idx(arr, i);
        knots[i].z = json_object_get_double(json_object_array_get_idx(row, 0));
        knots[i].x = json_object_get_double(json_object_array_get_idx(row, 1));
        knots[i].y = json_object_get_double(json_object_array_get_idx(row, 2));
    }
    json_object_put(root);
    *knots_out = knots;
    *count_out = (int64_t)n;
    return n >= 2 ? 0 : -1;
}

static int write_npy_u8(const char *path, const uint8_t *data, int h, int w) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    char dict[128];
    int dlen = snprintf(
        dict, sizeof(dict), "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d), }", h, w);
    int pad = 16 - ((10 + dlen + 1) % 16);
    if (pad == 16)
        pad = 0;
    uint16_t hlen = (uint16_t)(dlen + pad + 1);
    uint8_t pre[10] = {
        0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (uint8_t)(hlen & 0xff), (uint8_t)(hlen >> 8)};
    fwrite(pre, 1, 10, f);
    fwrite(dict, 1, (size_t)dlen, f);
    for (int i = 0; i < pad; i++)
        fputc(' ', f);
    fputc('\n', f);
    fwrite(data, 1, (size_t)h * w, f);
    fclose(f);
    return 0;
}

static int write_tiff_f32(const char *path, const float *data, int h, int w) {
    TIFF *t = TIFFOpen(path, "w");
    if (!t)
        return -1;
    TIFFSetField(t, TIFFTAG_IMAGEWIDTH, (uint32_t)w);
    TIFFSetField(t, TIFFTAG_IMAGELENGTH, (uint32_t)h);
    TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(t, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
    for (int r = 0; r < h; r++) {
        if (TIFFWriteScanline(t, (void *)(data + (size_t)r * w), (uint32_t)r, 0) < 0) {
            TIFFClose(t);
            return -1;
        }
    }
    TIFFClose(t);
    return 0;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline float fused_score_at(
    const uint8_t *su8, const uint8_t *bu8, const int8_t *pi8, size_t vox, size_t idx) {
    (void)vox;
    float s = su8[idx] * (1.0f / 255.0f);
    if (s < 0.02f)
        return 0.0f;
    if (bu8)
        s *= fmaxf(0.0f, 1.0f - 0.80f * (bu8[idx] * (1.0f / 255.0f)));
    if (pi8) {
        float cw = pi8[idx] * (1.0f / 127.0f);
        s *= fmaxf(0.15f, 0.85f + 0.25f * cw);
    }
    return s;
}

static inline float sample_fused_trilinear(const uint8_t *su8, const uint8_t *bu8,
    const int8_t *pi8, int Z, int Y, int X, float z, float y, float x) {
    if (z < 0.0f)
        z = 0.0f;
    else if (z > Z - 1.0001f)
        z = Z - 1.0001f;
    if (y < 0.0f)
        y = 0.0f;
    else if (y > Y - 1.0001f)
        y = Y - 1.0001f;
    if (x < 0.0f)
        x = 0.0f;
    else if (x > X - 1.0001f)
        x = X - 1.0001f;
    int iz = (int)z, iy = (int)y, ix = (int)x;
    float fz = z - iz, fy = y - iy, fx = x - ix;
    int iz1 = iz + (iz + 1 < Z), iy1 = iy + (iy + 1 < Y), ix1 = ix + (ix + 1 < X);
    size_t pl = (size_t)Y * X, vox = (size_t)Z * pl;
    float s000 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz * pl + (size_t)iy * X + ix);
    float s001 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz * pl + (size_t)iy * X + ix1);
    float s010 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz * pl + (size_t)iy1 * X + ix);
    float s011 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz * pl + (size_t)iy1 * X + ix1);
    float s100 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz1 * pl + (size_t)iy * X + ix);
    float s101 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz1 * pl + (size_t)iy * X + ix1);
    float s110 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz1 * pl + (size_t)iy1 * X + ix);
    float s111 = fused_score_at(su8, bu8, pi8, vox, (size_t)iz1 * pl + (size_t)iy1 * X + ix1);
    float c00 = s000 * (1.0f - fx) + s001 * fx;
    float c01 = s010 * (1.0f - fx) + s011 * fx;
    float c10 = s100 * (1.0f - fx) + s101 * fx;
    float c11 = s110 * (1.0f - fx) + s111 * fx;
    float c0 = c00 * (1.0f - fy) + c01 * fy;
    float c1 = c10 * (1.0f - fy) + c11 * fy;
    return c0 * (1.0f - fz) + c1 * fz;
}

static inline float sample_u8_trilinear(
    const uint8_t *vol, int Z, int Y, int X, float z, float y, float x) {
    if (z < 0.0f)
        z = 0.0f;
    else if (z > Z - 1.0001f)
        z = Z - 1.0001f;
    if (y < 0.0f)
        y = 0.0f;
    else if (y > Y - 1.0001f)
        y = Y - 1.0001f;
    if (x < 0.0f)
        x = 0.0f;
    else if (x > X - 1.0001f)
        x = X - 1.0001f;
    int iz = (int)z, iy = (int)y, ix = (int)x;
    float fz = z - iz, fy = y - iy, fx = x - ix;
    int iz1 = iz + (iz + 1 < Z), iy1 = iy + (iy + 1 < Y), ix1 = ix + (ix + 1 < X);
    size_t pl = (size_t)Y * X;
    float s000 = vol[(size_t)iz * pl + (size_t)iy * X + ix];
    float s001 = vol[(size_t)iz * pl + (size_t)iy * X + ix1];
    float s010 = vol[(size_t)iz * pl + (size_t)iy1 * X + ix];
    float s011 = vol[(size_t)iz * pl + (size_t)iy1 * X + ix1];
    float s100 = vol[(size_t)iz1 * pl + (size_t)iy * X + ix];
    float s101 = vol[(size_t)iz1 * pl + (size_t)iy * X + ix1];
    float s110 = vol[(size_t)iz1 * pl + (size_t)iy1 * X + ix];
    float s111 = vol[(size_t)iz1 * pl + (size_t)iy1 * X + ix1];
    float c00 = s000 * (1.0f - fx) + s001 * fx;
    float c01 = s010 * (1.0f - fx) + s011 * fx;
    float c10 = s100 * (1.0f - fx) + s101 * fx;
    float c11 = s110 * (1.0f - fx) + s111 * fx;
    float c0 = c00 * (1.0f - fy) + c01 * fy;
    float c1 = c10 * (1.0f - fy) + c11 * fy;
    return (c0 * (1.0f - fz) + c1 * fz) * (1.0f / 255.0f);
}

static inline float sample_dir_aff(const uint8_t *au8, size_t vox, int Z, int Y, int X, float z,
    float y, float x, float tx, float ty, float tz) {
    float az = sample_u8_trilinear(au8, Z, Y, X, z, y, x);
    float ay = sample_u8_trilinear(au8 + vox, Z, Y, X, z, y, x);
    float ax = sample_u8_trilinear(au8 + 2 * vox, Z, Y, X, z, y, x);
    float denom = tx * tx + ty * ty + tz * tz + 1e-6f;
    return (tz * tz * az + ty * ty * ay + tx * tx * ax) / denom;
}

static inline float sample_centerline_score(const uint8_t *su8, const uint8_t *bu8,
    const int8_t *pi8, int Z, int Y, int X, float z, float y, float x, float nz, float ny,
    float nx) {
    static const float off[5] = {-1.5f, -0.75f, 0.0f, 0.75f, 1.5f};
    static const float wt[5] = {0.10f, 0.24f, 0.32f, 0.24f, 0.10f};
    float acc = 0.0f;
    for (int k = 0; k < 5; k++)
        acc += wt[k] *
            sample_fused_trilinear(
                su8, bu8, pi8, Z, Y, X, z + off[k] * nz, y + off[k] * ny, x + off[k] * nx);
    if (bu8)
        acc -= 0.75f * sample_u8_trilinear(bu8, Z, Y, X, z, y, x);
    return acc;
}

static int gaussian_1d_kernel(float sigma, float *k, int max_r) {
    int r = (int)ceilf(3.0f * sigma);
    if (r < 1)
        r = 1;
    if (r > max_r)
        r = max_r;
    double sum = 0.0;
    for (int i = -r; i <= r; i++) {
        double v = exp(-0.5 * (double)(i * i) / ((double)sigma * sigma));
        k[i + r] = (float)v;
        sum += v;
    }
    for (int i = 0; i <= 2 * r; i++)
        k[i] = (float)(k[i] / sum);
    return r;
}

typedef struct {
    const float *in;
    float *out;
    int Z, Y, X, axis, radius;
    float k[65];
} BlurCtx;

static void blur_axis_job(void *arg, size_t idx) {
    const BlurCtx *c = arg;
    int Z = c->Z, Y = c->Y, X = c->X, r = c->radius;
    const float *k = c->k;
    if (c->axis == 2) {
        int z = (int)idx;
        for (int y = 0; y < Y; y++) {
            const float *src = c->in + ((size_t)z * Y + y) * X;
            float *dst = c->out + ((size_t)z * Y + y) * X;
            for (int x = 0; x < X; x++) {
                float s = 0.0f;
                for (int d = -r; d <= r; d++)
                    s += k[d + r] * src[clampi(x + d, 0, X - 1)];
                dst[x] = s;
            }
        }
    } else if (c->axis == 1) {
        int z = (int)idx;
        const float *plane = c->in + (size_t)z * Y * X;
        float *outp = c->out + (size_t)z * Y * X;
        for (int y = 0; y < Y; y++) {
            float *dst = outp + (size_t)y * X;
            memset(dst, 0, (size_t)X * sizeof(float));
            for (int d = -r; d <= r; d++) {
                float w = k[d + r];
                const float *src = plane + (size_t)clampi(y + d, 0, Y - 1) * X;
                for (int x = 0; x < X; x++)
                    dst[x] += w * src[x];
            }
        }
    } else {
        int z = (int)idx;
        size_t plane = (size_t)Y * X;
        float *dst = c->out + (size_t)z * plane;
        memset(dst, 0, plane * sizeof(float));
        for (int d = -r; d <= r; d++) {
            float w = k[d + r];
            const float *src = c->in + (size_t)clampi(z + d, 0, Z - 1) * plane;
            for (size_t i = 0; i < plane; i++)
                dst[i] += w * src[i];
        }
    }
}

static void blur3d(const float *in, float *out, float *tmp, int Z, int Y, int X, float sigma) {
    BlurCtx ctx = {.Z = Z, .Y = Y, .X = X};
    ctx.radius = gaussian_1d_kernel(sigma, ctx.k, 32);
    ctx.in = in;
    ctx.out = tmp;
    ctx.axis = 2;
    each_block((size_t)Z, &ctx, blur_axis_job);
    ctx.in = tmp;
    ctx.out = out;
    ctx.axis = 1;
    each_block((size_t)Z, &ctx, blur_axis_job);
    ctx.in = out;
    ctx.out = tmp;
    ctx.axis = 0;
    each_block((size_t)Z, &ctx, blur_axis_job);
    memcpy(out, tmp, (size_t)Z * Y * X * sizeof(float));
}

static void jacobi3_top(
    double a00, double a11, double a22, double a01, double a02, double a12, float out[3]) {
    double m[3][3] = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int sw = 0; sw < 16; sw++) {
        double off = fabs(m[0][1]) + fabs(m[0][2]) + fabs(m[1][2]);
        if (off < 1e-18)
            break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++) {
                if (fabs(m[p][q]) < 1e-20)
                    continue;
                double th = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {
                    double x = m[k][p], y = m[k][q];
                    m[k][p] = c * x - s * y;
                    m[k][q] = s * x + c * y;
                }
                for (int k = 0; k < 3; k++) {
                    double x = m[p][k], y = m[q][k];
                    m[p][k] = c * x - s * y;
                    m[q][k] = s * x + c * y;
                }
                for (int k = 0; k < 3; k++) {
                    double x = v[k][p], y = v[k][q];
                    v[k][p] = c * x - s * y;
                    v[k][q] = s * x + c * y;
                }
            }
    }
    int top = 0;
    if (m[1][1] > m[top][top])
        top = 1;
    if (m[2][2] > m[top][top])
        top = 2;
    double nx = v[0][top], ny = v[1][top], nz = v[2][top];
    double l = sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 1e-12) {
        out[0] = (float)(nx / l);
        out[1] = (float)(ny / l);
        out[2] = (float)(nz / l);
    } else {
        out[0] = 1.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
    }
}

typedef struct {
    int Z, Y, X;
    int64_t z0, y0, x0;
    const float *s_sm;
    float *j[6];
    float *nx, *ny, *nz;
    const HxAxisKnot *axis;
    int64_t axis_count;
} NormalCtx;

static void tensor_grad_job(void *arg, size_t idx) {
    NormalCtx *c = arg;
    int z = (int)idx, Z = c->Z, Y = c->Y, X = c->X;
    size_t plane = (size_t)Y * X;
    int zm = z > 0 ? z - 1 : 0, zp = z + 1 < Z ? z + 1 : Z - 1;
    float zscale = (z > 0 && z + 1 < Z) ? 0.5f : 1.0f;
    for (int y = 0; y < Y; y++) {
        int ym = y > 0 ? y - 1 : 0, yp = y + 1 < Y ? y + 1 : Y - 1;
        float yscale = (y > 0 && y + 1 < Y) ? 0.5f : 1.0f;
        for (int x = 0; x < X; x++) {
            int xm = x > 0 ? x - 1 : 0, xp = x + 1 < X ? x + 1 : X - 1;
            float xscale = (x > 0 && x + 1 < X) ? 0.5f : 1.0f;
            size_t i = (size_t)z * plane + (size_t)y * X + x;
            float gz = (c->s_sm[(size_t)zp * plane + (size_t)y * X + x] -
                           c->s_sm[(size_t)zm * plane + (size_t)y * X + x]) *
                zscale;
            float gy = (c->s_sm[(size_t)z * plane + (size_t)yp * X + x] -
                           c->s_sm[(size_t)z * plane + (size_t)ym * X + x]) *
                yscale;
            float gx = (c->s_sm[(size_t)z * plane + (size_t)y * X + xp] -
                           c->s_sm[(size_t)z * plane + (size_t)y * X + xm]) *
                xscale;
            c->j[0][i] = gx * gx;
            c->j[1][i] = gy * gy;
            c->j[2][i] = gz * gz;
            c->j[3][i] = gx * gy;
            c->j[4][i] = gx * gz;
            c->j[5][i] = gy * gz;
        }
    }
}

static void tensor_eig_job(void *arg, size_t idx) {
    NormalCtx *c = arg;
    int z = (int)idx, Y = c->Y, X = c->X;
    size_t plane = (size_t)Y * X;
    double cx, cy;
    hx_centre_at(c->axis, c->axis_count, (double)(c->z0 + z), &cx, &cy);
    for (int y = 0; y < Y; y++) {
        double ry = (double)(c->y0 + y) - cy;
        for (int x = 0; x < X; x++) {
            size_t i = (size_t)z * plane + (size_t)y * X + x;
            float n[3];
            jacobi3_top(c->j[0][i], c->j[1][i], c->j[2][i], c->j[3][i], c->j[4][i], c->j[5][i], n);
            double rx = (double)(c->x0 + x) - cx;
            if (n[0] * rx + n[1] * ry < 0.0) {
                n[0] = -n[0];
                n[1] = -n[1];
                n[2] = -n[2];
            }
            c->nx[i] = n[0];
            c->ny[i] = n[1];
            c->nz[i] = n[2];
        }
    }
}

typedef struct {
    int Z, Y, X, hZ, hY, hX;
    int64_t z0, y0, x0;
    const float *s_sm;
    float *hs;
    float *hj[6];
    float *hnx, *hny, *hnz;
    float *nx, *ny, *nz;
    float *s_crest;
    const HxAxisKnot *axis;
    int64_t axis_count;
} HalfTensorCtx;

static void half_tensor_accum_job(void *arg, size_t idx) {
    HalfTensorCtx *c = arg;
    int hz = (int)idx, Z = c->Z, Y = c->Y, X = c->X, hY = c->hY, hX = c->hX;
    size_t plane = (size_t)Y * X;
    size_t hplane = (size_t)hY * hX;
    int z_lo = 2 * hz, z_hi = z_lo + 2 < Z ? z_lo + 2 : Z;
    for (int hy = 0; hy < hY; hy++) {
        int y_lo = 2 * hy, y_hi = y_lo + 2 < Y ? y_lo + 2 : Y;
        for (int hx = 0; hx < hX; hx++) {
            int x_lo = 2 * hx, x_hi = x_lo + 2 < X ? x_lo + 2 : X;
            float a00 = 0, a11 = 0, a22 = 0, a01 = 0, a02 = 0, a12 = 0, sv = 0;
            int cnt = 0;
            for (int z = z_lo; z < z_hi; z++) {
                int zm = z > 0 ? z - 1 : 0, zp = z + 1 < Z ? z + 1 : Z - 1;
                float zsc = (z > 0 && z + 1 < Z) ? 0.5f : 1.0f;
                for (int y = y_lo; y < y_hi; y++) {
                    int ym = y > 0 ? y - 1 : 0, yp = y + 1 < Y ? y + 1 : Y - 1;
                    float ysc = (y > 0 && y + 1 < Y) ? 0.5f : 1.0f;
                    for (int x = x_lo; x < x_hi; x++) {
                        int xm = x > 0 ? x - 1 : 0, xp = x + 1 < X ? x + 1 : X - 1;
                        float xsc = (x > 0 && x + 1 < X) ? 0.5f : 1.0f;
                        size_t i = (size_t)z * plane + (size_t)y * X + x;
                        sv += c->s_sm[i];
                        float gz = (c->s_sm[(size_t)zp * plane + (size_t)y * X + x] -
                                       c->s_sm[(size_t)zm * plane + (size_t)y * X + x]) *
                            zsc;
                        float gy = (c->s_sm[(size_t)z * plane + (size_t)yp * X + x] -
                                       c->s_sm[(size_t)z * plane + (size_t)ym * X + x]) *
                            ysc;
                        float gx = (c->s_sm[(size_t)z * plane + (size_t)y * X + xp] -
                                       c->s_sm[(size_t)z * plane + (size_t)y * X + xm]) *
                            xsc;
                        a00 += gx * gx;
                        a11 += gy * gy;
                        a22 += gz * gz;
                        a01 += gx * gy;
                        a02 += gx * gz;
                        a12 += gy * gz;
                        cnt++;
                    }
                }
            }
            float inv = cnt > 0 ? 1.0f / (float)cnt : 0.0f;
            size_t hi = (size_t)hz * hplane + (size_t)hy * hX + hx;
            c->hs[hi] = sv * inv;
            c->hj[0][hi] = a00 * inv;
            c->hj[1][hi] = a11 * inv;
            c->hj[2][hi] = a22 * inv;
            c->hj[3][hi] = a01 * inv;
            c->hj[4][hi] = a02 * inv;
            c->hj[5][hi] = a12 * inv;
        }
    }
}

static void half_tensor_eig_job(void *arg, size_t idx) {
    HalfTensorCtx *c = arg;
    int hz = (int)idx, hY = c->hY, hX = c->hX;
    size_t hplane = (size_t)hY * hX;
    double cx, cy;
    hx_centre_at(c->axis, c->axis_count, (double)(c->z0 + 2 * hz) + 0.5, &cx, &cy);
    for (int hy = 0; hy < hY; hy++) {
        double ry = (double)(c->y0 + 2 * hy) + 0.5 - cy;
        for (int hx = 0; hx < hX; hx++) {
            size_t hi = (size_t)hz * hplane + (size_t)hy * hX + hx;
            float n[3];
            jacobi3_top(c->hj[0][hi], c->hj[1][hi], c->hj[2][hi], c->hj[3][hi], c->hj[4][hi],
                c->hj[5][hi], n);
            double rx = (double)(c->x0 + 2 * hx) + 0.5 - cx;
            if (n[0] * rx + n[1] * ry < 0.0) {
                n[0] = -n[0];
                n[1] = -n[1];
                n[2] = -n[2];
            }
            c->hnx[hi] = n[0];
            c->hny[hi] = n[1];
            c->hnz[hi] = n[2];
        }
    }
}

static void half_tensor_upsample_job(void *arg, size_t idx) {
    HalfTensorCtx *c = arg;
    int z = (int)idx, Y = c->Y, X = c->X, hZ = c->hZ, hY = c->hY, hX = c->hX;
    size_t plane = (size_t)Y * X;
    size_t hplane = (size_t)hY * hX;
    double cx, cy;
    hx_centre_at(c->axis, c->axis_count, (double)(c->z0 + z), &cx, &cy);
    float gz = 0.5f * ((float)z - 0.5f);
    if (gz < 0.0f)
        gz = 0.0f;
    else if (gz > hZ - 1.0001f)
        gz = hZ - 1.0001f;
    int iz0 = (int)gz, iz1 = iz0 + (iz0 + 1 < hZ);
    float fz = gz - iz0;
    for (int y = 0; y < Y; y++) {
        double ry = (double)(c->y0 + y) - cy;
        float gy = 0.5f * ((float)y - 0.5f);
        if (gy < 0.0f)
            gy = 0.0f;
        else if (gy > hY - 1.0001f)
            gy = hY - 1.0001f;
        int iy0 = (int)gy, iy1 = iy0 + (iy0 + 1 < hY);
        float fy = gy - iy0;
        for (int x = 0; x < X; x++) {
            float gx = 0.5f * ((float)x - 0.5f);
            if (gx < 0.0f)
                gx = 0.0f;
            else if (gx > hX - 1.0001f)
                gx = hX - 1.0001f;
            int ix0 = (int)gx, ix1 = ix0 + (ix0 + 1 < hX);
            float fx = gx - ix0;
            size_t ids[8] = {
                (size_t)iz0 * hplane + (size_t)iy0 * hX + ix0,
                (size_t)iz0 * hplane + (size_t)iy0 * hX + ix1,
                (size_t)iz0 * hplane + (size_t)iy1 * hX + ix0,
                (size_t)iz0 * hplane + (size_t)iy1 * hX + ix1,
                (size_t)iz1 * hplane + (size_t)iy0 * hX + ix0,
                (size_t)iz1 * hplane + (size_t)iy0 * hX + ix1,
                (size_t)iz1 * hplane + (size_t)iy1 * hX + ix0,
                (size_t)iz1 * hplane + (size_t)iy1 * hX + ix1,
            };
            float wts[8] = {
                (1.0f - fz) * (1.0f - fy) * (1.0f - fx),
                (1.0f - fz) * (1.0f - fy) * fx,
                (1.0f - fz) * fy * (1.0f - fx),
                (1.0f - fz) * fy * fx,
                fz * (1.0f - fy) * (1.0f - fx),
                fz * (1.0f - fy) * fx,
                fz * fy * (1.0f - fx),
                fz * fy * fx,
            };
            float rnx = c->hnx[ids[0]], rny = c->hny[ids[0]], rnz = c->hnz[ids[0]];
            float sx = 0, sy = 0, sz = 0, sw = 0;
            for (int k = 0; k < 8; k++) {
                float knx = c->hnx[ids[k]], kny = c->hny[ids[k]], knz = c->hnz[ids[k]];
                float sgn = (knx * rnx + kny * rny + knz * rnz) < 0.0f ? -1.0f : 1.0f;
                sx += wts[k] * sgn * knx;
                sy += wts[k] * sgn * kny;
                sz += wts[k] * sgn * knz;
                sw += wts[k] * c->hs[ids[k]];
            }
            float l = sqrtf(sx * sx + sy * sy + sz * sz) + 1e-6f;
            sx /= l;
            sy /= l;
            sz /= l;
            double rx = (double)(c->x0 + x) - cx;
            if (sx * rx + sy * ry < 0.0) {
                sx = -sx;
                sy = -sy;
                sz = -sz;
            }
            size_t i = (size_t)z * plane + (size_t)y * X + x;
            c->nx[i] = sx;
            c->ny[i] = sy;
            c->nz[i] = sz;
            c->s_crest[i] = 0.72f * c->s_sm[i] + 0.28f * sw;
        }
    }
}

static inline float trilinear(const float *vol, int Z, int Y, int X, float z, float y, float x) {
    if (z < 0.0f)
        z = 0.0f;
    else if (z > Z - 1.0001f)
        z = Z - 1.0001f;
    if (y < 0.0f)
        y = 0.0f;
    else if (y > Y - 1.0001f)
        y = Y - 1.0001f;
    if (x < 0.0f)
        x = 0.0f;
    else if (x > X - 1.0001f)
        x = X - 1.0001f;
    int iz = (int)z, iy = (int)y, ix = (int)x;
    float fz = z - iz, fy = y - iy, fx = x - ix;
    int iz1 = iz + (iz + 1 < Z), iy1 = iy + (iy + 1 < Y), ix1 = ix + (ix + 1 < X);
    size_t pl = (size_t)Y * X;
    size_t i000 = (size_t)iz * pl + (size_t)iy * X + ix;
    size_t i001 = (size_t)iz * pl + (size_t)iy * X + ix1;
    size_t i010 = (size_t)iz * pl + (size_t)iy1 * X + ix;
    size_t i011 = (size_t)iz * pl + (size_t)iy1 * X + ix1;
    size_t i100 = (size_t)iz1 * pl + (size_t)iy * X + ix;
    size_t i101 = (size_t)iz1 * pl + (size_t)iy * X + ix1;
    size_t i110 = (size_t)iz1 * pl + (size_t)iy1 * X + ix;
    size_t i111 = (size_t)iz1 * pl + (size_t)iy1 * X + ix1;
    float c00 = vol[i000] * (1.0f - fx) + vol[i001] * fx;
    float c01 = vol[i010] * (1.0f - fx) + vol[i011] * fx;
    float c10 = vol[i100] * (1.0f - fx) + vol[i101] * fx;
    float c11 = vol[i110] * (1.0f - fx) + vol[i111] * fx;
    float c0 = c00 * (1.0f - fy) + c01 * fy;
    float c1 = c10 * (1.0f - fy) + c11 * fy;
    return c0 * (1.0f - fz) + c1 * fz;
}

typedef struct {
    float x, y, z;
    float nx, ny, nz;
    float score, cos_w, sin_w, spacing;
} CrestPt;

typedef struct {
    CrestPt *pts;
    int64_t count, cap;
} CrestList;

typedef struct {
    int tZ, tY, tX;
    int gZ, gY, gX;
    int pz0, py0, px0;
    int cz0, cz1, cy0, cy1, cx0, cx1;
    const float *s_sm, *nx, *ny, *nz;
    const uint8_t *barrier, *spacing;
    const int8_t *phase;
    float s_min, curv_min;
    CrestList *per_z;
} CrestCtx;

static void extract_crest_job(void *arg, size_t idx) {
    CrestCtx *c = arg;
    int z = c->cz0 + (int)idx;
    if (z <= 0 || z >= c->tZ - 1)
        return;
    int gz = c->pz0 + z;
    if (gz <= 0 || gz >= c->gZ - 1)
        return;
    size_t tpl = (size_t)c->tY * c->tX;
    size_t gpl = (size_t)c->gY * c->gX;
    size_t gvox = (size_t)c->gZ * gpl;
    CrestList *out = &c->per_z[idx];
    int64_t cap = 2048;
    CrestPt *pts = malloc((size_t)cap * sizeof(*pts));
    int64_t cnt = 0;
    if (!pts)
        return;
    int y_lo = c->cy0 > 1 ? c->cy0 : 1;
    int y_hi = c->cy1 < c->tY - 1 ? c->cy1 : c->tY - 1;
    int x_lo = c->cx0 > 1 ? c->cx0 : 1;
    int x_hi = c->cx1 < c->tX - 1 ? c->cx1 : c->tX - 1;
    for (int y = y_lo; y < y_hi; y++) {
        int gy = c->py0 + y;
        if (gy <= 0 || gy >= c->gY - 1)
            continue;
        for (int x = x_lo; x < x_hi; x++) {
            int gx = c->px0 + x;
            if (gx <= 0 || gx >= c->gX - 1)
                continue;
            size_t i = (size_t)z * tpl + (size_t)y * c->tX + x;
            float v0 = c->s_sm[i];
            if (v0 < c->s_min)
                continue;
            size_t gi = (size_t)gz * gpl + (size_t)gy * c->gX + gx;
            if (c->barrier && c->barrier[gi] > 140)
                continue;
            float nxi = c->nx[i], nyi = c->ny[i], nzi = c->nz[i];
            float vp1 = trilinear(c->s_sm, c->tZ, c->tY, c->tX, z + nzi, y + nyi, x + nxi);
            float vm1 = trilinear(c->s_sm, c->tZ, c->tY, c->tX, z - nzi, y - nyi, x - nxi);
            if (v0 < vp1 || v0 < vm1)
                continue;
            float curv = 2.0f * v0 - vp1 - vm1;
            if (curv < c->curv_min)
                continue;
            float denom = vm1 - 2.0f * v0 + vp1;
            float delta = fabsf(denom) > 1e-6f ? 0.5f * (vm1 - vp1) / denom : 0.0f;
            if (delta < -0.5f)
                delta = -0.5f;
            else if (delta > 0.5f)
                delta = 0.5f;
            if (cnt == cap) {
                cap *= 2;
                CrestPt *grown = realloc(pts, (size_t)cap * sizeof(*pts));
                if (!grown) {
                    free(pts);
                    return;
                }
                pts = grown;
            }
            float cw = 1.0f, sw = 0.0f, spc = 12.0f;
            if (c->phase) {
                cw = c->phase[gi] / 127.0f;
                sw = c->phase[gvox + gi] / 127.0f;
            }
            if (c->spacing)
                spc = fmaxf(3.0f, (float)c->spacing[gi] * 0.25f);
            pts[cnt++] = (CrestPt){.x = (float)gx + delta * nxi,
                .y = (float)gy + delta * nyi,
                .z = (float)gz + delta * nzi,
                .nx = nxi,
                .ny = nyi,
                .nz = nzi,
                .score = v0,
                .cos_w = cw,
                .sin_w = sw,
                .spacing = spc};
        }
    }
    out->pts = pts;
    out->count = cnt;
    out->cap = cap;
}

typedef struct {
    int64_t cell;
    int32_t idx;
    float depth;
} BinEntry;

static int bin_cell_cmp(const void *a, const void *b) {
    const BinEntry *x = a, *y = b;
    if (x->cell != y->cell)
        return x->cell < y->cell ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

static int bin_depth_cmp(const void *a, const void *b) {
    const BinEntry *x = a, *y = b;
    return (x->depth > y->depth) - (x->depth < y->depth);
}

typedef struct {
    float x, y, z;
    float nx, ny, nz;
    float score, cos_w, sin_w, spacing, theta;
    int32_t cell, count, patch, winding;
} Surfel;

typedef struct {
    int32_t a, b;
    float w;
    int8_t cut, kind;
} SEdge;

static int sedge_cmp(const void *a, const void *b) {
    const SEdge *x = a, *y = b;
    if (x->a != y->a)
        return x->a < y->a ? -1 : 1;
    return (x->b > y->b) - (x->b < y->b);
}

typedef struct {
    int32_t p, q, k, kind;
    double up, down;
} PTerm;

static int pterm_cmp(const void *a, const void *b) {
    const PTerm *x = a, *y = b;
    if (x->p != y->p)
        return x->p < y->p ? -1 : 1;
    if (x->q != y->q)
        return x->q < y->q ? -1 : 1;
    if (x->k != y->k)
        return (x->k > y->k) - (x->k < y->k);
    return (x->kind > y->kind) - (x->kind < y->kind);
}

typedef struct {
    int32_t a, b;
} PairI32;

static int pair_i32_cmp(const void *a, const void *b) {
    const PairI32 *x = a, *y = b;
    if (x->a != y->a)
        return x->a < y->a ? -1 : 1;
    return (x->b > y->b) - (x->b < y->b);
}

typedef struct {
    int32_t p, q, kind;
    double w;
} MergeCand;

static int merge_cand_cmp(const void *a, const void *b) {
    const MergeCand *x = a, *y = b;
    if (x->kind != y->kind)
        return x->kind < y->kind ? -1 : 1;
    return (x->w < y->w) - (x->w > y->w);
}

typedef struct {
    int32_t ca, cb, u, v;
    float dist, align;
} StitchLink;

static int stitch_link_cmp(const void *a, const void *b) {
    const StitchLink *x = a, *y = b;
    if (x->ca != y->ca)
        return x->ca < y->ca ? -1 : 1;
    if (x->cb != y->cb)
        return x->cb < y->cb ? -1 : 1;
    if (x->dist != y->dist)
        return x->dist < y->dist ? -1 : 1;
    if (x->u != y->u)
        return x->u < y->u ? -1 : 1;
    return (x->v > y->v) - (x->v < y->v);
}

typedef struct {
    int32_t ca, cb;
    int64_t first, count;
    double score;
} StitchGroup;

static int stitch_group_cmp(const void *a, const void *b) {
    const StitchGroup *x = a, *y = b;
    return (x->score < y->score) - (x->score > y->score);
}

static int32_t seam_turns(double from, double to) {
    return -(int32_t)lrint((to - from) / (2.0 * M_PI));
}

typedef struct {
    int32_t lower, upper;
    float dist, dw_phase;
    int32_t count;
} RayObs;

static int ray_obs_cmp(const void *a, const void *b) {
    const RayObs *x = a, *y = b;
    if (x->lower != y->lower)
        return x->lower < y->lower ? -1 : 1;
    return (x->upper > y->upper) - (x->upper < y->upper);
}

static int float_cmp(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double rng_exp(uint64_t *s) {
    double u = ((rng_next(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return -log(u);
}

static int32_t uf_find(int32_t *uf, int32_t x) {
    while (uf[x] != x)
        x = uf[x] = uf[uf[x]];
    return x;
}

static void arap_refine(int n, const Surfel *surfels, const int32_t *members,
    const int64_t *adj_off, const int32_t *adj_to, float *u_arr, float *v_arr, int rounds) {
    double *u = malloc((size_t)n * sizeof(double));
    double *v = malloc((size_t)n * sizeof(double));
    double *e1 = malloc((size_t)n * 3 * sizeof(double));
    double *e2 = malloc((size_t)n * 3 * sizeof(double));
    double *rot_c = malloc((size_t)n * sizeof(double));
    double *rot_s = malloc((size_t)n * sizeof(double));
    double *bu = malloc((size_t)n * sizeof(double));
    double *bv = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    double *d = malloc((size_t)n * sizeof(double));
    double *q = malloc((size_t)n * sizeof(double));
    double *invd = malloc((size_t)n * sizeof(double));
    if (!u || !v || !e1 || !e2 || !rot_c || !rot_s || !bu || !bv || !r || !d || !q || !invd) {
        free(u);
        free(v);
        free(e1);
        free(e2);
        free(rot_c);
        free(rot_s);
        free(bu);
        free(bv);
        free(r);
        free(d);
        free(q);
        free(invd);
        return;
    }
    for (int i = 0; i < n; i++) {
        u[i] = u_arr[i];
        v[i] = v_arr[i];
        int deg = (int)(adj_off[i + 1] - adj_off[i]);
        invd[i] = deg > 0 ? 1.0 / (deg + (i == 0 ? 1.0 : 0.0)) : 0.0;
        const Surfel *s = &surfels[members[i]];
        double nx = s->nx, ny = s->ny, nz = s->nz;
        double ax = -ny, ay = nx, az = 0.0;
        double al = sqrt(ax * ax + ay * ay);
        if (al < 1e-4) {
            ax = 1.0;
            ay = 0.0;
            az = 0.0;
        } else {
            ax /= al;
            ay /= al;
        }
        double dotn = ax * nx + ay * ny + az * nz;
        ax -= dotn * nx;
        ay -= dotn * ny;
        az -= dotn * nz;
        al = sqrt(ax * ax + ay * ay + az * az) + 1e-12;
        e1[3 * i + 0] = ax / al;
        e1[3 * i + 1] = ay / al;
        e1[3 * i + 2] = az / al;
        e2[3 * i + 0] = ny * e1[3 * i + 2] - nz * e1[3 * i + 1];
        e2[3 * i + 1] = nz * e1[3 * i + 0] - nx * e1[3 * i + 2];
        e2[3 * i + 2] = nx * e1[3 * i + 1] - ny * e1[3 * i + 0];
    }
    for (int iter = 0; iter < rounds; iter++) {
        for (int i = 0; i < n; i++) {
            const Surfel *si = &surfels[members[i]];
            double a = 0.0, b = 0.0;
            for (int64_t e = adj_off[i]; e < adj_off[i + 1]; e++) {
                int j = adj_to[e];
                const Surfel *sj = &surfels[members[j]];
                double dx = sj->x - si->x, dy = sj->y - si->y, dz = sj->z - si->z;
                double px = dx * e1[3 * i] + dy * e1[3 * i + 1] + dz * e1[3 * i + 2];
                double py = dx * e2[3 * i] + dy * e2[3 * i + 1] + dz * e2[3 * i + 2];
                double qu = u[j] - u[i], qv = v[j] - v[i];
                a += px * qu + py * qv;
                b += px * qv - py * qu;
            }
            double mag = hypot(a, b);
            if (mag > 1e-12) {
                rot_c[i] = a / mag;
                rot_s[i] = b / mag;
            } else {
                rot_c[i] = 1.0;
                rot_s[i] = 0.0;
            }
        }
        memset(bu, 0, (size_t)n * sizeof(double));
        memset(bv, 0, (size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) {
            const Surfel *si = &surfels[members[i]];
            for (int64_t e = adj_off[i]; e < adj_off[i + 1]; e++) {
                int j = adj_to[e];
                if (j <= i)
                    continue;
                const Surfel *sj = &surfels[members[j]];
                double dx = sj->x - si->x, dy = sj->y - si->y, dz = sj->z - si->z;
                double pix = dx * e1[3 * i] + dy * e1[3 * i + 1] + dz * e1[3 * i + 2];
                double piy = dx * e2[3 * i] + dy * e2[3 * i + 1] + dz * e2[3 * i + 2];
                double pjx = dx * e1[3 * j] + dy * e1[3 * j + 1] + dz * e1[3 * j + 2];
                double pjy = dx * e2[3 * j] + dy * e2[3 * j + 1] + dz * e2[3 * j + 2];
                double du_ij =
                    0.5 * ((rot_c[i] * pix - rot_s[i] * piy) + (rot_c[j] * pjx - rot_s[j] * pjy));
                double dv_ij =
                    0.5 * ((rot_s[i] * pix + rot_c[i] * piy) + (rot_s[j] * pjx + rot_c[j] * pjy));
                bu[i] -= du_ij;
                bu[j] += du_ij;
                bv[i] -= dv_ij;
                bv[j] += dv_ij;
            }
        }
        bu[0] += u[0];
        bv[0] += v[0];
        for (int axis = 0; axis < 2; axis++) {
            double *x = axis ? v : u;
            const double *rhs = axis ? bv : bu;
            double rz = 0.0, scale = 1e-12;
            for (int i = 0; i < n; i++) {
                double deg = (double)(adj_off[i + 1] - adj_off[i]) + (i == 0 ? 1.0 : 0.0);
                double ax = deg * x[i];
                for (int64_t e = adj_off[i]; e < adj_off[i + 1]; e++)
                    ax -= x[adj_to[e]];
                r[i] = rhs[i] - ax;
                d[i] = invd[i] * r[i];
                rz += r[i] * d[i];
                scale += rhs[i] * rhs[i] * invd[i];
            }
            for (int step = 0; step < 35 && rz > 1e-8 * scale; step++) {
                double pq = 0.0;
                for (int i = 0; i < n; i++) {
                    double deg = (double)(adj_off[i + 1] - adj_off[i]) + (i == 0 ? 1.0 : 0.0);
                    double ad = deg * d[i];
                    for (int64_t e = adj_off[i]; e < adj_off[i + 1]; e++)
                        ad -= d[adj_to[e]];
                    q[i] = ad;
                    pq += d[i] * ad;
                }
                if (pq <= 1e-18)
                    break;
                double alpha = rz / pq, next_rz = 0.0;
                for (int i = 0; i < n; i++) {
                    x[i] += alpha * d[i];
                    r[i] -= alpha * q[i];
                    next_rz += r[i] * r[i] * invd[i];
                }
                double beta = next_rz / rz;
                for (int i = 0; i < n; i++)
                    d[i] = invd[i] * r[i] + beta * d[i];
                rz = next_rz;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        u_arr[i] = (float)u[i];
        v_arr[i] = (float)v[i];
    }
    free(u);
    free(v);
    free(e1);
    free(e2);
    free(rot_c);
    free(rot_s);
    free(bu);
    free(bv);
    free(r);
    free(d);
    free(q);
    free(invd);
}

static void biharmonic_fill_mesh(double *grid, const uint8_t *known, const uint8_t *hole, int h,
    int w, double stride, int rounds) {
    (void)rounds;
    size_t n = (size_t)h * w;
    size_t hcnt = 0;
    for (size_t i = 0; i < n; i++)
        if (hole[i])
            hcnt++;
    if (!hcnt)
        return;
    size_t *hidx = malloc(hcnt * sizeof(size_t));
    double *harm = malloc(3 * hcnt * sizeof(double));
    double *curv = calloc(3 * n, sizeof(double));
    uint8_t *has_c = calloc(n, 1);
    if (!hidx || !harm || !curv || !has_c) {
        free(hidx);
        free(harm);
        free(curv);
        free(has_c);
        return;
    }
    size_t pos = 0;
    for (size_t i = 0; i < n; i++)
        if (hole[i])
            hidx[pos++] = i;

    for (size_t k = 0; k < hcnt; k++) {
        size_t i = hidx[k];
        int r = (int)(i / w), c = (int)(i % w);
        int ru = r - 1, rd = r + 1, cl = c - 1, cr = c + 1;
        while (ru >= 0 && (!known[(size_t)ru * w + c] || hole[(size_t)ru * w + c]))
            ru--;
        while (rd < h && (!known[(size_t)rd * w + c] || hole[(size_t)rd * w + c]))
            rd++;
        while (cl >= 0 && (!known[(size_t)r * w + cl] || hole[(size_t)r * w + cl]))
            cl--;
        while (cr < w && (!known[(size_t)r * w + cr] || hole[(size_t)r * w + cr]))
            cr++;
        double wsum = 0.0, px = 0.0, py = 0.0, pz = 0.0;
        if (ru >= 0 && rd < h) {
            double span = (double)(rd - ru);
            double t = (double)(r - ru) / span;
            double wt = 1.0 / span;
            size_t iu = (size_t)ru * w + c, id = (size_t)rd * w + c;
            px += wt * ((1.0 - t) * grid[iu] + t * grid[id]);
            py += wt * ((1.0 - t) * grid[n + iu] + t * grid[n + id]);
            pz += wt * ((1.0 - t) * grid[2 * n + iu] + t * grid[2 * n + id]);
            wsum += wt;
        }
        if (cl >= 0 && cr < w) {
            double span = (double)(cr - cl);
            double t = (double)(c - cl) / span;
            double wt = 1.0 / span;
            size_t il = (size_t)r * w + cl, ir = (size_t)r * w + cr;
            px += wt * ((1.0 - t) * grid[il] + t * grid[ir]);
            py += wt * ((1.0 - t) * grid[n + il] + t * grid[n + ir]);
            pz += wt * ((1.0 - t) * grid[2 * n + il] + t * grid[2 * n + ir]);
            wsum += wt;
        }
        if (wsum > 0.0) {
            grid[i] = px / wsum;
            grid[n + i] = py / wsum;
            grid[2 * n + i] = pz / wsum;
        }
    }

    double om_h = 1.65;
    for (int it = 0; it < 160; it++) {
        for (size_t k = 0; k < hcnt; k++) {
            size_t i = hidx[k];
            int r = (int)(i / w), c = (int)(i % w);
            int cnt = 0;
            double sx = 0, sy = 0, sz = 0;
            if (r > 0 && known[i - w]) {
                sx += grid[i - w];
                sy += grid[n + i - w];
                sz += grid[2 * n + i - w];
                cnt++;
            }
            if (r + 1 < h && known[i + w]) {
                sx += grid[i + w];
                sy += grid[n + i + w];
                sz += grid[2 * n + i + w];
                cnt++;
            }
            if (c > 0 && known[i - 1]) {
                sx += grid[i - 1];
                sy += grid[n + i - 1];
                sz += grid[2 * n + i - 1];
                cnt++;
            }
            if (c + 1 < w && known[i + 1]) {
                sx += grid[i + 1];
                sy += grid[n + i + 1];
                sz += grid[2 * n + i + 1];
                cnt++;
            }
            if (cnt > 0) {
                double inv = 1.0 / cnt;
                grid[i] += om_h * (sx * inv - grid[i]);
                grid[n + i] += om_h * (sy * inv - grid[n + i]);
                grid[2 * n + i] += om_h * (sz * inv - grid[2 * n + i]);
            }
        }
    }
    for (size_t k = 0; k < hcnt; k++) {
        size_t i = hidx[k];
        harm[k] = grid[i];
        harm[hcnt + k] = grid[n + i];
        harm[2 * hcnt + k] = grid[2 * n + i];
    }

    double max_h = 0.09 * stride;
    for (int r = 1; r + 1 < h; r++) {
        for (int c = 1; c + 1 < w; c++) {
            size_t i = (size_t)r * w + c;
            if (!known[i] || hole[i])
                continue;
            if (!known[i - w] || hole[i - w] || !known[i + w] || hole[i + w] || !known[i - 1] ||
                hole[i - 1] || !known[i + 1] || hole[i + 1])
                continue;
            double hx = 0.25 * (grid[i - w] + grid[i + w] + grid[i - 1] + grid[i + 1]) - grid[i];
            double hy =
                0.25 * (grid[n + i - w] + grid[n + i + w] + grid[n + i - 1] + grid[n + i + 1]) -
                grid[n + i];
            double hz = 0.25 *
                    (grid[2 * n + i - w] + grid[2 * n + i + w] + grid[2 * n + i - 1] +
                        grid[2 * n + i + 1]) -
                grid[2 * n + i];
            double hl = sqrt(hx * hx + hy * hy + hz * hz);
            if (hl > max_h) {
                double sc = max_h / hl;
                hx *= sc;
                hy *= sc;
                hz *= sc;
            }
            curv[i] = hx;
            curv[n + i] = hy;
            curv[2 * n + i] = hz;
            has_c[i] = 1;
        }
    }
    for (int it = 0; it < 80; it++) {
        for (size_t k = 0; k < hcnt; k++) {
            size_t i = hidx[k];
            int r = (int)(i / w), c = (int)(i % w);
            int cnt = 0;
            double sx = 0, sy = 0, sz = 0;
            if (r > 0 && (hole[i - w] || has_c[i - w])) {
                sx += curv[i - w];
                sy += curv[n + i - w];
                sz += curv[2 * n + i - w];
                cnt++;
            }
            if (r + 1 < h && (hole[i + w] || has_c[i + w])) {
                sx += curv[i + w];
                sy += curv[n + i + w];
                sz += curv[2 * n + i + w];
                cnt++;
            }
            if (c > 0 && (hole[i - 1] || has_c[i - 1])) {
                sx += curv[i - 1];
                sy += curv[n + i - 1];
                sz += curv[2 * n + i - 1];
                cnt++;
            }
            if (c + 1 < w && (hole[i + 1] || has_c[i + 1])) {
                sx += curv[i + 1];
                sy += curv[n + i + 1];
                sz += curv[2 * n + i + 1];
                cnt++;
            }
            if (cnt > 0) {
                double inv = 1.0 / cnt;
                curv[i] += 1.45 * (sx * inv - curv[i]);
                curv[n + i] += 1.45 * (sy * inv - curv[n + i]);
                curv[2 * n + i] += 1.45 * (sz * inv - curv[2 * n + i]);
            }
        }
    }

    double max_dev = 3.0 * stride;
    double om_p = 1.55;
    for (int it = 0; it < 160; it++) {
        for (size_t k = 0; k < hcnt; k++) {
            size_t i = hidx[k];
            int r = (int)(i / w), c = (int)(i % w);
            int cnt = 0;
            double sx = 0, sy = 0, sz = 0;
            if (r > 0 && known[i - w]) {
                sx += grid[i - w];
                sy += grid[n + i - w];
                sz += grid[2 * n + i - w];
                cnt++;
            }
            if (r + 1 < h && known[i + w]) {
                sx += grid[i + w];
                sy += grid[n + i + w];
                sz += grid[2 * n + i + w];
                cnt++;
            }
            if (c > 0 && known[i - 1]) {
                sx += grid[i - 1];
                sy += grid[n + i - 1];
                sz += grid[2 * n + i - 1];
                cnt++;
            }
            if (c + 1 < w && known[i + 1]) {
                sx += grid[i + 1];
                sy += grid[n + i + 1];
                sz += grid[2 * n + i + 1];
                cnt++;
            }
            if (cnt > 0) {
                double inv = 1.0 / cnt;
                double tx = sx * inv - (cnt == 4 ? curv[i] : 0.0);
                double ty = sy * inv - (cnt == 4 ? curv[n + i] : 0.0);
                double tz = sz * inv - (cnt == 4 ? curv[2 * n + i] : 0.0);
                double nx = grid[i] + om_p * (tx - grid[i]);
                double ny = grid[n + i] + om_p * (ty - grid[n + i]);
                double nz = grid[2 * n + i] + om_p * (tz - grid[2 * n + i]);
                double dx = nx - harm[k], dy = ny - harm[hcnt + k], dz = nz - harm[2 * hcnt + k];
                double dl = sqrt(dx * dx + dy * dy + dz * dz);
                if (dl > max_dev) {
                    double sc = max_dev / dl;
                    nx = harm[k] + dx * sc;
                    ny = harm[hcnt + k] + dy * sc;
                    nz = harm[2 * hcnt + k] + dz * sc;
                }
                grid[i] = nx;
                grid[n + i] = ny;
                grid[2 * n + i] = nz;
            }
        }
    }
    free(hidx);
    free(harm);
    free(curv);
    free(has_c);
}

static void tangential_isometric_fair(double *grid, const uint8_t *known, const uint8_t *locked,
    int h, int w, double stride, int rounds) {
    if (rounds <= 0)
        return;
    size_t n = (size_t)h * w;
    double *next_g = malloc(3 * n * sizeof(double));
    if (!next_g)
        return;
    for (int round = 0; round < rounds; round++) {
        memcpy(next_g, grid, 3 * n * sizeof(double));
        for (int r = 1; r + 1 < h; r++) {
            for (int c = 1; c + 1 < w; c++) {
                size_t i = (size_t)r * w + c;
                if (!known[i])
                    continue;
                size_t iu = i - w, id = i + w, il = i - 1, ir = i + 1;
                if (!known[iu] || !known[id] || !known[il] || !known[ir])
                    continue;
                double px = grid[i], py = grid[n + i], pz = grid[2 * n + i];
                double ux = grid[ir] - grid[il], uy = grid[n + ir] - grid[n + il],
                       uz = grid[2 * n + ir] - grid[2 * n + il];
                double vx = grid[id] - grid[iu], vy = grid[n + id] - grid[n + iu],
                       vz = grid[2 * n + id] - grid[2 * n + iu];
                double nx = vy * uz - vz * uy, ny = vz * ux - vx * uz, nz = vx * uy - vy * ux;
                double nl = sqrt(nx * nx + ny * ny + nz * nz);
                if (nl < 1e-6)
                    continue;
                nx /= nl;
                ny /= nl;
                nz /= nl;
                double mx = 0.25 * (grid[iu] + grid[id] + grid[il] + grid[ir]);
                double my = 0.25 * (grid[n + iu] + grid[n + id] + grid[n + il] + grid[n + ir]);
                double mz = 0.25 *
                    (grid[2 * n + iu] + grid[2 * n + id] + grid[2 * n + il] + grid[2 * n + ir]);
                size_t nbs[4] = {iu, id, il, ir};
                double ix = 0.0, iy = 0.0, iz = 0.0;
                for (int k = 0; k < 4; k++) {
                    size_t j = nbs[k];
                    double dx = px - grid[j], dy = py - grid[n + j], dz = pz - grid[2 * n + j];
                    double d = sqrt(dx * dx + dy * dy + dz * dz);
                    if (d > 1e-3) {
                        double sc = stride / d;
                        ix += grid[j] + dx * sc;
                        iy += grid[n + j] + dy * sc;
                        iz += grid[2 * n + j] + dz * sc;
                    } else {
                        ix += px;
                        iy += py;
                        iz += pz;
                    }
                }
                ix *= 0.25;
                iy *= 0.25;
                iz *= 0.25;
                double tx = 0.55 * mx + 0.45 * ix - px;
                double ty = 0.55 * my + 0.45 * iy - py;
                double tz = 0.55 * mz + 0.45 * iz - pz;
                double dn = tx * nx + ty * ny + tz * nz;
                tx -= dn * nx;
                ty -= dn * ny;
                tz -= dn * nz;
                double tl = sqrt(tx * tx + ty * ty + tz * tz);
                double max_step = 0.25 * stride;
                if (tl > max_step) {
                    double sc = max_step / tl;
                    tx *= sc;
                    ty *= sc;
                    tz *= sc;
                }
                double step_w = locked[i] ? 0.25 : 0.45;
                next_g[i] = px + step_w * tx;
                next_g[n + i] = py + step_w * ty;
                next_g[2 * n + i] = pz + step_w * tz;
            }
        }
        memcpy(grid, next_g, 3 * n * sizeof(double));
    }
    free(next_g);
}

typedef struct {
    const double *grid;
    const uint8_t *known, *locked;
    int h, w, Z, Y, X;
    int64_t z0, y0, x0;
    const uint8_t *su8, *bu8;
    const int8_t *pi8;
    double search_r;
    double *delta, *weight, *gnx, *gny, *gnz;
} LockCtx;

static void lock_row_job(void *arg, size_t r_idx) {
    LockCtx *lc = arg;
    int r = (int)r_idx, h = lc->h, w = lc->w;
    size_t n = (size_t)h * w;
    const double *grid = lc->grid;
    const uint8_t *known = lc->known, *locked = lc->locked;
    for (int c = 0; c < w; c++) {
        size_t i = (size_t)r * w + c;
        if (!known[i])
            continue;
        int ru = r > 0 && known[(size_t)(r - 1) * w + c] ? r - 1 : r;
        int rd = r + 1 < h && known[(size_t)(r + 1) * w + c] ? r + 1 : r;
        int cl = c > 0 && known[(size_t)r * w + c - 1] ? c - 1 : c;
        int cr = c + 1 < w && known[(size_t)r * w + c + 1] ? c + 1 : c;
        if (ru == rd || cl == cr)
            continue;
        size_t iu = (size_t)ru * w + c, id = (size_t)rd * w + c;
        size_t il = (size_t)r * w + cl, ir = (size_t)r * w + cr;
        double ax = grid[id] - grid[iu], ay = grid[n + id] - grid[n + iu],
               az = grid[2 * n + id] - grid[2 * n + iu];
        double bx = grid[ir] - grid[il], by = grid[n + ir] - grid[n + il],
               bz = grid[2 * n + ir] - grid[2 * n + il];
        double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        double len = sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6)
            continue;
        nx /= len;
        ny /= len;
        nz /= len;
        lc->gnx[i] = nx;
        lc->gny[i] = ny;
        lc->gnz[i] = nz;
        double px = grid[i] - lc->x0, py = grid[n + i] - lc->y0, pz = grid[2 * n + i] - lc->z0;
        if (px < 3 || px > lc->X - 4 || py < 3 || py > lc->Y - 4 || pz < 3 || pz > lc->Z - 4)
            continue;
        double loc_r = locked[i] ? lc->search_r : fmax(lc->search_r, 3.6);
        double dt = 0.35;
        int half = (int)ceil(loc_r / dt);
        if (half > 12)
            half = 12;
        int pad = 3;
        int npts = 2 * (half + pad) + 1;
        float sbuf[35];
        for (int k = 0; k < npts; k++) {
            double t = (k - half - pad) * dt;
            sbuf[k] = sample_fused_trilinear(lc->su8, lc->bu8, lc->pi8, lc->Z, lc->Y, lc->X,
                (float)(pz + t * nz), (float)(py + t * ny), (float)(px + t * nx));
        }
        float best_val = -1e9f, best_raw = 0.0f;
        double best_t = 0.0;
        for (int s = -half; s <= half; s++) {
            int k = s + half + pad;
            float val = 0.10f * sbuf[k - 2] + 0.24f * sbuf[k - 1] + 0.32f * sbuf[k] +
                0.24f * sbuf[k + 1] + 0.10f * sbuf[k + 2];
            double t = s * dt;
            float pen = val - (float)(0.009 * t * t);
            if (pen > best_val) {
                best_val = pen;
                best_t = t;
                best_raw = val;
            }
        }
        float min_p = locked[i] ? 0.12f : 0.16f;
        if (best_raw > min_p) {
            lc->delta[i] = best_t;
            lc->weight[i] =
                locked[i] ? fmin(1.0, (double)best_raw * 2.0) : fmin(0.85, (double)best_raw * 1.6);
        }
    }
}

static void lock_grid_to_crest(double *grid, const uint8_t *known, const uint8_t *locked, int h,
    int w, int64_t z0, int64_t y0, int64_t x0, int Z, int Y, int X, const uint8_t *su8,
    const uint8_t *bu8, const int8_t *pi8, const HxAxisKnot *axis, int64_t axis_count,
    double search_r) {
    (void)axis;
    (void)axis_count;
    size_t n = (size_t)h * w;
    double *delta = calloc(n, sizeof(double));
    double *weight = calloc(n, sizeof(double));
    double *next_d = calloc(n, sizeof(double));
    double *gnx = calloc(n, sizeof(double));
    double *gny = calloc(n, sizeof(double));
    double *gnz = calloc(n, sizeof(double));
    if (!delta || !weight || !next_d || !gnx || !gny || !gnz) {
        free(delta);
        free(weight);
        free(next_d);
        free(gnx);
        free(gny);
        free(gnz);
        return;
    }
    LockCtx lctx = {.grid = grid,
        .known = known,
        .locked = locked,
        .h = h,
        .w = w,
        .Z = Z,
        .Y = Y,
        .X = X,
        .z0 = z0,
        .y0 = y0,
        .x0 = x0,
        .su8 = su8,
        .bu8 = bu8,
        .pi8 = pi8,
        .search_r = search_r,
        .delta = delta,
        .weight = weight,
        .gnx = gnx,
        .gny = gny,
        .gnz = gnz};
    each_block((size_t)h, &lctx, lock_row_job);
    double *obs = malloc(n * sizeof(double));
    if (obs) {
        memcpy(obs, delta, n * sizeof(double));
        for (int round = 0; round < 24; round++) {
            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    size_t i = (size_t)r * w + c;
                    if (!known[i])
                        continue;
                    double nsum = 0.0;
                    int ncnt = 0;
                    if (r > 0 && known[i - w]) {
                        nsum += delta[i - w];
                        ncnt++;
                    }
                    if (r + 1 < h && known[i + w]) {
                        nsum += delta[i + w];
                        ncnt++;
                    }
                    if (c > 0 && known[i - 1]) {
                        nsum += delta[i - 1];
                        ncnt++;
                    }
                    if (c + 1 < w && known[i + 1]) {
                        nsum += delta[i + 1];
                        ncnt++;
                    }
                    double lam = 0.90 * ncnt;
                    double denom = weight[i] + lam;
                    next_d[i] = denom > 1e-6 ? (weight[i] * obs[i] + 0.90 * nsum) / denom : 0.0;
                }
            }
            memcpy(delta, next_d, n * sizeof(double));
        }
        for (size_t i = 0; i < n; i++) {
            if (!known[i] || (gnx[i] == 0 && gny[i] == 0 && gnz[i] == 0))
                continue;
            double cap_r = locked[i] ? search_r : fmax(search_r, 3.6);
            double d = fmax(-cap_r, fmin(cap_r, delta[i]));
            grid[i] += d * gnx[i];
            grid[n + i] += d * gny[i];
            grid[2 * n + i] += d * gnz[i];
        }
        free(obs);
    }
    free(delta);
    free(weight);
    free(next_d);
    free(gnx);
    free(gny);
    free(gnz);
}

typedef struct {
    int32_t *members;
    int32_t count;
} PieceSpec;

static int piece_spec_cmp(const void *a, const void *b) {
    const PieceSpec *x = a, *y = b;
    return (x->count < y->count) - (x->count > y->count);
}

static int parse_box_arg(const char *s, int64_t b[6]) {
    return sscanf(s, "%lld:%lld,%lld:%lld,%lld:%lld", (long long *)&b[0], (long long *)&b[1],
               (long long *)&b[2], (long long *)&b[3], (long long *)&b[4], (long long *)&b[5]) == 6
        ? 0
        : -1;
}

typedef struct {
    const char *out_dir, *sample, *scan;
    int64_t box[6];
    int Z, Y, X, curl, stride, min_nodes, arap_rounds, no_lock, fill_reach, fair_rounds;
    int max_hole_cells, verbose;
    double voxel_um;
    const uint8_t *su8, *bu8;
    const int8_t *pi8;
    const HxAxisKnot *axis;
    int64_t axis_count;
    const Surfel *surfels;
    const SheetNode *snodes;
    int64_t nsurf;
    json_object *rec_pieces;
    double total_cm2, area_4cm2, largest_cm2;
    int written_count, pieces_4, iso_pass, fold_splits;
} EmitCtx;

static void emit_piece(EmitCtx *ec, const int32_t *mem, int32_t n, const int32_t *ea,
    const int32_t *eb, int64_t pm, const float *in_u, const float *in_v, int allow_split) {
    if (n < ec->min_nodes)
        return;
    float *pu = malloc((size_t)n * sizeof(float));
    float *pv = malloc((size_t)n * sizeof(float));
    int64_t *loff = calloc((size_t)n + 1, sizeof(int64_t));
    for (int64_t e = 0; e < pm; e++) {
        loff[ea[e] + 1]++;
        loff[eb[e] + 1]++;
    }
    for (int32_t i = 0; i < n; i++)
        loff[i + 1] += loff[i];
    int32_t *lto = malloc((size_t)(2 * pm + 1) * sizeof(int32_t));
    int64_t *lc = malloc((size_t)(n + 1) * sizeof(int64_t));
    memcpy(lc, loff, (size_t)(n + 1) * sizeof(int64_t));
    for (int64_t e = 0; e < pm; e++) {
        lto[lc[ea[e]]++] = eb[e];
        lto[lc[eb[e]]++] = ea[e];
    }
    free(lc);

    if (in_u && in_v) {
        memcpy(pu, in_u, (size_t)n * sizeof(float));
        memcpy(pv, in_v, (size_t)n * sizeof(float));
    } else {
        int32_t *ord = malloc((size_t)n * sizeof(int32_t));
        int64_t pf[2] = {0, n};
        int64_t ef[2] = {0, pm};
        int32_t *gea = malloc((size_t)(pm ? pm : 1) * sizeof(int32_t));
        int32_t *geb = malloc((size_t)(pm ? pm : 1) * sizeof(int32_t));
        SheetNode *sub_sn = malloc((size_t)n * sizeof(SheetNode));
        for (int32_t i = 0; i < n; i++) {
            ord[i] = i;
            sub_sn[i] = ec->snodes[mem[i]];
        }
        for (int64_t e = 0; e < pm; e++) {
            gea[e] = ea[e];
            geb[e] = eb[e];
        }
        int32_t *lmap = malloc((size_t)n * sizeof(int32_t));
        int8_t *fsgn = calloc((size_t)n, 1);
        memset(pu, 0, (size_t)n * sizeof(float));
        memset(pv, 0, (size_t)n * sizeof(float));
        hx_level_flatten(sub_sn, ord, pf, 1, gea, geb, ef, lmap, fsgn, pu, pv, 400, 1e-3, 0.01, 1);
        free(ord);
        free(gea);
        free(geb);
        free(sub_sn);
        free(lmap);
        free(fsgn);
    }

    if (ec->arap_rounds > 0 && pm > n / 2)
        arap_refine(n, ec->surfels, mem, loff, lto, pu, pv, ec->arap_rounds);

    double umin = DBL_MAX, umax = -DBL_MAX, vmin = DBL_MAX, vmax = -DBL_MAX;
    for (int32_t i = 0; i < n; i++) {
        if (pu[i] < umin)
            umin = pu[i];
        if (pu[i] > umax)
            umax = pu[i];
        if (pv[i] < vmin)
            vmin = pv[i];
        if (pv[i] > vmax)
            vmax = pv[i];
    }
    int gw = (int)floor((umax - umin) / ec->stride) + 1;
    int gh = (int)floor((vmax - vmin) / ec->stride) + 1;
    if (gw < 2 || gh < 2 || (int64_t)gw * gh > 16000000) {
        free(pu);
        free(pv);
        free(loff);
        free(lto);
        return;
    }
    double *xyz = malloc((size_t)n * 3 * sizeof(double));
    double *nrm = malloc((size_t)n * 3 * sizeof(double));
    double *du = malloc((size_t)n * sizeof(double));
    double *dv = malloc((size_t)n * sizeof(double));
    int64_t *flat_cell = malloc((size_t)n * sizeof(int64_t));
    size_t gsize = (size_t)gh * gw;
    double *grid = calloc(3 * gsize, sizeof(double));
    uint8_t *locked = calloc(gsize, 1);
    double *spread = calloc(gsize, sizeof(double));
    for (int32_t i = 0; i < n; i++) {
        const Surfel *s = &ec->surfels[mem[i]];
        xyz[3 * i + 0] = s->x;
        xyz[3 * i + 1] = s->y;
        xyz[3 * i + 2] = s->z;
        nrm[3 * i + 0] = s->nx;
        nrm[3 * i + 1] = s->ny;
        nrm[3 * i + 2] = s->nz;
        du[i] = pu[i] - umin;
        dv[i] = pv[i] - vmin;
    }
    int64_t occ = hx_raster_cells(
        gh, gw, n, xyz, nrm, du, dv, (double)ec->stride, 2.2, flat_cell, grid, locked, spread);
    int64_t lock_cnt = 0;
    for (size_t j = 0; j < gsize; j++)
        lock_cnt += locked[j];
    double fold_share = occ > 0 ? 1.0 - (double)lock_cnt / (double)occ : 1.0;
    if (ec->verbose && allow_split && n >= 200) {
        int32_t wmin = 1000000, wmax = -1000000;
        int32_t ncc = 0;
        int8_t *vv = calloc((size_t)n, 1);
        int32_t *qq = malloc((size_t)n * sizeof(int32_t));
        for (int32_t i = 0; i < n; i++) {
            int32_t w = ec->surfels[mem[i]].winding;
            if (w < wmin)
                wmin = w;
            if (w > wmax)
                wmax = w;
            if (!vv[i]) {
                ncc++;
                int32_t h0 = 0, t0 = 0;
                vv[i] = 1;
                qq[t0++] = i;
                while (h0 < t0) {
                    int32_t u = qq[h0++];
                    for (int64_t e = loff[u]; e < loff[u + 1]; e++)
                        if (!vv[lto[e]]) {
                            vv[lto[e]] = 1;
                            qq[t0++] = lto[e];
                        }
                }
            }
        }
        free(vv);
        free(qq);
        printf("  [diag] n=%d pm=%lld ncc=%d w=[%d..%d] gw=%d gh=%d occ=%lld lock=%lld fold=%.3f\n",
            n, (long long)pm, ncc, wmin, wmax, gw, gh, (long long)occ, (long long)lock_cnt,
            fold_share);
    }

    if (fold_share > 0.12 && allow_split && lock_cnt >= ec->min_nodes) {
        uint8_t *keep = calloc((size_t)n, 1);
        for (int32_t i = 0; i < n; i++)
            if (locked[flat_cell[i]])
                keep[i] = 1;
        int32_t *vis = calloc((size_t)n, sizeof(int32_t));
        int32_t *q = malloc((size_t)n * sizeof(int32_t));
        int32_t *old2sub = malloc((size_t)n * sizeof(int32_t));
        int32_t *sub_mem = malloc((size_t)n * sizeof(int32_t));
        int32_t *sub_ea = malloc((size_t)(pm ? pm : 1) * sizeof(int32_t));
        int32_t *sub_eb = malloc((size_t)(pm ? pm : 1) * sizeof(int32_t));
        ec->fold_splits++;
        for (int32_t seed = 0; seed < n; seed++) {
            if (!keep[seed] || vis[seed])
                continue;
            int32_t hd = 0, tl = 0;
            vis[seed] = 1;
            q[tl++] = seed;
            while (hd < tl) {
                int32_t u = q[hd++];
                for (int64_t e = loff[u]; e < loff[u + 1]; e++) {
                    int32_t v = lto[e];
                    if (keep[v] && !vis[v]) {
                        vis[v] = 1;
                        q[tl++] = v;
                    }
                }
            }
            if (tl < ec->min_nodes)
                continue;
            for (int32_t i = 0; i < tl; i++) {
                old2sub[q[i]] = i;
                sub_mem[i] = mem[q[i]];
            }
            int64_t spm = 0;
            for (int32_t i = 0; i < tl; i++) {
                int32_t u = q[i];
                for (int64_t e = loff[u]; e < loff[u + 1]; e++) {
                    int32_t v = lto[e];
                    if (keep[v] && v > u && vis[v]) {
                        sub_ea[spm] = old2sub[u];
                        sub_eb[spm] = old2sub[v];
                        spm++;
                    }
                }
            }
            emit_piece(ec, sub_mem, tl, sub_ea, sub_eb, spm, NULL, NULL, 0);
        }
        free(keep);
        free(vis);
        free(q);
        free(old2sub);
        free(sub_mem);
        free(sub_ea);
        free(sub_eb);
        free(pu);
        free(pv);
        free(loff);
        free(lto);
        free(xyz);
        free(nrm);
        free(du);
        free(dv);
        free(flat_cell);
        free(grid);
        free(locked);
        free(spread);
        return;
    }

    if (occ <= 0 || fold_share > 0.30 || lock_cnt < ec->min_nodes / 2) {
        free(pu);
        free(pv);
        free(loff);
        free(lto);
        free(xyz);
        free(nrm);
        free(du);
        free(dv);
        free(flat_cell);
        free(grid);
        free(locked);
        free(spread);
        return;
    }

    hx_centre_cells(gh, gw, n, flat_cell, du, dv, xyz, NULL, locked, (double)ec->stride, 4, 0.1,
        0.5, 1.5, grid);

    uint8_t *known = malloc(gsize);
    memcpy(known, locked, gsize);
    uint8_t *hole = calloc(gsize, 1);
    int64_t hole_cnt = 0;
    int fr = ec->fill_reach > 0 ? ec->fill_reach : 1;
    static const int dirs4[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int r = 0; r < gh; r++) {
        for (int c = 0; c < gw; c++) {
            size_t i = (size_t)r * gw + c;
            if (locked[i])
                continue;
            int best_span = 1000000;
            double init_p[3] = {0, 0, 0};
            for (int d = 0; d < 4; d++) {
                int dr = dirs4[d][0], dc = dirs4[d][1];
                int d1 = 0, d2 = 0;
                for (int k = 1; k <= fr; k++) {
                    int rr = r - k * dr, cc = c - k * dc;
                    if (rr < 0 || rr >= gh || cc < 0 || cc >= gw)
                        break;
                    if (locked[(size_t)rr * gw + cc]) {
                        d1 = k;
                        break;
                    }
                }
                if (!d1)
                    continue;
                for (int k = 1; k <= fr + 1 - d1; k++) {
                    int rr = r + k * dr, cc = c + k * dc;
                    if (rr < 0 || rr >= gh || cc < 0 || cc >= gw)
                        break;
                    if (locked[(size_t)rr * gw + cc]) {
                        d2 = k;
                        break;
                    }
                }
                if (!d2)
                    continue;
                int span = d1 + d2;
                if (span >= best_span)
                    continue;
                size_t i1 = (size_t)(r - d1 * dr) * gw + (c - d1 * dc);
                size_t i2 = (size_t)(r + d2 * dr) * gw + (c + d2 * dc);
                double dx = grid[i2] - grid[i1];
                double dy = grid[gsize + i2] - grid[gsize + i1];
                double dz = grid[2 * gsize + i2] - grid[2 * gsize + i1];
                double dist3 = sqrt(dx * dx + dy * dy + dz * dz);
                double step_len = (d >= 2 ? 1.41421356 : 1.0) * ec->stride * span;
                if (dist3 < 0.72 * step_len || dist3 > 1.32 * step_len)
                    continue;
                best_span = span;
                double t = (double)d1 / (double)span;
                init_p[0] = (1.0 - t) * grid[i1] + t * grid[i2];
                init_p[1] = (1.0 - t) * grid[gsize + i1] + t * grid[gsize + i2];
                init_p[2] = (1.0 - t) * grid[2 * gsize + i1] + t * grid[2 * gsize + i2];
            }
            if (best_span <= fr + 1) {
                hole[i] = 1;
                known[i] = 1;
                hole_cnt++;
                grid[i] = init_p[0];
                grid[gsize + i] = init_p[1];
                grid[2 * gsize + i] = init_p[2];
            }
        }
    }
    uint8_t *ext = calloc(gsize, 1);
    int64_t *hq = malloc(gsize * sizeof(int64_t));
    if (ext && hq) {
        int64_t hd = 0, tl = 0;
        for (int r = 0; r < gh; r++) {
            size_t i0 = (size_t)r * gw, i1 = i0 + (size_t)(gw - 1);
            if (!known[i0] && !ext[i0]) {
                ext[i0] = 1;
                hq[tl++] = (int64_t)i0;
            }
            if (!known[i1] && !ext[i1]) {
                ext[i1] = 1;
                hq[tl++] = (int64_t)i1;
            }
        }
        for (int c = 0; c < gw; c++) {
            size_t i0 = (size_t)c, i1 = (size_t)(gh - 1) * gw + (size_t)c;
            if (!known[i0] && !ext[i0]) {
                ext[i0] = 1;
                hq[tl++] = (int64_t)i0;
            }
            if (!known[i1] && !ext[i1]) {
                ext[i1] = 1;
                hq[tl++] = (int64_t)i1;
            }
        }
        while (hd < tl) {
            int64_t cur_i = hq[hd++];
            int rr = (int)(cur_i / gw), cc = (int)(cur_i % gw);
            if (rr > 0 && !known[cur_i - gw] && !ext[cur_i - gw]) {
                ext[cur_i - gw] = 1;
                hq[tl++] = cur_i - gw;
            }
            if (rr + 1 < gh && !known[cur_i + gw] && !ext[cur_i + gw]) {
                ext[cur_i + gw] = 1;
                hq[tl++] = cur_i + gw;
            }
            if (cc > 0 && !known[cur_i - 1] && !ext[cur_i - 1]) {
                ext[cur_i - 1] = 1;
                hq[tl++] = cur_i - 1;
            }
            if (cc + 1 < gw && !known[cur_i + 1] && !ext[cur_i + 1]) {
                ext[cur_i + 1] = 1;
                hq[tl++] = cur_i + 1;
            }
        }
        for (size_t seed = 0; seed < gsize; seed++) {
            if (known[seed] || ext[seed])
                continue;
            int64_t chd = 0, ctl = 0;
            ext[seed] = 2;
            hq[ctl++] = (int64_t)seed;
            double sx = 0.0, sy = 0.0, sz = 0.0;
            int64_t bcnt_h = 0;
            while (chd < ctl) {
                int64_t cur_i = hq[chd++];
                int rr = (int)(cur_i / gw), cc = (int)(cur_i % gw);
                int64_t nbs[4] = {rr > 0 ? cur_i - gw : -1, rr + 1 < gh ? cur_i + gw : -1,
                    cc > 0 ? cur_i - 1 : -1, cc + 1 < gw ? cur_i + 1 : -1};
                for (int d = 0; d < 4; d++) {
                    int64_t ni = nbs[d];
                    if (ni < 0)
                        continue;
                    if (known[ni]) {
                        sx += grid[ni];
                        sy += grid[gsize + ni];
                        sz += grid[2 * gsize + ni];
                        bcnt_h++;
                    } else if (!ext[ni]) {
                        ext[ni] = 2;
                        hq[ctl++] = ni;
                    }
                }
            }
            if (ctl <= ec->max_hole_cells && bcnt_h >= 4) {
                double mx = sx / bcnt_h, my = sy / bcnt_h, mz = sz / bcnt_h;
                for (int64_t k = 0; k < ctl; k++) {
                    size_t idx = (size_t)hq[k];
                    hole[idx] = 1;
                    known[idx] = 1;
                    hole_cnt++;
                    grid[idx] = mx;
                    grid[gsize + idx] = my;
                    grid[2 * gsize + idx] = mz;
                }
            }
        }
    }
    free(ext);
    free(hq);
    if (hole_cnt > 0) {
        biharmonic_fill_mesh(grid, known, hole, gh, gw, (double)ec->stride, 160);
        for (int r = 0; r < gh; r++) {
            for (int c = 0; c < gw; c++) {
                size_t i = (size_t)r * gw + c;
                if (!hole[i])
                    continue;
                int bad = 0;
                double lu = 0, lv = 0, ux = 0, uy = 0, uz = 0, vx = 0, vy = 0, vz = 0;
                int has_u = 0, has_v = 0;
                if (c + 1 < gw && known[i + 1]) {
                    ux = grid[i + 1] - grid[i];
                    uy = grid[gsize + i + 1] - grid[gsize + i];
                    uz = grid[2 * gsize + i + 1] - grid[2 * gsize + i];
                    lu = sqrt(ux * ux + uy * uy + uz * uz);
                    has_u = 1;
                } else if (c > 0 && known[i - 1]) {
                    ux = grid[i] - grid[i - 1];
                    uy = grid[gsize + i] - grid[gsize + i - 1];
                    uz = grid[2 * gsize + i] - grid[2 * gsize + i - 1];
                    lu = sqrt(ux * ux + uy * uy + uz * uz);
                    has_u = 1;
                }
                if (r + 1 < gh && known[i + gw]) {
                    vx = grid[i + gw] - grid[i];
                    vy = grid[gsize + i + gw] - grid[gsize + i];
                    vz = grid[2 * gsize + i + gw] - grid[2 * gsize + i];
                    lv = sqrt(vx * vx + vy * vy + vz * vz);
                    has_v = 1;
                } else if (r > 0 && known[i - gw]) {
                    vx = grid[i] - grid[i - gw];
                    vy = grid[gsize + i] - grid[gsize + i - gw];
                    vz = grid[2 * gsize + i] - grid[2 * gsize + i - gw];
                    lv = sqrt(vx * vx + vy * vy + vz * vz);
                    has_v = 1;
                }
                if (has_u && (lu < 0.45 * ec->stride || lu > 1.75 * ec->stride))
                    bad = 1;
                if (has_v && (lv < 0.45 * ec->stride || lv > 1.75 * ec->stride))
                    bad = 1;
                if (has_u && has_v && lu > 1e-3 && lv > 1e-3) {
                    double sh = fabs(ux * vx + uy * vy + uz * vz) / (lu * lv);
                    if (sh > 0.58)
                        bad = 1;
                }
                if (bad) {
                    known[i] = 0;
                    hole[i] = 0;
                    hole_cnt--;
                }
            }
        }
    }

    if (ec->fair_rounds > 0)
        tangential_isometric_fair(grid, known, locked, gh, gw, (double)ec->stride, ec->fair_rounds);

    if (!ec->no_lock)
        lock_grid_to_crest(grid, known, locked, gh, gw, ec->box[0], ec->box[2], ec->box[4], ec->Z,
            ec->Y, ec->X, ec->su8, ec->bu8, ec->pi8, ec->axis, ec->axis_count, 2.5);

    if (ec->fair_rounds > 0)
        tangential_isometric_fair(grid, known, locked, gh, gw, (double)ec->stride,
            ec->fair_rounds / 2 > 4 ? ec->fair_rounds / 2 : 4);

    double iso[14] = {0};
    hx_mesh_isometry(grid, gh, gw, known, locked, (double)ec->stride, iso);
    int64_t known_cnt = lock_cnt + hole_cnt;
    double area_mm2 =
        (double)known_cnt * (ec->stride * ec->voxel_um * 1e-3) * (ec->stride * ec->voxel_um * 1e-3);
    double area_cm2 = area_mm2 * 0.01;
    int is_iso = iso[7 + 5] >= 0.70 && iso[7 + 6] <= 0.07;

    float *tx = malloc(gsize * sizeof(float));
    float *ty = malloc(gsize * sizeof(float));
    float *tz = malloc(gsize * sizeof(float));
    float *tw = malloc(gsize * sizeof(float));
    double *wsum = calloc(gsize, sizeof(double));
    int32_t *wcnt = calloc(gsize, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) {
        int64_t at = flat_cell[i];
        const Surfel *s = &ec->surfels[mem[i]];
        double wcont = (double)s->winding + (double)ec->curl * (double)s->theta / (2.0 * M_PI);
        wsum[at] += wcont;
        wcnt[at]++;
    }
    double piece_w_sum = 0;
    int64_t piece_w_cnt = 0;
    double bb_min[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, bb_max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    for (size_t j = 0; j < gsize; j++) {
        if (!known[j]) {
            tx[j] = ty[j] = tz[j] = -1.0f;
            tw[j] = NAN;
        } else {
            tx[j] = (float)grid[j];
            ty[j] = (float)grid[gsize + j];
            tz[j] = (float)grid[2 * gsize + j];
            if (tx[j] < bb_min[0])
                bb_min[0] = tx[j];
            if (tx[j] > bb_max[0])
                bb_max[0] = tx[j];
            if (ty[j] < bb_min[1])
                bb_min[1] = ty[j];
            if (ty[j] > bb_max[1])
                bb_max[1] = ty[j];
            if (tz[j] < bb_min[2])
                bb_min[2] = tz[j];
            if (tz[j] > bb_max[2])
                bb_max[2] = tz[j];
            if (wcnt[j] > 0) {
                tw[j] = (float)(wsum[j] / wcnt[j]);
                piece_w_sum += tw[j];
                piece_w_cnt++;
            } else {
                tw[j] = NAN;
            }
        }
    }
    int32_t mean_w = piece_w_cnt ? (int32_t)lrint(piece_w_sum / piece_w_cnt) : 0;
    for (int pass = 0; pass < 12; pass++) {
        int changed = 0;
        for (int r = 0; r < gh; r++) {
            for (int c = 0; c < gw; c++) {
                size_t j = (size_t)r * gw + c;
                if (!known[j] || !isnan(tw[j]))
                    continue;
                double s = 0.0;
                int nc = 0;
                if (r > 0 && known[j - gw] && !isnan(tw[j - gw])) {
                    s += tw[j - gw];
                    nc++;
                }
                if (r + 1 < gh && known[j + gw] && !isnan(tw[j + gw])) {
                    s += tw[j + gw];
                    nc++;
                }
                if (c > 0 && known[j - 1] && !isnan(tw[j - 1])) {
                    s += tw[j - 1];
                    nc++;
                }
                if (c + 1 < gw && known[j + 1] && !isnan(tw[j + 1])) {
                    s += tw[j + 1];
                    nc++;
                }
                if (nc > 0) {
                    tw[j] = (float)(s / nc);
                    changed = 1;
                }
            }
        }
        if (!changed)
            break;
    }
    for (size_t j = 0; j < gsize; j++)
        if (known[j] && isnan(tw[j]))
            tw[j] = (float)mean_w;

    char pdir[4096], fpath[4096], uuid[64];
    snprintf(uuid, sizeof(uuid), "piece-%04d", ec->written_count);
    snprintf(pdir, sizeof(pdir), "%s/%s", ec->out_dir, uuid);
    make_dirs(pdir);
    snprintf(fpath, sizeof(fpath), "%s/x.tif", pdir);
    write_tiff_f32(fpath, tx, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/y.tif", pdir);
    write_tiff_f32(fpath, ty, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/z.tif", pdir);
    write_tiff_f32(fpath, tz, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/w.tif", pdir);
    write_tiff_f32(fpath, tw, gh, gw);
    uint8_t *qual = calloc(gsize, 1);
    for (size_t j = 0; j < gsize; j++)
        qual[j] = locked[j] ? 1 : (known[j] ? 2 : 0);
    snprintf(fpath, sizeof(fpath), "%s/quality.npy", pdir);
    write_npy_u8(fpath, qual, gh, gw);
    free(qual);

    json_object *meta = json_object_new_object();
    json_object_object_add(meta, "sample", json_object_new_string(ec->sample));
    json_object_object_add(meta, "scan", json_object_new_string(ec->scan));
    json_object_object_add(meta, "voxel_um", json_object_new_double(ec->voxel_um));
    json_object_object_add(meta, "piece", json_object_new_int(ec->written_count));
    json_object_object_add(meta, "winding", json_object_new_int(mean_w));
    json_object_object_add(meta, "cells", json_object_new_int64(lock_cnt));
    json_object_object_add(meta, "nodes", json_object_new_int(n));
    json_object_object_add(
        meta, "crowding", json_object_new_double((double)n / fmax(1.0, (double)occ)));
    json_object_object_add(meta, "fold_share", json_object_new_double(fold_share));
    json_object_object_add(meta, "area_mm2", json_object_new_double(area_mm2));
    json_object_object_add(
        meta, "width_mm", json_object_new_double(gw * ec->stride * ec->voxel_um * 1e-3));
    json_object_object_add(
        meta, "height_mm", json_object_new_double(gh * ec->stride * ec->voxel_um * 1e-3));
    json_object_object_add(meta, "written", json_object_new_boolean(1));
    json_object_object_add(meta, "format", json_object_new_string("tifxyz"));
    json_object_object_add(meta, "type", json_object_new_string("seg"));
    json_object_object_add(meta, "uuid", json_object_new_string(uuid));
    json_object_object_add(meta, "stride", json_object_new_double((double)ec->stride));
    json_object_object_add(
        meta, "locked_share", json_object_new_double((double)lock_cnt / (double)gsize));
    json_object *sc = json_object_new_array_ext(2);
    json_object_array_add(sc, json_object_new_double(1.0 / ec->stride));
    json_object_array_add(sc, json_object_new_double(1.0 / ec->stride));
    json_object_object_add(meta, "scale", sc);
    json_object *bb = json_object_new_array_ext(2);
    json_object *b0 = json_object_new_array_ext(3), *b1 = json_object_new_array_ext(3);
    for (int a = 0; a < 3; a++) {
        json_object_array_add(b0, json_object_new_double(bb_min[a]));
        json_object_array_add(b1, json_object_new_double(bb_max[a]));
    }
    json_object_array_add(bb, b0);
    json_object_array_add(bb, b1);
    json_object_object_add(meta, "bbox", bb);
    json_object *iso_obj = json_object_new_object();
    json_object *lk = json_object_new_object();
    json_object_object_add(lk, "within_10pct", json_object_new_double(iso[7 + 5]));
    json_object_object_add(lk, "jitter", json_object_new_double(iso[7 + 6]));
    json_object_object_add(iso_obj, "locked", lk);
    json_object_object_add(iso_obj, "isometric", json_object_new_boolean(is_iso));
    json_object_object_add(meta, "isometry", iso_obj);
    snprintf(fpath, sizeof(fpath), "%s/meta.json", pdir);
    json_object_to_file_ext(fpath, meta, JSON_C_TO_STRING_PRETTY);
    json_object_array_add(ec->rec_pieces, meta);

    ec->total_cm2 += area_cm2;
    if (area_cm2 >= 4.0) {
        ec->area_4cm2 += area_cm2;
        ec->pieces_4++;
    }
    if (area_cm2 > ec->largest_cm2)
        ec->largest_cm2 = area_cm2;
    ec->iso_pass += is_iso;
    ec->written_count++;

    free(tx);
    free(ty);
    free(tz);
    free(tw);
    free(wsum);
    free(wcnt);
    free(known);
    free(hole);
    free(pu);
    free(pv);
    free(loff);
    free(lto);
    free(xyz);
    free(nrm);
    free(du);
    free(dv);
    free(flat_cell);
    free(grid);
    free(locked);
    free(spread);
}

typedef struct {
    const Surfel *surfels;
    const int32_t *cell_first;
    const uint8_t *su8, *bu8, *au8;
    const int8_t *pi8;
    size_t vox;
    int64_t X, Y, Z, GX, GY, GZ;
    int64_t box[6];
    int stride;
    float max_dt, max_dn;
    SEdge **edges_ptr;
    int64_t *nedges_ptr, *ecap_ptr;
    pthread_mutex_t mu;
} DirectEdgeJob;

static void direct_edge_worker(void *ctx, size_t cz_sz) {
    DirectEdgeJob *j = ctx;
    int cz = (int)cz_sz;
    int64_t lcap = 65536, lcnt = 0;
    SEdge *loc = malloc((size_t)lcap * sizeof(*loc));
    if (!loc)
        return;
    const Surfel *surfels = j->surfels;
    const int32_t *cell_first = j->cell_first;
    const uint8_t *su8 = j->su8, *bu8 = j->bu8, *au8 = j->au8;
    const int8_t *pi8 = j->pi8;
    size_t vox = j->vox;
    int64_t X = j->X, Y = j->Y, Z = j->Z, GX = j->GX, GY = j->GY, GZ = j->GZ;
    int stride = j->stride;
    float max_dt = j->max_dt, max_dn = j->max_dn;
    for (int cy = 0; cy < GY; cy++) {
        for (int cx = 0; cx < GX; cx++) {
            int32_t cell = (int32_t)(((int64_t)cz * GY + cy) * GX + cx);
            for (int32_t u = cell_first[cell]; u < cell_first[cell + 1]; u++) {
                const Surfel *su = &surfels[u];
                for (int dz = -1; dz <= 1; dz++) {
                    int nz_c = cz + dz;
                    if (nz_c < 0 || nz_c >= GZ)
                        continue;
                    for (int dy = -1; dy <= 1; dy++) {
                        int ny_c = cy + dy;
                        if (ny_c < 0 || ny_c >= GY)
                            continue;
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx_c = cx + dx;
                            if (nx_c < 0 || nx_c >= GX)
                                continue;
                            if (dz == 0 && dy == 0 && dx == 0)
                                continue;
                            int32_t ncell = (int32_t)(((int64_t)nz_c * GY + ny_c) * GX + nx_c);
                            if (ncell < cell)
                                continue;
                            for (int32_t v = cell_first[ncell]; v < cell_first[ncell + 1]; v++) {
                                const Surfel *sv = &surfels[v];
                                float ca = su->nx * sv->nx + su->ny * sv->ny + su->nz * sv->nz;
                                if (ca < 0.85f)
                                    continue;
                                float vx = sv->x - su->x, vy = sv->y - su->y, vz = sv->z - su->z;
                                float nax = su->nx + sv->nx, nay = su->ny + sv->ny,
                                      naz = su->nz + sv->nz;
                                float nl = sqrtf(nax * nax + nay * nay + naz * naz) + 1e-6f;
                                nax /= nl;
                                nay /= nl;
                                naz /= nl;
                                float dn = fabsf(vx * nax + vy * nay + vz * naz);
                                float d2 = vx * vx + vy * vy + vz * vz;
                                float dt = sqrtf(fmaxf(0.0f, d2 - dn * dn));
                                if (dt > max_dt || dt < 0.15f * stride)
                                    continue;
                                if (dn > max_dn || dn > 0.28f * dt + 0.7f)
                                    continue;
                                float mx = 0.5f * (su->x + sv->x) - j->box[4];
                                float my = 0.5f * (su->y + sv->y) - j->box[2];
                                float mz = 0.5f * (su->z + sv->z) - j->box[0];
                                float s0 =
                                    sample_fused_trilinear(su8, bu8, pi8, Z, Y, X, mz, my, mx);
                                float sp0 = sample_fused_trilinear(su8, bu8, pi8, Z, Y, X,
                                    mz + 0.75f * naz, my + 0.75f * nay, mx + 0.75f * nax);
                                float sm0 = sample_fused_trilinear(su8, bu8, pi8, Z, Y, X,
                                    mz - 0.75f * naz, my - 0.75f * nay, mx - 0.75f * nax);
                                float smid = fmaxf(s0, fmaxf(sp0, sm0));
                                if (smid < 0.12f)
                                    continue;
                                if (bu8 && sample_u8_trilinear(bu8, Z, Y, X, mz, my, mx) > 0.57f)
                                    continue;
                                float aff_mul = 1.0f;
                                if (au8) {
                                    float a_dir =
                                        sample_dir_aff(au8, vox, Z, Y, X, mz, my, mx, vx, vy, vz);
                                    if (a_dir < 0.22f)
                                        continue;
                                    aff_mul = 0.40f + 0.90f * a_dir;
                                }
                                float sp = sample_fused_trilinear(su8, bu8, pi8, Z, Y, X,
                                    mz + 2.2f * naz, my + 2.2f * nay, mx + 2.2f * nax);
                                float sm = sample_fused_trilinear(su8, bu8, pi8, Z, Y, X,
                                    mz - 2.2f * naz, my - 2.2f * nay, mx - 2.2f * nax);
                                if (smid < 0.72f * fmaxf(sp, sm))
                                    continue;
                                if (pi8) {
                                    float dotp = su->cos_w * sv->cos_w + su->sin_w * sv->sin_w;
                                    if (dotp < 0.0f)
                                        continue;
                                }
                                float w = sqrtf(fmaxf(0.0f, fminf(su->score, sv->score) * smid)) *
                                    ca * expf(-0.5f * (dn / 1.2f) * (dn / 1.2f)) * aff_mul;
                                if (lcnt == lcap) {
                                    lcap *= 2;
                                    loc = realloc(loc, (size_t)lcap * sizeof(*loc));
                                }
                                loc[lcnt++] = (SEdge){u, v, w, 0, 0};
                            }
                        }
                    }
                }
            }
        }
    }
    if (lcnt > 0) {
        pthread_mutex_lock(&j->mu);
        if (*j->nedges_ptr + lcnt > *j->ecap_ptr) {
            while (*j->nedges_ptr + lcnt > *j->ecap_ptr)
                *j->ecap_ptr *= 2;
            *j->edges_ptr = realloc(*j->edges_ptr, (size_t)(*j->ecap_ptr) * sizeof(SEdge));
        }
        memcpy(*j->edges_ptr + *j->nedges_ptr, loc, (size_t)lcnt * sizeof(SEdge));
        *j->nedges_ptr += lcnt;
        pthread_mutex_unlock(&j->mu);
    }
    free(loc);
}

typedef struct {
    const Surfel *surfels;
    const SEdge *edges;
    const int64_t *adj_off;
    const int32_t *adj_eid;
    const int32_t *adj_to;
    const int32_t *cell_first;
    const int32_t *patch_root;
    const int32_t *patch_wc;
    const int32_t *root_sz;
    const uint8_t *bu8;
    const int8_t *pi8;
    size_t vox;
    int64_t nsurf, X, Y, Z, GX, GY, GZ;
    int64_t box[6];
    int stride, curl, cur_min_root;
    double cur_reach;
    float cur_min_ca, cur_max_dn_arc, cur_max_dn_uv;
    StitchLink **slinks_ptr;
    int64_t *slcnt_ptr, *slcap_ptr;
    pthread_mutex_t mu;
} StitchRayJob;

static void stitch_ray_worker(void *ctx, size_t blk) {
    StitchRayJob *j = ctx;
    int64_t u0 = (int64_t)blk * 32768;
    int64_t u1 = u0 + 32768;
    if (u1 > j->nsurf)
        u1 = j->nsurf;
    static const double cos12[12] = {
        1.0, 0.8660254, 0.5, 0.0, -0.5, -0.8660254, -1.0, -0.8660254, -0.5, 0.0, 0.5, 0.8660254};
    static const double sin12[12] = {
        0.0, 0.5, 0.8660254, 1.0, 0.8660254, 0.5, 0.0, -0.5, -0.8660254, -1.0, -0.8660254, -0.5};
    int64_t lcap = 4096, lcnt = 0;
    StitchLink *loc = malloc((size_t)lcap * sizeof(*loc));
    if (!loc)
        return;
    const Surfel *surfels = j->surfels;
    const SEdge *edges = j->edges;
    const int64_t *adj_off = j->adj_off;
    const int32_t *adj_eid = j->adj_eid;
    const int32_t *adj_to = j->adj_to;
    const int32_t *cell_first = j->cell_first;
    const int32_t *patch_root = j->patch_root;
    const int32_t *patch_wc = j->patch_wc;
    const int32_t *root_sz = j->root_sz;
    const uint8_t *bu8 = j->bu8;
    const int8_t *pi8 = j->pi8;
    size_t vox = j->vox;
    int64_t X = j->X, Y = j->Y, Z = j->Z, GX = j->GX, GY = j->GY, GZ = j->GZ;
    int stride = j->stride, curl = j->curl, cur_min_root = j->cur_min_root;
    double cur_reach = j->cur_reach;
    float cur_min_ca = j->cur_min_ca, cur_max_dn_arc = j->cur_max_dn_arc,
          cur_max_dn_uv = j->cur_max_dn_uv;
    for (int64_t u = u0; u < u1; u++) {
        const Surfel *su = &surfels[u];
        int32_t ra = patch_root[su->patch];
        if (root_sz[ra] < cur_min_root)
            continue;
        int same_deg = 0;
        for (int64_t e = adj_off[u]; e < adj_off[u + 1]; e++) {
            if (edges[adj_eid[e]].cut)
                continue;
            if (patch_root[surfels[adj_to[e]].patch] == ra)
                same_deg++;
        }
        if (same_deg >= 6)
            continue;
        double e1x = -su->ny, e1y = su->nx, e1z = 0.0;
        double l1 = hypot(e1x, e1y);
        if (l1 < 1e-3) {
            e1x = 1.0;
            e1y = 0.0;
        } else {
            e1x /= l1;
            e1y /= l1;
        }
        double dn1 = e1x * su->nx + e1y * su->ny;
        e1x -= dn1 * su->nx;
        e1y -= dn1 * su->ny;
        e1z -= dn1 * su->nz;
        l1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z) + 1e-9;
        e1x /= l1;
        e1y /= l1;
        e1z /= l1;
        double e2x = su->ny * e1z - su->nz * e1y;
        double e2y = su->nz * e1x - su->nx * e1z;
        double e2z = su->nx * e1y - su->ny * e1x;
        int32_t wca = patch_wc[su->patch];
        for (int d = 0; d < 12; d++) {
            double tx = cos12[d] * e1x + sin12[d] * e2x;
            double ty = cos12[d] * e1y + sin12[d] * e2y;
            double tz = cos12[d] * e1z + sin12[d] * e2z;
            int32_t best_v = -1;
            float best_d = 1e9f, best_al = 0.0f;
            for (double dist = (double)stride; dist <= cur_reach && best_v < 0;
                dist += 0.75 * stride) {
                double qx = (su->x - j->box[4]) + dist * tx;
                double qy = (su->y - j->box[2]) + dist * ty;
                double qz = (su->z - j->box[0]) + dist * tz;
                if (qx < 1 || qx >= X - 1 || qy < 1 || qy >= Y - 1 || qz < 1 || qz >= Z - 1)
                    break;
                for (int ns = -1; ns <= 1; ns++) {
                    double sx = qx + ns * 0.55 * stride * su->nx;
                    double sy = qy + ns * 0.55 * stride * su->ny;
                    double sz = qz + ns * 0.55 * stride * su->nz;
                    int cz = clampi((int)(sz / stride), 0, (int)GZ - 1);
                    int cy = clampi((int)(sy / stride), 0, (int)GY - 1);
                    int cx = clampi((int)(sx / stride), 0, (int)GX - 1);
                    int32_t cell = (int32_t)(((int64_t)cz * GY + cy) * GX + cx);
                    for (int32_t v = cell_first[cell]; v < cell_first[cell + 1]; v++) {
                        const Surfel *sv = &surfels[v];
                        int32_t rb = patch_root[sv->patch];
                        if (rb == ra || root_sz[rb] < cur_min_root)
                            continue;
                        if (patch_wc[sv->patch] != wca)
                            continue;
                        int32_t st = curl * seam_turns(su->theta, sv->theta);
                        if (sv->winding - su->winding != st)
                            continue;
                        float ca = su->nx * sv->nx + su->ny * sv->ny + su->nz * sv->nz;
                        if (ca < cur_min_ca)
                            continue;
                        float vx = sv->x - su->x, vy = sv->y - su->y, vz = sv->z - su->z;
                        float d3 = sqrtf(vx * vx + vy * vy + vz * vz);
                        if (d3 < 0.7f * stride || d3 > (float)cur_reach || d3 >= best_d)
                            continue;
                        float nax = su->nx + sv->nx, nay = su->ny + sv->ny, naz = su->nz + sv->nz;
                        float nl = sqrtf(nax * nax + nay * nay + naz * naz) + 1e-6f;
                        nax /= nl;
                        nay /= nl;
                        naz /= nl;
                        float dn_arc = fabsf(vx * nax + vy * nay + vz * naz);
                        float dn_u = fabsf(vx * su->nx + vy * su->ny + vz * su->nz);
                        float dn_v = fabsf(vx * sv->nx + vy * sv->ny + vz * sv->nz);
                        if (dn_arc > cur_max_dn_arc || dn_u > cur_max_dn_uv ||
                            dn_v > cur_max_dn_uv || dn_arc > 0.08f * d3 + 0.8f)
                            continue;
                        int blocked = 0;
                        double phase_acc = 0.0;
                        double prev_phi = pi8 ? atan2(su->sin_w, su->cos_w) : 0.0;
                        int nsteps = (int)(d3 / 3.0f);
                        for (int stp = 1; stp < nsteps && !blocked; stp++) {
                            double f = (double)stp / (double)nsteps;
                            double mx = (su->x - j->box[4]) + f * vx;
                            double my = (su->y - j->box[2]) + f * vy;
                            double mz = (su->z - j->box[0]) + f * vz;
                            if (mx < 1 || mx >= X - 1 || my < 1 || my >= Y - 1 || mz < 1 ||
                                mz >= Z - 1) {
                                blocked = 1;
                                break;
                            }
                            size_t vi = ((size_t)(int)mz * Y + (int)my) * X + (int)mx;
                            if (bu8 && bu8[vi] > 150) {
                                blocked = 1;
                                break;
                            }
                            if (pi8) {
                                double cur_phi = atan2((double)pi8[vox + vi], (double)pi8[vi]);
                                double dphi = cur_phi - prev_phi;
                                if (dphi > M_PI)
                                    dphi -= 2.0 * M_PI;
                                else if (dphi < -M_PI)
                                    dphi += 2.0 * M_PI;
                                phase_acc += dphi;
                                prev_phi = cur_phi;
                            }
                            int mcz = clampi((int)(mz / stride), 0, (int)GZ - 1);
                            int mcy = clampi((int)(my / stride), 0, (int)GY - 1);
                            int mcx = clampi((int)(mx / stride), 0, (int)GX - 1);
                            int32_t mcell = (int32_t)(((int64_t)mcz * GY + mcy) * GX + mcx);
                            for (int32_t w = cell_first[mcell]; w < cell_first[mcell + 1]; w++) {
                                int32_t rc = patch_root[surfels[w].patch];
                                if (rc == ra || rc == rb)
                                    continue;
                                const Surfel *sw = &surfels[w];
                                double dxw = (sw->x - j->box[4]) - mx;
                                double dyw = (sw->y - j->box[2]) - my;
                                double dzw = (sw->z - j->box[0]) - mz;
                                double dnw = fabs(dxw * sw->nx + dyw * sw->ny + dzw * sw->nz);
                                double d2w = dxw * dxw + dyw * dyw + dzw * dzw;
                                double dtw2 = d2w - dnw * dnw;
                                if (dnw < 2.5 && dtw2 < (0.85 * stride) * (0.85 * stride)) {
                                    blocked = 1;
                                    break;
                                }
                            }
                        }
                        if (blocked)
                            continue;
                        if (pi8 && fabs(phase_acc / (2.0 * M_PI)) > 0.28)
                            continue;
                        best_v = v;
                        best_d = d3;
                        best_al = ca * (1.0f - dn_arc / 3.5f);
                    }
                }
            }
            if (best_v >= 0) {
                int32_t ca0 = ra, cb0 = patch_root[surfels[best_v].patch];
                int32_t uu = (int32_t)u, vv = best_v;
                if (ca0 > cb0) {
                    int32_t sw = ca0;
                    ca0 = cb0;
                    cb0 = sw;
                    sw = uu;
                    uu = vv;
                    vv = sw;
                }
                if (lcnt == lcap) {
                    lcap *= 2;
                    loc = realloc(loc, (size_t)lcap * sizeof(*loc));
                }
                loc[lcnt++] = (StitchLink){ca0, cb0, uu, vv, best_d, best_al};
            }
        }
    }
    if (lcnt > 0) {
        pthread_mutex_lock(&j->mu);
        if (*j->slcnt_ptr + lcnt > *j->slcap_ptr) {
            while (*j->slcnt_ptr + lcnt > *j->slcap_ptr)
                *j->slcap_ptr *= 2;
            *j->slinks_ptr = realloc(*j->slinks_ptr, (size_t)(*j->slcap_ptr) * sizeof(StitchLink));
        }
        memcpy(*j->slinks_ptr + *j->slcnt_ptr, loc, (size_t)lcnt * sizeof(StitchLink));
        *j->slcnt_ptr += lcnt;
        pthread_mutex_unlock(&j->mu);
    }
    free(loc);
}

typedef struct {
    int32_t grounds, main_ground, graphs;
    int64_t matched, main_count;
} AnchorStats;

static int name_order(const struct dirent **a, const struct dirent **b) {
    return strcmp((*a)->d_name, (*b)->d_name);
}

static int anchor_terms(const char *dir, const Surfel *surfels, const int32_t *cell_first,
    const float *patch_theta, const int64_t *box, int stride, int GZ, int GY, int GX, int curl,
    int32_t npatches, double weight, PTerm **terms_ptr, int64_t *tcnt_ptr, int64_t *tcap_ptr,
    AnchorStats *stats) {
    enum { BATCH = 65536 };
    char path[4096];
    snprintf(path, sizeof(path), "%s/surfels.bin", dir);
    FILE *fs = fopen(path, "rb"), *fp = fopen(path, "rb");
    int64_t hdr[2] = {0, 0};
    Surfel *as = malloc(BATCH * sizeof(*as));
    int32_t *ap = malloc(BATCH * sizeof(*ap));
    int32_t *ground = NULL;
    int64_t *ground_count = NULL;
    int32_t ground_cap = 0, base = stats->grounds;
    int failed = !fs || !fp || !as || !ap || fread(hdr, sizeof(int64_t), 2, fs) != 2 ||
        hdr[0] < 0 || fseeko(fp, (off_t)(16 + hdr[0] * (int64_t)sizeof(Surfel)), SEEK_SET);
    for (int64_t done = 0; !failed && done < hdr[0];) {
        size_t n = (size_t)(hdr[0] - done < BATCH ? hdr[0] - done : BATCH);
        if (fread(as, sizeof(*as), n, fs) != n || fread(ap, sizeof(*ap), n, fp) != n) {
            failed = 1;
            break;
        }
        done += (int64_t)n;
        for (size_t i = 0; i < n; i++) {
            const Surfel *sa = &as[i];
            if (ap[i] < 0 || sa->patch < 0)
                continue;
            double lx = sa->x - box[4], ly = sa->y - box[2], lz = sa->z - box[0];
            if (lx < 0 || ly < 0 || lz < 0)
                continue;
            int cz = (int)(lz / stride), cy = (int)(ly / stride), cx = (int)(lx / stride);
            if (cz >= GZ || cy >= GY || cx >= GX)
                continue;
            int32_t best = -1;
            float best_d2 = 1.0f;
            for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int nz = cz + dz, ny = cy + dy, nx = cx + dx;
                        if (nz < 0 || nz >= GZ || ny < 0 || ny >= GY || nx < 0 || nx >= GX)
                            continue;
                        int64_t cell = ((int64_t)nz * GY + ny) * GX + nx;
                        for (int32_t v = cell_first[cell]; v < cell_first[cell + 1]; v++) {
                            float vx = surfels[v].x - sa->x, vy = surfels[v].y - sa->y,
                                  vz = surfels[v].z - sa->z;
                            float d2 = vx * vx + vy * vy + vz * vz;
                            if (d2 < best_d2) {
                                best_d2 = d2;
                                best = v;
                            }
                        }
                    }
            if (best < 0)
                continue;
            if (sa->patch >= ground_cap) {
                int32_t cap = sa->patch + 65536;
                int32_t *grown = realloc(ground, (size_t)cap * sizeof(*ground));
                int64_t *counts =
                    grown ? realloc(ground_count, (size_t)cap * sizeof(*counts)) : NULL;
                if (grown)
                    ground = grown;
                if (!counts) {
                    failed = 1;
                    break;
                }
                ground_count = counts;
                memset(ground + ground_cap, -1, (size_t)(cap - ground_cap) * sizeof(*ground));
                memset(ground_count + ground_cap, 0, (size_t)(cap - ground_cap) * sizeof(*counts));
                ground_cap = cap;
            }
            if (ground[sa->patch] < 0)
                ground[sa->patch] = stats->grounds++;
            int32_t g = ground[sa->patch];
            if (++ground_count[g - base] > stats->main_count) {
                stats->main_count = ground_count[g - base];
                stats->main_ground = g;
            }
            if (*tcnt_ptr == *tcap_ptr) {
                *tcap_ptr *= 2;
                *terms_ptr = realloc(*terms_ptr, (size_t)(*tcap_ptr) * sizeof(PTerm));
            }
            int32_t p = surfels[best].patch;
            int32_t k = curl * seam_turns(patch_theta[p], surfels[best].theta) - sa->winding;
            (*terms_ptr)[(*tcnt_ptr)++] = (PTerm){p, npatches + g, k, 4, weight, weight};
            stats->matched++;
        }
    }
    if (fs)
        fclose(fs);
    if (fp)
        fclose(fp);
    free(ground);
    free(ground_count);
    free(as);
    free(ap);
    stats->graphs += !failed;
    return failed ? -1 : 0;
}

typedef struct {
    int64_t nsolve, mterms;
    const int32_t *tp, *tq, *tk;
    int32_t *up, *down, *windings;
} DrawJob;

static void draw_worker(void *ctx, size_t d) {
    DrawJob *j = ctx;
    int64_t dlog[3] = {0};
    hx_level_minimize(j->nsolve, j->mterms, j->tp, j->tq, j->tk, j->up + d * j->mterms,
        j->down + d * j->mterms, j->windings + d * j->nsolve, 8, dlog);
}

static struct {
    const char *name;
    double seconds;
} stage_log[40];
static int stage_count;
static double stage_from;

static void stage_done(const char *name) {
    double now = clock_sec();
    if (stage_count < 40) {
        stage_log[stage_count].name = name;
        stage_log[stage_count++].seconds = now - stage_from;
    }
    stage_from = now;
}

#ifdef HERCULEX_CUDA_FLOW
static int64_t flow_opened, flow_refused, flow_runs, flow_failed, flow_micros;

static void *counted_flow_open(int32_t n, int32_t m, const int32_t *first, const void *arcs) {
    void *flow = hx_flow_open(n, m, first, arcs);
    __atomic_add_fetch(flow ? &flow_opened : &flow_refused, 1, __ATOMIC_RELAXED);
    return flow;
}

static int64_t counted_flow_run(void *flow, int32_t *cap, int64_t *terminal) {
    double from = clock_sec();
    int64_t sent = hx_flow_run(flow, cap, terminal);
    __atomic_add_fetch(&flow_micros, (int64_t)(1e6 * (clock_sec() - from)), __ATOMIC_RELAXED);
    __atomic_add_fetch(sent < 0 ? &flow_failed : &flow_runs, 1, __ATOMIC_RELAXED);
    return sent;
}
#endif

#ifdef TRACE_CUDA
static int cuda_slab(const uint8_t *su8, size_t gpl, int Z, const int64_t *box,
    const HxAxisKnot *axis, int64_t axis_count, int pz0, int tZ, int tile_z, double *centres) {
    int hZ = (tZ + 1) / 2;
    double *cxf = centres, *cyf = cxf + tZ, *cxh = cyf + tZ, *cyh = cxh + hZ;
    for (int z = 0; z < tZ; z++)
        hx_centre_at(axis, axis_count, (double)(box[0] + pz0 + z), &cxf[z], &cyf[z]);
    for (int hz = 0; hz < hZ; hz++)
        hx_centre_at(axis, axis_count, (double)(box[0] + pz0 + 2 * hz) + 0.5, &cxh[hz], &cyh[hz]);
    int next0 = pz0 + tZ, next1 = next0 + tile_z < Z ? next0 + tile_z : Z;
    if (next1 > next0) {
        uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
        uintptr_t from = (uintptr_t)(su8 + (size_t)next0 * gpl) & ~(page - 1);
        madvise((void *)from, (uintptr_t)(su8 + (size_t)next1 * gpl) - from, MADV_WILLNEED);
    }
    return trace_cuda_crest_slab(su8 + (size_t)pz0 * gpl, pz0, tZ, centres);
}

static int cuda_surfels(const TraceCudaSurfel *made, int64_t count, const HxAxisKnot *axis,
    int64_t axis_count, Surfel **surfels_ptr, int64_t *nsurf_ptr, int64_t *cap_ptr) {
    if (*nsurf_ptr + count > *cap_ptr) {
        int64_t cap = *cap_ptr;
        while (*nsurf_ptr + count > cap)
            cap *= 2;
        Surfel *grown = realloc(*surfels_ptr, (size_t)cap * sizeof(*grown));
        if (!grown)
            return -1;
        *surfels_ptr = grown;
        *cap_ptr = cap;
    }
    Surfel *out = *surfels_ptr + *nsurf_ptr;
    for (int64_t i = 0; i < count; i++) {
        const TraceCudaSurfel *m = &made[i];
        double gx = m->x, gy = m->y, gz = m->z;
        double ax_x, ax_y;
        hx_centre_at(axis, axis_count, gz, &ax_x, &ax_y);
        double nx0 = m->nx, ny0 = m->ny, nz0 = m->nz;
        if (nx0 * (gx - ax_x) + ny0 * (gy - ax_y) < 0.0) {
            nx0 = -nx0;
            ny0 = -ny0;
            nz0 = -nz0;
        }
        out[i] = (Surfel){.x = (float)gx,
            .y = (float)gy,
            .z = (float)gz,
            .nx = (float)nx0,
            .ny = (float)ny0,
            .nz = (float)nz0,
            .score = m->score,
            .cos_w = m->cos_w,
            .sin_w = m->sin_w,
            .spacing = m->spacing,
            .theta = (float)atan2(gy - ax_y, gx - ax_x),
            .cell = m->cell,
            .count = m->count,
            .patch = -1,
            .winding = 0};
    }
    *nsurf_ptr += count;
    return 0;
}
#endif

static int usage(void) {
    fputs(
        "Usage: unroll-trace (--surface UNIT.npy | --field DIR) --box z0:z1,y0:y1,x0:x1 "
        "--axis AXIS.json --out DIR [OPTION...]\n"
        "Input and output\n"
        "  --surface UNIT.npy       m7 surface prediction of the box (uint8, z y x)\n"
        "  --field DIR              a field folder instead of --surface\n"
        "  --box z0:z1,y0:y1,x0:x1  where the array lies in the scan, in voxels\n"
        "  --axis AXIS.json         the scan's umbilicus\n"
        "  --out DIR                the unit's record, and its pieces unless --skip-emit\n"
        "  --save-graph DIR         write the unit graph (surfels.bin) that unroll-stitch reads\n"
        "  --skip-emit              write no pieces; unroll-stitch emits them\n"
        "  --sample NAME --scan ID  names for the records\n"
        "Scan\n"
        "  --curl +1|-1             hand-set curl (default +1)\n"
        "  --voxel UM               voxel size in micrometres (default 9.362)\n"
        "  --stride N               surfel grid in voxels (default 8)\n"
        "Sheet tracing\n"
        "  --smin X                 lowest surface value on a crest (default 0.14)\n"
        "  --curvmin X              lowest crest curvature (default 0.006)\n"
        "  --bridge VOX             tangential gap-bridge reach in voxels (default 28)\n"
        "  --patch-r X              largest patch radius in strides (default 4)\n"
        "  --stitch VOX             first stitch reach (default 40)\n"
        "  --consolidate VOX        second stitch reach (default 84)\n"
        "  --min-nodes N            smallest piece in surfels (default 48)\n"
        "Winding solve\n"
        "  --anchor PATH            tie the unit to a saved unit graph, or to every overlapping "
        "graph in a folder\n"
        "  --anchor-weight W        weight per matched anchor surfel (default 20)\n"
        "  --skip-cost X            relative cost of a ray step over a layer (default 0.35)\n"
        "  --min-votes N            reweighted draws of 8 that a merge needs (default 6)\n"
        "  --draw-jobs N            draws solved at once (default min(8, online CPU cores))\n"
        "Pieces written without --skip-emit\n"
        "  --fill-reach N  --fair N  --max-hole N  --arap N  --no-lock\n"
        "Field folders\n"
        "  --no-bar  --no-nrm  --no-pha  --no-aff   ignore the barrier, normal, phase or affinity "
        "map\n"
        "CUDA builds\n"
        "  --no-cuda  --no-cuda-crest  --no-cuda-flow   every stage, the crest or the min-cut on "
        "the "
        "CPU\n"
        "Diagnostics\n"
        "  --timings  --verbose\n",
        stderr);
    return 2;
}

int main(int argc, char **argv) {
    const char *field_dir = NULL, *surface_path = NULL, *axis_path = NULL, *out_dir = NULL;
    const char *save_graph_dir = NULL;
    const char *anchor_dir = NULL;
    double anchor_weight = 20.0;
    const char *sample = "unknown", *scan = "unknown";
    int64_t box[6] = {0, 0, 0, 0, 0, 0};
    int have_box = 0, curl = 1, stride = 8, min_nodes = 48, arap_rounds = 0, no_lock = 0;
    int no_bar = 0, no_nrm = 0, no_pha = 0, no_aff = 0, skip_emit = 0;
    int tile_z = 160, tile_yx = 480, tile_pad = 20, min_votes = 6, fill_reach = 5, fair_rounds = 20;
    int max_hole_cells = 3600, verbose = 0;
    double voxel_um = 9.362, s_min = 0.14, curv_min = 0.006, min_layer_gap = 2.2,
           bridge_reach = 28.0;
    double max_patch_mul = 4.0, stitch_reach = 40.0, consolidate_reach = 84.0;
    int cuda_crest_wanted = 1, cuda_flow_wanted = 1, timings = 0;
    long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int draw_jobs = online_cpus < 1 ? 1 : (int)online_cpus;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--field") && i + 1 < argc)
            field_dir = argv[++i];
        else if (!strcmp(argv[i], "--surface") && i + 1 < argc)
            surface_path = argv[++i];
        else if (!strcmp(argv[i], "--axis") && i + 1 < argc)
            axis_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_dir = argv[++i];
        else if (!strcmp(argv[i], "--save-graph") && i + 1 < argc)
            save_graph_dir = argv[++i];
        else if (!strcmp(argv[i], "--box") && i + 1 < argc)
            have_box = !parse_box_arg(argv[++i], box);
        else if (!strcmp(argv[i], "--curl") && i + 1 < argc)
            curl = atoi(argv[++i]) >= 0 ? 1 : -1;
        else if (!strcmp(argv[i], "--stride") && i + 1 < argc)
            stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--voxel") && i + 1 < argc)
            voxel_um = atof(argv[++i]);
        else if (!strcmp(argv[i], "--smin") && i + 1 < argc)
            s_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--curvmin") && i + 1 < argc)
            curv_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--min-nodes") && i + 1 < argc)
            min_nodes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bridge") && i + 1 < argc)
            bridge_reach = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stitch") && i + 1 < argc)
            stitch_reach = atof(argv[++i]);
        else if (!strcmp(argv[i], "--consolidate") && i + 1 < argc)
            consolidate_reach = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fill-reach") && i + 1 < argc)
            fill_reach = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fair") && i + 1 < argc)
            fair_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-hole") && i + 1 < argc)
            max_hole_cells = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--arap") && i + 1 < argc)
            arap_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-votes") && i + 1 < argc)
            min_votes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--anchor") && i + 1 < argc)
            anchor_dir = argv[++i];
        else if (!strcmp(argv[i], "--anchor-weight") && i + 1 < argc)
            anchor_weight = atof(argv[++i]);
        else if (!strcmp(argv[i], "--skip-cost") && i + 1 < argc)
            g_skip_cost = atof(argv[++i]);
        else if (!strcmp(argv[i], "--patch-r") && i + 1 < argc)
            max_patch_mul = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-lock"))
            no_lock = 1;
        else if (!strcmp(argv[i], "--no-bar"))
            no_bar = 1;
        else if (!strcmp(argv[i], "--no-nrm"))
            no_nrm = 1;
        else if (!strcmp(argv[i], "--no-pha"))
            no_pha = 1;
        else if (!strcmp(argv[i], "--no-aff"))
            no_aff = 1;
        else if (!strcmp(argv[i], "--skip-emit"))
            skip_emit = 1;
        else if (!strcmp(argv[i], "--verbose"))
            verbose = 1;
        else if (!strcmp(argv[i], "--no-cuda"))
            cuda_crest_wanted = cuda_flow_wanted = 0;
        else if (!strcmp(argv[i], "--no-cuda-crest"))
            cuda_crest_wanted = 0;
        else if (!strcmp(argv[i], "--no-cuda-flow"))
            cuda_flow_wanted = 0;
        else if (!strcmp(argv[i], "--timings"))
            timings = 1;
        else if (!strcmp(argv[i], "--draw-jobs") && i + 1 < argc)
            draw_jobs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample") && i + 1 < argc)
            sample = argv[++i];
        else if (!strcmp(argv[i], "--scan") && i + 1 < argc)
            scan = argv[++i];
        else {
            fprintf(stderr, "trace: unknown option or missing value: %s\n", argv[i]);
            return usage();
        }
    }
    if ((!field_dir && !surface_path) || !axis_path || !out_dir || !have_box)
        return usage();
    double t0 = clock_sec();
    stage_from = t0;
    (void)cuda_crest_wanted;
    (void)cuda_flow_wanted;
    if (draw_jobs < 1)
        draw_jobs = 1;
    HxAxisKnot *axis = NULL;
    int64_t axis_count = 0;
    if (load_axis_json(axis_path, &axis, &axis_count)) {
        fprintf(stderr, "trace: cannot load axis %s\n", axis_path);
        return 1;
    }
    char pathbuf[4096];
    NpyArray surf_npy = {0}, bar_npy = {0}, nrm_npy = {0}, pha_npy = {0}, spc_npy = {0},
             aff_npy = {0};
    if (surface_path) {
        if (map_npy(surface_path, &surf_npy)) {
            fprintf(stderr, "trace: cannot map %s\n", surface_path);
            return 1;
        }
    } else {
        snprintf(pathbuf, sizeof(pathbuf), "%s/surface.npy", field_dir);
        if (map_npy(pathbuf, &surf_npy)) {
            fprintf(stderr, "trace: cannot map %s\n", pathbuf);
            return 1;
        }
    }
    if (field_dir) {
        snprintf(pathbuf, sizeof(pathbuf), "%s/barrier.npy", field_dir);
        if (!no_bar)
            map_npy(pathbuf, &bar_npy);
        snprintf(pathbuf, sizeof(pathbuf), "%s/normals.npy", field_dir);
        if (!no_nrm)
            map_npy(pathbuf, &nrm_npy);
        snprintf(pathbuf, sizeof(pathbuf), "%s/phase.npy", field_dir);
        if (!no_pha)
            map_npy(pathbuf, &pha_npy);
        snprintf(pathbuf, sizeof(pathbuf), "%s/aff.npy", field_dir);
        if (!no_aff)
            map_npy(pathbuf, &aff_npy);
        snprintf(pathbuf, sizeof(pathbuf), "%s/spacing.npy", field_dir);
        map_npy(pathbuf, &spc_npy);
    }
    if (surf_npy.ndim != 3 || surf_npy.itemsize != 1) {
        fprintf(stderr, "trace: expected 3D uint8 surface.npy\n");
        return 1;
    }
    int Z = (int)surf_npy.shape[0], Y = (int)surf_npy.shape[1], X = (int)surf_npy.shape[2];
    size_t vox = (size_t)Z * Y * X;
    const uint8_t *su8 = surf_npy.data;
    const uint8_t *bu8 = bar_npy.data;
    const int8_t *ni8 =
        (nrm_npy.data && nrm_npy.ndim == 4 && nrm_npy.shape[0] == 3) ? nrm_npy.data : NULL;
    const int8_t *pi8 =
        (pha_npy.data && pha_npy.ndim == 4 && pha_npy.shape[0] == 2) ? pha_npy.data : NULL;
    const uint8_t *au8 =
        (aff_npy.data && aff_npy.ndim == 4 && aff_npy.shape[0] == 3) ? aff_npy.data : NULL;
    const uint8_t *spu8 = spc_npy.data;
    printf("trace v6: mapped %dx%dx%d (%.1fM vox) bar=%d nrm=%d pha=%d aff=%d spc=%d (%.2fs)\n", Z,
        Y, X, vox / 1e6, bu8 != NULL, ni8 != NULL, pi8 != NULL, au8 != NULL, spu8 != NULL,
        clock_sec() - t0);
    fflush(stdout);

    int GZ = (Z + stride - 1) / stride, GY = (Y + stride - 1) / stride,
        GX = (X + stride - 1) / stride;
    int64_t ncells = (int64_t)GZ * GY * GX;
    tile_z = (tile_z / stride) * stride;
    if (tile_z < stride)
        tile_z = stride;
    tile_yx = (tile_yx / stride) * stride;
    if (tile_yx < stride)
        tile_yx = stride;
    int64_t total_crest = 0;
    int64_t surf_cap = 1 << 18, nsurf = 0;
    Surfel *surfels = malloc((size_t)surf_cap * sizeof(*surfels));
    int min_cell_pts = stride >= 8 ? 5 : (stride >= 6 ? 4 : 3);
    size_t gpl = (size_t)Y * X;
    int tiles_run = 0, tiles_skipped = 0;
    stage_done("load");
#ifdef TRACE_CUDA
    int cuda_crest = 0;
    double *cuda_centres = NULL;
    if (cuda_crest_wanted && !bu8 && !ni8 && !pi8 && !spu8 && !trace_cuda_open()) {
        TraceCudaCrest setup = {.Z = Z,
            .Y = Y,
            .X = X,
            .GY = GY,
            .GX = GX,
            .stride = stride,
            .tile_z = tile_z,
            .tile_yx = tile_yx,
            .tile_pad = tile_pad,
            .min_cell_pts = min_cell_pts,
            .s_min = (float)s_min,
            .curv_min = (float)curv_min,
            .min_layer_gap = (float)min_layer_gap};
        memcpy(setup.box, box, sizeof(box));
        setup.r_sm = gaussian_1d_kernel(0.85f, setup.k_sm, 32);
        setup.r_hs = gaussian_1d_kernel(0.90f, setup.k_hs, 32);
        setup.r_hj = gaussian_1d_kernel(2.5f, setup.k_hj, 32);
        cuda_centres = malloc((size_t)4 * (tile_z + 2 * tile_pad) * sizeof(double));
        cuda_crest = cuda_centres && !trace_cuda_crest_begin(&setup);
    }
#endif
    for (int za = 0; za < Z; za += tile_z) {
        int zb = za + tile_z < Z ? za + tile_z : Z;
        int pz0 = za - tile_pad > 0 ? za - tile_pad : 0;
        int pz1 = zb + tile_pad < Z ? zb + tile_pad : Z;
        int tZ = pz1 - pz0;
#ifdef TRACE_CUDA
        if (cuda_crest &&
            cuda_slab(su8, gpl, Z, box, axis, axis_count, pz0, tZ, tile_z, cuda_centres)) {
            cuda_crest = 0;
            fprintf(stderr, "trace: CUDA crest failed, continuing on the CPU\n");
        }
#endif
        for (int ya = 0; ya < Y; ya += tile_yx) {
            int yb = ya + tile_yx < Y ? ya + tile_yx : Y;
            int py0 = ya - tile_pad > 0 ? ya - tile_pad : 0;
            int py1 = yb + tile_pad < Y ? yb + tile_pad : Y;
            int tY = py1 - py0;
            for (int xa = 0; xa < X; xa += tile_yx) {
                int xb = xa + tile_yx < X ? xa + tile_yx : X;
                int px0 = xa - tile_pad > 0 ? xa - tile_pad : 0;
                int px1 = xb + tile_pad < X ? xb + tile_pad : X;
                int tX = px1 - px0;
                int any_fg = 0;
                for (int z = za; z < zb && !any_fg; z += 4)
                    for (int y = ya; y < yb && !any_fg; y += 4) {
                        const uint8_t *row = su8 + (size_t)z * gpl + (size_t)y * X;
                        for (int x = xa; x < xb; x += 4)
                            if (row[x] >= 24) {
                                any_fg = 1;
                                break;
                            }
                    }
                if (!any_fg) {
                    tiles_skipped++;
                    continue;
                }
                tiles_run++;
#ifdef TRACE_CUDA
                if (cuda_crest) {
                    TraceCudaTile tile = {za, zb, ya, yb, xa, xb, pz0, pz1, py0, py1, px0, px1};
                    const TraceCudaSurfel *made = NULL;
                    int64_t made_count = 0, tile_points = 0;
                    if (!trace_cuda_crest_tile(&tile, &made, &made_count, &tile_points) &&
                        !cuda_surfels(
                            made, made_count, axis, axis_count, &surfels, &nsurf, &surf_cap)) {
                        total_crest += tile_points;
                        continue;
                    }
                    cuda_crest = 0;
                    fprintf(stderr, "trace: CUDA crest failed, continuing on the CPU\n");
                }
#endif
                size_t tvox = (size_t)tZ * tY * tX;
                size_t tpl = (size_t)tY * tX;
                float *s_raw = malloc(tvox * sizeof(float));
                float *s_sm = malloc(tvox * sizeof(float));
                float *tmp = malloc(tvox * sizeof(float));
                for (int z = 0; z < tZ; z++) {
                    size_t gz = (size_t)(pz0 + z) * gpl;
                    for (int y = 0; y < tY; y++) {
                        size_t gbase = gz + (size_t)(py0 + y) * X + px0;
                        float *dst = s_raw + (size_t)z * tpl + (size_t)y * tX;
                        for (int x = 0; x < tX; x++)
                            dst[x] = fused_score_at(su8, bu8, pi8, vox, gbase + x);
                    }
                }
                blur3d(s_raw, s_sm, tmp, tZ, tY, tX, 0.85f);
                free(s_raw);

                float *tnx = malloc(tvox * sizeof(float));
                float *tny = malloc(tvox * sizeof(float));
                float *tnz = malloc(tvox * sizeof(float));
                float *s_crest = s_sm;
                float *s_crest_alloc = NULL;
                if (ni8) {
                    for (int z = 0; z < tZ; z++) {
                        size_t gz = (size_t)(pz0 + z) * gpl;
                        for (int y = 0; y < tY; y++) {
                            size_t gbase = gz + (size_t)(py0 + y) * X + px0;
                            size_t tbase = (size_t)z * tpl + (size_t)y * tX;
                            for (int x = 0; x < tX; x++) {
                                tnx[tbase + x] = ni8[gbase + x] * (1.0f / 127.0f);
                                tny[tbase + x] = ni8[vox + gbase + x] * (1.0f / 127.0f);
                                tnz[tbase + x] = ni8[2 * vox + gbase + x] * (1.0f / 127.0f);
                            }
                        }
                    }
                    blur3d(tnx, tnx, tmp, tZ, tY, tX, 2.0f);
                    blur3d(tny, tny, tmp, tZ, tY, tX, 2.0f);
                    blur3d(tnz, tnz, tmp, tZ, tY, tX, 2.0f);
                    for (int z = 0; z < tZ; z++) {
                        double cx, cy;
                        hx_centre_at(axis, axis_count, (double)(box[0] + pz0 + z), &cx, &cy);
                        for (int y = 0; y < tY; y++) {
                            double ry = (double)(box[2] + py0 + y) - cy;
                            size_t tbase = (size_t)z * tpl + (size_t)y * tX;
                            for (int x = 0; x < tX; x++) {
                                size_t i = tbase + x;
                                float l =
                                    sqrtf(tnx[i] * tnx[i] + tny[i] * tny[i] + tnz[i] * tnz[i]) +
                                    1e-6f;
                                tnx[i] /= l;
                                tny[i] /= l;
                                tnz[i] /= l;
                                double rx = (double)(box[4] + px0 + x) - cx;
                                if (tnx[i] * rx + tny[i] * ry < 0.0) {
                                    tnx[i] = -tnx[i];
                                    tny[i] = -tny[i];
                                    tnz[i] = -tnz[i];
                                }
                            }
                        }
                    }
                } else {
                    int hZ = (tZ + 1) / 2, hY = (tY + 1) / 2, hX = (tX + 1) / 2;
                    size_t hvox = (size_t)hZ * hY * hX;
                    s_crest_alloc = malloc(tvox * sizeof(float));
                    s_crest = s_crest_alloc;
                    HalfTensorCtx hctx = {.Z = tZ,
                        .Y = tY,
                        .X = tX,
                        .hZ = hZ,
                        .hY = hY,
                        .hX = hX,
                        .z0 = box[0] + pz0,
                        .y0 = box[2] + py0,
                        .x0 = box[4] + px0,
                        .s_sm = s_sm,
                        .hs = malloc(hvox * sizeof(float)),
                        .hnx = malloc(hvox * sizeof(float)),
                        .hny = malloc(hvox * sizeof(float)),
                        .hnz = malloc(hvox * sizeof(float)),
                        .nx = tnx,
                        .ny = tny,
                        .nz = tnz,
                        .s_crest = s_crest,
                        .axis = axis,
                        .axis_count = axis_count};
                    for (int k = 0; k < 6; k++)
                        hctx.hj[k] = malloc(hvox * sizeof(float));
                    float *htmp = malloc(hvox * sizeof(float));
                    each_block((size_t)hZ, &hctx, half_tensor_accum_job);
                    blur3d(hctx.hs, hctx.hs, htmp, hZ, hY, hX, 0.90f);
                    for (int k = 0; k < 6; k++)
                        blur3d(hctx.hj[k], hctx.hj[k], htmp, hZ, hY, hX, 2.5f);
                    each_block((size_t)hZ, &hctx, half_tensor_eig_job);
                    each_block((size_t)tZ, &hctx, half_tensor_upsample_job);
                    free(htmp);
                    for (int k = 0; k < 6; k++)
                        free(hctx.hj[k]);
                    free(hctx.hs);
                    free(hctx.hnx);
                    free(hctx.hny);
                    free(hctx.hnz);
                }
                free(tmp);

                int cz0 = za - pz0, cz1 = zb - pz0;
                int cy0 = ya - py0, cy1 = yb - py0;
                int cx0 = xa - px0, cx1 = xb - px0;
                int core_z = cz1 - cz0;
                CrestList *per_z = calloc((size_t)core_z, sizeof(*per_z));
                CrestCtx cctx = {.tZ = tZ,
                    .tY = tY,
                    .tX = tX,
                    .gZ = Z,
                    .gY = Y,
                    .gX = X,
                    .pz0 = pz0,
                    .py0 = py0,
                    .px0 = px0,
                    .cz0 = cz0,
                    .cz1 = cz1,
                    .cy0 = cy0,
                    .cy1 = cy1,
                    .cx0 = cx0,
                    .cx1 = cx1,
                    .s_sm = s_crest,
                    .nx = tnx,
                    .ny = tny,
                    .nz = tnz,
                    .barrier = bu8,
                    .spacing = spu8,
                    .phase = pi8,
                    .s_min = (float)s_min,
                    .curv_min = (float)curv_min,
                    .per_z = per_z};
                each_block((size_t)core_z, &cctx, extract_crest_job);
                int64_t tile_crest = 0;
                for (int z = 0; z < core_z; z++)
                    tile_crest += per_z[z].count;
                total_crest += tile_crest;
                if (tile_crest > 0) {
                    CrestPt *tpts = malloc((size_t)tile_crest * sizeof(*tpts));
                    BinEntry *entries = malloc((size_t)tile_crest * sizeof(*entries));
                    int64_t pos = 0;
                    for (int z = 0; z < core_z; z++) {
                        if (per_z[z].count) {
                            memcpy(
                                tpts + pos, per_z[z].pts, (size_t)per_z[z].count * sizeof(*tpts));
                            pos += per_z[z].count;
                        }
                        free(per_z[z].pts);
                    }
                    int cz_lo = za / stride, cz_hi = (zb - 1) / stride;
                    int cy_lo = ya / stride, cy_hi = (yb - 1) / stride;
                    int cx_lo = xa / stride, cx_hi = (xb - 1) / stride;
                    for (int64_t i = 0; i < tile_crest; i++) {
                        int cz = clampi((int)(tpts[i].z / stride), cz_lo, cz_hi);
                        int cy = clampi((int)(tpts[i].y / stride), cy_lo, cy_hi);
                        int cx = clampi((int)(tpts[i].x / stride), cx_lo, cx_hi);
                        entries[i] =
                            (BinEntry){((int64_t)cz * GY + cy) * GX + cx, (int32_t)i, 0.0f};
                    }
                    qsort(entries, (size_t)tile_crest, sizeof(*entries), bin_cell_cmp);
                    for (int64_t a = 0; a < tile_crest;) {
                        int64_t b = a;
                        while (b < tile_crest && entries[b].cell == entries[a].cell)
                            b++;
                        double mx = 0, my = 0, mz = 0;
                        const CrestPt *ref = &tpts[entries[a].idx];
                        for (int64_t i = a; i < b; i++) {
                            const CrestPt *p = &tpts[entries[i].idx];
                            double sgn = (p->nx * ref->nx + p->ny * ref->ny + p->nz * ref->nz) < 0
                                ? -1.0
                                : 1.0;
                            mx += sgn * p->nx;
                            my += sgn * p->ny;
                            mz += sgn * p->nz;
                        }
                        double ml = sqrt(mx * mx + my * my + mz * mz) + 1e-9;
                        mx /= ml;
                        my /= ml;
                        mz /= ml;
                        for (int64_t i = a; i < b; i++) {
                            const CrestPt *p = &tpts[entries[i].idx];
                            entries[i].depth = (float)(p->x * mx + p->y * my + p->z * mz);
                        }
                        qsort(entries + a, (size_t)(b - a), sizeof(*entries), bin_depth_cmp);
                        for (int64_t c0 = a; c0 < b;) {
                            int64_t c1 = c0 + 1;
                            while (c1 < b &&
                                entries[c1].depth - entries[c1 - 1].depth <= (float)min_layer_gap)
                                c1++;
                            if (c1 - c0 >= min_cell_pts) {
                                double sx = 0, sy = 0, sz = 0, snx = 0, sny = 0, snz = 0, ssc = 0,
                                       scw = 0, ssw = 0, ssp = 0;
                                double wtot = 0;
                                for (int64_t k = c0; k < c1; k++) {
                                    const CrestPt *p = &tpts[entries[k].idx];
                                    double w = p->score;
                                    double sgn =
                                        (p->nx * mx + p->ny * my + p->nz * mz) < 0 ? -1.0 : 1.0;
                                    sx += w * p->x;
                                    sy += w * p->y;
                                    sz += w * p->z;
                                    snx += w * sgn * p->nx;
                                    sny += w * sgn * p->ny;
                                    snz += w * sgn * p->nz;
                                    ssc += p->score;
                                    scw += w * p->cos_w;
                                    ssw += w * p->sin_w;
                                    ssp += w * p->spacing;
                                    wtot += w;
                                }
                                double nl = sqrt(snx * snx + sny * sny + snz * snz) + 1e-9;
                                double gx = box[4] + sx / wtot, gy = box[2] + sy / wtot,
                                       gz = box[0] + sz / wtot;
                                double ax_x, ax_y;
                                hx_centre_at(axis, axis_count, gz, &ax_x, &ax_y);
                                double nx0 = snx / nl, ny0 = sny / nl, nz0 = snz / nl;
                                if (nx0 * (gx - ax_x) + ny0 * (gy - ax_y) < 0.0) {
                                    nx0 = -nx0;
                                    ny0 = -ny0;
                                    nz0 = -nz0;
                                }
                                if (nsurf == surf_cap) {
                                    surf_cap *= 2;
                                    surfels = realloc(surfels, (size_t)surf_cap * sizeof(*surfels));
                                }
                                surfels[nsurf++] = (Surfel){.x = (float)gx,
                                    .y = (float)gy,
                                    .z = (float)gz,
                                    .nx = (float)nx0,
                                    .ny = (float)ny0,
                                    .nz = (float)nz0,
                                    .score = (float)(ssc / (c1 - c0)),
                                    .cos_w = (float)(scw / wtot),
                                    .sin_w = (float)(ssw / wtot),
                                    .spacing = (float)(ssp / wtot),
                                    .theta = (float)atan2(gy - ax_y, gx - ax_x),
                                    .cell = (int32_t)entries[a].cell,
                                    .count = (int32_t)(c1 - c0),
                                    .patch = -1,
                                    .winding = 0};
                            }
                            c0 = c1;
                        }
                        a = b;
                    }
                    free(entries);
                    free(tpts);
                } else {
                    for (int z = 0; z < core_z; z++)
                        free(per_z[z].pts);
                }
                free(per_z);
                free(s_crest_alloc);
                free(s_sm);
                free(tnx);
                free(tny);
                free(tnz);
            }
        }
    }
#ifdef TRACE_CUDA
    TraceCudaStats cuda_stats = {0};
    if (cuda_centres) {
        trace_cuda_crest_end(&cuda_stats);
        free(cuda_centres);
    }
    if (cuda_stats.tiles)
        printf("trace: CUDA crest on %lld of %d tiles (%.2f GB in %lld slabs %.2fs, "
               "kernels %.2fs, binning %.2fs)\n",
            (long long)cuda_stats.tiles, tiles_run, (double)cuda_stats.bytes / 1e9,
            (long long)cuda_stats.slabs, cuda_stats.upload, cuda_stats.kernels, cuda_stats.binning);
#endif
    printf("trace: %d tiles (%d skipped) -> %lld sub-voxel crest points -> %lld clean "
           "stride-%d surfels (%.2fs)\n",
        tiles_run, tiles_skipped, (long long)total_crest, (long long)nsurf, stride,
        clock_sec() - t0);
    fflush(stdout);
    stage_done("crest");

    int32_t *cell_first = calloc((size_t)ncells + 1, sizeof(int32_t));
    for (int64_t i = 0; i < nsurf; i++)
        cell_first[surfels[i].cell + 1]++;
    for (int64_t c = 0; c < ncells; c++)
        cell_first[c + 1] += cell_first[c];
    Surfel *sorted_surfels = malloc((size_t)(nsurf ? nsurf : 1) * sizeof(*sorted_surfels));
    int32_t *cell_cur = malloc((size_t)(ncells + 1) * sizeof(int32_t));
    memcpy(cell_cur, cell_first, (size_t)(ncells + 1) * sizeof(int32_t));
    for (int64_t i = 0; i < nsurf; i++)
        sorted_surfels[cell_cur[surfels[i].cell]++] = surfels[i];
    free(cell_cur);
    free(surfels);
    surfels = sorted_surfels;

    int64_t edge_cap = nsurf * 8 + 65536;
    SEdge *edges = malloc((size_t)edge_cap * sizeof(*edges));
    int64_t nedges = 0;
    float max_dt = 1.85f * stride;
    float max_dn = fminf(2.4f, 0.35f * stride);
    DirectEdgeJob dej = {.surfels = surfels,
        .cell_first = cell_first,
        .su8 = su8,
        .bu8 = bu8,
        .au8 = au8,
        .pi8 = pi8,
        .vox = vox,
        .X = X,
        .Y = Y,
        .Z = Z,
        .GX = GX,
        .GY = GY,
        .GZ = GZ,
        .stride = stride,
        .max_dt = max_dt,
        .max_dn = max_dn,
        .edges_ptr = &edges,
        .nedges_ptr = &nedges,
        .ecap_ptr = &edge_cap};
    memcpy(dej.box, box, sizeof(box));
    pthread_mutex_init(&dej.mu, NULL);
    each_block((size_t)GZ, &dej, direct_edge_worker);
    pthread_mutex_destroy(&dej.mu);
    qsort(edges, (size_t)nedges, sizeof(*edges), sedge_cmp);

    int64_t *adj_off = calloc((size_t)nsurf + 1, sizeof(int64_t));
    for (int64_t e = 0; e < nedges; e++) {
        adj_off[edges[e].a + 1]++;
        adj_off[edges[e].b + 1]++;
    }
    for (int64_t i = 0; i < nsurf; i++)
        adj_off[i + 1] += adj_off[i];
    int32_t *adj_to = malloc((size_t)(2 * nedges + 1) * sizeof(int32_t));
    int32_t *adj_eid = malloc((size_t)(2 * nedges + 1) * sizeof(int32_t));
    int64_t *cur = malloc((size_t)(nsurf + 1) * sizeof(int64_t));
    memcpy(cur, adj_off, (size_t)(nsurf + 1) * sizeof(int64_t));
    for (int64_t e = 0; e < nedges; e++) {
        int32_t a = edges[e].a, b = edges[e].b;
        adj_to[cur[a]] = b;
        adj_eid[cur[a]++] = (int32_t)e;
        adj_to[cur[b]] = a;
        adj_eid[cur[b]++] = (int32_t)e;
    }
    free(cur);

    int64_t y_cuts = 0;
    for (int64_t u = 0; u < nsurf; u++) {
        const Surfel *su = &surfels[u];
        int64_t f0 = adj_off[u], f1 = adj_off[u + 1];
        for (int64_t i = f0; i < f1; i++) {
            if (edges[adj_eid[i]].cut)
                continue;
            const Surfel *s1 = &surfels[adj_to[i]];
            for (int64_t j = i + 1; j < f1; j++) {
                if (edges[adj_eid[j]].cut)
                    continue;
                const Surfel *s2 = &surfels[adj_to[j]];
                float dx = s2->x - s1->x, dy = s2->y - s1->y, dz = s2->z - s1->z;
                float dn = fabsf(dx * su->nx + dy * su->ny + dz * su->nz);
                if (dn <= 2.5f)
                    continue;
                float dt = sqrtf(fmaxf(0.0f, dx * dx + dy * dy + dz * dz - dn * dn));
                if (dt < 1.15f * stride) {
                    int32_t drop =
                        edges[adj_eid[i]].w < edges[adj_eid[j]].w ? adj_eid[i] : adj_eid[j];
                    if (!edges[drop].cut) {
                        edges[drop].cut = 1;
                        y_cuts++;
                    }
                }
            }
        }
    }
    printf("trace: %lld direct edges (%lld Y-junction cuts) (%.2fs)\n", (long long)nedges,
        (long long)y_cuts, clock_sec() - t0);
    fflush(stdout);
    stage_done("direct_edges");

    int32_t *queue = malloc((size_t)(nsurf ? nsurf : 1) * sizeof(int32_t));
    int32_t *cell_stamp = malloc((size_t)ncells * sizeof(int32_t));
    memset(cell_stamp, -1, (size_t)ncells * sizeof(int32_t));
    int32_t npatches = 0;
    float max_patch_r = (float)(max_patch_mul * stride);
    for (int64_t seed = 0; seed < nsurf; seed++) {
        if (surfels[seed].patch >= 0)
            continue;
        int32_t pid = npatches++;
        surfels[seed].patch = pid;
        cell_stamp[surfels[seed].cell] = pid;
        int64_t head = 0, tail = 0;
        queue[tail++] = (int32_t)seed;
        float sx0 = surfels[seed].x, sy0 = surfels[seed].y, sz0 = surfels[seed].z;
        float snx0 = surfels[seed].nx, sny0 = surfels[seed].ny, snz0 = surfels[seed].nz;
        while (head < tail) {
            int32_t u = queue[head++];
            for (int64_t e = adj_off[u]; e < adj_off[u + 1]; e++) {
                int32_t eid = adj_eid[e];
                if (edges[eid].cut || edges[eid].w < 0.20f)
                    continue;
                int32_t v = adj_to[e];
                if (surfels[v].patch >= 0)
                    continue;
                if (cell_stamp[surfels[v].cell] == pid)
                    continue;
                float dx = surfels[v].x - sx0, dy = surfels[v].y - sy0, dz = surfels[v].z - sz0;
                if (dx * dx + dy * dy + dz * dz > max_patch_r * max_patch_r)
                    continue;
                float dn_seed = fabsf(dx * snx0 + dy * sny0 + dz * snz0);
                if (dn_seed > 2.8f)
                    continue;
                float ca_seed = surfels[v].nx * snx0 + surfels[v].ny * sny0 + surfels[v].nz * snz0;
                if (ca_seed < 0.90f)
                    continue;
                surfels[v].patch = pid;
                cell_stamp[surfels[v].cell] = pid;
                queue[tail++] = v;
            }
        }
    }
    free(cell_stamp);

    float *patch_theta = calloc((size_t)(npatches ? npatches : 1), sizeof(float));
    int32_t *patch_size = calloc((size_t)(npatches ? npatches : 1), sizeof(int32_t));
    float *patch_best = calloc((size_t)(npatches ? npatches : 1), sizeof(float));
    for (int64_t i = 0; i < nsurf; i++) {
        int32_t p = surfels[i].patch;
        patch_size[p]++;
        if (surfels[i].score > patch_best[p]) {
            patch_best[p] = surfels[i].score;
            patch_theta[p] = surfels[i].theta;
        }
    }
    printf("trace: %d atomic single-sheet patches (%.2fs)\n", npatches, clock_sec() - t0);
    fflush(stdout);
    stage_done("patches");

    int64_t bcap = nsurf / 2 + 4096, bcnt = 0;
    SEdge *bridges = malloc((size_t)bcap * sizeof(*bridges));
    if (bridge_reach > 1.0) {
        for (int64_t u = 0; u < nsurf; u++) {
            const Surfel *su = &surfels[u];
            if (su->score < 0.20f)
                continue;
            double e1x = -su->ny, e1y = su->nx, e1z = 0.0;
            double l1 = hypot(e1x, e1y);
            if (l1 < 1e-3) {
                e1x = 1.0;
                e1y = 0.0;
            } else {
                e1x /= l1;
                e1y /= l1;
            }
            double dn1 = e1x * su->nx + e1y * su->ny;
            e1x -= dn1 * su->nx;
            e1y -= dn1 * su->ny;
            e1z -= dn1 * su->nz;
            l1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z) + 1e-9;
            e1x /= l1;
            e1y /= l1;
            e1z /= l1;
            double e2x = su->ny * e1z - su->nz * e1y;
            double e2y = su->nz * e1x - su->nx * e1z;
            double e2z = su->nx * e1y - su->ny * e1x;
            int has_dir[4] = {0, 0, 0, 0};
            for (int64_t e = adj_off[u]; e < adj_off[u + 1]; e++) {
                if (edges[adj_eid[e]].cut)
                    continue;
                const Surfel *sv = &surfels[adj_to[e]];
                double dx = sv->x - su->x, dy = sv->y - su->y, dz = sv->z - su->z;
                double d1 = dx * e1x + dy * e1y + dz * e1z;
                double d2 = dx * e2x + dy * e2y + dz * e2z;
                if (d1 > 0.4 * stride)
                    has_dir[0] = 1;
                if (d1 < -0.4 * stride)
                    has_dir[1] = 1;
                if (d2 > 0.4 * stride)
                    has_dir[2] = 1;
                if (d2 < -0.4 * stride)
                    has_dir[3] = 1;
            }
            double dirs[4][3] = {
                {e1x, e1y, e1z}, {-e1x, -e1y, -e1z}, {e2x, e2y, e2z}, {-e2x, -e2y, -e2z}};
            for (int d = 0; d < 4; d++) {
                if (has_dir[d])
                    continue;
                double px = su->x - box[4], py = su->y - box[2], pz = su->z - box[0];
                double tx = dirs[d][0], ty = dirs[d][1], tz = dirs[d][2];
                int hit_v = -1;
                float hit_d = 1e9f;
                for (double dist = 2.0; dist <= bridge_reach && hit_v < 0; dist += 2.0) {
                    px += 2.0 * tx;
                    py += 2.0 * ty;
                    pz += 2.0 * tz;
                    if (px < 2 || px >= X - 2 || py < 2 || py >= Y - 2 || pz < 2 || pz >= Z - 2)
                        break;
                    if (ni8) {
                        size_t vi = ((size_t)(int)pz * Y + (int)py) * X + (int)px;
                        double nxi = ni8[vi] * (1.0 / 127.0);
                        double nyi = ni8[vox + vi] * (1.0 / 127.0);
                        double nzi = ni8[2 * vox + vi] * (1.0 / 127.0);
                        double nl = sqrt(nxi * nxi + nyi * nyi + nzi * nzi);
                        if (nl > 0.25) {
                            nxi /= nl;
                            nyi /= nl;
                            nzi /= nl;
                            double tn = tx * nxi + ty * nyi + tz * nzi;
                            tx -= tn * nxi;
                            ty -= tn * nyi;
                            tz -= tn * nzi;
                            double tl = sqrt(tx * tx + ty * ty + tz * tz);
                            if (tl < 0.3)
                                break;
                            tx /= tl;
                            ty /= tl;
                            tz /= tl;
                        }
                    }
                    if (dist < 0.9 * stride)
                        continue;
                    int cz = clampi((int)(pz / stride), 0, GZ - 1);
                    int cy = clampi((int)(py / stride), 0, GY - 1);
                    int cx = clampi((int)(px / stride), 0, GX - 1);
                    int32_t cell = (int32_t)(((int64_t)cz * GY + cy) * GX + cx);
                    for (int32_t v = cell_first[cell]; v < cell_first[cell + 1]; v++) {
                        if (surfels[v].patch == su->patch)
                            continue;
                        const Surfel *sv = &surfels[v];
                        float ca = su->nx * sv->nx + su->ny * sv->ny + su->nz * sv->nz;
                        if (ca < 0.88f)
                            continue;
                        double dx = (sv->x - box[4]) - px, dy = (sv->y - box[2]) - py,
                               dz = (sv->z - box[0]) - pz;
                        double dn = fabs(dx * sv->nx + dy * sv->ny + dz * sv->nz);
                        double dd = sqrt(dx * dx + dy * dy + dz * dz);
                        if (dn < 2.0 && dd < 0.85 * stride && (float)dd < hit_d) {
                            hit_d = (float)dd;
                            hit_v = v;
                        }
                    }
                }
                if (hit_v >= 0) {
                    const Surfel *sv = &surfels[hit_v];
                    if (pi8 && (su->cos_w * sv->cos_w + su->sin_w * sv->sin_w) < 0.25f)
                        continue;
                    if (bcnt == bcap) {
                        bcap *= 2;
                        bridges = realloc(bridges, (size_t)bcap * sizeof(*bridges));
                    }
                    int32_t a = (int32_t)u < hit_v ? (int32_t)u : hit_v;
                    int32_t b = (int32_t)u < hit_v ? hit_v : (int32_t)u;
                    bridges[bcnt++] = (SEdge){a, b, 0.35f * fminf(su->score, sv->score), 0, 1};
                }
            }
        }
    }
    printf("trace: %lld tangential gap-bridge candidates (%.2fs)\n", (long long)bcnt,
        clock_sec() - t0);
    fflush(stdout);
    stage_done("gap_bridges");

    int64_t rcap = nsurf + 65536, rcnt = 0;
    RayObs *rays = malloc((size_t)rcap * sizeof(*rays));
    float *ray_dists = malloc((size_t)rcap * sizeof(float));
    int64_t rdist_cnt = 0;
    int64_t mcap = nsurf * 4 + 65536, mcnt = 0;
    PairI32 *mpairs = malloc((size_t)mcap * sizeof(*mpairs));
    for (int64_t u = 0; u < nsurf; u++) {
        const Surfel *su = &surfels[u];
        double px = su->x - box[4], py = su->y - box[2], pz = su->z - box[0];
        double rx = su->nx, ry = su->ny, rz = su->nz;
        int hit_v = -1;
        float hit_t = 0.0f;
        double hit_phase = 0.0;
        double phase_acc = 0.0;
        double prev_phi = pi8 ? atan2(su->sin_w, su->cos_w) : 0.0;
        int32_t last_mp = -1;
        for (double t = 2.5; t <= 44.0; t += 1.0) {
            double qx = px + t * rx, qy = py + t * ry, qz = pz + t * rz;
            if (qx < 1 || qx >= X - 1 || qy < 1 || qy >= Y - 1 || qz < 1 || qz >= Z - 1)
                break;
            if (pi8 && hit_v < 0) {
                size_t vi = ((size_t)(int)qz * Y + (int)qy) * X + (int)qx;
                double cur_phi = atan2((double)pi8[vox + vi], (double)pi8[vi]);
                double dphi = cur_phi - prev_phi;
                if (dphi > M_PI)
                    dphi -= 2.0 * M_PI;
                else if (dphi < -M_PI)
                    dphi += 2.0 * M_PI;
                phase_acc += dphi;
                prev_phi = cur_phi;
            }
            int cz = clampi((int)(qz / stride), 0, GZ - 1);
            int cy = clampi((int)(qy / stride), 0, GY - 1);
            int cx = clampi((int)(qx / stride), 0, GX - 1);
            int32_t cell = (int32_t)(((int64_t)cz * GY + cy) * GX + cx);
            for (int32_t v = cell_first[cell]; v < cell_first[cell + 1]; v++) {
                if (surfels[v].patch == su->patch)
                    continue;
                const Surfel *sv = &surfels[v];
                if (su->nx * sv->nx + su->ny * sv->ny + su->nz * sv->nz < 0.72f)
                    continue;
                double dx = sv->x - su->x, dy = sv->y - su->y, dz = sv->z - su->z;
                double proj = dx * rx + dy * ry + dz * rz;
                if (proj < 2.2 || proj > 44.0)
                    continue;
                double lat2 = dx * dx + dy * dy + dz * dz - proj * proj;
                if (lat2 <= (0.75 * stride) * (0.75 * stride)) {
                    if (hit_v < 0 && proj <= 44.0) {
                        hit_v = v;
                        hit_t = (float)proj;
                        hit_phase = phase_acc;
                    }
                    if (proj <= 28.0 && sv->patch != last_mp) {
                        last_mp = sv->patch;
                        if (mcnt + 2 >= mcap) {
                            mcap *= 2;
                            mpairs = realloc(mpairs, (size_t)mcap * sizeof(*mpairs));
                        }
                        mpairs[mcnt++] = (PairI32){su->patch, sv->patch};
                        mpairs[mcnt++] = (PairI32){sv->patch, su->patch};
                    }
                    break;
                }
            }
        }
        if (hit_v >= 0) {
            if (rcnt == rcap) {
                rcap *= 2;
                rays = realloc(rays, (size_t)rcap * sizeof(*rays));
                ray_dists = realloc(ray_dists, (size_t)rcap * sizeof(float));
            }
            rays[rcnt++] = (RayObs){
                su->patch, surfels[hit_v].patch, hit_t, (float)(hit_phase / (2.0 * M_PI)), 1};
            ray_dists[rdist_cnt++] = hit_t;
        }
    }
    float med_gap = 12.0f;
    if (rdist_cnt > 10) {
        qsort(ray_dists, (size_t)rdist_cnt, sizeof(float), float_cmp);
        med_gap = ray_dists[rdist_cnt / 2];
    }
    free(ray_dists);
    printf("trace: %lld normal rays, median layer gap %.2f vox (%.1f um) (%.2fs)\n",
        (long long)rcnt, med_gap, med_gap * voxel_um, clock_sec() - t0);
    fflush(stdout);
    stage_done("normal_rays");

    int64_t tcap = nedges + bcnt + rcnt + 65536, tcnt = 0;
    PTerm *terms = malloc((size_t)tcap * sizeof(*terms));
    for (int64_t e = 0; e < nedges; e++) {
        if (edges[e].cut)
            continue;
        int32_t pa = surfels[edges[e].a].patch, pb = surfels[edges[e].b].patch;
        if (pa == pb)
            continue;
        double ta = surfels[edges[e].a].theta, tb = surfels[edges[e].b].theta;
        int32_t off = curl *
            (seam_turns(ta, tb) + seam_turns(patch_theta[pa], ta) -
                seam_turns(patch_theta[pb], tb));
        double w = 5.0 * edges[e].w;
        int32_t p0 = pa, q0 = pb;
        if (p0 > q0) {
            int32_t sw = p0;
            p0 = q0;
            q0 = sw;
            off = -off;
        }
        terms[tcnt++] = (PTerm){p0, q0, off, 1, w, w};
    }
    for (int64_t e = 0; e < bcnt; e++) {
        int32_t pa = surfels[bridges[e].a].patch, pb = surfels[bridges[e].b].patch;
        if (pa == pb)
            continue;
        double ta = surfels[bridges[e].a].theta, tb = surfels[bridges[e].b].theta;
        int32_t off = curl *
            (seam_turns(ta, tb) + seam_turns(patch_theta[pa], ta) -
                seam_turns(patch_theta[pb], tb));
        double w = 1.2 * bridges[e].w;
        int32_t p0 = pa, q0 = pb;
        if (p0 > q0) {
            int32_t sw = p0;
            p0 = q0;
            q0 = sw;
            off = -off;
        }
        terms[tcnt++] = (PTerm){p0, q0, off, 2, w, w};
    }

    for (int64_t c = 0; c < ncells; c++) {
        int32_t f0 = cell_first[c], f1 = cell_first[c + 1];
        if (f1 - f0 < 2)
            continue;
        for (int32_t i = f0; i < f1; i++)
            for (int32_t j = i + 1; j < f1; j++) {
                int32_t pa = surfels[i].patch, pb = surfels[j].patch;
                if (pa == pb)
                    continue;
                if (mcnt + 4 >= mcap) {
                    mcap *= 2;
                    mpairs = realloc(mpairs, (size_t)mcap * sizeof(*mpairs));
                }
                mpairs[mcnt++] = (PairI32){pa, pb};
                mpairs[mcnt++] = (PairI32){pa, pb};
                mpairs[mcnt++] = (PairI32){pb, pa};
                mpairs[mcnt++] = (PairI32){pb, pa};
            }
    }

    qsort(rays, (size_t)rcnt, sizeof(*rays), ray_obs_cmp);
    int64_t ray_pairs = 0;
    for (int64_t a = 0; a < rcnt;) {
        int64_t b = a;
        double dsum = 0, psum = 0;
        while (b < rcnt && rays[b].lower == rays[a].lower && rays[b].upper == rays[a].upper) {
            dsum += rays[b].dist;
            psum += rays[b].dw_phase;
            b++;
        }
        int64_t cnt = b - a;
        if (cnt >= 2) {
            int32_t pa = rays[a].lower, pb = rays[a].upper;
            double dmean = dsum / cnt;
            double pmean = psum / cnt;
            int32_t k_step = 1;
            if (pi8 && pmean > 1.55)
                k_step = (int32_t)lrint(fmin(4.0, pmean));
            int32_t off = curl * seam_turns(patch_theta[pa], patch_theta[pb]) + k_step;
            double w_down = 4.5 * (double)cnt;
            double skip_ratio = dmean > 1.65 * med_gap ? 0.12 : g_skip_cost;
            double w_up = skip_ratio * w_down;
            int32_t p0 = pa, q0 = pb;
            if (p0 > q0) {
                int32_t sw = p0;
                p0 = q0;
                q0 = sw;
                off = -off;
                double sw_w = w_up;
                w_up = w_down;
                w_down = sw_w;
            }
            terms[tcnt++] = (PTerm){p0, q0, off, 3, w_up, w_down};
            ray_pairs++;
        }
        a = b;
    }
    free(rays);

    AnchorStats anchor = {0, 0, 0, 0, 0};
    if (anchor_dir) {
        int failed = 0;
        snprintf(pathbuf, sizeof(pathbuf), "%s/surfels.bin", anchor_dir);
        struct dirent **entries = NULL;
        int count =
            access(pathbuf, R_OK) == 0 ? -1 : scandir(anchor_dir, &entries, NULL, name_order);
        if (count < 0) {
            failed = anchor_terms(anchor_dir, surfels, cell_first, patch_theta, box, stride, GZ, GY,
                GX, curl, npatches, anchor_weight, &terms, &tcnt, &tcap, &anchor);
        } else {
            for (int k = 0; k < count && !failed; k++) {
                const char *entry = entries[k]->d_name;
                long long uz0, uz1;
                if (sscanf(entry, "unit_z%lld_%lld", &uz0, &uz1) != 2 || uz1 <= box[0] ||
                    uz0 >= box[1] || (uz0 == box[0] && uz1 == box[1]))
                    continue;
                snprintf(pathbuf, sizeof(pathbuf), "%s/%s/surfels.bin", anchor_dir, entry);
                if (access(pathbuf, R_OK))
                    continue;
                snprintf(pathbuf, sizeof(pathbuf), "%s/%s", anchor_dir, entry);
                failed = anchor_terms(pathbuf, surfels, cell_first, patch_theta, box, stride, GZ,
                    GY, GX, curl, npatches, anchor_weight, &terms, &tcnt, &tcap, &anchor);
            }
            for (int k = 0; k < count; k++)
                free(entries[k]);
            free((void *)entries);
        }
        if (failed) {
            fprintf(stderr, "trace: cannot read anchor graph in %s\n", anchor_dir);
            return 1;
        }
        printf("trace: anchor %s: %d graphs, %lld surfels matched, %d winding components "
               "(%.2fs)\n",
            anchor_dir, anchor.graphs, (long long)anchor.matched, anchor.grounds, clock_sec() - t0);
        fflush(stdout);
    }
#ifdef HERCULEX_CUDA_FLOW
    HxDeviceFlow df = {counted_flow_open, counted_flow_run, hx_flow_close};
    if (cuda_flow_wanted && anchor.grounds == 0)
        hx_level_device(&df);
    else if (cuda_flow_wanted)
        printf("trace: anchored solve: min-cut on the CPU\n");
#endif
    int32_t nsolve = npatches + anchor.grounds;

    qsort(mpairs, (size_t)mcnt, sizeof(*mpairs), pair_i32_cmp);
    int64_t umcnt = 0;
    for (int64_t a = 0; a < mcnt;) {
        int64_t b = a;
        while (b < mcnt && mpairs[b].a == mpairs[a].a && mpairs[b].b == mpairs[a].b)
            b++;
        if (b - a >= 3)
            mpairs[umcnt++] = mpairs[a];
        a = b;
    }
    int64_t *mutex_off = calloc((size_t)npatches + 1, sizeof(int64_t));
    for (int64_t i = 0; i < umcnt; i++)
        mutex_off[mpairs[i].a + 1]++;
    for (int32_t i = 0; i < npatches; i++)
        mutex_off[i + 1] += mutex_off[i];
    int32_t *mutex_to = malloc((size_t)(umcnt ? umcnt : 1) * sizeof(int32_t));
    for (int64_t i = 0; i < umcnt; i++)
        mutex_to[i] = mpairs[i].b;
    free(mpairs);

    qsort(terms, (size_t)tcnt, sizeof(*terms), pterm_cmp);
    int64_t mterms = 0;
    for (int64_t i = 0; i < tcnt; i++) {
        if (mterms && pterm_cmp(&terms[mterms - 1], &terms[i]) == 0) {
            terms[mterms - 1].up += terms[i].up;
            terms[mterms - 1].down += terms[i].down;
        } else {
            terms[mterms++] = terms[i];
        }
    }

    int32_t *tp = malloc((size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *tq = malloc((size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *tk = malloc((size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *tup = malloc((size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *tdn = malloc((size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *pw = calloc((size_t)(nsolve ? nsolve : 1), sizeof(int32_t));
    for (int64_t i = 0; i < mterms; i++) {
        tp[i] = terms[i].p;
        tq[i] = terms[i].q;
        tk[i] = terms[i].k;
        tup[i] = (int32_t)fmin(1000000.0, fmax(1.0, lrint(8.0 * terms[i].up)));
        tdn[i] = (int32_t)fmin(1000000.0, fmax(1.0, lrint(8.0 * terms[i].down)));
    }
    int64_t slog[3] = {0};
    stage_done("terms");
    if (npatches > 1 && mterms > 0)
        hx_level_minimize(nsolve, mterms, tp, tq, tk, tup, tdn, pw, 32, slog);
    stage_done("exact_solve");
    if (anchor.grounds > 0) {
        int32_t shift = pw[npatches + anchor.main_ground];
        for (int32_t i = 0; i < nsolve; i++)
            pw[i] -= shift;
    }

    int64_t ord_tot = 0, ord_met = 0, same_tot = 0, same_met = 0, anchor_tot = 0, anchor_met = 0;
    for (int64_t i = 0; i < mterms; i++) {
        int met = (pw[tq[i]] - pw[tp[i]] == tk[i]);
        if (terms[i].kind == 3) {
            ord_tot++;
            ord_met += (pw[tq[i]] - pw[tp[i]] >= tk[i]);
        } else if (terms[i].kind == 4) {
            anchor_tot++;
            anchor_met += met;
        } else {
            same_tot++;
            same_met += met;
        }
    }
    printf("trace: exact solve %d patches, %lld terms (rays=%lld), order_ge=%lld/%lld, "
           "same=%lld/%lld, energy=%lld (%.2fs)\n",
        npatches, (long long)mterms, (long long)ray_pairs, (long long)ord_met, (long long)ord_tot,
        (long long)same_met, (long long)same_tot, (long long)slog[2], clock_sec() - t0);
    if (anchor_tot > 0)
        printf("trace: anchor terms met %lld/%lld\n", (long long)anchor_met, (long long)anchor_tot);
    fflush(stdout);

    int draws = 8;
    if (draw_jobs > draws)
        draw_jobs = draws;
    int32_t *votes = calloc((size_t)(mterms ? mterms : 1), sizeof(int32_t));
    int32_t *dup = malloc((size_t)draw_jobs * (size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *ddn = malloc((size_t)draw_jobs * (size_t)(mterms ? mterms : 1) * sizeof(int32_t));
    int32_t *dw = malloc((size_t)draw_jobs * (size_t)(nsolve ? nsolve : 1) * sizeof(int32_t));
    DrawJob draw = {.nsolve = nsolve,
        .mterms = mterms,
        .tp = tp,
        .tq = tq,
        .tk = tk,
        .up = dup,
        .down = ddn,
        .windings = dw};
    uint64_t rng = UINT64_C(0x5eed1545cafe0001);
    for (int d = 0; d < draws && npatches > 1 && mterms > 0; d += draw_jobs) {
        int batch = draws - d < draw_jobs ? draws - d : draw_jobs;
        for (int b = 0; b < batch; b++) {
            int32_t *bup = dup + (size_t)b * mterms, *bdn = ddn + (size_t)b * mterms;
            memcpy(dw + (size_t)b * nsolve, pw, (size_t)nsolve * sizeof(int32_t));
            for (int64_t i = 0; i < mterms; i++) {
                double mul = 0.25 + 0.75 * rng_exp(&rng);
                bup[i] = (int32_t)fmin(1000000.0, fmax(1.0, lrint(tup[i] * mul)));
                bdn[i] = (int32_t)fmin(1000000.0, fmax(1.0, lrint(tdn[i] * mul)));
            }
        }
        if (batch == 1)
            draw_worker(&draw, 0);
        else
            each_block((size_t)batch, &draw, draw_worker);
        for (int b = 0; b < batch; b++) {
            const int32_t *bw = dw + (size_t)b * nsolve;
            for (int64_t i = 0; i < mterms; i++)
                votes[i] += (bw[tq[i]] - bw[tp[i]] == tk[i]);
        }
    }
    free(dup);
    free(ddn);
    free(dw);
    stage_done("draws");

    MergeCand *mcands = malloc((size_t)(mterms ? mterms : 1) * sizeof(*mcands));
    int64_t ncand = 0;
    for (int64_t i = 0; i < mterms; i++) {
        if (terms[i].kind >= 3)
            continue;
        if (pw[tq[i]] - pw[tp[i]] != tk[i])
            continue;
        int need_votes = min_votes;
        if (votes[i] < need_votes)
            continue;
        mcands[ncand++] = (MergeCand){tp[i], tq[i], terms[i].kind, terms[i].up};
    }
    qsort(mcands, (size_t)ncand, sizeof(*mcands), merge_cand_cmp);

    int32_t *wuf = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    for (int32_t i = 0; i < npatches; i++)
        wuf[i] = i;
    for (int64_t i = 0; i < mterms; i++) {
        if (terms[i].kind == 4)
            continue;
        int32_t ra = uf_find(wuf, tp[i]), rb = uf_find(wuf, tq[i]);
        if (ra != rb)
            wuf[rb] = ra;
    }

    int32_t *uf = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    int32_t *c_head = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    int32_t *c_tail = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    int32_t *c_next = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    int32_t *c_len = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    int32_t *c_surf = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    for (int32_t i = 0; i < npatches; i++) {
        uf[i] = i;
        c_head[i] = i;
        c_tail[i] = i;
        c_next[i] = -1;
        c_len[i] = 1;
        c_surf[i] = patch_size[i];
    }
    int64_t merges_ok = 0, mutex_blocked = 0;
    for (int64_t i = 0; i < ncand; i++) {
        int32_t ra = uf_find(uf, mcands[i].p);
        int32_t rb = uf_find(uf, mcands[i].q);
        if (ra == rb)
            continue;
        if (c_len[ra] < c_len[rb]) {
            int32_t sw = ra;
            ra = rb;
            rb = sw;
        }
        int32_t min_s = c_surf[ra] < c_surf[rb] ? c_surf[ra] : c_surf[rb];
        int32_t max_p = min_s >= 250 ? 4 : (min_s >= 80 ? 2 : 1);
        int32_t max_s = (int32_t)(0.015 * (double)min_s);
        int32_t conf_p = 0, conf_s = 0;
        int conflict = 0;
        for (int32_t p = c_head[rb]; p >= 0; p = c_next[p]) {
            int hit_p = 0;
            for (int64_t m = mutex_off[p]; m < mutex_off[p + 1]; m++) {
                int32_t q = mutex_to[m];
                if (uf_find(uf, q) == ra &&
                    pw[q] - pw[p] == curl * seam_turns(patch_theta[p], patch_theta[q])) {
                    hit_p = 1;
                    break;
                }
            }
            if (hit_p) {
                conf_p++;
                conf_s += patch_size[p];
                if (conf_p >= max_p && conf_s > max_s) {
                    conflict = 1;
                    break;
                }
            }
        }
        if (conflict) {
            mutex_blocked++;
            continue;
        }
        uf[rb] = ra;
        c_next[c_tail[ra]] = c_head[rb];
        c_tail[ra] = c_tail[rb];
        c_len[ra] += c_len[rb];
        c_surf[ra] += c_surf[rb];
        merges_ok++;
    }
    free(mcands);
    free(votes);
    free(tp);
    free(tq);
    free(tk);
    free(tup);
    free(tdn);
    free(terms);

    for (int64_t i = 0; i < nsurf; i++) {
        int32_t p = surfels[i].patch;
        surfels[i].winding = pw[p] + curl * seam_turns(patch_theta[p], surfels[i].theta);
    }
    stage_done("merge");

    int64_t scap = nsurf / 4 + 4096, scnt = 0;
    SEdge *stitches = malloc((size_t)scap * sizeof(*stitches));
    for (int st_stage = 0; st_stage < 2 && npatches > 1; st_stage++) {
        double cur_reach = st_stage == 0 ? stitch_reach : consolidate_reach;
        int32_t cur_min_root = st_stage == 0 ? 24 : 60;
        float cur_min_ca = st_stage == 0 ? 0.89f : 0.86f;
        float cur_max_dn_arc = st_stage == 0 ? 2.0f : 2.8f;
        float cur_max_dn_uv = st_stage == 0 ? 3.0f : 3.8f;
        if (cur_reach <= (st_stage == 0 ? (double)stride : stitch_reach))
            continue;
        int64_t stage_merges = 0, stage_blocked = 0;
        int32_t *patch_root = malloc((size_t)npatches * sizeof(int32_t));
        int32_t *patch_wc = malloc((size_t)npatches * sizeof(int32_t));
        for (int32_t i = 0; i < npatches; i++) {
            patch_root[i] = uf_find(uf, i);
            patch_wc[i] = uf_find(wuf, i);
        }
        int32_t *root_sz = calloc((size_t)npatches, sizeof(int32_t));
        for (int64_t i = 0; i < nsurf; i++)
            root_sz[patch_root[surfels[i].patch]]++;
        int64_t slcap = nsurf / 4 + 16384, slcnt = 0;
        StitchLink *slinks = malloc((size_t)slcap * sizeof(*slinks));
        StitchRayJob sj = {.surfels = surfels,
            .edges = edges,
            .adj_off = adj_off,
            .adj_eid = adj_eid,
            .adj_to = adj_to,
            .cell_first = cell_first,
            .patch_root = patch_root,
            .patch_wc = patch_wc,
            .root_sz = root_sz,
            .bu8 = bu8,
            .pi8 = pi8,
            .vox = vox,
            .nsurf = nsurf,
            .X = X,
            .Y = Y,
            .Z = Z,
            .GX = GX,
            .GY = GY,
            .GZ = GZ,
            .stride = stride,
            .curl = curl,
            .cur_min_root = cur_min_root,
            .cur_reach = cur_reach,
            .cur_min_ca = cur_min_ca,
            .cur_max_dn_arc = cur_max_dn_arc,
            .cur_max_dn_uv = cur_max_dn_uv,
            .slinks_ptr = &slinks,
            .slcnt_ptr = &slcnt,
            .slcap_ptr = &slcap};
        memcpy(sj.box, box, sizeof(box));
        pthread_mutex_init(&sj.mu, NULL);
        size_t nblk = (size_t)((nsurf + 32767) / 32768);
        each_block(nblk, &sj, stitch_ray_worker);
        pthread_mutex_destroy(&sj.mu);
        stage_done(st_stage == 0 ? "stitch_rays_1" : "stitch_rays_2");
        free(patch_root);
        free(patch_wc);
        qsort(slinks, (size_t)slcnt, sizeof(*slinks), stitch_link_cmp);
        int64_t gcap = slcnt / 2 + 1024, gcnt = 0;
        StitchGroup *groups = malloc((size_t)gcap * sizeof(*groups));
        for (int64_t a = 0; a < slcnt;) {
            int64_t b = a;
            double asum = 0.0, dsum = 0.0;
            while (b < slcnt && slinks[b].ca == slinks[a].ca && slinks[b].cb == slinks[a].cb) {
                asum += slinks[b].align;
                dsum += slinks[b].dist;
                b++;
            }
            int64_t cnt = b - a;
            int32_t min_sz = root_sz[slinks[a].ca] < root_sz[slinks[a].cb] ? root_sz[slinks[a].ca]
                                                                           : root_sz[slinks[a].cb];
            int need_cnt = st_stage == 0 ? (min_sz < 80 ? 2 : 3) : 5;
            if (cnt >= need_cnt) {
                if (gcnt == gcap) {
                    gcap *= 2;
                    groups = realloc(groups, (size_t)gcap * sizeof(*groups));
                }
                double sc = (double)cnt * (asum / cnt) / (1.0 + (dsum / cnt) / 32.0);
                groups[gcnt++] = (StitchGroup){slinks[a].ca, slinks[a].cb, a, cnt, sc};
            }
            a = b;
        }
        qsort(groups, (size_t)gcnt, sizeof(*groups), stitch_group_cmp);
        for (int64_t gi = 0; gi < gcnt; gi++) {
            int32_t ra = uf_find(uf, groups[gi].ca);
            int32_t rb = uf_find(uf, groups[gi].cb);
            if (ra == rb)
                continue;
            if (c_len[ra] < c_len[rb]) {
                int32_t sw = ra;
                ra = rb;
                rb = sw;
            }
            int32_t min_s = c_surf[ra] < c_surf[rb] ? c_surf[ra] : c_surf[rb];
            int32_t max_p = min_s >= 250 ? 4 : (min_s >= 80 ? 2 : 1);
            int32_t max_s = (int32_t)(0.015 * (double)min_s);
            int32_t conf_p = 0, conf_s = 0;
            int conflict = 0;
            for (int32_t p = c_head[rb]; p >= 0; p = c_next[p]) {
                int hit_p = 0;
                for (int64_t m = mutex_off[p]; m < mutex_off[p + 1]; m++) {
                    int32_t q = mutex_to[m];
                    if (uf_find(uf, q) == ra &&
                        pw[q] - pw[p] == curl * seam_turns(patch_theta[p], patch_theta[q])) {
                        hit_p = 1;
                        break;
                    }
                }
                if (hit_p) {
                    conf_p++;
                    conf_s += patch_size[p];
                    if (conf_p >= max_p && conf_s > max_s) {
                        conflict = 1;
                        break;
                    }
                }
            }
            if (conflict) {
                stage_blocked++;
                continue;
            }
            uf[rb] = ra;
            c_next[c_tail[ra]] = c_head[rb];
            c_tail[ra] = c_tail[rb];
            c_len[ra] += c_len[rb];
            c_surf[ra] += c_surf[rb];
            stage_merges++;
            int64_t take = groups[gi].count < 32 ? groups[gi].count : 32;
            int64_t step_i = groups[gi].count / take;
            if (step_i < 1)
                step_i = 1;
            for (int64_t j = 0; j < groups[gi].count && j < take * step_i; j += step_i) {
                const StitchLink *sl = &slinks[groups[gi].first + j];
                if (scnt == scap) {
                    scap *= 2;
                    stitches = realloc(stitches, (size_t)scap * sizeof(*stitches));
                }
                stitches[scnt++] = (SEdge){sl->u, sl->v, sl->align, 0, 2};
            }
        }
        free(groups);
        free(slinks);
        free(root_sz);
        printf("trace: stitch stage %d (reach=%.0f): %lld merges (%lld mutex-blocked, %lld "
               "links) (%.2fs)\n",
            st_stage + 1, cur_reach, (long long)stage_merges, (long long)stage_blocked,
            (long long)scnt, clock_sec() - t0);
        fflush(stdout);
        stage_done(st_stage == 0 ? "stitch_merge_1" : "stitch_merge_2");
    }
    int32_t *patch_wcomp = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    for (int32_t i = 0; i < npatches; i++)
        patch_wcomp[i] = uf_find(wuf, i);
    free(wuf);
    free(c_head);
    free(c_tail);
    free(c_next);
    free(c_len);
    free(c_surf);
    free(mutex_off);
    free(mutex_to);

    int32_t *comp_id = malloc((size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    memset(comp_id, -1, (size_t)(npatches ? npatches : 1) * sizeof(int32_t));
    int32_t ncomps = 0;
    for (int32_t i = 0; i < npatches; i++) {
        int32_t r = uf_find(uf, i);
        if (comp_id[r] < 0)
            comp_id[r] = ncomps++;
        comp_id[i] = comp_id[r];
    }
    int32_t *csizes = calloc((size_t)(ncomps ? ncomps : 1), sizeof(int32_t));
    for (int64_t i = 0; i < nsurf; i++)
        csizes[comp_id[surfels[i].patch]]++;
    int32_t kept_pieces = 0;
    int32_t *comp_to_piece = malloc((size_t)(ncomps ? ncomps : 1) * sizeof(int32_t));
    for (int32_t c = 0; c < ncomps; c++)
        comp_to_piece[c] = csizes[c] >= min_nodes ? kept_pieces++ : -1;
    PieceSpec *pspecs = calloc((size_t)(kept_pieces ? kept_pieces : 1), sizeof(*pspecs));
    for (int32_t c = 0; c < ncomps; c++) {
        int32_t k = comp_to_piece[c];
        if (k >= 0) {
            pspecs[k].members = malloc((size_t)csizes[c] * sizeof(int32_t));
            pspecs[k].count = 0;
        }
    }
    for (int64_t i = 0; i < nsurf; i++) {
        int32_t k = comp_to_piece[comp_id[surfels[i].patch]];
        if (k >= 0)
            pspecs[k].members[pspecs[k].count++] = (int32_t)i;
    }
    qsort(pspecs, (size_t)kept_pieces, sizeof(*pspecs), piece_spec_cmp);
    printf("trace: %lld merges (%lld mutex-blocked) -> %d pieces >= %d surfels (%.2fs)\n",
        (long long)merges_ok, (long long)mutex_blocked, kept_pieces, min_nodes, clock_sec() - t0);
    fflush(stdout);

    int32_t *surfel_piece = malloc((size_t)(nsurf ? nsurf : 1) * sizeof(int32_t));
    memset(surfel_piece, -1, (size_t)(nsurf ? nsurf : 1) * sizeof(int32_t));
    for (int32_t k = 0; k < kept_pieces; k++)
        for (int32_t j = 0; j < pspecs[k].count; j++)
            surfel_piece[pspecs[k].members[j]] = k;

    SheetNode *snodes = malloc((size_t)(nsurf ? nsurf : 1) * sizeof(*snodes));
    for (int64_t i = 0; i < nsurf; i++) {
        snodes[i] = (SheetNode){.x = surfels[i].x,
            .y = surfels[i].y,
            .z = surfels[i].z,
            .nx = (int8_t)lrintf(surfels[i].nx * 127.0f),
            .ny = (int8_t)lrintf(surfels[i].ny * 127.0f),
            .nz = (int8_t)lrintf(surfels[i].nz * 127.0f),
            .prominence = (uint8_t)fminf(255.0f, surfels[i].score * 255.0f),
            .width = 8,
            .coherence = 255};
    }

    int64_t *pfirst = calloc((size_t)kept_pieces + 1, sizeof(int64_t));
    for (int32_t k = 0; k < kept_pieces; k++)
        pfirst[k + 1] = pfirst[k] + pspecs[k].count;
    int32_t *porder =
        malloc((size_t)(pfirst[kept_pieces] ? pfirst[kept_pieces] : 1) * sizeof(int32_t));
    for (int32_t k = 0; k < kept_pieces; k++)
        memcpy(porder + pfirst[k], pspecs[k].members, (size_t)pspecs[k].count * sizeof(int32_t));

    int64_t *efirst = calloc((size_t)kept_pieces + 1, sizeof(int64_t));
    for (int64_t e = 0; e < nedges; e++) {
        if (edges[e].cut)
            continue;
        int32_t ka = surfel_piece[edges[e].a], kb = surfel_piece[edges[e].b];
        if (ka >= 0 && ka == kb) {
            int32_t st = curl * seam_turns(surfels[edges[e].a].theta, surfels[edges[e].b].theta);
            if (surfels[edges[e].b].winding - surfels[edges[e].a].winding == st)
                efirst[ka + 1]++;
        }
    }
    for (int64_t e = 0; e < bcnt; e++) {
        int32_t ka = surfel_piece[bridges[e].a], kb = surfel_piece[bridges[e].b];
        if (ka >= 0 && ka == kb) {
            int32_t st =
                curl * seam_turns(surfels[bridges[e].a].theta, surfels[bridges[e].b].theta);
            if (surfels[bridges[e].b].winding - surfels[bridges[e].a].winding == st)
                efirst[ka + 1]++;
        }
    }
    for (int64_t e = 0; e < scnt; e++) {
        int32_t ka = surfel_piece[stitches[e].a], kb = surfel_piece[stitches[e].b];
        if (ka >= 0 && ka == kb) {
            int32_t st =
                curl * seam_turns(surfels[stitches[e].a].theta, surfels[stitches[e].b].theta);
            if (surfels[stitches[e].b].winding - surfels[stitches[e].a].winding == st)
                efirst[ka + 1]++;
        }
    }
    for (int32_t k = 0; k < kept_pieces; k++)
        efirst[k + 1] += efirst[k];
    int64_t tot_pe = efirst[kept_pieces];
    int32_t *pea = malloc((size_t)(tot_pe ? tot_pe : 1) * sizeof(int32_t));
    int32_t *peb = malloc((size_t)(tot_pe ? tot_pe : 1) * sizeof(int32_t));
    int64_t *ecur = malloc((size_t)(kept_pieces + 1) * sizeof(int64_t));
    memcpy(ecur, efirst, (size_t)(kept_pieces + 1) * sizeof(int64_t));
    for (int64_t e = 0; e < nedges; e++) {
        if (edges[e].cut)
            continue;
        int32_t ka = surfel_piece[edges[e].a], kb = surfel_piece[edges[e].b];
        if (ka >= 0 && ka == kb) {
            int32_t st = curl * seam_turns(surfels[edges[e].a].theta, surfels[edges[e].b].theta);
            if (surfels[edges[e].b].winding - surfels[edges[e].a].winding == st) {
                pea[ecur[ka]] = edges[e].a;
                peb[ecur[ka]++] = edges[e].b;
            }
        }
    }
    for (int64_t e = 0; e < bcnt; e++) {
        int32_t ka = surfel_piece[bridges[e].a], kb = surfel_piece[bridges[e].b];
        if (ka >= 0 && ka == kb) {
            int32_t st =
                curl * seam_turns(surfels[bridges[e].a].theta, surfels[bridges[e].b].theta);
            if (surfels[bridges[e].b].winding - surfels[bridges[e].a].winding == st) {
                pea[ecur[ka]] = bridges[e].a;
                peb[ecur[ka]++] = bridges[e].b;
            }
        }
    }
    for (int64_t e = 0; e < scnt; e++) {
        int32_t ka = surfel_piece[stitches[e].a], kb = surfel_piece[stitches[e].b];
        if (ka >= 0 && ka == kb) {
            int32_t st =
                curl * seam_turns(surfels[stitches[e].a].theta, surfels[stitches[e].b].theta);
            if (surfels[stitches[e].b].winding - surfels[stitches[e].a].winding == st) {
                pea[ecur[ka]] = stitches[e].a;
                peb[ecur[ka]++] = stitches[e].b;
            }
        }
    }
    free(ecur);

    int32_t *local_map = malloc((size_t)(nsurf ? nsurf : 1) * sizeof(int32_t));
    int8_t *flat_sign = calloc((size_t)(nsurf ? nsurf : 1), 1);
    float *flat_u = calloc((size_t)(nsurf ? nsurf : 1), sizeof(float));
    float *flat_v = calloc((size_t)(nsurf ? nsurf : 1), sizeof(float));
    if (!skip_emit && kept_pieces > 0) {
        int32_t flat_threads = online_cpus < 1 ? 1 : (int32_t)online_cpus;
        hx_level_flatten(snodes, porder, pfirst, kept_pieces, pea, peb, efirst, local_map,
            flat_sign, flat_u, flat_v, 400, 1e-3, 0.01, flat_threads);
        printf("trace: initial Poisson UV flatten done (%.2fs)\n", clock_sec() - t0);
        fflush(stdout);
    }

    if (save_graph_dir) {
        make_dirs(save_graph_dir);
        snprintf(pathbuf, sizeof(pathbuf), "%s/surfels.bin", save_graph_dir);
        FILE *sf = fopen(pathbuf, "wb");
        if (sf) {
            for (int64_t i = 0; i < nsurf; i++)
                surfels[i].patch = patch_wcomp[surfels[i].patch];
            int64_t hdr[2] = {nsurf, (int64_t)kept_pieces};
            fwrite(hdr, sizeof(int64_t), 2, sf);
            fwrite(surfels, sizeof(Surfel), (size_t)nsurf, sf);
            fwrite(surfel_piece, sizeof(int32_t), (size_t)nsurf, sf);
            fclose(sf);
            printf("trace: saved graph %s (%lld surfels, %d pieces) (%.2fs)\n", pathbuf,
                (long long)nsurf, kept_pieces, clock_sec() - t0);
            fflush(stdout);
        }
    }
    stage_done("pieces_and_graph");
    free(patch_wcomp);

    free(surfel_piece);
    free(comp_to_piece);
    free(csizes);
    free(comp_id);
    free(uf);
    free(pw);
    free(patch_theta);
    free(patch_size);
    free(patch_best);
    free(stitches);
    free(bridges);
    free(queue);
    free(adj_off);
    free(adj_to);
    free(adj_eid);
    free(edges);
    free(cell_first);
    free(pfirst);
    free(porder);
    free(flat_sign);

    make_dirs(out_dir);
    EmitCtx ec = {.out_dir = out_dir,
        .sample = sample,
        .scan = scan,
        .Z = Z,
        .Y = Y,
        .X = X,
        .curl = curl,
        .stride = stride,
        .min_nodes = min_nodes,
        .arap_rounds = arap_rounds,
        .no_lock = no_lock,
        .fill_reach = fill_reach,
        .fair_rounds = fair_rounds,
        .max_hole_cells = max_hole_cells,
        .verbose = verbose,
        .voxel_um = voxel_um,
        .su8 = su8,
        .bu8 = bu8,
        .pi8 = pi8,
        .axis = axis,
        .axis_count = axis_count,
        .surfels = surfels,
        .snodes = snodes,
        .nsurf = nsurf,
        .rec_pieces = json_object_new_array()};
    memcpy(ec.box, box, sizeof(box));

    for (int32_t k = 0; !skip_emit && k < kept_pieces; k++) {
        int32_t n = pspecs[k].count;
        const int32_t *mem = pspecs[k].members;
        for (int32_t i = 0; i < n; i++)
            local_map[mem[i]] = i;
        float *pu = malloc((size_t)n * sizeof(float));
        float *pv = malloc((size_t)n * sizeof(float));
        for (int32_t i = 0; i < n; i++) {
            pu[i] = flat_u[mem[i]];
            pv[i] = flat_v[mem[i]];
        }
        int64_t pe0 = efirst[k], pe1 = efirst[k + 1], pm = pe1 - pe0;
        int32_t *lea = malloc((size_t)(pm ? pm : 1) * sizeof(int32_t));
        int32_t *leb = malloc((size_t)(pm ? pm : 1) * sizeof(int32_t));
        for (int64_t e = 0; e < pm; e++) {
            lea[e] = local_map[pea[pe0 + e]];
            leb[e] = local_map[peb[pe0 + e]];
        }
        emit_piece(&ec, mem, n, lea, leb, pm, pu, pv, 1);
        free(pu);
        free(pv);
        free(lea);
        free(leb);
    }

    json_object *record = json_object_new_object();
    json_object_object_add(record, "sample", json_object_new_string(sample));
    json_object_object_add(record, "volume", json_object_new_string(scan));
    json_object_object_add(record, "voxel_um", json_object_new_double(voxel_um));
    json_object_object_add(record, "stride", json_object_new_int(stride));
    json_object_object_add(record, "chirality", json_object_new_int(curl));
    json_object_object_add(record, "surfels", json_object_new_int64(nsurf));
    json_object_object_add(record, "patches", json_object_new_int(npatches));
    json_object_object_add(record, "mutex_blocked", json_object_new_int64(mutex_blocked));
    json_object_object_add(record, "fold_splits", json_object_new_int(ec.fold_splits));
    json_object_object_add(record, "area_cm2", json_object_new_double(ec.total_cm2));
    json_object_object_add(record, "area_4cm2", json_object_new_double(ec.area_4cm2));
    json_object_object_add(record, "pieces_4cm2", json_object_new_int(ec.pieces_4));
    json_object_object_add(record, "largest_cm2", json_object_new_double(ec.largest_cm2));
    json_object_object_add(record, "isometric_pass", json_object_new_int(ec.iso_pass));
    json_object_object_add(record, "piece_count", json_object_new_int(ec.written_count));
    json_object_object_add(record, "seconds", json_object_new_double(clock_sec() - t0));
    json_object_object_add(record, "pieces", ec.rec_pieces);
    snprintf(pathbuf, sizeof(pathbuf), "%s/record.json", out_dir);
    json_object_to_file_ext(pathbuf, record, JSON_C_TO_STRING_PRETTY);
    json_object_put(record);

    printf("trace DONE: %d pieces (%.2f cm2 total, %d >=4cm2 totaling %.2f cm2, largest "
           "%.2f cm2, "
           "isometric %d/%d, fold_splits=%d) in %.2fs\n",
        ec.written_count, ec.total_cm2, ec.pieces_4, ec.area_4cm2, ec.largest_cm2, ec.iso_pass,
        ec.written_count, ec.fold_splits, clock_sec() - t0);
    stage_done("emit");
    if (timings) {
        for (int i = 0; i < stage_count; i++)
            printf("trace timing: %s %.2f\n", stage_log[i].name, stage_log[i].seconds);
        printf("trace timing: draw_jobs %d\n", draw_jobs);
#ifdef TRACE_CUDA
        printf("trace timing: cuda_crest_tiles %lld\n", (long long)cuda_stats.tiles);
        printf("trace timing: cuda_crest_upload %.2f\n", cuda_stats.upload);
        printf("trace timing: cuda_crest_kernels %.2f\n", cuda_stats.kernels);
        printf("trace timing: cuda_crest_binning %.2f\n", cuda_stats.binning);
#endif
#ifdef HERCULEX_CUDA_FLOW
        printf("trace timing: cuda_flow_graphs %lld\n", (long long)flow_opened);
        printf("trace timing: cuda_flow_refused %lld\n", (long long)flow_refused);
        printf("trace timing: cuda_flow_cuts %lld\n", (long long)flow_runs);
        printf("trace timing: cuda_flow_failed %lld\n", (long long)flow_failed);
        printf("trace timing: cuda_flow_seconds %.2f\n", (double)flow_micros / 1e6);
#endif
    }

    for (int32_t k = 0; k < kept_pieces; k++)
        free(pspecs[k].members);
    free(pspecs);
    free(efirst);
    free(pea);
    free(peb);
    free(local_map);
    free(flat_u);
    free(flat_v);
    free(snodes);
    free(surfels);
    unmap_npy(&surf_npy);
    unmap_npy(&bar_npy);
    unmap_npy(&nrm_npy);
    unmap_npy(&pha_npy);
    unmap_npy(&aff_npy);
    unmap_npy(&spc_npy);
    free(axis);
    return 0;
}
