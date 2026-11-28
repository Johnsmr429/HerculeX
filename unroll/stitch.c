#define _GNU_SOURCE
#include "parallel_blocks.h"
#include "stitch_cuda.h"

#include <errno.h>
#include <float.h>
#include <json-c/json.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

typedef struct {
    union {
        struct {
            float x, y, z;
            float nx, ny, nz;
        };
        StitchSpot spot;
    };
    float score, cos_w, sin_w, spacing, theta;
    int32_t cell, count, patch, winding;
} Surfel;
_Static_assert(sizeof(Surfel) == sizeof(StitchSurfel), "surfel layout");

void hx_centre_at(const HxAxisKnot *knots, int64_t count, double z, double *x, double *y);
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

static double clock_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int64_t tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

enum { T_LOAD, T_SEAMS, T_RAYS, T_SOLVE, T_GATHER, T_EDGES, T_FLATTEN, T_GRIDS, T_WRITE, T_COUNT };
static int64_t g_spent[T_COUNT];

static void lap(int stage, int64_t since) {
    __atomic_fetch_add(&g_spent[stage], tick() - since, __ATOMIC_RELAXED);
}

static FILE *g_check, *g_check_uv;
static pthread_mutex_t g_check_mu = PTHREAD_MUTEX_INITIALIZER;

static uint64_t fnv(uint64_t h, const void *data, size_t size) {
    const uint8_t *p = data;
    for (size_t i = 0; i < size; i++)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}

static uint64_t members_hash(const Surfel *pts, const int32_t *members, int32_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (int32_t i = 0; i < n; i++)
        h = fnv(h, &pts[members ? members[i] : i], 3 * sizeof(float));
    return h;
}

static void check_raw(const char *stage, int k, const void *raw, int64_t count, size_t size) {
    if (!g_check)
        return;
    fprintf(g_check, "%s %d %lld %016llx\n", stage, k, (long long)count,
        (unsigned long long)fnv(UINT64_C(14695981039346656037), raw, (size_t)count * size));
}

static void check_sheet(
    const Surfel *pts, int32_t n, const int32_t *ea, const int32_t *eb, int64_t bridges) {
    if (!g_check)
        return;
    uint64_t h = UINT64_C(14695981039346656037);
    for (int64_t e = 0; e < bridges; e++) {
        h = fnv(h, &ea[e], sizeof(int32_t));
        h = fnv(h, &eb[e], sizeof(int32_t));
    }
    uint64_t m = members_hash(pts, NULL, n);
    pthread_mutex_lock(&g_check_mu);
    fprintf(g_check, "sheet %016llx %d %lld %016llx\n", (unsigned long long)m, n,
        (long long)bridges, (unsigned long long)h);
    pthread_mutex_unlock(&g_check_mu);
}

static void check_flat(
    const Surfel *pts, const int32_t *members, int32_t n, const float *pu, const float *pv) {
    if (!g_check_uv)
        return;
    uint64_t h = members_hash(pts, members, n);
    pthread_mutex_lock(&g_check_mu);
    fwrite(&h, sizeof(h), 1, g_check_uv);
    fwrite(&n, sizeof(n), 1, g_check_uv);
    fwrite(pu, sizeof(float), (size_t)n, g_check_uv);
    fwrite(pv, sizeof(float), (size_t)n, g_check_uv);
    pthread_mutex_unlock(&g_check_mu);
}

static void check_piece(const Surfel *pts, const int32_t *members, int32_t n, int depth, int id) {
    if (!g_check)
        return;
    uint64_t h = members_hash(pts, members, n);
    pthread_mutex_lock(&g_check_mu);
    fprintf(g_check, "piece %016llx %d %d %d\n", (unsigned long long)h, n, depth, id);
    pthread_mutex_unlock(&g_check_mu);
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
    char dir[1024];
    double z_lo, z_hi;
    int64_t nsurf;
    int32_t npieces;
    int32_t gid_base;
    Surfel *surfels;
    int32_t *piece;
    int32_t *p_count;
    int32_t *p_wcomp;
    double *p_wsum;
    double *p_tsum;
    double *p_x0, *p_x1, *p_y0, *p_y1, *p_z0, *p_z1;
} UnitData;

typedef struct {
    int32_t a, b;
    int32_t count, k;
    double dist_sum;
} SeamPair;

static int seam_pair_cmp(const void *p1, const void *p2) {
    const SeamPair *a = p1, *b = p2;
    if (a->a != b->a)
        return a->a < b->a ? -1 : 1;
    if (a->b != b->b)
        return a->b < b->b ? -1 : 1;
    return (a->k > b->k) - (a->k < b->k);
}

static const uint8_t *g_mask;
static int64_t g_mask_dims[3];
static double g_mask_scale = 16.0;

static int load_mask(const char *path, const char *dims) {
    if (sscanf(dims, "%lld,%lld,%lld", (long long *)&g_mask_dims[0], (long long *)&g_mask_dims[1],
            (long long *)&g_mask_dims[2]) != 3)
        return -1;
    size_t n = (size_t)(g_mask_dims[0] * g_mask_dims[1] * g_mask_dims[2]);
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    uint8_t *m = malloc(n);
    size_t got = fread(m, 1, n, f);
    fclose(f);
    if (got != n) {
        free(m);
        return -1;
    }
    g_mask = m;
    return 0;
}

static inline int in_mask(const Surfel *s) {
    if (!g_mask)
        return 1;
    int64_t z = (int64_t)floor(s->z / g_mask_scale), y = (int64_t)floor(s->y / g_mask_scale),
            x = (int64_t)floor(s->x / g_mask_scale);
    if (z < 0 || y < 0 || x < 0 || z >= g_mask_dims[0] || y >= g_mask_dims[1] ||
        x >= g_mask_dims[2])
        return 0;
    return g_mask[(z * g_mask_dims[1] + y) * g_mask_dims[2] + x] != 0;
}

static void apply_mask(int32_t *piece, const Surfel *surfels, int64_t n) {
    if (!g_mask)
        return;
    for (int64_t i = 0; i < n; i++)
        if (piece[i] >= 0 && !in_mask(&surfels[i]))
            piece[i] = -1;
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

static int seam_pair_cnt_desc(const void *p1, const void *p2) {
    const SeamPair *a = p1, *b = p2;
    return (a->count < b->count) - (a->count > b->count);
}

typedef struct {
    int64_t key;
    int32_t idx;
} HashItem;

static int hash_item_cmp(const void *p1, const void *p2) {
    const HashItem *a = p1, *b = p2;
    return (a->key > b->key) - (a->key < b->key);
}

static inline int64_t pack_cell(int cz, int cy, int cx) {
    return ((int64_t)(cz + 100000) << 40) | ((int64_t)(cy + 100000) << 20) | (int64_t)(cx + 100000);
}

static int64_t lower_bound_key(const HashItem *arr, int64_t n, int64_t key) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (arr[mid].key < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

static int32_t uf_find(int32_t *uf, int32_t x) {
    while (uf[x] != x)
        x = uf[x] = uf[uf[x]];
    return x;
}

typedef struct {
    int32_t sheet_id;
    int64_t count;
} SheetOrder;

static int sheet_order_cmp(const void *p1, const void *p2) {
    const SheetOrder *a = p1, *b = p2;
    return (a->count < b->count) - (a->count > b->count);
}

typedef struct {
    int32_t wcomp, count, p;
} WCompItem;

static int wcomp_item_cmp(const void *p1, const void *p2) {
    const WCompItem *a = p1, *b = p2;
    if (a->wcomp != b->wcomp)
        return a->wcomp < b->wcomp ? -1 : 1;
    if (a->count != b->count)
        return a->count > b->count ? -1 : 1;
    return (a->p > b->p) - (a->p < b->p);
}

typedef struct {
    int32_t a, b;
} EdgePair;

static int g_keep_units;

static int load_unit(UnitData *u) {
    int64_t since = tick();
    char path[1200];
    snprintf(path, sizeof(path), "%s/surfels.bin", u->dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "stitch: cannot open %s\n", path);
        return -1;
    }
    int64_t hdr[2] = {0, 0};
    if (fread(hdr, sizeof(int64_t), 2, f) != 2) {
        fclose(f);
        return -1;
    }
    u->nsurf = hdr[0];
    u->npieces = (int32_t)hdr[1];
    u->surfels = malloc((size_t)(u->nsurf ? u->nsurf : 1) * sizeof(Surfel));
    u->piece = malloc((size_t)(u->nsurf ? u->nsurf : 1) * sizeof(int32_t));
    if (!u->surfels || !u->piece) {
        fclose(f);
        return -1;
    }
    if (fread(u->surfels, sizeof(Surfel), (size_t)u->nsurf, f) != (size_t)u->nsurf ||
        fread(u->piece, sizeof(int32_t), (size_t)u->nsurf, f) != (size_t)u->nsurf) {
        fclose(f);
        return -1;
    }
    fclose(f);
    apply_mask(u->piece, u->surfels, u->nsurf);
    u->p_count = calloc((size_t)(u->npieces ? u->npieces : 1), sizeof(int32_t));
    u->p_wcomp = calloc((size_t)(u->npieces ? u->npieces : 1), sizeof(int32_t));
    u->p_wsum = calloc((size_t)(u->npieces ? u->npieces : 1), sizeof(double));
    u->p_tsum = calloc((size_t)(u->npieces ? u->npieces : 1), sizeof(double));
    u->p_x0 = malloc((size_t)(u->npieces ? u->npieces : 1) * sizeof(double));
    u->p_x1 = malloc((size_t)(u->npieces ? u->npieces : 1) * sizeof(double));
    u->p_y0 = malloc((size_t)(u->npieces ? u->npieces : 1) * sizeof(double));
    u->p_y1 = malloc((size_t)(u->npieces ? u->npieces : 1) * sizeof(double));
    u->p_z0 = malloc((size_t)(u->npieces ? u->npieces : 1) * sizeof(double));
    u->p_z1 = malloc((size_t)(u->npieces ? u->npieces : 1) * sizeof(double));
    for (int32_t p = 0; p < u->npieces; p++) {
        u->p_x0[p] = u->p_y0[p] = u->p_z0[p] = 1e18;
        u->p_x1[p] = u->p_y1[p] = u->p_z1[p] = -1e18;
        u->p_wcomp[p] = -1;
    }
    for (int64_t i = 0; i < u->nsurf; i++) {
        int32_t p = u->piece[i];
        if (p < 0 || p >= u->npieces)
            continue;
        const Surfel *sf = &u->surfels[i];
        u->p_count[p]++;
        u->p_wsum[p] += sf->winding;
        u->p_tsum[p] += sf->theta;
        if (u->p_wcomp[p] < 0)
            u->p_wcomp[p] = sf->patch;
        if (sf->x < u->p_x0[p])
            u->p_x0[p] = sf->x;
        if (sf->x > u->p_x1[p])
            u->p_x1[p] = sf->x;
        if (sf->y < u->p_y0[p])
            u->p_y0[p] = sf->y;
        if (sf->y > u->p_y1[p])
            u->p_y1[p] = sf->y;
        if (sf->z < u->p_z0[p])
            u->p_z0[p] = sf->z;
        if (sf->z > u->p_z1[p])
            u->p_z1[p] = sf->z;
    }
    if (!g_keep_units) {
        free(u->surfels);
        free(u->piece);
        u->surfels = NULL;
        u->piece = NULL;
    }
    lap(T_LOAD, since);
    return 0;
}

static int load_unit_surfels(UnitData *u) {
    if (u->surfels && u->piece)
        return 0;
    int64_t since = tick();
    char path[1200];
    snprintf(path, sizeof(path), "%s/surfels.bin", u->dir);
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    int64_t hdr[2] = {0, 0};
    if (fread(hdr, sizeof(int64_t), 2, f) != 2) {
        fclose(f);
        return -1;
    }
    u->surfels = malloc((size_t)(u->nsurf ? u->nsurf : 1) * sizeof(Surfel));
    u->piece = malloc((size_t)(u->nsurf ? u->nsurf : 1) * sizeof(int32_t));
    if (!u->surfels || !u->piece) {
        fclose(f);
        return -1;
    }
    if (fread(u->surfels, sizeof(Surfel), (size_t)u->nsurf, f) != (size_t)u->nsurf ||
        fread(u->piece, sizeof(int32_t), (size_t)u->nsurf, f) != (size_t)u->nsurf) {
        fclose(f);
        return -1;
    }
    fclose(f);
    apply_mask(u->piece, u->surfels, u->nsurf);
    lap(T_LOAD, since);
    return 0;
}

static void unload_unit_surfels(UnitData *u) {
    if (g_keep_units)
        return;
    free(u->surfels);
    free(u->piece);
    u->surfels = NULL;
    u->piece = NULL;
}

static HashItem *seam_cells(
    const UnitData *ua, double ov_lo, double ov_hi, double cell_sz, int64_t *count) {
    int64_t na = 0;
    for (int64_t i = 0; i < ua->nsurf; i++)
        if (ua->piece[i] >= 0 && ua->surfels[i].z >= ov_lo - 4.0 && ua->surfels[i].z <= ov_hi + 4.0)
            na++;
    HashItem *ha = malloc((size_t)(na ? na : 1) * sizeof(HashItem));
    int64_t pos = 0;
    for (int64_t i = 0; i < ua->nsurf; i++) {
        if (ua->piece[i] < 0 || ua->surfels[i].z < ov_lo - 4.0 || ua->surfels[i].z > ov_hi + 4.0)
            continue;
        const Surfel *s = &ua->surfels[i];
        ha[pos++] = (HashItem){pack_cell((int)floor(s->z / cell_sz), (int)floor(s->y / cell_sz),
                                   (int)floor(s->x / cell_sz)),
            (int32_t)i};
    }
    qsort(ha, (size_t)na, sizeof(HashItem), hash_item_cmp);
    *count = na;
    return ha;
}

static int32_t seam_nearest(const UnitData *ua, const HashItem *ha, int64_t na, const Surfel *sb,
    double cell_sz, float reach, float normal_reach, float *dist) {
    int cz0 = (int)floor(sb->z / cell_sz);
    int cy0 = (int)floor(sb->y / cell_sz);
    int cx0 = (int)floor(sb->x / cell_sz);
    int32_t best = -1;
    float best_d = reach;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int64_t key = pack_cell(cz0 + dz, cy0 + dy, cx0 + dx);
                int64_t p0 = lower_bound_key(ha, na, key);
                for (; p0 < na && ha[p0].key == key; p0++) {
                    int32_t ia = ha[p0].idx;
                    float d;
                    if (!stitch_seam_near(&ua->surfels[ia].spot, &sb->spot, normal_reach, &d) ||
                        d >= best_d)
                        continue;
                    best_d = d;
                    best = ia;
                }
            }
        }
    }
    *dist = best_d;
    return best;
}

static SeamPair seam_pair(
    const UnitData *ua, const UnitData *ub, int32_t ia, int64_t ib, float dist, const int32_t *wl) {
    int32_t ga = ua->gid_base + ua->piece[ia], gb = ub->gid_base + ub->piece[ib];
    int32_t k = (ua->surfels[ia].winding - wl[ga]) - (ub->surfels[ib].winding - wl[gb]);
    return (SeamPair){ga, gb, 1, k, dist};
}

static SeamPair *seam_raw(const UnitData *ua, const UnitData *ub, double ov_lo, double ov_hi,
    double seam_dist, double seam_dn, const int32_t *wl, int64_t *count) {
    double cell_sz = 12.0;
    int64_t na = 0, rcap = 65536, rcnt = 0;
    HashItem *ha = seam_cells(ua, ov_lo, ov_hi, cell_sz, &na);
    SeamPair *raw = malloc((size_t)rcap * sizeof(SeamPair));
    for (int64_t i = 0; i < ub->nsurf; i++) {
        if (ub->piece[i] < 0)
            continue;
        const Surfel *sb = &ub->surfels[i];
        if (sb->z < ov_lo || sb->z > ov_hi)
            continue;
        float best_d;
        int32_t ia =
            seam_nearest(ua, ha, na, sb, cell_sz, (float)seam_dist, (float)seam_dn, &best_d);
        if (ia < 0)
            continue;
        if (rcnt == rcap) {
            rcap *= 2;
            raw = realloc(raw, (size_t)rcap * sizeof(SeamPair));
        }
        raw[rcnt++] = seam_pair(ua, ub, ia, i, best_d, wl);
    }
    free(ha);
    *count = rcnt;
    return raw;
}

#ifdef STITCH_CUDA
static int64_t g_ties, g_fell;

static SeamPair *seam_raw_cuda(StitchCuda *gpu, int k, const UnitData *ua, const UnitData *ub,
    double ov_lo, double ov_hi, double seam_dist, double seam_dn, const int32_t *wl,
    int64_t *count) {
    StitchSeamHit *hits = NULL;
    if (stitch_cuda_unit(gpu, k, (const StitchSurfel *)ua->surfels, ua->piece, ua->nsurf) ||
        stitch_cuda_unit(gpu, k + 1, (const StitchSurfel *)ub->surfels, ub->piece, ub->nsurf))
        return NULL;
    int64_t found =
        stitch_cuda_seam(gpu, k, k + 1, ov_lo, ov_hi, (float)seam_dist, (float)seam_dn, &hits);
    if (found < 0)
        return NULL;
    SeamPair *raw = malloc((size_t)(found ? found : 1) * sizeof(SeamPair));
    HashItem *ha = NULL;
    int64_t na = 0, rcnt = 0;
    for (int64_t h = 0; h < found; h++) {
        int32_t ia = hits[h].hit;
        float best_d = hits[h].dist;
        if (hits[h].tie) {
            if (!ha)
                ha = seam_cells(ua, ov_lo, ov_hi, 12.0, &na);
            ia = seam_nearest(ua, ha, na, &ub->surfels[hits[h].at], 12.0, (float)seam_dist,
                (float)seam_dn, &best_d);
            g_ties++;
        }
        if (ia >= 0)
            raw[rcnt++] = seam_pair(ua, ub, ia, hits[h].at, best_d, wl);
    }
    free(ha);
    free(hits);
    *count = rcnt;
    return raw;
}
#endif

typedef struct {
    int32_t a, b, k, count;
} NbrVote;

static int g_nbr_trust = 0;
static int g_bridge = 3;
static const int32_t *g_comp_of;

static int nbr_vote_cmp(const void *p1, const void *p2) {
    const NbrVote *x = p1, *y = p2;
    if (x->a != y->a)
        return x->a < y->a ? -1 : 1;
    if (x->b != y->b)
        return x->b < y->b ? -1 : 1;
    return (x->k > y->k) - (x->k < y->k);
}

static int32_t ray_nearest(
    const UnitData *u, const HashItem *h, int64_t m, int64_t i, double max_hit, double cs) {
    int32_t pa = u->piece[i];
    const Surfel *sa = &u->surfels[i];
    int32_t best = -1;
    double best_t = 1e9;
    for (double t = 4.0; t <= max_hit && best < 0; t += 2.0) {
        double qx = sa->x + sa->nx * t, qy = sa->y + sa->ny * t, qz = sa->z + sa->nz * t;
        int cz = (int)floor(qz / cs), cy = (int)floor(qy / cs), cx = (int)floor(qx / cs);
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int64_t key = pack_cell(cz + dz, cy + dy, cx + dx);
                    int64_t p0 = lower_bound_key(h, m, key);
                    for (; p0 < m && h[p0].key == key; p0++) {
                        double along;
                        if (u->piece[h[p0].idx] == pa ||
                            !stitch_ray_near(
                                &sa->spot, &u->surfels[h[p0].idx].spot, qx, qy, qz, &along))
                            continue;
                        double tt = t + along;
                        if (tt < best_t) {
                            best_t = tt;
                            best = h[p0].idx;
                        }
                    }
                }
            }
        }
    }
    return best;
}

static NbrVote ray_vote(const UnitData *u, const int32_t *wl, int64_t i, int32_t best) {
    int32_t pa = u->piece[i], pb = u->piece[best];
    const Surfel *sa = &u->surfels[i], *sb = &u->surfels[best];
    int32_t ga = u->gid_base + pa, gb = u->gid_base + pb;
    int32_t dw = sb->winding - sa->winding;
    int same_comp = g_comp_of && g_comp_of[ga] >= 0 && g_comp_of[ga] == g_comp_of[gb];
    int32_t step = g_nbr_trust && same_comp ? dw : 1;
    int32_t k = step - dw + (wl[gb] - wl[ga]);
    return (NbrVote){ga, gb, k, dw};
}

typedef struct {
    UnitData *u;
    const int32_t *wl;
    const HashItem *h;
    int64_t m, n;
    int sample_every;
    double max_hit, cs;
    int64_t block;
    NbrVote **bv;
    int64_t *bn;
} RayJob;

static void ray_block(void *ctx, size_t bi) {
    RayJob *j = ctx;
    UnitData *u = j->u;
    int64_t i0 = (int64_t)bi * j->block, i1 = i0 + j->block;
    if (i1 > j->n)
        i1 = j->n;
    int64_t cap = 4096, cnt = 0;
    NbrVote *v = malloc((size_t)cap * sizeof(NbrVote));
    for (int64_t i = i0; i < i1; i += j->sample_every) {
        if (u->piece[i] < 0)
            continue;
        int32_t best = ray_nearest(u, j->h, j->m, i, j->max_hit, j->cs);
        if (best < 0)
            continue;
        if (cnt == cap) {
            cap *= 2;
            v = realloc(v, (size_t)cap * sizeof(NbrVote));
        }
        v[cnt++] = ray_vote(u, j->wl, i, best);
    }
    j->bv[bi] = v;
    j->bn[bi] = cnt;
}

static HashItem *ray_cells(const UnitData *u, double cs, int64_t *count) {
    int64_t n = u->nsurf, m = 0;
    for (int64_t i = 0; i < n; i++)
        m += u->piece[i] >= 0;
    HashItem *h = malloc((size_t)(m ? m : 1) * sizeof(HashItem));
    int64_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        if (u->piece[i] < 0)
            continue;
        const Surfel *sf = &u->surfels[i];
        h[pos++] = (HashItem){
            pack_cell((int)floor(sf->z / cs), (int)floor(sf->y / cs), (int)floor(sf->x / cs)),
            (int32_t)i};
    }
    qsort(h, (size_t)m, sizeof(HashItem), hash_item_cmp);
    *count = m;
    return h;
}

static NbrVote *ray_raw(
    UnitData *u, const int32_t *wl, int sample_every, double max_hit, int64_t *count) {
    double cs = 6.0;
    int64_t n = u->nsurf, m = 0;
    HashItem *h = ray_cells(u, cs, &m);
    int64_t block = 65536 * (int64_t)sample_every;
    int64_t nb = (n + block - 1) / block;
    RayJob job = {.u = u,
        .wl = wl,
        .h = h,
        .m = m,
        .n = n,
        .sample_every = sample_every,
        .max_hit = max_hit,
        .cs = cs,
        .block = block,
        .bv = calloc((size_t)(nb ? nb : 1), sizeof(NbrVote *)),
        .bn = calloc((size_t)(nb ? nb : 1), sizeof(int64_t))};
    each_block((size_t)nb, &job, ray_block);
    int64_t cnt = 0;
    for (int64_t b = 0; b < nb; b++)
        cnt += job.bn[b];
    NbrVote *v = malloc((size_t)(cnt ? cnt : 1) * sizeof(NbrVote));
    cnt = 0;
    for (int64_t b = 0; b < nb; b++) {
        memcpy(v + cnt, job.bv[b], (size_t)job.bn[b] * sizeof(NbrVote));
        cnt += job.bn[b];
        free(job.bv[b]);
    }
    free(job.bv);
    free(job.bn);
    free(h);
    *count = cnt;
    return v;
}

#ifdef STITCH_CUDA
static NbrVote *ray_raw_cuda(StitchCuda *gpu, int k, const UnitData *u, const int32_t *wl,
    int sample_every, double max_hit, int64_t *count) {
    StitchRayHit *hits = NULL;
    if (stitch_cuda_unit(gpu, k, (const StitchSurfel *)u->surfels, u->piece, u->nsurf))
        return NULL;
    int64_t found = stitch_cuda_rays(gpu, k, sample_every, max_hit, &hits);
    if (found < 0)
        return NULL;
    NbrVote *v = malloc((size_t)(found ? found : 1) * sizeof(NbrVote));
    HashItem *h = NULL;
    int64_t m = 0, cnt = 0;
    for (int64_t i = 0; i < found; i++) {
        int32_t best = hits[i].hit;
        if (hits[i].tie) {
            if (!h)
                h = ray_cells(u, 6.0, &m);
            best = ray_nearest(u, h, m, hits[i].at, max_hit, 6.0);
            g_ties++;
        }
        if (best >= 0)
            v[cnt++] = ray_vote(u, wl, hits[i].at, best);
    }
    free(h);
    free(hits);
    *count = cnt;
    return v;
}
#endif

static int64_t radial_votes(StitchCuda *gpu, int unit, UnitData *u, const int32_t *wl,
    int sample_every, double max_hit, NbrVote **out) {
    *out = NULL;
    if (load_unit_surfels(u))
        return -1;
    int64_t since = tick(), cnt = 0;
    NbrVote *v = NULL;
#ifdef STITCH_CUDA
    if (gpu) {
        v = ray_raw_cuda(gpu, unit, u, wl, sample_every, max_hit, &cnt);
        g_fell += !v;
    }
#endif
    if (!v)
        v = ray_raw(u, wl, sample_every, max_hit, &cnt);
    for (int64_t i = 0; i < cnt; i++)
        v[i].count = 1;
    unload_unit_surfels(u);
    check_raw("rays", unit, v, cnt, sizeof(NbrVote));
    qsort(v, (size_t)cnt, sizeof(NbrVote), nbr_vote_cmp);
    int64_t outn = 0;
    for (int64_t a = 0; a < cnt;) {
        int64_t b = a;
        int32_t best_k = 0, best_c = 0, tot = 0;
        while (b < cnt && v[b].a == v[a].a && v[b].b == v[a].b) {
            int64_t c = b;
            while (c < cnt && v[c].a == v[b].a && v[c].b == v[b].b && v[c].k == v[b].k)
                c++;
            tot += (int32_t)(c - b);
            if (c - b > best_c) {
                best_c = (int32_t)(c - b);
                best_k = v[b].k;
            }
            b = c;
        }
        if (best_c >= 4 && best_c * 10 >= tot * 6)
            v[outn++] = (NbrVote){v[a].a, v[a].b, best_k, best_c};
        a = b;
    }
    *out = v;
    lap(T_RAYS, since);
    return outn;
}

typedef struct {
    const char *out_dir, *sample, *scan;
    int curl, stride, min_nodes, fill_reach, fair_rounds, max_hole_cells, split_depth;
    double voxel_um, phi_gate;
    const HxAxisKnot *axis;
    int64_t axis_count;
    json_object *rec_pieces;
    pthread_mutex_t mu;
    double total_cm2, area_4cm2, area_20cm2, largest_cm2;
    int written_count, pieces_4, pieces_20, iso_pass, fold_splits, fold_dropped;
} GlobalEmitCtx;

static inline double surfel_phi(const Surfel *s, int curl) {
    return stitch_phi(s->theta, s->winding, curl);
}

static double cell_phi(const GlobalEmitCtx *ec, double x, double y, double z, double near) {
    double ax = 0.0, ay = 0.0;
    hx_centre_at(ec->axis, ec->axis_count, z, &ax, &ay);
    double th = (double)ec->curl * atan2(y - ay, x - ax);
    return th + 2.0 * M_PI * floor((near - th) / (2.0 * M_PI) + 0.5);
}

static int double_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef struct {
    int gw, gh, depth, units_spanned, is_iso;
    int32_t nodes;
    int64_t cells;
    size_t gsize;
    float *tx, *ty, *tz, *tw, *tp;
    uint8_t *qual;
    double bb_min[3], bb_max[3];
    double phi_min, phi_max, phi_mean, fold_share, area_mm2, area_cm2, within, jitter;
} PieceGrid;

static int piece_grid(const GlobalEmitCtx *ec, const Surfel *pts, const int32_t *members,
    int32_t keep_n, const float *pu, const float *pv, const double *sphi, int units_spanned,
    int depth, PieceGrid *pg) {
    int stride = ec->stride;
    double umin = DBL_MAX, umax = -DBL_MAX, vmin = DBL_MAX, vmax = -DBL_MAX;
    for (int32_t i = 0; i < keep_n; i++) {
        if (pu[i] < umin)
            umin = pu[i];
        if (pu[i] > umax)
            umax = pu[i];
        if (pv[i] < vmin)
            vmin = pv[i];
        if (pv[i] > vmax)
            vmax = pv[i];
    }
    int gw = (int)floor((umax - umin) / stride) + 1;
    int gh = (int)floor((vmax - vmin) / stride) + 1;
    int too_big = gw < 2 || gh < 2 || (int64_t)gw * gh > 32000000;
    double *xyz = NULL, *nrm = NULL, *du = NULL, *dv = NULL, *grid = NULL, *spread = NULL;
    int64_t *flat_cell = NULL;
    uint8_t *locked = NULL;
    size_t gsize = 0;
    int64_t occ = 0, lock_cnt = 0;
    double fold_share = 1.0;
    if (!too_big) {
        xyz = malloc((size_t)keep_n * 3 * sizeof(double));
        nrm = malloc((size_t)keep_n * 3 * sizeof(double));
        du = malloc((size_t)keep_n * sizeof(double));
        dv = malloc((size_t)keep_n * sizeof(double));
        flat_cell = malloc((size_t)keep_n * sizeof(int64_t));
        gsize = (size_t)gh * gw;
        grid = calloc(3 * gsize, sizeof(double));
        locked = calloc(gsize, 1);
        spread = calloc(gsize, sizeof(double));
        for (int32_t i = 0; i < keep_n; i++) {
            const Surfel *s = &pts[members[i]];
            xyz[3 * i + 0] = s->x;
            xyz[3 * i + 1] = s->y;
            xyz[3 * i + 2] = s->z;
            nrm[3 * i + 0] = s->nx;
            nrm[3 * i + 1] = s->ny;
            nrm[3 * i + 2] = s->nz;
            du[i] = pu[i] - umin;
            dv[i] = pv[i] - vmin;
        }
        occ = hx_raster_cells(
            gh, gw, keep_n, xyz, nrm, du, dv, (double)stride, 2.2, flat_cell, grid, locked, spread);
        for (size_t j = 0; j < gsize; j++)
            lock_cnt += locked[j];
        fold_share = occ > 0 ? 1.0 - (double)lock_cnt / (double)occ : 1.0;
    }
    if (too_big || occ <= 0 || fold_share > 0.30 || lock_cnt < ec->min_nodes / 2) {
        free(xyz);
        free(nrm);
        free(du);
        free(dv);
        free(flat_cell);
        free(grid);
        free(locked);
        free(spread);
        return 0;
    }

    hx_centre_cells(gh, gw, keep_n, flat_cell, du, dv, xyz, NULL, locked, (double)stride, 4, 0.1,
        0.5, 1.5, grid);

    double *cphi = calloc(gsize, sizeof(double));
    int32_t *cphi_n = calloc(gsize, sizeof(int32_t));
    for (int32_t i = 0; i < keep_n; i++) {
        int64_t c = flat_cell[i];
        if (c < 0 || (size_t)c >= gsize)
            continue;
        if (cphi_n[c] == 0)
            cphi[c] = sphi[i];
        else
            cphi[c] +=
                sphi[i] + 2.0 * M_PI * floor((cphi[c] / cphi_n[c] - sphi[i]) / (2.0 * M_PI) + 0.5);
        cphi_n[c]++;
    }
    for (size_t j = 0; j < gsize; j++)
        if (cphi_n[j] > 0)
            cphi[j] /= cphi_n[j];

    uint8_t *known = malloc(gsize);
    memcpy(known, locked, gsize);
    uint8_t *hole = calloc(gsize, 1);
    int64_t hole_cnt = 0;
    int fr = ec->fill_reach > 0 ? ec->fill_reach : 5;
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
                double step_len = (d >= 2 ? 1.41421356 : 1.0) * stride * span;
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
    if (hole_cnt > 0) {
        biharmonic_fill_mesh(grid, known, hole, gh, gw, (double)stride, 160);
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
                if (has_u && (lu < 0.45 * stride || lu > 1.75 * stride))
                    bad = 1;
                if (has_v && (lv < 0.45 * stride || lv > 1.75 * stride))
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
        tangential_isometric_fair(grid, known, locked, gh, gw, (double)stride, ec->fair_rounds);

    uint8_t *phi_set = calloc(gsize, 1);
    int64_t qh = 0, qt = 0;
    for (size_t j = 0; j < gsize; j++) {
        if (known[j] && cphi_n[j] > 0) {
            cphi[j] = cell_phi(ec, grid[j], grid[gsize + j], grid[2 * gsize + j], cphi[j]);
            phi_set[j] = 1;
            hq[qt++] = (int64_t)j;
        }
    }
    while (qh < qt) {
        int64_t cur_i = hq[qh++];
        int rr = (int)(cur_i / gw), cc = (int)(cur_i % gw);
        int64_t nbs[4] = {rr > 0 ? cur_i - gw : -1, rr + 1 < gh ? cur_i + gw : -1,
            cc > 0 ? cur_i - 1 : -1, cc + 1 < gw ? cur_i + 1 : -1};
        for (int d = 0; d < 4; d++) {
            int64_t ni = nbs[d];
            if (ni < 0 || !known[ni] || phi_set[ni])
                continue;
            cphi[ni] = cell_phi(ec, grid[ni], grid[gsize + ni], grid[2 * gsize + ni], cphi[cur_i]);
            phi_set[ni] = 1;
            hq[qt++] = ni;
        }
    }
    free(hq);

    double iso[14] = {0};
    hx_mesh_isometry(grid, gh, gw, known, locked, (double)stride, iso);
    int64_t known_cnt = 0;
    for (size_t j = 0; j < gsize; j++)
        known_cnt += known[j] && phi_set[j];
    double area_mm2 =
        (double)known_cnt * (stride * ec->voxel_um * 1e-3) * (stride * ec->voxel_um * 1e-3);
    double area_cm2 = area_mm2 * 0.01;
    int is_iso = iso[7 + 5] >= 0.70 && iso[7 + 6] <= 0.07;

    float *tx = malloc(gsize * sizeof(float));
    float *ty = malloc(gsize * sizeof(float));
    float *tz = malloc(gsize * sizeof(float));
    float *tw = malloc(gsize * sizeof(float));
    float *tp = malloc(gsize * sizeof(float));
    double bb_min[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, bb_max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    double phi_min = DBL_MAX, phi_max = -DBL_MAX, phi_sum = 0.0;
    for (size_t j = 0; j < gsize; j++) {
        if (!known[j] || !phi_set[j]) {
            tx[j] = ty[j] = tz[j] = -1.0f;
            tw[j] = NAN;
            tp[j] = NAN;
            continue;
        }
        tx[j] = (float)grid[j];
        ty[j] = (float)grid[gsize + j];
        tz[j] = (float)grid[2 * gsize + j];
        tp[j] = (float)cphi[j];
        double ax = 0.0, ay = 0.0;
        hx_centre_at(ec->axis, ec->axis_count, grid[2 * gsize + j], &ax, &ay);
        double th = (double)ec->curl * atan2(grid[gsize + j] - ay, grid[j] - ax);
        tw[j] = (float)lrint((cphi[j] - th) / (2.0 * M_PI));
        if (cphi[j] < phi_min)
            phi_min = cphi[j];
        if (cphi[j] > phi_max)
            phi_max = cphi[j];
        phi_sum += cphi[j];
        for (int a = 0; a < 3; a++) {
            double val = grid[(size_t)a * gsize + j];
            if (val < bb_min[a])
                bb_min[a] = val;
            if (val > bb_max[a])
                bb_max[a] = val;
        }
    }
    free(phi_set);
    free(cphi);
    free(cphi_n);
    uint8_t *qual = calloc(gsize, 1);
    for (size_t j = 0; j < gsize; j++)
        qual[j] = locked[j] ? 1 : (known[j] ? 2 : 0);
    *pg = (PieceGrid){.gw = gw,
        .gh = gh,
        .depth = depth,
        .units_spanned = units_spanned,
        .is_iso = is_iso,
        .nodes = keep_n,
        .cells = lock_cnt,
        .gsize = gsize,
        .tx = tx,
        .ty = ty,
        .tz = tz,
        .tw = tw,
        .tp = tp,
        .qual = qual,
        .phi_min = phi_min,
        .phi_max = phi_max,
        .phi_mean = known_cnt > 0 ? phi_sum / (double)known_cnt : 0.0,
        .fold_share = fold_share,
        .area_mm2 = area_mm2,
        .area_cm2 = area_cm2,
        .within = iso[7 + 5],
        .jitter = iso[7 + 6]};
    memcpy(pg->bb_min, bb_min, sizeof(bb_min));
    memcpy(pg->bb_max, bb_max, sizeof(bb_max));
    free(known);
    free(hole);
    free(xyz);
    free(nrm);
    free(du);
    free(dv);
    free(flat_cell);
    free(grid);
    free(locked);
    free(spread);
    return 1;
}

static void piece_free(PieceGrid *pg) {
    free(pg->tx);
    free(pg->ty);
    free(pg->tz);
    free(pg->tw);
    free(pg->tp);
    free(pg->qual);
}

static json_object *piece_write(const GlobalEmitCtx *ec, const PieceGrid *pg, int my_id) {
    int stride = ec->stride, gw = pg->gw, gh = pg->gh;
    char pdir[4096], fpath[4096], uuid[64];
    snprintf(uuid, sizeof(uuid), "piece-%04d", my_id);
    snprintf(pdir, sizeof(pdir), "%s/%s", ec->out_dir, uuid);
    make_dirs(pdir);
    snprintf(fpath, sizeof(fpath), "%s/x.tif", pdir);
    write_tiff_f32(fpath, pg->tx, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/y.tif", pdir);
    write_tiff_f32(fpath, pg->ty, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/z.tif", pdir);
    write_tiff_f32(fpath, pg->tz, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/w.tif", pdir);
    write_tiff_f32(fpath, pg->tw, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/phi.tif", pdir);
    write_tiff_f32(fpath, pg->tp, gh, gw);
    snprintf(fpath, sizeof(fpath), "%s/quality.npy", pdir);
    write_npy_u8(fpath, pg->qual, gh, gw);

    json_object *meta = json_object_new_object();
    json_object_object_add(meta, "sample", json_object_new_string(ec->sample));
    json_object_object_add(meta, "scan", json_object_new_string(ec->scan));
    json_object_object_add(meta, "voxel_um", json_object_new_double(ec->voxel_um));
    json_object_object_add(meta, "piece", json_object_new_int(my_id));
    json_object_object_add(
        meta, "winding", json_object_new_int((int)floor(pg->phi_mean / (2.0 * M_PI) + 0.5)));
    json_object_object_add(meta, "phi_min", json_object_new_double(pg->phi_min));
    json_object_object_add(meta, "phi_max", json_object_new_double(pg->phi_max));
    json_object_object_add(meta, "phi_mean", json_object_new_double(pg->phi_mean));
    json_object_object_add(meta, "split_depth", json_object_new_int(pg->depth));
    json_object_object_add(meta, "units_spanned", json_object_new_int(pg->units_spanned));
    json_object_object_add(meta, "cells", json_object_new_int64(pg->cells));
    json_object_object_add(meta, "nodes", json_object_new_int(pg->nodes));
    json_object_object_add(meta, "fold_share", json_object_new_double(pg->fold_share));
    json_object_object_add(meta, "area_mm2", json_object_new_double(pg->area_mm2));
    json_object_object_add(
        meta, "width_mm", json_object_new_double(gw * stride * ec->voxel_um * 1e-3));
    json_object_object_add(
        meta, "height_mm", json_object_new_double(gh * stride * ec->voxel_um * 1e-3));
    json_object_object_add(meta, "written", json_object_new_boolean(1));
    json_object_object_add(meta, "format", json_object_new_string("tifxyz"));
    json_object_object_add(meta, "type", json_object_new_string("seg"));
    json_object_object_add(meta, "uuid", json_object_new_string(uuid));
    json_object_object_add(meta, "stride", json_object_new_double((double)stride));
    json_object *sc = json_object_new_array_ext(2);
    json_object_array_add(sc, json_object_new_double(1.0 / stride));
    json_object_array_add(sc, json_object_new_double(1.0 / stride));
    json_object_object_add(meta, "scale", sc);
    json_object *bb = json_object_new_array_ext(2);
    json_object *b0 = json_object_new_array_ext(3), *b1 = json_object_new_array_ext(3);
    for (int a = 0; a < 3; a++) {
        json_object_array_add(b0, json_object_new_double(pg->bb_min[a]));
        json_object_array_add(b1, json_object_new_double(pg->bb_max[a]));
    }
    json_object_array_add(bb, b0);
    json_object_array_add(bb, b1);
    json_object_object_add(meta, "bbox", bb);
    json_object *iso_obj = json_object_new_object();
    json_object *lk = json_object_new_object();
    json_object_object_add(lk, "within_10pct", json_object_new_double(pg->within));
    json_object_object_add(lk, "jitter", json_object_new_double(pg->jitter));
    json_object_object_add(iso_obj, "locked", lk);
    json_object_object_add(iso_obj, "isometric", json_object_new_boolean(pg->is_iso));
    json_object_object_add(meta, "isometry", iso_obj);
    snprintf(fpath, sizeof(fpath), "%s/meta.json", pdir);
    json_object_to_file_ext(fpath, meta, JSON_C_TO_STRING_PRETTY);
    return meta;
}

enum { PIECE_KEYS = 12 };

typedef struct {
    json_object *meta;
    double key[PIECE_KEYS];
    int old;
} PieceOrder;

static double meta_number(json_object *meta, const char *name) {
    json_object *v = NULL;
    return json_object_object_get_ex(meta, name, &v) ? json_object_get_double(v) : 0.0;
}

static double meta_corner(json_object *meta, int corner, int axis) {
    json_object *bb = NULL;
    if (!json_object_object_get_ex(meta, "bbox", &bb))
        return 0.0;
    return json_object_get_double(
        json_object_array_get_idx(json_object_array_get_idx(bb, corner), axis));
}

static int piece_order_cmp(const void *p1, const void *p2) {
    const PieceOrder *a = p1, *b = p2;
    for (int k = 0; k < PIECE_KEYS; k++)
        if (a->key[k] != b->key[k])
            return a->key[k] < b->key[k] ? -1 : 1;
    return (a->old > b->old) - (a->old < b->old);
}

static int renumber_pieces(GlobalEmitCtx *ec) {
    size_t n = json_object_array_length(ec->rec_pieces);
    PieceOrder *order = calloc(n ? n : 1, sizeof(*order));
    if (!order)
        return 1;
    for (size_t i = 0; i < n; i++) {
        json_object *meta = json_object_array_get_idx(ec->rec_pieces, i);
        PieceOrder *o = &order[i];
        o->meta = meta;
        o->old = (int)meta_number(meta, "piece");
        double key[PIECE_KEYS] = {meta_number(meta, "phi_min"), meta_corner(meta, 0, 2),
            meta_corner(meta, 0, 1), meta_corner(meta, 0, 0), meta_number(meta, "phi_max"),
            meta_corner(meta, 1, 2), meta_corner(meta, 1, 1), meta_corner(meta, 1, 0),
            meta_number(meta, "phi_mean"), meta_number(meta, "area_mm2"),
            meta_number(meta, "nodes"), meta_number(meta, "cells")};
        memcpy(o->key, key, sizeof(key));
    }
    qsort(order, n, sizeof(*order), piece_order_cmp);
    char from[4096], to[4096], uuid[64];
    int failed = 0;
    for (size_t i = 0; i < n && !failed; i++) {
        snprintf(from, sizeof(from), "%s/piece-%04d", ec->out_dir, order[i].old);
        snprintf(to, sizeof(to), "%s/.renumber-%04d", ec->out_dir, order[i].old);
        failed = rename(from, to) != 0;
    }
    json_object *list = json_object_new_array_ext((int)n);
    ec->total_cm2 = ec->area_4cm2 = ec->area_20cm2 = ec->largest_cm2 = 0.0;
    ec->pieces_4 = ec->pieces_20 = ec->iso_pass = 0;
    for (size_t i = 0; i < n && !failed; i++) {
        json_object *meta = order[i].meta;
        snprintf(uuid, sizeof(uuid), "piece-%04d", (int)i);
        snprintf(from, sizeof(from), "%s/.renumber-%04d", ec->out_dir, order[i].old);
        snprintf(to, sizeof(to), "%s/%s", ec->out_dir, uuid);
        if (rename(from, to) != 0) {
            failed = 1;
            break;
        }
        json_object_object_add(meta, "piece", json_object_new_int((int)i));
        json_object_object_add(meta, "uuid", json_object_new_string(uuid));
        snprintf(to, sizeof(to), "%s/%s/meta.json", ec->out_dir, uuid);
        failed = json_object_to_file_ext(to, meta, JSON_C_TO_STRING_PRETTY) != 0;
        json_object_array_add(list, json_object_get(meta));
        double area_cm2 = meta_number(meta, "area_mm2") * 0.01;
        ec->total_cm2 += area_cm2;
        if (area_cm2 >= 4.0) {
            ec->area_4cm2 += area_cm2;
            ec->pieces_4++;
        }
        if (area_cm2 >= 20.0) {
            ec->area_20cm2 += area_cm2;
            ec->pieces_20++;
        }
        if (area_cm2 > ec->largest_cm2)
            ec->largest_cm2 = area_cm2;
        json_object *iso = NULL, *flag = NULL;
        if (json_object_object_get_ex(meta, "isometry", &iso) &&
            json_object_object_get_ex(iso, "isometric", &flag))
            ec->iso_pass += json_object_get_boolean(flag);
    }
    json_object_put(ec->rec_pieces);
    ec->rec_pieces = list;
    free(order);
    return failed;
}

static void piece_count(GlobalEmitCtx *ec, const PieceGrid *pg, json_object *meta) {
    double area_cm2 = pg->area_cm2;
    json_object_array_add(ec->rec_pieces, meta);
    ec->total_cm2 += area_cm2;
    if (area_cm2 >= 4.0) {
        ec->area_4cm2 += area_cm2;
        ec->pieces_4++;
    }
    if (area_cm2 >= 20.0) {
        ec->area_20cm2 += area_cm2;
        ec->pieces_20++;
    }
    if (area_cm2 > ec->largest_cm2)
        ec->largest_cm2 = area_cm2;
    ec->iso_pass += pg->is_iso;
}

typedef void (*PartFn)(void *ctx, const int32_t *members, int32_t count, int32_t *ea, int32_t *eb,
    int64_t edges, int side, int32_t index);

static void each_component(int32_t min_nodes, const int32_t *members, int32_t cnt,
    const int32_t *ea, const int32_t *eb, int64_t ecnt, int side, PartFn part, void *ctx) {
    if (cnt < min_nodes)
        return;
    int32_t *cuf = malloc((size_t)cnt * sizeof(int32_t));
    for (int32_t i = 0; i < cnt; i++)
        cuf[i] = i;
    for (int64_t e = 0; e < ecnt; e++) {
        int32_t ra = uf_find(cuf, ea[e]), rb = uf_find(cuf, eb[e]);
        if (ra != rb)
            cuf[rb] = ra;
    }
    int32_t *comp = malloc((size_t)cnt * sizeof(int32_t));
    int32_t *root_comp = malloc((size_t)cnt * sizeof(int32_t));
    for (int32_t i = 0; i < cnt; i++)
        root_comp[i] = -1;
    int32_t ncomp = 0;
    for (int32_t i = 0; i < cnt; i++) {
        int32_t r = uf_find(cuf, i);
        if (root_comp[r] < 0)
            root_comp[r] = ncomp++;
        comp[i] = root_comp[r];
    }
    free(root_comp);
    free(cuf);
    int64_t *msize = calloc((size_t)ncomp + 1, sizeof(int64_t));
    int64_t *esize = calloc((size_t)ncomp + 1, sizeof(int64_t));
    for (int32_t i = 0; i < cnt; i++)
        msize[comp[i] + 1]++;
    for (int64_t e = 0; e < ecnt; e++)
        esize[comp[ea[e]] + 1]++;
    for (int32_t c = 0; c < ncomp; c++) {
        msize[c + 1] += msize[c];
        esize[c + 1] += esize[c];
    }
    int32_t *cm = malloc((size_t)(cnt ? cnt : 1) * sizeof(int32_t));
    int32_t *loc = malloc((size_t)(cnt ? cnt : 1) * sizeof(int32_t));
    int32_t *cea = malloc((size_t)(ecnt ? ecnt : 1) * sizeof(int32_t));
    int32_t *ceb = malloc((size_t)(ecnt ? ecnt : 1) * sizeof(int32_t));
    int64_t *mfill = malloc((size_t)(ncomp + 1) * sizeof(int64_t));
    int64_t *efill = malloc((size_t)(ncomp + 1) * sizeof(int64_t));
    memcpy(mfill, msize, (size_t)(ncomp + 1) * sizeof(int64_t));
    memcpy(efill, esize, (size_t)(ncomp + 1) * sizeof(int64_t));
    for (int32_t i = 0; i < cnt; i++) {
        int32_t c = comp[i];
        loc[i] = (int32_t)(mfill[c] - msize[c]);
        cm[mfill[c]++] = members[i];
    }
    for (int64_t e = 0; e < ecnt; e++) {
        int32_t c = comp[ea[e]];
        cea[efill[c]] = loc[ea[e]];
        ceb[efill[c]++] = loc[eb[e]];
    }
    free(mfill);
    free(efill);
    free(loc);
    free(comp);
    for (int32_t c = 0; c < ncomp; c++) {
        int32_t n_c = (int32_t)(msize[c + 1] - msize[c]);
        if (n_c < min_nodes)
            continue;
        part(ctx, cm + msize[c], n_c, cea + esize[c], ceb + esize[c], esize[c + 1] - esize[c], side,
            c);
    }
    free(msize);
    free(esize);
    free(cm);
    free(cea);
    free(ceb);
}

static void each_half(int32_t min_nodes, const double *sphi, const int32_t *members, int32_t keep_n,
    const int32_t *ea, const int32_t *eb, int64_t ke, PartFn part, void *ctx) {
    double *tmp = malloc((size_t)keep_n * sizeof(double));
    memcpy(tmp, sphi, (size_t)keep_n * sizeof(double));
    qsort(tmp, (size_t)keep_n, sizeof(double), double_cmp);
    double cut = tmp[keep_n / 2];
    double span = tmp[keep_n - 1] - tmp[0];
    free(tmp);
    int32_t *half = malloc((size_t)keep_n * sizeof(int32_t));
    int32_t *hloc = malloc((size_t)keep_n * sizeof(int32_t));
    uint8_t *lo_side = malloc((size_t)keep_n);
    int32_t *hea = malloc((size_t)(ke ? ke : 1) * sizeof(int32_t));
    int32_t *heb = malloc((size_t)(ke ? ke : 1) * sizeof(int32_t));
    for (int32_t i = 0; i < keep_n; i++)
        lo_side[i] = span > 1e-6 ? sphi[i] < cut : (i < keep_n / 2);
    for (int side = 0; side < 2; side++) {
        int32_t hn = 0;
        for (int32_t i = 0; i < keep_n; i++) {
            if (lo_side[i] == (side == 0)) {
                hloc[i] = hn;
                half[hn++] = members[i];
            } else {
                hloc[i] = -1;
            }
        }
        int64_t hk = 0;
        for (int64_t e = 0; e < ke; e++) {
            if (hloc[ea[e]] >= 0 && hloc[eb[e]] >= 0) {
                hea[hk] = hloc[ea[e]];
                heb[hk] = hloc[eb[e]];
                hk++;
            }
        }
        each_component(min_nodes, half, hn, hea, heb, hk, side, part, ctx);
    }
    free(hea);
    free(heb);
    free(lo_side);
    free(hloc);
    free(half);
}

typedef struct {
    GlobalEmitCtx *ec;
    const Surfel *pts;
    int units_spanned, flat_threads, depth;
} PartCall;

static void flatten_part(const Surfel *pts, const int32_t *members, int32_t keep_n,
    const int32_t *ea, const int32_t *eb, int64_t ke, int stride, int threads, float *pu,
    float *pv) {
    SheetNode *snodes = malloc((size_t)keep_n * sizeof(SheetNode));
    int32_t *ord = malloc((size_t)keep_n * sizeof(int32_t));
    for (int32_t i = 0; i < keep_n; i++) {
        const Surfel *s = &pts[members[i]];
        ord[i] = i;
        snodes[i] = (SheetNode){.x = s->x,
            .y = s->y,
            .z = s->z,
            .nx = (int8_t)lrintf(s->nx * 127.0f),
            .ny = (int8_t)lrintf(s->ny * 127.0f),
            .nz = (int8_t)lrintf(s->nz * 127.0f),
            .prominence = (uint8_t)fminf(255.0f, s->score * 255.0f),
            .width = (uint8_t)stride,
            .coherence = 255};
    }
    int32_t *lmap = malloc((size_t)keep_n * sizeof(int32_t));
    int8_t *fsgn = calloc((size_t)keep_n, 1);
    int64_t pf[2] = {0, keep_n}, ef[2] = {0, ke};
    hx_level_flatten(snodes, ord, pf, 1, ea, eb, ef, lmap, fsgn, pu, pv, 600, 1e-3, 0.01, threads);
    free(lmap);
    free(fsgn);
    free(ord);
    free(snodes);
}

static void emit_part(void *ctx, const int32_t *members, int32_t keep_n, int32_t *ea, int32_t *eb,
    int64_t ke, int side, int32_t index) {
    const PartCall *call = ctx;
    GlobalEmitCtx *ec = call->ec;
    const Surfel *pts = call->pts;
    int depth = call->depth;
    int64_t since = tick();
    double *sphi = malloc((size_t)keep_n * sizeof(double));
    float *pu = calloc((size_t)keep_n, sizeof(float));
    float *pv = calloc((size_t)keep_n, sizeof(float));
    for (int32_t i = 0; i < keep_n; i++)
        sphi[i] = surfel_phi(&pts[members[i]], ec->curl);
    flatten_part(pts, members, keep_n, ea, eb, ke, ec->stride, call->flat_threads, pu, pv);
    lap(T_FLATTEN, since);
    check_flat(pts, members, keep_n, pu, pv);

    since = tick();
    PieceGrid pg;
    int made = piece_grid(ec, pts, members, keep_n, pu, pv, sphi, call->units_spanned, depth, &pg);
    lap(T_GRIDS, since);
    free(pu);
    free(pv);
    if (!made) {
        if (depth < ec->split_depth && keep_n >= 2 * ec->min_nodes) {
            PartCall next = *call;
            next.depth = depth + 1;
            each_half(ec->min_nodes, sphi, members, keep_n, ea, eb, ke, emit_part, &next);
            pthread_mutex_lock(&ec->mu);
            ec->fold_splits++;
            pthread_mutex_unlock(&ec->mu);
        } else {
            pthread_mutex_lock(&ec->mu);
            ec->fold_dropped++;
            pthread_mutex_unlock(&ec->mu);
        }
        free(sphi);
        return;
    }
    pthread_mutex_lock(&ec->mu);
    int my_id = ec->written_count++;
    pthread_mutex_unlock(&ec->mu);
    since = tick();
    json_object *meta = piece_write(ec, &pg, my_id);
    lap(T_WRITE, since);
    check_piece(pts, members, keep_n, depth, my_id);
    pthread_mutex_lock(&ec->mu);
    piece_count(ec, &pg, meta);
    pthread_mutex_unlock(&ec->mu);
    piece_free(&pg);
    free(sphi);
}

static HashItem *sheet_cells(const Surfel *pts, int32_t n, int stride) {
    HashItem *hitems = malloc((size_t)n * sizeof(HashItem));
    for (int32_t i = 0; i < n; i++) {
        int cz = (int)floor(pts[i].z / stride);
        int cy = (int)floor(pts[i].y / stride);
        int cx = (int)floor(pts[i].x / stride);
        hitems[i] = (HashItem){pack_cell(cz, cy, cx), i};
    }
    qsort(hitems, (size_t)n, sizeof(HashItem), hash_item_cmp);
    return hitems;
}

static void emit_global_sheet(GlobalEmitCtx *ec, const Surfel *pts, int32_t n, int32_t global_w,
    int units_spanned, int flat_threads) {
    (void)global_w;
    if (n < ec->min_nodes)
        return;
    int64_t since = tick();
    int stride = ec->stride;
    double *phi = malloc((size_t)n * sizeof(double));
    for (int32_t i = 0; i < n; i++)
        phi[i] = surfel_phi(&pts[i], ec->curl);
    HashItem *hitems = sheet_cells(pts, n, stride);

    int64_t ecap = (int64_t)n * 8 + 4096, ecnt = 0;
    int32_t *ea = malloc((size_t)ecap * sizeof(int32_t));
    int32_t *eb = malloc((size_t)ecap * sizeof(int32_t));
    for (int32_t u = 0; u < n; u++) {
        const Surfel *su = &pts[u];
        int cz0 = (int)floor(su->z / stride);
        int cy0 = (int)floor(su->y / stride);
        int cx0 = (int)floor(su->x / stride);
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int64_t key = pack_cell(cz0 + dz, cy0 + dy, cx0 + dx);
                    int64_t pos = lower_bound_key(hitems, n, key);
                    for (; pos < n && hitems[pos].key == key; pos++) {
                        int32_t v = hitems[pos].idx;
                        if (v <= u)
                            continue;
                        if (fabs(phi[u] - phi[v]) > ec->phi_gate)
                            continue;
                        if (!stitch_edge_near(&su->spot, &pts[v].spot, stride))
                            continue;
                        if (ecnt == ecap) {
                            ecap *= 2;
                            ea = realloc(ea, (size_t)ecap * sizeof(int32_t));
                            eb = realloc(eb, (size_t)ecap * sizeof(int32_t));
                        }
                        ea[ecnt] = u;
                        eb[ecnt] = v;
                        ecnt++;
                    }
                }
            }
        }
    }

    int32_t *cuf = malloc((size_t)n * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++)
        cuf[i] = i;
    for (int64_t e = 0; e < ecnt; e++) {
        int32_t ra = uf_find(cuf, ea[e]), rb = uf_find(cuf, eb[e]);
        if (ra != rb)
            cuf[rb] = ra;
    }
    int r2 = g_bridge;
    int64_t joined = ecnt;
    for (int32_t u = 0; u < n; u++) {
        const Surfel *su = &pts[u];
        int32_t ra = uf_find(cuf, u);
        int cz0 = (int)floor(su->z / stride);
        int cy0 = (int)floor(su->y / stride);
        int cx0 = (int)floor(su->x / stride);
        for (int dz = -r2; dz <= r2; dz++) {
            for (int dy = -r2; dy <= r2; dy++) {
                for (int dx = -r2; dx <= r2; dx++) {
                    if (abs(dz) <= 1 && abs(dy) <= 1 && abs(dx) <= 1)
                        continue;
                    int64_t key = pack_cell(cz0 + dz, cy0 + dy, cx0 + dx);
                    int64_t pos = lower_bound_key(hitems, n, key);
                    for (; pos < n && hitems[pos].key == key; pos++) {
                        int32_t v = hitems[pos].idx;
                        int32_t rb = uf_find(cuf, v);
                        if (rb == ra)
                            continue;
                        if (fabs(phi[u] - phi[v]) > ec->phi_gate)
                            continue;
                        if (!stitch_bridge_near(&su->spot, &pts[v].spot, stride, r2))
                            continue;
                        if (ecnt == ecap) {
                            ecap *= 2;
                            ea = realloc(ea, (size_t)ecap * sizeof(int32_t));
                            eb = realloc(eb, (size_t)ecap * sizeof(int32_t));
                        }
                        ea[ecnt] = u < v ? u : v;
                        eb[ecnt] = u < v ? v : u;
                        ecnt++;
                        cuf[rb] = ra;
                        ra = uf_find(cuf, u);
                    }
                }
            }
        }
    }
    free(hitems);
    free(cuf);
    free(phi);
    lap(T_EDGES, since);
    check_sheet(pts, n, ea + joined, eb + joined, ecnt - joined);

    int32_t *all = malloc((size_t)n * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++)
        all[i] = i;
    PartCall call = {ec, pts, units_spanned, flat_threads, 0};
    each_component(ec->min_nodes, all, n, ea, eb, ecnt, 0, emit_part, &call);
    free(all);
    free(ea);
    free(eb);
}

typedef struct {
    GlobalEmitCtx *ec;
    Surfel **b_buf;
    int32_t *b_len;
    const SheetOrder *sord;
    const int32_t *sh_w;
    const int32_t *sh_units;
    int32_t b0;
} BatchEmitJob;

static void batch_emit_worker(void *ctx, size_t bi) {
    BatchEmitJob *j = ctx;
    int32_t s = j->sord[j->b0 + (int32_t)bi].sheet_id;
    emit_global_sheet(j->ec, j->b_buf[bi], j->b_len[bi], j->sh_w[s], j->sh_units[s], 2);
}

#ifdef STITCH_CUDA
enum { PIECE_NEW, PIECE_MADE, PIECE_FOLDED };

typedef struct {
    int32_t sheet, count, depth, state, id;
    int32_t *path, *members;
    double *sphi;
    int64_t slot;
    PieceGrid grid;
    json_object *meta;
} CudaPiece;

typedef struct {
    CudaPiece **list;
    int64_t count, room;
} PieceList;

typedef struct {
    int32_t from, to;
    int64_t seq;
} Arc;

typedef struct {
    GlobalEmitCtx *ec;
    int32_t sheets;
    Surfel **pts;
    int32_t *len, *spanned;
    int64_t *first;
    int32_t *order, *label, *pairs;
    int64_t *pair_first;
    int32_t **bridge_a, **bridge_b;
    int64_t *bridges;
    Arc **arcs;
    PieceList *made;
    CudaPiece **round;
    float *u, *v;
    int64_t *row_first, *row_at;
    int32_t *row_to;
} CudaBatch;

typedef struct {
    CudaBatch *batch;
    CudaPiece *piece;
    PieceList *out;
} SplitCall;

typedef struct {
    GlobalEmitCtx *ec;
    CudaPiece **list;
    int64_t count;
} WriteJob;

static pthread_t g_writer;
static int g_writing, g_flatten_cpu;

static void list_add(PieceList *l, CudaPiece *p) {
    if (l->count == l->room) {
        l->room = l->room ? l->room * 2 : 64;
        l->list = realloc(l->list, (size_t)l->room * sizeof(CudaPiece *));
    }
    l->list[l->count++] = p;
}

static CudaPiece *piece_new(const Surfel *pts, int curl, int32_t sheet, const int32_t *members,
    int32_t count, int depth, const int32_t *parent, int side, int32_t index) {
    CudaPiece *p = calloc(1, sizeof(*p));
    p->sheet = sheet;
    p->count = count;
    p->depth = depth;
    p->path = malloc((size_t)(1 + 2 * depth) * sizeof(int32_t));
    if (depth) {
        memcpy(p->path, parent, (size_t)(2 * depth - 1) * sizeof(int32_t));
        p->path[2 * depth - 1] = side;
    }
    p->path[2 * depth] = index;
    p->members = malloc((size_t)count * sizeof(int32_t));
    p->sphi = malloc((size_t)count * sizeof(double));
    memcpy(p->members, members, (size_t)count * sizeof(int32_t));
    for (int32_t i = 0; i < count; i++)
        p->sphi[i] = surfel_phi(&pts[members[i]], curl);
    return p;
}

static void piece_drop(CudaPiece *p) {
    if (p->state == PIECE_MADE)
        piece_free(&p->grid);
    free(p->path);
    free(p->members);
    free(p->sphi);
    free(p);
}

static int piece_cmp(const void *p1, const void *p2) {
    const CudaPiece *a = *(CudaPiece *const *)p1, *b = *(CudaPiece *const *)p2;
    if (a->sheet != b->sheet)
        return a->sheet < b->sheet ? -1 : 1;
    int na = 1 + 2 * a->depth, nb = 1 + 2 * b->depth;
    for (int i = 0; i < na && i < nb; i++)
        if (a->path[i] != b->path[i])
            return a->path[i] < b->path[i] ? -1 : 1;
    return (na > nb) - (na < nb);
}

static int size_cmp(const void *p1, const void *p2) {
    const CudaPiece *a = *(CudaPiece *const *)p1, *b = *(CudaPiece *const *)p2;
    if (a->count != b->count)
        return a->count > b->count ? -1 : 1;
    return piece_cmp(p1, p2);
}

static int arc_cmp(const void *p1, const void *p2) {
    const Arc *a = p1, *b = p2;
    if (a->from != b->from)
        return a->from < b->from ? -1 : 1;
    return (a->seq > b->seq) - (a->seq < b->seq);
}

static void cuda_order(void *ctx, size_t si) {
    CudaBatch *b = ctx;
    int32_t n = b->len[si];
    HashItem *h = sheet_cells(b->pts[si], n, b->ec->stride);
    for (int32_t r = 0; r < n; r++)
        b->order[b->first[si] + h[r].idx] = (int32_t)(b->first[si] + r);
    free(h);
}

static void cuda_join(void *ctx, size_t si) {
    CudaBatch *b = ctx;
    const GlobalEmitCtx *ec = b->ec;
    const Surfel *pts = b->pts[si];
    int32_t n = b->len[si], ncomp = 0;
    int64_t base = b->first[si], p0 = b->pair_first[si], p1 = b->pair_first[si + 1], nb = 0;
    int32_t *cuf = malloc((size_t)n * sizeof(int32_t));
    int32_t *la = malloc((size_t)(p1 - p0 + 1) * sizeof(int32_t));
    int32_t *lb = malloc((size_t)(p1 - p0 + 1) * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++)
        cuf[i] = (int32_t)(b->label[base + i] - base);
    for (int64_t p = p0; p < p1;) {
        int32_t u = (int32_t)(b->pairs[2 * p] - base);
        int32_t ra = uf_find(cuf, u);
        for (; p < p1 && b->pairs[2 * p] - base == u; p++) {
            int32_t v = (int32_t)(b->pairs[2 * p + 1] - base);
            int32_t rb = uf_find(cuf, v);
            if (rb == ra)
                continue;
            la[nb] = u < v ? u : v;
            lb[nb] = u < v ? v : u;
            nb++;
            cuf[rb] = ra;
            ra = uf_find(cuf, u);
        }
    }
    check_sheet(pts, n, la, lb, nb);
    Arc *arcs = malloc((size_t)(2 * nb + 1) * sizeof(Arc));
    for (int64_t e = 0; e < nb; e++) {
        arcs[2 * e] = (Arc){(int32_t)(base + la[e]), (int32_t)(base + lb[e]), 2 * e};
        arcs[2 * e + 1] = (Arc){(int32_t)(base + lb[e]), (int32_t)(base + la[e]), 2 * e + 1};
    }
    qsort(arcs, (size_t)(2 * nb), sizeof(Arc), arc_cmp);
    b->bridge_a[si] = la;
    b->bridge_b[si] = lb;
    b->bridges[si] = nb;
    b->arcs[si] = arcs;

    int32_t *comp = malloc((size_t)n * sizeof(int32_t));
    int32_t *root_comp = malloc((size_t)n * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++)
        root_comp[i] = -1;
    for (int32_t i = 0; i < n; i++) {
        int32_t r = uf_find(cuf, i);
        if (root_comp[r] < 0)
            root_comp[r] = ncomp++;
        comp[i] = root_comp[r];
    }
    int64_t *msize = calloc((size_t)ncomp + 1, sizeof(int64_t));
    int64_t *mfill = malloc((size_t)(ncomp + 1) * sizeof(int64_t));
    int32_t *cm = malloc((size_t)n * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++)
        msize[comp[i] + 1]++;
    for (int32_t c = 0; c < ncomp; c++)
        msize[c + 1] += msize[c];
    memcpy(mfill, msize, (size_t)(ncomp + 1) * sizeof(int64_t));
    for (int32_t i = 0; i < n; i++)
        cm[mfill[comp[i]]++] = i;
    for (int32_t c = 0; c < ncomp; c++) {
        int32_t n_c = (int32_t)(msize[c + 1] - msize[c]);
        if (n_c >= ec->min_nodes)
            list_add(&b->made[si],
                piece_new(pts, ec->curl, (int32_t)si, cm + msize[c], n_c, 0, NULL, 0, c));
    }
    free(cm);
    free(mfill);
    free(msize);
    free(root_comp);
    free(comp);
    free(cuf);
}

static void cuda_finish(void *ctx, size_t k) {
    CudaBatch *b = ctx;
    CudaPiece *p = b->round[k];
    const Surfel *pts = b->pts[p->sheet];
    const float *pu = b->u + p->slot, *pv = b->v + p->slot;
    check_flat(pts, p->members, p->count, pu, pv);
    int made = piece_grid(b->ec, pts, p->members, p->count, pu, pv, p->sphi, b->spanned[p->sheet],
        p->depth, &p->grid);
    p->state = made ? PIECE_MADE : PIECE_FOLDED;
}

static void cuda_part(void *ctx, const int32_t *members, int32_t count, int32_t *ea, int32_t *eb,
    int64_t edges, int side, int32_t index) {
    SplitCall *call = ctx;
    const CudaPiece *p = call->piece;
    list_add(call->out,
        piece_new(call->batch->pts[p->sheet], call->batch->ec->curl, p->sheet, members, count,
            p->depth + 1, p->path, side, index));
}

static int64_t piece_edges(
    const CudaBatch *b, const CudaPiece *p, int64_t r0, int32_t **out_a, int32_t **out_b) {
    int32_t n = b->len[p->sheet];
    int64_t base = b->first[p->sheet], ke = 0;
    int64_t room = b->row_first[r0 + p->count] - b->row_first[r0] + b->bridges[p->sheet] + 1;
    int32_t *loc = malloc((size_t)n * sizeof(int32_t));
    int32_t *ea = malloc((size_t)room * sizeof(int32_t));
    int32_t *eb = malloc((size_t)room * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++)
        loc[i] = -1;
    for (int32_t i = 0; i < p->count; i++)
        loc[p->members[i]] = i;
    for (int32_t i = 0; i < p->count; i++) {
        for (int64_t e = b->row_first[r0 + i]; e < b->row_first[r0 + i + 1]; e++) {
            int64_t other = b->row_to[e] - base;
            if (other <= p->members[i] || loc[other] < 0)
                continue;
            ea[ke] = i;
            eb[ke] = loc[other];
            ke++;
        }
    }
    for (int64_t e = 0; e < b->bridges[p->sheet]; e++) {
        int32_t i = loc[b->bridge_a[p->sheet][e]], j = loc[b->bridge_b[p->sheet][e]];
        if (i < 0 || j < 0)
            continue;
        ea[ke] = i;
        eb[ke] = j;
        ke++;
    }
    free(loc);
    *out_a = ea;
    *out_b = eb;
    return ke;
}

static void cuda_split(void *ctx, size_t k) {
    CudaBatch *b = ctx;
    CudaPiece *p = b->round[k];
    int32_t *ea = NULL, *eb = NULL;
    int64_t ke = piece_edges(b, p, b->row_at[k], &ea, &eb);
    SplitCall call = {b, p, &b->made[k]};
    each_half(b->ec->min_nodes, p->sphi, p->members, p->count, ea, eb, ke, cuda_part, &call);
    free(ea);
    free(eb);
}

static void cuda_flatten(void *ctx, size_t k) {
    CudaBatch *b = ctx;
    const CudaPiece *p = b->round[k];
    int32_t *ea = NULL, *eb = NULL;
    int64_t ke = piece_edges(b, p, b->row_at[k], &ea, &eb);
    flatten_part(b->pts[p->sheet], p->members, p->count, ea, eb, ke, b->ec->stride, 1,
        b->u + p->slot, b->v + p->slot);
    free(ea);
    free(eb);
}

static int rows_fetch(CudaBatch *b, StitchCuda *gpu, CudaPiece **list, int64_t count) {
    int64_t total = 0;
    b->row_at = malloc((size_t)count * sizeof(int64_t));
    for (int64_t k = 0; k < count; k++) {
        b->row_at[k] = total;
        total += list[k]->count;
    }
    int32_t *nodes = malloc((size_t)(total ? total : 1) * sizeof(int32_t));
    b->row_first = malloc((size_t)(total + 1) * sizeof(int64_t));
    b->row_to = NULL;
    for (int64_t k = 0; k < count; k++)
        for (int32_t i = 0; i < list[k]->count; i++)
            nodes[b->row_at[k] + i] = (int32_t)(b->first[list[k]->sheet] + list[k]->members[i]);
    int failed = stitch_cuda_emit_rows(gpu, nodes, total, b->row_first, &b->row_to);
    free(nodes);
    b->round = list;
    return failed;
}

static void rows_drop(CudaBatch *b) {
    free(b->row_first);
    free(b->row_to);
    free(b->row_at);
}

static void write_block(void *ctx, size_t k) {
    WriteJob *job = ctx;
    CudaPiece *p = job->list[k];
    p->meta = piece_write(job->ec, &p->grid, p->id);
}

static void *write_run(void *arg) {
    WriteJob *job = arg;
    int64_t since = tick();
    each_block((size_t)job->count, job, write_block);
    pthread_mutex_lock(&job->ec->mu);
    for (int64_t k = 0; k < job->count; k++)
        piece_count(job->ec, &job->list[k]->grid, job->list[k]->meta);
    pthread_mutex_unlock(&job->ec->mu);
    for (int64_t k = 0; k < job->count; k++)
        piece_drop(job->list[k]);
    lap(T_WRITE, since);
    free(job->list);
    free(job);
    return NULL;
}

static void writer_wait(void) {
    if (g_writing)
        pthread_join(g_writer, NULL);
    g_writing = 0;
}

static int cuda_round(
    CudaBatch *b, StitchCuda *gpu, PieceList *round, PieceList *all, int *splits, int *drops) {
    const GlobalEmitCtx *ec = b->ec;
    int64_t pieces = round->count, slots = 0, since = tick();
    qsort(round->list, (size_t)pieces, sizeof(CudaPiece *), size_cmp);
    int64_t *piece_first = malloc((size_t)(pieces + 1) * sizeof(int64_t));
    for (int64_t k = 0; k < pieces; k++) {
        round->list[k]->slot = slots;
        piece_first[k] = slots;
        slots += ((int64_t)round->list[k]->count + STITCH_BLOCK - 1) / STITCH_BLOCK * STITCH_BLOCK;
    }
    piece_first[pieces] = slots;
    int32_t *slot_node = malloc((size_t)slots * sizeof(int32_t));
    b->u = malloc((size_t)slots * sizeof(float));
    b->v = malloc((size_t)slots * sizeof(float));
    memset(slot_node, 0xff, (size_t)slots * sizeof(int32_t));
    for (int64_t k = 0; k < pieces; k++) {
        const CudaPiece *p = round->list[k];
        for (int32_t i = 0; i < p->count; i++)
            slot_node[p->slot + i] = (int32_t)(b->first[p->sheet] + p->members[i]);
    }
    int failed = pieces > INT32_MAX;
    if (!failed && g_flatten_cpu) {
        failed = rows_fetch(b, gpu, round->list, pieces);
        if (!failed)
            each_block((size_t)pieces, b, cuda_flatten);
        rows_drop(b);
    } else if (!failed) {
        failed = stitch_cuda_emit_flatten(
            gpu, slot_node, slots, piece_first, (int32_t)pieces, 600, 1e-3, 0.01, b->u, b->v);
    }
    free(slot_node);
    free(piece_first);
    lap(T_FLATTEN, since);
    PieceList split = {0};
    if (!failed) {
        since = tick();
        b->round = round->list;
        each_block((size_t)pieces, b, cuda_finish);
        lap(T_GRIDS, since);
    }
    for (int64_t k = 0; k < pieces; k++) {
        CudaPiece *p = round->list[k];
        list_add(all, p);
        if (p->state != PIECE_FOLDED)
            continue;
        if (p->depth < ec->split_depth && p->count >= 2 * ec->min_nodes)
            list_add(&split, p);
        else
            (*drops)++;
    }
    round->count = 0;
    free(b->u);
    free(b->v);
    if (!failed && split.count) {
        failed = rows_fetch(b, gpu, split.list, split.count);
        if (!failed) {
            b->made = calloc((size_t)split.count, sizeof(PieceList));
            each_block((size_t)split.count, b, cuda_split);
            for (int64_t k = 0; k < split.count; k++) {
                for (int64_t i = 0; i < b->made[k].count; i++)
                    list_add(round, b->made[k].list[i]);
                free(b->made[k].list);
            }
            free(b->made);
            *splits += (int)split.count;
        }
        rows_drop(b);
    }
    free(split.list);
    return failed;
}

static int emit_batch_cuda(GlobalEmitCtx *ec, StitchCuda *gpu, Surfel **b_buf, const int32_t *b_len,
    int32_t bn, const SheetOrder *sord, const int32_t *sh_units) {
    CudaBatch b = {.ec = ec};
    int64_t nodes = 0, npairs = 0, narcs = 0, since = tick();
    b.pts = malloc((size_t)bn * sizeof(Surfel *));
    b.len = malloc((size_t)bn * sizeof(int32_t));
    b.spanned = malloc((size_t)bn * sizeof(int32_t));
    b.first = malloc((size_t)(bn + 1) * sizeof(int64_t));
    for (int32_t bi = 0; bi < bn; bi++) {
        if (b_len[bi] < ec->min_nodes)
            continue;
        b.pts[b.sheets] = b_buf[bi];
        b.len[b.sheets] = b_len[bi];
        b.spanned[b.sheets] = sh_units[sord[bi].sheet_id];
        b.first[b.sheets++] = nodes;
        nodes += b_len[bi];
    }
    b.first[b.sheets] = nodes;
    PieceList all = {0}, round = {0};
    int splits = 0, drops = 0, failed = !b.sheets ? 0 : nodes > INT32_MAX - 1;
    if (b.sheets && !failed) {
        b.order = malloc((size_t)nodes * sizeof(int32_t));
        b.label = malloc((size_t)nodes * sizeof(int32_t));
        each_block((size_t)b.sheets, &b, cuda_order);
        failed =
            stitch_cuda_emit_open(gpu, nodes, b.sheets, b.first, (const StitchSurfel *const *)b.pts,
                b.order, ec->stride, ec->curl, g_bridge, ec->phi_gate, b.label, &b.pairs, &npairs);
    }
    if (b.sheets && !failed) {
        b.pair_first = malloc((size_t)(b.sheets + 1) * sizeof(int64_t));
        b.bridge_a = calloc((size_t)b.sheets, sizeof(int32_t *));
        b.bridge_b = calloc((size_t)b.sheets, sizeof(int32_t *));
        b.bridges = calloc((size_t)b.sheets, sizeof(int64_t));
        b.arcs = calloc((size_t)b.sheets, sizeof(Arc *));
        b.made = calloc((size_t)b.sheets, sizeof(PieceList));
        for (int64_t p = 0, si = 0; si <= b.sheets; si++) {
            while (p < npairs && b.pairs[2 * p] < b.first[si])
                p++;
            b.pair_first[si] = si < b.sheets ? p : npairs;
        }
        each_block((size_t)b.sheets, &b, cuda_join);
        for (int32_t si = 0; si < b.sheets; si++)
            narcs += 2 * b.bridges[si];
        int32_t *from = malloc((size_t)(narcs + 1) * sizeof(int32_t));
        int32_t *to = malloc((size_t)(narcs + 1) * sizeof(int32_t));
        narcs = 0;
        for (int32_t si = 0; si < b.sheets; si++) {
            for (int64_t e = 0; e < 2 * b.bridges[si]; e++) {
                from[narcs] = b.arcs[si][e].from;
                to[narcs++] = b.arcs[si][e].to;
            }
            for (int64_t i = 0; i < b.made[si].count; i++)
                list_add(&round, b.made[si].list[i]);
            free(b.made[si].list);
            free(b.arcs[si]);
        }
        free(b.made);
        free(b.arcs);
        failed = stitch_cuda_emit_bridges(gpu, from, to, narcs);
        free(from);
        free(to);
        lap(T_EDGES, since);
        while (round.count && !failed)
            failed = cuda_round(&b, gpu, &round, &all, &splits, &drops);
        stitch_cuda_emit_close(gpu);
    }
    for (int64_t k = 0; k < round.count; k++)
        list_add(&all, round.list[k]);
    free(round.list);
    PieceList made = {0};
    for (int64_t k = 0; k < all.count; k++) {
        if (!failed && all.list[k]->state == PIECE_MADE)
            list_add(&made, all.list[k]);
        else
            piece_drop(all.list[k]);
    }
    free(all.list);
    if (!failed) {
        qsort(made.list, (size_t)made.count, sizeof(CudaPiece *), piece_cmp);
        writer_wait();
        for (int64_t k = 0; k < made.count; k++) {
            CudaPiece *p = made.list[k];
            p->id = ec->written_count++;
            check_piece(b.pts[p->sheet], p->members, p->count, p->depth, p->id);
        }
        ec->fold_splits += splits;
        ec->fold_dropped += drops;
    }
    if (made.count) {
        WriteJob *job = malloc(sizeof(*job));
        *job = (WriteJob){ec, made.list, made.count};
        if (pthread_create(&g_writer, NULL, write_run, job))
            write_run(job);
        else
            g_writing = 1;
    } else {
        free(made.list);
    }
    for (int32_t si = 0; si < b.sheets && b.bridge_a; si++) {
        free(b.bridge_a[si]);
        free(b.bridge_b[si]);
    }
    free(b.bridge_a);
    free(b.bridge_b);
    free(b.bridges);
    free(b.pair_first);
    free(b.pairs);
    free(b.order);
    free(b.label);
    free(b.pts);
    free(b.len);
    free(b.spanned);
    free(b.first);
    return failed ? -1 : 0;
}

typedef struct {
    UnitData *units;
    int nunits, fill;
    uint64_t mask;
    const int32_t *gid_sheet, *sh_batch_idx, *gw_sol, *gid_wlocal, *b_cap;
    Surfel **b_buf;
    int32_t bn;
    int64_t *at;
} GatherJob;

static void gather_block(void *ctx, size_t ki) {
    GatherJob *j = ctx;
    int k = (int)ki;
    const UnitData *u = &j->units[k];
    if (!(j->mask & (UINT64_C(1) << k)))
        return;
    double z_cut_lo = k > 0
        ? 0.5 * (fmax(j->units[k - 1].z_lo, u->z_lo) + fmin(j->units[k - 1].z_hi, u->z_hi))
        : -1e18;
    double z_cut_hi = k + 1 < j->nunits
        ? 0.5 * (fmax(u->z_lo, j->units[k + 1].z_lo) + fmin(u->z_hi, j->units[k + 1].z_hi))
        : 1e18;
    int64_t *at = j->at + (int64_t)k * j->bn;
    for (int64_t i = 0; i < u->nsurf; i++) {
        int32_t p = u->piece[i];
        if (p < 0 || p >= u->npieces)
            continue;
        int32_t g = u->gid_base + p;
        int32_t bi = j->sh_batch_idx[j->gid_sheet[g]];
        if (bi < 0)
            continue;
        const Surfel *sf = &u->surfels[i];
        if (sf->z < z_cut_lo || sf->z >= z_cut_hi)
            continue;
        int64_t pos = at[bi]++;
        if (!j->fill || pos >= j->b_cap[bi])
            continue;
        Surfel cp = *sf;
        cp.winding += j->gw_sol[g] - j->gid_wlocal[g];
        j->b_buf[bi][pos] = cp;
    }
}

static void gather_units(GatherJob *j, int32_t *b_len) {
    j->at = calloc((size_t)j->nunits * j->bn, sizeof(int64_t));
    j->fill = 0;
    each_block((size_t)j->nunits, j, gather_block);
    for (int32_t bi = 0; bi < j->bn; bi++) {
        int64_t run = 0;
        for (int k = 0; k < j->nunits; k++) {
            int64_t count = j->at[(int64_t)k * j->bn + bi];
            j->at[(int64_t)k * j->bn + bi] = run;
            run += count;
        }
        b_len[bi] = (int32_t)(run < j->b_cap[bi] ? run : j->b_cap[bi]);
    }
    j->fill = 1;
    each_block((size_t)j->nunits, j, gather_block);
    free(j->at);
}

static int units_fit(const UnitData *units, int nunits) {
    double bytes = 0;
    for (int k = 0; k < nunits; k++) {
        char path[1200];
        int64_t hdr[2] = {0, 0};
        snprintf(path, sizeof(path), "%s/surfels.bin", units[k].dir);
        FILE *f = fopen(path, "rb");
        if (!f)
            return 0;
        size_t got = fread(hdr, sizeof(int64_t), 2, f);
        fclose(f);
        if (got != 2)
            return 0;
        bytes += (double)hdr[0] * (sizeof(Surfel) + sizeof(int32_t));
    }
    return bytes <= 0.45 * (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGE_SIZE);
}

static void load_block(void *ctx, size_t k) { load_unit((UnitData *)ctx + k); }
#endif

static int usage(void) {
    fputs("Usage: unroll-stitch --unit GRAPH:z0:z1 --unit GRAPH:z0:z1 ... --axis AXIS.json "
          "--out "
          "DIR [OPTION...]\n"
          "Input and output\n"
          "  --unit GRAPH:z0:z1        a unit graph folder (surfels.bin) and its planes; 2 to 64 "
          "units\n"
          "  --axis AXIS.json          the scan's umbilicus\n"
          "  --out DIR                 piece folders and record.json\n"
          "  --sample NAME --scan ID   names for the records\n"
          "  --mask MASK.u8 --mask-dims Z,Y,X   papyrus mask at 1/16 scale; surfels outside it are "
          "dropped\n"
          "Scan\n"
          "  --curl +1|-1              hand-set curl (default +1)\n"
          "  --voxel UM                voxel size in micrometres (default 9.362)\n"
          "  --stride N                surfel grid in voxels (default 8)\n"
          "Winding solve\n"
          "  --seam-dist VOX           seam matches within this distance (default 4.5)\n"
          "  --seam-dn VOX             and within this distance along the normal (default 2.2)\n"
          "  --neighbours W            weight of radial-neighbour terms in the turn refine "
          "(default 0)\n"
          "  --neighbours-trust        radial pairs inside one unit component keep the unit's own "
          "step\n"
          "  --refine D                tie each piece's turn with its size divided by D (default "
          "0, off)\n"
          "Pieces\n"
          "  --phi-gate RAD            join surfels only within this spiral-angle difference "
          "(default 1.0)\n"
          "  --bridge N                reach across sheet gaps in grid cells (default 3; 6 gives "
          "larger pieces)\n"
          "  --split-depth N           times a folded part may be halved by angle (default 3)\n"
          "  --min-nodes N             smallest piece in surfels (default 400)\n"
          "  --fair N  --max-hole N    flattening rounds (default 20) and largest filled hole in "
          "cells (default 4800)\n"
          "  --batch-surfels N         surfels per emit batch (default 40000000)\n"
          "Environment\n"
          "  STITCH_CUDA=0 (CPU lane)  STITCH_FLATTEN=gpu  STITCH_TIMES=1  STITCH_CHECK=DIR\n",
        stderr);
    return 2;
}

int main(int argc, char **argv) {
    UnitData units[64];
    int nunits = 0;
    const char *out_dir = NULL, *sample = "unknown", *scan = "unknown";
    int curl = 1, stride = 8, min_nodes = 400, fill_reach = 5, fair_rounds = 20;
    int max_hole_cells = 4800, split_depth = 3;
    double voxel_um = 9.362, phi_gate = 1.0, seam_dist = 4.5, seam_dn = 2.2, refine_div = 0.0;
    double nbr_weight = 0.0;
    int64_t batch_surfels = 40000000;
    const char *axis_path = NULL, *mask_path = NULL, *mask_dims = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--unit") && i + 1 < argc) {
            if (nunits == 64) {
                fprintf(stderr, "stitch: at most 64 units; unroll the scroll in parts\n");
                return 2;
            }
            const char *arg = argv[++i];
            const char *c2 = strrchr(arg, ':');
            const char *c1 = NULL;
            if (c2) {
                for (const char *p = c2 - 1; p > arg; p--)
                    if (*p == ':') {
                        c1 = p;
                        break;
                    }
            }
            if (!c1 || !c2) {
                fprintf(stderr, "stitch: bad --unit DIR:z_lo:z_hi (%s)\n", arg);
                return 2;
            }
            memset(&units[nunits], 0, sizeof(UnitData));
            size_t dlen = (size_t)(c1 - arg);
            if (dlen >= sizeof(units[nunits].dir))
                dlen = sizeof(units[nunits].dir) - 1;
            memcpy(units[nunits].dir, arg, dlen);
            units[nunits].dir[dlen] = 0;
            units[nunits].z_lo = atof(c1 + 1);
            units[nunits].z_hi = atof(c2 + 1);
            nunits++;
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_dir = argv[++i];
        else if (!strcmp(argv[i], "--sample") && i + 1 < argc)
            sample = argv[++i];
        else if (!strcmp(argv[i], "--scan") && i + 1 < argc)
            scan = argv[++i];
        else if (!strcmp(argv[i], "--curl") && i + 1 < argc)
            curl = atoi(argv[++i]) >= 0 ? 1 : -1;
        else if (!strcmp(argv[i], "--stride") && i + 1 < argc)
            stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--voxel") && i + 1 < argc)
            voxel_um = atof(argv[++i]);
        else if (!strcmp(argv[i], "--min-nodes") && i + 1 < argc)
            min_nodes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fair") && i + 1 < argc)
            fair_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-hole") && i + 1 < argc)
            max_hole_cells = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--axis") && i + 1 < argc)
            axis_path = argv[++i];
        else if (!strcmp(argv[i], "--mask") && i + 1 < argc)
            mask_path = argv[++i];
        else if (!strcmp(argv[i], "--mask-dims") && i + 1 < argc)
            mask_dims = argv[++i];
        else if (!strcmp(argv[i], "--phi-gate") && i + 1 < argc)
            phi_gate = atof(argv[++i]);
        else if (!strcmp(argv[i], "--split-depth") && i + 1 < argc)
            split_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seam-dist") && i + 1 < argc)
            seam_dist = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seam-dn") && i + 1 < argc)
            seam_dn = atof(argv[++i]);
        else if (!strcmp(argv[i], "--refine") && i + 1 < argc)
            refine_div = atof(argv[++i]);
        else if (!strcmp(argv[i], "--neighbours") && i + 1 < argc)
            nbr_weight = atof(argv[++i]);
        else if (!strcmp(argv[i], "--neighbours-trust"))
            g_nbr_trust = 1;
        else if (!strcmp(argv[i], "--bridge") && i + 1 < argc)
            g_bridge = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-surfels") && i + 1 < argc)
            batch_surfels = atoll(argv[++i]);
        else {
            fprintf(stderr, "stitch: unknown option or missing value: %s\n", argv[i]);
            return usage();
        }
    }
    if (nunits < 2 || !out_dir || !axis_path)
        return usage();
    HxAxisKnot *axis = NULL;
    int64_t axis_count = 0;
    if (load_axis_json(axis_path, &axis, &axis_count)) {
        fprintf(stderr, "stitch: cannot load axis %s\n", axis_path);
        return 2;
    }
    if (mask_path && (!mask_dims || load_mask(mask_path, mask_dims))) {
        fprintf(stderr, "stitch: cannot load mask %s\n", mask_path);
        return 2;
    }
    double t0 = clock_sec();
#ifdef HERCULEX_CUDA_FLOW
    HxDeviceFlow df = {hx_flow_open, hx_flow_run, hx_flow_close};
    hx_level_device(&df);
#endif
    StitchCuda *gpu = NULL;
    const char *check_path = getenv("STITCH_CHECK");
    if (check_path) {
        char name[4096];
        snprintf(name, sizeof(name), "%s.txt", check_path);
        g_check = fopen(name, "w");
        snprintf(name, sizeof(name), "%s.uv", check_path);
        g_check_uv = fopen(name, "wb");
    }
#ifdef STITCH_CUDA
    const char *lane = getenv("STITCH_CUDA"), *flatten = getenv("STITCH_FLATTEN");
    g_flatten_cpu = !flatten || strcmp(flatten, "gpu");
    if (!lane || strcmp(lane, "0"))
        gpu = stitch_cuda_open();
    if (gpu)
        printf("stitch: CUDA lane on %s, flattening on the %s\n", stitch_cuda_name(gpu),
            g_flatten_cpu ? "CPU" : "GPU");
    else
        printf("stitch: CPU lane\n");
    if (gpu && getenv("STITCH_TIMES"))
        stitch_cuda_timed(gpu);
    if (gpu && units_fit(units, nunits)) {
        g_keep_units = 1;
        each_block((size_t)nunits, units, load_block);
    }
#endif
    int32_t total_gids = 0;
    for (int k = 0; k < nunits; k++) {
        if (!units[k].p_count && load_unit(&units[k]))
            return 1;
        units[k].gid_base = total_gids;
        total_gids += units[k].npieces;
        printf("stitch: unit %d [%0.f..%0.f] loaded %lld surfels, %d pieces\n", k, units[k].z_lo,
            units[k].z_hi, (long long)units[k].nsurf, units[k].npieces);
    }
    fflush(stdout);

    int32_t *gid_unit = malloc((size_t)total_gids * sizeof(int32_t));
    int32_t *gid_local = malloc((size_t)total_gids * sizeof(int32_t));
    int32_t *gid_wlocal = malloc((size_t)total_gids * sizeof(int32_t));
    int32_t *gid_size = malloc((size_t)total_gids * sizeof(int32_t));
    for (int k = 0; k < nunits; k++) {
        for (int32_t p = 0; p < units[k].npieces; p++) {
            int32_t g = units[k].gid_base + p;
            gid_unit[g] = k;
            gid_local[g] = p;
            gid_size[g] = units[k].p_count[p];
            gid_wlocal[g] =
                units[k].p_count[p] ? (int32_t)lrint(units[k].p_wsum[p] / units[k].p_count[p]) : 0;
        }
    }

    int64_t scap = 65536, scnt = 0;
    SeamPair *spairs = malloc((size_t)scap * sizeof(SeamPair));
    for (int k = 0; k + 1 < nunits; k++) {
        UnitData *ua = &units[k], *ub = &units[k + 1];
        double ov_lo = fmax(ua->z_lo, ub->z_lo);
        double ov_hi = fmin(ua->z_hi, ub->z_hi);
        if (ov_hi <= ov_lo) {
            unload_unit_surfels(ua);
            continue;
        }
        if (load_unit_surfels(ua) || load_unit_surfels(ub))
            return 1;
        int64_t since = tick(), rcnt = 0;
        SeamPair *raw = NULL;
#ifdef STITCH_CUDA
        if (gpu) {
            raw =
                seam_raw_cuda(gpu, k, ua, ub, ov_lo, ov_hi, seam_dist, seam_dn, gid_wlocal, &rcnt);
            g_fell += !raw;
        }
#endif
        if (!raw)
            raw = seam_raw(ua, ub, ov_lo, ov_hi, seam_dist, seam_dn, gid_wlocal, &rcnt);
        check_raw("seam", k, raw, rcnt, sizeof(SeamPair));
        qsort(raw, (size_t)rcnt, sizeof(SeamPair), seam_pair_cmp);
        int64_t seam_kept = 0;
        for (int64_t a = 0; a < rcnt;) {
            int64_t b = a;
            int32_t best_cnt = 0, best_k = 0;
            double best_dsum = 0.0;
            while (b < rcnt && raw[b].a == raw[a].a && raw[b].b == raw[a].b) {
                int64_t c = b;
                double dsum = 0.0;
                while (c < rcnt && raw[c].a == raw[b].a && raw[c].b == raw[b].b &&
                    raw[c].k == raw[b].k) {
                    dsum += raw[c].dist_sum;
                    c++;
                }
                if ((int32_t)(c - b) > best_cnt) {
                    best_cnt = (int32_t)(c - b);
                    best_k = raw[b].k;
                    best_dsum = dsum;
                }
                b = c;
            }
            if (best_cnt >= 16 && best_dsum / best_cnt <= 3.5) {
                if (scnt == scap) {
                    scap *= 2;
                    spairs = realloc(spairs, (size_t)scap * sizeof(SeamPair));
                }
                spairs[scnt++] = (SeamPair){raw[a].a, raw[a].b, best_cnt, best_k, best_dsum};
                seam_kept++;
            }
            a = b;
        }
        free(raw);
        lap(T_SEAMS, since);
        unload_unit_surfels(ua);
        printf("stitch: seam %d-%d [%.0f..%.0f]: %lld matched piece pairs\n", k, k + 1, ov_lo,
            ov_hi, (long long)seam_kept);
    }
    if (nunits > 0)
        unload_unit_surfels(&units[nunits - 1]);
    for (int64_t i = 0; g_check && i < scnt; i++)
        fprintf(g_check, "pair %d %d %d %d %a\n", spairs[i].a, spairs[i].b, spairs[i].count,
            spairs[i].k, spairs[i].dist_sum);

    int32_t *comp_of = malloc((size_t)(total_gids ? total_gids : 1) * sizeof(int32_t));
    int32_t ncomps = 0;
    for (int k = 0; k < nunits; k++) {
        int32_t base = units[k].gid_base, np = units[k].npieces;
        WCompItem *witems = malloc((size_t)(np ? np : 1) * sizeof(WCompItem));
        for (int32_t p = 0; p < np; p++)
            witems[p] = (WCompItem){units[k].p_wcomp[p], units[k].p_count[p], p};
        qsort(witems, (size_t)np, sizeof(WCompItem), wcomp_item_cmp);
        for (int32_t s = 0; s < np;) {
            int32_t e = s;
            while (e < np && witems[e].wcomp == witems[s].wcomp)
                e++;
            for (int32_t j = s; j < e; j++)
                comp_of[base + witems[j].p] = witems[s].wcomp >= 0 ? ncomps : -1;
            if (witems[s].wcomp >= 0)
                ncomps++;
            s = e;
        }
        free(witems);
    }
    int64_t *comp_size = calloc((size_t)(ncomps ? ncomps : 1), sizeof(int64_t));
    for (int32_t g = 0; g < total_gids; g++)
        if (comp_of[g] >= 0)
            comp_size[comp_of[g]] += gid_size[g];
    int64_t vcnt = 0;
    SeamPair *votes = malloc((size_t)(scnt ? scnt : 1) * sizeof(SeamPair));
    for (int64_t i = 0; i < scnt; i++) {
        int32_t ca = comp_of[spairs[i].a], cb = comp_of[spairs[i].b];
        if (ca < 0 || cb < 0)
            continue;
        int32_t delta = spairs[i].k + gid_wlocal[spairs[i].a] - gid_wlocal[spairs[i].b];
        votes[vcnt++] = (SeamPair){ca, cb, spairs[i].count, delta, 0.0};
    }
    qsort(votes, (size_t)vcnt, sizeof(SeamPair), seam_pair_cmp);
    int64_t ecap2 = 0;
    SeamPair *cedges = malloc((size_t)(vcnt ? vcnt : 1) * sizeof(SeamPair));
    double *cagree = malloc((size_t)(vcnt ? vcnt : 1) * sizeof(double));
    for (int64_t s = 0; s < vcnt;) {
        int64_t e = s;
        int64_t total = 0, best = 0;
        int32_t best_delta = 0;
        while (e < vcnt && votes[e].a == votes[s].a && votes[e].b == votes[s].b) {
            int64_t f = e, w = 0;
            while (f < vcnt && votes[f].a == votes[e].a && votes[f].b == votes[e].b &&
                votes[f].k == votes[e].k) {
                w += votes[f].count;
                f++;
            }
            total += w;
            if (w > best) {
                best = w;
                best_delta = votes[e].k;
            }
            e = f;
        }
        cedges[ecap2] = (SeamPair){
            votes[s].a, votes[s].b, (int32_t)fmin(2e9, (double)best), best_delta, (double)total};
        cagree[ecap2] = total > 0 ? (double)best / (double)total : 0.0;
        ecap2++;
        s = e;
    }
    free(votes);
    qsort(cedges, (size_t)ecap2, sizeof(SeamPair), seam_pair_cnt_desc);
    int32_t *cuf2 = malloc((size_t)(ncomps ? ncomps : 1) * sizeof(int32_t));
    for (int32_t c = 0; c < ncomps; c++)
        cuf2[c] = c;
    int64_t *tfirst = calloc((size_t)ncomps + 1, sizeof(int64_t));
    int32_t *tto = malloc((size_t)(2 * ecap2 + 1) * sizeof(int32_t));
    int32_t *tdel = malloc((size_t)(2 * ecap2 + 1) * sizeof(int32_t));
    uint8_t *intree = calloc((size_t)(ecap2 ? ecap2 : 1), 1);
    for (int64_t i = 0; i < ecap2; i++) {
        if (cedges[i].count < 64)
            continue;
        int32_t ra = uf_find(cuf2, cedges[i].a), rb = uf_find(cuf2, cedges[i].b);
        if (ra == rb)
            continue;
        cuf2[rb] = ra;
        intree[i] = 1;
        tfirst[cedges[i].a + 1]++;
        tfirst[cedges[i].b + 1]++;
    }
    for (int32_t c = 0; c < ncomps; c++)
        tfirst[c + 1] += tfirst[c];
    int64_t *tfill = malloc((size_t)(ncomps + 1) * sizeof(int64_t));
    memcpy(tfill, tfirst, (size_t)(ncomps + 1) * sizeof(int64_t));
    for (int64_t i = 0; i < ecap2; i++) {
        if (!intree[i])
            continue;
        tto[tfill[cedges[i].a]] = cedges[i].b;
        tdel[tfill[cedges[i].a]++] = cedges[i].k;
        tto[tfill[cedges[i].b]] = cedges[i].a;
        tdel[tfill[cedges[i].b]++] = -cedges[i].k;
    }
    free(tfill);
    int32_t *cshift = calloc((size_t)(ncomps ? ncomps : 1), sizeof(int32_t));
    uint8_t *cseen = calloc((size_t)(ncomps ? ncomps : 1), 1);
    int32_t *cq = malloc((size_t)(ncomps ? ncomps : 1) * sizeof(int32_t));
    int32_t *corder = malloc((size_t)(ncomps ? ncomps : 1) * sizeof(int32_t));
    for (int32_t c = 0; c < ncomps; c++)
        corder[c] = c;
    for (int32_t i = 1; i < ncomps; i++) {
        int32_t c = corder[i], j = i - 1;
        while (j >= 0 && comp_size[corder[j]] < comp_size[c]) {
            corder[j + 1] = corder[j];
            j--;
        }
        corder[j + 1] = c;
    }
    int32_t ntrees = 0, main_tree = -1;
    int32_t *ctree = malloc((size_t)(ncomps ? ncomps : 1) * sizeof(int32_t));
    int64_t main_size = 0;
    for (int32_t oi = 0; oi < ncomps; oi++) {
        int32_t root = corder[oi];
        if (cseen[root])
            continue;
        int32_t qh2 = 0, qt2 = 0;
        cq[qt2++] = root;
        cseen[root] = 1;
        cshift[root] = 0;
        int64_t tsize = 0;
        while (qh2 < qt2) {
            int32_t c = cq[qh2++];
            ctree[c] = ntrees;
            tsize += comp_size[c];
            for (int64_t e = tfirst[c]; e < tfirst[c + 1]; e++) {
                int32_t d = tto[e];
                if (cseen[d])
                    continue;
                cseen[d] = 1;
                cshift[d] = cshift[c] + tdel[e];
                cq[qt2++] = d;
            }
        }
        if (tsize > main_size) {
            main_size = tsize;
            main_tree = ntrees;
        }
        ntrees++;
    }
    int64_t agree_edges = 0, tree_edges = 0;
    double agree_sum = 0.0;
    for (int64_t i = 0; i < ecap2; i++) {
        if (!intree[i])
            continue;
        tree_edges++;
        agree_sum += cagree[i];
        agree_edges += cagree[i] >= 0.8;
    }
    int64_t main_pieces_size = 0, all_size = 0;
    for (int32_t c = 0; c < ncomps; c++) {
        all_size += comp_size[c];
        if (ctree[c] == main_tree)
            main_pieces_size += comp_size[c];
    }
    printf("stitch: %d unit components, %lld seam links, %d trees; main tree holds %.1f%% of "
           "surfels; tree links agree %.3f on average (%lld of %lld above 0.8)\n",
        ncomps, (long long)ecap2, ntrees, 100.0 * main_pieces_size / fmax(1.0, (double)all_size),
        tree_edges ? agree_sum / tree_edges : 0.0, (long long)agree_edges, (long long)tree_edges);
    fflush(stdout);
    int32_t *gw_sol = calloc((size_t)(total_gids ? total_gids : 1), sizeof(int32_t));
    for (int32_t g = 0; g < total_gids; g++)
        gw_sol[g] = gid_wlocal[g] + (comp_of[g] >= 0 ? cshift[comp_of[g]] : 0);
    NbrVote *nvotes = NULL;
    int64_t nnv = 0;
    g_comp_of = comp_of;
    if (nbr_weight > 0) {
        for (int k = 0; k < nunits; k++) {
            NbrVote *vk = NULL;
            int64_t nk = radial_votes(gpu, k, &units[k], gid_wlocal, 3, 26.0, &vk);
            if (nk < 0)
                return 1;
            nvotes = realloc(nvotes, (size_t)(nnv + nk + 1) * sizeof(NbrVote));
            memcpy(nvotes + nnv, vk, (size_t)nk * sizeof(NbrVote));
            nnv += nk;
            free(vk);
        }
        int64_t agree = 0;
        for (int64_t i = 0; i < nnv; i++)
            agree += nvotes[i].k == gid_wlocal[nvotes[i].b] - gid_wlocal[nvotes[i].a];
        printf("stitch: %lld radial neighbour pairs, %.1f%% agree with the units' labels "
               "(%.2fs)\n",
            (long long)nnv, 100.0 * agree / fmax(1.0, (double)nnv), clock_sec() - t0);
        fflush(stdout);
    }
#ifdef STITCH_CUDA
    if (gpu)
        stitch_cuda_units_drop(gpu);
#endif
    for (int64_t i = 0; g_check && i < nnv; i++)
        fprintf(
            g_check, "vote %d %d %d %d\n", nvotes[i].a, nvotes[i].b, nvotes[i].k, nvotes[i].count);
    int64_t solve_since = tick();
    if (refine_div > 0 || nbr_weight > 0) {
        int64_t tcap = scnt + total_gids + nnv + 4096, tcnt = 0;
        int32_t *tp = malloc((size_t)tcap * sizeof(int32_t));
        int32_t *tq = malloc((size_t)tcap * sizeof(int32_t));
        int32_t *tk = malloc((size_t)tcap * sizeof(int32_t));
        int32_t *tup = malloc((size_t)tcap * sizeof(int32_t));
        int32_t *tdn = malloc((size_t)tcap * sizeof(int32_t));
        for (int k = 0; k < nunits; k++) {
            int32_t base = units[k].gid_base, np = units[k].npieces;
            WCompItem *witems = malloc((size_t)(np ? np : 1) * sizeof(WCompItem));
            for (int32_t p = 0; p < np; p++)
                witems[p] = (WCompItem){units[k].p_wcomp[p], units[k].p_count[p], p};
            qsort(witems, (size_t)np, sizeof(WCompItem), wcomp_item_cmp);
            for (int32_t a = 0; a < np;) {
                int32_t b = a;
                while (b < np && witems[b].wcomp == witems[a].wcomp)
                    b++;
                if (refine_div > 0 && witems[a].wcomp >= 0 && witems[a].count >= 48) {
                    int32_t anchor = base + witems[a].p;
                    for (int32_t j = a + 1; j < b; j++) {
                        if (witems[j].count < 48)
                            continue;
                        int32_t g = base + witems[j].p;
                        tp[tcnt] = anchor;
                        tq[tcnt] = g;
                        tk[tcnt] = gid_wlocal[g] - gid_wlocal[anchor];
                        int32_t w =
                            (int32_t)fmin(1000000.0, fmax(1.0, (double)gid_size[g] / refine_div));
                        tup[tcnt] = w;
                        tdn[tcnt] = w;
                        tcnt++;
                    }
                }
                a = b;
            }
            free(witems);
        }
        for (int64_t i = 0; i < scnt; i++) {
            int32_t w = (int32_t)fmin(1000000.0, (double)spairs[i].count);
            tp[tcnt] = spairs[i].a;
            tq[tcnt] = spairs[i].b;
            tk[tcnt] = spairs[i].k;
            tup[tcnt] = w;
            tdn[tcnt] = w;
            tcnt++;
        }
        for (int64_t i = 0; i < nnv; i++) {
            int32_t w = (int32_t)fmin(1000000.0, nvotes[i].count * nbr_weight);
            tp[tcnt] = nvotes[i].a;
            tq[tcnt] = nvotes[i].b;
            tk[tcnt] = nvotes[i].k;
            tup[tcnt] = w;
            tdn[tcnt] = w;
            tcnt++;
        }
        int64_t slog[3] = {0};
        hx_level_minimize(total_gids, tcnt, tp, tq, tk, tup, tdn, gw_sol, 4, slog);
        int64_t nb_ok = 0;
        for (int64_t i = 0; i < nnv; i++)
            nb_ok += gw_sol[nvotes[i].b] - gw_sol[nvotes[i].a] == nvotes[i].k;
        printf("stitch: after the solve %.1f%% of radial neighbour pairs hold\n",
            100.0 * nb_ok / fmax(1.0, (double)nnv));
        printf("stitch: piece refine (anchor size/%.1f, neighbour weight %.2f): %lld terms, "
               "%lld "
               "cuts, %lld moves\n",
            refine_div, nbr_weight, (long long)tcnt, (long long)slog[0], (long long)slog[1]);
        free(tp);
        free(tq);
        free(tk);
        free(tup);
        free(tdn);
    }
    free(comp_size);
    free(cedges);
    free(cagree);
    free(cuf2);
    free(tfirst);
    free(tto);
    free(tdel);
    free(intree);
    free(cshift);
    free(cseen);
    free(cq);
    free(corder);
    free(ctree);

    const char *dump_path = getenv("STITCH_DUMP");
    if (dump_path) {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            for (int32_t g = 0; g < total_gids; g++)
                fprintf(df, "P %d %d %d %d %d %d %d\n", g, gid_unit[g], gid_local[g], gid_wlocal[g],
                    units[gid_unit[g]].p_wcomp[gid_local[g]], gid_size[g], gw_sol[g]);
            for (int64_t i = 0; i < scnt; i++)
                fprintf(df, "S %d %d %d %d %.2f\n", spairs[i].a, spairs[i].b, spairs[i].k,
                    spairs[i].count, spairs[i].dist_sum / fmax(1, spairs[i].count));
            fclose(df);
        }
    }
    qsort(spairs, (size_t)scnt, sizeof(SeamPair), seam_pair_cnt_desc);
    int32_t *guf = malloc((size_t)total_gids * sizeof(int32_t));
    uint64_t *unit_mask = calloc((size_t)total_gids, sizeof(uint64_t));
    for (int32_t g = 0; g < total_gids; g++) {
        guf[g] = g;
        unit_mask[g] = UINT64_C(1) << gid_unit[g];
    }
    int64_t seam_consistent = 0, seam_merged = 0;
    double seam_dist_sum = 0.0;
    double *sh_z0 = malloc((size_t)total_gids * sizeof(double));
    double *sh_z1 = malloc((size_t)total_gids * sizeof(double));
    double *sh_x0 = malloc((size_t)total_gids * sizeof(double));
    double *sh_x1 = malloc((size_t)total_gids * sizeof(double));
    double *sh_y0 = malloc((size_t)total_gids * sizeof(double));
    double *sh_y1 = malloc((size_t)total_gids * sizeof(double));
    for (int k = 0; k < nunits; k++) {
        for (int32_t p = 0; p < units[k].npieces; p++) {
            int32_t g = units[k].gid_base + p;
            sh_x0[g] = units[k].p_x0[p];
            sh_x1[g] = units[k].p_x1[p];
            sh_y0[g] = units[k].p_y0[p];
            sh_y1[g] = units[k].p_y1[p];
            sh_z0[g] = units[k].p_z0[p];
            sh_z1[g] = units[k].p_z1[p];
        }
    }
    for (int64_t i = 0; i < scnt; i++) {
        int32_t a = spairs[i].a, b = spairs[i].b;
        if (gw_sol[b] - gw_sol[a] != spairs[i].k)
            continue;
        seam_consistent++;
        seam_dist_sum += spairs[i].dist_sum / spairs[i].count;
        int32_t ra = uf_find(guf, a), rb = uf_find(guf, b);
        if (ra == rb)
            continue;
        if (unit_mask[ra] & unit_mask[rb]) {
            if (spairs[i].count < 40)
                continue;
            double ox = fmin(sh_x1[ra], sh_x1[rb]) - fmax(sh_x0[ra], sh_x0[rb]);
            double oy = fmin(sh_y1[ra], sh_y1[rb]) - fmax(sh_y0[ra], sh_y0[rb]);
            double minx = fmin(sh_x1[ra] - sh_x0[ra], sh_x1[rb] - sh_x0[rb]);
            double miny = fmin(sh_y1[ra] - sh_y0[ra], sh_y1[rb] - sh_y0[rb]);
            if (ox > 0.35 * minx && oy > 0.35 * miny)
                continue;
        }
        guf[rb] = ra;
        unit_mask[ra] |= unit_mask[rb];
        if (sh_x0[rb] < sh_x0[ra])
            sh_x0[ra] = sh_x0[rb];
        if (sh_x1[rb] > sh_x1[ra])
            sh_x1[ra] = sh_x1[rb];
        if (sh_y0[rb] < sh_y0[ra])
            sh_y0[ra] = sh_y0[rb];
        if (sh_y1[rb] > sh_y1[ra])
            sh_y1[ra] = sh_y1[rb];
        if (sh_z0[rb] < sh_z0[ra])
            sh_z0[ra] = sh_z0[rb];
        if (sh_z1[rb] > sh_z1[ra])
            sh_z1[ra] = sh_z1[rb];
        seam_merged++;
    }
    free(sh_x0);
    free(sh_x1);
    free(sh_y0);
    free(sh_y1);
    free(sh_z0);
    free(sh_z1);
    printf("stitch: exact sync %d global pieces | %lld/%lld (%.1f%%) winding-consistent seams, "
           "%lld multi-unit merges, mean_dist=%.2f vox (%.2fs)\n",
        total_gids, (long long)seam_consistent, (long long)scnt,
        100.0 * seam_consistent / fmax(1.0, (double)scnt), (long long)seam_merged,
        seam_consistent ? seam_dist_sum / seam_consistent : 0.0, clock_sec() - t0);
    fflush(stdout);
    lap(T_SOLVE, solve_since);

    int32_t *root_to_sh = malloc((size_t)total_gids * sizeof(int32_t));
    memset(root_to_sh, -1, (size_t)total_gids * sizeof(int32_t));
    int32_t nsheets = 0;
    for (int32_t g = 0; g < total_gids; g++) {
        int32_t r = uf_find(guf, g);
        if (root_to_sh[r] < 0)
            root_to_sh[r] = nsheets++;
    }
    int64_t *sh_cnt = calloc((size_t)(nsheets ? nsheets : 1), sizeof(int64_t));
    int32_t *sh_w = calloc((size_t)(nsheets ? nsheets : 1), sizeof(int32_t));
    int32_t *sh_units = calloc((size_t)(nsheets ? nsheets : 1), sizeof(int32_t));
    uint64_t *sh_umask = calloc((size_t)(nsheets ? nsheets : 1), sizeof(uint64_t));
    for (int32_t g = 0; g < total_gids; g++) {
        int32_t r = uf_find(guf, g);
        int32_t s = root_to_sh[r];
        sh_cnt[s] += gid_size[g];
        sh_w[s] = gw_sol[r];
        sh_umask[s] = unit_mask[r];
        sh_units[s] = __builtin_popcountll(unit_mask[r]);
    }
    SheetOrder *sord = malloc((size_t)(nsheets ? nsheets : 1) * sizeof(SheetOrder));
    for (int32_t s = 0; s < nsheets; s++)
        sord[s] = (SheetOrder){s, sh_cnt[s]};
    qsort(sord, (size_t)nsheets, sizeof(SheetOrder), sheet_order_cmp);

    make_dirs(out_dir);
    GlobalEmitCtx ec = {.out_dir = out_dir,
        .sample = sample,
        .scan = scan,
        .curl = curl,
        .stride = stride,
        .min_nodes = min_nodes,
        .fill_reach = fill_reach,
        .fair_rounds = fair_rounds,
        .max_hole_cells = max_hole_cells,
        .split_depth = split_depth,
        .voxel_um = voxel_um,
        .phi_gate = phi_gate,
        .axis = axis,
        .axis_count = axis_count,
        .rec_pieces = json_object_new_array()};
    pthread_mutex_init(&ec.mu, NULL);

    int32_t nkeep = 0;
    while (nkeep < nsheets && sord[nkeep].count >= min_nodes)
        nkeep++;
#ifdef STITCH_CUDA
    int32_t *gid_sheet = malloc((size_t)(total_gids ? total_gids : 1) * sizeof(int32_t));
    for (int32_t g = 0; g < total_gids; g++)
        gid_sheet[g] = root_to_sh[uf_find(guf, g)];
#endif
    int32_t *sh_batch_idx = malloc((size_t)(nsheets ? nsheets : 1) * sizeof(int32_t));
    memset(sh_batch_idx, -1, (size_t)(nsheets ? nsheets : 1) * sizeof(int32_t));
    int32_t max_batch = 200000;
    Surfel **b_buf = calloc((size_t)max_batch, sizeof(Surfel *));
    int32_t *b_len = calloc((size_t)max_batch, sizeof(int32_t));
    int32_t *b_cap = calloc((size_t)max_batch, sizeof(int32_t));
    for (int32_t b0 = 0; b0 < nkeep;) {
        int32_t b1 = b0;
        int64_t bsurf = 0;
        uint64_t bumask = 0;
        while (b1 < nkeep &&
            (b1 == b0 || (b1 - b0 < max_batch && bsurf + sord[b1].count <= batch_surfels))) {
            int32_t s = sord[b1].sheet_id;
            int32_t bi = b1 - b0;
            sh_batch_idx[s] = bi;
            b_cap[bi] = (int32_t)(sh_cnt[s] + 16);
            b_len[bi] = 0;
            b_buf[bi] = malloc((size_t)b_cap[bi] * sizeof(Surfel));
            bsurf += sord[b1].count;
            bumask |= sh_umask[s];
            b1++;
        }
        int32_t bn = b1 - b0;
        int64_t gather_since = tick();
        int gathered = 0, emitted = 0;
#ifdef STITCH_CUDA
        if (gpu && g_keep_units) {
            GatherJob gather = {.units = units,
                .nunits = nunits,
                .mask = bumask,
                .gid_sheet = gid_sheet,
                .sh_batch_idx = sh_batch_idx,
                .gw_sol = gw_sol,
                .gid_wlocal = gid_wlocal,
                .b_cap = b_cap,
                .b_buf = b_buf,
                .bn = bn};
            gather_units(&gather, b_len);
            gathered = 1;
        }
#endif
        for (int k = 0; k < nunits && !gathered; k++) {
            if (!(bumask & (UINT64_C(1) << k)))
                continue;
            if (load_unit_surfels(&units[k]))
                return 1;
            double z_cut_lo = k > 0 ? 0.5 *
                    (fmax(units[k - 1].z_lo, units[k].z_lo) +
                        fmin(units[k - 1].z_hi, units[k].z_hi))
                                    : -1e18;
            double z_cut_hi = k + 1 < nunits ? 0.5 *
                    (fmax(units[k].z_lo, units[k + 1].z_lo) +
                        fmin(units[k].z_hi, units[k + 1].z_hi))
                                             : 1e18;
            for (int64_t i = 0; i < units[k].nsurf; i++) {
                int32_t p = units[k].piece[i];
                if (p < 0 || p >= units[k].npieces)
                    continue;
                int32_t g = units[k].gid_base + p;
                int32_t s = root_to_sh[uf_find(guf, g)];
                int32_t bi = sh_batch_idx[s];
                if (bi < 0)
                    continue;
                const Surfel *sf = &units[k].surfels[i];
                if (sf->z < z_cut_lo || sf->z >= z_cut_hi)
                    continue;
                if (b_len[bi] < b_cap[bi]) {
                    Surfel cp = *sf;
                    cp.winding += gw_sol[g] - gid_wlocal[g];
                    b_buf[bi][b_len[bi]++] = cp;
                }
            }
            unload_unit_surfels(&units[k]);
        }
        lap(T_GATHER, gather_since);
#ifdef STITCH_CUDA
        if (gpu) {
            emitted = !emit_batch_cuda(&ec, gpu, b_buf, b_len, bn, sord + b0, sh_units);
            if (!emitted) {
                g_fell++;
                writer_wait();
                printf("stitch: CUDA emit failed for sheets %d..%d, CPU lane\n", b0, b1);
            }
        }
#endif
        int32_t big = emitted ? bn : 0;
        long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        int32_t big_threads = online_cpus < 1 ? 1 : (int32_t)online_cpus;
        while (big < bn && b_len[big] >= 400000) {
            int32_t s = sord[b0 + big].sheet_id;
            emit_global_sheet(&ec, b_buf[big], b_len[big], sh_w[s], sh_units[s], big_threads);
            big++;
        }
        if (big < bn) {
            BatchEmitJob ej = {.ec = &ec,
                .b_buf = b_buf + big,
                .b_len = b_len + big,
                .sord = sord,
                .sh_w = sh_w,
                .sh_units = sh_units,
                .b0 = b0 + big};
            each_block((size_t)(bn - big), &ej, batch_emit_worker);
        }
        printf("stitch: emitted sheets %d..%d (%lld surfels) -> %d pieces (%.2fs)\n", b0, b1,
            (long long)bsurf, ec.written_count, clock_sec() - t0);
        fflush(stdout);
        for (int32_t bi = 0; bi < bn; bi++) {
            int32_t s = sord[b0 + bi].sheet_id;
            free(b_buf[bi]);
            b_buf[bi] = NULL;
            sh_batch_idx[s] = -1;
        }
        b0 = b1;
    }
#ifdef STITCH_CUDA
    writer_wait();
    free(gid_sheet);
#endif
    pthread_mutex_destroy(&ec.mu);
    free(b_buf);
    free(b_len);
    free(b_cap);
    if (renumber_pieces(&ec)) {
        fprintf(stderr, "stitch: could not renumber the pieces in %s\n", out_dir);
        return 1;
    }

    char pathbuf[4096];
    json_object *record = json_object_new_object();
    json_object_object_add(record, "sample", json_object_new_string(sample));
    json_object_object_add(record, "volume", json_object_new_string(scan));
    json_object_object_add(record, "voxel_um", json_object_new_double(voxel_um));
    json_object_object_add(record, "stride", json_object_new_int(stride));
    json_object_object_add(record, "chirality", json_object_new_int(curl));
    json_object_object_add(record, "units", json_object_new_int(nunits));
    json_object_object_add(record, "seam_pairs", json_object_new_int64(scnt));
    json_object_object_add(record, "seam_consistent", json_object_new_int64(seam_consistent));
    json_object_object_add(record, "seam_merges", json_object_new_int64(seam_merged));
    json_object_object_add(record, "area_cm2", json_object_new_double(ec.total_cm2));
    json_object_object_add(record, "area_4cm2", json_object_new_double(ec.area_4cm2));
    json_object_object_add(record, "pieces_4cm2", json_object_new_int(ec.pieces_4));
    json_object_object_add(record, "area_20cm2", json_object_new_double(ec.area_20cm2));
    json_object_object_add(record, "pieces_20cm2", json_object_new_int(ec.pieces_20));
    json_object_object_add(record, "largest_cm2", json_object_new_double(ec.largest_cm2));
    json_object_object_add(record, "isometric_pass", json_object_new_int(ec.iso_pass));
    json_object_object_add(record, "fold_splits", json_object_new_int(ec.fold_splits));
    json_object_object_add(record, "fold_dropped", json_object_new_int(ec.fold_dropped));
    json_object_object_add(record, "piece_count", json_object_new_int(ec.written_count));
    json_object_object_add(record, "seconds", json_object_new_double(clock_sec() - t0));
    json_object_object_add(record, "pieces", ec.rec_pieces);
    snprintf(pathbuf, sizeof(pathbuf), "%s/record.json", out_dir);
    json_object_to_file_ext(pathbuf, record, JSON_C_TO_STRING_PRETTY);
    json_object_put(record);

    printf("stitch DONE: %d global sheets (%.2f cm2 total, %d >=4cm2 totaling %.2f cm2, "
           "%d >=20cm2 totaling %.2f cm2, largest %.2f cm2, isometric %d/%d) in %.2fs\n",
        ec.written_count, ec.total_cm2, ec.pieces_4, ec.area_4cm2, ec.pieces_20, ec.area_20cm2,
        ec.largest_cm2, ec.iso_pass, ec.written_count, clock_sec() - t0);

    if (getenv("STITCH_TIMES")) {
        double spent[T_COUNT];
        for (int s = 0; s < T_COUNT; s++)
            spent[s] = (double)g_spent[s] * 1e-9;
        printf("stitch: seconds load %.2f seams %.2f rays %.2f solve %.2f gather %.2f "
               "edges %.2f flatten %.2f grids %.2f write %.2f (%s)\n",
            spent[T_LOAD], spent[T_SEAMS], spent[T_RAYS], spent[T_SOLVE], spent[T_GATHER],
            spent[T_EDGES], spent[T_FLATTEN], spent[T_GRIDS], spent[T_WRITE],
            gpu ? "CUDA lane, emit stages by the clock" : "CPU lane, emit stages over threads");
    }
#ifdef STITCH_CUDA
    if (gpu && getenv("STITCH_TIMES"))
        stitch_cuda_report(gpu);
    if (gpu)
        printf("stitch: %lld ties and %lld failed GPU stages settled on the CPU\n",
            (long long)g_ties, (long long)g_fell);
    stitch_cuda_close(gpu);
#endif
    if (g_check)
        fclose(g_check);
    if (g_check_uv)
        fclose(g_check_uv);
    free(sh_batch_idx);
    free(sh_umask);
    free(sord);
    free(sh_cnt);
    free(sh_w);
    free(sh_units);
    free(root_to_sh);
    free(guf);
    free(unit_mask);
    free(gw_sol);
    free(comp_of);
    free(nvotes);
    free(spairs);
    free(gid_unit);
    free(gid_local);
    free(gid_wlocal);
    free(gid_size);
    free(axis);
    free((void *)g_mask);
    for (int k = 0; k < nunits; k++) {
        free(units[k].surfels);
        free(units[k].piece);
        free(units[k].p_count);
        free(units[k].p_wcomp);
        free(units[k].p_wsum);
        free(units[k].p_tsum);
        free(units[k].p_x0);
        free(units[k].p_x1);
        free(units[k].p_y0);
        free(units[k].p_y1);
        free(units[k].p_z0);
        free(units[k].p_z1);
    }
    return 0;
}