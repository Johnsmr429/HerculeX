#include "stitch_cuda.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef STITCH_RUN
#define STITCH_RUN(kernel, count, ...) kernel<<<stitch_grid(count), 256>>>(__VA_ARGS__)
#endif
#define STITCH_EACH(i, n)                                                                          \
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n);                  \
        i += (long long)blockDim.x * gridDim.x)

enum { SCAN_BLOCK = 1024, HELD = 512, UNITS = 64 };
enum {
    P_UNITS,
    P_CELLS,
    P_SEAMS,
    P_RAYS,
    P_EDGES,
    P_LABELS,
    P_BRIDGES,
    P_ROWS,
    P_WALK,
    P_TURN,
    P_FLAT,
    P_COUNT
};

typedef struct {
    double x, y, z;
} V;

typedef struct {
    int lo[3], dim[3];
} Box;

typedef struct {
    Box box;
    long long cells, items;
    int *end, *item;
} Cells;

typedef struct {
    double rz, scale, alpha, beta;
    int steps;
} Pace;

typedef struct {
    int id;
    long long n;
    StitchSpot *spot;
    int *wind, *piece;
} Unit;

struct StitchCuda {
    char name[256];
    size_t memory;
    int failed, nheld, nunits;
    void *held[HELD];
    int *scan_sum, *scan_host;
    long long scan_room, rounds, levels, steps;
    int timed;
    double since, spent[P_COUNT];
    Unit units[UNITS];
    size_t unit_bytes;
    long long nodes, bridges;
    int stride;
    StitchSpot *spot;
    int *row, *adj, *node_slot, *bfrom, *bto;
};

static unsigned stitch_grid(long long count) {
    long long blocks = (count + 255) / 256;
    return blocks < 1 ? 1u : blocks > 4194304 ? 4194304u : (unsigned)blocks;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void lap(StitchCuda *g, int phase) {
    if (!g->timed)
        return;
    cudaDeviceSynchronize();
    double t = now();
    if (phase >= 0)
        g->spent[phase] += t - g->since;
    g->since = t;
}

static void *take(StitchCuda *g, size_t bytes) {
    void *p = NULL;
    if (g->failed || g->nheld == HELD || cudaMalloc(&p, bytes ? bytes : 8) != cudaSuccess) {
        g->failed = 1;
        return NULL;
    }
    g->held[g->nheld++] = p;
    return p;
}

static void forget(StitchCuda *g, void *p) {
    for (int i = g->nheld - 1; i >= 0; i--) {
        if (g->held[i] != p)
            continue;
        memmove(g->held + i, g->held + i + 1, (size_t)(g->nheld - i - 1) * sizeof(void *));
        g->nheld--;
        return;
    }
}

static void give(StitchCuda *g, void *p) {
    if (!p)
        return;
    forget(g, p);
    cudaFree(p);
}

static void push(StitchCuda *g, void *device, const void *host, size_t bytes) {
    if (!g->failed && bytes &&
        cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        g->failed = 1;
}

static void pull(StitchCuda *g, void *host, const void *device, size_t bytes) {
    if (!g->failed && bytes &&
        cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        g->failed = 1;
}

static void fill(StitchCuda *g, void *device, int value, size_t bytes) {
    if (!g->failed && bytes && cudaMemset(device, value, bytes) != cudaSuccess)
        g->failed = 1;
}

static int done(StitchCuda *g) {
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess)
        g->failed = 1;
    while (g->nheld > 0)
        cudaFree(g->held[--g->nheld]);
    int failed = g->failed;
    g->failed = 0;
    return failed ? -1 : 0;
}

__global__ void k_scan_sum(const int *a, int *sum, long long n, long long blocks) {
    STITCH_EACH(b, blocks) {
        long long i1 = (b + 1) * SCAN_BLOCK < n ? (b + 1) * SCAN_BLOCK : n;
        int s = 0;
        for (long long i = b * SCAN_BLOCK; i < i1; i++)
            s += a[i];
        sum[b] = s;
    }
}

__global__ void k_scan_put(int *a, const int *base, long long n, long long blocks) {
    STITCH_EACH(b, blocks) {
        long long i1 = (b + 1) * SCAN_BLOCK < n ? (b + 1) * SCAN_BLOCK : n;
        int run = base[b];
        for (long long i = b * SCAN_BLOCK; i < i1; i++) {
            int v = a[i];
            a[i] = run;
            run += v;
        }
    }
}

static long long scan(StitchCuda *g, int *a, long long n) {
    long long blocks = (n + SCAN_BLOCK - 1) / SCAN_BLOCK, run = 0;
    if (!g->failed && blocks > g->scan_room) {
        long long room = blocks > 2 * g->scan_room ? blocks : 2 * g->scan_room;
        room = room < 65536 ? 65536 : room;
        cudaFree(g->scan_sum);
        free(g->scan_host);
        g->scan_sum = NULL;
        g->scan_room = 0;
        g->scan_host = (int *)malloc((size_t)room * sizeof(int));
        if (!g->scan_host ||
            cudaMalloc((void **)&g->scan_sum, (size_t)room * sizeof(int)) != cudaSuccess)
            g->failed = 1;
        else
            g->scan_room = room;
    }
    int *sum = g->scan_sum, *host = g->scan_host;
    if (g->failed)
        return -1;
    STITCH_RUN(k_scan_sum, blocks, a, sum, n, blocks);
    pull(g, host, sum, (size_t)blocks * sizeof(int));
    for (long long b = 0; b < blocks && !g->failed; b++) {
        int v = host[b];
        if (v < 0 || run + v > INT_MAX)
            g->failed = 1;
        host[b] = (int)run;
        run += v;
    }
    push(g, sum, host, (size_t)blocks * sizeof(int));
    if (!g->failed)
        STITCH_RUN(k_scan_put, blocks, a, sum, n, blocks);
    return g->failed ? -1 : run;
}

__device__ int cell_of(float p, double size, int stride) {
    return stride ? (int)floor((double)(p / (float)stride)) : (int)floor((double)p / size);
}

__device__ int cell_at(Box b, int cz, int cy, int cx) {
    int z = cz - b.lo[0], y = cy - b.lo[1], x = cx - b.lo[2];
    if (z < 0 || y < 0 || x < 0 || z >= b.dim[0] || y >= b.dim[1] || x >= b.dim[2])
        return -1;
    return (int)(((long long)z * b.dim[1] + y) * b.dim[2] + x);
}

__global__ void k_cell_box(const StitchSpot *spot, const int *tag, double size, int stride,
    double zlo, double zhi, int *box, long long n) {
    STITCH_EACH(i, n) {
        if ((tag && tag[i] < 0) || spot[i].z < zlo || spot[i].z > zhi)
            continue;
        int c[3] = {cell_of(spot[i].z, size, stride), cell_of(spot[i].y, size, stride),
            cell_of(spot[i].x, size, stride)};
        for (int k = 0; k < 3; k++) {
            if (c[k] < box[k])
                atomicMin(box + k, c[k]);
            if (c[k] > box[3 + k])
                atomicMax(box + 3 + k, c[k]);
        }
    }
}

__global__ void k_cell_count(const StitchSpot *spot, const int *tag, double size, int stride,
    double zlo, double zhi, Box box, int *cell, int *end, long long n) {
    STITCH_EACH(i, n) {
        cell[i] = -1;
        if ((tag && tag[i] < 0) || spot[i].z < zlo || spot[i].z > zhi)
            continue;
        int c = cell_at(box, cell_of(spot[i].z, size, stride), cell_of(spot[i].y, size, stride),
            cell_of(spot[i].x, size, stride));
        cell[i] = c;
        if (c >= 0)
            atomicAdd(end + c, 1);
    }
}

__global__ void k_cell_fill(const int *cell, int *end, int *loose, long long n) {
    STITCH_EACH(i, n) {
        if (cell[i] >= 0)
            loose[atomicAdd(end + cell[i], 1)] = (int)i;
    }
}

__global__ void k_cell_sort(const int *cell, const int *end, const int *loose, const int *order,
    int *item, long long items) {
    STITCH_EACH(s, items) {
        int i = loose[s], c = cell[i], lo = c ? end[c - 1] : 0, hi = end[c], rank = 0;
        int key = order ? order[i] : i;
        for (int j = lo; j < hi; j++)
            rank += (order ? order[loose[j]] : loose[j]) < key;
        item[lo + rank] = i;
    }
}

static int cells_build(StitchCuda *g, Cells *c, const StitchSpot *spot, const int *tag,
    const int *order, long long n, double size, int stride, double zlo, double zhi) {
    int box[6] = {INT_MAX, INT_MAX, INT_MAX, INT_MIN, INT_MIN, INT_MIN};
    int *dbox = (int *)take(g, sizeof(box));
    memset(c, 0, sizeof(*c));
    push(g, dbox, box, sizeof(box));
    if (g->failed)
        return -1;
    STITCH_RUN(k_cell_box, n, spot, tag, size, stride, zlo, zhi, dbox, n);
    pull(g, box, dbox, sizeof(box));
    give(g, dbox);
    if (g->failed)
        return -1;
    double cells = 1;
    for (int k = 0; k < 3; k++) {
        c->box.lo[k] = box[k];
        c->box.dim[k] = box[3 + k] >= box[k] ? box[3 + k] - box[k] + 1 : 0;
        cells *= (double)c->box.dim[k];
    }
    if (cells > 2.0e9 || cells * 4.0 > 0.35 * (double)g->memory) {
        g->failed = 1;
        return -1;
    }
    c->cells = (long long)cells;
    c->end = (int *)take(g, (size_t)(c->cells + 1) * sizeof(int));
    int *cell = (int *)take(g, (size_t)n * sizeof(int));
    fill(g, c->end, 0, (size_t)(c->cells + 1) * sizeof(int));
    if (g->failed)
        return -1;
    STITCH_RUN(k_cell_count, n, spot, tag, size, stride, zlo, zhi, c->box, cell, c->end, n);
    c->items = scan(g, c->end, c->cells + 1);
    if (c->items < 0)
        return -1;
    int *loose = (int *)take(g, (size_t)c->items * sizeof(int));
    c->item = (int *)take(g, (size_t)c->items * sizeof(int));
    if (g->failed)
        return -1;
    STITCH_RUN(k_cell_fill, n, cell, c->end, loose, n);
    STITCH_RUN(k_cell_sort, c->items, cell, c->end, loose, order, c->item, c->items);
    give(g, loose);
    give(g, cell);
    return g->failed ? -1 : 0;
}

__global__ void k_unit_split(const StitchSurfel *src, StitchSpot *spot, int *wind, long long n) {
    STITCH_EACH(i, n) {
        spot[i] = src[i].spot;
        wind[i] = src[i].winding;
    }
}

__global__ void k_unit_gather(const StitchSpot *spot, const int *wind, const int *piece,
    const int *item, StitchSpot *sspot, int *swind, int *spiece, long long items) {
    STITCH_EACH(s, items) {
        int i = item[s];
        sspot[s] = spot[i];
        swind[s] = wind[i];
        spiece[s] = piece[i];
    }
}

__global__ void k_seam(const StitchSpot *spot, const int *piece, const StitchSpot *sspot,
    const int *swind, const int *spiece, const int *item, const int *end, Box box, double size,
    double lo, double hi, float reach, float normal_reach, int *hit, float *dist, int *tie,
    int *keep, long long n) {
    STITCH_EACH(i, n) {
        const StitchSpot *sb = spot + i;
        hit[i] = -1;
        keep[i] = 0;
        if (piece[i] < 0 || sb->z < lo || sb->z > hi)
            continue;
        int cz = cell_of(sb->z, size, 0), cy = cell_of(sb->y, size, 0),
            cx = cell_of(sb->x, size, 0);
        int best = -1, best_p = 0, best_w = 0, tied = 0, odd = 0;
        float best_d = reach;
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int c = cell_at(box, cz + dz, cy + dy, cx + dx);
                    if (c < 0)
                        continue;
                    for (int s = c ? end[c - 1] : 0; s < end[c]; s++) {
                        float d;
                        if (!stitch_seam_near(sspot + s, sb, normal_reach, &d))
                            continue;
                        if (d != d)
                            odd = 1;
                        if (!(d >= best_d)) {
                            best_d = d;
                            best = item[s];
                            best_p = spiece[s];
                            best_w = swind[s];
                            tied = 0;
                        } else if (d == best_d && best >= 0 &&
                            (spiece[s] != best_p || swind[s] != best_w)) {
                            tied = 1;
                        }
                    }
                }
            }
        }
        hit[i] = best;
        dist[i] = best_d;
        tie[i] = tied | odd;
        keep[i] = best >= 0;
    }
}

__global__ void k_seam_pack(const int *hit, const float *dist, const int *tie, const int *keep,
    StitchSeamHit *out, long long n) {
    STITCH_EACH(i, n) {
        if (hit[i] < 0)
            continue;
        StitchSeamHit h = {(int32_t)i, hit[i], tie[i], dist[i]};
        out[keep[i]] = h;
    }
}

__global__ void k_ray(const StitchSpot *spot, const int *piece, const StitchSpot *sspot,
    const int *swind, const int *spiece, const int *item, const int *end, Box box, double size,
    double reach, int every, int *hit, int *tie, int *keep, long long samples) {
    STITCH_EACH(k, samples) {
        long long i = k * every;
        const StitchSpot *sa = spot + i;
        int pa = piece[i], best = -1, best_p = 0, best_w = 0, tied = 0;
        double best_t = 1e9;
        hit[k] = -1;
        tie[k] = 0;
        keep[k] = 0;
        if (pa < 0)
            continue;
        for (double t = 4.0; t <= reach && best < 0; t += 2.0) {
            double qx = sa->x + sa->nx * t, qy = sa->y + sa->ny * t, qz = sa->z + sa->nz * t;
            int cz = (int)floor(qz / size), cy = (int)floor(qy / size), cx = (int)floor(qx / size);
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int c = cell_at(box, cz + dz, cy + dy, cx + dx);
                        if (c < 0)
                            continue;
                        for (int s = c ? end[c - 1] : 0; s < end[c]; s++) {
                            double along;
                            if (spiece[s] == pa ||
                                !stitch_ray_near(sa, sspot + s, qx, qy, qz, &along))
                                continue;
                            double tt = t + along;
                            if (tt < best_t) {
                                best_t = tt;
                                best = item[s];
                                best_p = spiece[s];
                                best_w = swind[s];
                                tied = 0;
                            } else if (tt == best_t &&
                                (spiece[s] != best_p || swind[s] != best_w)) {
                                tied = 1;
                            }
                        }
                    }
                }
            }
        }
        hit[k] = best;
        tie[k] = tied;
        keep[k] = best >= 0;
    }
}

__global__ void k_ray_pack(const int *hit, const int *tie, const int *keep, int every,
    StitchRayHit *out, long long samples) {
    STITCH_EACH(k, samples) {
        if (hit[k] < 0)
            continue;
        StitchRayHit h = {(int32_t)(k * every), hit[k], tie[k]};
        out[keep[k]] = h;
    }
}

__global__ void k_node_split(const StitchSurfel *src, const long long *first, int sheets, int curl,
    StitchSpot *spot, double *phi, int *sheet, long long n) {
    STITCH_EACH(i, n) {
        int lo = 0, hi = sheets;
        while (hi - lo > 1) {
            int mid = lo + (hi - lo) / 2;
            if (first[mid] <= i)
                lo = mid;
            else
                hi = mid;
        }
        spot[i] = src[i].spot;
        phi[i] = stitch_phi(src[i].theta, src[i].winding, curl);
        sheet[i] = lo;
    }
}

__global__ void k_node_gather(const StitchSpot *spot, const double *phi, const int *sheet,
    const int *item, StitchSpot *sspot, double *sphi, int *ssheet, long long n) {
    STITCH_EACH(s, n) {
        int i = item[s];
        sspot[s] = spot[i];
        sphi[s] = phi[i];
        ssheet[s] = sheet[i];
    }
}

__global__ void k_edge(const StitchSpot *spot, const double *phi, const int *sheet,
    const StitchSpot *sspot, const double *sphi, const int *ssheet, const int *item, const int *end,
    Box box, int stride, double phi_gate, int *row, int *adj, long long n) {
    STITCH_EACH(u, n) {
        const StitchSpot *su = spot + u;
        int cz = cell_of(su->z, 0.0, stride), cy = cell_of(su->y, 0.0, stride),
            cx = cell_of(su->x, 0.0, stride);
        int first = adj ? row[u] : 0, lo = first, hi = adj ? row[u + 1] - 1 : 0, count = 0;
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int c = cell_at(box, cz + dz, cy + dy, cx + dx);
                    if (c < 0)
                        continue;
                    for (int s = c ? end[c - 1] : 0; s < end[c]; s++) {
                        int v = item[s];
                        if (ssheet[s] != sheet[u] || v == u)
                            continue;
                        if (fabs(phi[u] - sphi[s]) > phi_gate)
                            continue;
                        if (!(v > u ? stitch_edge_near(su, sspot + s, stride)
                                    : stitch_edge_near(sspot + s, su, stride)))
                            continue;
                        count++;
                        if (!adj)
                            continue;
                        if (v < u)
                            adj[lo++] = v;
                        else
                            adj[hi--] = v;
                    }
                }
            }
        }
        if (!adj) {
            row[u] = count;
            continue;
        }
        for (int a = first + 1; a < lo; a++) {
            int v = adj[a], b = a;
            while (b > first && adj[b - 1] > v) {
                adj[b] = adj[b - 1];
                b--;
            }
            adj[b] = v;
        }
        for (int a = lo, b = row[u + 1] - 1; a < b; a++, b--) {
            int v = adj[a];
            adj[a] = adj[b];
            adj[b] = v;
        }
    }
}

__global__ void k_label_start(int *label, long long n) {
    STITCH_EACH(i, n) { label[i] = (int)i; }
}

__global__ void k_label_hook(
    const int *row, const int *adj, int *label, int *changed, long long n) {
    STITCH_EACH(u, n) {
        int mine = label[u], moved = 0;
        for (int e = row[u]; e < row[u + 1]; e++) {
            int other = label[adj[e]];
            if (other < mine) {
                atomicMin(label + mine, other);
                moved = 1;
            }
        }
        if (moved)
            *changed = 1;
    }
}

__global__ void k_label_jump(int *label, long long n) {
    STITCH_EACH(u, n) {
        int l = label[u];
        while (label[l] != l)
            l = label[l];
        label[u] = l;
    }
}

__global__ void k_label_gather(const int *label, const int *item, int *slabel, long long n) {
    STITCH_EACH(s, n) { slabel[s] = label[item[s]]; }
}

__global__ void k_bridge(const StitchSpot *spot, const double *phi, const int *sheet,
    const int *label, const StitchSpot *sspot, const double *sphi, const int *ssheet,
    const int *slabel, const int *item, const int *end, Box box, int stride, int reach,
    double phi_gate, int *row, int *pairs, long long n) {
    STITCH_EACH(u, n) {
        const StitchSpot *su = spot + u;
        int cz = cell_of(su->z, 0.0, stride), cy = cell_of(su->y, 0.0, stride),
            cx = cell_of(su->x, 0.0, stride);
        int count = 0, at = pairs ? row[u] : 0;
        for (int dz = -reach; dz <= reach; dz++) {
            for (int dy = -reach; dy <= reach; dy++) {
                for (int dx = -reach; dx <= reach; dx++) {
                    if (abs(dz) <= 1 && abs(dy) <= 1 && abs(dx) <= 1)
                        continue;
                    int c = cell_at(box, cz + dz, cy + dy, cx + dx);
                    if (c < 0)
                        continue;
                    for (int s = c ? end[c - 1] : 0; s < end[c]; s++) {
                        if (ssheet[s] != sheet[u] || slabel[s] == label[u])
                            continue;
                        if (fabs(phi[u] - sphi[s]) > phi_gate)
                            continue;
                        if (!stitch_bridge_near(su, sspot + s, stride, reach))
                            continue;
                        if (pairs) {
                            pairs[2 * (at + count)] = (int)u;
                            pairs[2 * (at + count) + 1] = item[s];
                        }
                        count++;
                    }
                }
            }
        }
        if (!pairs)
            row[u] = count;
    }
}

__device__ V v_of(double x, double y, double z) {
    V v = {x, y, z};
    return v;
}
__device__ double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ V cross(V a, V b) {
    return v_of(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ V plus(V a, V b, double k) { return v_of(a.x + k * b.x, a.y + k * b.y, a.z + k * b.z); }
__device__ V unit_of(V a) {
    double l = sqrt(dot(a, a)) + 1e-12;
    return v_of(a.x / l, a.y / l, a.z / l);
}
__device__ V normal_of(const signed char *n, int s) {
    return unit_of(v_of((double)s * n[0], (double)s * n[1], (double)s * n[2]));
}
__device__ V carried(V e, V a, V b) {
    double along = dot(b, e) / (1.0 + dot(a, b) + 1e-9);
    V r = plus(e, plus(a, b, 1.0), -along);
    return unit_of(plus(r, b, -dot(r, b)));
}
__device__ V round_axis(V n) { return v_of(-n.y, n.x, 0.0); }

__device__ void edge_step(
    const float *p, const float *q, V fi, V ni, V fj, V nj, double *du, double *dv) {
    V d = v_of(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
    V a = unit_of(plus(fi, fj, 1.0));
    V c = unit_of(plus(cross(ni, fi), cross(nj, fj), 1.0));
    *du = dot(d, a);
    *dv = dot(d, c);
}

__global__ void k_slot_start(const StitchSpot *spot, const int *snode, const int *pfirst,
    int pieces, int *spiece, int *node_slot, float *pos, signed char *tilt, int *claim,
    signed char *sign, long long slots) {
    STITCH_EACH(s, slots) {
        int x = snode[s], lo = 0, hi = pieces;
        claim[s] = INT_MAX;
        sign[s] = 0;
        spiece[s] = -1;
        if (x < 0)
            continue;
        while (hi - lo > 1) {
            int mid = lo + (hi - lo) / 2;
            if (pfirst[mid] <= s)
                lo = mid;
            else
                hi = mid;
        }
        spiece[s] = lo;
        node_slot[x] = (int)s;
        pos[3 * s] = spot[x].x;
        pos[3 * s + 1] = spot[x].y;
        pos[3 * s + 2] = spot[x].z;
        tilt[3 * s] = (signed char)lrintf(spot[x].nx * 127.0f);
        tilt[3 * s + 1] = (signed char)lrintf(spot[x].ny * 127.0f);
        tilt[3 * s + 2] = (signed char)lrintf(spot[x].nz * 127.0f);
    }
}

__global__ void k_slot_end(const int *snode, int *node_slot, long long slots) {
    STITCH_EACH(s, slots) {
        if (snode[s] >= 0)
            node_slot[snode[s]] = -1;
    }
}

__device__ int bridge_first(const int *bfrom, long long bridges, int x) {
    long long lo = 0, hi = bridges;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (bfrom[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (int)lo;
}

__global__ void k_slot_rows(const int *snode, const int *spiece, const int *node_slot,
    const int *row, const int *adj, const int *bfrom, const int *bto, long long bridges, int *first,
    int *to, long long slots) {
    STITCH_EACH(s, slots) {
        int x = snode[s], count = 0, at = to ? first[s] : 0;
        if (x >= 0) {
            for (int e = row[x]; e < row[x + 1]; e++) {
                int t = node_slot[adj[e]];
                if (t < 0 || spiece[t] != spiece[s])
                    continue;
                if (to)
                    to[at + count] = t;
                count++;
            }
            for (long long b = bridge_first(bfrom, bridges, x); b < bridges && bfrom[b] == x; b++) {
                int t = node_slot[bto[b]];
                if (t < 0 || spiece[t] != spiece[s])
                    continue;
                if (to)
                    to[at + count] = t;
                count++;
            }
        }
        if (!to)
            first[s] = count;
    }
}

__global__ void k_walk_root(const int *pfirst, const signed char *tilt, int *order,
    signed char *sign, V *normal, V *frame, int *parent, long long pieces) {
    STITCH_EACH(p, pieces) {
        int s = pfirst[p];
        order[p] = s;
        sign[s] = 1;
        V n = normal_of(tilt + 3 * s, 1);
        V r = round_axis(n);
        if (dot(r, r) < 1e-4)
            r = v_of(1, 0, 0);
        normal[s] = n;
        frame[s] = unit_of(plus(r, n, -dot(r, n)));
        parent[s] = -1;
    }
}

__global__ void k_walk_claim(const int *order, const int *first, const int *to,
    const signed char *sign, int *claim, int head, long long count) {
    STITCH_EACH(k, count) {
        int r = head + (int)k, i = order[r];
        for (int e = first[i]; e < first[i + 1]; e++) {
            int j = to[e];
            if (!sign[j] && claim[j] > r)
                atomicMin(claim + j, r);
        }
    }
}

__global__ void k_walk_count(const int *order, const int *first, const int *to, const int *claim,
    int *count, int head, long long n) {
    STITCH_EACH(k, n) {
        int r = head + (int)k, i = order[r], c = 0;
        for (int e = first[i]; e < first[i + 1]; e++)
            c += claim[to[e]] == r;
        count[k] = c;
    }
}

__global__ void k_walk_grow(int *order, const int *first, const int *to, const int *claim,
    const int *count, const signed char *tilt, signed char *sign, V *normal, V *frame, int *parent,
    int head, int tail, long long n) {
    STITCH_EACH(k, n) {
        int r = head + (int)k, i = order[r], at = tail + count[k];
        for (int e = first[i]; e < first[i + 1]; e++) {
            int j = to[e];
            if (claim[j] != r)
                continue;
            const signed char *p = tilt + 3 * i, *q = tilt + 3 * j;
            int s =
                ((int)p[0] * q[0] + (int)p[1] * q[1] + (int)p[2] * q[2]) >= 0 ? sign[i] : -sign[i];
            order[at++] = j;
            parent[j] = i;
            sign[j] = (signed char)s;
            normal[j] = normal_of(q, s);
            frame[j] = carried(frame[i], normal[i], normal[j]);
        }
    }
}

__global__ void k_turn_terms(const int *snode, const int *first, const int *to, const V *normal,
    const V *frame, double held, double *b, double *weight, double *angle, long long slots) {
    STITCH_EACH(i, slots) {
        if (snode[i] < 0)
            continue;
        double sum = 0;
        for (int e = first[i]; e < first[i + 1]; e++) {
            int j = to[e];
            if (j < i) {
                V w = carried(frame[j], normal[j], normal[i]);
                sum += atan2(dot(w, cross(normal[i], frame[i])), dot(w, frame[i]));
            } else {
                V w = carried(frame[i], normal[i], normal[j]);
                sum -= atan2(dot(w, cross(normal[j], frame[j])), dot(w, frame[j]));
            }
        }
        V r = round_axis(normal[i]);
        double level = sqrt(dot(r, r));
        weight[i] = held * level * level;
        angle[i] = 0;
        if (!(level < 1e-3)) {
            r = v_of(r.x / level, r.y / level, 0);
            double a = atan2(dot(cross(normal[i], frame[i]), r), dot(frame[i], r));
            if (a > STITCH_PI / 2)
                a -= STITCH_PI;
            else if (a < -STITCH_PI / 2)
                a += STITCH_PI;
            sum += weight[i] * a;
        }
        b[i] = sum;
    }
}

__global__ void k_turn_frames(
    const int *snode, const V *normal, const double *angle, V *frame, long long slots) {
    STITCH_EACH(i, slots) {
        if (snode[i] < 0)
            continue;
        double c = cos(angle[i]), d = sin(angle[i]);
        V f = plus(
            v_of(c * frame[i].x, c * frame[i].y, c * frame[i].z), cross(normal[i], frame[i]), d);
        frame[i] = unit_of(plus(f, normal[i], -dot(f, normal[i])));
    }
}

__global__ void k_flat_walk(const int *order, const int *parent, const float *pos, const V *normal,
    const V *frame, double *u, double *v, long long n) {
    STITCH_EACH(k, n) {
        int j = order[k], i = parent[j];
        if (i < 0) {
            u[j] = 0;
            v[j] = pos[3 * j + 2];
            continue;
        }
        double du, dv;
        edge_step(pos + 3 * i, pos + 3 * j, frame[i], normal[i], frame[j], normal[j], &du, &dv);
        u[j] = u[i] + du;
        v[j] = v[i] + dv;
    }
}

__global__ void k_flat_terms(const int *snode, const int *first, const int *to, const float *pos,
    const V *normal, const V *frame, double *bu, double *bv, long long slots) {
    STITCH_EACH(i, slots) {
        if (snode[i] < 0)
            continue;
        double su = 0, sv = 0, du, dv;
        for (int e = first[i]; e < first[i + 1]; e++) {
            int j = to[e];
            if (j < i) {
                edge_step(
                    pos + 3 * j, pos + 3 * i, frame[j], normal[j], frame[i], normal[i], &du, &dv);
                su += du;
                sv += dv;
            } else {
                edge_step(
                    pos + 3 * i, pos + 3 * j, frame[i], normal[i], frame[j], normal[j], &du, &dv);
                su -= du;
                sv -= dv;
            }
        }
        bu[i] = su;
        bv[i] = sv;
    }
}

__global__ void k_flat_out(
    const double *u, const double *v, float *fu, float *fv, long long slots) {
    STITCH_EACH(i, slots) {
        fu[i] = (float)u[i];
        fv[i] = (float)v[i];
    }
}

__global__ void k_cg_start(const int *snode, const int *bpiece, const int *first, const int *to,
    const double *diag, const double *b, const double *x, double *m, double *r, double *d,
    double *part, long long slots, int systems, long long blocks) {
    STITCH_EACH(blk, blocks) {
        if (bpiece[blk] < 0)
            continue;
        double prz[2] = {0, 0}, psc[2] = {0, 0};
        for (long long i = blk * STITCH_BLOCK; i < (blk + 1) * STITCH_BLOCK; i++) {
            if (snode[i] < 0)
                continue;
            double degree = (double)(first[i + 1] - first[i]) + (diag ? diag[i] : 0.0);
            double mi = degree > 0 ? 1.0 / degree : 0.0;
            m[i] = mi;
            for (int k = 0; k < systems; k++) {
                const double *xk = x + k * slots;
                double applied = degree * xk[i];
                for (int e = first[i]; e < first[i + 1]; e++)
                    applied -= xk[to[e]];
                double bi = b[k * slots + i], ri = bi - applied, di = mi * ri;
                r[k * slots + i] = ri;
                d[k * slots + i] = di;
                prz[k] += ri * di;
                psc[k] += bi * bi * mi;
            }
        }
        for (int k = 0; k < systems; k++) {
            part[k * blocks + blk] = prz[k];
            part[(systems + k) * blocks + blk] = psc[k];
        }
    }
}

__global__ void k_cg_start_piece(const double *part, const int *pblock, Pace *pace, int *active,
    int systems, long long blocks, int steps, double tolerance, long long pieces) {
    STITCH_EACH(p, pieces) {
        for (int k = 0; k < systems; k++) {
            double rz = 0, scale = 1e-12;
            for (int blk = pblock[p]; blk < pblock[p + 1]; blk++) {
                rz += part[k * blocks + blk];
                scale += part[(systems + k) * blocks + blk];
            }
            Pace start = {rz, scale, 0, 0, 0};
            pace[k * pieces + p] = start;
            active[k * pieces + p] = 0 < steps && rz >= tolerance * tolerance * scale;
        }
    }
}

__global__ void k_cg_apply(const int *snode, const int *bpiece, const int *first, const int *to,
    const double *diag, const double *d, double *q, double *part, const int *active,
    long long slots, int systems, long long blocks, long long pieces, long long live) {
    STITCH_EACH(blk, live) {
        int p = bpiece[blk];
        if (p < 0)
            continue;
        for (int k = 0; k < systems; k++) {
            if (!active[k * pieces + p])
                continue;
            const double *dk = d + k * slots;
            double sum = 0;
            for (long long i = blk * STITCH_BLOCK; i < (blk + 1) * STITCH_BLOCK; i++) {
                if (snode[i] < 0)
                    continue;
                double degree = (double)(first[i + 1] - first[i]) + (diag ? diag[i] : 0.0);
                double a = degree * dk[i];
                for (int e = first[i]; e < first[i + 1]; e++)
                    a -= dk[to[e]];
                q[k * slots + i] = a;
                sum += dk[i] * a;
            }
            part[k * blocks + blk] = sum;
        }
    }
}

__global__ void k_cg_step(const double *part, const int *pblock, Pace *pace, int *active,
    int systems, long long blocks, long long pieces, long long live) {
    STITCH_EACH(p, live) {
        for (int k = 0; k < systems; k++) {
            if (!active[k * pieces + p])
                continue;
            double pq = 0;
            for (int blk = pblock[p]; blk < pblock[p + 1]; blk++)
                pq += part[k * blocks + blk];
            if (pq <= 0) {
                active[k * pieces + p] = 0;
                continue;
            }
            pace[k * pieces + p].alpha = pace[k * pieces + p].rz / pq;
        }
    }
}

__global__ void k_cg_move(const int *snode, const int *bpiece, const double *m, const double *d,
    const double *q, double *x, double *r, double *part, const Pace *pace, const int *active,
    long long slots, int systems, long long blocks, long long pieces, long long live) {
    STITCH_EACH(blk, live) {
        int p = bpiece[blk];
        if (p < 0)
            continue;
        for (int k = 0; k < systems; k++) {
            if (!active[k * pieces + p])
                continue;
            double alpha = pace[k * pieces + p].alpha, sum = 0;
            for (long long i = blk * STITCH_BLOCK + k * slots;
                i < (blk + 1) * STITCH_BLOCK + k * slots; i++) {
                if (snode[i - k * slots] < 0)
                    continue;
                x[i] += alpha * d[i];
                r[i] -= alpha * q[i];
                sum += r[i] * r[i] * m[i - k * slots];
            }
            part[k * blocks + blk] = sum;
        }
    }
}

__global__ void k_cg_fresh(const double *part, const int *pblock, Pace *pace, int *active,
    int systems, long long blocks, int steps, double tolerance, long long pieces, long long live) {
    STITCH_EACH(p, live) {
        for (int k = 0; k < systems; k++) {
            if (!active[k * pieces + p])
                continue;
            Pace *now = pace + k * pieces + p;
            double fresh = 0;
            for (int blk = pblock[p]; blk < pblock[p + 1]; blk++)
                fresh += part[k * blocks + blk];
            now->beta = fresh / now->rz;
            now->rz = fresh;
            now->steps++;
            active[k * pieces + p] =
                now->steps < steps && fresh >= tolerance * tolerance * now->scale;
        }
    }
}

__global__ void k_cg_turn(const int *snode, const int *bpiece, const double *m, const double *r,
    double *d, const Pace *pace, const int *active, long long slots, int systems, long long pieces,
    long long live) {
    STITCH_EACH(blk, live) {
        int p = bpiece[blk];
        if (p < 0)
            continue;
        for (int k = 0; k < systems; k++) {
            if (!active[k * pieces + p])
                continue;
            double beta = pace[k * pieces + p].beta;
            for (long long i = blk * STITCH_BLOCK + k * slots;
                i < (blk + 1) * STITCH_BLOCK + k * slots; i++) {
                if (snode[i - k * slots] >= 0)
                    d[i] = m[i - k * slots] * r[i] + beta * d[i];
            }
        }
    }
}

__global__ void k_block_piece(const int *spiece, int *bpiece, long long blocks) {
    STITCH_EACH(blk, blocks) { bpiece[blk] = spiece[blk * STITCH_BLOCK]; }
}

__global__ void k_rows_count(const int *nodes, const int *row, int *first, long long n) {
    STITCH_EACH(k, n) { first[k] = row[nodes[k] + 1] - row[nodes[k]]; }
}

__global__ void k_rows_fill(
    const int *nodes, const int *row, const int *adj, const int *first, int *to, long long n) {
    STITCH_EACH(k, n) {
        for (int e = row[nodes[k]], at = first[k]; e < row[nodes[k] + 1]; e++, at++)
            to[at] = adj[e];
    }
}

typedef struct {
    const int *snode, *bpiece, *first, *to, *pblock;
    const int *pblock_host;
    double *m, *x, *r, *d, *q, *b, *part;
    Pace *pace;
    int *active, *active_host;
    long long slots, blocks, pieces;
} Solver;

static long long last_active(const Solver *s, int systems, long long live) {
    long long last = -1;
    for (int k = 0; k < systems; k++)
        for (long long p = live - 1; p > last; p--)
            if (s->active_host[k * s->pieces + p])
                last = p;
    return last;
}

static void solve(
    StitchCuda *g, const Solver *s, const double *diag, int systems, int steps, double tolerance) {
    if (g->failed)
        return;
    STITCH_RUN(k_cg_start, s->blocks, s->snode, s->bpiece, s->first, s->to, diag, s->b, s->x, s->m,
        s->r, s->d, s->part, s->slots, systems, s->blocks);
    STITCH_RUN(k_cg_start_piece, s->pieces, s->part, s->pblock, s->pace, s->active, systems,
        s->blocks, steps, tolerance, s->pieces);
    pull(g, s->active_host, s->active, (size_t)systems * s->pieces * sizeof(int));
    long long last = g->failed ? -1 : last_active(s, systems, s->pieces);
    while (last >= 0) {
        long long live = last + 1, blocks = s->pblock_host[live];
        g->steps++;
        STITCH_RUN(k_cg_apply, blocks, s->snode, s->bpiece, s->first, s->to, diag, s->d, s->q,
            s->part, s->active, s->slots, systems, s->blocks, s->pieces, blocks);
        STITCH_RUN(k_cg_step, live, s->part, s->pblock, s->pace, s->active, systems, s->blocks,
            s->pieces, live);
        STITCH_RUN(k_cg_move, blocks, s->snode, s->bpiece, s->m, s->d, s->q, s->x, s->r, s->part,
            s->pace, s->active, s->slots, systems, s->blocks, s->pieces, blocks);
        STITCH_RUN(k_cg_fresh, live, s->part, s->pblock, s->pace, s->active, systems, s->blocks,
            steps, tolerance, s->pieces, live);
        STITCH_RUN(k_cg_turn, blocks, s->snode, s->bpiece, s->m, s->r, s->d, s->pace, s->active,
            s->slots, systems, s->pieces, blocks);
        for (int k = 0; k < systems; k++)
            pull(g, s->active_host + k * s->pieces, s->active + k * s->pieces,
                (size_t)live * sizeof(int));
        last = g->failed ? -1 : last_active(s, systems, live);
    }
}

extern "C" StitchCuda *stitch_cuda_open(void) {
    int count = 0;
    size_t free_bytes = 0, total = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1 || cudaSetDevice(0) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, 0) != cudaSuccess ||
        cudaMemGetInfo(&free_bytes, &total) != cudaSuccess)
        return NULL;
    StitchCuda *g = (StitchCuda *)calloc(1, sizeof(StitchCuda));
    if (!g)
        return NULL;
    snprintf(g->name, sizeof(g->name), "%.200s, %.1f of %.1f GB free", prop.name,
        (double)free_bytes / 1073741824.0, (double)total / 1073741824.0);
    g->memory = free_bytes;
    return g;
}

extern "C" const char *stitch_cuda_name(const StitchCuda *g) { return g->name; }

extern "C" void stitch_cuda_timed(StitchCuda *g) { g->timed = 1; }

extern "C" void stitch_cuda_report(const StitchCuda *g) {
    const double *t = g->spent;
    printf("stitch: GPU seconds units %.2f cells %.2f seams %.2f rays %.2f edges %.2f labels "
           "%.2f (%lld rounds) bridges %.2f rows %.2f walk %.2f (%lld levels) turn %.2f flat %.2f "
           "(%lld solver steps)\n",
        t[P_UNITS], t[P_CELLS], t[P_SEAMS], t[P_RAYS], t[P_EDGES], t[P_LABELS], g->rounds,
        t[P_BRIDGES], t[P_ROWS], t[P_WALK], g->levels, t[P_TURN], t[P_FLAT], g->steps);
}

static void unit_free(StitchCuda *g, int at) {
    Unit *u = g->units + at;
    cudaFree(u->spot);
    cudaFree(u->wind);
    cudaFree(u->piece);
    g->unit_bytes -= (size_t)u->n * 32;
    g->units[at] = g->units[--g->nunits];
}

extern "C" void stitch_cuda_units_drop(StitchCuda *g) {
    while (g->nunits > 0)
        unit_free(g, g->nunits - 1);
}

static Unit *unit_find(StitchCuda *g, int id) {
    for (int i = 0; i < g->nunits; i++)
        if (g->units[i].id == id)
            return g->units + i;
    return NULL;
}

extern "C" int stitch_cuda_unit(
    StitchCuda *g, int32_t unit, const StitchSurfel *surfels, const int32_t *piece, int64_t count) {
    if (unit_find(g, unit))
        return 0;
    size_t need = (size_t)count * 32;
    for (;;) {
        int oldest = -1;
        for (int i = 0; i < g->nunits; i++)
            if (g->units[i].id < unit - 1 && (oldest < 0 || g->units[i].id < g->units[oldest].id))
                oldest = i;
        if (oldest < 0 || (g->nunits < UNITS && g->unit_bytes + need <= g->memory * 45 / 100))
            break;
        unit_free(g, oldest);
    }
    if (g->nunits == UNITS)
        return -1;
    Unit u = {unit, count, NULL, NULL, NULL};
    lap(g, -1);
    StitchSurfel *src = (StitchSurfel *)take(g, (size_t)count * sizeof(StitchSurfel));
    u.spot = (StitchSpot *)take(g, (size_t)count * sizeof(StitchSpot));
    u.wind = (int *)take(g, (size_t)count * sizeof(int));
    u.piece = (int *)take(g, (size_t)count * sizeof(int));
    push(g, src, surfels, (size_t)count * sizeof(StitchSurfel));
    push(g, u.piece, piece, (size_t)count * sizeof(int));
    if (!g->failed)
        STITCH_RUN(k_unit_split, count, src, u.spot, u.wind, count);
    lap(g, P_UNITS);
    int kept = !g->failed;
    if (kept) {
        forget(g, u.spot);
        forget(g, u.wind);
        forget(g, u.piece);
    }
    if (done(g)) {
        if (kept) {
            cudaFree(u.spot);
            cudaFree(u.wind);
            cudaFree(u.piece);
        }
        return -1;
    }
    g->units[g->nunits++] = u;
    g->unit_bytes += need;
    return 0;
}

extern "C" int64_t stitch_cuda_seam(StitchCuda *g, int32_t unit_a, int32_t unit_b, double lo,
    double hi, float reach, float normal_reach, StitchSeamHit **hits) {
    Unit *a = unit_find(g, unit_a), *b = unit_find(g, unit_b);
    *hits = NULL;
    if (!a || !b)
        return -1;
    Cells c;
    long long kept = -1;
    StitchSeamHit *host = NULL;
    lap(g, -1);
    if (!cells_build(g, &c, a->spot, a->piece, NULL, a->n, 12.0, 0, lo - 4.0, hi + 4.0)) {
        lap(g, P_CELLS);
        StitchSpot *sspot = (StitchSpot *)take(g, (size_t)c.items * sizeof(StitchSpot));
        int *swind = (int *)take(g, (size_t)c.items * sizeof(int));
        int *spiece = (int *)take(g, (size_t)c.items * sizeof(int));
        int *hit = (int *)take(g, (size_t)b->n * sizeof(int));
        float *dist = (float *)take(g, (size_t)b->n * sizeof(float));
        int *tie = (int *)take(g, (size_t)b->n * sizeof(int));
        int *keep = (int *)take(g, (size_t)(b->n + 1) * sizeof(int));
        fill(g, keep, 0, (size_t)(b->n + 1) * sizeof(int));
        if (!g->failed) {
            STITCH_RUN(k_unit_gather, c.items, a->spot, a->wind, a->piece, c.item, sspot, swind,
                spiece, c.items);
            STITCH_RUN(k_seam, b->n, b->spot, b->piece, sspot, swind, spiece, c.item, c.end, c.box,
                12.0, lo, hi, reach, normal_reach, hit, dist, tie, keep, b->n);
            kept = scan(g, keep, b->n + 1);
        }
        if (kept >= 0) {
            StitchSeamHit *out = (StitchSeamHit *)take(g, (size_t)kept * sizeof(StitchSeamHit));
            host = (StitchSeamHit *)malloc((size_t)(kept ? kept : 1) * sizeof(StitchSeamHit));
            if (!host)
                g->failed = 1;
            if (!g->failed)
                STITCH_RUN(k_seam_pack, b->n, hit, dist, tie, keep, out, b->n);
            pull(g, host, out, (size_t)kept * sizeof(StitchSeamHit));
        }
    }
    lap(g, P_SEAMS);
    if (done(g)) {
        free(host);
        return -1;
    }
    *hits = host;
    return kept;
}

extern "C" int64_t stitch_cuda_rays(
    StitchCuda *g, int32_t unit, int32_t every, double reach, StitchRayHit **hits) {
    Unit *u = unit_find(g, unit);
    *hits = NULL;
    if (!u || every < 1)
        return -1;
    Cells c;
    long long kept = -1, samples = (u->n + every - 1) / every;
    StitchRayHit *host = NULL;
    lap(g, -1);
    if (!cells_build(g, &c, u->spot, u->piece, NULL, u->n, 6.0, 0, -HUGE_VAL, HUGE_VAL)) {
        lap(g, P_CELLS);
        StitchSpot *sspot = (StitchSpot *)take(g, (size_t)c.items * sizeof(StitchSpot));
        int *swind = (int *)take(g, (size_t)c.items * sizeof(int));
        int *spiece = (int *)take(g, (size_t)c.items * sizeof(int));
        int *hit = (int *)take(g, (size_t)samples * sizeof(int));
        int *tie = (int *)take(g, (size_t)samples * sizeof(int));
        int *keep = (int *)take(g, (size_t)(samples + 1) * sizeof(int));
        fill(g, keep, 0, (size_t)(samples + 1) * sizeof(int));
        if (!g->failed) {
            STITCH_RUN(k_unit_gather, c.items, u->spot, u->wind, u->piece, c.item, sspot, swind,
                spiece, c.items);
            STITCH_RUN(k_ray, samples, u->spot, u->piece, sspot, swind, spiece, c.item, c.end,
                c.box, 6.0, reach, every, hit, tie, keep, samples);
            kept = scan(g, keep, samples + 1);
        }
        if (kept >= 0) {
            StitchRayHit *out = (StitchRayHit *)take(g, (size_t)kept * sizeof(StitchRayHit));
            host = (StitchRayHit *)malloc((size_t)(kept ? kept : 1) * sizeof(StitchRayHit));
            if (!host)
                g->failed = 1;
            if (!g->failed)
                STITCH_RUN(k_ray_pack, samples, hit, tie, keep, every, out, samples);
            pull(g, host, out, (size_t)kept * sizeof(StitchRayHit));
        }
    }
    lap(g, P_RAYS);
    if (done(g)) {
        free(host);
        return -1;
    }
    *hits = host;
    return kept;
}

extern "C" void stitch_cuda_emit_close(StitchCuda *g) {
    cudaFree(g->spot);
    cudaFree(g->row);
    cudaFree(g->adj);
    cudaFree(g->node_slot);
    cudaFree(g->bfrom);
    cudaFree(g->bto);
    g->spot = NULL;
    g->row = g->adj = g->node_slot = g->bfrom = g->bto = NULL;
    g->nodes = g->bridges = 0;
}

extern "C" int stitch_cuda_emit_open(StitchCuda *g, int64_t nodes, int32_t sheets,
    const int64_t *sheet_first, const StitchSurfel *const *sheet_nodes, const int32_t *order,
    int32_t stride, int32_t curl, int32_t bridge, double phi_gate, int32_t *label, int32_t **pairs,
    int64_t *pair_count) {
    *pairs = NULL;
    *pair_count = 0;
    if (nodes < 1 || nodes > INT_MAX - 1 || stride < 1 || g->spot)
        return -1;
    int32_t *host = NULL;
    long long edges = -1, found = -1;
    Cells c;
    memset(&c, 0, sizeof(c));
    lap(g, -1);
    StitchSpot *spot = (StitchSpot *)take(g, (size_t)nodes * sizeof(StitchSpot));
    int *row = (int *)take(g, (size_t)(nodes + 1) * sizeof(int));
    int *node_slot = (int *)take(g, (size_t)nodes * sizeof(int));
    double *phi = (double *)take(g, (size_t)nodes * sizeof(double));
    int *sheet = (int *)take(g, (size_t)nodes * sizeof(int));
    int *dorder = (int *)take(g, (size_t)nodes * sizeof(int));
    long long *first = (long long *)take(g, (size_t)(sheets + 1) * sizeof(long long));
    StitchSurfel *src = (StitchSurfel *)take(g, (size_t)nodes * sizeof(StitchSurfel));
    for (int32_t s = 0; s < sheets && !g->failed; s++)
        push(g, src + sheet_first[s], sheet_nodes[s],
            (size_t)(sheet_first[s + 1] - sheet_first[s]) * sizeof(StitchSurfel));
    push(g, first, sheet_first, (size_t)(sheets + 1) * sizeof(long long));
    push(g, dorder, order, (size_t)nodes * sizeof(int));
    fill(g, node_slot, 0xff, (size_t)nodes * sizeof(int));
    fill(g, row, 0, (size_t)(nodes + 1) * sizeof(int));
    if (!g->failed)
        STITCH_RUN(k_node_split, nodes, src, first, sheets, curl, spot, phi, sheet, nodes);
    give(g, src);
    int *adj = NULL;
    if (!g->failed &&
        !cells_build(g, &c, spot, NULL, dorder, nodes, 0.0, stride, -HUGE_VAL, HUGE_VAL)) {
        lap(g, P_CELLS);
        StitchSpot *sspot = (StitchSpot *)take(g, (size_t)nodes * sizeof(StitchSpot));
        double *sphi = (double *)take(g, (size_t)nodes * sizeof(double));
        int *ssheet = (int *)take(g, (size_t)nodes * sizeof(int));
        if (!g->failed) {
            STITCH_RUN(k_node_gather, nodes, spot, phi, sheet, c.item, sspot, sphi, ssheet, nodes);
            STITCH_RUN(k_edge, nodes, spot, phi, sheet, sspot, sphi, ssheet, c.item, c.end, c.box,
                stride, phi_gate, row, (int *)NULL, nodes);
            edges = scan(g, row, nodes + 1);
        }
        if (edges >= 0) {
            adj = (int *)take(g, (size_t)edges * sizeof(int));
            int *dlabel = (int *)take(g, (size_t)nodes * sizeof(int));
            int *slabel = (int *)take(g, (size_t)nodes * sizeof(int));
            int *changed = (int *)take(g, sizeof(int));
            int *count = (int *)take(g, (size_t)(nodes + 1) * sizeof(int));
            if (!g->failed) {
                STITCH_RUN(k_edge, nodes, spot, phi, sheet, sspot, sphi, ssheet, c.item, c.end,
                    c.box, stride, phi_gate, row, adj, nodes);
                STITCH_RUN(k_label_start, nodes, dlabel, nodes);
            }
            lap(g, P_EDGES);
            for (int moved = 1; moved && !g->failed;) {
                g->rounds++;
                fill(g, changed, 0, sizeof(int));
                if (g->failed)
                    break;
                STITCH_RUN(k_label_hook, nodes, row, adj, dlabel, changed, nodes);
                STITCH_RUN(k_label_jump, nodes, dlabel, nodes);
                pull(g, &moved, changed, sizeof(int));
            }
            lap(g, P_LABELS);
            fill(g, count, 0, (size_t)(nodes + 1) * sizeof(int));
            if (!g->failed) {
                STITCH_RUN(k_label_gather, nodes, dlabel, c.item, slabel, nodes);
                STITCH_RUN(k_bridge, nodes, spot, phi, sheet, dlabel, sspot, sphi, ssheet, slabel,
                    c.item, c.end, c.box, stride, bridge, phi_gate, count, (int *)NULL, nodes);
                found = scan(g, count, nodes + 1);
            }
            if (found >= 0) {
                int *dpairs = (int *)take(g, (size_t)found * 2 * sizeof(int));
                host = (int32_t *)malloc((size_t)(found ? found : 1) * 2 * sizeof(int32_t));
                if (!host)
                    g->failed = 1;
                if (!g->failed)
                    STITCH_RUN(k_bridge, nodes, spot, phi, sheet, dlabel, sspot, sphi, ssheet,
                        slabel, c.item, c.end, c.box, stride, bridge, phi_gate, count, dpairs,
                        nodes);
                pull(g, host, dpairs, (size_t)found * 2 * sizeof(int));
                pull(g, label, dlabel, (size_t)nodes * sizeof(int));
            }
        }
    }
    lap(g, P_BRIDGES);
    if (!g->failed && found >= 0) {
        forget(g, spot);
        forget(g, row);
        forget(g, adj);
        forget(g, node_slot);
        g->spot = spot;
        g->row = row;
        g->adj = adj;
        g->node_slot = node_slot;
        g->nodes = nodes;
        g->stride = stride;
    }
    if (done(g) || found < 0) {
        free(host);
        stitch_cuda_emit_close(g);
        return -1;
    }
    *pairs = host;
    *pair_count = found;
    return 0;
}

extern "C" int stitch_cuda_emit_bridges(
    StitchCuda *g, const int32_t *from, const int32_t *to, int64_t count) {
    if (!g->spot || g->bfrom)
        return -1;
    int *bfrom = (int *)take(g, (size_t)count * sizeof(int));
    int *bto = (int *)take(g, (size_t)count * sizeof(int));
    push(g, bfrom, from, (size_t)count * sizeof(int));
    push(g, bto, to, (size_t)count * sizeof(int));
    if (!g->failed) {
        forget(g, bfrom);
        forget(g, bto);
        g->bfrom = bfrom;
        g->bto = bto;
        g->bridges = count;
    }
    return done(g);
}

extern "C" int stitch_cuda_emit_rows(
    StitchCuda *g, const int32_t *nodes, int64_t count, int64_t *first, int32_t **to) {
    *to = NULL;
    if (!g->spot || count < 1)
        return -1;
    int32_t *host = NULL;
    int *counts = (int *)malloc((size_t)(count + 1) * sizeof(int));
    int *dnodes = (int *)take(g, (size_t)count * sizeof(int));
    int *dfirst = (int *)take(g, (size_t)(count + 1) * sizeof(int));
    long long total = -1;
    push(g, dnodes, nodes, (size_t)count * sizeof(int));
    fill(g, dfirst, 0, (size_t)(count + 1) * sizeof(int));
    if (!counts)
        g->failed = 1;
    if (!g->failed) {
        STITCH_RUN(k_rows_count, count, dnodes, g->row, dfirst, count);
        total = scan(g, dfirst, count + 1);
    }
    if (total >= 0) {
        int *dto = (int *)take(g, (size_t)total * sizeof(int));
        host = (int32_t *)malloc((size_t)(total ? total : 1) * sizeof(int32_t));
        if (!host)
            g->failed = 1;
        if (!g->failed)
            STITCH_RUN(k_rows_fill, count, dnodes, g->row, g->adj, dfirst, dto, count);
        pull(g, host, dto, (size_t)total * sizeof(int));
        pull(g, counts, dfirst, (size_t)(count + 1) * sizeof(int));
    }
    if (done(g) || total < 0) {
        free(host);
        free(counts);
        return -1;
    }
    for (int64_t k = 0; k <= count; k++)
        first[k] = counts[k];
    free(counts);
    *to = host;
    return 0;
}

extern "C" int stitch_cuda_emit_flatten(StitchCuda *g, const int32_t *slot_node, int64_t slots,
    const int64_t *piece_first, int32_t pieces, int32_t steps, double tolerance, double held,
    float *u, float *v) {
    if (!g->spot || !g->bfrom || slots < 1 || pieces < 1 || slots > INT_MAX - 1 ||
        slots % STITCH_BLOCK)
        return -1;
    long long blocks = slots / STITCH_BLOCK, links = -1, real = 0;
    int *pfirst_host = (int *)malloc((size_t)(pieces + 1) * sizeof(int));
    int *pblock_host = (int *)malloc((size_t)(pieces + 1) * sizeof(int));
    int *active_host = (int *)malloc((size_t)pieces * 2 * sizeof(int));
    int *level = NULL;
    long long levels = 0, room = 0;
    if (!pfirst_host || !pblock_host || !active_host)
        g->failed = 1;
    for (int32_t p = 0; p <= pieces && !g->failed; p++) {
        pfirst_host[p] = (int)piece_first[p];
        pblock_host[p] = (int)(piece_first[p] / STITCH_BLOCK);
    }
    int *snode = (int *)take(g, (size_t)slots * sizeof(int));
    int *spiece = (int *)take(g, (size_t)slots * sizeof(int));
    int *pfirst = (int *)take(g, (size_t)(pieces + 1) * sizeof(int));
    int *pblock = (int *)take(g, (size_t)(pieces + 1) * sizeof(int));
    int *bpiece = (int *)take(g, (size_t)blocks * sizeof(int));
    int *first = (int *)take(g, (size_t)(slots + 1) * sizeof(int));
    float *pos = (float *)take(g, (size_t)slots * 3 * sizeof(float));
    signed char *tilt = (signed char *)take(g, (size_t)slots * 3);
    signed char *sign = (signed char *)take(g, (size_t)slots);
    int *claim = (int *)take(g, (size_t)slots * sizeof(int));
    int *to = NULL;
    lap(g, -1);
    push(g, snode, slot_node, (size_t)slots * sizeof(int));
    push(g, pfirst, pfirst_host, (size_t)(pieces + 1) * sizeof(int));
    push(g, pblock, pblock_host, (size_t)(pieces + 1) * sizeof(int));
    fill(g, first, 0, (size_t)(slots + 1) * sizeof(int));
    if (!g->failed) {
        STITCH_RUN(k_slot_start, slots, g->spot, snode, pfirst, pieces, spiece, g->node_slot, pos,
            tilt, claim, sign, slots);
        STITCH_RUN(k_block_piece, blocks, spiece, bpiece, blocks);
        STITCH_RUN(k_slot_rows, slots, snode, spiece, g->node_slot, g->row, g->adj, g->bfrom,
            g->bto, g->bridges, first, (int *)NULL, slots);
        links = scan(g, first, slots + 1);
    }
    if (links >= 0) {
        to = (int *)take(g, (size_t)links * sizeof(int));
        if (!g->failed)
            STITCH_RUN(k_slot_rows, slots, snode, spiece, g->node_slot, g->row, g->adj, g->bfrom,
                g->bto, g->bridges, first, to, slots);
    }
    if (!g->failed)
        STITCH_RUN(k_slot_end, slots, snode, g->node_slot, slots);
    for (int64_t s = 0; s < slots; s++)
        real += slot_node[s] >= 0;
    V *normal = (V *)take(g, (size_t)slots * sizeof(V));
    V *frame = (V *)take(g, (size_t)slots * sizeof(V));
    int *parent = (int *)take(g, (size_t)slots * sizeof(int));
    int *order = (int *)take(g, (size_t)slots * sizeof(int));
    int *count = (int *)take(g, (size_t)(slots + 1) * sizeof(int));
    long long head = 0, tail = pieces;
    lap(g, P_ROWS);
    if (!g->failed && links >= 0)
        STITCH_RUN(k_walk_root, pieces, pfirst, tilt, order, sign, normal, frame, parent, pieces);
    while (!g->failed && links >= 0 && head < tail) {
        long long n = tail - head;
        if (levels == room) {
            room = room ? room * 2 : 1024;
            int *wider = (int *)realloc(level, (size_t)room * sizeof(int));
            if (!wider) {
                g->failed = 1;
                break;
            }
            level = wider;
        }
        level[levels++] = (int)head;
        g->levels++;
        STITCH_RUN(k_walk_claim, n, order, first, to, sign, claim, (int)head, n);
        STITCH_RUN(k_walk_count, n, order, first, to, claim, count, (int)head, n);
        fill(g, count + n, 0, sizeof(int));
        long long grown = scan(g, count, n + 1);
        if (grown < 0 || tail + grown > real) {
            g->failed = 1;
            break;
        }
        if (grown)
            STITCH_RUN(k_walk_grow, n, order, first, to, claim, count, tilt, sign, normal, frame,
                parent, (int)head, (int)tail, n);
        head = tail;
        tail += grown;
    }
    if (tail != real)
        g->failed = 1;
    lap(g, P_WALK);
    give(g, claim);
    give(g, count);
    give(g, sign);
    give(g, tilt);
    Solver s;
    memset(&s, 0, sizeof(s));
    s.snode = snode;
    s.bpiece = bpiece;
    s.first = first;
    s.to = to;
    s.pblock = pblock;
    s.pblock_host = pblock_host;
    s.active_host = active_host;
    s.slots = slots;
    s.blocks = blocks;
    s.pieces = pieces;
    s.m = (double *)take(g, (size_t)slots * sizeof(double));
    s.x = (double *)take(g, (size_t)slots * 2 * sizeof(double));
    s.r = (double *)take(g, (size_t)slots * 2 * sizeof(double));
    s.d = (double *)take(g, (size_t)slots * 2 * sizeof(double));
    s.q = (double *)take(g, (size_t)slots * 2 * sizeof(double));
    s.b = (double *)take(g, (size_t)slots * 2 * sizeof(double));
    s.part = (double *)take(g, (size_t)blocks * 4 * sizeof(double));
    s.pace = (Pace *)take(g, (size_t)pieces * 2 * sizeof(Pace));
    s.active = (int *)take(g, (size_t)pieces * 2 * sizeof(int));
    double *weight = (double *)take(g, (size_t)slots * sizeof(double));
    fill(g, s.x, 0, (size_t)slots * 2 * sizeof(double));
    fill(g, s.b, 0, (size_t)slots * 2 * sizeof(double));
    fill(g, weight, 0, (size_t)slots * sizeof(double));
    for (int round = 0; round < 2 && !g->failed; round++) {
        STITCH_RUN(
            k_turn_terms, slots, snode, first, to, normal, frame, held, s.b, weight, s.x, slots);
        solve(g, &s, weight, 1, steps, tolerance);
        if (!g->failed)
            STITCH_RUN(k_turn_frames, slots, snode, normal, s.x, frame, slots);
    }
    lap(g, P_TURN);
    for (long long l = 0; l < levels && !g->failed; l++) {
        long long from = level[l], n = (l + 1 < levels ? level[l + 1] : tail) - from;
        STITCH_RUN(k_flat_walk, n, order + from, parent, pos, normal, frame, s.x, s.x + slots, n);
    }
    if (!g->failed) {
        STITCH_RUN(
            k_flat_terms, slots, snode, first, to, pos, normal, frame, s.b, s.b + slots, slots);
        solve(g, &s, NULL, 2, steps, tolerance);
    }
    float *fu = (float *)take(g, (size_t)slots * sizeof(float));
    float *fv = (float *)take(g, (size_t)slots * sizeof(float));
    if (!g->failed)
        STITCH_RUN(k_flat_out, slots, s.x, s.x + slots, fu, fv, slots);
    pull(g, u, fu, (size_t)slots * sizeof(float));
    pull(g, v, fv, (size_t)slots * sizeof(float));
    lap(g, P_FLAT);
    free(pfirst_host);
    free(pblock_host);
    free(active_host);
    free(level);
    return done(g);
}

extern "C" void stitch_cuda_close(StitchCuda *g) {
    if (!g)
        return;
    stitch_cuda_units_drop(g);
    stitch_cuda_emit_close(g);
    cudaFree(g->scan_sum);
    free(g->scan_host);
    free(g);
}
