// Transport the local frame along the sheet to avoid folding at horizontal sections.

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float x, y, z; int8_t nx, ny, nz; uint8_t prominence, width, coherence; uint16_t pad; } SheetNode;
typedef struct { double x, y, z; } V;

static inline V v_of(double x, double y, double z) { V v = {x, y, z}; return v; }
static inline double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V cross(V a, V b) { return v_of(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline V plus(V a, V b, double k) { return v_of(a.x + k * b.x, a.y + k * b.y, a.z + k * b.z); }
static inline V unit_of(V a) { double l = sqrt(dot(a, a)) + 1e-12; return v_of(a.x / l, a.y / l, a.z / l); }

static inline V normal_of(const SheetNode *n, int s) { return unit_of(v_of((double)s * n->nx, (double)s * n->ny, (double)s * n->nz)); }

static inline V carried(V e, V a, V b) {
    double along = dot(b, e) / (1.0 + dot(a, b) + 1e-9);
    V r = plus(e, plus(a, b, 1.0), -along);
    return unit_of(plus(r, b, -dot(r, b)));
}

static inline V round_axis(V n) { return v_of(-n.y, n.x, 0.0); }

typedef struct {
    int64_t n;
    const int64_t *first;
    const int32_t *to;
} Net;

static void solve(const Net *g, const double *diag, const double *b, double *x, int steps, double tolerance) {
    int64_t n = g->n;
    double *r = malloc(sizeof(double) * (size_t)(n + 1)), *d = malloc(sizeof(double) * (size_t)(n + 1)), *q = malloc(sizeof(double) * (size_t)(n + 1)), *m = malloc(sizeof(double) * (size_t)(n + 1));
    double rz = 0, scale = 1e-12;
    for (int64_t i = 0; i < n; i++) {
        double degree = (double)(g->first[i + 1] - g->first[i]) + (diag ? diag[i] : 0.0);
        m[i] = degree > 0 ? 1.0 / degree : 0.0;
        double applied = degree * x[i];
        for (int64_t e = g->first[i]; e < g->first[i + 1]; e++) applied -= x[g->to[e]];
        r[i] = b[i] - applied;
        d[i] = m[i] * r[i];
        rz += r[i] * d[i];
        scale += b[i] * b[i] * m[i];
    }
    for (int step = 0; step < steps && rz >= tolerance * tolerance * scale; step++) {
        double pq = 0;
        for (int64_t i = 0; i < n; i++) {
            double degree = (double)(g->first[i + 1] - g->first[i]) + (diag ? diag[i] : 0.0);
            double a = degree * d[i];
            for (int64_t e = g->first[i]; e < g->first[i + 1]; e++) a -= d[g->to[e]];
            q[i] = a;
            pq += d[i] * a;
        }
        if (pq <= 0) break;
        double alpha = rz / pq, fresh = 0;
        for (int64_t i = 0; i < n; i++) {
            x[i] += alpha * d[i];
            r[i] -= alpha * q[i];
            fresh += r[i] * r[i] * m[i];
        }
        double beta = fresh / rz;
        for (int64_t i = 0; i < n; i++) d[i] = m[i] * r[i] + beta * d[i];
        rz = fresh;
    }
    free(r); free(d); free(q); free(m);
}

static void flatten_piece(const SheetNode *nodes, const int32_t *members, int64_t n, const int32_t *ea, const int32_t *eb, int64_t m, int32_t *local, int8_t *sign, float *out_u, float *out_v,
                          int steps, double tolerance, double held) {
    for (int64_t i = 0; i < n; i++) local[members[i]] = (int32_t)i;
    int64_t *first = calloc((size_t)n + 2, sizeof(int64_t));
    int32_t *to = malloc(sizeof(int32_t) * (2 * (size_t)m + 1));
    for (int64_t e = 0; e < m; e++) { first[local[ea[e]] + 1]++; first[local[eb[e]] + 1]++; }
    for (int64_t i = 0; i < n; i++) first[i + 1] += first[i];
    int64_t *cursor = malloc(sizeof(int64_t) * ((size_t)n + 1));
    memcpy(cursor, first, sizeof(int64_t) * ((size_t)n + 1));
    for (int64_t e = 0; e < m; e++) {
        int32_t x = local[ea[e]], y = local[eb[e]];
        to[cursor[x]++] = y;
        to[cursor[y]++] = x;
    }
    free(cursor);
    Net net = {n, first, to};
    int8_t *s = calloc((size_t)n, 1);
    int32_t *queue = malloc(sizeof(int32_t) * ((size_t)n + 1));
    V *normal = malloc(sizeof(V) * (size_t)(n + 1)), *frame = malloc(sizeof(V) * (size_t)(n + 1));
    for (int64_t start = 0; start < n; start++) {
        if (s[start]) continue;
        s[start] = 1;
        normal[start] = normal_of(&nodes[members[start]], 1);
        V r = round_axis(normal[start]);
        if (dot(r, r) < 1e-4) r = v_of(1, 0, 0);
        frame[start] = unit_of(plus(r, normal[start], -dot(r, normal[start])));
        int64_t head = 0, end = 0;
        queue[end++] = (int32_t)start;
        while (head < end) {
            int32_t i = queue[head++];
            const SheetNode *p = &nodes[members[i]];
            for (int64_t e = first[i]; e < first[i + 1]; e++) {
                int32_t j = to[e];
                if (s[j]) continue;
                const SheetNode *q = &nodes[members[j]];
                s[j] = ((int32_t)p->nx * q->nx + (int32_t)p->ny * q->ny + (int32_t)p->nz * q->nz) >= 0 ? s[i] : -s[i];
                normal[j] = normal_of(q, s[j]);
                frame[j] = carried(frame[i], normal[i], normal[j]);
                queue[end++] = j;
            }
        }
    }
    double *angle = calloc((size_t)n + 1, sizeof(double)), *b = calloc((size_t)n + 1, sizeof(double)), *weight = calloc((size_t)n + 1, sizeof(double));
    for (int round = 0; round < 2; round++) {
        memset(b, 0, sizeof(double) * (size_t)n);
        for (int64_t e = 0; e < m; e++) {
            int32_t i = local[ea[e]], j = local[eb[e]];
            V w = carried(frame[i], normal[i], normal[j]);
            double o = atan2(dot(w, cross(normal[j], frame[j])), dot(w, frame[j]));
            b[j] += o;
            b[i] -= o;
        }
        for (int64_t i = 0; i < n; i++) {
            V r = round_axis(normal[i]);
            double level = sqrt(dot(r, r));
            weight[i] = held * level * level;
            angle[i] = 0;
            if (level < 1e-3) continue;
            r = v_of(r.x / level, r.y / level, 0);
            double a = atan2(dot(cross(normal[i], frame[i]), r), dot(frame[i], r));
            if (a > M_PI / 2) a -= M_PI;
            else if (a < -M_PI / 2) a += M_PI;
            b[i] += weight[i] * a;
        }
        solve(&net, weight, b, angle, steps, tolerance);
        for (int64_t i = 0; i < n; i++) {
            double c = cos(angle[i]), d = sin(angle[i]);
            V f = plus(v_of(c * frame[i].x, c * frame[i].y, c * frame[i].z), cross(normal[i], frame[i]), d);
            frame[i] = unit_of(plus(f, normal[i], -dot(f, normal[i])));
        }
    }
    double *u = calloc((size_t)n + 1, sizeof(double)), *v = calloc((size_t)n + 1, sizeof(double)), *bu = calloc((size_t)n + 1, sizeof(double)), *bv = calloc((size_t)n + 1, sizeof(double));
    uint8_t *seen = calloc((size_t)n + 1, 1);
    for (int64_t start = 0; start < n; start++) {
        if (seen[start]) continue;
        seen[start] = 1;
        v[start] = nodes[members[start]].z;
        int64_t head = 0, end = 0;
        queue[end++] = (int32_t)start;
        while (head < end) {
            int32_t i = queue[head++];
            const SheetNode *p = &nodes[members[i]];
            for (int64_t e = first[i]; e < first[i + 1]; e++) {
                int32_t j = to[e];
                if (seen[j]) continue;
                const SheetNode *q = &nodes[members[j]];
                V d = v_of(q->x - p->x, q->y - p->y, q->z - p->z), a = unit_of(plus(frame[i], frame[j], 1.0)), c = unit_of(plus(cross(normal[i], frame[i]), cross(normal[j], frame[j]), 1.0));
                u[j] = u[i] + dot(d, a);
                v[j] = v[i] + dot(d, c);
                seen[j] = 1;
                queue[end++] = j;
            }
        }
    }
    for (int64_t e = 0; e < m; e++) {
        int32_t i = local[ea[e]], j = local[eb[e]];
        const SheetNode *p = &nodes[ea[e]], *q = &nodes[eb[e]];
        V d = v_of(q->x - p->x, q->y - p->y, q->z - p->z), a = unit_of(plus(frame[i], frame[j], 1.0)), c = unit_of(plus(cross(normal[i], frame[i]), cross(normal[j], frame[j]), 1.0));
        double du = dot(d, a), dv = dot(d, c);
        bu[j] += du; bu[i] -= du;
        bv[j] += dv; bv[i] -= dv;
    }
    solve(&net, NULL, bu, u, steps, tolerance);
    solve(&net, NULL, bv, v, steps, tolerance);
    for (int64_t i = 0; i < n; i++) {
        sign[members[i]] = s[i];
        out_u[members[i]] = (float)u[i];
        out_v[members[i]] = (float)v[i];
    }
    free(first); free(to); free(s); free(queue); free(normal); free(frame); free(angle); free(b); free(weight); free(u); free(v); free(bu); free(bv); free(seen);
}

typedef struct {
    const SheetNode *nodes;
    const int32_t *order, *ea, *eb;
    const int64_t *piece_first, *edge_first;
    int32_t *local;
    int8_t *sign;
    float *u, *v;
    int32_t pieces, steps, next;
    double tolerance, held;
} Job;

static void *worker(void *argument) {
    Job *j = argument;
    for (;;) {
        int32_t p = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (p >= j->pieces) return NULL;
        flatten_piece(j->nodes, j->order + j->piece_first[p], j->piece_first[p + 1] - j->piece_first[p], j->ea + j->edge_first[p], j->eb + j->edge_first[p], j->edge_first[p + 1] - j->edge_first[p],
                      j->local, j->sign, j->u, j->v, j->steps, j->tolerance, j->held);
    }
}

void hx_level_flatten(const SheetNode *nodes, const int32_t *order, const int64_t *piece_first, int32_t pieces, const int32_t *ea, const int32_t *eb, const int64_t *edge_first, int32_t *local, int8_t *sign,
                      float *u, float *v, int32_t steps, double tolerance, double held, int32_t threads) {
    Job job = {nodes, order, ea, eb, piece_first, edge_first, local, sign, u, v, pieces, steps, 0, tolerance, held};
    pthread_t pool[64];
    if (threads > 64) threads = 64;
    for (int32_t i = 0; i < threads; i++) pthread_create(&pool[i], NULL, worker, &job);
    for (int32_t i = 0; i < threads; i++) pthread_join(pool[i], NULL);
}
