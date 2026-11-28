#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t at, adjacent[4];
    int count;
} FreeCell;

int hx_relax_mesh(void *grid, int single, int64_t planes, int64_t h, int64_t w,
    const uint8_t *known, const uint8_t *free_cells, int64_t free_count, int rounds) {
    int64_t size = h * w, used = 0;
    FreeCell *plan = malloc(free_count * sizeof(*plan));
    double *values = malloc((size + 1) * sizeof(*values));
    double *update = malloc(free_count * sizeof(*update));
    if (!plan || !values || !update) {
        free(plan);
        free(values);
        free(update);
        return -1;
    }
    values[size] = 0;
    for (int64_t i = 0; i < size; i++) {
        if (!free_cells[i])
            continue;
        int64_t edge = known ? size : i;
        FreeCell cell = {.at = i,
            .adjacent = {i >= w ? i - w : edge, i + w < size ? i + w : edge, i % w ? i - 1 : edge,
                i % w + 1 < w ? i + 1 : edge}};
        for (int k = 0; k < 4; k++) {
            int64_t j = cell.adjacent[k];
            if (known && (j == size || !known[j]))
                cell.adjacent[k] = size;
            else
                cell.count++;
        }
        if (cell.count)
            plan[used++] = cell;
    }
    for (int64_t p = 0; p < planes; p++) {
        if (single) {
            float *src = (float *)grid + p * size;
            for (int64_t i = 0; i < size; i++)
                values[i] = src[i];
        } else
            memcpy(values, (double *)grid + p * size, size * sizeof(*values));
        for (int step = 0; step < rounds; step++) {
            for (int64_t i = 0; i < used; i++) {
                const int64_t *adj = plan[i].adjacent;
                double a = values[adj[0]], b = values[adj[1]];
                double c = values[adj[2]], d = values[adj[3]];
                if (single) {
                    float sum = (float)a + (float)b + (float)c + (float)d;
                    update[i] = (float)((double)sum / plan[i].count);
                } else
                    update[i] = (a + b + c + d) / plan[i].count;
            }
            int equal = step % 8 == 7;
            for (int64_t i = 0; i < used; i++) {
                double *dst = &values[plan[i].at];
                if (equal) {
                    uint64_t before, after;
                    memcpy(&before, dst, sizeof(before));
                    memcpy(&after, &update[i], sizeof(after));
                    equal = before == after;
                }
                *dst = update[i];
            }
            if (equal)
                break;
        }
        for (int64_t i = 0; i < used; i++) {
            int64_t at = plan[i].at;
            if (single)
                ((float *)grid)[p * size + at] = values[at];
            else
                ((double *)grid)[p * size + at] = values[at];
        }
    }
    free(update);
    free(values);
    free(plan);
    return 0;
}

#ifndef HX_MESH_RELAX_ONLY
typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} SheetNode;
typedef struct {
    double x, y, z;
} V;

static V v_of(double x, double y, double z) { return (V){x, y, z}; }
static double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V cross(V a, V b) {
    return v_of(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static V plus(V a, V b, double k) { return v_of(a.x + k * b.x, a.y + k * b.y, a.z + k * b.z); }
static V unit_of(V a) {
    double l = sqrt(dot(a, a)) + 1e-12;
    return v_of(a.x / l, a.y / l, a.z / l);
}

static V normal_of(const SheetNode *n, int s) {
    return unit_of(v_of((double)s * n->nx, (double)s * n->ny, (double)s * n->nz));
}

static V carried(V e, V a, V b) {
    double along = dot(b, e) / (1.0 + dot(a, b) + 1e-9);
    V r = plus(e, plus(a, b, 1.0), -along);
    return unit_of(plus(r, b, -dot(r, b)));
}

static V round_axis(V n) { return v_of(-n.y, n.x, 0.0); }

typedef struct {
    int64_t n;
    const int64_t *first;
    const int32_t *to;
} Net;

typedef struct {
    const Net *net;
    const double *diag, *b;
    double *x;
    int steps;
    double tolerance;
} Solve;

static void *solve(void *argument) {
    Solve *call = argument;
    const Net *g = call->net;
    const double *diag = call->diag, *b = call->b;
    double *x = call->x;
    int64_t n = g->n;
    double *r = malloc(4 * (size_t)n * sizeof(*r));
    double *d = r + n, *q = d + n, *m = q + n;
    double rz = 0, scale = 1e-12;
    for (int64_t i = 0; i < n; i++) {
        double degree = (double)(g->first[i + 1] - g->first[i]) + (diag ? diag[i] : 0.0);
        m[i] = degree > 0 ? 1.0 / degree : 0.0;
        double applied = degree * x[i];
        for (int64_t e = g->first[i]; e < g->first[i + 1]; e++)
            applied -= x[g->to[e]];
        r[i] = b[i] - applied;
        d[i] = m[i] * r[i];
        rz += r[i] * d[i];
        scale += b[i] * b[i] * m[i];
    }
    for (int step = 0; step < call->steps && rz >= call->tolerance * call->tolerance * scale;
        step++) {
        double pq = 0;
        for (int64_t i = 0; i < n; i++) {
            double degree = (double)(g->first[i + 1] - g->first[i]) + (diag ? diag[i] : 0.0);
            double a = degree * d[i];
            for (int64_t e = g->first[i]; e < g->first[i + 1]; e++)
                a -= d[g->to[e]];
            q[i] = a;
            pq += d[i] * a;
        }
        if (pq <= 0)
            break;
        double alpha = rz / pq, fresh = 0;
        for (int64_t i = 0; i < n; i++) {
            x[i] += alpha * d[i];
            r[i] -= alpha * q[i];
            fresh += r[i] * r[i] * m[i];
        }
        double beta = fresh / rz;
        for (int64_t i = 0; i < n; i++)
            d[i] = m[i] * r[i] + beta * d[i];
        rz = fresh;
    }
    free(r);
    return NULL;
}

static void edge_step(const SheetNode *p, const SheetNode *q, const V *frame, const V *normal,
    int32_t i, int32_t j, double *du, double *dv) {
    V d = v_of(q->x - p->x, q->y - p->y, q->z - p->z);
    V a = unit_of(plus(frame[i], frame[j], 1.0));
    V c = unit_of(plus(cross(normal[i], frame[i]), cross(normal[j], frame[j]), 1.0));
    *du = dot(d, a);
    *dv = dot(d, c);
}

enum { SPLIT_NODES = 20000 };

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
    Job *job = argument;
    const SheetNode *nodes = job->nodes;
    int32_t *local = job->local;
    for (;;) {
        int32_t piece = __atomic_fetch_add(&job->next, 1, __ATOMIC_RELAXED);
        if (piece >= job->pieces)
            return NULL;
        int64_t node = job->piece_first[piece], edge = job->edge_first[piece];
        int64_t n = job->piece_first[piece + 1] - node, m = job->edge_first[piece + 1] - edge;
        if (n <= 0)
            continue;
        const int32_t *members = job->order + node, *ea = job->ea + edge, *eb = job->eb + edge;
        for (int64_t i = 0; i < n; i++)
            local[members[i]] = (int32_t)i;
        int64_t *first = calloc((size_t)n + 2, sizeof(int64_t));
        int32_t *to = malloc(sizeof(int32_t) * (2 * (size_t)m + 1));
        for (int64_t e = 0; e < m; e++) {
            first[local[ea[e]] + 2]++;
            first[local[eb[e]] + 2]++;
        }
        for (int64_t i = 1; i <= n; i++)
            first[i + 1] += first[i];
        for (int64_t e = 0; e < m; e++) {
            int32_t x = local[ea[e]], y = local[eb[e]];
            to[first[x + 1]++] = y;
            to[first[y + 1]++] = x;
        }
        Net net = {.n = n, .first = first, .to = to};
        int8_t *s = calloc((size_t)n, 1);
        int32_t *order = malloc(2 * (size_t)n * sizeof(*order)), *parent = order + n;
        int64_t head = 0, end = 0;
        V *normal = malloc(2 * (size_t)n * sizeof(*normal)), *frame = normal + n;
        for (int64_t start = 0; start < n; start++) {
            if (s[start])
                continue;
            s[start] = 1;
            normal[start] = normal_of(&nodes[members[start]], 1);
            V r = round_axis(normal[start]);
            if (dot(r, r) < 1e-4)
                r = v_of(1, 0, 0);
            frame[start] = unit_of(plus(r, normal[start], -dot(r, normal[start])));
            parent[start] = -1;
            order[end++] = start;
            while (head < end) {
                int32_t i = order[head++];
                const SheetNode *p = &nodes[members[i]];
                for (int64_t e = first[i]; e < first[i + 1]; e++) {
                    int32_t j = to[e];
                    if (s[j])
                        continue;
                    const SheetNode *q = &nodes[members[j]];
                    s[j] = ((int32_t)p->nx * q->nx + (int32_t)p->ny * q->ny +
                               (int32_t)p->nz * q->nz) >= 0
                        ? s[i]
                        : -s[i];
                    normal[j] = normal_of(q, s[j]);
                    frame[j] = carried(frame[i], normal[i], normal[j]);
                    parent[j] = i;
                    order[end++] = j;
                }
            }
        }
        double *angle = malloc(3 * (size_t)n * sizeof(*angle)), *b = angle + n, *weight = b + n;
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
                weight[i] = job->held * level * level;
                angle[i] = 0;
                if (level < 1e-3)
                    continue;
                r = v_of(r.x / level, r.y / level, 0);
                double a = atan2(dot(cross(normal[i], frame[i]), r), dot(frame[i], r));
                if (a > M_PI / 2)
                    a -= M_PI;
                else if (a < -M_PI / 2)
                    a += M_PI;
                b[i] += weight[i] * a;
            }
            solve(&(Solve){.net = &net,
                .diag = weight,
                .b = b,
                .x = angle,
                .steps = job->steps,
                .tolerance = job->tolerance});
            for (int64_t i = 0; i < n; i++) {
                double c = cos(angle[i]), d = sin(angle[i]);
                V f = plus(v_of(c * frame[i].x, c * frame[i].y, c * frame[i].z),
                    cross(normal[i], frame[i]), d);
                frame[i] = unit_of(plus(f, normal[i], -dot(f, normal[i])));
            }
        }
        free(angle);
        double *u = calloc(4 * (size_t)n, sizeof(*u)), *v = u + n, *bu = v + n, *bv = bu + n;
        for (int64_t k = 0; k < n; k++) {
            int32_t j = order[k], i = parent[j];
            if (i < 0) {
                v[j] = nodes[members[j]].z;
                continue;
            }
            const SheetNode *p = &nodes[members[i]], *q = &nodes[members[j]];
            double du, dv;
            edge_step(p, q, frame, normal, i, j, &du, &dv);
            u[j] = u[i] + du;
            v[j] = v[i] + dv;
        }
        for (int64_t e = 0; e < m; e++) {
            int32_t i = local[ea[e]], j = local[eb[e]];
            const SheetNode *p = &nodes[ea[e]], *q = &nodes[eb[e]];
            double du, dv;
            edge_step(p, q, frame, normal, i, j, &du, &dv);
            bu[j] += du;
            bu[i] -= du;
            bv[j] += dv;
            bv[i] -= dv;
        }
        Solve along_u = {
            .net = &net, .b = bu, .x = u, .steps = job->steps, .tolerance = job->tolerance};
        Solve along_v = {
            .net = &net, .b = bv, .x = v, .steps = job->steps, .tolerance = job->tolerance};
        pthread_t second;
        int split = n >= SPLIT_NODES && pthread_create(&second, NULL, solve, &along_v) == 0;
        solve(&along_u);
        if (split)
            pthread_join(second, NULL);
        else
            solve(&along_v);
        for (int64_t i = 0; i < n; i++) {
            job->sign[members[i]] = s[i];
            job->u[members[i]] = (float)u[i];
            job->v[members[i]] = (float)v[i];
        }
        free(first);
        free(to);
        free(s);
        free(order);
        free(normal);
        free(u);
    }
}

void hx_level_flatten(const SheetNode *nodes, const int32_t *order, const int64_t *piece_first,
    int32_t pieces, const int32_t *ea, const int32_t *eb, const int64_t *edge_first, int32_t *local,
    int8_t *sign, float *u, float *v, int32_t steps, double tolerance, double held,
    int32_t threads) {
    Job job = {.nodes = nodes,
        .order = order,
        .ea = ea,
        .eb = eb,
        .piece_first = piece_first,
        .edge_first = edge_first,
        .local = local,
        .sign = sign,
        .u = u,
        .v = v,
        .pieces = pieces,
        .steps = steps,
        .tolerance = tolerance,
        .held = held};
    pthread_t pool[63];
    int32_t started = 0;
    while (started + 1 < threads && started < 63) {
        if (pthread_create(&pool[started], NULL, worker, &job))
            break;
        started++;
    }
    worker(&job);
    for (int32_t i = 0; i < started; i++)
        pthread_join(pool[i], NULL);
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

typedef struct {
    double count, u, v, uu, uv, vv;
    double xyz[3], xu[3], xv[3];
} Moments;

typedef struct {
    Moments local;
    double take, du, dv;
} MeshCell;

int64_t hx_raster_cells(int64_t height, int64_t width, int64_t nodes, const double *xyz,
    const double *normal, double *u, double *v, double step, double fold, int64_t *flat,
    double *grid, uint8_t *locked, double *spread) {
    int64_t size = height * width, occupied = 0;
    int64_t *count = calloc((size_t)size, sizeof(*count));
    if (!count)
        return -1;
    for (int64_t i = 0; i < nodes; i++) {
        if (!(u[i] >= 0 && u[i] / step < width && v[i] >= 0 && v[i] / step < height)) {
            free(count);
            return -2;
        }
        int64_t x = u[i] / step, y = v[i] / step, at = y * width + x;
        flat[i] = at;
        u[i] -= (x + 0.5) * step;
        v[i] -= (y + 0.5) * step;
        count[at]++;
        for (int k = 0; k < 3; k++)
            grid[k * size + at] += xyz[3 * i + k];
    }
    for (int64_t j = 0; j < size; j++) {
        if (!count[j])
            continue;
        occupied++;
        for (int k = 0; k < 3; k++)
            grid[k * size + j] /= count[j];
    }
    for (int64_t i = 0; i < nodes; i++) {
        int64_t at = flat[i];
        double depth = 0;
        for (int k = 0; k < 3; k++)
            depth += (xyz[3 * i + k] - grid[k * size + at]) * normal[3 * i + k];
        spread[at] += depth * depth;
    }
    for (int64_t j = 0; j < size; j++) {
        if (count[j])
            spread[j] = sqrt(spread[j] / count[j]);
        locked[j] = count[j] && spread[j] <= fold;
    }
    free(count);
    return occupied;
}

int hx_centre_cells(int64_t height, int64_t width, int64_t nodes, const int64_t *flat,
    const double *du, const double *dv, const double *xyz, const uint8_t *take,
    const uint8_t *source, double step, int min_nodes, double min_spread, double min_length,
    double max_length, double *mean) {
    int64_t size = height * width;
    MeshCell *cells = calloc(size, sizeof(*cells));
    if (!cells)
        return -1;
    for (int64_t i = 0; i < nodes; i++) {
        if (flat[i] < 0 || flat[i] >= size) {
            free(cells);
            return -2;
        }
        MeshCell *cell = cells + flat[i];
        double u = du[i], v = dv[i];
        if (!take || take[i]) {
            cell->take++;
            cell->du += u;
            cell->dv += v;
        }
        if (!source[take ? i : flat[i]])
            continue;
        Moments *s = &cell->local;
        s->count++;
        s->u += u;
        s->v += v;
        s->uu += u * u;
        s->uv += u * v;
        s->vv += v * v;
        for (int k = 0; k < 3; k++) {
            double x = xyz[3 * i + k];
            s->xyz[k] += x;
            s->xu[k] += x * u;
            s->xv[k] += x * v;
        }
    }
    for (int64_t i = 0; i < size; i++) {
        if (!cells[i].take)
            continue;
        Moments m = {0};
        int64_t y = i / width, x = i % width;
        for (int dy = -1; dy <= 1; dy++) {
            if (y + dy < 0 || y + dy >= height)
                continue;
            for (int dx = -1; dx <= 1; dx++) {
                if (x + dx < 0 || x + dx >= width)
                    continue;
                Moments *s = &cells[i + dy * width + dx].local;
                double u = dx * step, v = dy * step;
                m.count += s->count;
                m.u += s->u + u * s->count;
                m.v += s->v + v * s->count;
                m.uu += s->uu + 2 * u * s->u + u * u * s->count;
                m.vv += s->vv + 2 * v * s->v + v * v * s->count;
                m.uv += s->uv + v * s->u + u * s->v + u * v * s->count;
                for (int k = 0; k < 3; k++) {
                    m.xyz[k] += s->xyz[k];
                    m.xu[k] += s->xu[k] + u * s->xyz[k];
                    m.xv[k] += s->xv[k] + v * s->xyz[k];
                }
            }
        }
        if (m.count < min_nodes)
            continue;
        double u = m.u / m.count, v = m.v / m.count;
        double uu = m.uu / m.count - u * u, vv = m.vv / m.count - v * v;
        double uv = m.uv / m.count - u * v, det = uu * vv - uv * uv;
        double delta = uu - vv;
        double narrow = (uu + vv) / 2 - sqrt(fmax(delta * delta / 4 + uv * uv, 0));
        if (!(narrow > min_spread))
            continue;
        double along[3], across[3], lu = 0, lv = 0;
        for (int k = 0; k < 3; k++) {
            double centre = m.xyz[k] / m.count;
            double xu = m.xu[k] / m.count - centre * u;
            double xv = m.xv[k] / m.count - centre * v;
            along[k] = (xu * vv - xv * uv) / det;
            across[k] = (xv * uu - xu * uv) / det;
            lu += along[k] * along[k];
            lv += across[k] * across[k];
        }
        lu = sqrt(lu);
        lv = sqrt(lv);
        if (lu > min_length && lu < max_length && lv > min_length && lv < max_length) {
            double move_u = cells[i].du / cells[i].take, move_v = cells[i].dv / cells[i].take;
            for (int k = 0; k < 3; k++)
                mean[k * size + i] -= along[k] * move_u + across[k] * move_v;
        }
    }
    free(cells);
    return 0;
}

static int compare_length(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    if (isnan(a) || isnan(b)) {
        return isnan(a) - isnan(b);
    }
    return (a > b) - (a < b);
}

static void select_lengths(
    double *x, size_t lo, size_t hi, const size_t *ranks, size_t n, unsigned int depth) {
    if (!n || hi - lo <= 1) {
        return;
    }
    if (hi - lo <= 24) {
        for (size_t i = lo + 1; i < hi; i++) {
            double v = x[i];
            size_t j = i;
            while (j > lo && x[j - 1] > v) {
                x[j] = x[j - 1];
                j--;
            }
            x[j] = v;
        }
        return;
    }
    if (depth == 40) {
        qsort(x + lo, hi - lo, sizeof(*x), compare_length);
        return;
    }
    double first = x[lo], last = x[hi - 1], middle = x[lo + (hi - lo) / 2];
    double pivot = first < last ? fmax(first, fmin(middle, last)) : fmax(last, fmin(middle, first));
    size_t below = lo, above = hi;
    for (size_t j = lo; j < above;) {
        double v = x[j];
        if (v < pivot) {
            x[j++] = x[below];
            x[below++] = v;
        } else if (v > pivot) {
            x[j] = x[--above];
            x[above] = v;
        } else {
            j++;
        }
    }
    size_t left = 0, right = n;
    while (left < n && ranks[left] < below) {
        left++;
    }
    while (right > left && ranks[right - 1] >= above) {
        right--;
    }
    select_lengths(x, lo, below, ranks, left, depth + 1);
    select_lengths(x, above, hi, ranks + right, n - right, depth + 1);
}

static double length_percentile(const double *sorted, size_t size, double fraction) {
    double index = fraction * (double)(size - 1);
    size_t lo = (size_t)index, hi = lo + (lo + 1 < size);
    double part = index - (double)lo, delta = sorted[hi] - sorted[lo];
    return part >= 0.5 ? sorted[hi] - delta * (1.0 - part) : sorted[lo] + delta * part;
}

int hx_mesh_isometry(const double *grid, int64_t h, int64_t w, const uint8_t *known,
    const uint8_t *locked, double stride, double *results) {
    if (h < 0 || w < 0 || (h && (uint64_t)w > SIZE_MAX / 64 / (uint64_t)h)) {
        return -1;
    }
    size_t cells = (size_t)h * (size_t)w, capacity = cells ? 2 * cells : 1;
    double *storage = malloc(2 * capacity * sizeof(*storage));
    if (!storage) {
        return -1;
    }
    double *steps = storage, *changes = storage + capacity;
    for (int set = 0; set < 2; set++) {
        const uint8_t *mask = set ? locked : known;
        size_t pairs = 0, triples = 0, within = 0;
        int nan_step = 0, nan_change = 0;
        for (int axis = 0; axis < 2; axis++) {
            int64_t lines = axis ? h : w, length = axis ? w : h, advance = axis ? 1 : w;
            for (int64_t line = 0; line < lines; line++) {
                int64_t cell = axis ? line * w : line;
                double previous = 0;
                int connected = 0;
                for (int64_t j = 1; j < length; j++, cell += advance) {
                    int64_t next = cell + advance;
                    if (!mask[cell] || !mask[next]) {
                        connected = 0;
                        continue;
                    }
                    double x = grid[next] - grid[cell];
                    double y = grid[cells + next] - grid[cells + cell];
                    double z = grid[2 * cells + next] - grid[2 * cells + cell];
                    double step = sqrt((x * x + y * y) + z * z) / stride;
                    steps[pairs++] = step;
                    nan_step |= isnan(step);
                    within += fabs(step - 1.0) <= 0.1;
                    if (connected) {
                        double difference = fabs(step - previous);
                        changes[triples++] = difference;
                        nan_change |= isnan(difference);
                    }
                    previous = step;
                    connected = 1;
                }
            }
        }
        double *out = results + 7 * set;
        out[0] = (double)pairs;
        out[1] = (double)triples;
        if (pairs) {
            double fractions[] = {0.05, 0.5, 0.95};
            size_t ranks[6];
            for (int k = 0; k < 3; k++) {
                size_t rank = (size_t)(fractions[k] * (double)(pairs - 1));
                ranks[2 * k] = rank;
                ranks[2 * k + 1] = rank + (rank + 1 < pairs);
            }
            if (!nan_step) {
                select_lengths(steps, 0, pairs, ranks, 6, 0);
            }
            for (int k = 0; k < 3; k++) {
                out[2 + k] = nan_step ? NAN : length_percentile(steps, pairs, fractions[k]);
            }
            out[5] = (double)within / (double)pairs;
        }
        if (triples) {
            size_t ranks[] = {(triples - 1) / 2, triples / 2};
            if (nan_change) {
                out[6] = NAN;
            } else {
                select_lengths(changes, 0, triples, ranks, 2, 0);
                out[6] =
                    triples % 2 ? changes[ranks[0]] : (changes[ranks[0]] + changes[ranks[1]]) * 0.5;
            }
        }
    }
    free(storage);
    return 0;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
#endif
