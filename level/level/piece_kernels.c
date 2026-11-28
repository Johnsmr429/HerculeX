#include "parallel_blocks.h"
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 8
#define AHEAD 16
#define EDGE_BLOCKS 64
#define TWO_PI 6.283185307179586f

typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} Node;

void hx_node_quality(const Node *nodes, int64_t count, double span, float *out) {
    if (!count)
        return;
    int64_t histogram[256] = {0};
    for (int64_t i = 0; i < count; i++)
        histogram[nodes[i].prominence]++;
    int lo = 0, hi = 0;
    int64_t sum = 0;
    for (int v = 0; v < 256; v++) {
        sum += histogram[v];
        if (sum <= (count - 1) / 2)
            lo = v + 1;
        if (sum <= count / 2)
            hi = v + 1;
    }
    float scale = span * (lo + hi) * 0.5;
    for (int64_t i = 0; i < count; i++) {
        float strength = nodes[i].prominence / scale;
        float coherence = (float)nodes[i].coherence / 255.0f + 0.25f;
        out[i] = (strength > 1 ? 1 : strength) * (coherence > 1 ? 1 : coherence);
    }
}

void hx_node_area(const Node *nodes, int64_t count, float cell_area, float *out) {
    for (int64_t i = 0; i < count; i++) {
        float x = nodes[i].nx, y = nodes[i].ny, z = nodes[i].nz;
        float norm = sqrtf(x * x + y * y + z * z);
        float spread = (fabsf(x) + fabsf(y) + fabsf(z)) / fmaxf(norm, 1e-6f);
        out[i] = cell_area / fmaxf(spread, 1.0f);
    }
}

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} Edge;

static uint32_t curve_cell(float coordinate, float size) {
    float cell = floorf(coordinate / size);
    if (!(cell >= -2147483648.0f && cell < 2147483648.0f)) {
        return 0;
    }
    return (uint32_t)(int32_t)cell & 4095;
}

int hx_curve_patches(
    const Node *nodes, const int32_t *patch, int32_t n, int32_t count, float size, int32_t *out) {
    if (n == 0) {
        return 0;
    }
    int error = -1;
    int32_t *order = calloc((size_t)count + 1, sizeof(*order));
    int32_t *next = malloc(((size_t)count + 1) * sizeof(*next));
    uint64_t *key = malloc(((size_t)count + 1) * sizeof(*key));
    if (!order || !next || !key) {
        goto finish;
    }
    for (int32_t i = n; i-- > 0;) {
        int32_t id = patch[i] < 0 ? patch[i] + count : patch[i];
        if (id < 0 || id >= count) {
            error = -2;
            goto finish;
        }
        order[id] = i;
    }
    uint64_t bits[4096] = {0};
    for (int i = 1; i < 4096; i++) {
        bits[i] = (bits[i >> 1] << 3) | (i & 1);
    }
    for (int32_t j = 0; j < count; j++) {
        Node node = nodes[order[j]];
        key[j] = bits[curve_cell(node.x, size)] | bits[curve_cell(node.y, size)] << 1 |
            bits[curve_cell(node.z, size)] << 2;
        order[j] = j;
    }
    for (int shift = 0; shift < 36; shift += 12) {
        int32_t offsets[4096] = {0};
        for (int32_t j = 0; j < count; j++) {
            offsets[(key[j] >> shift) & 4095]++;
        }
        int32_t total = 0;
        for (int i = 0; i < 4096; i++) {
            int32_t length = offsets[i];
            offsets[i] = total;
            total += length;
        }
        for (int32_t j = 0; j < count; j++) {
            int32_t id = order[j];
            next[offsets[(key[id] >> shift) & 4095]++] = id;
        }
        int32_t *swap = order;
        order = next;
        next = swap;
    }
    for (int32_t i = 0; i < count; i++) {
        next[order[i]] = i;
    }
    for (int32_t i = 0; i < n; i++) {
        out[i] = next[patch[i] < 0 ? patch[i] + count : patch[i]];
    }
    error = 0;
finish:
    free(key);
    free(next);
    free(order);
    return error;
}

void hx_patch_start(const int32_t *x, const int32_t *piece, const int32_t *potential,
    const int32_t *patch, const int32_t *inner, int32_t chirality, int64_t nodes, int32_t *start) {
    for (int64_t i = 0; i < nodes; i++) {
        int64_t delta = (int64_t)potential[i] - inner[i];
        start[patch[i]] = (int32_t)(x[piece[i]] + (int64_t)chirality * delta);
    }
}

void hx_firm_levels(const int32_t *x, const int32_t *p, const int32_t *q, const int32_t *k,
    int64_t terms, int32_t *votes, int draws, const int32_t *patch, const int32_t *inner,
    int64_t nodes, int chirality, int32_t *level) {
    for (int64_t t = 0; t < terms; t++)
        votes[t] = votes[t] >= draws && (uint32_t)x[q[t]] - (uint32_t)x[p[t]] == (uint32_t)k[t];
    for (int64_t i = 0; i < nodes; i++)
        level[i] = (int32_t)((int64_t)x[patch[i]] + (int64_t)chirality * inner[i]);
}

void hx_firm_edges(const int32_t *votes, const int32_t *x, const int32_t *patch,
    const int32_t *inner, int chirality, const Edge *edges, int64_t count, const int32_t *term,
    uint8_t *kept, uint8_t *firm) {
    for (int64_t i = 0; i < count; i++) {
        int32_t a = edges[i].a, b = edges[i].b, t = term[i];
        int64_t delta =
            (int64_t)x[patch[b]] - x[patch[a]] + chirality * ((int64_t)inner[b] - inner[a]);
        kept[i] = delta == (int64_t)chirality * edges[i].step;
        firm[i] = kept[i] && (t == -1 || votes[t >= 0 ? t : -2LL - t]);
    }
}

void hx_selected_piece_edges(const Edge *edges, int64_t length, const int32_t *label,
    const int32_t *potential, int64_t groups, int64_t *cursor, int32_t *ea, int32_t *eb) {
    for (int64_t i = 0; i < length; i++) {
        int32_t a = edges[i].a, b = edges[i].b, group = label[a];
        if (group < 0 || group >= groups || group != label[b] ||
            (uint32_t)potential[b] - (uint32_t)potential[a] != (uint32_t)edges[i].step) {
            continue;
        }
        int64_t at = cursor[group]++;
        if (ea) {
            ea[at] = a;
            eb[at] = b;
        }
    }
}

static inline float unit(int8_t v) { return v / 127.0f; }

typedef struct {
    float x, y, z, length;
} Direction;

static int64_t match_cells(const Node *nodes, const float *theta, const float *quality, int64_t a0,
    int64_t a1, int64_t b0, int64_t b1, const Direction *left, const Direction *right, float sigma,
    float max_depth, float min_cos, float max_lateral, Edge *out) {
    int na = (int)(a1 - a0), nb = (int)(b1 - b0);
    float depth[MAXN][MAXN];
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++) {
            const Node *p = &nodes[a0 + i], *q = &nodes[b0 + j];
            float ax = unit(p->nx), ay = unit(p->ny), az = unit(p->nz);
            float bx = unit(q->nx), by = unit(q->ny), bz = unit(q->nz);
            float la = left[i].length, lb = right[j].length;
            float dot = (ax * bx + ay * by + az * bz) / (la * lb);
            float sign = dot < 0 ? -1.0f : 1.0f;
            depth[i][j] = 1e9f;
            if (dot * sign < min_cos)
                continue;
            float mx = left[i].x + sign * right[j].x;
            float my = left[i].y + sign * right[j].y;
            float mz = left[i].z + sign * right[j].z;
            float lm = sqrtf(mx * mx + my * my + mz * mz);
            mx /= lm;
            my /= lm;
            mz /= lm;
            float dx = q->x - p->x, dy = q->y - p->y, dz = q->z - p->z;
            float along = dx * mx + dy * my + dz * mz;
            float lateral = sqrtf(fmaxf(dx * dx + dy * dy + dz * dz - along * along, 0));
            if (fabsf(along) > max_depth || lateral > max_lateral)
                continue;
            depth[i][j] = fabsf(along);
        }
    int64_t written = 0;
    for (int i = 0; i < na; i++) {
        int best = -1;
        float m1 = 1e9f, m2 = 1e9f;
        for (int j = 0; j < nb; j++) {
            float d = depth[i][j];
            if (d < m1) {
                m2 = m1;
                m1 = d;
                best = j;
            } else if (d < m2)
                m2 = d;
        }
        if (best < 0)
            continue;
        int back = -1;
        float c1 = 1e9f, c2 = 1e9f;
        for (int k = 0; k < na; k++) {
            float d = depth[k][best];
            if (d < c1) {
                c2 = c1;
                c1 = d;
                back = k;
            } else if (d < c2)
                c2 = d;
        }
        if (back != i)
            continue;
        float rival = fminf(m2, c2);
        float unique = rival > 1e8f ? 1.0f : (rival - m1) / (rival + m1 + 1e-6f);
        int64_t u = a0 + i, v = b0 + best;
        float q = fminf(quality[u], quality[v]);
        float e = m1 / sigma;
        float dtheta = theta[v] - theta[u];
        out[written].a = (int32_t)u;
        out[written].b = (int32_t)v;
        out[written].weight = q * expf(-e * e) * unique;
        out[written].step = -(int32_t)lrintf(dtheta / TWO_PI);
        written++;
    }
    return written;
}

int64_t hx_sheet_edges(const Node *nodes, const int32_t *first, int GZ, int GY, int GX, int plane,
    const float *theta, const float *quality, float sigma, float max_depth, float min_cos,
    float max_lateral, Edge *out, int64_t capacity) {
    static const int forward[13][3] = {{0, 0, 1}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1}, {1, -1, -1},
        {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1}, {1, 1, -1}, {1, 1, 0}, {1, 1, 1}};
    int64_t cells = (int64_t)GY * GX;
    int start = first[plane * cells];
    int end = first[(plane + (plane + 1 < GZ ? 2 : 1)) * cells];
    if (end == start) {
        return 0;
    }
    Direction *directions = malloc((size_t)(end - start) * sizeof(*directions));
    if (!directions) {
        return -1;
    }
    for (int i = start; i < end; i++) {
        float x = unit(nodes[i].nx), y = unit(nodes[i].ny), z = unit(nodes[i].nz);
        float length = sqrtf(x * x + y * y + z * z);
        directions[i - start] =
            (Direction){.x = x / length, .y = y / length, .z = z / length, .length = length};
    }
    int64_t count = 0;
    for (int y = 0; y < GY; y++)
        for (int x = 0; x < GX; x++) {
            int64_t c = ((int64_t)plane * GY + y) * GX + x;
            if (first[c] == first[c + 1])
                continue;
            for (int k = 0; k < 13; k++) {
                int zz = plane + forward[k][0], yy = y + forward[k][1], xx = x + forward[k][2];
                if (zz >= GZ || yy < 0 || yy >= GY || xx < 0 || xx >= GX)
                    continue;
                int64_t d = ((int64_t)zz * GY + yy) * GX + xx;
                if (first[d] == first[d + 1])
                    continue;
                int na = first[c + 1] - first[c], nb = first[d + 1] - first[d];
                if (na > MAXN || nb > MAXN || count + na > capacity) {
                    free(directions);
                    return -1;
                }
                count += match_cells(nodes, theta, quality, first[c], first[c + 1], first[d],
                    first[d + 1], directions + (first[c] - start), directions + (first[d] - start),
                    sigma, max_depth, min_cos, max_lateral, out + count);
            }
        }
    free(directions);
    return count;
}

void hx_cell_table(const int32_t *cell_of, int64_t count, int32_t *first, int64_t cells) {
    memset(first, 0, sizeof(int32_t) * (size_t)(cells + 1));
    for (int64_t i = 0; i < count; i++)
        first[cell_of[i] + 1]++;
    for (int64_t c = 0; c < cells; c++)
        first[c + 1] += first[c];
}

static size_t merge_slice(
    const Edge **head, int64_t *left, int count, Edge *out, size_t capacity, const uint8_t *more) {
    size_t used = 0;
    for (; used < capacity; used++) {
        int best = -1;
        float top = 0;
        for (int r = 0; r < count; r++) {
            if (left[r] && (best < 0 || head[r]->weight > top)) {
                best = r;
                top = head[r]->weight;
            }
        }
        if (best < 0) {
            break;
        }
        *out++ = *head[best]++;
        left[best]--;
        if (!left[best] && more && more[best]) {
            return used + 1;
        }
    }
    return used;
}

void hx_merge_edges(const int64_t *runs, const int64_t *lengths, int count, Edge *out) {
    const Edge *head[count];
    int64_t left[count];
    for (int r = 0; r < count; r++) {
        head[r] = (const Edge *)runs[r];
        left[r] = lengths[r];
    }
    merge_slice(head, left, count, out, SIZE_MAX, NULL);
}

static inline uint32_t falling_key(float weight) {
    if (weight == 0)
        weight = 0;
    uint32_t bits;
    memcpy(&bits, &weight, sizeof(bits));
    return (bits & 0x80000000u) ? bits : ~(bits | 0x80000000u);
}

void hx_sort_edges(Edge *edges, int64_t count, Edge *scratch) {
    int64_t totals[4][256];
    memset(totals, 0, sizeof(totals));
    for (int64_t e = 0; e < count; e++) {
        uint32_t key = falling_key(edges[e].weight);
        for (int pass = 0; pass < 4; pass++)
            totals[pass][(key >> (8 * pass)) & 255]++;
    }
    Edge *from = edges, *to = scratch;
    for (int pass = 0; pass < 4; pass++) {
        int64_t at[256], sum = 0;
        for (int d = 0; d < 256; d++) {
            at[d] = sum;
            sum += totals[pass][d];
        }
        for (int64_t e = 0; e < count; e++)
            to[at[(falling_key(from[e].weight) >> (8 * pass)) & 255]++] = from[e];
        Edge *swap = from;
        from = to;
        to = swap;
    }
}

typedef struct {
    const Edge *edges;
    int64_t count;
    const Node *nodes;
    Edge *turned;
} Turns;

static void turn_block(void *context, size_t block) {
    Turns *t = context;
    const Edge *edges = t->edges;
    int64_t lo = t->count * (int64_t)block / EDGE_BLOCKS;
    int64_t hi = t->count * (int64_t)(block + 1) / EDGE_BLOCKS;
    for (int64_t e = lo; e < hi; e++) {
        if (e + AHEAD < hi) {
            __builtin_prefetch(&t->nodes[edges[e + AHEAD].a]);
            __builtin_prefetch(&t->nodes[edges[e + AHEAD].b]);
        }
        const Node *p = &t->nodes[edges[e].a], *q = &t->nodes[edges[e].b];
        t->turned[e] = edges[e];
        t->turned[e].step = p->nx * q->nx + p->ny * q->ny + p->nz * q->nz < 0;
    }
}

int64_t hx_agreeing_edges(const Edge *edges, int64_t count, const Node *nodes, Edge *out) {
    int64_t kept = 0;
    for (int64_t e = 0; e < count; e++) {
        if (e + AHEAD < count) {
            __builtin_prefetch(&nodes[edges[e + AHEAD].a]);
            __builtin_prefetch(&nodes[edges[e + AHEAD].b]);
        }
        const Node *p = &nodes[edges[e].a], *q = &nodes[edges[e].b];
        if (p->nx * q->nx + p->ny * q->ny + p->nz * q->nz >= 0) {
            out[kept++] = edges[e];
        }
    }
    return kept;
}

void hx_turned_steps(const Edge *edges, int64_t count, const Node *nodes, Edge *out) {
    Turns t = {.edges = edges, .count = count, .nodes = nodes, .turned = out};
    each_block(EDGE_BLOCKS, &t, turn_block);
}

typedef struct {
    const Edge *edges;
    int64_t count;
    const void *values;
    uint8_t *out;
    int wide, potential;
} Labels;

static void compare_block(void *context, size_t block) {
    Labels *l = context;
    const int32_t *label = l->values;
    const int64_t *wide = l->values;
    const char *bytes = l->values;
    size_t stride = l->wide ? sizeof(*wide) : sizeof(*label);
    int64_t lo = l->count * (int64_t)block / EDGE_BLOCKS;
    int64_t hi = l->count * (int64_t)(block + 1) / EDGE_BLOCKS;
    for (int64_t e = lo; e < hi; e++) {
        if (e + AHEAD < hi) {
            __builtin_prefetch(bytes + stride * l->edges[e + AHEAD].a);
            __builtin_prefetch(bytes + stride * l->edges[e + AHEAD].b);
        }
        int32_t a = l->edges[e].a, b = l->edges[e].b;
        if (l->wide)
            l->out[e] = wide[a] == wide[b];
        else if (l->potential)
            l->out[e] = label[b] - label[a] == l->edges[e].step;
        else
            l->out[e] = label[a] == label[b];
    }
}

void hx_same_label(const Edge *edges, int64_t count, const int32_t *label, uint8_t *out) {
    Labels l = {.edges = edges, .count = count, .values = label, .out = out};
    each_block(EDGE_BLOCKS, &l, compare_block);
}

void hx_same_label_wide(const Edge *edges, int64_t count, const int64_t *label, uint8_t *out) {
    Labels l = {.edges = edges, .count = count, .values = label, .out = out, .wide = 1};
    each_block(EDGE_BLOCKS, &l, compare_block);
}

void hx_potential_agrees(const Edge *edges, int64_t count, const int32_t *potential, uint8_t *out) {
    Labels l = {.edges = edges, .count = count, .values = potential, .out = out, .potential = 1};
    each_block(EDGE_BLOCKS, &l, compare_block);
}

typedef struct {
    int32_t *parent, *potential, *size;
} Forest;

static Forest forest_of(int32_t *data, int64_t count) {
    return (Forest){.parent = data, .potential = data + count, .size = data + 2 * count};
}

static int32_t find(Forest *f, int32_t i, int32_t *n) {
    int32_t parent = f->parent[i];
    if (f->parent[parent] == parent) {
        *n = f->potential[i];
        return parent;
    }
    int32_t root = i, sum = 0;
    while (f->parent[root] != root) {
        sum += f->potential[root];
        root = f->parent[root];
    }
    int32_t node = i, remaining = sum;
    while (f->parent[node] != node) {
        int32_t up = f->parent[node], own = f->potential[node];
        f->parent[node] = root;
        f->potential[node] = remaining;
        remaining -= own;
        node = up;
    }
    *n = sum;
    return root;
}

static int32_t unite(Forest *f, int32_t a, int32_t b, int32_t shift) {
    int32_t small = a, big = b;
    if (f->size[a] > f->size[b]) {
        small = b;
        big = a;
        shift = -shift;
    }
    f->parent[small] = big;
    f->potential[small] = -shift;
    f->size[big] += f->size[small];
    return big;
}

void hx_forest_init(int64_t count, int32_t *data) {
    Forest f = forest_of(data, count);
    for (int64_t i = 0; i < count; i++) {
        f.parent[i] = i;
        f.potential[i] = 0;
        f.size[i] = 1;
    }
}

void hx_grow_turns(const Node *nodes, int64_t count, int32_t *data, const Edge *edges, int64_t n) {
    Forest f = forest_of(data, count);
    for (int64_t i = 0; i < n; i++) {
        if (i + AHEAD < n) {
            __builtin_prefetch(&f.parent[edges[i + AHEAD].a]);
            __builtin_prefetch(&f.parent[edges[i + AHEAD].b]);
            __builtin_prefetch(&nodes[edges[i + AHEAD].a]);
            __builtin_prefetch(&nodes[edges[i + AHEAD].b]);
        }
        int32_t x = edges[i].a, y = edges[i].b, u, v;
        int32_t a = find(&f, x, &u), b = find(&f, y, &v);
        if (a != b) {
            const Node *p = nodes + x, *q = nodes + y;
            int step = p->nx * q->nx + p->ny * q->ny + p->nz * q->nz < 0;
            unite(&f, a, b, u + step - v);
        }
    }
}

void hx_labels(int64_t count, int32_t *data, int32_t *root, int32_t *n) {
    Forest f = forest_of(data, count);
    for (int64_t i = 0; i < count; i++)
        root[i] = find(&f, i, &n[i]);
}

typedef struct {
    int32_t a, b, rank;
} Pair;

void hx_ordered_rays(
    const Pair *pairs, const float *weight, const intptr_t *order, int64_t count, Edge *out) {
    for (int64_t i = 0; i < count; i++) {
        intptr_t at = order[i];
        out[i] = (Edge){.a = pairs[at].a, .b = pairs[at].b, .weight = weight[at]};
    }
}

static int compare_length(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median_length(double *values, size_t count) {
    size_t rank = count / 2, lo = 0, hi = count;
    for (int pass = 0; hi - lo > 1; pass++) {
        if (pass == 64) {
            qsort(values + lo, hi - lo, sizeof(*values), compare_length);
            break;
        }
        double a = values[lo], b = values[lo + (hi - lo) / 2], c = values[hi - 1];
        double pivot = a < b ? (b < c ? b : fmax(a, c)) : (a < c ? a : fmax(b, c));
        size_t first = lo, last = hi, i = lo;
        while (i < last) {
            double v = values[i];
            if (v < pivot) {
                values[i++] = values[first];
                values[first++] = v;
            } else if (v > pivot) {
                values[i] = values[--last];
                values[last] = v;
            } else {
                i++;
            }
        }
        if (rank < first) {
            hi = first;
        } else if (rank >= last) {
            lo = last;
        } else {
            break;
        }
    }
    if (count % 2) {
        return values[rank];
    }
    double lower = values[0];
    for (size_t j = 1; j < rank; j++) {
        if (values[j] > lower) {
            lower = values[j];
        }
    }
    return (lower + values[rank]) * 0.5;
}

static double ray_length(const Node *nodes, Pair ray) {
    double x = (double)nodes[ray.b].x - nodes[ray.a].x;
    double y = (double)nodes[ray.b].y - nodes[ray.a].y;
    double z = (double)nodes[ray.b].z - nodes[ray.a].z;
    volatile double xx = x * x, yy = y * y, zz = z * z;
    return sqrt((xx + yy) + zz);
}

int hx_sampled_gap(
    const Node *nodes, int64_t count, const Pair *pairs, size_t rays, double *result) {
    *result = NAN;
    if (!rays) {
        return 0;
    }
    size_t capacity = rays / 7 + (rays % 7 != 0);
    double *lengths = malloc(capacity * sizeof(*lengths));
    if (!lengths) {
        return -1;
    }
    size_t rank = 0, size = 0;
    int status = 0, nan = 0;
    for (size_t i = 0; i < rays; i++) {
        Pair ray = pairs[i];
        if (ray.rank != 1 || rank++ % 7 != 0) {
            continue;
        }
        if (ray.a < 0 || ray.b < 0 || ray.a >= count || ray.b >= count) {
            status = 1;
            break;
        }
        double distance = ray_length(nodes, ray);
        nan |= isnan(distance);
        lengths[size++] = distance;
    }
    if (!status && !nan && size) {
        *result = median_length(lengths, size);
    }
    free(lengths);
    return status;
}

int hx_ray_weights(const Node *nodes, int64_t count, const Pair *pairs, size_t rays,
    const float *quality, double rim, double sure, float *out) {
    if (!rays || !isfinite(rim) || rim >= sure) {
        return 1;
    }
    double *lengths = malloc(rays * sizeof(*lengths));
    if (!lengths) {
        return -1;
    }
    int status = 1;
    for (size_t i = 0; i < rays; i++) {
        Pair ray = pairs[i];
        if (ray.a < 0 || ray.b < 0 || ray.a >= count || ray.b >= count) {
            goto finish;
        }
        lengths[i] = ray_length(nodes, ray);
        if (!isfinite(lengths[i]) || !isfinite(quality[ray.a]) || !isfinite(quality[ray.b])) {
            goto finish;
        }
    }
    double cut = 0.6 * median_length(lengths, rays);
    size_t kept = 0;
    for (size_t i = 0; i < rays; i++) {
        if (lengths[i] > cut) {
            lengths[kept++] = lengths[i];
        }
    }
    if (kept) {
        double gap = median_length(lengths, kept);
        for (size_t i = 0; i < rays; i++) {
            Pair ray = pairs[i];
            double s = (ray_length(nodes, ray) / gap - rim) / (sure - rim);
            float q = quality[ray.a] < quality[ray.b] ? quality[ray.a] : quality[ray.b];
            out[i] = fmin(fmax(s, 0), 1) * q;
        }
        status = 0;
    }
finish:
    free(lengths);
    return status;
}

typedef struct {
    float depth, lateral;
    int32_t node;
} Hit;

static int compare_hit(const void *p, const void *q) {
    float a = ((const Hit *)p)->depth, b = ((const Hit *)q)->depth;
    return a < b ? -1 : a > b;
}

int64_t hx_radial_pairs(const Node *nodes, const int32_t *first, int GZ, int GY, int GX, int plane,
    int cell, int plane0, const uint8_t *member, float reach, float tube, float same_sheet,
    int ranks, Pair *out, int64_t capacity) {
    int64_t count = 0;
    Hit hits[256];
    for (int y = 0; y < GY; y++)
        for (int x = 0; x < GX; x++) {
            int64_t c = ((int64_t)plane * GY + y) * GX + x;
            for (int64_t i = first[c]; i < first[c + 1]; i++) {
                if (!member[i])
                    continue;
                const Node *p = &nodes[i];
                float nx = unit(p->nx), ny = unit(p->ny), nz = unit(p->nz);
                float length = sqrtf(nx * nx + ny * ny + nz * nz);
                nx /= length;
                ny /= length;
                nz /= length;
                float ex = p->x + reach * nx, ey = p->y + reach * ny, ez = p->z + reach * nz;
                int x0 = (int)floorf((fminf(p->x, ex) - tube) / cell),
                    x1 = (int)floorf((fmaxf(p->x, ex) + tube) / cell);
                int y0 = (int)floorf((fminf(p->y, ey) - tube) / cell),
                    y1 = (int)floorf((fmaxf(p->y, ey) + tube) / cell);
                int z0 = (int)floorf((fminf(p->z, ez) - tube) / cell) - plane0;
                int z1 = (int)floorf((fmaxf(p->z, ez) + tube) / cell) - plane0;
                if (x0 < 0)
                    x0 = 0;
                if (y0 < 0)
                    y0 = 0;
                if (z0 < 0)
                    z0 = 0;
                if (x1 >= GX)
                    x1 = GX - 1;
                if (y1 >= GY)
                    y1 = GY - 1;
                if (z1 >= GZ)
                    z1 = GZ - 1;
                int found = 0;
                for (int zz = z0; zz <= z1; zz++)
                    for (int yy = y0; yy <= y1; yy++)
                        for (int xx = x0; xx <= x1; xx++) {
                            int64_t d = ((int64_t)zz * GY + yy) * GX + xx;
                            for (int64_t j = first[d]; j < first[d + 1]; j++) {
                                if (j == i || !member[j])
                                    continue;
                                const Node *q = &nodes[j];
                                float dx = q->x - p->x, dy = q->y - p->y, dz = q->z - p->z;
                                float along = dx * nx + dy * ny + dz * nz;
                                if (along < same_sheet || along > reach)
                                    continue;
                                float lateral = dx * dx + dy * dy + dz * dz - along * along;
                                if (lateral > tube * tube)
                                    continue;
                                if (found < 256)
                                    hits[found++] = (Hit){along, lateral, (int32_t)j};
                            }
                        }
                if (!found)
                    continue;
                qsort(hits, found, sizeof(Hit), compare_hit);
                int rank = 0, k = 0;
                while (k < found && rank < ranks) {
                    int best = k, end = k + 1;
                    while (end < found && hits[end].depth - hits[end - 1].depth < same_sheet) {
                        if (hits[end].lateral < hits[best].lateral)
                            best = end;
                        end++;
                    }
                    rank++;
                    if (count == capacity)
                        return -1;
                    out[count++] = (Pair){(int32_t)i, hits[best].node, rank};
                    k = end;
                }
            }
        }
    return count;
}

void hx_group_edges(const int32_t *a, const int32_t *b, int64_t count, const int32_t *label,
    int32_t pieces, int64_t *first, int32_t *out_a, int32_t *out_b) {
    memset(first, 0, sizeof(int64_t) * ((size_t)pieces + 1));
    for (int64_t e = 0; e < count; e++)
        if (label[a[e]] >= 0)
            first[label[a[e]] + 1]++;
    for (int32_t p = 0; p < pieces; p++)
        first[p + 1] += first[p];
    int64_t *cursor = malloc(sizeof(int64_t) * ((size_t)pieces + 1));
    memcpy(cursor, first, sizeof(int64_t) * ((size_t)pieces + 1));
    for (int64_t e = 0; e < count; e++) {
        int32_t l = label[a[e]];
        if (l < 0)
            continue;
        int64_t at = cursor[l]++;
        out_a[at] = a[e];
        out_b[at] = b[e];
    }
    free(cursor);
}

enum { EDGE_READ = 65536 };

typedef struct {
    const Edge *data;
    Edge *buffer;
    int fd;
    int64_t count, offset, first, end;
} EdgeSource;

static int open_edges(EdgeSource *s, const Edge *data, int64_t count, int fd, int64_t offset) {
    *s = (EdgeSource){.data = data, .count = count, .fd = fd, .offset = offset, .end = count};
    if (count < 0 || (count && fd < 0 && !data)) {
        return 0;
    }
    if (fd >= 0 && count) {
        s->buffer = malloc(EDGE_READ * sizeof(Edge));
        s->data = s->buffer;
        s->end = 0;
        return s->buffer != NULL;
    }
    return 1;
}

static int read_edges(EdgeSource *s, int64_t i) {
    if (i < s->end || i >= s->count) {
        return 1;
    }
    int64_t size = s->count - i;
    if (size > EDGE_READ) {
        size = EDGE_READ;
    }
    size_t bytes = (size_t)size * sizeof(Edge), done = 0;
    off_t offset = s->offset + (off_t)i * sizeof(Edge);
    while (done < bytes) {
        ssize_t n = pread(s->fd, (char *)s->buffer + done, bytes - done, offset + (off_t)done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return 0;
        }
        done += (size_t)n;
    }
    s->first = i;
    s->end = i + size;
    return 1;
}

int hx_merge_edge_file(
    const int64_t *runs, const int64_t *lengths, int count, int fd, const int64_t *files) {
    if (count <= 0) {
        return count ? -1 : 0;
    }
    EdgeSource *sources = calloc(count, sizeof(*sources));
    Edge *buffer = malloc(EDGE_READ * sizeof(*buffer));
    const Edge *head[count];
    int64_t left[count];
    uint8_t more[count];
    int status = -1;
    if (!sources || !buffer) {
        goto finish;
    }
    for (int r = 0; r < count; r++) {
        EdgeSource *s = sources + r;
        if (!open_edges(s, (const Edge *)runs[r], lengths[r], files[2 * r], files[2 * r + 1])) {
            goto finish;
        }
        head[r] = s->data;
        left[r] = s->end;
        more[r] = s->end < s->count;
    }
    size_t used = 0;
    for (;;) {
        for (int r = 0; r < count; r++) {
            if (!left[r] && more[r]) {
                EdgeSource *s = sources + r;
                if (!read_edges(s, s->end)) {
                    goto finish;
                }
                head[r] = s->data;
                left[r] = s->end - s->first;
                more[r] = s->end < s->count;
            }
        }
        size_t got = merge_slice(head, left, count, buffer + used, EDGE_READ - used, more);
        used += got;
        if (!got || used == EDGE_READ) {
            size_t bytes = used * sizeof(*buffer);
            const char *data = (const char *)buffer;
            while (bytes) {
                ssize_t n = write(fd, data, bytes);
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                if (n <= 0) {
                    goto finish;
                }
                data += n;
                bytes -= n;
            }
            used = 0;
        }
        if (!got) {
            break;
        }
    }
    status = 0;
finish:
    if (sources) {
        for (int r = 0; r < count; r++) {
            free(sources[r].buffer);
        }
    }
    free(sources);
    free(buffer);
    return status;
}

typedef struct {
    const void *label;
    const uint8_t *member;
    const int32_t *potential;
    int wide;
} Selection;

static int join_edges(EdgeSource *source, const uint8_t *use, Forest *f, const Selection *selection,
    int64_t *joined) {
    for (int64_t i = 0; i < source->count;) {
        while (use && i < source->count && !use[i]) {
            i++;
        }
        if (i == source->count) {
            break;
        }
        if (!read_edges(source, i)) {
            return -1;
        }
        for (; i < source->end; i++) {
            if (use && !use[i]) {
                continue;
            }
            const Edge *edge = source->data + (i - source->first);
            if (i + AHEAD < source->end) {
                __builtin_prefetch(f->parent + edge[AHEAD].a);
                __builtin_prefetch(f->parent + edge[AHEAD].b);
            }
            int32_t a = edge->a, b = edge->b;
            if (selection) {
                const uint8_t *m = selection->member;
                const int32_t *v = selection->potential;
                if ((m && !(m[a] && m[b])) ||
                    (v && (uint32_t)v[b] - (uint32_t)v[a] != (uint32_t)edge->step)) {
                    continue;
                }
                if (selection->label) {
                    const int32_t *small = selection->label;
                    const int64_t *large = selection->label;
                    if (selection->wide ? large[a] != large[b] : small[a] != small[b]) {
                        continue;
                    }
                }
            }
            int32_t na, nb;
            int32_t ra = find(f, a, &na), rb = find(f, b, &nb);
            if (ra != rb) {
                unite(f, ra, rb, na + edge->step - nb);
                (*joined)++;
            }
        }
    }
    return 0;
}

int hx_settled_levels(int64_t count, const Edge *edges, int64_t length, const uint8_t *use,
    const void *labels, int wide, const uint8_t *members, const int32_t *potential, int fd,
    int64_t offset, int32_t *root, int32_t *levels) {
    if (count < 0 || count > INT32_MAX) {
        return -1;
    }
    EdgeSource source = {0};
    Forest f = {
        .parent = root, .potential = levels, .size = malloc(((size_t)count + 1) * sizeof(*f.size))};
    int status = -1;
    if (f.size && open_edges(&source, edges, length, fd, offset)) {
        for (int64_t i = 0; i < count; i++) {
            root[i] = (int32_t)i;
            levels[i] = 0;
            f.size[i] = 1;
        }
        Selection selection = {
            .label = labels, .member = members, .potential = potential, .wide = wide};
        int64_t joined = 0;
        status = join_edges(
            &source, use, &f, labels || members || potential ? &selection : NULL, &joined);
        if (!status) {
            for (int64_t i = 0; i < count; i++) {
                root[i] = find(&f, (int32_t)i, levels + i);
            }
        }
    }
    free(source.buffer);
    free(f.size);
    return status;
}

static int grow_edges(EdgeSource *a, const uint8_t *use, EdgeSource *b, int64_t count,
    int32_t *data, int64_t *stats) {
    int64_t n_attract = a->count, n_repel = b->count;
    Forest f = forest_of(data, count);
    if (!n_repel) {
        return join_edges(a, use, &f, NULL, stats + 3);
    }
    if (n_repel > INT32_MAX / 2)
        return -1;
    int32_t *head = malloc(sizeof(int32_t) * count), *last = malloc(sizeof(int32_t) * count);
    int32_t *length = calloc(count, sizeof(int32_t));
    int32_t *other = malloc(sizeof(int32_t) * (2 * n_repel + 1));
    int32_t *link = malloc(sizeof(int32_t) * (2 * n_repel + 1));
    int status = -1;
    if (!head || !last || !length || !other || !link)
        goto finish;
    for (int64_t i = 0; i < count; i++)
        head[i] = last[i] = -1;
    int32_t used = 0;
    int64_t i = 0, j = 0;
    for (;;) {
        while (use && i < n_attract && !use[i])
            i++;
        if (!read_edges(a, i) || !read_edges(b, j)) {
            goto finish;
        }
        const Edge *attract = i < n_attract ? a->data + (i - a->first) : NULL;
        const Edge *repel = j < n_repel ? b->data + (j - b->first) : NULL;
        if (!attract && !repel) {
            break;
        }
        if (attract && a->end - i > AHEAD) {
            __builtin_prefetch(&f.parent[attract[AHEAD].a]);
            __builtin_prefetch(&f.parent[attract[AHEAD].b]);
        }
        if (repel && b->end - j > AHEAD) {
            __builtin_prefetch(&f.parent[repel[AHEAD].a]);
            __builtin_prefetch(&f.parent[repel[AHEAD].b]);
        }
        if (repel && (!attract || repel->weight >= attract->weight)) {
            int32_t na, nb;
            int32_t ra = find(&f, repel->a, &na), rb = find(&f, repel->b, &nb);
            if (ra == rb) {
                stats[1]++;
                j++;
                continue;
            }
            int32_t ends[2][2] = {{ra, repel->b}, {rb, repel->a}};
            for (int s = 0; s < 2; s++) {
                int32_t r = ends[s][0];
                other[used] = ends[s][1];
                link[used] = -1;
                if (last[r] >= 0)
                    link[last[r]] = used;
                else
                    head[r] = used;
                last[r] = used;
                length[r]++;
                used++;
            }
            stats[0]++;
            j++;
            continue;
        }
        int32_t na, nb;
        int32_t ra = find(&f, attract->a, &na), rb = find(&f, attract->b, &nb);
        if (ra == rb) {
            i++;
            continue;
        }
        int32_t probe = length[ra] <= length[rb] ? ra : rb, target = probe == ra ? rb : ra;
        int forbidden = 0;
        for (int32_t e = head[probe]; e >= 0 && !forbidden; e = link[e]) {
            int32_t n;
            if (find(&f, other[e], &n) == target)
                forbidden = 1;
        }
        if (forbidden) {
            stats[2]++;
            i++;
            continue;
        }
        int32_t big = unite(&f, ra, rb, na + attract->step - nb);
        int32_t small = big == rb ? ra : rb;
        if (head[small] >= 0) {
            if (last[big] >= 0)
                link[last[big]] = head[small];
            else
                head[big] = head[small];
            last[big] = last[small];
            length[big] += length[small];
        }
        stats[3]++;
        i++;
    }
    status = 0;
finish:
    free(head);
    free(last);
    free(length);
    free(other);
    free(link);
    return status;
}

int hx_mutex_sources(const Edge *attract, const uint8_t *use, int64_t n_attract, const Edge *repel,
    int64_t n_repel, int64_t count, int32_t *data, int64_t *stats, int a_fd, int64_t a_offset,
    int b_fd, int64_t b_offset) {
    EdgeSource a = {0}, b = {0};
    int status = -1;
    if (open_edges(&a, attract, n_attract, a_fd, a_offset) &&
        open_edges(&b, repel, n_repel, b_fd, b_offset)) {
        status = grow_edges(&a, use, &b, count, data, stats);
    }
    free(a.buffer);
    free(b.buffer);
    return status;
}

int hx_mutex_grow(const Edge *attract, const uint8_t *use, int64_t n_attract, const Edge *repel,
    int64_t n_repel, int64_t count, int32_t *data, int64_t *stats) {
    return hx_mutex_sources(
        attract, use, n_attract, repel, n_repel, count, data, stats, -1, 0, -1, 0);
}

int32_t hx_dense_roots(const int32_t *root, int64_t nodes, int32_t *rank, int32_t *label) {
    for (int64_t i = 0; i < nodes; i++) {
        if (root[i] < 0 || root[i] >= nodes)
            return -1;
        rank[root[i]] = 1;
    }
    int32_t count = 0;
    for (int64_t i = 0; i < nodes; i++)
        if (rank[i])
            rank[i] = count++;
    for (int64_t i = 0; i < nodes; i++)
        label[i] = rank[root[i]];
    return count;
}

int64_t hx_turn_roots(
    Node *nodes, int64_t count, const int32_t *root, const int32_t *parity, int32_t *balance) {
    for (int64_t i = 0; i < count; i++)
        balance[root[i]] += (parity[i] & 1) ? 1 : -1;
    int64_t changed = 0;
    for (int64_t i = 0; i < count; i++) {
        if ((parity[i] & 1) == (balance[root[i]] > 0))
            continue;
        nodes[i].nx = -nodes[i].nx;
        nodes[i].ny = -nodes[i].ny;
        nodes[i].nz = -nodes[i].nz;
        changed++;
    }
    return changed;
}

void hx_order_members(int64_t count, const int32_t *label, int64_t *cursor, int32_t *order) {
    for (int64_t i = 0; i < count; i++) {
        int32_t group = label[i];
        if (group >= 0)
            order[cursor[group]++] = i;
    }
}
