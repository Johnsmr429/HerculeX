
#include <dispatch/dispatch.h>
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

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} Edge;

static inline float unit(int8_t v) { return v / 127.0f; }

// Mutually nearest sheets of two cells; weight falls with depth mismatch and with ambiguity.
static int64_t match_cells(const Node *nodes, const float *theta, const float *quality, int64_t a0,
                           int64_t a1, int64_t b0, int64_t b1, float sigma, float max_depth,
                           float min_cos, float max_lateral, Edge *out) {
    int na = (int)(a1 - a0), nb = (int)(b1 - b0);
    float depth[MAXN][MAXN];
    for (int i = 0; i < na; i++) for (int j = 0; j < nb; j++) {
        const Node *p = &nodes[a0 + i], *q = &nodes[b0 + j];
        float ax = unit(p->nx), ay = unit(p->ny), az = unit(p->nz);
        float bx = unit(q->nx), by = unit(q->ny), bz = unit(q->nz);
        float la = sqrtf(ax * ax + ay * ay + az * az), lb = sqrtf(bx * bx + by * by + bz * bz);
        float dot = (ax * bx + ay * by + az * bz) / (la * lb);
        float sign = dot < 0 ? -1.0f : 1.0f;
        depth[i][j] = 1e9f;
        if (dot * sign < min_cos) continue;
        float mx = ax / la + sign * bx / lb, my = ay / la + sign * by / lb, mz = az / la + sign * bz / lb;
        float lm = sqrtf(mx * mx + my * my + mz * mz);
        mx /= lm; my /= lm; mz /= lm;
        float dx = q->x - p->x, dy = q->y - p->y, dz = q->z - p->z;
        float along = dx * mx + dy * my + dz * mz;
        float lateral = sqrtf(fmaxf(dx * dx + dy * dy + dz * dz - along * along, 0));
        if (fabsf(along) > max_depth || lateral > max_lateral) continue;
        depth[i][j] = fabsf(along);
    }
    int64_t written = 0;
    for (int i = 0; i < na; i++) {
        int best = -1;
        float m1 = 1e9f, m2 = 1e9f;
        for (int j = 0; j < nb; j++) {
            float d = depth[i][j];
            if (d < m1) { m2 = m1; m1 = d; best = j; }
            else if (d < m2) m2 = d;
        }
        if (best < 0) continue;
        int back = -1;
        float c1 = 1e9f, c2 = 1e9f;
        for (int k = 0; k < na; k++) {
            float d = depth[k][best];
            if (d < c1) { c2 = c1; c1 = d; back = k; }
            else if (d < c2) c2 = d;
        }
        if (back != i) continue;
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
                       const float *theta, const float *quality, float sigma, float max_depth,
                       float min_cos, float max_lateral, Edge *out, int64_t capacity) {
    static const int forward[13][3] = {{0, 0, 1},  {0, 1, -1}, {0, 1, 0},  {0, 1, 1},  {1, -1, -1},
                                       {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0},  {1, 0, 1},
                                       {1, 1, -1}, {1, 1, 0},  {1, 1, 1}};
    int64_t count = 0;
    for (int y = 0; y < GY; y++) for (int x = 0; x < GX; x++) {
        int64_t c = ((int64_t)plane * GY + y) * GX + x;
        if (first[c] == first[c + 1]) continue;
        for (int k = 0; k < 13; k++) {
            int zz = plane + forward[k][0], yy = y + forward[k][1], xx = x + forward[k][2];
            if (zz >= GZ || yy < 0 || yy >= GY || xx < 0 || xx >= GX) continue;
            int64_t d = ((int64_t)zz * GY + yy) * GX + xx;
            if (first[d] == first[d + 1]) continue;
            if (count + MAXN > capacity) return -1;
            count += match_cells(nodes, theta, quality, first[c], first[c + 1], first[d], first[d + 1],
                                 sigma, max_depth, min_cos, max_lateral, out + count);
        }
    }
    return count;
}

void hx_cell_table(const int32_t *cell_of, int64_t count, int32_t *first, int64_t cells) {
    memset(first, 0, sizeof(int32_t) * (size_t)(cells + 1));
    for (int64_t i = 0; i < count; i++) first[cell_of[i] + 1]++;
    for (int64_t c = 0; c < cells; c++) first[c + 1] += first[c];
}

// Runs each strongest first merged into one, an earlier run first on a tie: the order of a stable sort of their concatenation, with no index array.
void hx_merge_edges(const int64_t *runs, const int64_t *lengths, int count, Edge *out) {
    const Edge *head[count];
    int64_t left[count];
    for (int r = 0; r < count; r++) {
        head[r] = (const Edge *)runs[r];
        left[r] = lengths[r];
    }
    for (;;) {
        int best = -1;
        float top = 0;
        for (int r = 0; r < count; r++) {
            if (left[r] && (best < 0 || head[r]->weight > top)) {
                best = r;
                top = head[r]->weight;
            }
        }
        if (best < 0) return;
        *out++ = *head[best]++;
        left[best]--;
    }
}

// A float's bits as an integer that falls as the float rises; both zeros are one key, as they are one value to a sort.
static inline uint32_t falling_key(float weight) {
    if (weight == 0) weight = 0;
    uint32_t bits;
    memcpy(&bits, &weight, sizeof(bits));
    return (bits & 0x80000000u) ? bits : ~(bits | 0x80000000u);
}

// Edges strongest first, ties in their order: the order of a stable sort of their negated weights, by bytes of the key from the lowest up.
void hx_sort_edges(Edge *edges, int64_t count, Edge *scratch) {
    int64_t totals[4][256];
    memset(totals, 0, sizeof(totals));
    for (int64_t e = 0; e < count; e++) {
        uint32_t key = falling_key(edges[e].weight);
        for (int pass = 0; pass < 4; pass++) totals[pass][(key >> (8 * pass)) & 255]++;
    }
    Edge *from = edges, *to = scratch;
    for (int pass = 0; pass < 4; pass++) {
        int64_t at[256], sum = 0;
        for (int d = 0; d < 256; d++) {
            at[d] = sum;
            sum += totals[pass][d];
        }
        for (int64_t e = 0; e < count; e++) to[at[(falling_key(from[e].weight) >> (8 * pass)) & 255]++] = from[e];
        Edge *swap = from;
        from = to;
        to = swap;
    }
}

void hx_turned_steps(const Edge *edges, int64_t count, const Node *nodes, Edge *out) {
    dispatch_apply(EDGE_BLOCKS, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
        int64_t lo = count * (int64_t)block / EDGE_BLOCKS, hi = count * (int64_t)(block + 1) / EDGE_BLOCKS;
        for (int64_t e = lo; e < hi; e++) {
            if (e + AHEAD < hi) {
                __builtin_prefetch(&nodes[edges[e + AHEAD].a]);
                __builtin_prefetch(&nodes[edges[e + AHEAD].b]);
            }
            const Node *p = &nodes[edges[e].a], *q = &nodes[edges[e].b];
            out[e] = edges[e];
            out[e].step = (int32_t)p->nx * q->nx + (int32_t)p->ny * q->ny + (int32_t)p->nz * q->nz < 0;
        }
    });
}

#define EACH_EDGE(values, test) \
    dispatch_apply(EDGE_BLOCKS, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) { \
        int64_t lo = count * (int64_t)block / EDGE_BLOCKS, hi = count * (int64_t)(block + 1) / EDGE_BLOCKS; \
        for (int64_t e = lo; e < hi; e++) { \
            if (e + AHEAD < hi) { \
                __builtin_prefetch(&values[edges[e + AHEAD].a]); \
                __builtin_prefetch(&values[edges[e + AHEAD].b]); \
            } \
            int32_t a = edges[e].a, b = edges[e].b; \
            out[e] = (test); \
        } \
    })

void hx_same_label(const Edge *edges, int64_t count, const int32_t *label, uint8_t *out) { EACH_EDGE(label, label[a] == label[b]); }

void hx_same_label_wide(const Edge *edges, int64_t count, const int64_t *label, uint8_t *out) { EACH_EDGE(label, label[a] == label[b]); }

void hx_potential_agrees(const Edge *edges, int64_t count, const int32_t *potential, uint8_t *out) { EACH_EDGE(potential, potential[b] - potential[a] == edges[e].step); }

// Union-find with integer potentials: n_i = potential[i] + n_parent[i].
typedef struct {
    int32_t *parent, *potential, *next, *tail, *size;
} Forest;

static int32_t find(Forest *f, int32_t i, int32_t *n) {
    int32_t root = i, sum = 0;
    while (f->parent[root] != root) { sum += f->potential[root]; root = f->parent[root]; }
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

static void join(Forest *f, int32_t small, int32_t big, int32_t k) {
    f->parent[small] = big;
    f->potential[small] = k;
    f->next[f->tail[big]] = small;
    f->tail[big] = f->tail[small];
    f->size[big] += f->size[small];
}

void hx_forest_init(int64_t count, int32_t *parent, int32_t *potential, int32_t *next, int32_t *tail, int32_t *size) {
    for (int64_t i = 0; i < count; i++) {
        parent[i] = (int32_t)i;
        potential[i] = 0;
        next[i] = -1;
        tail[i] = (int32_t)i;
        size[i] = 1;
    }
}

void hx_labels(int64_t count, int32_t *parent, int32_t *potential, int32_t *next, int32_t *tail,
               int32_t *size, int32_t *root, int32_t *n) {
    Forest f = {parent, potential, next, tail, size};
    for (int64_t i = 0; i < count; i++) root[i] = find(&f, (int32_t)i, &n[i]);
}

typedef struct {
    int32_t a, b, rank;
} Pair;

typedef struct {
    float depth, lateral;
    int32_t node;
} Hit;

static int compare_hit(const void *p, const void *q) {
    float a = ((const Hit *)p)->depth, b = ((const Hit *)q)->depth;
    return a < b ? -1 : a > b;
}

// Cast each member's outward normal through neighbouring cells; hits close in depth are one sheet, and rank counts sheets.
int64_t hx_radial_pairs(const Node *nodes, const int32_t *first, int GZ, int GY, int GX, int plane, int cell,
                        int plane0, const uint8_t *member, float reach, float tube, float same_sheet, int ranks,
                        Pair *out, int64_t capacity) {
    int64_t count = 0;
    Hit hits[256];
    for (int y = 0; y < GY; y++) for (int x = 0; x < GX; x++) {
        int64_t c = ((int64_t)plane * GY + y) * GX + x;
        for (int64_t i = first[c]; i < first[c + 1]; i++) {
            if (!member[i]) continue;
            const Node *p = &nodes[i];
            float nx = unit(p->nx), ny = unit(p->ny), nz = unit(p->nz);
            float length = sqrtf(nx * nx + ny * ny + nz * nz);
            nx /= length; ny /= length; nz /= length;
            float ex = p->x + reach * nx, ey = p->y + reach * ny, ez = p->z + reach * nz;
            int x0 = (int)floorf((fminf(p->x, ex) - tube) / cell), x1 = (int)floorf((fmaxf(p->x, ex) + tube) / cell);
            int y0 = (int)floorf((fminf(p->y, ey) - tube) / cell), y1 = (int)floorf((fmaxf(p->y, ey) + tube) / cell);
            int z0 = (int)floorf((fminf(p->z, ez) - tube) / cell) - plane0;
            int z1 = (int)floorf((fmaxf(p->z, ez) + tube) / cell) - plane0;
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (z0 < 0) z0 = 0;
            if (x1 >= GX) x1 = GX - 1;
            if (y1 >= GY) y1 = GY - 1;
            if (z1 >= GZ) z1 = GZ - 1;
            int found = 0;
            for (int zz = z0; zz <= z1; zz++) for (int yy = y0; yy <= y1; yy++) for (int xx = x0; xx <= x1; xx++) {
                int64_t d = ((int64_t)zz * GY + yy) * GX + xx;
                for (int64_t j = first[d]; j < first[d + 1]; j++) {
                    if (j == i || !member[j]) continue;
                    const Node *q = &nodes[j];
                    float dx = q->x - p->x, dy = q->y - p->y, dz = q->z - p->z;
                    float along = dx * nx + dy * ny + dz * nz;
                    if (along < same_sheet || along > reach) continue;
                    float lateral = dx * dx + dy * dy + dz * dz - along * along;
                    if (lateral > tube * tube) continue;
                    if (found < 256) hits[found++] = (Hit){along, lateral, (int32_t)j};
                }
            }
            if (!found) continue;
            qsort(hits, found, sizeof(Hit), compare_hit);
            int rank = 0, k = 0;
            while (k < found && rank < ranks) {
                int best = k, end = k + 1;
                while (end < found && hits[end].depth - hits[end - 1].depth < same_sheet) {
                    if (hits[end].lateral < hits[best].lateral) best = end;
                    end++;
                }
                rank++;
                if (count == capacity) return -1;
                out[count++] = (Pair){(int32_t)i, hits[best].node, rank};
                k = end;
            }
        }
    }
    return count;
}

// Edges grouped by the piece of their first node: a counting sort into fresh arrays. label is -1 outside any piece.
void hx_group_edges(const int32_t *a, const int32_t *b, int64_t count, const int32_t *label, int32_t pieces, int64_t *first,
                    int32_t *out_a, int32_t *out_b) {
    memset(first, 0, sizeof(int64_t) * ((size_t)pieces + 1));
    for (int64_t e = 0; e < count; e++) if (label[a[e]] >= 0) first[label[a[e]] + 1]++;
    for (int32_t p = 0; p < pieces; p++) first[p + 1] += first[p];
    int64_t *cursor = malloc(sizeof(int64_t) * ((size_t)pieces + 1));
    memcpy(cursor, first, sizeof(int64_t) * ((size_t)pieces + 1));
    for (int64_t e = 0; e < count; e++) {
        int32_t l = label[a[e]];
        if (l < 0) continue;
        int64_t at = cursor[l]++;
        out_a[at] = a[e];
        out_b[at] = b[e];
    }
    free(cursor);
}

// Mutex watershed: sheet edges attract and rays repel, strongest first; a ray seen before a join forbids it, whatever the winding.
void hx_mutex_grow(const Edge *attract, const uint8_t *use, int64_t n_attract, const Edge *repel, int64_t n_repel, int64_t count, int32_t *parent,
                   int32_t *potential, int32_t *next, int32_t *tail, int32_t *size, int64_t *stats) {
    Forest f = {parent, potential, next, tail, size};
    int32_t *head = malloc(sizeof(int32_t) * count), *last = malloc(sizeof(int32_t) * count);
    int32_t *length = calloc(count, sizeof(int32_t));
    int32_t *other = malloc(sizeof(int32_t) * (2 * n_repel + 1));
    int32_t *link = malloc(sizeof(int32_t) * (2 * n_repel + 1));
    for (int64_t i = 0; i < count; i++) head[i] = last[i] = -1;
    int32_t used = 0;
    int64_t i = 0, j = 0;
    for (;;) {
        while (use && i < n_attract && !use[i]) i++;
        if (i >= n_attract && j >= n_repel) break;
        if (i + AHEAD < n_attract) {
            __builtin_prefetch(&parent[attract[i + AHEAD].a]);
            __builtin_prefetch(&parent[attract[i + AHEAD].b]);
        }
        if (j + AHEAD < n_repel) {
            __builtin_prefetch(&parent[repel[j + AHEAD].a]);
            __builtin_prefetch(&parent[repel[j + AHEAD].b]);
        }
        if (j < n_repel && (i >= n_attract || repel[j].weight >= attract[i].weight)) {
            int32_t na, nb;
            int32_t ra = find(&f, repel[j].a, &na), rb = find(&f, repel[j].b, &nb);
            if (ra == rb) { stats[1]++; j++; continue; }
            int32_t ends[2][2] = {{ra, repel[j].b}, {rb, repel[j].a}};
            for (int s = 0; s < 2; s++) {
                int32_t r = ends[s][0];
                other[used] = ends[s][1];
                link[used] = -1;
                if (last[r] >= 0) link[last[r]] = used; else head[r] = used;
                last[r] = used;
                length[r]++;
                used++;
            }
            stats[0]++;
            j++;
            continue;
        }
        int32_t na, nb;
        int32_t ra = find(&f, attract[i].a, &na), rb = find(&f, attract[i].b, &nb);
        if (ra == rb) { i++; continue; }
        int32_t probe = length[ra] <= length[rb] ? ra : rb, target = probe == ra ? rb : ra;
        int forbidden = 0;
        for (int32_t e = head[probe]; e >= 0 && !forbidden; e = link[e]) {
            int32_t n;
            if (find(&f, other[e], &n) == target) forbidden = 1;
        }
        if (forbidden) { stats[2]++; i++; continue; }
        int32_t k = na + attract[i].step - nb;
        int32_t small = ra, big = rb, shift = -k;
        if (size[ra] > size[rb]) { small = rb; big = ra; shift = k; }
        join(&f, small, big, shift);
        if (head[small] >= 0) {
            if (last[big] >= 0) link[last[big]] = head[small]; else head[big] = head[small];
            last[big] = last[small];
            length[big] += length[small];
        }
        stats[3]++;
        i++;
    }
    free(head); free(last); free(length); free(other); free(link);
}
