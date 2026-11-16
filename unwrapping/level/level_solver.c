// Exact minimiser of sum_t f_t(x[q_t] - x[p_t] - k_t) over integer x, each f_t convex with one kink: up*max(r,0) + dn*max(-r,0).

#include <math.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TERMINAL (-1)
#define FREE (-3)
#define NO_KINK 0x7fffffff
#define MOST_COST (1 << 24)

typedef struct {
    int64_t terminal;
    int32_t first, last, parent;
    uint8_t sink, queued;
} Node;

typedef struct { int32_t head, sister; } Arc;

typedef struct {
    Node *nodes;
    const Arc *arcs;
    int32_t *cap;
    int32_t count, source_side, sink_side;
    int64_t flow;
    int32_t *label, *fifo, *current, *ring;
    int32_t *cand, workers, relabels;
} Graph;

// Maximum flow by push-relabel from what excess and deficit the nodes hold, distances renewed by breadth-first search from the deficits;
static void pr_relabel_all(Graph *g, int32_t *label, int32_t *queue) {
    Node *N = g->nodes;
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    int32_t n = g->count, head = 0, end = 0;
    for (int32_t i = 0; i < n; i++) {
        label[i] = N[i].terminal < 0 ? 0 : n;
        if (N[i].terminal < 0) queue[end++] = i;
    }
    while (head < end) {
        int32_t v = queue[head++];
        for (int32_t e = N[v].first; e < N[v].last; e++) {
            int32_t u = A[e].head;
            if (label[u] == n && C[A[e].sister] > 0) {
                label[u] = label[v] + 1;
                queue[end++] = u;
            }
        }
    }
}

#define BFS_CHUNKS 64
#define BFS_PARALLEL_LEVEL 4096

// The same distances by levels: every node first reached from level d gets d + 1, whichever thread reaches it,
// so the labels are the serial breadth-first search's exactly; only the order nodes sit in the queue differs.
static void pr_relabel_parallel(Graph *g, int32_t *label, int32_t *queue) {
    Node *N = g->nodes;
    const Arc *A = g->arcs;
    int32_t *C = g->cap, *cand = g->cand;
    int32_t n = g->count, end = 0;
    for (int32_t i = 0; i < n; i++) {
        label[i] = N[i].terminal < 0 ? 0 : n;
        if (N[i].terminal < 0) queue[end++] = i;
    }
    int32_t lo = 0, hi = end, depth = 0;
    int64_t offset_store[BFS_CHUNKS + 1];
    int32_t found_store[BFS_CHUNKS];
    int64_t *offset = offset_store;
    int32_t *found = found_store;
    dispatch_queue_t pool = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    while (lo < hi) {
        int32_t size = hi - lo, next = depth + 1;
        if (size < BFS_PARALLEL_LEVEL) {
            for (int32_t at = lo; at < hi; at++) {
                int32_t v = queue[at];
                for (int32_t e = N[v].first; e < N[v].last; e++) {
                    int32_t u = A[e].head;
                    if (label[u] == n && C[A[e].sister] > 0) { label[u] = next; queue[end++] = u; }
                }
            }
        } else {
            int32_t chunk = (size + BFS_CHUNKS - 1) / BFS_CHUNKS;
            const int32_t *frontier = queue + lo;
            dispatch_apply(BFS_CHUNKS, pool, ^(size_t c) {
                int64_t arcs = 0;
                int32_t a = (int32_t)c * chunk, b = a + chunk < size ? a + chunk : size;
                for (int32_t i = a; i < b; i++) arcs += N[frontier[i]].last - N[frontier[i]].first;
                offset[c + 1] = arcs;
            });
            offset[0] = 0;
            for (int c = 0; c < BFS_CHUNKS; c++) offset[c + 1] += offset[c];
            dispatch_apply(BFS_CHUNKS, pool, ^(size_t c) {
                int32_t a = (int32_t)c * chunk, b = a + chunk < size ? a + chunk : size, got = 0;
                int32_t *out = cand + offset[c];
                for (int32_t i = a; i < b; i++) {
                    int32_t v = frontier[i];
                    for (int32_t e = N[v].first; e < N[v].last; e++) {
                        int32_t u = A[e].head;
                        if (__atomic_load_n(&label[u], __ATOMIC_RELAXED) != n || C[A[e].sister] <= 0) continue;
                        int32_t expected = n;
                        if (__atomic_compare_exchange_n(&label[u], &expected, next, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) out[got++] = u;
                    }
                }
                found[c] = got;
            });
            for (int c = 0; c < BFS_CHUNKS; c++) {
                memcpy(queue + end, cand + offset[c], sizeof(int32_t) * (size_t)found[c]);
                end += found[c];
            }
        }
        lo = hi;
        hi = end;
        depth++;
    }
}

static void relabel(Graph *g, int32_t *label, int32_t *queue) {
    if (g->workers > 1 && g->cand) pr_relabel_parallel(g, label, queue);
    else pr_relabel_all(g, label, queue);
}

// A completed flow needs no cut. Pass -1 when the caller needs the partition on every solve.
static void maxflow(Graph *g, int64_t offered) {
    Node *N = g->nodes;
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    int32_t n = g->count;
    int32_t *label = g->label, *fifo = g->fifo, *current = g->current, *ring = g->ring;
    g->flow = 0;
    if (!offered) return;
    int64_t work = 0, arcs = (int64_t)(N[n - 1].last);
    relabel(g, label, fifo);
    int32_t head = 0, tail = 0, held = 0;
    for (int32_t i = 0; i < n; i++) {
        current[i] = N[i].first;
        N[i].queued = 0;
    }
    for (int32_t i = 0; i < n; i++) {
        if (N[i].terminal > 0 && label[i] < n) {
            ring[tail] = i;
            if (++tail > n) tail = 0;
            held++;
            N[i].queued = 1;
        }
    }
    while (held) {
        int32_t u = ring[head];
        if (++head > n) head = 0;
        held--;
        N[u].queued = 0;
        while (N[u].terminal > 0 && label[u] < n) {
            int32_t e = current[u];
            if (e == N[u].last) {
                int32_t lowest = n;
                for (int32_t f = N[u].first; f < N[u].last; f++)
                    if (C[f] > 0 && label[A[f].head] + 1 < lowest) lowest = label[A[f].head] + 1;
                work += N[u].last - N[u].first + 12;
                label[u] = lowest;
                current[u] = N[u].first;
                continue;
            }
            int32_t v = A[e].head;
            if (C[e] > 0 && label[u] == label[v] + 1) {
                int64_t delta = N[u].terminal < C[e] ? N[u].terminal : C[e];
                C[e] -= (int32_t)delta;
                C[A[e].sister] += (int32_t)delta;
                N[u].terminal -= delta;
                int64_t before = N[v].terminal;
                N[v].terminal += delta;
                if (before < 0) g->flow += -before < delta ? -before : delta;
                if (N[v].terminal > 0 && !N[v].queued && label[v] < n) {
                    ring[tail] = v;
                    if (++tail > n) tail = 0;
                    held++;
                    N[v].queued = 1;
                }
                if (N[u].terminal == 0) break;
            } else {
                current[u] = e + 1;
            }
        }
        if (work * g->relabels > arcs) {
            work = 0;
            relabel(g, label, fifo);
            for (int32_t i = 0; i < n; i++) current[i] = N[i].first;
            head = tail = held = 0;
            for (int32_t i = 0; i < n; i++) {
                N[i].queued = 0;
                if (N[i].terminal > 0 && label[i] < n) {
                    ring[tail] = i;
                    if (++tail > n) tail = 0;
                    held++;
                    N[i].queued = 1;
                }
            }
        }
    }
    if (g->flow == offered) return;
    for (int32_t i = 0; i < n; i++) { N[i].parent = FREE; N[i].sink = 0; N[i].queued = 0; }
    int32_t end = 0, at = 0;
    for (int32_t i = 0; i < n; i++) if (N[i].terminal > 0) { N[i].parent = TERMINAL; fifo[end++] = i; }
    while (at < end) {
        int32_t u = fifo[at++];
        for (int32_t e = N[u].first; e < N[u].last; e++) {
            int32_t v = A[e].head;
            if (C[e] > 0 && N[v].parent == FREE) { N[v].parent = TERMINAL; fifo[end++] = v; }
        }
    }
    g->source_side = end;
    end = at = 0;
    for (int32_t i = 0; i < n; i++) if (N[i].terminal < 0) { N[i].parent = TERMINAL; N[i].sink = 1; fifo[end++] = i; }
    while (at < end) {
        int32_t v = fifo[at++];
        for (int32_t e = N[v].first; e < N[v].last; e++) {
            int32_t u = A[e].head;
            if (C[A[e].sister] > 0 && N[u].parent == FREE) { N[u].parent = TERMINAL; N[u].sink = 1; fifo[end++] = u; }
        }
    }
    g->sink_side = end;
}

static inline int64_t cost(int64_t r, int64_t up, int64_t dn) { return r >= 0 ? up * r : -dn * r; }

int64_t hx_lnat_energy(int64_t terms, const int32_t *p, const int32_t *q, const int32_t *k, const int32_t *up, const int32_t *dn, const int32_t *x) {
    int64_t total = 0;
    for (int64_t t = 0; t < terms; t++) total += cost((int64_t)x[q[t]] - x[p[t]] - k[t], up[t], dn[t]);
    return total;
}

// The proof of a finished solve: every flow a slope of its term at x, and none left over at any node.
static int proven(Graph *g, int64_t count, int64_t terms, const int32_t *p, const int32_t *q, const int32_t *k, const int32_t *up, const int32_t *dn, const int32_t *x, const int32_t *forward) {
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    int64_t *balance = calloc((size_t)count, sizeof(int64_t));
    int good = balance != NULL;
    for (int64_t t = 0; good && t < terms; t++) {
        int32_t r = x[q[t]] - x[p[t]] - k[t], a = forward[t], b = A[a].sister;
        int64_t phi = r > 0 ? up[t] : r < 0 ? -(int64_t)dn[t] : C[a] - dn[t];
        good = r != 0 || (C[a] >= 0 && C[b] >= 0 && C[a] + C[b] == (int64_t)up[t] + dn[t]);
        balance[p[t]] += phi;
        balance[q[t]] -= phi;
    }
    for (int64_t i = 0; good && i < count; i++) good = balance[i] == 0;
    free(balance);
    return good;
}

static int64_t unit_steps(Graph *g, int64_t count, int64_t terms, const int32_t *p, const int32_t *q, const int32_t *k, const int32_t *up, const int32_t *dn, int32_t *x, const int32_t *forward, const int32_t *term_of, int64_t cuts, int64_t *moved) {
    Node *N = g->nodes;
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    int32_t *residual = malloc(sizeof(int32_t) * (size_t)(terms + 1));
    int32_t *members = malloc(sizeof(int32_t) * (size_t)(count + 1));
    for (int64_t i = 0; i < count; i++) N[i].terminal = 0;
    for (int64_t t = 0; t < terms; t++) {
        int32_t a = forward[t], b = A[a].sister;
        int32_t r = x[q[t]] - x[p[t]] - k[t];
        residual[t] = r;
        int64_t phi = r > 0 ? up[t] : r < 0 ? -(int64_t)dn[t] : 0;
        N[p[t]].terminal += phi;
        N[q[t]].terminal -= phi;
        C[a] = r == 0 ? dn[t] : 0;
        C[b] = r == 0 ? up[t] : 0;
    }
    int64_t offered = 0;
    for (int64_t i = 0; i < count; i++)
        if (N[i].terminal > 0) offered += N[i].terminal;
    maxflow(g, offered);
    for (;;) {
        int64_t excess = offered - g->flow;
        cuts++;
        if (!excess) {
            if (!proven(g, count, terms, p, q, k, up, dn, x, forward)) cuts = -2;
            break;
        }
        uint8_t sink = g->sink_side < g->source_side;
        int32_t direction = sink ? -1 : 1, members_count = 0, jump = NO_KINK;
        for (int64_t i = 0; i < count; i++)
            if (N[i].parent != FREE && N[i].sink == sink) members[members_count++] = (int32_t)i;
        for (int32_t m = 0; m < members_count; m++) {
            int32_t i = members[m];
            for (int32_t e = N[i].first; e < N[i].last; e++) {
                int32_t j = A[e].head;
                if (N[j].parent != FREE && N[j].sink == sink) continue;
                int32_t t = term_of[e], r = residual[t], shift = q[t] == i ? direction : -direction;
                if (r != 0 && (r > 0) != (shift > 0) && abs(r) < jump) jump = abs(r);
            }
        }
        if (jump == NO_KINK) break;
        for (int32_t m = 0; m < members_count; m++) {
            int32_t i = members[m];
            for (int32_t e = N[i].first; e < N[i].last; e++) {
                int32_t j = A[e].head;
                if (N[j].parent != FREE && N[j].sink == sink) continue;
                int32_t t = term_of[e], r = residual[t], shift = q[t] == i ? direction : -direction;
                int32_t a = forward[t], b = A[a].sister;
                if (r == 0) {
                    C[a] = C[b] = 0;
                } else if (r + shift * jump == 0) {
                    int64_t phi = r > 0 ? up[t] : -(int64_t)dn[t];
                    C[a] = (int32_t)(phi + dn[t]);
                    C[b] = (int32_t)(up[t] - phi);
                }
                residual[t] = r + shift * jump;
            }
            x[i] += direction * jump;
            *moved += jump;
        }
        int64_t routed = g->flow;
        maxflow(g, excess);
        g->flow += routed;
    }
    free(residual);
    free(members);
    return cuts;
}

typedef struct {
    int64_t count, terms;
    Node *nodes;
    Arc *arcs;
    int32_t *forward, *term_of;
} Shape;

static void shape_open(Shape *s, int64_t count, int64_t terms, const int32_t *p, const int32_t *q) {
    s->count = count;
    s->terms = terms;
    s->nodes = calloc((size_t)count + 1, sizeof(Node));
    s->arcs = malloc(sizeof(Arc) * (size_t)(2 * terms + 1));
    s->forward = malloc(sizeof(int32_t) * (size_t)(terms + 1));
    s->term_of = malloc(sizeof(int32_t) * (size_t)(2 * terms + 1));
    int32_t *degree = calloc((size_t)count + 1, sizeof(int32_t));
    for (int64_t t = 0; t < terms; t++) { degree[p[t]]++; degree[q[t]]++; }
    int32_t at = 0;
    for (int64_t i = 0; i < count; i++) {
        s->nodes[i].first = at;
        at += degree[i];
        s->nodes[i].last = s->nodes[i].first;
    }
    free(degree);
    for (int64_t t = 0; t < terms; t++) {
        int32_t a = s->nodes[p[t]].last++, b = s->nodes[q[t]].last++;
        s->arcs[a].head = q[t];
        s->arcs[b].head = p[t];
        s->arcs[a].sister = b;
        s->arcs[b].sister = a;
        s->forward[t] = a;
        s->term_of[a] = s->term_of[b] = (int32_t)t;
    }
}

static void shape_close(Shape *s) {
    free(s->nodes);
    free(s->arcs);
    free(s->forward);
    free(s->term_of);
}

static void graph_open(Graph *g, const Shape *s) {
    memset(g, 0, sizeof(Graph));
    g->count = (int32_t)s->count;
    g->nodes = malloc(sizeof(Node) * (size_t)(s->count + 1));
    memcpy(g->nodes, s->nodes, sizeof(Node) * (size_t)s->count);
    g->arcs = s->arcs;
    g->cap = malloc(sizeof(int32_t) * (size_t)(2 * s->terms + 1));
    g->label = malloc(sizeof(int32_t) * (size_t)(s->count + 1));
    g->fifo = malloc(sizeof(int32_t) * (size_t)(s->count + 1));
    g->current = malloc(sizeof(int32_t) * (size_t)(s->count + 1));
    g->ring = malloc(sizeof(int32_t) * (size_t)(s->count + 1));
    g->cand = NULL;
    g->workers = 1;
    g->relabels = 2;
}

static void graph_close(Graph *g) {
    free(g->nodes);
    free(g->cap);
    free(g->label);
    free(g->fifo);
    free(g->current);
    free(g->ring);
    free(g->cand);
}

// x improved in place to a global minimum; step starts at first_step and halves, each scale run until neither direction helps.
int64_t hx_level_minimize(int64_t count, int64_t terms, const int32_t *p, const int32_t *q, const int32_t *k, const int32_t *up, const int32_t *dn, int32_t *x, int32_t first_step, int64_t *log) {
    Shape s;
    shape_open(&s, count, terms, p, q);
    Graph solve, *g = &solve;
    graph_open(g, &s);
    g->cand = malloc(sizeof(int32_t) * (size_t)(2 * terms + 1));
    g->workers = g->cand ? 8 : 1;
    g->relabels = g->cand ? 4 : 2;
    int64_t cuts = 0, moved = 0;
    for (int32_t step = first_step; step >= 2; step /= 2) {
        int quiet = 0;
        for (int direction = 1; quiet < 2; direction = -direction) {
            int32_t delta = direction * step;
            for (int64_t i = 0; i < count; i++) g->nodes[i].terminal = 0;
            int64_t offered = 0;
            for (int64_t t = 0; t < terms; t++) {
                int64_t r = (int64_t)x[q[t]] - x[p[t]] - k[t];
                int64_t e01 = cost(r + delta, up[t], dn[t]) - cost(r, up[t], dn[t]);
                int64_t e10 = cost(r - delta, up[t], dn[t]) - cost(r, up[t], dn[t]);
                int64_t u = 0;
                if (e01 < 0) u = e01;
                else if (e10 < 0) u = -e10;
                g->nodes[q[t]].terminal -= u;
                g->nodes[p[t]].terminal += u;
                int32_t a = s.forward[t];
                g->cap[a] = (int32_t)(e10 + u);
                g->cap[g->arcs[a].sister] = (int32_t)(e01 - u);
            }
            for (int64_t i = 0; i < count; i++)
                if (g->nodes[i].terminal > 0) offered += g->nodes[i].terminal;
            maxflow(g, offered);
            cuts++;
            if (g->flow - offered >= 0) { quiet++; continue; }
            quiet = 0;
            for (int64_t i = 0; i < count; i++) {
                if (g->nodes[i].parent != FREE && !g->nodes[i].sink) {
                    x[i] += delta;
                    moved++;
                }
            }
        }
    }
    cuts = unit_steps(g, count, terms, p, q, k, up, dn, x, s.forward, s.term_of, cuts, &moved);
    log[0] = cuts;
    log[1] = moved;
    log[2] = hx_lnat_energy(terms, p, q, k, up, dn, x);
    graph_close(g);
    shape_close(&s);
    return cuts;
}

int64_t hx_level_two_labels(int64_t count, int64_t terms, const int32_t *a, const int32_t *b, const int32_t *weight, const int64_t *unary, uint8_t *label) {
    Shape s;
    shape_open(&s, count, terms, a, b);
    Graph solve, *g = &solve;
    graph_open(g, &s);
    for (int64_t t = 0; t < terms; t++) {
        int32_t arc = s.forward[t];
        g->cap[arc] = weight[t];
        g->cap[g->arcs[arc].sister] = weight[t];
    }
    for (int64_t i = 0; i < count; i++) g->nodes[i].terminal = unary[i];
    maxflow(g, -1);
    for (int64_t i = 0; i < count; i++) label[i] = g->nodes[i].parent != FREE && !g->nodes[i].sink;
    int64_t flow = g->flow;
    graph_close(g);
    shape_close(&s);
    return flow;
}

typedef struct { int32_t q, k, element; } Entry;

static int compare_entries(const void *a, const void *b) {
    const Entry *x = a, *y = b;
    if (x->q != y->q) return x->q < y->q ? -1 : 1;
    if (x->k != y->k) return x->k < y->k ? -1 : 1;
    return 0;
}

typedef struct { int32_t a, b; float weight; int32_t step; } Edge;
typedef struct { int32_t a, b, rank; } Pair;

typedef struct { int64_t terms; int32_t *p, *q, *k; } Terms;

static inline int element_ends(int64_t e, const int32_t *patch, const int32_t *inner, const Edge *edges, int64_t edge_count, const Pair *rays, const float *ray_weight, const float *theta,
                               int32_t chirality, int32_t *p, int32_t *q, int32_t *k, int *swapped) {
    int32_t a, b, offset;
    if (e < edge_count) {
        a = edges[e].a; b = edges[e].b;
        if (patch[a] == patch[b]) return 0;
        offset = chirality * (edges[e].step + inner[a] - inner[b]);
    } else {
        int64_t r = e - edge_count;
        a = rays[r].a; b = rays[r].b;
        if (patch[a] == patch[b] || ray_weight[r] <= 0) return 0;
        int32_t turns = -(int32_t)rint(((double)theta[b] - theta[a]) / (2 * 3.14159265358979323846));
        offset = chirality * (turns + inner[a] - inner[b]) + 1;
    }
    *swapped = patch[a] > patch[b];
    *p = *swapped ? patch[b] : patch[a];
    *q = *swapped ? patch[a] : patch[b];
    *k = *swapped ? -offset : offset;
    return 1;
}

int64_t hx_level_terms(int64_t count, const int32_t *patch, const int32_t *inner, const Edge *edges, int64_t edge_count, const Pair *rays, const float *ray_weight, int64_t ray_count,
                       const float *theta, int32_t chirality, int32_t *element_term, uint64_t *handle) {
    int64_t elements = edge_count + ray_count;
    int64_t *start = calloc((size_t)count + 2, sizeof(int64_t));
    int32_t p, q, k;
    int swapped;
    for (int64_t e = 0; e < elements; e++)
        if (element_ends(e, patch, inner, edges, edge_count, rays, ray_weight, theta, chirality, &p, &q, &k, &swapped)) start[p + 1]++;
    for (int64_t i = 0; i < count; i++) start[i + 1] += start[i];
    int64_t used = start[count];
    Entry *entries = malloc(sizeof(Entry) * (size_t)(used + 1));
    int64_t *fill = malloc(sizeof(int64_t) * (size_t)(count + 1));
    memcpy(fill, start, sizeof(int64_t) * (size_t)(count + 1));
    for (int64_t e = 0; e < elements; e++) {
        element_term[e] = -1;
        if (!element_ends(e, patch, inner, edges, edge_count, rays, ray_weight, theta, chirality, &p, &q, &k, &swapped)) continue;
        Entry *out = &entries[fill[p]++];
        out->q = q;
        out->k = k;
        out->element = (int32_t)e;
    }
    free(fill);
    int64_t terms = 0;
    for (int64_t i = 0; i < count; i++) {
        int64_t from = start[i], to = start[i + 1];
        if (to - from > 1) qsort(entries + from, (size_t)(to - from), sizeof(Entry), compare_entries);
        for (int64_t j = from; j < to; j++) terms += j == from || entries[j].q != entries[j - 1].q || entries[j].k != entries[j - 1].k;
    }
    Terms *list = malloc(sizeof(Terms));
    list->terms = terms;
    list->p = malloc(sizeof(int32_t) * (size_t)(terms + 1));
    list->q = malloc(sizeof(int32_t) * (size_t)(terms + 1));
    list->k = malloc(sizeof(int32_t) * (size_t)(terms + 1));
    int64_t t = -1;
    for (int64_t i = 0; i < count; i++) {
        for (int64_t j = start[i]; j < start[i + 1]; j++) {
            if (j == start[i] || entries[j].q != entries[j - 1].q || entries[j].k != entries[j - 1].k) {
                t++;
                list->p[t] = (int32_t)i;
                list->q[t] = entries[j].q;
                list->k[t] = entries[j].k;
            }
            int32_t e = entries[j].element;
            element_ends(e, patch, inner, edges, edge_count, rays, ray_weight, theta, chirality, &p, &q, &k, &swapped);
            element_term[e] = swapped ? (int32_t)(-2 - t) : (int32_t)t;
        }
    }
    free(start);
    free(entries);
    *handle = (uint64_t)list;
    return terms;
}

void hx_level_terms_take(uint64_t handle, int32_t *p, int32_t *q, int32_t *k) {
    Terms *list = (Terms *)handle;
    memcpy(p, list->p, sizeof(int32_t) * (size_t)list->terms);
    memcpy(q, list->q, sizeof(int32_t) * (size_t)list->terms);
    memcpy(k, list->k, sizeof(int32_t) * (size_t)list->terms);
    free(list->p); free(list->q); free(list->k); free(list);
}

static inline void add_element(float *sum_up, float *sum_dn, int32_t v, float up, float dn) {
    if (v >= 0) {
        sum_up[v] += up;
        sum_dn[v] += dn;
    } else if (v < -1) {
        sum_up[-2 - v] += dn;
        sum_dn[-2 - v] += up;
    }
}

void hx_level_weights(int64_t terms, const Edge *edges, int64_t edge_count, const float *ray_weight, int64_t ray_count, float hidden, const int32_t *element_term, int32_t scale, int32_t *up, int32_t *dn) {
    float *sum_up = calloc((size_t)terms + 1, sizeof(float)), *sum_dn = calloc((size_t)terms + 1, sizeof(float));
    for (int64_t e = 0; e < edge_count; e++) add_element(sum_up, sum_dn, element_term[e], edges[e].weight, edges[e].weight);
    for (int64_t r = 0; r < ray_count; r++) add_element(sum_up, sum_dn, element_term[edge_count + r], ray_weight[r] * hidden, ray_weight[r]);
    for (int64_t t = 0; t < terms; t++) {
        int64_t a = llrintf(sum_up[t] * scale), b = llrintf(sum_dn[t] * scale);
        up[t] = a < 1 ? 1 : a > MOST_COST ? MOST_COST : (int32_t)a;
        dn[t] = b < 1 ? 1 : b > MOST_COST ? MOST_COST : (int32_t)b;
    }
    free(sum_up);
    free(sum_dn);
}

static int32_t find(int32_t *parent, int32_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

typedef struct { float x, y, z; int8_t nx, ny, nz; uint8_t prominence, width, coherence; uint16_t pad; } SheetNode;

static inline int agreeing(const SheetNode *nodes, const Edge *e) {
    const SheetNode *a = &nodes[e->a], *b = &nodes[e->b];
    return (int32_t)a->nx * b->nx + (int32_t)a->ny * b->ny + (int32_t)a->nz * b->nz >= 0;
}

static inline int one_block(const SheetNode *nodes, const Edge *e, float side) {
    const SheetNode *a = &nodes[e->a], *b = &nodes[e->b];
    return floorf(a->x / side) == floorf(b->x / side) && floorf(a->y / side) == floorf(b->y / side) && floorf(a->z / side) == floorf(b->z / side);
}

// Nodes joined by agreeing edges inside one block of the given side, numbered from zero. Returns how many.
int64_t hx_level_blocks(int64_t count, const SheetNode *nodes, const int64_t *runs, const int64_t *lengths, int64_t run_count, float side, int32_t *block) {
    int32_t *parent = malloc(sizeof(int32_t) * (size_t)(count + 1));
    for (int64_t i = 0; i < count; i++) parent[i] = (int32_t)i;
    for (int64_t r = 0; r < run_count; r++) {
        const Edge *run = (const Edge *)runs[r];
        for (int64_t e = 0; e < lengths[r]; e++) {
            if (!agreeing(nodes, &run[e]) || !one_block(nodes, &run[e], side)) continue;
            int32_t i = find(parent, run[e].a), j = find(parent, run[e].b);
            if (i != j) parent[i] = j;
        }
    }
    int32_t blocks = 0;
    for (int64_t i = 0; i < count; i++)
        if (parent[i] == i) block[i] = blocks++;
    for (int64_t i = 0; i < count; i++) block[i] = block[find(parent, (int32_t)i)];
    free(parent);
    return blocks;
}

// Agreeing edges between blocks, one pair each with weights summed, for hx_level_terms_take with the weight as k.
int64_t hx_level_block_pairs(int64_t blocks, const int32_t *block, const SheetNode *nodes, const int64_t *runs, const int64_t *lengths, int64_t run_count, int32_t units, uint64_t *handle) {
    int64_t *start = calloc((size_t)blocks + 2, sizeof(int64_t));
    for (int64_t r = 0; r < run_count; r++) {
        const Edge *run = (const Edge *)runs[r];
        for (int64_t e = 0; e < lengths[r]; e++) {
            int32_t i = block[run[e].a], j = block[run[e].b];
            if (i != j && agreeing(nodes, &run[e])) start[(i < j ? i : j) + 1]++;
        }
    }
    for (int64_t i = 0; i < blocks; i++) start[i + 1] += start[i];
    Entry *entries = malloc(sizeof(Entry) * (size_t)(start[blocks] + 1));
    int64_t *fill = malloc(sizeof(int64_t) * (size_t)(blocks + 1));
    memcpy(fill, start, sizeof(int64_t) * (size_t)(blocks + 1));
    for (int64_t r = 0; r < run_count; r++) {
        const Edge *run = (const Edge *)runs[r];
        for (int64_t e = 0; e < lengths[r]; e++) {
            int32_t i = block[run[e].a], j = block[run[e].b];
            if (i == j || !agreeing(nodes, &run[e])) continue;
            int64_t weight = llrintf(run[e].weight * (float)units);
            Entry *out = &entries[fill[i < j ? i : j]++];
            out->q = i < j ? j : i;
            out->k = weight < 1 ? 1 : (int32_t)weight;
            out->element = 0;
        }
    }
    free(fill);
    int64_t pairs = 0;
    for (int64_t i = 0; i < blocks; i++) {
        if (start[i + 1] - start[i] > 1) qsort(entries + start[i], (size_t)(start[i + 1] - start[i]), sizeof(Entry), compare_entries);
        for (int64_t j = start[i]; j < start[i + 1]; j++) pairs += j == start[i] || entries[j].q != entries[j - 1].q;
    }
    Terms *list = malloc(sizeof(Terms));
    list->terms = pairs;
    list->p = malloc(sizeof(int32_t) * (size_t)(pairs + 1));
    list->q = malloc(sizeof(int32_t) * (size_t)(pairs + 1));
    list->k = malloc(sizeof(int32_t) * (size_t)(pairs + 1));
    int64_t t = -1, sum = 0;
    for (int64_t i = 0; i < blocks; i++) {
        for (int64_t j = start[i]; j < start[i + 1]; j++) {
            if (j == start[i] || entries[j].q != entries[j - 1].q) {
                t++;
                sum = 0;
                list->p[t] = (int32_t)i;
                list->q[t] = entries[j].q;
            }
            sum += entries[j].k;
            list->k[t] = sum > 1000000000 ? 1000000000 : (int32_t)sum;
        }
    }
    free(start);
    free(entries);
    *handle = (uint64_t)list;
    return pairs;
}

int64_t hx_level_domains(int64_t count, const SheetNode *nodes, const int64_t *runs, const int64_t *lengths, int64_t run_count, const uint8_t *flip, uint8_t *keep, int64_t *log) {
    int32_t *parent = malloc(sizeof(int32_t) * (size_t)(count + 1));
    int64_t *gain = calloc((size_t)count + 1, sizeof(int64_t));
    for (int64_t i = 0; i < count; i++) parent[i] = (int32_t)i;
    for (int64_t r = 0; r < run_count; r++) {
        const Edge *run = (const Edge *)runs[r];
        for (int64_t e = 0; e < lengths[r]; e++) {
            if (!(flip[run[e].a] && flip[run[e].b] && agreeing(nodes, &run[e]))) continue;
            int32_t i = find(parent, run[e].a), j = find(parent, run[e].b);
            if (i != j) parent[i] = j;
        }
    }
    int64_t opposed = 0;
    for (int64_t r = 0; r < run_count; r++) {
        const Edge *run = (const Edge *)runs[r];
        for (int64_t e = 0; e < lengths[r]; e++) {
            int agree = agreeing(nodes, &run[e]);
            opposed += !agree;
            int32_t i = flip[run[e].a] ? find(parent, run[e].a) : -1, j = flip[run[e].b] ? find(parent, run[e].b) : -1;
            if (i == j) continue;
            if (i >= 0) gain[i] += agree ? -1 : 1;
            if (j >= 0) gain[j] += agree ? -1 : 1;
        }
    }
    int64_t kept = 0;
    for (int64_t i = 0; i < count; i++) {
        keep[i] = 0;
        if (!flip[i]) continue;
        int32_t r = find(parent, (int32_t)i);
        keep[i] = gain[r] >= 0;
        kept += keep[i] && r == i;
    }
    free(parent);
    free(gain);
    log[0] = opposed;
    return kept;
}

static inline uint64_t next_random(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

typedef struct {
    int64_t count, terms, edge_count, ray_count;
    const int32_t *p, *q, *k, *element_term, *x0;
    const Shape *shape;
    const Edge *edges;
    const float *ray_weight;
    float hidden;
    int32_t draws, scale, boot, join_shape;
    uint64_t seed;
    int32_t next;
    int32_t *votes;
    int64_t failed, cuts;
} Work;

typedef union { float sum; int32_t units; } Cost;

static inline void add_cost(Cost *up, Cost *dn, int32_t v, float a, float b) {
    if (v >= 0) {
        up[v].sum += a;
        dn[v].sum += b;
    } else if (v < -1) {
        up[-2 - v].sum += b;
        dn[-2 - v].sum += a;
    }
}

static inline float exponential(uint64_t *state) {
    return -logf(((float)(next_random(state) >> 40) + 0.5f) * (1.0f / 16777216.0f));
}

static void *draw_worker(void *argument) {
    Work *w = argument;
    Graph g;
    graph_open(&g, w->shape);
    Cost *up = malloc(sizeof(Cost) * (size_t)(w->terms + 1));
    Cost *dn = malloc(sizeof(Cost) * (size_t)(w->terms + 1));
    int32_t *x = malloc(sizeof(int32_t) * (size_t)(w->count + 1));
    for (;;) {
        int32_t draw = __atomic_fetch_add(&w->next, 1, __ATOMIC_RELAXED);
        if (draw >= w->draws) break;
        uint64_t state = w->seed + 0x632be59bd9b4e019ULL * (uint64_t)(draw + 1);
        memset(up, 0, sizeof(Cost) * (size_t)w->terms);
        memset(dn, 0, sizeof(Cost) * (size_t)w->terms);
        for (int64_t e = 0; e < w->edge_count; e++) {
            int32_t v = w->element_term[e];
            if (v == -1) continue;
            float weight = w->edges[e].weight * (w->boot ? exponential(&state) : 1.0f);
            add_cost(up, dn, v, weight, weight);
        }
        for (int64_t r = 0; r < w->ray_count; r++) {
            int32_t v = w->element_term[w->edge_count + r];
            if (v == -1) continue;
            float weight = w->ray_weight[r] * (w->boot ? exponential(&state) : 1.0f);
            add_cost(up, dn, v, weight * w->hidden, weight);
        }
        for (int64_t t = 0; t < w->terms; t++) {
            float gain = 0;
            for (int32_t i = 0; i < w->join_shape; i++) gain += exponential(&state);
            gain = (float)w->scale * (w->join_shape ? gain / (float)w->join_shape : 1.0f);
            int64_t a = llrintf(up[t].sum * gain), b = llrintf(dn[t].sum * gain);
            up[t].units = a < 1 ? 1 : a > 1000000000 ? 1000000000 : (int32_t)a;
            dn[t].units = b < 1 ? 1 : b > 1000000000 ? 1000000000 : (int32_t)b;
        }
        memcpy(x, w->x0, sizeof(int32_t) * (size_t)w->count);
        int64_t moved = 0;
        int64_t cuts = unit_steps(&g, w->count, w->terms, w->p, w->q, w->k, (const int32_t *)up, (const int32_t *)dn, x, w->shape->forward, w->shape->term_of, 0, &moved);
        if (cuts < 0) __atomic_fetch_add(&w->failed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&w->cuts, cuts, __ATOMIC_RELAXED);
        for (int64_t t = 0; t < w->terms; t++)
            if (x[w->q[t]] - x[w->p[t]] == w->k[t]) __atomic_fetch_add(&w->votes[t], 1, __ATOMIC_RELAXED);
    }
    free(up); free(dn); free(x);
    graph_close(&g);
    return NULL;
}

// Bootstrap draws use independent exponential weights.
int64_t hx_level_votes(int64_t count, int64_t terms, const int32_t *p, const int32_t *q, const int32_t *k, const Edge *edges, int64_t edge_count, const float *ray_weight, int64_t ray_count, float hidden, const int32_t *element_term,
                       const int32_t *x0, int32_t draws, int32_t threads, int32_t scale, int32_t boot, int32_t join_shape, uint64_t seed, int32_t *votes, int64_t *log) {
    Shape shape;
    shape_open(&shape, count, terms, p, q);
    Work w = {count, terms, edge_count, ray_count, p, q, k, element_term, x0, &shape, edges, ray_weight, hidden, draws, scale, boot, join_shape, seed, 0, votes, 0, 0};
    if (threads < 1) threads = 1;
    pthread_t pool[64];
    for (int32_t i = 0; i < threads && i < 64; i++) pthread_create(&pool[i], NULL, draw_worker, &w);
    for (int32_t i = 0; i < threads && i < 64; i++) pthread_join(pool[i], NULL);
    shape_close(&shape);
    log[0] = w.cuts;
    return w.failed;
}
