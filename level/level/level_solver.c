#include <errno.h>
#include <unistd.h>
#include <math.h>
#include "parallel_blocks.h"
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NO_KINK 0x7fffffff
#define MOST_COST (1 << 24)

typedef struct {
    int32_t head, sister;
} Arc;

typedef struct {
    void *(*open)(int32_t, int32_t, const int32_t *, const Arc *);
    int64_t (*run)(void *, int32_t *, int64_t *);
    void (*close)(void *);
} DeviceFlow;

static DeviceFlow device_flow;
static pthread_mutex_t device_lock = PTHREAD_MUTEX_INITIALIZER;

void hx_level_device(DeviceFlow *api) {
    pthread_mutex_lock(&device_lock);
    if (!device_flow.open)
        device_flow = *api;
    pthread_mutex_unlock(&device_lock);
}

typedef struct {
    const int32_t *first, *bias;
    int64_t *terminal;
    int8_t *side;
    const Arc *arcs;
    int32_t *cap;
    int32_t count, source_side, sink_side;
    int64_t flow;
    int32_t *label, *fifo, *current, *ring;
    int32_t relabels;
    void *device;
} Graph;

#define BFS_CHUNKS 64
#define BFS_PARALLEL_LEVEL 4096

typedef struct {
    Graph *g;
    int32_t *frontier;
    int32_t size, chunk, next, end;
} Frontier;

static void visit_chunk(void *context, size_t c) {
    Frontier *f = context;
    Graph *g = f->g;
    int32_t lo = (int32_t)c * f->chunk, hi = lo + f->chunk;
    int32_t saved[128], count = 0;
    for (int32_t i = lo; i < hi && i < f->size; i++) {
        int32_t v = f->frontier[i];
        for (int32_t e = g->first[v]; e < g->first[v + 1]; e++) {
            int32_t u = g->arcs[e].head, expected = g->count;
            if (__atomic_load_n(&g->label[u], __ATOMIC_RELAXED) != expected ||
                g->cap[g->arcs[e].sister] <= 0)
                continue;
            if (!__atomic_compare_exchange_n(
                    &g->label[u], &expected, f->next, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                continue;
            saved[count++] = u;
            if (count == 128) {
                int32_t start = __atomic_fetch_add(&f->end, count, __ATOMIC_RELAXED);
                memcpy(g->fifo + start, saved, sizeof(saved));
                count = 0;
            }
        }
    }
    if (count) {
        int32_t start = __atomic_fetch_add(&f->end, count, __ATOMIC_RELAXED);
        memcpy(g->fifo + start, saved, (size_t)count * sizeof(*saved));
    }
}

static void relabel(Graph *g) {
    int32_t n = g->count, end = 0;
    int32_t *label = g->label, *queue = g->fifo;
    for (int32_t i = 0; i < n; i++) {
        label[i] = g->terminal[i] < 0 ? 0 : n;
        if (label[i] == 0)
            queue[end++] = i;
    }
    Frontier f = {.g = g};
    for (int32_t lo = 0, hi; lo < end; lo = hi) {
        hi = end;
        f.size = hi - lo;
        f.next++;
        if (g->relabels <= 2 || f.size < BFS_PARALLEL_LEVEL) {
            for (int32_t at = lo; at < hi; at++) {
                int32_t v = queue[at];
                for (int32_t e = g->first[v]; e < g->first[v + 1]; e++) {
                    int32_t u = g->arcs[e].head;
                    if (label[u] == n && g->cap[g->arcs[e].sister] > 0) {
                        label[u] = f.next;
                        queue[end++] = u;
                    }
                }
            }
            continue;
        }
        f.frontier = queue + lo;
        f.chunk = (f.size + BFS_CHUNKS - 1) / BFS_CHUNKS;
        f.end = end;
        each_block(BFS_CHUNKS, &f, visit_chunk);
        end = f.end;
    }
}

static void mark_cut(Graph *g) {
    int32_t n = g->count, *fifo = g->fifo;
    const Arc *A = g->arcs;
    const int32_t *C = g->cap;
    memset(g->side, 0, (size_t)n);
    int32_t end = 0, at = 0;
    for (int32_t i = 0; i < n; i++)
        if (g->terminal[i] > 0) {
            g->side[i] = 1;
            fifo[end++] = i;
        }
    while (at < end) {
        int32_t u = fifo[at++];
        for (int32_t e = g->first[u]; e < g->first[u + 1]; e++) {
            int32_t v = A[e].head;
            if (C[e] > 0 && !g->side[v]) {
                g->side[v] = 1;
                fifo[end++] = v;
            }
        }
    }
    g->source_side = end;
    at = end;
    for (int32_t i = 0; i < n; i++)
        if (g->terminal[i] < 0) {
            g->side[i] = -1;
            fifo[end++] = i;
        }
    while (at < end) {
        int32_t v = fifo[at++];
        for (int32_t e = g->first[v]; e < g->first[v + 1]; e++) {
            int32_t u = A[e].head;
            if (C[A[e].sister] > 0 && !g->side[u]) {
                g->side[u] = -1;
                fifo[end++] = u;
            }
        }
    }
    g->sink_side = end - g->source_side;
}

static void maxflow(Graph *g, int64_t offered) {
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    int32_t n = g->count;
    int32_t *label = g->label, *current = g->current, *ring = g->ring;
    g->flow = 0;
    if (!offered)
        return;
    if (g->device) {
        int64_t flow = device_flow.run(g->device, C, g->terminal);
        if (flow >= 0) {
            g->flow = flow;
            if (flow != offered)
                mark_cut(g);
            return;
        }
        device_flow.close(g->device);
        g->device = NULL;
    }
    int64_t work = 0, arcs = g->first[n];
    relabel(g);
    int32_t head = 0, tail = 0, held = 0;
    for (int32_t i = 0; i < n; i++) {
        current[i] = g->first[i];
        g->side[i] = 0;
    }
    for (int32_t i = 0; i < n; i++) {
        if (g->terminal[i] > 0 && label[i] < n) {
            ring[tail] = i;
            if (++tail > n)
                tail = 0;
            held++;
            g->side[i] = 1;
        }
    }
    while (held) {
        int32_t u = ring[head];
        if (++head > n)
            head = 0;
        held--;
        g->side[u] = 0;
        while (g->terminal[u] > 0 && label[u] < n) {
            int32_t e = current[u];
            if (e == g->first[u + 1]) {
                int32_t lowest = n;
                for (int32_t f = g->first[u]; f < g->first[u + 1]; f++)
                    if (C[f] > 0 && label[A[f].head] + 1 < lowest)
                        lowest = label[A[f].head] + 1;
                work += g->first[u + 1] - g->first[u] + 12;
                label[u] = lowest;
                current[u] = g->first[u];
                continue;
            }
            int32_t v = A[e].head;
            if (C[e] > 0 && label[u] == label[v] + 1) {
                int64_t delta = g->terminal[u] < C[e] ? g->terminal[u] : C[e];
                C[e] -= (int32_t)delta;
                C[A[e].sister] += (int32_t)delta;
                g->terminal[u] -= delta;
                int64_t before = g->terminal[v];
                g->terminal[v] += delta;
                if (before < 0)
                    g->flow += -before < delta ? -before : delta;
                if (g->terminal[v] > 0 && !g->side[v] && label[v] < n) {
                    ring[tail] = v;
                    if (++tail > n)
                        tail = 0;
                    held++;
                    g->side[v] = 1;
                }
                if (g->terminal[u] == 0)
                    break;
            } else {
                current[u] = e + 1;
            }
        }
        if (work * g->relabels > arcs) {
            work = 0;
            relabel(g);
            for (int32_t i = 0; i < n; i++)
                current[i] = g->first[i];
            head = tail = held = 0;
            for (int32_t i = 0; i < n; i++) {
                g->side[i] = 0;
                if (g->terminal[i] > 0 && label[i] < n) {
                    ring[tail] = i;
                    if (++tail > n)
                        tail = 0;
                    held++;
                    g->side[i] = 1;
                }
            }
        }
    }
    if (g->flow == offered)
        return;
    mark_cut(g);
}

static inline int64_t cost(int64_t r, int64_t up, int64_t dn) { return r >= 0 ? up * r : -dn * r; }

int64_t hx_lnat_energy(int64_t terms, const int32_t *p, const int32_t *q, const int32_t *k,
    const int32_t *up, const int32_t *dn, const int32_t *x) {
    int64_t total = 0;
    for (int64_t t = 0; t < terms; t++)
        total += cost((int64_t)x[q[t]] - x[p[t]] - k[t], up[t], dn[t]);
    return total;
}

static int32_t forward_arc(const Graph *g, const int32_t *forward, int32_t t, int32_t p) {
    return g->bias ? t + g->bias[p] : forward[t];
}

static int32_t original_term(
    const Graph *g, const int32_t *by_arc, int32_t i, int32_t j, int32_t e) {
    if (!g->bias) {
        return by_arc[e];
    }
    return i < j ? e - g->bias[i] : g->arcs[e].sister - g->bias[j];
}

static int proven(Graph *g, int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *k, const int32_t *up, const int32_t *dn, const int32_t *x,
    const int32_t *forward) {
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    int64_t *balance = g->terminal;
    memset(balance, 0, (size_t)count * sizeof(*balance));
    int good = 1;
    for (int64_t t = 0; good && t < terms; t++) {
        int32_t r = x[q[t]] - x[p[t]] - k[t], a = forward_arc(g, forward, (int32_t)t, p[t]),
                b = A[a].sister;
        int64_t phi = r > 0 ? up[t] : r < 0 ? -(int64_t)dn[t] : C[a] - dn[t];
        good = r != 0 || (C[a] >= 0 && C[b] >= 0 && C[a] + C[b] == (int64_t)up[t] + dn[t]);
        balance[p[t]] += phi;
        balance[q[t]] -= phi;
    }
    for (int64_t i = 0; good && i < count; i++)
        good = balance[i] == 0;
    return good;
}

static int64_t unit_steps(Graph *g, int64_t count, int64_t terms, const int32_t *p,
    const int32_t *q, const int32_t *k, const int32_t *up, const int32_t *dn, int32_t *x,
    const int32_t *forward, const int32_t *term_of, int64_t cuts, int64_t *moved) {
    const Arc *A = g->arcs;
    int32_t *C = g->cap;
    for (int64_t i = 0; i < count; i++)
        g->terminal[i] = 0;
    for (int64_t t = 0; t < terms; t++) {
        int32_t a = forward_arc(g, forward, (int32_t)t, p[t]), b = A[a].sister;
        int32_t r = x[q[t]] - x[p[t]] - k[t];
        int64_t phi = r > 0 ? up[t] : r < 0 ? -(int64_t)dn[t] : 0;
        g->terminal[p[t]] += phi;
        g->terminal[q[t]] -= phi;
        C[a] = r == 0 ? dn[t] : 0;
        C[b] = r == 0 ? up[t] : 0;
    }
    int64_t offered = 0;
    for (int64_t i = 0; i < count; i++)
        if (g->terminal[i] > 0)
            offered += g->terminal[i];
    maxflow(g, offered);
    for (;;) {
        int64_t excess = offered - g->flow;
        cuts++;
        if (!excess) {
            if (!proven(g, count, terms, p, q, k, up, dn, x, forward))
                cuts = -2;
            break;
        }
        int32_t direction = g->sink_side < g->source_side ? -1 : 1, jump = NO_KINK;
        int32_t members_count = direction < 0 ? g->sink_side : g->source_side;
        const int32_t *members = g->fifo + (direction < 0 ? g->source_side : 0);
        for (int32_t m = 0; m < members_count; m++) {
            int32_t i = members[m];
            for (int32_t e = g->first[i]; e < g->first[i + 1]; e++) {
                int32_t j = A[e].head;
                if (g->side[j] == direction)
                    continue;
                int32_t t = original_term(g, term_of, i, j, e);
                int32_t r = x[q[t]] - x[p[t]] - k[t];
                int32_t shift = q[t] == i ? direction : -direction;
                if (r != 0 && (r > 0) != (shift > 0) && abs(r) < jump)
                    jump = abs(r);
            }
        }
        if (jump == NO_KINK)
            break;
        for (int32_t m = 0; m < members_count; m++) {
            int32_t i = members[m];
            for (int32_t e = g->first[i]; e < g->first[i + 1]; e++) {
                int32_t j = A[e].head;
                if (g->side[j] == direction)
                    continue;
                int32_t t = original_term(g, term_of, i, j, e);
                int32_t r = x[q[t]] - x[p[t]] - k[t];
                int32_t shift = q[t] == i ? direction : -direction;
                int32_t a = forward_arc(g, forward, (int32_t)t, p[t]), b = A[a].sister;
                if (r == 0) {
                    C[a] = C[b] = 0;
                } else if (r + shift * jump == 0) {
                    int64_t phi = r > 0 ? up[t] : -(int64_t)dn[t];
                    C[a] = (int32_t)(phi + dn[t]);
                    C[b] = (int32_t)(up[t] - phi);
                }
            }
            x[i] += direction * jump;
            *moved += jump;
        }
        int64_t routed = g->flow;
        maxflow(g, excess);
        g->flow += routed;
    }
    return cuts;
}

typedef struct {
    int32_t count;
    int64_t terms;
    int32_t *first, *forward, *term_of, *bias;
    Arc *arcs;
} Shape;

static void shape_close(Shape *s) {
    free(s->bias);
    free(s->first);
    free(s->arcs);
    free(s->forward);
    free(s->term_of);
}

enum { TERM_BATCH = 65536 };

static int load_part(int fd, void *buffer, size_t bytes, off_t offset) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, (char *)buffer + done, bytes - done, offset + (off_t)done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return 0;
        }
        done += (size_t)n;
    }
    return 1;
}

static int shape_open(Shape *s, int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *weight, int32_t *cap) {
    if (count < 0 || count >= INT32_MAX || terms < 0 || terms > INT32_MAX / 2)
        return -1;
    *s = (Shape){.count = count, .terms = terms};
    s->first = calloc((size_t)count + 1, sizeof(*s->first));
    s->arcs = malloc((size_t)(2 * terms + 1) * sizeof(*s->arcs));
    int ordered = !cap;
    int32_t previous = -1;
    int32_t *next = malloc((size_t)(count + 1) * sizeof(*next));
    if (!s->first || !s->arcs || !next) {
        goto failed;
    }
    for (int32_t t = 0; t < terms; t++) {
        if (ordered) {
            ordered = p[t] >= previous && p[t] < q[t];
            previous = p[t];
        }
        s->first[p[t] + 1]++;
        s->first[q[t] + 1]++;
    }
    if (ordered) {
        s->bias = malloc((size_t)(count + 1) * sizeof(*s->bias));
        if (!s->bias) {
            goto failed;
        }
    } else if (!cap) {
        s->forward = malloc((size_t)(terms + 1) * sizeof(*s->forward));
        s->term_of = malloc((size_t)(2 * terms + 1) * sizeof(*s->term_of));
        if (!s->forward || !s->term_of) {
            goto failed;
        }
    }
    for (int32_t i = 0; i < count; i++) {
        s->first[i + 1] += s->first[i];
        next[i] = s->first[i];
    }
    for (int32_t t = 0; t < terms; t++) {
        int32_t a = next[p[t]]++, b = next[q[t]]++;
        s->arcs[a] = (Arc){.head = q[t], .sister = b};
        s->arcs[b] = (Arc){.head = p[t], .sister = a};
        if (cap) {
            cap[a] = cap[b] = weight[t];
        } else if (ordered) {
            s->bias[p[t]] = a - t;
        } else {
            s->forward[t] = a;
            s->term_of[a] = s->term_of[b] = t;
        }
    }
    free(next);
    return 0;
failed:
    free(next);
    shape_close(s);
    return -1;
}

static void graph_close(Graph *g) {
    if (g->device)
        device_flow.close(g->device);
    free(g->terminal);
    free(g->side);
    free(g->cap);
    free(g->label);
    free(g->fifo);
    free(g->current);
    free(g->ring);
}

static int graph_open(Graph *g, const Shape *s, int32_t *cap) {
    size_t n = (size_t)s->count + 1;
    *g = (Graph){
        .count = s->count, .first = s->first, .arcs = s->arcs, .bias = s->bias, .relabels = 2};
    g->terminal = malloc(n * sizeof(*g->terminal));
    g->side = malloc(n * sizeof(*g->side));
    g->cap = cap ? cap : malloc((size_t)(2 * s->terms + 1) * sizeof(*g->cap));
    g->label = malloc(n * sizeof(*g->label));
    g->fifo = malloc(n * sizeof(*g->fifo));
    g->current = malloc(n * sizeof(*g->current));
    g->ring = malloc(n * sizeof(*g->ring));
    if (g->terminal && g->side && g->cap && g->label && g->fifo && g->current && g->ring)
        return 0;
    graph_close(g);
    return -1;
}

int64_t hx_level_minimize(int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *k, const int32_t *up, const int32_t *dn, int32_t *x, int32_t first_step,
    int64_t *log) {
    Shape s;
    if (shape_open(&s, count, terms, p, q, NULL, NULL))
        return -1;
    Graph solve, *g = &solve;
    if (graph_open(g, &s, NULL)) {
        shape_close(&s);
        return -1;
    }
    g->relabels = 4;
    pthread_mutex_lock(&device_lock);
    DeviceFlow gpu = device_flow;
    pthread_mutex_unlock(&device_lock);
    if (gpu.open && count >= 65536 && terms >= 2 * count)
        g->device = gpu.open(s.count, 2 * terms, s.first, s.arcs);
    int64_t cuts = 0, moved = 0;
    for (int32_t step = first_step; step >= 2; step /= 2) {
        int quiet = 0;
        for (int direction = 1; quiet < 2; direction = -direction) {
            int32_t delta = direction * step;
            for (int64_t i = 0; i < count; i++)
                g->terminal[i] = 0;
            int64_t offered = 0;
            for (int64_t t = 0; t < terms; t++) {
                int64_t r = (int64_t)x[q[t]] - x[p[t]] - k[t];
                int64_t e01 = cost(r + delta, up[t], dn[t]) - cost(r, up[t], dn[t]);
                int64_t e10 = cost(r - delta, up[t], dn[t]) - cost(r, up[t], dn[t]);
                int64_t u = 0;
                if (e01 < 0)
                    u = e01;
                else if (e10 < 0)
                    u = -e10;
                g->terminal[q[t]] -= u;
                g->terminal[p[t]] += u;
                int32_t a = forward_arc(g, s.forward, (int32_t)t, p[t]);
                g->cap[a] = (int32_t)(e10 + u);
                g->cap[g->arcs[a].sister] = (int32_t)(e01 - u);
            }
            for (int64_t i = 0; i < count; i++)
                if (g->terminal[i] > 0)
                    offered += g->terminal[i];
            maxflow(g, offered);
            cuts++;
            if (g->flow - offered >= 0) {
                quiet++;
                continue;
            }
            quiet = 0;
            for (int64_t i = 0; i < count; i++) {
                if (g->side[i] == 1) {
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

int64_t hx_level_two_labels(int64_t count, int64_t terms, const int32_t *a, const int32_t *b,
    const int32_t *weight, const int64_t *unary, uint8_t *label) {
    if (terms < 0 || terms > INT32_MAX / 2) {
        return -1;
    }
    int32_t *cap = malloc((size_t)(2 * terms + 1) * sizeof(*cap));
    if (!cap) {
        return -1;
    }
    Shape s;
    if (shape_open(&s, count, terms, a, b, weight, cap)) {
        free(cap);
        return -1;
    }
    Graph solve, *g = &solve;
    if (graph_open(g, &s, cap)) {
        shape_close(&s);
        return -1;
    }
    for (int64_t i = 0; i < count; i++)
        g->terminal[i] = unary[i];
    maxflow(g, -1);
    for (int64_t i = 0; i < count; i++)
        label[i] = g->side[i] == 1;
    int64_t flow = g->flow;
    graph_close(g);
    shape_close(&s);
    return flow;
}

typedef struct {
    int32_t q, k, element;
} Entry;

static int compare_entries(const void *a, const void *b) {
    const Entry *x = a, *y = b;
    if (x->q != y->q)
        return x->q < y->q ? -1 : 1;
    if (x->k != y->k)
        return x->k < y->k ? -1 : 1;
    return 0;
}

typedef struct {
    int32_t a, b;
    float weight;
    int32_t step;
} Edge;
typedef struct {
    int32_t a, b, rank;
} Pair;

typedef struct {
    int64_t terms;
    int32_t *p, *q, *k;
    int32_t data[];
} Terms;

static Terms *terms_new(int64_t terms) {
    size_t stride = (size_t)terms + 1;
    if (stride > (SIZE_MAX - sizeof(Terms)) / (3 * sizeof(int32_t))) {
        return NULL;
    }
    Terms *list = malloc(sizeof(*list) + 3 * stride * sizeof(*list->data));
    if (list) {
        list->terms = terms;
        list->p = list->data;
        list->q = list->p + stride;
        list->k = list->q + stride;
    }
    return list;
}

typedef struct {
    int32_t p, q, k;
} Key;

typedef struct {
    const Edge *edges;
    const Pair *rays;
    const int32_t *patch, *inner;
    const float *ray_weight, *theta;
    int64_t edge_count;
    int32_t chirality;
} TermInput;

static inline int term_key(const TermInput *in, int64_t e, Key *key, int *swapped) {
    int ray = e >= in->edge_count;
    int64_t r = e - in->edge_count;
    int32_t a = ray ? in->rays[r].a : in->edges[e].a;
    int32_t b = ray ? in->rays[r].b : in->edges[e].b;
    int32_t p = in->patch[a], q = in->patch[b];
    if (p == q || (ray && in->ray_weight[r] <= 0)) {
        return 0;
    }
    int32_t step = ray
        ? -(int32_t)rint(((double)in->theta[b] - in->theta[a]) / (2 * 3.14159265358979323846))
        : in->edges[e].step;
    int32_t offset = in->chirality * (step + in->inner[a] - in->inner[b]) + ray;
    *swapped = p > q;
    *key = *swapped ? (Key){q, p, -offset} : (Key){p, q, offset};
    return 1;
}

int64_t hx_level_terms(int64_t count, const int32_t *patch, const int32_t *inner, const Edge *edges,
    int64_t edge_count, const Pair *rays, const float *ray_weight, int64_t ray_count,
    const float *theta, int32_t chirality, int32_t *element_term, uint64_t *handle) {
    if (edge_count < 0 || ray_count < 0 || edge_count > INT32_MAX - 1 - ray_count) {
        return -1;
    }
    TermInput input = {edges, rays, patch, inner, ray_weight, theta, edge_count, chirality};
    int32_t elements = (int32_t)(edge_count + ray_count);
    int64_t *start = calloc((size_t)count + 2, sizeof(*start));
    if (!start) {
        return -1;
    }
    for (int32_t e = 0; e < elements; e++) {
        Key key;
        int swapped;
        if (term_key(&input, e, &key, &swapped)) {
            start[key.p + 1]++;
        }
    }
    for (int64_t i = 0; i < count; i++) {
        start[i + 1] += start[i];
    }
    Entry *entries = malloc(((size_t)start[count] + 1) * sizeof(*entries));
    int64_t *fill = malloc(((size_t)count + 1) * sizeof(*fill));
    int64_t result = -1;
    if (!entries || !fill) {
        goto finish;
    }
    memcpy(fill, start, ((size_t)count + 1) * sizeof(*fill));
    for (int32_t e = 0; e < elements; e++) {
        Key key;
        int swapped;
        element_term[e] = -1;
        if (term_key(&input, e, &key, &swapped)) {
            entries[fill[key.p]++] = (Entry){key.q, key.k, swapped ? ~e : e};
        }
    }
    free(fill);
    fill = NULL;
    int64_t terms = 0;
    for (int64_t i = 0; i < count; i++) {
        int64_t from = start[i], to = start[i + 1];
        if (to - from > 1)
            qsort(entries + from, (size_t)(to - from), sizeof(Entry), compare_entries);
        for (int64_t j = from; j < to; j++)
            terms += j == from || compare_entries(entries + j, entries + j - 1) != 0;
    }
    Terms *list = terms_new(terms);
    if (!list) {
        goto finish;
    }
    int64_t t = -1;
    for (int64_t i = 0; i < count; i++) {
        for (int64_t j = start[i]; j < start[i + 1]; j++) {
            if (j == start[i] || compare_entries(entries + j, entries + j - 1) != 0) {
                t++;
                list->p[t] = (int32_t)i;
                list->q[t] = entries[j].q;
                list->k[t] = entries[j].k;
            }
            int32_t e = entries[j].element;
            element_term[e < 0 ? ~e : e] = e < 0 ? (int32_t)(-2 - t) : (int32_t)t;
        }
    }
    *handle = (uint64_t)list;
    result = terms;
finish:
    free(start);
    free(fill);
    free(entries);
    return result;
}

void hx_level_terms_take(uint64_t handle, int32_t *p, int32_t *q, int32_t *k) {
    Terms *list = (Terms *)handle;
    memcpy(p, list->p, sizeof(int32_t) * (size_t)list->terms);
    memcpy(q, list->q, sizeof(int32_t) * (size_t)list->terms);
    memcpy(k, list->k, sizeof(int32_t) * (size_t)list->terms);
    free(list);
}

typedef union {
    float sum;
    int32_t units;
} Cost;

static inline void add_cost(Cost *up, Cost *dn, int32_t v, float a, float b) {
    if (v >= 0) {
        up[v].sum += a;
        dn[v].sum += b;
    } else if (v < -1) {
        up[-2 - v].sum += b;
        dn[-2 - v].sum += a;
    }
}

static inline uint64_t next_random(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline float exponential(uint64_t *state) {
    return -logf(((float)(next_random(state) >> 40) + 0.5f) * (1.0f / 16777216.0f));
}

typedef struct {
    const Edge *edges;
    const int32_t *ids;
    int64_t edge_count, ray_count;
    const float *ray_weight, *ray_up;
    float hidden;
} WeightInput;

static void accumulate_costs(
    const WeightInput *in, int64_t terms, Cost *up, Cost *dn, uint64_t *random) {
    memset(up, 0, (size_t)terms * sizeof(*up));
    memset(dn, 0, (size_t)terms * sizeof(*dn));
    for (int64_t e = 0; e < in->edge_count; e++) {
        int32_t id = in->ids[e];
        if (id != -1) {
            float weight = in->edges[e].weight * (random ? exponential(random) : 1.0f);
            add_cost(up, dn, id, weight, weight);
        }
    }
    for (int64_t r = 0; r < in->ray_count; r++) {
        int32_t id = in->ids[in->edge_count + r];
        if (id != -1) {
            float factor = random ? exponential(random) : 1.0f;
            float weight = in->ray_weight[r] * factor;
            float skip = in->ray_up ? in->ray_up[r] * factor : weight * in->hidden;
            add_cost(up, dn, id, skip, weight);
        }
    }
}

static inline int32_t cost_units(float sum, float gain, int32_t limit) {
    int64_t units = llrintf(sum * gain);
    return units < 1 ? 1 : units > limit ? limit : (int32_t)units;
}

void hx_level_weights(int64_t terms, const Edge *edges, int64_t edge_count, const float *ray_weight,
    const float *ray_up, int64_t ray_count, float hidden, const int32_t *element_term,
    int32_t scale, int32_t *up_units, int32_t *down_units) {
    WeightInput in = {edges, element_term, edge_count, ray_count, ray_weight, ray_up, hidden};
    Cost *up = (Cost *)up_units, *dn = (Cost *)down_units;
    accumulate_costs(&in, terms, up, dn, NULL);
    for (int64_t t = 0; t < terms; t++) {
        up[t].units = cost_units(up[t].sum, scale, MOST_COST);
        dn[t].units = cost_units(dn[t].sum, scale, MOST_COST);
    }
}

static int32_t find(int32_t *parent, int32_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

typedef struct {
    float x, y, z;
    int8_t nx, ny, nz;
    uint8_t prominence, width, coherence;
    uint16_t pad;
} SheetNode;

static inline int agreeing(const SheetNode *nodes, const Edge *e) {
    const SheetNode *a = &nodes[e->a], *b = &nodes[e->b];
    return (int32_t)a->nx * b->nx + (int32_t)a->ny * b->ny + (int32_t)a->nz * b->nz >= 0;
}

static inline int one_block(const SheetNode *nodes, const Edge *e, float side) {
    const SheetNode *a = &nodes[e->a], *b = &nodes[e->b];
    return floorf(a->x / side) == floorf(b->x / side) &&
        floorf(a->y / side) == floorf(b->y / side) && floorf(a->z / side) == floorf(b->z / side);
}

typedef struct {
    int64_t address, length, fd, offset;
} EdgeRun;

typedef struct {
    int64_t index, pos;
    Edge *buffer;
} EdgeReader;

static int64_t next_edges(
    EdgeReader *reader, const EdgeRun *runs, int64_t count, const Edge **out) {
    while (reader->index < count) {
        const EdgeRun *run = runs + reader->index;
        int64_t n = run->length - reader->pos;
        if (!n) {
            reader->index++;
            reader->pos = 0;
            continue;
        }
        if (run->fd < 0) {
            *out = (const Edge *)(uintptr_t)run->address + reader->pos;
        } else {
            if (!reader->buffer) {
                reader->buffer = malloc(TERM_BATCH * sizeof(Edge));
            }
            if (n > TERM_BATCH) {
                n = TERM_BATCH;
            }
            off_t at = run->offset + reader->pos * sizeof(Edge);
            if (!reader->buffer || !load_part(run->fd, reader->buffer, n * sizeof(Edge), at)) {
                return -1;
            }
            *out = reader->buffer;
        }
        reader->pos += n;
        return n;
    }
    return 0;
}

int64_t hx_level_blocks(int64_t count, const SheetNode *nodes, const EdgeRun *runs,
    int64_t run_count, float side, int32_t *block) {
    EdgeReader reader = {0};
    const Edge *run;
    int64_t n;
    int32_t *parent = malloc(sizeof(int32_t) * (size_t)(count + 1));
    if (!parent) {
        return -1;
    }
    for (int64_t i = 0; i < count; i++) {
        parent[i] = (int32_t)i;
    }
    while ((n = next_edges(&reader, runs, run_count, &run)) > 0) {
        for (int64_t e = 0; e < n; e++) {
            if (!agreeing(nodes, &run[e]) || !one_block(nodes, &run[e], side))
                continue;
            int32_t i = find(parent, run[e].a), j = find(parent, run[e].b);
            if (i != j)
                parent[i] = j;
        }
    }
    free(reader.buffer);
    if (n < 0) {
        free(parent);
        return -1;
    }
    int32_t blocks = 0;
    for (int64_t i = 0; i < count; i++)
        if (parent[i] == i)
            block[i] = blocks++;
    for (int64_t i = 0; i < count; i++)
        block[i] = block[find(parent, (int32_t)i)];
    free(parent);
    return blocks;
}

enum { LINK_SHIFT = 16, LINKS_PER_PAGE = 1 << LINK_SHIFT };

typedef struct {
    int32_t other, weight, previous;
} BlockLink;

int64_t hx_level_block_pairs(int64_t blocks, const int32_t *block, const SheetNode *nodes,
    const EdgeRun *runs, int64_t run_count, int32_t units, uint64_t *handle) {
    EdgeReader reader = {0};
    const Edge *run;
    int64_t n;
    int32_t *heads = malloc(((size_t)blocks + 1) * sizeof(*heads));
    size_t limit = 0;
    for (int64_t r = 0; r < run_count; r++) {
        limit += runs[r].length;
    }
    if (limit > INT32_MAX) {
        limit = INT32_MAX;
    }
    BlockLink **pages = calloc((limit >> LINK_SHIFT) + 1, sizeof(*pages));
    int32_t pairs = 0;
    size_t allocated = 0;
    Entry *row = NULL;
    int64_t result = -1;
    if (!heads || !pages) {
        goto finish;
    }
    memset(heads, -1, (size_t)blocks * sizeof(*heads));
    while ((n = next_edges(&reader, runs, run_count, &run)) > 0) {
        for (int64_t e = 0; e < n; e++) {
            int32_t a = block[run[e].a], b = block[run[e].b];
            if (a == b || !agreeing(nodes, run + e)) {
                continue;
            }
            int32_t first = a < b ? a : b, last = a < b ? b : a;
            int32_t pos = heads[first];
            BlockLink *link = NULL;
            while (pos >= 0) {
                link = &pages[pos >> LINK_SHIFT][pos & (LINKS_PER_PAGE - 1)];
                if (link->other == last) {
                    break;
                }
                pos = link->previous;
            }
            int64_t weight = llrintf(run[e].weight * (float)units);
            if (weight < 1) {
                weight = 1;
            }
            if (pos < 0) {
                if (pairs == INT32_MAX) {
                    goto finish;
                }
                int32_t offset = pairs & (LINKS_PER_PAGE - 1);
                if (!offset) {
                    size_t slots = limit - pairs;
                    if (slots > LINKS_PER_PAGE) {
                        slots = LINKS_PER_PAGE;
                    }
                    pages[allocated] = malloc(slots * sizeof(*link));
                    if (!pages[allocated]) {
                        goto finish;
                    }
                    allocated++;
                }
                link = &pages[pairs >> LINK_SHIFT][offset];
                *link = (BlockLink){.other = last, .previous = heads[first]};
                heads[first] = pairs++;
            }
            int64_t sum = (int64_t)link->weight + weight;
            link->weight = sum > 1000000000 ? 1000000000 : (int32_t)sum;
        }
    }
    if (n < 0) {
        goto finish;
    }
    Terms *list = terms_new(pairs);
    if (!list) {
        goto finish;
    }
    size_t capacity = 0, at = 0;
    for (int64_t i = 0; i < blocks; i++) {
        size_t count = 0;
        for (int32_t pos = heads[i]; pos >= 0;) {
            BlockLink *link = &pages[pos >> LINK_SHIFT][pos & (LINKS_PER_PAGE - 1)];
            if (count == capacity) {
                size_t next = capacity ? capacity * 2 : 16;
                Entry *larger = realloc(row, next * sizeof(*larger));
                if (!larger) {
                    free(list);
                    goto finish;
                }
                row = larger;
                capacity = next;
            }
            row[count++] = (Entry){.q = link->other, .k = link->weight};
            pos = link->previous;
        }
        if (count > 16) {
            qsort(row, count, sizeof(*row), compare_entries);
        } else {
            for (size_t j = 1; j < count; j++) {
                Entry next = row[j];
                size_t k = j;
                while (k && row[k - 1].q > next.q) {
                    row[k] = row[k - 1];
                    k--;
                }
                row[k] = next;
            }
        }
        for (size_t j = 0; j < count; j++) {
            list->p[at] = (int32_t)i;
            list->q[at] = row[j].q;
            list->k[at++] = row[j].k;
        }
    }
    *handle = (uint64_t)list;
    result = pairs;
finish:
    free(reader.buffer);
    free(row);
    for (size_t i = 0; i < allocated; i++) {
        free(pages[i]);
    }
    free(pages);
    free(heads);
    return result;
}

int64_t hx_level_domains(int64_t count, const SheetNode *nodes, const EdgeRun *runs,
    int64_t run_count, const uint8_t *flip, uint8_t *keep, int64_t *log) {
    EdgeReader reader = {0};
    const Edge *run;
    int64_t n;
    int32_t *parent = malloc(sizeof(int32_t) * (size_t)(count + 1));
    int64_t *gain = calloc((size_t)count + 1, sizeof(int64_t));
    if (!parent || !gain) {
        goto failed;
    }
    for (int64_t i = 0; i < count; i++) {
        parent[i] = (int32_t)i;
    }
    while ((n = next_edges(&reader, runs, run_count, &run)) > 0) {
        for (int64_t e = 0; e < n; e++) {
            if (!(flip[run[e].a] && flip[run[e].b] && agreeing(nodes, &run[e])))
                continue;
            int32_t i = find(parent, run[e].a), j = find(parent, run[e].b);
            if (i != j)
                parent[i] = j;
        }
    }
    if (n < 0) {
        goto failed;
    }
    reader.index = 0;
    reader.pos = 0;
    int64_t opposed = 0;
    while ((n = next_edges(&reader, runs, run_count, &run)) > 0) {
        for (int64_t e = 0; e < n; e++) {
            int agree = agreeing(nodes, &run[e]);
            opposed += !agree;
            int32_t i = flip[run[e].a] ? find(parent, run[e].a) : -1,
                    j = flip[run[e].b] ? find(parent, run[e].b) : -1;
            if (i == j)
                continue;
            if (i >= 0)
                gain[i] += agree ? -1 : 1;
            if (j >= 0)
                gain[j] += agree ? -1 : 1;
        }
    }
    if (n < 0) {
        goto failed;
    }
    int64_t kept = 0;
    for (int64_t i = 0; i < count; i++) {
        keep[i] = 0;
        if (!flip[i])
            continue;
        int32_t r = find(parent, (int32_t)i);
        keep[i] = gain[r] >= 0;
        kept += keep[i] && r == i;
    }
    free(parent);
    free(gain);
    log[0] = opposed;
    free(reader.buffer);
    return kept;
failed:
    free(reader.buffer);
    free(parent);
    free(gain);
    return -1;
}

typedef struct {
    WeightInput weights;
    int64_t count, terms;
    const int32_t *p, *q, *k, *x0;
    const Shape *shape;
    int32_t draws, scale, boot, join_shape;
    uint64_t seed;
    int32_t next;
    int32_t *votes;
    int64_t failed, cuts;
} Work;

static void *draw_worker(void *argument) {
    Work *w = argument;
    Graph g;
    if (graph_open(&g, w->shape, NULL)) {
        __atomic_fetch_add(&w->failed, 1, __ATOMIC_RELAXED);
        return NULL;
    }
    Cost *up = malloc(sizeof(Cost) * (size_t)(w->terms + 1));
    Cost *dn = malloc(sizeof(Cost) * (size_t)(w->terms + 1));
    int32_t *x = malloc(sizeof(int32_t) * (size_t)(w->count + 1));
    if (!up || !dn || !x) {
        __atomic_fetch_add(&w->failed, 1, __ATOMIC_RELAXED);
        goto finish_draw;
    }
    for (;;) {
        int32_t draw = __atomic_fetch_add(&w->next, 1, __ATOMIC_RELAXED);
        if (draw >= w->draws)
            break;
        uint64_t state = w->seed + 0x632be59bd9b4e019ULL * (uint64_t)(draw + 1);
        accumulate_costs(&w->weights, w->terms, up, dn, w->boot ? &state : NULL);
        for (int64_t t = 0; t < w->terms; t++) {
            float gain = 0;
            for (int32_t i = 0; i < w->join_shape; i++)
                gain += exponential(&state);
            gain = (float)w->scale * (w->join_shape ? gain / (float)w->join_shape : 1.0f);
            up[t].units = cost_units(up[t].sum, gain, 1000000000);
            dn[t].units = cost_units(dn[t].sum, gain, 1000000000);
        }
        memcpy(x, w->x0, sizeof(int32_t) * (size_t)w->count);
        int64_t moved = 0;
        int64_t cuts = unit_steps(&g, w->count, w->terms, w->p, w->q, w->k, (const int32_t *)up,
            (const int32_t *)dn, x, w->shape->forward, w->shape->term_of, 0, &moved);
        if (cuts < 0)
            __atomic_fetch_add(&w->failed, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&w->cuts, cuts, __ATOMIC_RELAXED);
        for (int64_t t = 0; t < w->terms; t++)
            if (x[w->q[t]] - x[w->p[t]] == w->k[t])
                __atomic_fetch_add(&w->votes[t], 1, __ATOMIC_RELAXED);
    }
finish_draw:
    free(up);
    free(dn);
    free(x);
    graph_close(&g);
    return NULL;
}

int64_t hx_level_votes(int64_t count, int64_t terms, const int32_t *p, const int32_t *q,
    const int32_t *k, const Edge *edges, int64_t edge_count, const float *ray_weight,
    const float *ray_up, int64_t ray_count, float hidden, const int32_t *element_term,
    const int32_t *x0, int32_t draws, int32_t threads, int32_t scale, int32_t boot,
    int32_t join_shape, uint64_t seed, int32_t *votes, int64_t *log) {
    Shape shape;
    if (shape_open(&shape, count, terms, p, q, NULL, NULL))
        return -1;
    Work w = {.weights = {edges, element_term, edge_count, ray_count, ray_weight, ray_up, hidden},
        .count = count,
        .terms = terms,
        .p = p,
        .q = q,
        .k = k,
        .x0 = x0,
        .shape = &shape,
        .draws = draws,
        .scale = scale,
        .boot = boot,
        .join_shape = join_shape,
        .seed = seed,
        .votes = votes};
    if (threads < 1)
        threads = 1;
    pthread_t pool[63];
    int32_t workers = 0;
    for (int32_t i = 1; i < threads && i < draws && i <= 63; i++) {
        if (pthread_create(&pool[workers], NULL, draw_worker, &w))
            break;
        workers++;
    }
    if (draws > 0)
        draw_worker(&w);
    for (int32_t i = 0; i < workers; i++)
        pthread_join(pool[i], NULL);
    shape_close(&shape);
    log[0] = w.cuts;
    return w.failed;
}
