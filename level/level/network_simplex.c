#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LOWER 1
#define TREE 0
#define UPPER (-1)
#define UP 1
#define DOWN (-1)
#define UNBOUNDED INT64_MAX

typedef struct {
    int32_t *parent, *pred, *depth, *first, *next, *prev;
    int8_t *dir;
} Tree;

static void detach(Tree *t, int32_t u) {
    int32_t p = t->parent[u];
    if (t->prev[u] >= 0)
        t->next[t->prev[u]] = t->next[u];
    else
        t->first[p] = t->next[u];
    if (t->next[u] >= 0)
        t->prev[t->next[u]] = t->prev[u];
}

static void attach(Tree *t, int32_t u, int32_t p) {
    t->parent[u] = p;
    t->prev[u] = -1;
    t->next[u] = t->first[p];
    if (t->first[p] >= 0)
        t->prev[t->first[p]] = u;
    t->first[p] = u;
}

static _Thread_local const int64_t *sort_key;

static int by_capacity(const void *p, const void *q) {
    int64_t a = sort_key[*(const int32_t *)p], b = sort_key[*(const int32_t *)q];
    if (a != b)
        return a > b ? -1 : 1;
    return *(const int32_t *)p - *(const int32_t *)q;
}

static int32_t root_of(int32_t *up, int32_t v) {
    while (up[v] != v) {
        up[v] = up[up[v]];
        v = up[v];
    }
    return v;
}

static int strong_tree(int32_t n, int64_t m, const int32_t *tail, const int32_t *head,
    const int64_t *cost, const int64_t *cap, int8_t *state, int64_t *pi, Tree *t) {
    int32_t *order = malloc(sizeof(int32_t) * (m + 1)),
            *up = malloc(sizeof(int32_t) * ((size_t)n + 1));
    int64_t *start = calloc((size_t)n + 2, sizeof(int64_t));
    int32_t *chosen = malloc(sizeof(*chosen) * ((size_t)n + 1));
    int64_t *cursor = NULL;
    int32_t *queue = NULL, *list = NULL;
    uint8_t *seen = NULL;
    int ok = 0;
    if (!order || !up || !chosen || !start) {
        goto finish_tree;
    }
    for (int64_t e = 0; e < m; e++)
        order[e] = (int32_t)e;
    sort_key = cap;
    qsort(order, m, sizeof(int32_t), by_capacity);
    for (int32_t v = 0; v < n; v++)
        up[v] = v;
    int32_t count = 0;
    for (int64_t i = 0; i < m && count < n - 1; i++) {
        int32_t e = order[i], a = root_of(up, tail[e]), b = root_of(up, head[e]);
        if (a == b)
            continue;
        up[a] = b;
        chosen[count++] = e;
    }
    free(order);
    order = NULL;
    free(up);
    up = NULL;
    cursor = malloc(((size_t)n + 1) * sizeof(*cursor));
    list = malloc((2 * (size_t)count + 1) * sizeof(*list));
    queue = malloc(((size_t)n + 1) * sizeof(*queue));
    seen = calloc((size_t)n + 1, 1);
    if (!cursor || !list || !queue || !seen) {
        goto finish_tree;
    }
    for (int32_t i = 0; i < count; i++) {
        start[tail[chosen[i]] + 1]++;
        start[head[chosen[i]] + 1]++;
    }
    for (int32_t v = 0; v < n; v++)
        start[v + 1] += start[v];
    memcpy(cursor, start, sizeof(int64_t) * ((size_t)n + 1));
    for (int32_t i = 0; i < count; i++) {
        list[cursor[tail[chosen[i]]]++] = chosen[i];
        list[cursor[head[chosen[i]]]++] = chosen[i];
    }
    for (int32_t s = 0; s < n; s++) {
        if (seen[s])
            continue;
        seen[s] = 1;
        int32_t front = 0, back = 0;
        queue[back++] = s;
        while (front < back) {
            int32_t v = queue[front++];
            for (int64_t k = start[v]; k < start[v + 1]; k++) {
                int32_t e = list[k], w = tail[e] == v ? head[e] : tail[e];
                if (seen[w])
                    continue;
                seen[w] = 1;
                state[m + w] = LOWER;
                state[e] = TREE;
                detach(t, w);
                attach(t, w, v);
                t->pred[w] = e;
                t->dir[w] = tail[e] == v ? DOWN : UP;
                t->depth[w] = t->depth[v] + 1;
                pi[w] = tail[e] == v ? pi[v] + cost[e] : pi[v] - cost[e];
                queue[back++] = w;
            }
        }
    }
    ok = 1;
finish_tree:
    free(order);
    free(up);
    free(start);
    free(chosen);
    free(cursor);
    free(list);
    free(queue);
    free(seen);
    return ok;
}

int64_t hx_network_simplex(int32_t n, int64_t m, const int32_t *tail_in, const int32_t *head_in,
    const int32_t *cost_in, const int64_t *cap_in, int64_t *pi, int64_t pivot_limit) {
    if (n < 0 || n >= INT32_MAX || m < 0 || m > INT32_MAX - n) {
        return -3;
    }
    if (!n) {
        pi[0] = 0;
        return 0;
    }
    int32_t root = n, nodes = n + 1;
    int64_t arcs = m + n;
    int32_t *tail = malloc(sizeof(int32_t) * arcs), *head = malloc(sizeof(int32_t) * arcs);
    int64_t *cost = malloc(sizeof(int64_t) * arcs), *cap = malloc(sizeof(int64_t) * arcs),
            *flow = calloc(arcs, sizeof(int64_t));
    int8_t *state = malloc(arcs);
    Tree t;
    t.parent = malloc(sizeof(int32_t) * nodes);
    t.pred = malloc(sizeof(int32_t) * nodes);
    t.depth = malloc(sizeof(int32_t) * nodes);
    t.first = malloc(sizeof(int32_t) * nodes);
    t.next = malloc(sizeof(int32_t) * nodes);
    t.prev = malloc(sizeof(int32_t) * nodes);
    t.dir = malloc(nodes);
    int32_t *stack = malloc(sizeof(int32_t) * nodes);

    int64_t pivots = -2;
    if (!tail || !head || !cost || !cap || !flow || !state || !stack || !t.parent || !t.pred ||
        !t.depth || !t.first || !t.next || !t.prev || !t.dir) {
        goto finish;
    }
    int64_t largest = 1;
    for (int64_t e = 0; e < m; e++) {
        tail[e] = tail_in[e];
        head[e] = head_in[e];
        cost[e] = cost_in[e];
        cap[e] = cap_in[e];
        state[e] = LOWER;
        int64_t magnitude = cost[e] < 0 ? -cost[e] : cost[e];
        if (magnitude > largest)
            largest = magnitude;
    }
    int64_t artificial = (largest + 1) * nodes;
    for (int32_t i = 0; i < nodes; i++)
        t.first[i] = -1;
    pi[root] = 0;
    t.parent[root] = -1;
    t.pred[root] = -1;
    t.depth[root] = 0;
    t.dir[root] = 0;
    t.next[root] = t.prev[root] = -1;
    for (int32_t v = 0; v < n; v++) {
        int64_t e = m + v;
        tail[e] = v;
        head[e] = root;
        cost[e] = artificial;
        cap[e] = UNBOUNDED;
        state[e] = TREE;
        pi[v] = -artificial;
        t.pred[v] = (int32_t)e;
        t.dir[v] = UP;
        t.depth[v] = 1;
        attach(&t, v, root);
    }

    if (!strong_tree(n, m, tail, head, cost, cap, state, pi, &t)) {
        goto finish;
    }

    int64_t block = (int64_t)sqrt((double)arcs);
    if (block < 10)
        block = 10;
    int64_t cursor = 0;
    pivots = 0;
    while (1) {
        int64_t best = 0, entering = -1, scanned = 0, in_block = 0;
        while (scanned < arcs) {
            int64_t e = cursor;
            if (++cursor == arcs)
                cursor = 0;
            scanned++;
            in_block++;
            int8_t s = state[e];
            if (s != TREE) {
                int64_t c = s * (cost[e] + pi[tail[e]] - pi[head[e]]);
                if (c < best) {
                    best = c;
                    entering = e;
                }
            }
            if (in_block == block) {
                if (entering >= 0)
                    break;
                in_block = 0;
            }
        }
        if (entering < 0)
            break;
        if (pivots++ >= pivot_limit) {
            pivots = -1;
            break;
        }

        int32_t first = state[entering] == LOWER ? tail[entering] : head[entering];
        int32_t second = state[entering] == LOWER ? head[entering] : tail[entering];
        int32_t u = first, v = second;
        while (u != v) {
            if (t.depth[u] >= t.depth[v])
                u = t.parent[u];
            else
                v = t.parent[v];
        }
        int32_t join = u, out_node = -1;
        int64_t delta = cap[entering];
        int result = 0;
        for (u = first; u != join; u = t.parent[u]) {
            int64_t e = t.pred[u], d = flow[e];
            if (t.dir[u] == DOWN)
                d = cap[e] == UNBOUNDED ? UNBOUNDED : cap[e] - d;
            if (d < delta) {
                delta = d;
                out_node = u;
                result = 1;
            }
        }
        for (u = second; u != join; u = t.parent[u]) {
            int64_t e = t.pred[u], d = flow[e];
            if (t.dir[u] == UP)
                d = cap[e] == UNBOUNDED ? UNBOUNDED : cap[e] - d;
            if (d <= delta) {
                delta = d;
                out_node = u;
                result = 2;
            }
        }
        if (delta > 0) {
            int64_t value = state[entering] * delta;
            flow[entering] += value;
            for (u = tail[entering]; u != join; u = t.parent[u])
                flow[t.pred[u]] -= t.dir[u] * value;
            for (u = head[entering]; u != join; u = t.parent[u])
                flow[t.pred[u]] += t.dir[u] * value;
        }
        if (!result) {
            state[entering] = -state[entering];
            continue;
        }
        int64_t leaving = t.pred[out_node];
        state[entering] = TREE;
        state[leaving] = flow[leaving] == 0 ? LOWER : UPPER;

        int32_t u_in = result == 1 ? first : second, v_in = result == 1 ? second : first;
        int32_t w = u_in, above = v_in, arc = (int32_t)entering;
        int8_t direction = tail[entering] == u_in ? UP : DOWN;
        while (1) {
            int32_t old_parent = t.parent[w], old_arc = t.pred[w];
            int8_t old_direction = t.dir[w];
            detach(&t, w);
            attach(&t, w, above);
            t.pred[w] = arc;
            t.dir[w] = direction;
            if (w == out_node)
                break;
            above = w;
            arc = old_arc;
            direction = -old_direction;
            w = old_parent;
        }
        int64_t target =
            tail[entering] == u_in ? pi[v_in] - cost[entering] : pi[v_in] + cost[entering];
        int64_t shift = target - pi[u_in];
        int32_t top = 0;
        stack[top++] = u_in;
        while (top) {
            int32_t x = stack[--top];
            pi[x] += shift;
            t.depth[x] = t.depth[t.parent[x]] + 1;
            for (int32_t c = t.first[x]; c >= 0; c = t.next[c])
                stack[top++] = c;
        }
    }
finish:
    free(tail);
    free(head);
    free(cost);
    free(cap);
    free(flow);
    free(state);
    free(stack);
    free(t.parent);
    free(t.pred);
    free(t.depth);
    free(t.first);
    free(t.next);
    free(t.prev);
    free(t.dir);
    return pivots;
}
