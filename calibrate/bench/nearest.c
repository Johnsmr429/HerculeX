enum { BENCH_NEIGHBORS = 16, BENCH_KEEP = 3, BENCH_LEAF = 16 };

typedef struct {
    i32 begin, end, left, right;
    f32 low[3], high[3];
} BenchNode;

typedef struct {
    const f32 *points;
    i32 *order;
    BenchNode *nodes;
    i32 count;
} BenchTree;

typedef struct {
    i32 index;
    double squared;
} BenchCandidate;

typedef struct {
    i64 index[BENCH_KEEP];
    f32 depth[BENCH_KEEP];
} BenchMatches;

static int bench_finite(const f32 *p) { return isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]); }

static void bench_swap(i32 *a, i32 *b) {
    i32 c = *a;
    *a = *b;
    *b = c;
}

static void bench_select(BenchTree *t, int axis, i32 first, i32 end, i32 target) {
    i32 lo = first, hi = end - 1;
    while (lo < hi) {
        i32 pivot = t->order[lo + (hi - lo) / 2];
        double key = t->points[3LL * pivot + axis];
        i32 left = lo, right = hi;
        while (left <= right) {
            while (left <= hi) {
                i32 row = t->order[left];
                double value = t->points[3LL * row + axis];
                if (value > key || (value == key && row >= pivot)) {
                    break;
                }
                left++;
            }
            while (right >= lo) {
                i32 row = t->order[right];
                double value = t->points[3LL * row + axis];
                if (value < key || (value == key && row <= pivot)) {
                    break;
                }
                right--;
            }
            if (left <= right) {
                bench_swap(t->order + left++, t->order + right--);
            }
        }
        if (target <= right) {
            hi = right;
        } else if (target >= left) {
            lo = left;
        } else {
            return;
        }
    }
}

static i32 bench_tree_node(BenchTree *t, i32 begin, i32 end) {
    i32 id = t->count++;
    BenchNode *node = t->nodes + id;
    *node = (BenchNode){.begin = begin,
        .end = end,
        .left = -1,
        .right = -1,
        .low = {INFINITY, INFINITY, INFINITY},
        .high = {-INFINITY, -INFINITY, -INFINITY}};
    for (i32 i = begin; i < end; i++) {
        const f32 *p = t->points + 3LL * t->order[i];
        for (int d = 0; d < 3; d++) {
            node->low[d] = fminf(node->low[d], p[d]);
            node->high[d] = fmaxf(node->high[d], p[d]);
        }
    }
    if (end - begin > BENCH_LEAF) {
        int axis = 0;
        for (int d = 1; d < 3; d++) {
            if (node->high[d] - node->low[d] > node->high[axis] - node->low[axis]) {
                axis = d;
            }
        }
        i32 mid = begin + (end - begin) / 2;
        bench_select(t, axis, begin, end, mid);
        node->left = bench_tree_node(t, begin, mid);
        node->right = bench_tree_node(t, mid, end);
    }
    return id;
}

static double bench_box_distance(const BenchNode *n, const f32 *point) {
    double sum = 0;
    for (int d = 0; d < 3; d++) {
        double at = point[d],
               x = at < n->low[d] ? n->low[d] - at
            : at > n->high[d]     ? at - n->high[d]
                                  : 0;
        sum += x * x;
    }
    return sum;
}

static void bench_candidates_limit(const BenchTree *t, i32 node, const f32 *point, double cutoff,
    BenchCandidate *near, int *used, int limit) {
    const BenchNode *n = t->nodes + node;
    if (n->left < 0) {
        for (i32 i = n->begin; i < n->end; i++) {
            i32 id = t->order[i];
            const f32 *p = t->points + 3LL * id;
            double x = (double)point[0] - p[0], y = (double)point[1] - p[1];
            double z = (double)point[2] - p[2], d = x * x + y * y + z * z;
            if (d >= cutoff) {
                continue;
            }
            int at = *used;
            while (at &&
                (near[at - 1].squared > d ||
                    (near[at - 1].squared == d && near[at - 1].index > id))) {
                at--;
            }
            if (at == limit) {
                continue;
            }
            int next = *used < limit ? ++*used : limit;
            memmove(near + at + 1, near + at, (next - at - 1) * sizeof(*near));
            near[at] = (BenchCandidate){id, d};
        }
        return;
    }
    i32 a = n->left, b = n->right;
    double p = bench_box_distance(t->nodes + a, point);
    double q = bench_box_distance(t->nodes + b, point);
    if (p > q) {
        i32 swap = a;
        a = b;
        b = swap;
        double f = p;
        p = q;
        q = f;
    }
    if (p <= (*used == limit ? near[*used - 1].squared : cutoff)) {
        bench_candidates_limit(t, a, point, cutoff, near, used, limit);
    }
    if (q <= (*used == limit ? near[*used - 1].squared : cutoff)) {
        bench_candidates_limit(t, b, point, cutoff, near, used, limit);
    }
}

static void bench_candidates(const BenchTree *t, i32 node, const f32 *point, double cutoff,
    BenchCandidate *near, int *used) {
    bench_candidates_limit(t, node, point, cutoff, near, used, BENCH_NEIGHBORS);
}

typedef struct {
    BenchTree tree;
    const f32 *queries, *normals;
    int query_normals, limit;
    double depth, across, cutoff;
    i64 query_count;
    BenchMatches *out;
} BenchSearch;

static void bench_fit_query(void *context, i64 block) {
    BenchSearch *s = context;
    for (i64 row = block * 128; row < (block + 1) * 128 && row < s->query_count; row++) {
        const f32 *point = s->queries + 3 * row;
        BenchMatches *m = s->out + row;
        for (int i = 0; i < BENCH_KEEP; i++) {
            m->index[i] = -1;
            m->depth[i] = INFINITY;
        }
        if (!s->tree.count || !bench_finite(point) ||
            (s->query_normals && !bench_finite(s->normals + 3 * row))) {
            continue;
        }
        BenchCandidate near[BENCH_NEIGHBORS];
        double depth[BENCH_KEEP] = {INFINITY, INFINITY, INFINITY};
        int used = 0;
        bench_candidates_limit(&s->tree, 0, point, s->cutoff, near, &used, s->limit);
        for (int i = 0; i < used; i++) {
            i32 p = near[i].index;
            const f32 *from = s->tree.points + 3LL * p;
            const f32 *normal = s->normals + 3 * (s->query_normals ? row : p);
            double dot = ((double)point[0] - from[0]) * normal[0] +
                ((double)point[1] - from[1]) * normal[1] + ((double)point[2] - from[2]) * normal[2];
            double along = fabs(dot), through = near[i].squared - dot * dot;
            if (!isfinite(along) || along > s->depth || fmax(through, 0) > s->across) {
                continue;
            }
            int slot = BENCH_KEEP - 1;
            if (along >= depth[slot]) {
                continue;
            }
            while (slot > 0 && along < depth[slot - 1]) {
                depth[slot] = depth[slot - 1];
                m->index[slot] = m->index[slot - 1];
                slot--;
            }
            depth[slot] = along;
            m->index[slot] = p;
        }
        for (int i = 0; i < BENCH_KEEP; i++) {
            m->depth[i] = depth[i];
        }
    }
}

static BenchTree bench_tree(Arena *a, const f32 *points, i64 n) {
    BenchTree t = {.points = points,
        .order = push(a, n * sizeof(i32)),
        .nodes = push(a, (n / (BENCH_LEAF / 2) * 2 + 4) * sizeof(BenchNode))};
    i32 count = 0;
    for (i64 i = 0; i < n; i++) {
        if (bench_finite(points + 3 * i)) {
            t.order[count++] = i;
        }
    }
    if (count) {
        bench_tree_node(&t, 0, count);
    }
    return t;
}

static BenchMatches *bench_on_sheet_limit(Arena *a, const f32 *points, i64 n, const f32 *queries,
    i64 m, const f32 *normals, double depth, double across, int query_normals, int limit) {
    need(n >= 0 && n <= INT32_MAX && m >= 0, "too many bench points");
    need(limit > 0 && limit <= BENCH_NEIGHBORS, "invalid bench neighbour count");
    BenchSearch search = {.limit = limit,
        .tree = {.points = points},
        .queries = queries,
        .normals = normals,
        .query_normals = query_normals,
        .depth = depth,
        .across = across * across,
        .cutoff = hypot(depth, across),
        .query_count = m,
        .out = push(a, m * sizeof(BenchMatches))};
    search.cutoff *= search.cutoff;
    size_t saved = a->used;
    search.tree = bench_tree(a, points, n);
    parallel((m + 127) / 128, 4, bench_fit_query, &search);
    a->used = saved;
    return search.out;
}

static BenchMatches *bench_on_sheet(Arena *a, const f32 *points, i64 n, const f32 *queries, i64 m,
    const f32 *normals, double depth, double across, int query_normals) {
    return bench_on_sheet_limit(
        a, points, n, queries, m, normals, depth, across, query_normals, BENCH_NEIGHBORS);
}
