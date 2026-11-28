#ifndef HX_UNITY
const char hx_usage[] = "Fit band windows and compare detector and solver patch features.";
#endif

#include <glob.h>

enum { VIEW_FEATURES = 15, VIEW_PATCH = 16, VIEW_WINDOW = 32, VIEW_HIDDEN = 64 };

static const char *const view_names[VIEW_FEATURES] = {"nodes_per_cell", "prominence_p10",
    "prominence_p50", "prominence_p90", "width_p50", "coherence_p10", "coherence_p50", "tilt_p50",
    "kept", "edges_per_node", "edge_agreement", "edges_between_pieces", "in_pieces",
    "pieces_per_1000", "piece_size"};

static const char *const view_controls[] = {"PHerc0139-20250728140407-rows-0007-0019",
    "PHerc1667-20231117161658-rows-0008-0020", "PHerc0172-20241024131839-rows-0010-0022",
    "PHercParis4-20230205180739-rows-0010-0022"};

json_object *hx_solver_window_fit(const char *unit, const char *out, int first, int last);

typedef struct {
    i64 raw, kept, edges, occupied, in_pieces, between, distinct;
    i64 start[3], next[3];
} ViewPatch;

typedef struct {
    double *rows;
    i64 count, capacity;
} ViewRows;

static int view_double_order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int view_pair_order(const void *a, const void *b) {
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return (x > y) - (x < y);
}

static double view_quantile(double *values, i64 n, double q) {
    qsort(values, n, sizeof(*values), view_double_order);
    return realism_percentile(values, n, q);
}

static void view_append(ViewRows *out, const double *sample) {
    if (out->count == out->capacity) {
        i64 capacity = out->capacity ? out->capacity * 2 : 128;
        double *rows = realloc(out->rows, capacity * VIEW_FEATURES * sizeof(*rows));
        need(rows != NULL, "cannot grow solver view patch rows");
        out->capacity = capacity;
        out->rows = rows;
    }
    memcpy(out->rows + out->count++ * VIEW_FEATURES, sample, VIEW_FEATURES * sizeof(*sample));
}

static i32 view_patch(i64 cell, int gy, int gx) {
    int py = (gy + VIEW_PATCH - 1) / VIEW_PATCH, px = (gx + VIEW_PATCH - 1) / VIEW_PATCH;
    i64 z = cell / ((i64)gy * gx), y = cell / gx % gy, x = cell % gx;
    i64 patch = ((z / VIEW_PATCH) * py + y / VIEW_PATCH) * px + x / VIEW_PATCH;
    need(cell >= 0 && patch < INT32_MAX, "invalid solver view cell index");
    return patch;
}

static i64 view_cell_boundary(const i32 *cell, i64 n, i64 value) {
    i64 left = 0, right = n;
    while (left < right) {
        i64 m = (left + right) / 2;
        if (cell[m] < value) {
            left = m + 1;
        } else {
            right = m;
        }
    }
    return left;
}

static i64 view_int(Array a, i64 i) {
    if (!strcmp(a.descr, "<i8")) {
        return ((const i64 *)a.data)[i];
    }
    need(!strcmp(a.descr, "<i4"), "unexpected solver archive type %s", a.descr);
    return ((const i32 *)a.data)[i];
}

static double view_normal(Array a, i64 i) {
    if (!strcmp(a.descr, "|i1")) {
        return ((const i8 *)a.data)[i];
    }
    if (!strcmp(a.descr, "<f4")) {
        return ((const f32 *)a.data)[i];
    }
    need(!strcmp(a.descr, "<f8"), "unexpected solver normal type %s", a.descr);
    return ((const double *)a.data)[i];
}

static void view_features(
    Arena *a, const char *control, const char *folder, const int *window, ViewRows *out) {
    size_t saved = a->used, total = 0;
    json_object *record = json_read(fmt(a, "%s/record.json", control));
    need(record != NULL, "no solver view control at %s", control);
    json_object *grid = jget(record, "grid"), *planes = jget(record, "cell_planes");
    int gy = json_object_get_int(json_object_array_get_idx(grid, 1));
    int gx = json_object_get_int(json_object_array_get_idx(grid, 2));
    int low = json_object_get_int(json_object_array_get_idx(planes, 0));
    int high = json_object_get_int(json_object_array_get_idx(planes, 1));
    json_object *rows = json_object_object_get(record, "rows");
    i64 base = rows ? (u64)json_object_get_int64(json_object_array_get_idx(rows, 0)) << 32 : 0;
    json_object_put(record);
    need(gy > 0 && gx > 0 && high > low, "invalid solver grid at %s", control);
    void *raw = NULL;
    i32 *cells = NULL;
    need(!hx_synthetic_raw_nodes(control, &raw, &cells, &total) && total <= INT32_MAX,
        "cannot read solver view nodes from %s", control);
    i64 per_plane = (i64)gy * gx;
    i64 zbase = window ? (window[0] - low) * per_plane : 0;
    i64 lo = view_cell_boundary(cells, total, zbase);
    i64 hi = window ? view_cell_boundary(cells, total, (window[1] - low) * per_plane) : (i64)total;
    if (hi <= lo) {
        free(cells);
        free(raw);
        a->used = saved;
        return;
    }
    i64 count = 0;
    i32 *group = push(a, (hi - lo) * sizeof(*group));
    for (i64 i = lo; i < hi; i++) {
        i64 p = view_patch(cells[i] - zbase, gy, gx);
        group[i - lo] = p;
        count = p >= count ? p + 1 : count;
    }
    ViewPatch *patch = pushz(a, count * sizeof(*patch));
    for (i64 i = lo; i < hi; i++) {
        ViewPatch *p = patch + group[i - lo];
        p->raw++;
        p->occupied += i == lo || cells[i] != cells[i - 1];
    }
    free(cells);
    const char *nodes = fmt(a, "%s/nodes.npz", folder), *edges = fmt(a, "%s/edges.npz", folder);
    Array ids = npz_get(a, nodes, "ids"), roots = npz_get(a, nodes, "root");
    Array normal = npz_get(a, nodes, "normal"), left = npz_get(a, edges, "a");
    Array right = npz_get(a, edges, "b");
    i64 n = elements(ids), edges_n = elements(left);
    need(n == elements(roots) && elements(normal) == n * 3 && edges_n == elements(right) &&
            n <= INT32_MAX && ids.ndim == 1 && normal.ndim == 2 && normal.shape[1] == 3,
        "invalid fitted nodes or edges at %s", folder);
    i32 *sizes = pushz(a, n * sizeof(*sizes)), *kept = push(a, n * sizeof(*kept));
    i32 *root = push(a, n * sizeof(*root));
    u64 *pairs = push(a, n * sizeof(*pairs));
    for (i64 i = 0; i < n; i++) {
        i64 id = view_int(ids, i) - base, r = view_int(roots, i);
        need(id >= lo && id < hi && r >= 0 && r < n, "fitted node lies outside %s", control);
        int p = group[id - lo];
        patch[p].kept++;
        sizes[r]++;
        kept[i] = p;
        root[i] = r;
        pairs[i] = (u64)p << 32 | (u32)r;
    }
    qsort(pairs, n, sizeof(*pairs), view_pair_order);
    for (i64 i = 0; i < n; i++) {
        ViewPatch *p = patch + kept[i];
        p->in_pieces += sizes[root[i]] >= 400;
        if (!i || pairs[i] != pairs[i - 1]) {
            patch[pairs[i] >> 32].distinct++;
        }
    }
    for (i64 i = 0; i < edges_n; i++) {
        i64 l = view_int(left, i), r = view_int(right, i);
        need(l >= 0 && l < n && r >= 0 && r < n, "invalid solver edge endpoint");
        ViewPatch *p = patch + kept[l];
        p->edges++;
        p->between += root[l] != root[r];
    }
    i64 size[3] = {0};
    for (i64 i = 0; i < count; i++) {
        ViewPatch *p = patch + i;
        if (p->raw < 200 || !p->kept || !p->edges) {
            continue;
        }
        for (int j = 0; j < 3; j++) {
            p->start[j] = p->next[j] = size[j];
        }
        size[0] += p->raw;
        size[1] += p->kept;
        size[2] += p->edges;
    }
    double *data = push(a, (4 * size[0] + size[1] + size[2]) * sizeof(*data));
    double *piece = data + 4 * size[0], *agreement = piece + size[1];
    const RealismNode *node = raw;
    for (i64 i = lo; i < hi; i++) {
        ViewPatch *p = patch + group[i - lo];
        if (p->raw < 200 || !p->kept || !p->edges) {
            continue;
        }
        i64 pos = p->next[0]++;
        double nx = node[i].nx, ny = node[i].ny, nz = node[i].nz;
        data[pos] = node[i].prominence;
        data[pos + size[0]] = node[i].width;
        data[pos + 2 * size[0]] = node[i].coherence;
        data[pos + 3 * size[0]] = fabs(nz) / fmax(sqrt(nx * nx + ny * ny + nz * nz), 1e-9);
    }
    free(raw);
    for (i64 i = 0; i < n; i++) {
        ViewPatch *p = patch + kept[i];
        if (p->raw >= 200 && p->edges) {
            piece[p->next[1]++] = log10((double)sizes[root[i]]);
        }
    }
    for (i64 i = 0; i < edges_n; i++) {
        i64 l = view_int(left, i), r = view_int(right, i);
        ViewPatch *p = patch + kept[l];
        if (p->raw < 200) {
            continue;
        }
        double dot = 0, nl = 0, nr = 0;
        for (int d = 0; d < 3; d++) {
            double x = view_normal(normal, l * 3 + d), y = view_normal(normal, r * 3 + d);
            dot += x * y;
            nl += x * x;
            nr += y * y;
        }
        agreement[p->next[2]++] = fabs(dot) / fmax(sqrt(nl) * sqrt(nr), 1e-9);
    }
    for (i64 i = 0; i < count; i++) {
        ViewPatch p = patch[i];
        if (p.raw < 200 || !p.kept || !p.edges) {
            continue;
        }
        double *prominence = data + p.start[0];
        double *coherence = prominence + 2 * size[0];
        qsort(prominence, p.raw, sizeof(*prominence), view_double_order);
        qsort(coherence, p.raw, sizeof(*coherence), view_double_order);
        double values[VIEW_FEATURES] = {(double)p.raw / p.occupied,
            realism_percentile(prominence, p.raw, 0.1), realism_percentile(prominence, p.raw, 0.5),
            realism_percentile(prominence, p.raw, 0.9),
            view_quantile(prominence + size[0], p.raw, 0.5),
            realism_percentile(coherence, p.raw, 0.1), realism_percentile(coherence, p.raw, 0.5),
            view_quantile(prominence + 3 * size[0], p.raw, 0.5), (double)p.kept / p.raw,
            (double)p.edges / p.kept, view_quantile(agreement + p.start[2], p.edges, 0.5),
            (double)p.between / p.edges, (double)p.in_pieces / p.kept, 1000.0 * p.distinct / p.kept,
            view_quantile(piece + p.start[1], p.kept, 0.5)};
        int valid = 1;
        for (int j = 0; j < VIEW_FEATURES; j++) {
            valid &= isfinite(values[j]);
        }
        if (valid) {
            view_append(out, values);
        }
    }
    a->used = saved;
}

typedef struct {
    const char *name;
    i64 first, count;
    int label, fold;
} ViewGroup;

static int view_group_order(const void *left, const void *right) {
    const ViewGroup *const *a = left, *const *b = right;
    return strcmp((*a)->name, (*b)->name);
}

static u32 view_interval(Pcg64 *rng, u32 max) {
    u32 mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    u32 n;
    do {
        n = pcg64_next32(rng) & mask;
    } while (n > max);
    return n;
}

static int view_folds(Arena *a, ViewGroup *group, int groups, int *fold, u8 *label) {
    size_t saved = a->used;
    ViewGroup **order = push(a, groups * sizeof(*order));
    int n[2] = {0};
    for (int i = 0; i < groups; i++) {
        if (group[i].count) {
            n[group[i].label]++;
        }
    }
    int count = n[0] < n[1] ? n[0] : n[1];
    count = count < 5 ? count : 5;
    need(count >= 2, "solver comparison needs at least two nonempty windows and bands");
    Pcg64 rng = pcg64(0);
    for (int side = 1; side >= 0; side--) {
        int used = 0;
        for (int i = 0; i < groups; i++) {
            if (group[i].count && group[i].label == side) {
                order[used++] = &group[i];
            }
        }
        qsort(order, used, sizeof(*order), view_group_order);
        for (int i = used - 1; i > 0; i--) {
            u32 j = view_interval(&rng, i);
            ViewGroup *one = order[i];
            order[i] = order[j];
            order[j] = one;
        }
        for (int i = 0; i < used; i++) {
            ViewGroup *g = order[i];
            g->fold = i % count;
            for (i64 j = g->first, end = g->first + g->count; j < end; j++) {
                fold[j] = g->fold;
                label[j] = g->label;
            }
        }
    }
    a->used = saved;
    return count;
}

static void view_scale(double *x, i64 n, int d, const int *fold, int f) {
    double mean[VIEW_FEATURES] = {0}, variance[VIEW_FEATURES] = {0};
    i64 count = 0;
    for (i64 i = 0; i < n; i++) {
        if (!fold || fold[i] != f) {
            count++;
            for (int j = 0; j < d; j++) {
                mean[j] += x[i * d + j];
            }
        }
    }
    for (int j = 0; j < d; j++) {
        mean[j] /= count;
    }
    for (i64 i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            double *z = x + i * d + j;
            *z -= mean[j];
            if (!fold || fold[i] != f) {
                variance[j] += *z * *z;
            }
        }
    }
    for (int j = 0; j < d; j++) {
        variance[j] = sqrt(variance[j] / count) + 1e-9;
    }
    for (i64 i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            x[i * d + j] /= variance[j];
        }
    }
}

static double view_linear(const double *x, const double *w, int d) {
    double linear = w[d];
    for (int j = 0; j < d; j++) {
        linear += x[j] * w[j];
    }
    return linear;
}

static void view_logistic(
    const double *x, const u8 *label, const int *fold, int f, i64 n, int d, double *w) {
    int width = d + 1;
    for (int iter = 0; iter < 300; iter++) {
        double h[(VIEW_FEATURES + 1) * (VIEW_FEATURES + 1)] = {0};
        double g[VIEW_FEATURES + 1] = {0};
        for (i64 i = 0; i < n; i++) {
            if (fold[i] == f) {
                continue;
            }
            const double *v = x + i * d;
            double p = 1 / (1 + exp(-syn_clip(view_linear(v, w, d), -30, 30)));
            double error = p - label[i], variance = p * (1 - p);
            for (int j = 0; j < width; j++) {
                double z = j == d ? 1 : v[j];
                g[j] += z * error;
                for (int k = 0; k <= j; k++) {
                    h[j * width + k] += z * variance * (k == d ? 1 : v[k]);
                }
            }
        }
        for (int j = 0; j < width; j++) {
            if (j < d) {
                g[j] += w[j];
                h[j * width + j] += 1;
            }
            h[j * width + j] += 1e-9;
            for (int k = 0; k < j; k++) {
                h[k * width + j] = h[j * width + k];
            }
        }
        int pivot[VIEW_FEATURES + 1], one = 1, info;
        dgesv_(&width, &one, h, &width, pivot, g, &width, &info);
        need(!info, "cannot fit solver view logistic regression");
        double maximum = 0;
        for (int j = 0; j < width; j++) {
            w[j] -= g[j];
            maximum = fmax(maximum, fabs(g[j]));
        }
        if (maximum < 1e-8) {
            break;
        }
    }
}

typedef struct {
    u32 state[624];
    int index;
} ViewMt;

static ViewMt view_mt(u32 seed) {
    ViewMt mt = {.state = {seed}, .index = 624};
    for (u32 i = 1; i < 624; i++) {
        u32 last = mt.state[i - 1];
        mt.state[i] = 1812433253u * (last ^ (last >> 30)) + i;
    }
    return mt;
}

static u32 view_mt_next(ViewMt *mt) {
    if (mt->index == 624) {
        for (int i = 0; i < 624; i++) {
            u32 x = (mt->state[i] & 0x80000000u) | (mt->state[(i + 1) % 624] & 0x7fffffffu);
            mt->state[i] = mt->state[(i + 397) % 624] ^ (x >> 1) ^ (x & 1 ? 0x9908b0dfu : 0);
        }
        mt->index = 0;
    }
    u32 value = mt->state[mt->index++];
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    return value ^ (value >> 18);
}

void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
    const float *beta, float *c, const int *ldc);

static void view_gemm(
    int m, int n, int k, const float *a, int at, const float *b, int bt, float *c) {
    char ma = at ? 'T' : 'N', mb = bt ? 'T' : 'N';
    int lda = at ? m : k, ldb = bt ? k : n;
    float alpha = 1, beta = 0;
    sgemm_(&mb, &ma, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &n);
}

typedef struct {
    int in, out, n;
    f32 *param, *grad, *mean, *var;
} ViewLayer;

static ViewLayer view_layer(Arena *a, int in, int out, ViewMt *rng) {
    int n = out * (in + 1);
    ViewLayer l = {.in = in,
        .out = out,
        .n = n,
        .param = push(a, n * sizeof(f32)),
        .grad = push(a, n * sizeof(f32)),
        .mean = pushz(a, n * sizeof(f32)),
        .var = pushz(a, n * sizeof(f32))};
    f32 bound = 1.0 / sqrt(in);
    for (int i = 0; i < n; i++) {
        f32 fraction = (view_mt_next(rng) & 0xffffffu) * 0x1p-24f;
        l.param[i] = fmaf(fraction, 2 * bound, -bound);
    }
    return l;
}

static void view_forward(ViewLayer l, const f32 *x, int batch, f32 *z, int relu) {
    view_gemm(batch, l.out, l.in, x, 0, l.param, 1, z);
    const f32 *bias = l.param + l.in * l.out;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < l.out; j++) {
            f32 *out = z + i * l.out + j;
            *out += bias[j];
            if (relu && *out < 0) {
                *out = 0;
            }
        }
    }
}

static void view_backward(ViewLayer l, const f32 *x, int batch, const f32 *dz, f32 *dx) {
    view_gemm(l.out, l.in, batch, dz, 1, x, 0, l.grad);
    f32 *bias = l.grad + l.in * l.out;
    memset(bias, 0, l.out * sizeof(*bias));
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < l.out; j++) {
            bias[j] += dz[i * l.out + j];
        }
    }
    if (dx) {
        view_gemm(batch, l.in, l.out, dz, 0, l.param, 0, dx);
        for (int i = 0; i < batch * l.in; i++) {
            if (x[i] <= 0) {
                dx[i] = 0;
            }
        }
    }
}

static void view_adam(ViewLayer l, double step, double scale) {
    for (int i = 0; i < l.n; i++) {
        f32 g = l.grad[i] + 1e-4f * l.param[i];
        l.mean[i] += 0.1f * (g - l.mean[i]);
        l.var[i] = 0.999f * l.var[i] + 0.001f * (g * g);
        f32 denom = sqrtf(l.var[i]) / (f32)scale + 1e-8f;
        l.param[i] -= (f32)step * l.mean[i] / denom;
    }
}

static void view_network(Arena *a, const double *x, const u8 *label, const int *fold, int f, i64 n,
    int d, RealismRank *scores) {
    enum { VIEW_BATCH = 256, H = VIEW_HIDDEN };
    size_t saved = a->used;
    i32 *train = push(a, n * sizeof(*train));
    i64 count = 0, positive = 0;
    for (i64 i = 0; i < n; i++) {
        if (fold[i] != f) {
            train[count++] = i;
            positive += label[i];
        }
    }
    f32 weight[2] = {(0.5 * count) / (count - positive), (0.5 * count) / positive};
    Pcg64 rng = pcg64(f);
    ViewMt mt = view_mt(f);
    ViewLayer layer[3];
    layer[0] = view_layer(a, d, H, &mt);
    layer[1] = view_layer(a, H, H, &mt);
    layer[2] = view_layer(a, H, 1, &mt);
    f32 *input = push(a, VIEW_BATCH * d * sizeof(*input));
    f32 *h1 = push(a, VIEW_BATCH * H * sizeof(*h1)), *h2 = push(a, VIEW_BATCH * H * sizeof(*h2));
    f32 *v1 = push(a, VIEW_BATCH * H * sizeof(*v1)), *v2 = push(a, VIEW_BATCH * H * sizeof(*v2));
    for (int step = 1; step <= 1500; step++) {
        u8 y[VIEW_BATCH];
        f32 z[VIEW_BATCH], dz[VIEW_BATCH];
        for (int i = 0; i < VIEW_BATCH; i++) {
            int j = train[draw(&rng, count)];
            y[i] = label[j];
            for (int k = 0; k < d; k++) {
                input[i * d + k] = x[j * d + k];
            }
        }
        view_forward(layer[0], input, VIEW_BATCH, h1, 1);
        view_forward(layer[1], h1, VIEW_BATCH, h2, 1);
        view_forward(layer[2], h2, VIEW_BATCH, z, 0);
        for (int i = 0; i < VIEW_BATCH; i++) {
            f32 p = 1 / (1 + expf(-z[i]));
            dz[i] = (p - y[i]) * weight[y[i]] / VIEW_BATCH;
        }
        view_backward(layer[2], h2, VIEW_BATCH, dz, v2);
        view_backward(layer[1], h1, VIEW_BATCH, v2, v1);
        view_backward(layer[0], input, VIEW_BATCH, v1, NULL);
        double lr = 0.001 / (1 - pow(0.9, step)), scale = sqrt(1 - pow(0.999, step));
        for (int i = 0; i < 3; i++) {
            view_adam(layer[i], lr, scale);
        }
    }
    int batch = 0;
    i64 position = 0, ids[VIEW_BATCH];
    for (i64 i = 0; i <= n; i++) {
        if (i < n && fold[i] == f) {
            ids[batch] = i;
            for (int j = 0; j < d; j++) {
                input[batch * d + j] = x[i * d + j];
            }
            batch++;
        }
        if (batch == VIEW_BATCH || (i == n && batch)) {
            f32 scores_f[VIEW_BATCH];
            view_forward(layer[0], input, batch, h1, 1);
            view_forward(layer[1], h1, batch, h2, 1);
            view_forward(layer[2], h2, batch, scores_f, 0);
            for (int j = 0; j < batch; j++) {
                scores[position++] = (RealismRank){scores_f[j], label[ids[j]]};
            }
            batch = 0;
        }
    }
    a->used = saved;
}

static double view_auc(Arena *a, const ViewRows *data, const u8 *label, const int *fold, int folds,
    int first, int d, int network) {
    size_t saved = a->used;
    i64 n = data->count;
    double *x = push(a, n * d * sizeof(*x));
    RealismRank *score = push(a, n * sizeof(*score));
    double total = 0;
    for (int f = 0; f < folds; f++) {
        i64 nn = 0, positive = 0;
        for (i64 i = 0; i < n; i++) {
            memcpy(x + i * d, data->rows + i * VIEW_FEATURES + first, d * sizeof(*x));
            if (fold[i] == f) {
                nn++;
                positive += label[i];
            }
        }
        view_scale(x, n, d, fold, f);
        if (network) {
            view_network(a, x, label, fold, f, n, d, score);
        } else {
            double w[VIEW_FEATURES + 1] = {0};
            view_logistic(x, label, fold, f, n, d, w);
            i64 j = 0;
            for (i64 i = 0; i < n; i++) {
                if (fold[i] == f) {
                    score[j++] = (RealismRank){view_linear(x + i * d, w, d), label[i]};
                }
            }
        }
        total += nn * realism_auc(score, nn, positive);
    }
    a->used = saved;
    return total / n;
}

static double view_ks(const double *first, i64 n, const double *second, i64 m) {
    double maximum = 0;
    for (i64 i = 0, j = 0; i < n || j < m;) {
        double x = i < n ? first[i] : INFINITY, y = j < m ? second[j] : INFINITY;
        double key = fmin(x, y);
        while (i < n && first[i] <= key) {
            i++;
        }
        while (j < m && second[j] <= key) {
            j++;
        }
        maximum = fmax(maximum, fabs((double)i / n - (double)j / m));
    }
    return maximum;
}

static void view_number(json_object *out, const char *name, double value, int places) {
    double scale = places == 4 ? 10000 : 1000;
    json_object_object_add(out, name, jnum(nearbyint(value * scale) / scale));
}

static json_object *view_report(Arena *a, const char *scroll, ViewRows data, ViewGroup *groups,
    int windows, int bands, i64 real) {
    size_t saved = a->used;
    i64 other = data.count - real;
    u8 *label = push(a, data.count);
    int *fold = push(a, data.count * sizeof(*fold));
    int nfold = view_folds(a, groups, windows + bands, fold, label);
    json_object *out = json_object_new_object(), *per = json_object_new_object(),
                *names = json_object_new_array();
    double *values = push(a, data.count * sizeof(*values));
    for (int j = 0; j < VIEW_FEATURES; j++) {
        for (i64 i = 0; i < data.count; i++) {
            values[i] = data.rows[i * VIEW_FEATURES + j];
        }
        double a50 = view_quantile(values, real, 0.5),
               b50 = view_quantile(values + real, other, 0.5);
        json_object *one = json_object_new_object();
        view_number(one, "real_p50", a50, 4);
        view_number(one, "synthetic_p50", b50, 4);
        view_number(one, "ks", view_ks(values, real, values + real, other), 3);
        view_number(one, "auc", view_auc(a, &data, label, fold, nfold, j, 1, 0), 3);
        json_object_object_add(per, view_names[j], one);
    }
    int with_rows = 0;
    for (int i = 0; i < windows; i++) {
        with_rows += groups[i].count > 0;
    }
    for (int i = windows; i < windows + bands; i++) {
        json_object_array_add(names, json_object_new_string(groups[i].name));
    }
    json_object_object_add(out, "scroll", json_object_new_string(scroll));
    json_object_object_add(out, "real_windows", json_object_new_int(with_rows));
    json_object_object_add(out, "real_patches", json_object_new_int64(real));
    json_object_object_add(out, "bands", json_object_new_int(bands));
    json_object_object_add(out, "synthetic_patches", json_object_new_int64(other));
    view_number(
        out, "auc_logistic", view_auc(a, &data, label, fold, nfold, 0, VIEW_FEATURES, 0), 3);
    view_number(out, "auc_network", view_auc(a, &data, label, fold, nfold, 0, VIEW_FEATURES, 1), 3);
    view_number(out, "auc_detector_only", view_auc(a, &data, label, fold, nfold, 0, 8, 0), 3);
    view_number(
        out, "auc_fit_only", view_auc(a, &data, label, fold, nfold, 8, VIEW_FEATURES - 8, 0), 3);
    for (int i = 0; i < windows + bands; i++) {
        ViewGroup g = groups[i];
        if (g.count) {
            view_scale(data.rows + g.first * VIEW_FEATURES, g.count, VIEW_FEATURES, NULL, 0);
        }
    }
    view_number(
        out, "auc_within_band", view_auc(a, &data, label, fold, nfold, 0, VIEW_FEATURES, 0), 3);
    view_number(out, "auc_within_band_network",
        view_auc(a, &data, label, fold, nfold, 0, VIEW_FEATURES, 1), 3);
    json_object_object_add(out, "features", per);
    json_object_object_add(out, "band_names", names);
    a->used = saved;
    return out;
}

static const char *view_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static const char *view_control(Arena *a, const char *scroll) {
    size_t n = strlen(scroll);
    for (size_t j = 0; j < sizeof(view_controls) / sizeof(*view_controls); j++) {
        const char *control = view_controls[j];
        if (!strncmp(control, scroll, n) && control[n] == '-') {
            return fmt(a, "%s/controls/sheets/%s", LIBRARY, control);
        }
    }
    die("unknown solver view scroll %s", scroll);
    return NULL;
}

static const char *view_window_fit(
    Arena *a, const char *control, const char *out, int low, int high) {
    const char *name = view_basename(control);
    const char *folder = fmt(a, "%s/%s-p%d-%d", out, name, low, high);
    if (exists(fmt(a, "%s/nodes.npz", folder)) && exists(fmt(a, "%s/edges.npz", folder)) &&
        exists(fmt(a, "%s/window.json", folder))) {
        return folder;
    }
    mkdirs(folder);
    int lock = lock_file(fmt(a, "%s/.fit.lock", folder), 1, 1);
    need(lock >= 0, "cannot lock solver window %s", folder);
    const char *window = fmt(a, "%s/window.json", folder);
    if (!exists(window) || !exists(fmt(a, "%s/nodes.npz", folder)) ||
        !exists(fmt(a, "%s/edges.npz", folder))) {
        json_object *fit = hx_solver_window_fit(control, folder, low, high);
        need(fit != NULL, "cannot fit %s; check its axis and hand-set curl", folder);
        json_object *planes = json_object_new_array_ext(2);
        json_object_array_add(planes, json_object_new_int(low));
        json_object_array_add(planes, json_object_new_int(high));
        json_object_object_add(fit, "planes", planes);
        json_object_object_add(fit, "control", json_object_new_string(name));
        json_write(window, fit);
        printf("%s: %lld nodes\n", view_basename(folder), (long long)ji64(fit, "nodes"));
        fflush(stdout);
        json_object_put(fit);
    }
    close(lock);
    return folder;
}

static int syn_solver_view_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("synthetic.three_d.solver_view fit SCROLL | compare SCROLL [--bands N] "
             "[--root FOLDER] [--tag TAG]");
        return !argc;
    }
    int run_fit = !strcmp(argv[0], "fit");
    need(argc >= 2 && (run_fit || !strcmp(argv[0], "compare")),
        "expected fit or compare and SCROLL");
    Arena a = arena(32ull << 30);
    const char *scroll = argv[1], *control = view_control(&a, scroll);
    const char *out = fmt(&a, "%s/synthetic/realism/solver", OUTPUTS);
    const char *root = fmt(&a, "%s/synthetic/bands", OUTPUTS), *tag = "";
    int limit = 0;
    for (int i = 2; i < argc; i += 2) {
        need(!run_fit && i + 1 < argc, "invalid solver view option %s", argv[i]);
        if (!strcmp(argv[i], "--root")) {
            root = argv[i + 1];
        } else if (!strcmp(argv[i], "--tag")) {
            tag = argv[i + 1];
            need(!strchr(tag, '/'), "solver tag cannot contain a slash");
        } else if (!strcmp(argv[i], "--bands")) {
            char *end;
            long n = strtol(argv[i + 1], &end, 10);
            need(*argv[i + 1] && !*end && n >= 0 && n <= INT_MAX, "invalid band count");
            limit = n;
        } else {
            die("unknown solver view option %s", argv[i]);
        }
    }
    json_object *record = json_read(fmt(&a, "%s/record.json", control));
    need(record != NULL, "no solver view control at %s", control);
    json_object *span = jget(record, "cell_planes");
    int low = json_object_get_int(json_object_array_get_idx(span, 0));
    int high = json_object_get_int(json_object_array_get_idx(span, 1));
    int windows = high > low ? (high - low) / VIEW_WINDOW : 0;
    json_object_put(record);
    glob_t candidates = {0};
    if (!run_fit) {
        glob(fmt(&a, "%s/*", root), 0, NULL, &candidates);
    }
    need(candidates.gl_pathc <= INT32_MAX - (size_t)windows, "too many solver band folders");
    ViewGroup *group = pushz(&a, (windows + candidates.gl_pathc) * sizeof(*group));
    json_object *names = json_object_new_array();
    ViewRows data = {0};
    for (int j = 0; j < windows; j++) {
        int window[] = {low + j * VIEW_WINDOW, low + (j + 1) * VIEW_WINDOW};
        const char *folder = view_window_fit(&a, control, out, window[0], window[1]);
        const char *name = view_basename(folder);
        json_object_array_add(names, json_object_new_string(name));
        if (!run_fit) {
            i64 first = data.count;
            view_features(&a, control, folder, window, &data);
            group[j] =
                (ViewGroup){.name = name, .first = first, .count = data.count - first, .label = 1};
        }
    }
    if (run_fit) {
        puts(json_object_to_json_string_ext(names, JSON_C_TO_STRING_SPACED));
    } else {
        i64 real = data.count;
        int bands = 0;
        for (size_t j = 0; j < candidates.gl_pathc; j++) {
            const char *folder = candidates.gl_pathv[j];
            size_t saved = a.used;
            if (!exists(fmt(&a, "%s/truth.json", folder))) {
                a.used = saved;
                continue;
            }
            json_object *band = json_read(fmt(&a, "%s/band.json", folder));
            const char *look = json_object_get_string(json_object_object_get(band, "look"));
            int chosen = look && !strcmp(look, scroll);
            json_object_put(band);
            if (chosen) {
                i64 first = data.count;
                view_features(&a, fmt(&a, "%s/control", folder), folder, NULL, &data);
                group[windows + bands++] = (ViewGroup){
                    .name = view_basename(folder), .first = first, .count = data.count - first};
            }
            a.used = saved;
            if (limit && bands >= limit) {
                break;
            }
        }
        json_object *report = view_report(&a, scroll, data, group, windows, bands, real);
        const char *path = fmt(&a, "%s/compare-%s%s%s.json", out, scroll, *tag ? "-" : "", tag);
        json_write(path, report);
        json_object_object_del(report, "band_names");
        puts(json_object_to_json_string_ext(report, JSON_C_TO_STRING_PRETTY));
        json_object_put(report);
    }
    free(data.rows);
    json_object_put(names);
    globfree(&candidates);
    munmap(a.base, a.size);
    return 0;
}
