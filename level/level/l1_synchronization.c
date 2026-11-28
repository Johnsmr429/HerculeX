#include "network_simplex.c"

enum { WIND_WINDOW = 4, WIND_PASSES = 8, WIND_SCALE = 1024, WIND_CHUNK = 4000000 };
typedef struct {
    int32_t a, b, order;
    double k, w, bound;
} Vote;

static int by_vote(const void *a, const void *b) {
    const Vote *x = a, *y = b;
    if (x->a != y->a) {
        return x->a < y->a ? -1 : 1;
    }
    if (x->b != y->b) {
        return x->b < y->b ? -1 : 1;
    }
    if (x->k != y->k) {
        return x->k < y->k ? -1 : 1;
    }
    if (x->bound != y->bound) {
        return x->bound < y->bound ? -1 : 1;
    }
    return (x->order > y->order) - (x->order < y->order);
}

static int collect_votes(Vote *votes, int length, int chunk) {
    int ordered = 1;
    for (int i = 0; i < length; i++) {
        if (!(votes[i].k >= -0x1p63 && votes[i].k < 0x1p63) ||
            !(votes[i].bound >= -0x1p63 && votes[i].bound < 0x1p63)) {
            return -5;
        }
        votes[i].order = i;
        votes[i].k = trunc(votes[i].k);
        votes[i].bound = trunc(votes[i].bound);
        if (ordered && i && by_vote(votes + i - 1, votes + i) > 0) {
            ordered = 0;
        }
    }
    if (!ordered) {
        qsort(votes, length, sizeof(*votes), by_vote);
    }
    int used = 0;
    for (int i = 0, j; i < length; i = j) {
        Vote v = votes[i];
        int batch = v.order / chunk;
        double subtotal = 0, total = 0;
        for (j = i; j < length; j++) {
            Vote *w = votes + j;
            if (w->a != v.a || w->b != v.b || w->k != v.k || w->bound != v.bound) {
                break;
            }
            if (w->order / chunk != batch) {
                total += subtotal;
                subtotal = 0;
                batch = w->order / chunk;
            }
            subtotal += w->w;
        }
        v.w = total + subtotal;
        if (v.w > 1e-6) {
            votes[used++] = v;
        }
    }
    return used;
}

static int offsets_for(int n, const Vote *v, int length, double *out) {
    if (length < 0) {
        return length;
    }
    size_t capacity = (size_t)length * 2 + 1;
    int32_t *edge = malloc(capacity * 3 * sizeof(*edge));
    int64_t *caps = malloc(capacity * sizeof(*caps));
    int64_t *levels = calloc((size_t)n + 1, sizeof(*levels));
    int status = -1;
    if (!edge || !caps || !levels) {
        goto finish;
    }
    int32_t *head = edge + capacity, *costs = head + capacity;
    int64_t links = 0;
    for (int back = 0; back < 2; back++) {
        for (int i = 0; i < length; i++) {
            double scale = (back ? v[i].bound > 0 : v[i].bound < 0) ? 0.5 : 1.0;
            double cap = nearbyint((v[i].w * scale) * WIND_SCALE);
            if (cap <= 0 || !isfinite(cap) || cap >= 0x1p63) {
                continue;
            }
            edge[links] = back ? v[i].b : v[i].a;
            head[links] = back ? v[i].a : v[i].b;
            double cost = trunc(back ? -v[i].k : v[i].k);
            costs[links] = cost >= INT32_MIN && cost <= INT32_MAX ? (int32_t)cost : INT32_MIN;
            caps[links++] = (int64_t)cap;
        }
    }
    status = 0;
    if (n) {
        if ((int64_t)n + links >= INT32_MAX) {
            status = -3;
            goto finish;
        }
        int64_t pivots = hx_network_simplex(n, links, edge, head, costs, caps, levels, 2000000000);
        if (pivots < 0) {
            status = pivots == -2 ? -1 : pivots == -3 ? -3 : -2;
            goto finish;
        }
    }
    for (int i = 0; i < n; i++) {
        out[i] = (double)levels[i];
    }
finish:
    free(levels);
    free(caps);
    free(edge);
    return status;
}

static double term_cost(const Vote *v, const double *f) {
    double r = f[v->b] - f[v->a] - v->k;
    double scale = (r > 0 ? v->bound < 0 : v->bound > 0) ? 0.5 : 1.0;
    return (r > 0 ? scale * r : -scale * r) * v->w;
}

static double block_cost(const Vote *v, int n, const double *f) {
    if (n > 128) {
        int split = n / 2 / 8 * 8;
        return block_cost(v, split, f) + block_cost(v + split, n - split, f);
    }
    double accum[8] = {0};
    int i = 0;
    for (; i + 7 < n; i += 8) {
        for (int j = 0; j < 8; j++) {
            accum[j] += term_cost(v + i + j, f);
        }
    }
    double sum = ((accum[0] + accum[1]) + (accum[2] + accum[3])) +
        ((accum[4] + accum[5]) + (accum[6] + accum[7]));
    for (; i < n; i++) {
        sum += term_cost(v + i, f);
    }
    return sum;
}

static double objective(const Vote *v, int n, const double *f) {
    double sum = 0;
    for (int64_t i = 0; i < n; i += 8192) {
        int block = n - i < 8192 ? (int)(n - i) : 8192;
        sum += block_cost(v + i, block, f);
    }
    return sum;
}

static int components(int count, const Vote *v, int n, const int64_t *window, int32_t *out) {
    int32_t *parent = malloc(((size_t)count + 1) * sizeof(*parent));
    if (!parent) {
        return -1;
    }
    for (int i = 0; i < count; i++) {
        parent[i] = i;
        out[i] = -1;
    }
    for (int j = 0; j < n; j++) {
        int a = v[j].a, b = v[j].b;
        if (window && window[a] != window[b]) {
            continue;
        }
        a = root_of(parent, a);
        b = root_of(parent, b);
        if (a != b) {
            if (a < b) {
                parent[b] = a;
            } else {
                parent[a] = b;
            }
        }
    }
    int blocks = 0;
    for (int i = 0; i < count; i++) {
        int root = root_of(parent, i);
        out[i] = root == i ? blocks++ : out[root];
    }
    free(parent);
    return blocks;
}

static int solve_window(const Vote *votes, int length, const int32_t *local, int n, int grounded,
    const double *fixed, Vote *selected, double *offset) {
    int used = 0;
    for (int j = 0; j < length; j++) {
        Vote v = votes[j];
        int a = local[v.a], b = local[v.b];
        if (a == b && (grounded || a < 0)) {
            continue;
        }
        if (a < 0 || b < 0) {
            if (!grounded) {
                continue;
            }
            v.k += a < 0 ? fixed[v.a] : -fixed[v.b];
        }
        v.a = a < 0 ? n : a;
        v.b = b < 0 ? n : b;
        selected[used++] = v;
    }
    if (grounded && !used) {
        return -4;
    }
    if (grounded) {
        used = collect_votes(selected, used, WIND_CHUNK);
    }
    int status = offsets_for(n + grounded, selected, used, offset);
    if (!status && grounded) {
        for (int i = 0; i < n; i++) {
            offset[i] -= offset[n];
        }
    }
    return status;
}

static int by_int64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static int64_t zone_of(int64_t z, int shift) {
    int64_t q = z / WIND_WINDOW, r = z % WIND_WINDOW + shift;
    return q + r / WIND_WINDOW - (r % WIND_WINDOW < 0);
}

static int window_pass(const Vote *v, int n, int count, const int64_t *zone, int shift,
    int grounded, double *f, int64_t *window, int64_t *keys, int32_t *local, Vote *selected,
    double *offset) {
    for (int i = 0; i < count; i++) {
        keys[i] = window[i] = zone_of(zone[i], shift);
    }
    qsort(keys, count, sizeof(*keys), by_int64);
    if (keys[0] == keys[count - 1] && !grounded) {
        return 1;
    }
    for (int g = 0; g < count;) {
        int64_t key = keys[g];
        do {
            g++;
        } while (g < count && keys[g] == key);
        int free_nodes = 0;
        for (int i = 0; i < count; i++) {
            local[i] = window[i] == key ? free_nodes++ : -1;
        }
        int error = solve_window(v, n, local, free_nodes, grounded, f, selected, offset);
        if (error) {
            return error;
        }
        for (int i = 0; i < count; i++) {
            if (local[i] >= 0) {
                f[i] = offset[local[i]];
            }
        }
    }
    return 0;
}

static int synchronize_votes(
    const Vote *v, int n, int count, const int64_t *zone, double *f, int32_t *labels) {
    int status = -1;
    int64_t *windows = malloc(((size_t)count + 1) * 2 * sizeof(*windows));
    int32_t *local = malloc(((size_t)count + 1) * sizeof(*local));
    Vote *selected = malloc(((size_t)n + 1) * sizeof(*selected));
    double *offset = malloc(((size_t)count + 1) * sizeof(*offset));
    if (!windows || !local || !selected || !offset || components(count, v, n, NULL, labels) < 0) {
        goto finish;
    }
    int64_t *keys = windows + count;
    status = window_pass(v, n, count, zone, 0, 0, f, windows, keys, local, selected, offset);
    if (status == 1) {
        status = offsets_for(count, v, n, f);
        goto finish;
    }
    if (status) {
        goto finish;
    }
    int blocks = components(count, v, n, windows, local);
    if (blocks < 0) {
        status = -1;
        goto finish;
    }
    int used = 0;
    for (int j = 0; j < n; j++) {
        Vote edge = v[j];
        if (local[edge.a] != local[edge.b]) {
            edge.k = edge.k + f[edge.a] - f[edge.b];
            edge.a = local[edge.a];
            edge.b = local[edge.b];
            selected[used++] = edge;
        }
    }
    if (!used) {
        status = -4;
        goto finish;
    }
    used = collect_votes(selected, used, WIND_CHUNK);
    status = offsets_for(blocks, selected, used, offset);
    if (status) {
        goto finish;
    }
    for (int i = 0; i < count; i++) {
        f[i] += offset[local[i]];
    }
    double best = objective(v, n, f);
    for (int r = 0; r < WIND_PASSES; r++) {
        int shift = (r % 2 + 1) * (WIND_WINDOW / 2);
        status =
            window_pass(v, n, count, zone, shift, 1, f, windows, keys, local, selected, offset);
        if (status) {
            break;
        }
        double now = objective(v, n, f);
        if (best - now < 1e-5 * best) {
            break;
        }
        best = now;
    }
finish:
    free(offset);
    free(selected);
    free(local);
    free(windows);
    return status;
}

enum { WIND_AGGREGATE, WIND_OFFSETS, WIND_SUBSET, WIND_SYNCHRONIZE };

int hx_wind_solve(int mode, int count, int length, const int64_t *a, const int64_t *b,
    const double *k, const double *w, const double *bound, int chunk, const void *selection,
    const double *initial, int64_t *terms, double *output, int32_t *labels) {
    if (count < 0 || count >= INT32_MAX - 1 || length < 0 || length == INT32_MAX || chunk < 1) {
        return -5;
    }
    if (!length && (mode == WIND_AGGREGATE || mode == WIND_SYNCHRONIZE)) {
        return -4;
    }
    Vote *votes = malloc(((size_t)length + 1) * sizeof(*votes));
    if (!votes) {
        return -1;
    }
    int result;
    for (int i = 0; i < length; i++) {
        if (a[i] < 0 || a[i] >= count || b[i] < 0 || b[i] >= count) {
            free(votes);
            return -6;
        }
        votes[i] = (Vote){(int32_t)a[i], (int32_t)b[i], i, k[i], w[i], bound[i]};
    }
    if (mode == WIND_OFFSETS) {
        result = offsets_for(count, votes, length, output);
    } else if (mode == WIND_SUBSET) {
        Vote *selected = malloc(((size_t)length + 1) * sizeof(*selected));
        int32_t *local = malloc(((size_t)count + 1) * sizeof(*local));
        const uint8_t *on = selection;
        int used = 0;
        result = -1;
        if (selected && local) {
            for (int i = 0; i < count; i++) {
                local[i] = on[i] ? used++ : -1;
            }
            result = solve_window(votes, length, local, used, 1, initial, selected, output);
            if (!result) {
                result = used;
            }
        }
        free(local);
        free(selected);
    } else {
        result = collect_votes(votes, length, chunk);
        if (result >= 0 && mode == WIND_SYNCHRONIZE) {
            int status = synchronize_votes(votes, result, count, selection, output, labels);
            if (status) {
                result = status;
            }
        } else if (result >= 0) {
            for (int i = 0; i < result; i++) {
                terms[4 * (size_t)i] = votes[i].a;
                terms[4 * (size_t)i + 1] = votes[i].b;
                terms[4 * (size_t)i + 2] = (int64_t)votes[i].k;
                terms[4 * (size_t)i + 3] = (int64_t)votes[i].bound;
                output[i] = votes[i].w;
            }
        }
    }
    free(votes);
    return result;
}
