#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef int64_t Index;
static const double pi = 0x1.921fb54442d18p+1;

static int tangent(
    const double **points, const uint8_t *valid, Index i, Index previous, Index next, double *out) {
    int before = previous >= 0 && valid[previous], after = next >= 0 && valid[next];
    if (!before && !after)
        return 0;
    for (int a = 0; a < 3; a++) {
        const double *p = points[a];
        if (before && after)
            out[a] = ((p[next] - p[i]) + (p[i] - p[previous])) / 2;
        else
            out[a] = after ? p[next] - p[i] : p[i] - p[previous];
    }
    return 1;
}

void hx_reference_normals(const double *x, const double *y, const double *z, const uint8_t *v,
    Index h, Index w, float *out, uint8_t *keep) {
    const double *p[] = {x, y, z};
    for (Index r = 0; r < h; r++) {
        for (Index c = 0; c < w; c++) {
            Index i = r * w + c;
            double u[3], t[3];
            if (!v[i] || !tangent(p, v, i, c ? i - 1 : -1, c + 1 < w ? i + 1 : -1, u) ||
                !tangent(p, v, i, r ? i - w : -1, r + 1 < h ? i + w : -1, t))
                continue;
            double x = u[1] * t[2] - u[2] * t[1], y = u[2] * t[0] - u[0] * t[2],
                   z = u[0] * t[1] - u[1] * t[0];
            double length = sqrt(x * x + y * y + z * z);
            if (length > 1e-9) {
                keep[i] = 1;
                out[3 * i] = (float)(x / length);
                out[3 * i + 1] = (float)(y / length);
                out[3 * i + 2] = (float)(z / length);
            }
        }
    }
}

typedef struct {
    Index row, begin, end, up_begin, up_end, down_begin, down_end;
    int64_t shift, label;
} Run;

static Index runs_of(const uint8_t *mask, Index h, Index w, Run *runs, Index *first) {
    Index n = 0;
    for (Index r = 0; r < h; r++) {
        if (first)
            first[r] = n;
        const uint8_t *row = mask + r * w;
        for (Index c = 0; c < w;) {
            if (!row[c]) {
                c++;
                continue;
            }
            Index start = c++;
            while (c < w && row[c])
                c++;
            if (runs)
                runs[n] = (Run){.row = r, .begin = start, .end = c, .label = -1};
            n++;
        }
    }
    if (first)
        first[h] = n;
    return n;
}

Index hx_reference_runs(const uint8_t *mask, Index h, Index w, Index *out) {
    Index n = runs_of(mask, h, w, NULL, NULL);
    if (!out || !n) {
        return n;
    }
    Run *runs = (size_t)n <= SIZE_MAX / sizeof(*runs) ? malloc((size_t)n * sizeof(*runs)) : NULL;
    if (!runs) {
        return -1;
    }
    runs_of(mask, h, w, runs, NULL);
    for (Index i = 0; i < n; i++) {
        out[3 * i] = runs[i].row;
        out[3 * i + 1] = runs[i].begin;
        out[3 * i + 2] = runs[i].end;
    }
    free(runs);
    return n;
}

typedef struct {
    Index i, length;
} Seed;

static int longest_first(const void *a, const void *b) {
    const Seed *x = a, *y = b;
    if (x->length != y->length)
        return x->length > y->length ? -1 : 1;
    return (x->i > y->i) - (x->i < y->i);
}

static int number_order(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void unwrap_row(const double *theta, double *phi, Index begin, Index end) {
    double shift = 0;
    phi[begin] = theta[begin];
    for (Index c = begin + 1; c < end; c++) {
        double step = theta[c] - theta[c - 1];
        if (!(fabs(step) < pi)) {
            double d = fmod(step + pi, 2 * pi);
            if (d < 0)
                d += 2 * pi;
            d -= pi;
            if (d == -pi && step > 0)
                d = pi;
            shift += d - step;
        }
        phi[c] = theta[c] + shift;
    }
}

static int link_rows(Run *runs, const Index *first, Index h, Index w, const double *phi) {
    for (Index row = 0; row + 1 < h; row++) {
        Index i = first[row], j = first[row + 1];
        while (i < first[row + 1] && j < first[row + 2]) {
            Run *a = runs + i, *b = runs + j;
            Index lo = a->begin > b->begin ? a->begin : b->begin,
                  hi = a->end < b->end ? a->end : b->end;
            if (lo < hi) {
                if (a->down_end == 0)
                    a->down_begin = j;
                a->down_end = j + 1;
                if (b->up_end == 0)
                    b->up_begin = i;
                b->up_end = i + 1;
                for (Index c = lo; c < hi; c++) {
                    if (!isfinite(phi[row * w + c] - phi[(row + 1) * w + c]))
                        return -1;
                }
            }
            Index left = a->end, right = b->end;
            if (left <= right)
                i++;
            if (right <= left)
                j++;
        }
    }
    return 1;
}

static int connect(Run *runs, Index start, Index end, Run *a, Index w, const double *phi,
    double *temp, Index *queue, Index *tail) {
    for (Index j = start; j < end; j++) {
        Run *b = runs + j;
        if (b->label >= 0)
            continue;
        Index lo = a->begin > b->begin ? a->begin : b->begin,
              hi = a->end < b->end ? a->end : b->end;
        Index n = hi - lo;
        for (Index c = lo; c < hi; c++)
            temp[c - lo] = phi[a->row * w + c] - phi[b->row * w + c];
        if (n > 1)
            qsort(temp, (size_t)n, sizeof(*temp), number_order);
        double med = n % 2 ? temp[n / 2] : (temp[n / 2 - 1] + temp[n / 2]) / 2;
        double turns = nearbyint(med / (2 * pi));
        if (!(turns >= -0x1p63 && turns < 0x1p63))
            return -1;
        b->shift = (int64_t)((uint64_t)a->shift + (uint64_t)(int64_t)turns);
        b->label = a->label;
        queue[(*tail)++] = j;
    }
    return 1;
}

int hx_reference_unwrap(
    const double *theta, const uint8_t *valid, Index h, Index w, double *phi, int64_t *parts) {
    for (Index i = 0; i < h * w; i++) {
        phi[i] = NAN;
        parts[i] = -1;
    }
    Index n = runs_of(valid, h, w, NULL, NULL);
    if (!n)
        return 1;
    if ((size_t)n > SIZE_MAX / sizeof(Run) || (size_t)h >= SIZE_MAX / sizeof(Index))
        return 0;
    Run *runs = malloc((size_t)n * sizeof(*runs));
    Index *first = malloc((size_t)(h + 1) * sizeof(*first)),
          *queue = malloc((size_t)n * sizeof(*queue));
    Seed *seeds = malloc((size_t)n * sizeof(*seeds));
    double *temp = malloc((size_t)w * sizeof(*temp));
    int ok = runs && first && queue && seeds && temp;
    if (ok) {
        runs_of(valid, h, w, runs, first);
        for (Index i = 0; i < n; i++) {
            Run *r = runs + i;
            seeds[i] = (Seed){.i = i, .length = r->end - r->begin};
            unwrap_row(theta + r->row * w, phi + r->row * w, r->begin, r->end);
        }
        ok = link_rows(runs, first, h, w, phi);
        qsort(seeds, (size_t)n, sizeof(*seeds), longest_first);
    }
    int64_t label = 0;
    for (Index i = 0; ok == 1 && i < n; i++) {
        Index seed = seeds[i].i, tail = 1;
        if (runs[seed].label >= 0)
            continue;
        runs[seed].label = label++;
        queue[0] = seed;
        for (Index head = 0; ok == 1 && head < tail; head++) {
            Run *r = runs + queue[head];
            ok = connect(runs, r->up_begin, r->up_end, r, w, phi, temp, queue, &tail);
            if (ok == 1)
                ok = connect(runs, r->down_begin, r->down_end, r, w, phi, temp, queue, &tail);
        }
    }
    for (Index i = 0; ok == 1 && i < n; i++) {
        Run *r = runs + i;
        for (Index c = r->begin; c < r->end; c++) {
            Index at = r->row * w + c;
            parts[at] = r->label;
            phi[at] += r->shift * (2 * pi);
        }
    }
    free(temp);
    free(seeds);
    free(queue);
    free(first);
    free(runs);
    return ok;
}
