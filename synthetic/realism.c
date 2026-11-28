#define REALISM_SIDE 256
#define REALISM_N (REALISM_SIDE * REALISM_SIDE)
#define REALISM_FEATURES 31

static const char *realism_keys[REALISM_FEATURES] = {"p5", "p25", "p50", "p75", "p95", "spectrum0",
    "spectrum1", "spectrum2", "spectrum3", "spectrum4", "spectrum5", "spectrum6", "spectrum7",
    "coherence", "coherence_p25", "coherence_p75", "coherence_wide", "bright", "dark", "skew",
    "kurtosis", "pitch_px", "pitch_peak", "along_px", "sharpness", "noise", "peaks_per_100px",
    "peak_width_px", "peak_height", "valley", "double_share"};

typedef struct {
    double re, im;
} RealismComplex;

static void realism_fft(RealismComplex *v, const RealismComplex *phase, int reverse) {
    const int n = REALISM_SIDE;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            RealismComplex swap = v[i];
            v[i] = v[j];
            v[j] = swap;
        }
    }
    for (int stride = 1; stride < n; stride <<= 1) {
        int jump = n / (2 * stride);
        for (int base = 0; base < n; base += 2 * stride) {
            for (int k = 0; k < stride; k++) {
                RealismComplex a = v[base + k], b = v[base + k + stride], w = phase[k * jump];
                double sign = reverse ? -1 : 1;
                double re = b.re * w.re - b.im * (w.im * sign);
                double im = b.re * (w.im * sign) + b.im * w.re;
                v[base + k] = (RealismComplex){.re = a.re + re, .im = a.im + im};
                v[base + k + stride] = (RealismComplex){.re = a.re - re, .im = a.im - im};
            }
        }
    }
}

static void realism_fft2(RealismComplex *v, int reverse) {
    RealismComplex phase[REALISM_SIDE / 2], column[REALISM_SIDE];
    for (int k = 0; k < REALISM_SIDE / 2; k++) {
        double t = -2 * M_PI * k / REALISM_SIDE;
        phase[k] = (RealismComplex){.re = cos(t), .im = sin(t)};
    }
    for (int k = 0; k < REALISM_SIDE; k++) {
        realism_fft(v + k * REALISM_SIDE, phase, reverse);
    }
    for (int x = 0; x < REALISM_SIDE; x++) {
        for (int y = 0; y < REALISM_SIDE; y++) {
            column[y] = v[y * REALISM_SIDE + x];
        }
        realism_fft(column, phase, reverse);
        for (int y = 0; y < REALISM_SIDE; y++) {
            v[y * REALISM_SIDE + x] = column[y];
        }
    }
}

static void realism_blur(Arena *a, const double *in, double *out, double *scratch, double sigma,
    int y_order, int x_order) {
    syn_filter_order(a, in, scratch, REALISM_N, REALISM_SIDE, REALISM_SIDE, sigma, 4, 0, y_order);
    syn_filter_order(a, scratch, out, REALISM_N, 1, REALISM_SIDE, sigma, 4, 0, x_order);
}

static double realism_sample(const double *v, double y, double x) {
    if (y < 0 || y > REALISM_SIDE - 1 || x < 0 || x > REALISM_SIDE - 1) {
        return 0;
    }
    int y0 = floor(y), x0 = floor(x);
    int y1 = y0 + (y0 < REALISM_SIDE - 1), x1 = x0 + (x0 < REALISM_SIDE - 1);
    double w = y - y0, u = x - x0;
    return v[y0 * REALISM_SIDE + x0] * (1 - w) * (1 - u) + v[y0 * REALISM_SIDE + x1] * (1 - w) * u +
        v[y1 * REALISM_SIDE + x0] * w * (1 - u) + v[y1 * REALISM_SIDE + x1] * w * u;
}

static double realism_percentile(const double *sorted, size_t n, double q) {
    double at = q * (n - 1);
    size_t low = at, high = low + (low + 1 < n);
    double fraction = at - low, distance = sorted[high] - sorted[low];
    return fraction < 0.5 ? sorted[low] + distance * fraction
                          : sorted[high] - distance * (1 - fraction);
}

static void realism_profiles(
    Arena *a, const double *v, const u8 *mask, double y, double x, double *out) {
    size_t mark = a->used;
    double *height = push(a, 24 * 200 * sizeof(*height));
    double *width = push(a, 24 * 200 * sizeof(*width));
    double *valleys = push(a, 24 * 200 * sizeof(*valleys));
    double counts[24];
    size_t peaks = 0, gaps = 0, rows = 0, doubled = 0;
    for (int k = 0; k < 24; k++) {
        double shift = -80 + k * (160.0 / 23);
        double line[200];
        int kept = 0;
        for (int i = 0; i < 200; i++) {
            double t = i - 100, row = 128 - shift * x + t * y, col = 128 + shift * y + t * x;
            line[i] = realism_sample(v, row, col);
            if (row >= 0 && row <= 255 && col >= 0 && col <= 255) {
                int iy = floor(row + 0.5), ix = floor(col + 0.5);
                kept += mask[iy * REALISM_SIDE + ix] > 0;
            }
        }
        if (kept < 180) {
            continue;
        }
        int previous = -1, count = 0;
        for (int i = 1; i < 199; i++) {
            if (!(line[i] > line[i - 1] && line[i] >= line[i + 1])) {
                continue;
            }
            int begin = i >= 6 ? i - 6 : 0, end = i + 7 < 200 ? i + 7 : 200;
            double low = line[begin], high = line[i + 1];
            for (int j = begin + 1; j < i; j++) {
                low = fmin(low, line[j]);
            }
            for (int j = i + 2; j < end; j++) {
                high = fmin(high, line[j]);
            }
            double prominence = line[i] - fmax(low, high);
            if (prominence <= 0.15) {
                continue;
            }
            double h = line[i] - 0.5 * prominence;
            int l = i, r = i;
            while (l > 0 && line[l] > h) {
                l--;
            }
            while (r < 199 && line[r] > h) {
                r++;
            }
            double a0 = l < i ? l + (h - line[l]) / fmax(line[l + 1] - line[l], 1e-9) : l;
            double a1 = r > i ? r - (h - line[r]) / fmax(line[r - 1] - line[r], 1e-9) : r;
            height[peaks] = line[i];
            width[peaks++] = a1 - a0;
            if (previous >= 0) {
                double low = line[previous];
                for (int j = previous + 1; j < i; j++) {
                    low = fmin(low, line[j]);
                }
                valleys[gaps++] = low;
                doubled += i - previous <= 4;
            }
            previous = i;
            count++;
        }
        counts[rows++] = count * 0.5;
    }
    double *samples[] = {counts, width, height, valleys};
    size_t sizes[] = {rows, peaks, peaks, gaps};
    for (int i = 0; i < 4; i++) {
        qsort(samples[i], sizes[i], sizeof(*samples[i]), syn_double_order);
        out[i] = sizes[i] ? realism_percentile(samples[i], sizes[i], 0.5) : 0;
    }
    out[4] = (double)doubled / fmax(gaps, 1);
    a->used = mark;
}

static void realism_features(Arena *a, const u8 *crop, double out[REALISM_FEATURES]) {
    size_t mark = a->used, pixels = REALISM_N;
    double *v = push(a, pixels * sizeof(*v)), *sorted = push(a, pixels * sizeof(*sorted));
    size_t n = 0;
    for (size_t i = 0; i < pixels; i++) {
        if (crop[i]) {
            sorted[n++] = crop[i];
        }
    }
    need(n != 0, "empty CT realism crop");
    qsort(sorted, n, sizeof(*sorted), syn_double_order);
    double quantiles[] = {0.05, 0.25, 0.5, 0.75, 0.95};
    for (int i = 0; i < 5; i++) {
        out[i] = realism_percentile(sorted, n, quantiles[i]);
    }
    double scale = fmax(out[4] - out[0], 1), mean = 0;
    size_t bright = 0, dark = 0;
    for (size_t i = 0; i < pixels; i++) {
        v[i] = crop[i] ? (crop[i] - out[2]) / scale : 0;
        mean += v[i];
        bright += v[i] > 0.25;
        dark += v[i] < -0.15;
    }
    mean /= n;
    RealismComplex *plane = push(a, pixels * sizeof(*plane));
    double hann[REALISM_SIDE];
    for (int i = 0; i < REALISM_SIDE; i++) {
        hann[i] = 0.5 - 0.5 * cos(2 * M_PI * i / (REALISM_SIDE - 1));
    }
    for (size_t i = 0; i < pixels; i++) {
        double window = hann[i / REALISM_SIDE] * hann[i % REALISM_SIDE];
        plane[i] = (RealismComplex){.re = (v[i] - mean * (crop[i] > 0)) * window};
    }
    realism_fft2(plane, 0);
    const double bands[] = {0x1p-7, 0x1.ae89f995ad3afp-7, 0x1.6a09e667f3bcep-6,
        0x1.306fe0a31b715p-5, 0x1p-4, 0x1.ae89f995ad3aep-4, 0x1.6a09e667f3bcap-3,
        0x1.306fe0a31b714p-2, 0x1p-1};
    double bucket[8] = {0};
    int count[8] = {0};
    for (int y = 0; y < REALISM_SIDE; y++) {
        for (int x = 0; x < REALISM_SIDE; x++) {
            int fy = y < 128 ? y : y - REALISM_SIDE, fx = x < 128 ? x : x - REALISM_SIDE;
            double r = hypot(fy, fx) / REALISM_SIDE;
            for (int b = 0; b < 8; b++) {
                if (r >= bands[b] && r < bands[b + 1]) {
                    RealismComplex z = plane[y * REALISM_SIDE + x];
                    bucket[b] += z.re * z.re + z.im * z.im;
                    count[b]++;
                    break;
                }
            }
        }
    }
    for (int b = 0; b < 8; b++) {
        bucket[b] /= count[b];
    }
    double energy = syn_sum(bucket, 8);
    for (int b = 0; b < 8; b++) {
        bucket[b] = log(bucket[b] / energy);
    }
    double average = syn_sum(bucket, 8) / 8;
    for (int b = 0; b < 8; b++) {
        out[5 + b] = bucket[b] - average;
    }
    double *gy = push(a, pixels * sizeof(*gy)), *gx = push(a, pixels * sizeof(*gx));
    double *jyy = push(a, pixels * sizeof(*jyy)), *jxx = push(a, pixels * sizeof(*jxx));
    double *jxy = push(a, pixels * sizeof(*jxy)), *scratch = push(a, pixels * sizeof(*scratch));
    double *squares = push(a, pixels * sizeof(*squares));
    realism_blur(a, v, gy, scratch, 1.5, 1, 0);
    realism_blur(a, v, gx, scratch, 1.5, 0, 1);
    double sine = 0, cosine = 0;
    for (int t = 0; t < 2; t++) {
        double sigma = t == 0 ? 4 : 12;
        double *result[] = {jyy, jxx, jxy};
        const double *left[] = {gy, gx, gx}, *right[] = {gy, gx, gy};
        for (int j = 0; j < 3; j++) {
            for (size_t i = 0; i < pixels; i++) {
                squares[i] = left[j][i] * right[j][i];
            }
            realism_blur(a, squares, result[j], scratch, sigma, 0, 0);
        }
        size_t s = 0;
        for (size_t i = 0; i < pixels; i++) {
            if (!crop[i]) {
                continue;
            }
            double diff = jyy[i] - jxx[i], pair = jxy[i];
            sorted[s++] = sqrt(diff * diff + 4 * pair * pair) / fmax(jyy[i] + jxx[i], 1e-12);
            if (!t) {
                sine += pair;
                cosine += diff;
            }
        }
        double co = syn_sum(sorted, n) / n;
        if (!t) {
            qsort(sorted, n, sizeof(*sorted), syn_double_order);
            out[13] = co;
            out[14] = realism_percentile(sorted, n, 0.25);
            out[15] = realism_percentile(sorted, n, 0.75);
        } else {
            out[16] = co;
        }
    }
    double angle = 0.5 * atan2(2 * sine, cosine), ny = cos(angle), nx = sin(angle);
    double full_mean = mean * n / pixels;
    for (size_t i = 0; i < pixels; i++) {
        plane[i] = (RealismComplex){.re = v[i] - full_mean};
    }
    realism_fft2(plane, 0);
    for (size_t i = 0; i < pixels; i++) {
        plane[i] = (RealismComplex){.re = plane[i].re * plane[i].re + plane[i].im * plane[i].im};
    }
    realism_fft2(plane, 1);
    double zero = fmax(plane[0].re / pixels, 1e-12);
    for (size_t i = 0; i < pixels; i++) {
        size_t y = ((i / REALISM_SIDE) + 128) % REALISM_SIDE;
        size_t x = ((i % REALISM_SIDE) + 128) % REALISM_SIDE;
        squares[y * REALISM_SIDE + x] = plane[i].re / pixels / zero;
    }
    double across[64], along[64];
    for (int i = 0; i < 64; i++) {
        across[i] = realism_sample(squares, 128 + i * ny, 128 + i * nx);
        along[i] = realism_sample(squares, 128 - i * nx, 128 + i * ny);
    }
    int first = 63, end = 63;
    double ac = 0;
    for (int i = 1; i < 63; i++) {
        if (across[i] > across[i - 1] && across[i] >= across[i + 1]) {
            first = i;
            ac = across[i];
            break;
        }
    }
    for (int i = 0; i < 63; i++) {
        if (along[i] < 0.5) {
            end = i;
            break;
        }
    }
    realism_blur(a, v, jyy, scratch, 1, 2, 0);
    realism_blur(a, v, jxx, scratch, 1, 0, 2);
    realism_blur(a, v, jxy, scratch, 1, 0, 0);
    double lap = 0, grad = 0, ns = 0, n2 = 0, s2 = 0, s3 = 0, s4 = 0;
    for (size_t i = 0; i < pixels; i++) {
        if (crop[i]) {
            double d = v[i] - mean, noise = v[i] - jxy[i];
            lap += fabs(jyy[i] + jxx[i]);
            grad += hypot(gy[i], gx[i]);
            ns += noise;
            n2 += noise * noise;
            s2 += d * d;
            s3 += d * d * d;
            s4 += d * d * d * d;
        }
    }
    double standard = sqrt(s2 / n) + 1e-12;
    out[17] = (double)bright / n;
    out[18] = (double)dark / n;
    out[19] = (s3 / n) / pow(standard, 3);
    out[20] = (s4 / n) / pow(standard, 4);
    out[21] = first;
    out[22] = ac;
    out[23] = end;
    out[24] = (lap / n) / fmax(grad / n, 1e-9);
    out[25] = sqrt(fmax(n2 / n - (ns / n) * (ns / n), 0));
    realism_profiles(a, v, crop, ny, nx, out + 26);
    a->used = mark;
}

void dgesv_(const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, double *b,
    const int *ldb, int *info);

typedef struct {
    double score;
    int label;
} RealismRank;

static int realism_rank_order(const void *left, const void *right) {
    const RealismRank *a = left, *b = right;
    return (a->score > b->score) - (a->score < b->score);
}

static double realism_auc(RealismRank *ranks, size_t count, size_t positives) {
    qsort(ranks, count, sizeof(*ranks), realism_rank_order);
    double sum = 0;
    for (size_t first = 0; first < count;) {
        size_t last = first + 1, labels = ranks[first].label;
        while (last < count && ranks[last].score == ranks[first].score) {
            labels += ranks[last++].label;
        }
        sum += labels * (0.5 * (first + last + 1));
        first = last;
    }
    return (sum - 0.5 * positives * (positives + 1)) / fmax(positives * (count - positives), 1);
}

static void realism_measure(const double *rows, size_t count, double *mean, double *std) {
    memset(mean, 0, REALISM_FEATURES * sizeof(*mean));
    memset(std, 0, REALISM_FEATURES * sizeof(*std));
    for (size_t i = 0; i < count; i++) {
        for (int j = 0; j < REALISM_FEATURES; j++) {
            mean[j] += rows[i * REALISM_FEATURES + j];
        }
    }
    for (int j = 0; j < REALISM_FEATURES; j++) {
        mean[j] /= count;
    }
    for (size_t i = 0; i < count; i++) {
        for (int j = 0; j < REALISM_FEATURES; j++) {
            double d = rows[i * REALISM_FEATURES + j] - mean[j];
            std[j] += d * d;
        }
    }
    for (int j = 0; j < REALISM_FEATURES; j++) {
        std[j] = sqrt(std[j] / count);
    }
}

static double realism_separability(
    Arena *a, const double *real, size_t nr, const double *syn, size_t ns, int first) {
    size_t mark = a->used, count = nr + ns;
    int d = REALISM_FEATURES - first + 1;
    double *x = push(a, count * d * sizeof(*x));
    u8 *fold = push(a, count);
    RealismRank *row = push(a, count * sizeof(*row));
    Pcg64 rng = pcg64(0);
    for (size_t i = 0; i < count; i++) {
        fold[i] = draw(&rng, 5);
        row[i] = (RealismRank){.label = i < nr};
    }
    for (int f = 0; f < 5; f++) {
        double mean[REALISM_FEATURES] = {0}, std[REALISM_FEATURES] = {0};
        size_t train = 0;
        for (size_t i = 0; i < count; i++) {
            const double *raw =
                i < nr ? real + i * REALISM_FEATURES : syn + (i - nr) * REALISM_FEATURES;
            memcpy(x + i * d, raw + first, (d - 1) * sizeof(*x));
            x[i * d + d - 1] = 1;
            if (fold[i] != f) {
                train++;
                for (int k = 0; k < d - 1; k++) {
                    mean[k] += raw[k + first];
                }
            }
        }
        if (!train) {
            continue;
        }
        for (int k = 0; k < d - 1; k++) {
            mean[k] /= train;
        }
        for (size_t i = 0; i < count; i++) {
            for (int k = 0; k < d - 1; k++) {
                double *v = x + i * d + k;
                *v -= mean[k];
                if (fold[i] != f) {
                    std[k] += *v * *v;
                }
            }
        }
        for (int k = 0; k < d - 1; k++) {
            std[k] = sqrt(std[k] / train) + 1e-9;
        }
        for (size_t i = 0; i < count; i++) {
            for (int k = 0; k < d - 1; k++) {
                x[i * d + k] /= std[k];
            }
        }
        double w[REALISM_FEATURES + 1] = {0};
        for (int iter = 0; iter < 300; iter++) {
            enum { WIDTH = REALISM_FEATURES + 1 };
            double h[WIDTH * WIDTH] = {0}, g[WIDTH] = {0};
            for (size_t i = 0; i < count; i++) {
                if (fold[i] == f) {
                    continue;
                }
                const double *v = x + i * d;
                double linear = 0;
                for (int j = 0; j < d; j++) {
                    linear += v[j] * w[j];
                }
                double p = 1 / (1 + exp(-syn_clip(linear, -30, 30)));
                double error = p - row[i].label, variance = p * (1 - p);
                for (int j = 0; j < d; j++) {
                    g[j] += v[j] * error;
                    for (int k = 0; k <= j; k++) {
                        h[j * d + k] += v[j] * (v[k] * variance);
                    }
                }
            }
            for (int j = 0; j < d; j++) {
                if (j < d - 1) {
                    g[j] += w[j];
                    h[j * d + j] += 1;
                }
                h[j * d + j] += 1e-9;
                for (int k = 0; k < j; k++) {
                    h[k * d + j] = h[j * d + k];
                }
            }
            int pivot[WIDTH], one = 1, info;
            dgesv_(&d, &one, h, &d, pivot, g, &d, &info);
            need(!info, "cannot fit CT realism classifier");
            double maximum = 0;
            for (int j = 0; j < d; j++) {
                w[j] -= g[j];
                maximum = fmax(maximum, fabs(g[j]));
            }
            if (maximum < 1e-8) {
                break;
            }
        }
        for (size_t i = 0; i < count; i++) {
            if (fold[i] == f) {
                for (int j = 0; j < d; j++) {
                    row[i].score += x[i * d + j] * w[j];
                }
            }
        }
    }
    double auc = realism_auc(row, count, nr);
    a->used = mark;
    return auc;
}

static json_object *realism_compare(
    Arena *a, const double *real, size_t nr, const double *syn, size_t ns, double *distance) {
    double rm[REALISM_FEATURES], rs[REALISM_FEATURES], sm[REALISM_FEATURES], ss[REALISM_FEATURES];
    realism_measure(real, nr, rm, rs);
    realism_measure(syn, ns, sm, ss);
    json_object *differences = json_object_new_object();
    *distance = 0;
    for (int i = 0; i < REALISM_FEATURES; i++) {
        double std = fmax(rs[i], fmax(0.05 * fabs(rm[i]), 1e-3));
        double shift = (sm[i] - rm[i]) / std, spread = log(fmax(ss[i], 1e-3 * std) / std);
        *distance += shift * shift + 0.5 * spread * spread;
        syn_add(differences, realism_keys[i], nearbyint(100 * shift) / 100);
    }
    *distance /= REALISM_FEATURES;
    return differences;
}

#include "../scans/chunks/cached_ct.h"

int hx_synthetic_raw_nodes(const char *unit, void **nodes, int32_t **cells, size_t *count);

typedef struct {
    float x, y, z;
    i8 nx, ny, nz;
    u8 prominence, width, coherence;
    u16 pad;
} RealismNode;

typedef struct {
    const char *sample, *scan;
    const char *windows[3];
} RealismTraining;

static const RealismTraining realism_training[] = {
    {"PHerc0139", "20250728140407",
        {"PHerc0139-train", "PHerc0139-train-low", "PHerc0139-train-high"}},
    {"PHerc1667", "20231117161658", {"PHerc1667-train-2560", "PHerc1667-train-7168"}},
    {"PHerc0172", "20241024131839", {"PHerc0172-train-5760", "PHerc0172-train-9728"}},
    {"PHercParis4", "20230205180739", {"PHercParis4-train-3072", "PHercParis4-train-9216"}}};

static void realism_box(const i64 *centre, int across, i64 *box) {
    int height = REALISM_SIDE / 2;
    box[0] = centre[0] - (across ? height : 0);
    box[1] = centre[0] + (across ? height : 1);
    box[2] = centre[1] - (across ? 0 : height);
    box[3] = centre[1] + (across ? 1 : height);
    box[4] = centre[2] - height;
    box[5] = centre[2] + height;
}

static int realism_average(const u8 *raw, u8 *crop, int across, int stride) {
    size_t width = REALISM_SIDE * 2, height = across ? 2 : 2 * REALISM_SIDE;
    int nz = across ? REALISM_SIDE : 1, ny = across ? 1 : REALISM_SIDE;
    int filled = 0;
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < REALISM_SIDE; x++) {
                size_t at = ((size_t)(2 * z) * height + 2 * y) * width + 2 * x;
                int total = 0;
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        size_t i = at + (dz * height + dy) * width;
                        total += raw[stride * i] + raw[stride * (i + 1)];
                    }
                }
                int whole = total / 8, rest = total % 8;
                filled += total != 0;
                crop[(z * ny + y) * REALISM_SIDE + x] =
                    whole + (rest > 4 || (rest == 4 && (whole & 1)));
            }
        }
    }
    return filled;
}

static size_t realism_real_crops(Arena *a, int profile, int count, int level, int across, u8 *crops,
    const char **names, int progress) {
    const char *chunks = env("HERCULEX_CHUNKS", fmt(a, "%s/cache/chunks", OUTPUTS));
    const char *labels = fmt(a, "%s/unwrapping/network-training/labels", OUTPUTS);
    int first = profile == 0 ? 0 : 1, end = profile == 0 ? 1 : 4;
    Pcg64 rng = pcg64(0);
    size_t kept = 0;
    for (int j = first; j < end; j++) {
        const RealismTraining *r = realism_training + j;
        HxCachedCt base, selected;
        need(!hx_cached_ct_open(&base, chunks, r->sample, r->scan, 0),
            "no level-0 CT cache for %s %s", r->sample, r->scan);
        int have_level = level && !hx_cached_ct_open(&selected, chunks, r->sample, r->scan, level);
        size_t mark = a->used;
        Array vertices[3] = {0};
        size_t candidates = 0;
        for (int i = 0; i < 3 && r->windows[i]; i++) {
            vertices[i] = npz_get(a, fmt(a, "%s/%s.npz", labels, r->windows[i]), "V");
            need(vertices[i].ndim == 2 && vertices[i].shape[1] == 3 &&
                    (!strcmp(vertices[i].descr, "<f4") || !strcmp(vertices[i].descr, "<f8")),
                "invalid training vertices in %s", r->windows[i]);
            candidates += vertices[i].shape[0];
        }
        need(candidates && candidates <= UINT32_MAX, "no training vertices for %s", r->sample);
        u8 *raw = push(a, 8 * REALISM_N);
        int tries = 0, found = 0, scale = 1 << level;
        for (; found < count && tries < 30 * count; tries++) {
            size_t pick = draw(&rng, candidates);
            int i = 0;
            while (i < 2 && (i64)pick >= vertices[i].shape[0]) {
                pick -= vertices[i++].shape[0];
            }
            const Array *v = vertices + i;
            i64 centre[3], box[6], full[6];
            for (int k = 0; k < 3; k++) {
                double x = v->descr[2] == '4' ? ((const float *)v->data)[3 * pick + k]
                                              : ((const double *)v->data)[3 * pick + k];
                centre[k] = floor(x / scale);
            }
            realism_box(centre, across, box);
            int available = 1;
            for (int k = 0; k < 3; k++) {
                full[2 * k] = scale * box[2 * k];
                full[2 * k + 1] = scale * box[2 * k + 1];
                available &= full[2 * k] >= 0 && full[2 * k + 1] <= base.shape[k];
            }
            if (!available) {
                continue;
            }
            u8 *crop = crops + kept * REALISM_N;
            i64 missing = 0;
            int ready =
                have_level && !hx_cached_ct_read(&selected, box, crop, &missing) && !missing;
            if (!ready) {
                u8 *pixels = level ? raw : crop;
                ready = !hx_cached_ct_read(&base, full, pixels, &missing) && !missing;
                if (ready && level) {
                    realism_average(raw, crop, across, 1);
                }
            }
            if (ready) {
                int filled = 0;
                for (int k = 0; k < REALISM_N; k++) {
                    filled += crop[k] != 0;
                }
                if ((double)filled / REALISM_N >= 0.85) {
                    names[kept++] = r->sample;
                    found++;
                }
            }
        }
        if (progress) {
            printf("%s: %d crops in %d tries\n", r->sample, found, tries);
        }
        if (have_level) {
            hx_cached_ct_close(&selected);
        }
        hx_cached_ct_close(&base);
        a->used = mark;
    }
    return kept;
}

#include <glob.h>

typedef struct {
    HxCachedCt cache;
    const char *url, *chunks;
    i64 shape[3];
    int level, cached;
} RealismVolume;

static int realism_volume(Arena *a, RealismVolume *v, const char *source, int level) {
    const char *chunks = env("HERCULEX_CHUNKS", fmt(a, "%s/cache/chunks", OUTPUTS));
    *v = (RealismVolume){.url = source, .chunks = chunks, .level = level, .cache = {.fd = -1}};
    int status = hx_cached_ct_open(&v->cache, chunks, NULL, source, level);
    if (status == 0) {
        v->cached = 1;
        memcpy(v->shape, v->cache.shape, sizeof(v->shape));
        return 1;
    }
    return !hx_chunk_url_shape(source, level, v->shape);
}

static int realism_target_read(RealismVolume *v, const i64 box[6], u8 *pixels) {
    i64 unknown;
    if (v->cached && !hx_cached_ct_read(&v->cache, box, pixels, &unknown) && !unknown) {
        return 1;
    }
    hx_cached_ct_close(&v->cache);
    v->cached = 0;
    if (hx_chunk_fetch_url_box(v->url, v->level, box, 0) ||
        hx_cached_ct_open(&v->cache, v->chunks, NULL, v->url, v->level)) {
        return 0;
    }
    v->cached = 1;
    return !hx_cached_ct_read(&v->cache, box, pixels, &unknown) && !unknown;
}

static size_t realism_target_crops(
    Arena *a, const char *sample, int count, int level, int across, u8 *crops, const char **names) {
    const char *targets[][2] = {{"PHerc0211", "20250821151803"}, {"PHerc0125", "20250821151825"},
        {"PHerc0826", "20250821151701"}, {"PHerc1545", "20250821151648"},
        {"PHerc0846A", "20250728152254"}, {"PHerc0175A", "20250521115057"},
        {"PHerc1218", "20250521120456"}, {"PHerc0268", "20251110183117"}};
    const char *scan = NULL;
    for (int k = 0; k < 8; k++) {
        if (!strcmp(targets[k][0], sample)) {
            scan = targets[k][1];
        }
    }
    need(scan != NULL, "unknown target scroll %s", sample);
    json_object *manifest = json_read(fmt(a, "%s/calibrate/bench/controls.json", HOME));
    json_object *scans = jget(manifest, "scans");
    json_object_object_foreach(scans, k, row) {
        if (k && !strcmp(jstr(row, "sample"), sample)) {
            json_object *zones = jget(row, "zones");
            for (size_t i = 0; i < json_object_array_length(zones); i++) {
                json_object *zone = json_object_array_get_idx(zones, i);
                const char *kind = json_object_get_string(json_object_array_get_idx(zone, 2));
                need(kind && strncmp(kind, "holdout", 7) != 0, "%s is a holdout scroll", sample);
            }
        }
    }
    json_object_put(manifest);
    const char *url = NULL;
    FILE *file = fopen(fmt(a, "%s/catalog/products.jsonl", LIBRARY), "r");
    char *line = NULL;
    size_t cap = 0;
    while (file && !url && getline(&line, &cap, file) >= 0) {
        json_object *row = json_tokener_parse(line);
        if (row && !strcmp(jstr(row, "sample"), sample) && !strcmp(jstr(row, "owner"), scan) &&
            !strcmp(jstr(row, "kind"), "ome-zarr")) {
            const char *path = jstr(row, "path"), *root = jstr(row, "root");
            if (!strstr(path, "vcd3d")) {
                int n = strlen(root), len = strlen(path);
                while (n && root[n - 1] == '/') {
                    n--;
                }
                while (*path == '/') {
                    path++;
                    len--;
                }
                while (len && path[len - 1] == '/') {
                    len--;
                }
                url = fmt(a, "%.*s/%.*s", n, root, len, path);
            }
        }
        json_object_put(row);
    }
    free(line);
    if (file) {
        fclose(file);
    }
    if (!url) {
        char *source = NULL;
        int default_level;
        need(!hx_chunk_resolve(sample, scan, 0, &source, &default_level),
            "cannot resolve CT for %s %s", sample, scan);
        url = fmt(a, "%s", source);
        free(source);
    }
    RealismVolume base, picked;
    need(realism_volume(a, &base, url, 0), "%s has no level-0 CT", sample);
    int own = level && realism_volume(a, &picked, url, level);
    RealismVolume *v = own ? &picked : &base;
    glob_t files = {0};
    glob(fmt(a, "%s/unwrapping/nodes/%s/%s/nodes/rows-*", OUTPUTS, sample, scan), 0, NULL, &files);
    need(files.gl_pathc && files.gl_pathc <= UINT32_MAX, "%s has no node units", sample);
    typedef struct {
        float *xyz;
        size_t count;
        double scale;
    } Unit;
    Unit *units = pushz(a, files.gl_pathc * sizeof(*units));
    int factor = level && !own ? 2 : 1, tries = 0;
    size_t found = 0;
    u8 *raw = push(a, 8 * REALISM_N);
    Pcg64 rng = pcg64(0);
    for (; found < (size_t)count && tries < 20 * count; tries++) {
        size_t slot = draw(&rng, files.gl_pathc);
        Unit *u = units + slot;
        if (!u->xyz) {
            void *data;
            i32 *cells;
            size_t n = 0;
            const char *folder = files.gl_pathv[slot];
            need(!hx_synthetic_raw_nodes(folder, &data, &cells, &n), "cannot read %s", folder);
            const RealismNode *node = data;
            for (size_t i = 0; i < n; i++) {
                u->count += node[i].prominence >= 20;
            }
            need(u->count <= UINT32_MAX, "too many nodes in %s", folder);
            u->xyz = push(a, u->count * 3 * sizeof(*u->xyz));
            size_t at = 0;
            for (size_t i = 0; i < n; i++) {
                if (node[i].prominence >= 20) {
                    u->xyz[at++] = node[i].z;
                    u->xyz[at++] = node[i].y;
                    u->xyz[at++] = node[i].x;
                }
            }
            free(cells);
            free(data);
            size_t mark = a->used;
            json_object *rec = json_read(fmt(a, "%s/record.json", folder));
            u->scale = ldexp(1, ji64(rec, "level") - level);
            json_object_put(rec);
            a->used = mark;
        }
        if (!u->count) {
            continue;
        }
        const float *point = u->xyz + 3 * draw(&rng, u->count);
        i64 centre[] = {
            (i64)(point[0] * u->scale), (i64)(point[1] * u->scale), (i64)(point[2] * u->scale)};
        i64 box[6], full[6];
        realism_box(centre, across, box);
        int valid = 1;
        for (int i = 0; i < 3; i++) {
            full[2 * i] = box[2 * i] * factor;
            full[2 * i + 1] = box[2 * i + 1] * factor;
            valid &= full[2 * i] >= 0 && full[2 * i + 1] <= v->shape[i];
        }
        if (!valid) {
            continue;
        }
        u8 *crop = crops + found * REALISM_N;
        need(realism_target_read(v, full, factor == 2 ? raw : crop), "cannot fetch CT for %s",
            sample);
        if (factor == 2) {
            realism_average(raw, crop, across, 1);
        }
        int filled = 0;
        for (int i = 0; i < REALISM_N; i++) {
            filled += crop[i] != 0;
        }
        if ((double)filled / REALISM_N >= 0.85) {
            names[found++] = sample;
        }
    }
    printf("%s: %zu crops in %d tries (%s)\n", sample, found, tries,
        factor == 2 ? "level 0 averaged" : "level 1");
    globfree(&files);
    hx_cached_ct_close(&base.cache);
    if (own) {
        hx_cached_ct_close(&picked.cache);
    }
    return found;
}

static void realism_save(Arena *a, const char *path, const u8 *crops, const char **names,
    size_t count, int sample_size) {
    size_t share = count / sample_size;
    size_t stride = share > 1 ? share : 1, kept = 1 + (count - 1) / stride, width = 0;
    for (size_t j = 0; j < count; j++) {
        size_t n = strlen(names[j]);
        width = n > width ? n : width;
    }
    u32 *scans = pushz(a, count * width * sizeof(*scans));
    u32 *retained = push(a, kept * width * sizeof(*retained));
    u8 *tiles = push(a, kept * REALISM_N);
    double *stats = push(a, count * REALISM_FEATURES * sizeof(*stats));
    size_t at = 0;
    for (size_t j = 0; j < count; j++) {
        const u8 *crop = crops + j * REALISM_N;
        realism_features(a, crop, stats + j * REALISM_FEATURES);
        const char *name = names[j];
        for (size_t k = 0; name[k]; k++) {
            scans[j * width + k] = (u8)name[k];
        }
        if (j % stride == 0) {
            memcpy(tiles + at * REALISM_N, crop, REALISM_N);
            memcpy(retained + at++ * width, scans + j * width, width * sizeof(*retained));
        }
    }
    const char *members[] = {"features", "scans", "crops", "crop_scans"};
    Array arrays[] = {{.descr = "<f8",
                          .ndim = 2,
                          .shape = {count, REALISM_FEATURES},
                          .size = count * REALISM_FEATURES * sizeof(*stats),
                          .data = stats},
        {.ndim = 1, .shape = {count}, .size = count * width * sizeof(*scans), .data = scans},
        {.descr = "|u1",
            .ndim = 3,
            .shape = {kept, REALISM_SIDE, REALISM_SIDE},
            .size = kept * REALISM_N,
            .data = tiles},
        {.ndim = 1, .shape = {kept}, .size = kept * width * sizeof(*retained), .data = retained}};
    snprintf(arrays[1].descr, sizeof(arrays[1].descr), "<U%zu", width);
    memcpy(arrays[3].descr, arrays[1].descr, sizeof(arrays[3].descr));
    npz_write(path, 4, members, arrays);
}

static int realism_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.realism {real PROFILE|target SCROLL} "
                        "[--crops N] [--plane xy|zx] [--level 0|1]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    int target = argc >= 2 && !strcmp(argv[0], "target");
    need(target || (argc >= 2 && !strcmp(argv[0], "real")), "%s", usage);
    int profile = !strcmp(argv[1], "9.362") ? 0 : !strcmp(argv[1], "7.91") ? 1 : -1;
    need(target || profile >= 0, "%s", usage);
    int count = target ? 60 : 150, across = 0, level = 1;
    for (int k = 2; k < argc; k++) {
        const char *option = argv[k];
        need(k + 1 < argc, "%s", usage);
        const char *text = argv[++k];
        if (!strcmp(option, "--crops")) {
            char *end;
            long n = strtol(text, &end, 10);
            need(!*end && n > 0 && n < INT_MAX / 90, "%s", usage);
            count = n;
        } else if (!strcmp(option, "--plane")) {
            need(!strcmp(text, "xy") || !strcmp(text, "zx"), "%s", usage);
            across = !strcmp(text, "zx");
        } else if (!strcmp(option, "--level")) {
            need(!strcmp(text, "0") || !strcmp(text, "1"), "%s", usage);
            level = !strcmp(text, "1");
        } else {
            die("%s", usage);
        }
    }
    Arena a = arena(64 * GB);
    size_t maximum = (size_t)count * (!target && profile ? 3 : 1);
    u8 *crops = push(&a, maximum * REALISM_N);
    const char **names = push(&a, maximum * sizeof(*names));
    size_t n = target ? realism_target_crops(&a, argv[1], count, level, across, crops, names)
                      : realism_real_crops(&a, profile, count, level, across, crops, names, 1);
    need(n != 0, "no cached crops for scanner profile %s", argv[1]);
    const char *path = fmt(&a, "%s/synthetic/realism/%s-%s%s%s.npz", OUTPUTS,
        target ? "target" : "real", argv[1], across ? "-zx" : "", level ? "" : "-L0");
    realism_save(&a, path, crops, names, n, target ? 16 : 48);
    if (target) {
        printf("{\"target\":\"%s\", \"plane\":\"%s\", \"level\":%d, \"crops\":%zu}\n", argv[1],
            across ? "zx" : "xy", level, n);
        return 0;
    }
    const char **sorted = push(&a, n * sizeof(*sorted));
    memcpy(sorted, names, n * sizeof(*sorted));
    qsort(sorted, n, sizeof(*sorted), name_order);
    json_object *scans = json_object_new_array(), *out = json_object_new_object();
    for (size_t k = 0; k < n; k++) {
        if (!k || strcmp(sorted[k], sorted[k - 1]) != 0) {
            json_object_array_add(scans, json_object_new_string(sorted[k]));
        }
    }
    json_object_object_add(out, "profile", json_object_new_string(argv[1]));
    json_object_object_add(out, "crops", json_object_new_int64(n));
    json_object_object_add(out, "scans", scans);
    puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_SPACED));
    json_object_put(out);
    return 0;
}
