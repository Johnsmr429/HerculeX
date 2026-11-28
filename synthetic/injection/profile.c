typedef struct {
    double re, im;
} InjectionComplex;

typedef struct {
    int dim[3];
    size_t n;
} InjectionShape;

static InjectionShape injection_shape(const i64 *dim) {
    InjectionShape shape = {.n = 1};
    for (int k = 0; k < 3; k++) {
        need(dim[k] > 0 && dim[k] <= INT_MAX && !(dim[k] & (dim[k] - 1)) &&
                (u64)dim[k] <= SIZE_MAX / shape.n,
            "injection spectra require a power-of-two grid");
        shape.dim[k] = (int)dim[k];
        shape.n *= dim[k];
    }
    return shape;
}

static void injection_fft_line(
    InjectionComplex *line, const InjectionComplex *phase, int len, int inverse) {
    for (unsigned i = 1, j = 0; i < (unsigned)len; i++) {
        unsigned bit = (unsigned)len >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            InjectionComplex swap = line[i];
            line[i] = line[j];
            line[j] = swap;
        }
    }
    for (int step = 1; step < len; step *= 2) {
        int stride = len / (2 * step);
        for (int start = 0; start < len; start += 2 * step) {
            for (int i = 0; i < step; i++) {
                InjectionComplex p = phase[i * stride], r = line[start + i + step];
                double x = p.re * r.re - p.im * r.im;
                double y = p.re * r.im + p.im * r.re;
                InjectionComplex v = line[start + i];
                line[start + i] = (InjectionComplex){v.re + x, v.im + y};
                line[start + i + step] = (InjectionComplex){v.re - x, v.im - y};
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < len; i++) {
            line[i].re /= len;
            line[i].im /= len;
        }
    }
}

static void injection_fft(Arena *a, InjectionComplex *data, InjectionShape shape, int inverse) {
    size_t mark = a->used;
    int longest = fmax(shape.dim[0], fmax(shape.dim[1], shape.dim[2]));
    InjectionComplex *line = push(a, (size_t)longest * sizeof(*line));
    InjectionComplex *phase = push(a, (size_t)longest * sizeof(*phase));
    size_t stride = 1;
    for (int axis = 2; axis >= 0; axis--) {
        int length = shape.dim[axis];
        if (length == 1) {
            continue;
        }
        double t = (inverse ? 2 : -2) * M_PI / length;
        for (int j = 0; j < length / 2; j++) {
            phase[j] = (InjectionComplex){cos(t * j), sin(t * j)};
        }
        for (size_t block = 0; block < shape.n; block += length * stride) {
            for (size_t i = 0; i < stride; i++) {
                for (int j = 0; j < length; j++) {
                    line[j] = data[block + i + j * stride];
                }
                injection_fft_line(line, phase, length, inverse);
                for (int j = 0; j < length; j++) {
                    data[block + i + j * stride] = line[j];
                }
            }
        }
        stride *= length;
    }
    a->used = mark;
}

typedef struct {
    double mean, variance, skew, kurtosis;
} InjectionMoments;

static InjectionMoments injection_moments(const double *values, size_t n) {
    InjectionMoments s = {0};
    s.mean = pairwise64(values, n) / n;
    double v = 0, skew = 0, kurtosis = 0;
    for (size_t i = 0; i < n; i++) {
        double delta = values[i] - s.mean, square = delta * delta;
        v += square;
        skew += square * delta;
        kurtosis += square * square;
    }
    s.variance = v / n;
    s.skew = (skew / n) / pow(s.variance, 1.5);
    s.kurtosis = (kurtosis / n) / (s.variance * s.variance) - 3;
    return s;
}

static double injection_round(double value, double scale) {
    return nearbyint(value * scale) / scale;
}

static json_object *injection_number(double value) {
    return isfinite(value) ? jnum(value) : json_object_new_double(value);
}

static double injection_center(double *values, size_t n) {
    double mean = pairwise64(values, n) / n;
    for (size_t i = 0; i < n; i++) {
        values[i] -= mean;
    }
    return mean;
}

static void injection_power(
    Arena *a, const double *cubes, size_t count, InjectionShape shape, double *power) {
    size_t mark = a->used;
    InjectionComplex *f = push(a, shape.n * sizeof(*f));
    memset(power, 0, shape.n * sizeof(*power));
    for (size_t cube = 0; cube < count; cube++) {
        for (size_t i = 0; i < shape.n; i++) {
            f[i] = (InjectionComplex){cubes[cube * shape.n + i], 0};
        }
        injection_fft(a, f, shape, 0);
        for (size_t i = 0; i < shape.n; i++) {
            power[i] += f[i].re * f[i].re + f[i].im * f[i].im;
        }
    }
    for (size_t i = 0; i < shape.n; i++) {
        power[i] = power[i] / count / shape.n;
    }
    a->used = mark;
}

static void injection_lags(Arena *a, const double *power, InjectionShape shape, double lags[3][5]) {
    size_t mark = a->used;
    InjectionComplex *f = push(a, shape.n * sizeof(*f));
    for (size_t i = 0; i < shape.n; i++) {
        f[i] = (InjectionComplex){power[i], 0};
    }
    injection_fft(a, f, shape, 1);
    size_t stride = 1;
    for (int axis = 2; axis >= 0; axis--) {
        for (int lag = 0; lag <= 4; lag++) {
            lags[axis][lag] = f[(lag % shape.dim[axis]) * stride].re;
        }
        stride *= shape.dim[axis];
    }
    a->used = mark;
}

static double injection_ks(double *a, size_t na, double *b, size_t nb) {
    qsort(a, na, sizeof(*a), slab_order);
    qsort(b, nb, sizeof(*b), slab_order);
    size_t i = 0, j = 0;
    double max = 0;
    while (i < na || j < nb) {
        double next = i < na && j < nb ? fmin(a[i], b[j]) : i < na ? a[i] : b[j];
        while (i < na && a[i] <= next) {
            i++;
        }
        while (j < nb && b[j] <= next) {
            j++;
        }
        max = fmax(max, fabs((double)i / na - (double)j / nb));
    }
    return max;
}

static Array injection_air(Arena *a, const char *folder, InjectionShape *shape) {
    const char *path = fmt(a, "%s/air.npz", folder);
    Array blocks = npz_get(a, path, "blocks");
    need(blocks.ndim == 4 && !strcmp(blocks.descr, "|u1") && blocks.shape[0] > 0,
        "%s: expected uint8 air cubes", path);
    *shape = injection_shape(blocks.shape + 1);
    need(blocks.size / shape->n == (size_t)blocks.shape[0] && blocks.size % shape->n == 0,
        "%s: invalid air cube dimensions", path);
    return blocks;
}

static const char *injection_profile_folder(Arena *a, const char *scroll) {
    need(*scroll && !strchr(scroll, '/') && strcmp(scroll, ".") != 0 && strcmp(scroll, "..") != 0,
        "invalid injection profile scroll: %s", scroll);
    return fmt(a, "%s/synthetic/library/%s", OUTPUTS, scroll);
}

typedef struct {
    InjectionShape shape;
    const float *spectrum, *quantiles;
    size_t quantile_count;
} InjectionNoise;

static InjectionNoise injection_noise_open(Arena *a, const char *path) {
    Array spectrum = npz_get(a, path, "spectrum");
    Array quantiles = npz_get(a, path, "quantiles");
    need(spectrum.ndim == 3 && !strcmp(spectrum.descr, "<f4") && quantiles.ndim == 1 &&
            !strcmp(quantiles.descr, "<f4") && quantiles.shape[0] >= 2 &&
            quantiles.size / sizeof(float) == (size_t)quantiles.shape[0] &&
            quantiles.size % sizeof(float) == 0,
        "%s: unsupported noise spectrum or quantiles", path);
    InjectionShape shape = injection_shape(spectrum.shape);
    need(spectrum.size / sizeof(float) == shape.n && spectrum.size % sizeof(float) == 0,
        "%s: noise spectrum dimensions differ", path);
    const float *power = spectrum.data, *q = quantiles.data;
    size_t nq = (size_t)quantiles.shape[0];
    for (size_t i = 0; i < shape.n; i++) {
        need(isfinite(power[i]) && power[i] >= 0, "%s: invalid noise spectrum", path);
    }
    for (size_t i = 0; i < nq; i++) {
        need(isfinite(q[i]) && (!i || q[i] >= q[i - 1]), "%s: invalid noise quantiles", path);
    }
    return (InjectionNoise){shape, power, q, nq};
}

static void injection_draw_noise(
    Arena *a, InjectionNoise profile, double level, size_t count, Pcg64 *rng, double *values) {
    size_t mark = a->used;
    InjectionShape shape = profile.shape;
    InjectionComplex *f = push(a, shape.n * sizeof(*f));
    const float *s = profile.spectrum, *q = profile.quantiles;
    size_t nq = profile.quantile_count, total = shape.n * count;
    double *scale = push(a, shape.n * sizeof(*scale));
    for (size_t i = 0; i < shape.n; i++) {
        scale[i] = sqrt(s[i]);
    }
    for (size_t c = 0; c < count; c++) {
        for (size_t i = 0; i < shape.n; i++) {
            f[i] = (InjectionComplex){syn_normal(rng), 0};
        }
        injection_fft(a, f, shape, 0);
        for (size_t i = 0; i < shape.n; i++) {
            f[i].re *= scale[i];
            f[i].im *= scale[i];
        }
        injection_fft(a, f, shape, 1);
        for (size_t i = 0; i < shape.n; i++) {
            values[c * shape.n + i] = f[i].re;
        }
    }
    double std = sqrt(injection_moments(values, total).variance);
    need(std > 0 && isfinite(std), "synthetic noise has zero or invalid spread");
    for (size_t i = 0; i < total; i++) {
        double index = .5 * erfc(-values[i] / (std * M_SQRT2)) * (nq - 1);
        size_t lo = (size_t)floor(index);
        size_t hi = lo < nq - 1 ? lo + 1 : lo;
        values[i] = level + (q[lo] + (index - lo) * ((double)q[hi] - q[lo]));
    }
    a->used = mark;
}

static int injection_profile_check(Arena *a, const char *scroll) {
    enum { CUBES = 64 };
    const char *folder = injection_profile_folder(a, scroll);
    json_object *summary = json_read(fmt(a, "%s/profile.json", folder)),
                *air = jget(summary, "air");
    const char *path = fmt(a, "%s/profile.npz", folder);
    InjectionNoise profile = injection_noise_open(a, path);
    InjectionShape shape;
    Array blocks = injection_air(a, folder, &shape);
    need(!memcmp(profile.shape.dim, shape.dim, sizeof(shape.dim)), "%s: air and spectrum differ",
        path);
    size_t total = CUBES * shape.n;
    double *values = push(a, total * sizeof(*values));
    double *rounded = push(a, total * sizeof(*rounded));
    double *real = push(a, blocks.size * sizeof(*real));
    Pcg64 rng = pcg64(0);
    injection_draw_noise(a, profile, jf64(air, "level"), CUBES, &rng, values);
    for (size_t i = 0; i < total; i++) {
        rounded[i] = fmax(0, fmin(255, nearbyint(values[i])));
    }
    for (size_t i = 0; i < blocks.size; i++) {
        real[i] = ((const u8 *)blocks.data)[i];
    }
    for (size_t c = 0; c < (size_t)blocks.shape[0]; c++) {
        injection_center(real + c * shape.n, shape.n);
    }
    for (int c = 0; c < CUBES; c++) {
        injection_center(values + (size_t)c * shape.n, shape.n);
        injection_center(rounded + (size_t)c * shape.n, shape.n);
    }
    double *power = push(a, shape.n * sizeof(*power)), lags[3][5];
    injection_power(a, rounded, CUBES, shape, power);
    injection_lags(a, power, shape, lags);
    json_object *out = json_object_new_object();
    json_object *expected = jget(air, "autocovariance");
    int passed = 1;
    for (int ax = 0; ax < 3; ax++) {
        char axis[] = {"zyx"[ax], 0};
        json_object *base = jget(expected, axis), *record = json_object_new_array();
        need(json_object_array_length(base) >= 5, "%s: invalid %s autocovariance", path, axis);
        for (int lag = 0; lag <= 4; lag++) {
            double want = json_object_get_double(json_object_array_get_idx(base, lag));
            double ratio = lags[ax][lag] / want;
            if (lag < 3 && !(fabs(ratio - 1) <= .1)) {
                passed = 0;
            }
            json_object_array_add(record, injection_number(injection_round(ratio, 1000)));
        }
        json_object_object_add(out, fmt(a, "autocovariance_%s_drawn_over_real", axis), record);
    }
    InjectionMoments drawn = injection_moments(values, total),
                     measured = injection_moments(real, blocks.size);
    double ks = injection_ks(values, total, real, blocks.size);
    json_object_object_add(out, "ks_distance", injection_number(injection_round(ks, 10000)));
    json_object_object_add(
        out, "kurtosis_drawn", injection_number(injection_round(drawn.kurtosis, 1000)));
    json_object_object_add(
        out, "kurtosis_real", injection_number(injection_round(measured.kurtosis, 1000)));
    json_object_object_add(out, "skew_drawn", injection_number(injection_round(drawn.skew, 1000)));
    json_object_object_add(
        out, "skew_real", injection_number(injection_round(measured.skew, 1000)));
    json_object_object_add(out, "passed", json_object_new_boolean(passed && ks <= .05));
    puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_PRETTY));
    json_object_put(out);
    json_object_put(summary);
    return 0;
}

static json_object *injection_quantiles(
    double *values, size_t n, const double *qs, int count, double scale) {
    qsort(values, n, sizeof(*values), slab_order);
    json_object *out = json_object_new_array();
    for (int i = 0; i < count; i++) {
        double value = slab_quantile(values, n, qs[i]);
        json_object_array_add(out, injection_number(scale ? injection_round(value, scale) : value));
    }
    return out;
}

static json_object *injection_air_measure(
    Arena *a, Array air, InjectionShape shape, float *power, float *quantiles, int nq) {
    size_t mark = a->used, count = (size_t)air.shape[0], total = air.size;
    double *raw = push(a, total * sizeof(*raw)), *centred = push(a, total * sizeof(*centred));
    double *levels = push(a, count * sizeof(*levels));
    for (size_t i = 0; i < total; i++) {
        raw[i] = centred[i] = ((const u8 *)air.data)[i];
    }
    for (size_t i = 0; i < count; i++) {
        levels[i] = injection_center(centred + i * shape.n, shape.n);
    }
    InjectionMoments all = injection_moments(raw, total);
    double spread = sqrt(injection_moments(centred, total).variance);
    double *spectrum = push(a, shape.n * sizeof(*spectrum));
    injection_power(a, centred, count, shape, spectrum);
    double lags[3][5];
    injection_lags(a, spectrum, shape, lags);
    for (size_t i = 0; i < shape.n; i++) {
        power[i] = (float)spectrum[i];
    }
    qsort(centred, total, sizeof(*centred), slab_order);
    for (int i = 0; i < nq; i++) {
        quantiles[i] = (float)slab_quantile(centred, total, i * (1.0 / (nq - 1)));
    }
    json_object *cov = json_object_new_object(), *json = json_object_new_object();
    for (int k = 0; k < 3; k++) {
        char axis[] = {"zyx"[k], 0};
        json_object *values = json_object_new_array();
        for (int j = 0; j <= 4; j++) {
            json_object_array_add(values, jnum(lags[k][j]));
        }
        json_object_object_add(cov, axis, values);
    }
    double white = sqrt(fmax(0, lags[0][0] - (lags[0][1] + lags[1][1] + lags[2][1]) / 3));
    json_object_object_add(json, "level", injection_number(all.mean));
    json_object_object_add(json, "spread", injection_number(spread));
    json_object_object_add(json, "white", injection_number(white));
    json_object_object_add(json, "autocovariance", cov);
    json_object_object_add(json, "skew", injection_number(all.skew));
    json_object_object_add(json, "kurtosis", injection_number(all.kurtosis));
    json_object_object_add(json, "percentiles",
        injection_quantiles(raw, total, (double[]){.01, .05, .25, .5, .75, .95, .99}, 7, 0));
    json_object_object_add(
        json, "level_spread", injection_number(sqrt(injection_moments(levels, count).variance)));
    a->used = mark;
    return json;
}

enum { INJECTION_GRAIN_HALF = 7, INJECTION_GRAIN_SIZE = 15 };

static int injection_peak(const u8 *box, const i64 *dims, int z, int y, int x, u8 peak) {
    for (int k = -2; k <= 2; k++) {
        for (int j = -2; j <= 2; j++) {
            const u8 *p = box + ((z + k) * dims[1] + y + j) * dims[2] + x;
            for (int i = -2; i <= 2; i++) {
                if (p[i] > peak) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int injection_grain(
    const u8 *box, const i64 *dims, int z, int y, int x, float sum[3][INJECTION_GRAIN_SIZE]) {
    enum { H = INJECTION_GRAIN_HALF, SIDE = INJECTION_GRAIN_SIZE, FACE = SIDE * SIDE };
    double shell[6 * FACE];
    i64 row = dims[2], plane = dims[1] * row;
    const u8 *centre = box + ((size_t)z * dims[1] + y) * dims[2] + x;
    int n = 0;
    for (int j = -H; j <= H; j++) {
        for (int i = -H; i <= H; i++) {
            for (int side = -H; side <= H; side += 2 * H) {
                shell[n++] = centre[side * plane + j * row + i];
                shell[n++] = centre[j * plane + side * row + i];
                shell[n++] = centre[j * plane + i * row + side];
            }
        }
    }
    qsort(shell, n, sizeof(*shell), slab_order);
    if (slab_quantile(shell, n, .9) > 75) {
        return 0;
    }
    double background = slab_quantile(shell, n, .5), height = *centre - background;
    double halfway = background + .5 * height;
    int bright = 0;
    for (int k = -H; k <= H; k++) {
        for (int j = -H; j <= H; j++) {
            for (int i = -H; i <= H; i++) {
                if (centre[k * plane + j * row + i] > halfway && ++bright > 40) {
                    return 0;
                }
            }
        }
    }
    for (int i = -H; i <= H; i++) {
        sum[0][i + H] += (float)(centre[i * plane] - background) / (float)height;
        sum[1][i + H] += (float)(centre[i * row] - background) / (float)height;
        sum[2][i + H] += (float)(centre[i] - background) / (float)height;
    }
    return 1;
}

static size_t *injection_truth_keep(Arena *a, const SlabLibrary *library, size_t *kept) {
    const char *truth = fmt(a, "%s/truth.npz", library->folder);
    Array winding = npz_get(a, truth, "winding");
    Array frame = npz_get(a, truth, "frame");
    Array flags = npz_get(a, truth, "flags");
    size_t points = (size_t)library->points.shape[0], count = 0;
    int wide_w = !strcmp(winding.descr, "<f8"), wide_f = !strcmp(frame.descr, "<i4");
    need((wide_w || !strcmp(winding.descr, "<f4")) && winding.size == points * (wide_w ? 8 : 4) &&
            (wide_f || !strcmp(frame.descr, "<i2")) && frame.size == points * (wide_f ? 4 : 2) &&
            !strcmp(flags.descr, "|u1") && flags.size == points,
        "%s: invalid grain truth labels", truth);
    size_t *keep = push(a, points * sizeof(*keep));
    for (size_t i = 0; i < points; i++) {
        double w = wide_w ? ((double *)winding.data)[i] : ((f32 *)winding.data)[i];
        int f = wide_f ? ((i32 *)frame.data)[i] : ((i16 *)frame.data)[i];
        if (isfinite(w) && f >= 0 && !(((u8 *)flags.data)[i] & (1 | 32 | 64))) {
            keep[count++] = i;
        }
    }
    need(count && count <= UINT32_MAX, "%s: no usable truth points", truth);
    *kept = count;
    return keep;
}

static json_object *injection_grains(Arena *a, SlabLibrary *library) {
    enum { H = INJECTION_GRAIN_HALF, S = INJECTION_GRAIN_SIZE, BOXES = 120 };
    size_t mark = a->used, count;
    size_t *keep = injection_truth_keep(a, library, &count);
    size_t box_mark = a->used;
    Pcg64 rng = pcg64(0);
    float sums[3][S] = {{0}};
    i64 grains = 0;
    for (int k = 0; k < BOXES; k++) {
        const f32 *point = (const f32 *)library->points.data + 3 * keep[draw(&rng, count)];
        i64 x = (i64)nearbyint(point[0]), y = (i64)nearbyint(point[1]),
            z = (i64)nearbyint(point[2]);
        i64 box[] = {z - 48, z + 48, y - 128, y + 128, x - 128, x + 128}, dims[3];
        const u8 *ct = slab_read(a, library, box, dims);
        size_t total = (size_t)dims[0] * dims[1] * dims[2], nonzero = 0;
        for (size_t i = 0; i < total; i++) {
            nonzero += ct[i] != 0;
        }
        if ((double)(total - nonzero) / total <= .001) {
            for (int iz = H + 1; iz < dims[0] - H - 1; iz++) {
                for (int iy = H + 1; iy < dims[1] - H - 1; iy++) {
                    for (int ix = H + 1; ix < dims[2] - H - 1; ix++) {
                        u8 value = ct[((size_t)iz * dims[1] + iy) * dims[2] + ix];
                        if (value >= 120 && value <= 240 &&
                            injection_peak(ct, dims, iz, iy, ix, value)) {
                            grains += injection_grain(ct, dims, iz, iy, ix, sums);
                        }
                    }
                }
            }
        }
        a->used = box_mark;
    }
    json_object *out = json_object_new_object();
    json_object_object_add(out, "grains", json_object_new_int64(grains));
    if (grains) {
        for (int k = 0; k < 3; k++) {
            int n = 0;
            double sigma = 0;
            json_object *axis = json_object_new_array();
            for (int i = -H; i <= H; i++) {
                f32 value = sums[k][i + H] / (f32)grains;
                json_object_array_add(axis, injection_number(injection_round(value, 1000)));
                if (i && abs(i) <= 3 && value > .05f) {
                    sigma += -(i * i) / (2.0 * logf(value));
                    n++;
                }
            }
            char profile[] = "profile_z", width[] = "sigma_z";
            profile[8] = width[6] = "zyx"[k];
            json_object_object_add(out, profile, axis);
            json_object_object_add(
                out, width, n ? injection_number(injection_round(sqrt(sigma / n), 1000)) : NULL);
        }
    }
    a->used = mark;
    return out;
}

static double injection_hist_rank(const size_t *hist, size_t rank) {
    for (int i = 0; i < 256; i++) {
        if (rank < hist[i]) {
            return i;
        }
        rank -= hist[i];
    }
    return 255;
}

static json_object *injection_papyrus(Arena *a, const glob_t *paths) {
    size_t mark = a->used, n = 0, hist[256] = {0};
    float *core = NULL;
    double profile[2 * SLAB_DEPTH + 1] = {0}, saturation = 0;
    int depth = 0;
    for (size_t s = 0; s < paths->gl_pathc; s++) {
        Array ct = npz_get(a, paths->gl_pathv[s], "ct");
        need(ct.ndim == 3 && !strcmp(ct.descr, "|u1") && ct.shape[0] > 0 && ct.shape[1] > 0 &&
                ct.shape[2] >= 3 && ct.shape[2] <= 2 * SLAB_DEPTH + 1 &&
                (!depth || ct.shape[2] == depth) && ct.size == (size_t)elements(ct),
            "%s: invalid papyrus slab", paths->gl_pathv[s]);
        depth = (int)ct.shape[2];
        size_t pixels = ct.size / depth, saturated = 0;
        need(pixels <= (SIZE_MAX / sizeof(*core) - n) / 3, "papyrus slabs are too large");
        float *grow = realloc(core, (n + pixels * 3) * sizeof(*core));
        need(grow != NULL, "cannot allocate papyrus core");
        core = grow;
        double levels[2 * SLAB_DEPTH + 1] = {0};
        const u8 *in = ct.data;
        for (size_t i = 0; i < pixels; i++) {
            for (int d = 0; d < depth; d++) {
                u8 value = in[i * depth + d];
                levels[d] += value;
                saturated += value == 255;
            }
            for (int d = depth / 2 - 1; d <= depth / 2 + 1; d++) {
                u8 value = in[i * depth + d];
                core[n++] = value;
                hist[value]++;
            }
        }
        for (int d = 0; d < depth; d++) {
            profile[d] += levels[d] / pixels;
        }
        saturation += (double)saturated / ct.size;
        a->used = mark;
    }
    float level = mean_of(core, n);
    for (size_t i = 0; i < n; i++) {
        float d = core[i] - level;
        core[i] = d * d;
    }
    float spread = sqrtf(mean_of(core, n));
    free(core);
    json_object *record = json_object_new_object(), *curve = json_object_new_array();
    json_object *percentiles = json_object_new_array();
    const double p[] = {.05, .25, .5, .75, .95};
    for (int i = 0; i < 5; i++) {
        double index = p[i] * (n - 1);
        double low = injection_hist_rank(hist, (size_t)floor(index));
        double high = injection_hist_rank(hist, (size_t)ceil(index));
        json_object_array_add(percentiles, jnum(low + (index - floor(index)) * (high - low)));
    }
    for (int i = 0; i < depth; i++) {
        json_object_array_add(curve, jnum(injection_round(profile[i] / paths->gl_pathc, 100)));
    }
    json_object_object_add(record, "slabs", json_object_new_int64(paths->gl_pathc));
    json_object_object_add(record, "core_level", jnum(level));
    json_object_object_add(record, "core_spread", jnum(spread));
    json_object_object_add(record, "core_percentiles", percentiles);
    json_object_object_add(record, "profile", curve);
    json_object_object_add(record, "saturated_share", jnum(saturation / paths->gl_pathc));
    return record;
}

static int injection_slab_has(const char *path, const char *name) {
    int error;
    zip_t *archive = zip_open(path, ZIP_RDONLY, &error);
    need(archive != NULL, "%s: cannot read slab archive", path);
    int exists = zip_name_locate(archive, name, 0) >= 0;
    zip_close(archive);
    return exists;
}

static void injection_ridge(Arena *a, const char *path, SlabPoint *position, SlabPoint *normal) {
    enum { SMALL = SLAB_SIDE / SLAB_GAP_STEP, PAD = 12, PADDED = SMALL + 2 * PAD };
    size_t mark = a->used, pixels = (size_t)SLAB_SIDE * SLAB_SIDE;
    Array p = npz_get(a, path, "points"), n = npz_get(a, path, "normals");
    Array r = npz_get(a, path, "ridge");
    need(p.ndim == 3 && p.shape[0] == SMALL && p.shape[1] == SMALL && p.shape[2] == 3 &&
            n.ndim == 3 && !memcmp(p.shape, n.shape, 3 * sizeof(i64)) && !strcmp(p.descr, "<f4") &&
            !strcmp(n.descr, "<f4") && !strcmp(r.descr, "<f2") && r.ndim == 2 &&
            r.shape[0] == SLAB_SIDE && r.shape[1] == SLAB_SIDE &&
            p.size == SMALL * SMALL * 3 * sizeof(float) && n.size == p.size &&
            r.size == pixels * sizeof(_Float16),
        "%s: unsupported ridge, points or normal grids", path);
    SlabSpline spline = slab_spline(a, 1, PADDED, PADDED);
    const float *src = p.data, *norm = n.data;
    for (int k = 0; k < 3; k++) {
        for (int y = 0; y < PADDED; y++) {
            int row = (int)slab_nearest(y - PAD, SMALL);
            for (int x = 0; x < PADDED; x++) {
                int col = (int)slab_nearest(x - PAD, SMALL);
                spline.data[y * PADDED + x] = src[(row * SMALL + col) * 3 + k];
            }
        }
        slab_prefilter(spline);
        for (int y = 0; y < SLAB_SIDE; y++) {
            double gy = y / (double)SLAB_GAP_STEP;
            SlabTaps ty = slab_taps(gy + PAD, PADDED, PADDED);
            int lo = (int)gy, hi = (int)slab_nearest(lo + 1, SMALL);
            double wy = gy - lo;
            for (int x = 0; x < SLAB_SIDE; x++) {
                double gx = x / (double)SLAB_GAP_STEP;
                SlabTaps tx = slab_taps(gx + PAD, PADDED, 1);
                double value = 0;
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < 4; i++) {
                        value +=
                            spline.data[ty.index[j] + tx.index[i]] * ty.weight[j] * tx.weight[i];
                    }
                }
                size_t at = (size_t)y * SLAB_SIDE + x;
                position[at].v[k] = value;
                int left = (int)gx, right = (int)slab_nearest(left + 1, SMALL);
                double wx = gx - left;
                double top = norm[(lo * SMALL + left) * 3 + k] * (1 - wx) +
                    norm[(lo * SMALL + right) * 3 + k] * wx;
                double bottom = norm[(hi * SMALL + left) * 3 + k] * (1 - wx) +
                    norm[(hi * SMALL + right) * 3 + k] * wx;
                normal[at].v[k] = top * (1 - wy) + bottom * wy;
            }
        }
    }
    const _Float16 *height = r.data;
    for (size_t i = 0; i < pixels; i++) {
        SlabPoint *normal_at = normal + i;
        double norm2 = 0;
        for (int k = 0; k < 3; k++) {
            norm2 += normal_at->v[k] * normal_at->v[k];
        }
        need(norm2 > 0, "%s: zero relief normal", path);
        double norm_len = sqrt(norm2);
        for (int k = 0; k < 3; k++) {
            normal_at->v[k] /= norm_len;
            position[i].v[k] += (double)height[i] * normal_at->v[k];
        }
    }
    a->used = mark;
}

enum { INJECTION_RELIEF_TILE = 128, INJECTION_RELIEF_PIXELS = 128 * 128 };

static void injection_basis(double row, double col, double mean_square, double *q) {
    q[0] = 1;
    q[1] = row;
    q[2] = col;
    q[3] = row * row * .5 - mean_square;
    q[4] = row * col;
    q[5] = col * col * .5 - mean_square;
}

static double injection_relief_tile(Arena *a, const SlabPoint *positions, const SlabPoint *normals,
    int y, int x, double *power, double *bend) {
    enum { T = INJECTION_RELIEF_TILE, N = INJECTION_RELIEF_PIXELS };
    size_t mark = a->used;
    SlabPoint mean = {0}, axis = {0};
    for (int j = 0; j < T; j++) {
        for (int i = 0; i < T; i++) {
            size_t at = (size_t)(y + j) * SLAB_SIDE + x + i;
            for (int k = 0; k < 3; k++) {
                mean.v[k] += positions[at].v[k];
                axis.v[k] += normals[at].v[k];
            }
        }
    }
    double axis_len = sqrt(axis.v[0] * axis.v[0] + axis.v[1] * axis.v[1] + axis.v[2] * axis.v[2]);
    for (int k = 0; k < 3; k++) {
        mean.v[k] /= N;
        axis.v[k] /= axis_len;
    }
    double mean_square = (T * T - 1) / 24.0, mid = (T - 1) / 2.0;
    double *height = push(a, N * sizeof(*height));
    double fit[6] = {0}, scale[6] = {0}, weights[T];
    for (int j = 0; j < T; j++) {
        weights[j] = .5 - .5 * cos(2 * M_PI * j / (T - 1));
        for (int i = 0; i < T; i++) {
            const SlabPoint *p = positions + (size_t)(y + j) * SLAB_SIDE + x + i;
            double h = 0, q[6];
            for (int k = 0; k < 3; k++) {
                h += (p->v[k] - mean.v[k]) * axis.v[k];
            }
            height[j * T + i] = h;
            injection_basis(j - mid, i - mid, mean_square, q);
            for (int k = 0; k < 6; k++) {
                fit[k] += q[k] * h;
                scale[k] += q[k] * q[k];
            }
        }
    }
    for (int k = 0; k < 6; k++) {
        fit[k] /= scale[k];
    }
    memcpy(bend, fit + 3, 3 * sizeof(*bend));
    double norm = 0;
    InjectionComplex *f = push(a, N * sizeof(*f));
    for (int j = 0; j < T; j++) {
        for (int i = 0; i < T; i++) {
            double q[6], model = 0;
            injection_basis(j - mid, i - mid, mean_square, q);
            for (int k = 0; k < 6; k++) {
                model += fit[k] * q[k];
            }
            int at = j * T + i;
            double window = weights[j] * weights[i];
            height[at] -= model;
            f[at] = (InjectionComplex){height[at] * window, 0};
            norm += window * window;
        }
    }
    injection_fft(a, f, (InjectionShape){.dim = {1, T, T}, .n = N}, 0);
    for (int i = 0; i < N; i++) {
        power[i] += (f[i].re * f[i].re + f[i].im * f[i].im) / norm;
    }
    double spread = sqrt(injection_moments(height, N).variance);
    a->used = mark;
    return spread;
}

static json_object *injection_relief(
    Arena *a, const glob_t *paths, float *spectrum, float *bends, int *tiles) {
    enum { T = INJECTION_RELIEF_TILE, N = INJECTION_RELIEF_PIXELS };
    size_t mark = a->used, pixels = (size_t)SLAB_SIDE * SLAB_SIDE;
    size_t limit = paths->gl_pathc * (SLAB_SIDE / T) * (SLAB_SIDE / T);
    double *power = pushz(a, N * sizeof(*power)), *spreads = push(a, limit * sizeof(*spreads));
    double *rows = push(a, limit * sizeof(*rows)), *cols = push(a, limit * sizeof(*cols));
    SlabPoint *points = push(a, pixels * sizeof(*points)),
              *normals = push(a, pixels * sizeof(*normals));
    double total = 0;
    size_t count = 0;
    for (size_t i = 0; i < paths->gl_pathc; i++) {
        const char *path = paths->gl_pathv[i];
        if (!injection_slab_has(path, "ridge.npy")) {
            continue;
        }
        injection_ridge(a, path, points, normals);
        for (int y = 0; y + T <= SLAB_SIDE; y += T) {
            for (int x = 0; x + T <= SLAB_SIDE; x += T) {
                double bend[3];
                double spread = injection_relief_tile(a, points, normals, y, x, power, bend);
                for (int k = 0; k < 3; k++) {
                    bends[count * 3 + k] = (float)bend[k];
                }
                rows[count] = bend[0];
                cols[count] = bend[2];
                spreads[count++] = spread;
                total += spread * spread;
            }
        }
    }
    need(count > 0 && count <= INT_MAX, "no slab relief tiles");
    double mean_power = pairwise64(power, N) / N;
    double scale = total / count / mean_power;
    for (int i = 0; i < N; i++) {
        spectrum[i] = (float)(power[i] * scale);
    }
    *tiles = (int)count;
    const double q[] = {.1, .5, .9};
    json_object *out = json_object_new_object();
    json_object_object_add(out, "tiles", json_object_new_int(*tiles));
    json_object_object_add(out, "tile_voxels", json_object_new_int(T));
    json_object_object_add(out, "spread_voxels", injection_quantiles(spreads, count, q, 3, 1000));
    json_object_object_add(out, "bend_rows_per_voxel", injection_quantiles(rows, count, q, 3, 0));
    json_object_object_add(out, "bend_cols_per_voxel", injection_quantiles(cols, count, q, 3, 0));
    a->used = mark;
    return out;
}

static int injection_profile_measure(Arena *a, const char *scroll, const char *package) {
    enum { NQ = 100001, T = INJECTION_RELIEF_TILE, PIXELS = INJECTION_RELIEF_PIXELS };
    const char *folder = injection_profile_folder(a, scroll);
    SlabLibrary library = slab_open(a, package, scroll);
    json_object *response = injection_grains(a, &library);
    hx_cached_ct_close(&library.ct);
    glob_t paths = {0}, meta = {0};
    need(!glob(fmt(a, "%s/library-*.json", folder), GLOB_NOSORT, NULL, &meta) && meta.gl_pathc,
        "%s: no slab library metadata", folder);
    json_object *library_json = json_read(meta.gl_pathv[0]);
    double voxel = jf64(library_json, "voxel_um");
    json_object_put(library_json);
    globfree(&meta);
    need(!glob(fmt(a, "%s/slab-*.npz", folder), 0, NULL, &paths) && paths.gl_pathc &&
            paths.gl_pathc < INT_MAX / ((SLAB_SIDE / T) * (SLAB_SIDE / T)),
        "%s: no usable papyrus slabs", folder);
    qsort(paths.gl_pathv, paths.gl_pathc, sizeof(*paths.gl_pathv), slab_mesh_order);
    InjectionShape shape;
    Array air = injection_air(a, folder, &shape);
    float *power = push(a, shape.n * sizeof(*power)), *quantiles = push(a, NQ * sizeof(*quantiles));
    float *relief = push(a, PIXELS * sizeof(*relief));
    size_t tile_limit = paths.gl_pathc * (SLAB_SIDE / T) * (SLAB_SIDE / T);
    float *bends = push(a, tile_limit * 3 * sizeof(*bends));
    int tiles;
    json_object *summary = json_object_new_object(), *range = json_object_new_array();
    json_object_array_add(range, json_object_new_int(0));
    json_object_array_add(range, json_object_new_int(255));
    json_object_object_add(summary, "scroll", json_object_new_string(scroll));
    json_object_object_add(summary, "package", json_object_new_string(package));
    json_object_object_add(summary, "voxel_um", jnum(voxel));
    json_object_object_add(
        summary, "air", injection_air_measure(a, air, shape, power, quantiles, NQ));
    json_object_object_add(summary, "response", response);
    json_object_object_add(summary, "papyrus", injection_papyrus(a, &paths));
    json_object_object_add(summary, "range", range);
    json_object_object_add(summary, "relief", injection_relief(a, &paths, relief, bends, &tiles));
    globfree(&paths);
    const char *names[] = {"spectrum", "quantiles", "relief_spectrum", "bends"};
    Array arrays[] = {
        {.descr = "<f4",
            .ndim = 3,
            .shape = {shape.dim[0], shape.dim[1], shape.dim[2]},
            .data = power,
            .size = shape.n * sizeof(*power)},
        {.descr = "<f4",
            .ndim = 1,
            .shape = {NQ},
            .data = quantiles,
            .size = NQ * sizeof(*quantiles)},
        {.descr = "<f4",
            .ndim = 2,
            .shape = {T, T},
            .data = relief,
            .size = PIXELS * sizeof(*relief)},
        {.descr = "<f4",
            .ndim = 2,
            .shape = {tiles, 3},
            .data = bends,
            .size = (size_t)tiles * 3 * sizeof(*bends)},
    };
    npz_write_codec(fmt(a, "%s/profile.npz", folder), 4, names, arrays, ZIP_CM_DEFLATE);
    json_write(fmt(a, "%s/profile.json", folder), summary);
    puts(json_object_to_json_string_ext(summary, JSON_C_TO_STRING_PRETTY));
    json_object_put(summary);
    return 0;
}

static int injection_profile_command(int argc, char **argv) {
    const char *usage = "synthetic.injection.profile measure SCROLL [--package PACKAGE]\n"
                        "synthetic.injection.profile check SCROLL";
    if (!argc || !strcmp(argv[0], "-h") || !strcmp(argv[0], "--help")) {
        puts(usage);
        return 0;
    }
    need(argc >= 2, "%s", usage);
    Arena a = arena(4 * GB);
    int result;
    if (!strcmp(argv[0], "check")) {
        need(argc == 2, "%s", usage);
        result = injection_profile_check(&a, argv[1]);
    } else {
        need(!strcmp(argv[0], "measure"), "%s", usage);
        const char *package = NULL;
        for (int i = 2; i < argc; i++) {
            char split[32];
            const char *value;
            const char *key = slices_cli_option(argv[i], split, &value);
            need(!strcmp(key, "--package"), "unknown injection-profile option: %s", key);
            if (!value) {
                need(i + 1 < argc, "%s needs a value", key);
                value = argv[++i];
            }
            package = value;
        }
        result = injection_profile_measure(
            &a, argv[1], package ? package : fmt(&a, "%s-train", argv[1]));
    }
    munmap(a.base, a.size);
    return result;
}
