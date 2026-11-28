static int spot_reflect(int i, int n) {
    i %= 2 * n;
    if (i < 0) {
        i += 2 * n;
    }
    return i < n ? i : 2 * n - i - 1;
}

static void spot_gaussian(
    Arena *a, double *plane, double *work, int h, int w, double sigma, int single) {
    if (sigma <= 1e-15) {
        return;
    }
    size_t mark = a->used;
    int radius = (int)(sigma * 4 + 0.5);
    double *kernel = push(a, (2 * (size_t)radius + 1) * sizeof(*kernel));
    double total = 0;
    for (int i = -radius; i <= radius; i++) {
        double ratio = i / sigma;
        total += kernel[i + radius] = exp(-0.5 * ratio * ratio);
    }
    for (int i = 0; i <= 2 * radius; i++) {
        kernel[i] /= total;
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double value = 0;
            for (int i = -radius; i <= radius; i++) {
                value += kernel[i + radius] * plane[spot_reflect(y + i, h) * w + x];
            }
            work[y * w + x] = single ? (f32)value : value;
        }
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double value = 0;
            for (int i = -radius; i <= radius; i++) {
                value += kernel[i + radius] * work[y * w + spot_reflect(x + i, w)];
            }
            plane[y * w + x] = single ? (f32)value : value;
        }
    }
    a->used = mark;
}

static f32 spot_input(Array x, int kind, int width, i64 i) {
    const u8 *p = x.data;
    if (kind == 'f') {
        if (width == 2) {
            return ((const _Float16 *)p)[i];
        }
        return width == 4 ? ((const f32 *)p)[i] : (f32)((const f64 *)p)[i];
    }
    if (kind == 'i') {
        if (width == 1) {
            return ((const int8_t *)p)[i];
        }
        if (width == 2) {
            return ((const int16_t *)p)[i];
        }
        return width == 4 ? ((const int32_t *)p)[i] : (f32)((const int64_t *)p)[i];
    }
    if (width == 1) {
        return p[i];
    }
    if (width == 2) {
        return ((const uint16_t *)p)[i];
    }
    return width == 4 ? ((const uint32_t *)p)[i] : (f32)((const uint64_t *)p)[i];
}

static void spot_output(void *out, int kind, int width, i64 i, double value) {
    if (kind == 'f') {
        if (width == 2) {
            ((_Float16 *)out)[i] = value;
        } else if (width == 4) {
            ((f32 *)out)[i] = value;
        } else {
            ((f64 *)out)[i] = value;
        }
        return;
    }
    if (kind == 'b') {
        ((u8 *)out)[i] = value != 0;
        return;
    }
    double rounded = nearbyint(value);
    int bits = 8 * width - (kind == 'i');
    double limit = ldexp(1.0, bits);
    u64 raw;
    if (kind == 'i') {
        raw = rounded <= -limit ? (u64)(-(UINT64_C(1) << bits))
            : rounded >= limit  ? (UINT64_C(1) << bits) - 1
                                : (u64)(i64)rounded;
    } else {
        u64 max = width == 8 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
        raw = rounded <= 0 ? 0 : rounded >= limit ? max : (u64)rounded;
    }
    memcpy((u8 *)out + i * width, &raw, width);
}

static double spot_shift(const double *stack, i64 count, int layers, i64 i, f32 depth) {
    f32 clipped = fminf(layers - 1, fmaxf(0, depth));
    int low = (int)clipped, high = low + (low + 1 < layers);
    double t = (double)clipped - low;
    return stack[(i64)low * count + i] * (1 - t) + stack[(i64)high * count + i] * t;
}

int spot_flatten_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("Usage: ./run.sh spots.flatten STACK.npy OUT.npz [--mask VALID.npy]\n"
             "       [--rounds N] [--reach LAYERS] [--window RADIUS]\n"
             "       [--reference-px SIGMA] [--smooth-px SIGMA]");
        return 0;
    }
    need(argc >= 2 && argc % 2 == 0, "usage: spots.flatten STACK.npy OUT.npz [--mask VALID.npy]");
    int rounds = 5, window = 12;
    double reach = 3, reference_px = 24, smooth_px = 6;
    const char *mask_path = NULL;
    for (int i = 2; i < argc; i += 2) {
        const char *arg = argv[i], *value = argv[i + 1];
        if (!strcmp(arg, "--mask")) {
            mask_path = value;
            continue;
        }
        char *end;
        double number = strtod(value, &end);
        need(end != value && !*end && number >= 0 && number <= 4096, "invalid %s", arg);
        if (!strcmp(arg, "--rounds") || !strcmp(arg, "--window")) {
            need(number == (int)number, "%s needs an integer", arg);
            if (!strcmp(arg, "--rounds")) {
                rounds = (int)number;
            } else {
                window = (int)number;
            }
        } else if (!strcmp(arg, "--reach")) {
            reach = number;
        } else if (!strcmp(arg, "--reference-px")) {
            reference_px = number;
        } else if (!strcmp(arg, "--smooth-px")) {
            smooth_px = number;
        } else {
            die("unknown flattening option %s", arg);
        }
    }
    Arena a = arena(48 * GB);
    Array src = npy_read(&a, argv[0]);
    int kind = (unsigned char)src.descr[1], bytes = (unsigned char)src.descr[2] - '0';
    int supported = (src.descr[0] == '<' || src.descr[0] == '|') && strlen(src.descr) == 3 &&
        ((kind == 'f' && (bytes == 2 || bytes == 4 || bytes == 8)) ||
            ((kind == 'i' || kind == 'u') &&
                (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8)) ||
            (kind == 'b' && bytes == 1));
    need(src.ndim == 3 && src.shape[0] > 0 && src.shape[1] >= 2 && src.shape[2] >= 2 &&
            elements(src) <= INT_MAX && supported,
        "the stack must be a numeric (L,H,W) NPY array");
    int layers = src.shape[0], h = src.shape[1], w = src.shape[2], hh = h / 2, ww = w / 2;
    int n = h * w, m = hh * ww, band = 2 * window + 1;
    const u8 *valid = NULL;
    if (mask_path) {
        Array mask = npy_read(&a, mask_path);
        need(mask.ndim == 2 && mask.shape[0] == h && mask.shape[1] == w &&
                (!strcmp(mask.descr, "|b1") || !strcmp(mask.descr, "|u1")),
            "flattening mask must match the stack's height and width");
        valid = mask.data;
    }
    double *small = push(&a, (i64)layers * m * sizeof(*small));
    double *blur = push(&a, (i64)m * sizeof(*blur));
    double *weight = push(&a, (i64)m * sizeof(*weight));
    double *reference = push(&a, (i64)band * m * sizeof(*reference));
    double *update = push(&a, (i64)m * sizeof(*update));
    double *measures = push(&a, (i64)m * sizeof(*measures));
    double *column = push(&a, (i64)band * sizeof(*column));
    f32 *offset = pushz(&a, (i64)m * sizeof(*offset));
    f32 *best = push(&a, (i64)m * sizeof(*best));
    u8 *seen = push(&a, m);
    for (int i = 0; i < m; i++) {
        int at = 2 * (i / ww) * w + 2 * (i % ww);
        seen[i] = !valid || (valid[at] && valid[at + 1] && valid[at + w] && valid[at + w + 1]);
        weight[i] = seen[i];
        for (int z = 0; z < layers; z++) {
            i64 k = (i64)z * n + at;
            small[(i64)z * m + i] =
                (spot_input(src, kind, bytes, k) + spot_input(src, kind, bytes, k + 1) +
                    spot_input(src, kind, bytes, k + w) + spot_input(src, kind, bytes, k + w + 1)) *
                0.25f;
        }
    }
    for (int z = 0; z < layers; z++) {
        spot_gaussian(&a, small + (i64)z * m, blur, hh, ww, 1, 1);
    }
    spot_gaussian(&a, weight, blur, hh, ww, reference_px / 2, 1);
    double *history = push(&a, (i64)rounds * sizeof(*history));
    int first = layers / 2 - window, shifts = (int)ceil((2 * reach + 0.125) / 0.25);
    for (int pass = 0; pass < rounds; pass++) {
        for (int z = 0; z < band; z++) {
            double *r = reference + (i64)z * m;
            for (int i = 0; i < m; i++) {
                r[i] = seen[i] ? spot_shift(small, m, layers, i, (f32)(first + z) + offset[i]) : 0;
            }
            spot_gaussian(&a, r, blur, hh, ww, reference_px / 2, 0);
        }
        for (int i = 0; i < m; i++) {
            best[i] = 0;
            if (!seen[i]) {
                continue;
            }
            double average = 0, divisor = fmax(weight[i], (f32)1e-6);
            for (int z = 0; z < band; z++) {
                average += reference[(i64)z * m + i] /= divisor;
            }
            average /= band;
            double variance = 0;
            for (int z = 0; z < band; z++) {
                double d = reference[(i64)z * m + i] -= average;
                variance += d * d;
            }
            double denom = sqrt(variance / band) + 1e-6;
            for (int z = 0; z < band; z++) {
                reference[(i64)z * m + i] /= denom;
            }
            f32 best_score = -INFINITY;
            for (int s = 0; s < shifts; s++) {
                f32 start = (f32)-reach;
                f32 shift = start + s * ((f32)(-reach + 0.25) - start);
                f32 depth = offset[i] + shift;
                double mean = 0;
                for (int z = 0; z < band; z++) {
                    mean += column[z] = spot_shift(small, m, layers, i, (f32)(first + z) + depth);
                }
                mean /= band;
                double spread = 0;
                for (int z = 0; z < band; z++) {
                    column[z] -= mean;
                    spread += column[z] * column[z];
                }
                double scale = sqrt(spread / band) + 1e-6, score = 0;
                for (int z = 0; z < band; z++) {
                    score += column[z] / scale * reference[(i64)z * m + i];
                }
                score /= band;
                if (score > best_score) {
                    best_score = (f32)score;
                    best[i] = shift;
                }
            }
        }
        for (int y = 0; y < hh; y++) {
            for (int x = 0; x < ww; x++) {
                double nearby[25];
                int j = 0;
                for (int u = -2; u <= 2; u++) {
                    for (int v = -2; v <= 2; v++) {
                        nearby[j++] = best[spot_reflect(y + u, hh) * ww + spot_reflect(x + v, ww)];
                    }
                }
                qsort(nearby, 25, sizeof(*nearby), spot_length_order);
                update[y * ww + x] = nearby[12];
            }
        }
        spot_gaussian(&a, update, blur, hh, ww, smooth_px / 2, 1);
        int kept = 0;
        for (int i = 0; i < m; i++) {
            f32 next = fminf((f32)reach, fmaxf((f32)-reach, (f32)update[i]));
            offset[i] += next;
            if (seen[i]) {
                measures[kept++] = fabsf(next);
            }
        }
        qsort(measures, kept, sizeof(*measures), spot_length_order);
        f32 median = kept ? (f32)(measures[(kept - 1) / 2] + measures[kept / 2]) * 0.5f : NAN;
        history[pass] = nearbyint((double)median * 1000) / 1000;
    }
    f32 *whole = push(&a, (i64)n * sizeof(*whole));
    void *out = push(&a, src.size);
    double ratio_h = (hh - 1) / (double)(h - 1), ratio_w = (ww - 1) / (double)(w - 1);
    for (int i = 0; i < n; i++) {
        int row = i / w, col = i % w;
        double y = row * ratio_h, x = col * ratio_w;
        int low_y = (int)y, low_x = (int)x;
        int high_y = low_y + (low_y + 1 < hh), high_x = low_x + (low_x + 1 < ww);
        double u = y - low_y, v = x - low_x;
        whole[i] = valid && !valid[i]
            ? 0
            : (f32)((1 - u) *
                      ((1 - v) * offset[low_y * ww + low_x] + v * offset[low_y * ww + high_x]) +
                  u * ((1 - v) * offset[high_y * ww + low_x] + v * offset[high_y * ww + high_x]));
        for (int z = 0; z < layers; z++) {
            f32 at = fminf(layers - 1, fmaxf(0, (f32)z + whole[i]));
            int lo = (int)at, hi = lo + (lo + 1 < layers);
            double t = (double)at - lo;
            double a = spot_input(src, kind, bytes, (i64)lo * n + i);
            double b = spot_input(src, kind, bytes, (i64)hi * n + i);
            spot_output(out, kind, bytes, (i64)z * n + i, a * (1 - t) + b * t);
        }
    }
    src.data = out;
    Array fields[] = {src,
        {.descr = "<f4", .ndim = 2, .shape = {h, w}, .data = whole, .size = (size_t)n * 4},
        {.descr = "<f8",
            .ndim = 1,
            .shape = {rounds},
            .data = history,
            .size = (size_t)rounds * 8}};
    npz_write(argv[1], 3, (const char *[]){"stack", "offset", "history"}, fields);
    return 0;
}
