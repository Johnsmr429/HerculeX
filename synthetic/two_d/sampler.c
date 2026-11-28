enum { SLICES_WINDOW = 256, SLICES_STRIDE = 128, SLICES_PIXELS = SLICES_WINDOW * SLICES_WINDOW };

typedef struct {
    u32 state[624];
    unsigned cursor;
    void (*accelerated_fill)(void *, double *, size_t);
    void *accelerator;
} SliceNoise;

typedef void (*SliceNetwork)(void *context, int n, const float *input, const float *condition,
    const i64 *style, float log_sigma, float *output);

typedef struct {
    SliceNetwork network;
    void *network_context;
    int w, h, n, windows, low_precision;
    double *hann, *weights;
    float *input, *condition, *output;
    i64 *window_styles;
} SliceSample;

static SliceNoise slices_noise_seed(u64 seed) {
    SliceNoise n = {.cursor = 624};
    n.state[0] = (u32)seed;
    for (int i = 1; i < 624; i++) {
        u32 prev = n.state[i - 1];
        n.state[i] = 1812433253u * (prev ^ (prev >> 30)) + i;
    }
    return n;
}

static u32 slices_noise_next(SliceNoise *n) {
    if (n->cursor == 624) {
        for (int i = 0; i < 624; i++) {
            u32 next = n->state[(i + 1) % 624];
            u32 mix = (n->state[i] & 0x80000000u) | (next & 0x7fffffffu);
            n->state[i] = n->state[(i + 397) % 624] ^ (mix >> 1) ^ ((next & 1) ? 0x9908b0dfu : 0);
        }
        n->cursor = 0;
    }
    u32 y = n->state[n->cursor++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    return y ^ (y >> 18);
}

static double slices_noise_uniform(SliceNoise *n, int low_precision) {
    u64 bits = slices_noise_next(n);
    if (low_precision) {
        return (bits & 0xffffffu) * 0x1p-24;
    }
    bits = (bits << 32) | slices_noise_next(n);
    return (bits & ((1ull << 53) - 1)) * 0x1p-53;
}

static void slices_sampler_noise(SliceNoise *n, double *out, size_t count, int low_precision) {
    if (n->accelerated_fill) {
        n->accelerated_fill(n->accelerator, out, count);
        return;
    }
    need(count % 16 == 0, "sampler noise requires batches of 16 pixels");
    for (size_t start = 0; start < count; start += 16) {
        double u[16];
        for (int j = 0; j < 16; j++) {
            u[j] = slices_noise_uniform(n, low_precision);
        }
        for (int j = 0; j < 8; j++) {
            double a = (2.0 * M_PI) * u[8 + j];
            if (low_precision) {
                float radius = sqrtf(-2.0f * logf(1.0f - (float)u[j]));
                out[start + j] = radius * cosf((float)a);
                out[start + j + 8] = radius * sinf((float)a);
            } else {
                double radius = sqrt(-2.0 * log(1.0 - u[j]));
                out[start + j] = radius * cos(a);
                out[start + j + 8] = radius * sin(a);
            }
        }
    }
}

static double slices_precision(double x, int low_precision) {
    return low_precision ? (double)(float)x : x;
}

static void slices_window_origin(const SliceSample *s, int window, int *tile, int *y, int *x) {
    int cols = (s->w - SLICES_WINDOW) / SLICES_STRIDE + 1;
    int rows = (s->h - SLICES_WINDOW) / SLICES_STRIDE + 1;
    *tile = window / (cols * rows);
    *y = (window / cols % rows) * SLICES_STRIDE;
    *x = (window % cols) * SLICES_STRIDE;
}

static void slices_denoise(SliceSample *s, const double *canvas, const u8 *line, const u8 *dist,
    const i64 *styles, double sigma, double *out) {
    int ncols = (s->w - SLICES_WINDOW) / SLICES_STRIDE + 1;
    int nrows = (s->h - SLICES_WINDOW) / SLICES_STRIDE + 1;
    int n = ncols * nrows * s->n;
    size_t tile = (size_t)s->w * s->h, pixels = tile * s->n;
    memset(out, 0, pixels * sizeof(*out));
    memset(s->weights, 0, pixels * sizeof(*s->weights));
    float sig = (float)sigma;
    float variance = sig * sig + .25f, cskip = .25f / variance;
    float root = sqrtf(variance), cin = 1.0f / root, cout = .5f * sig / root;
    for (int offset = 0; offset < n; offset += s->windows) {
        int count = slices_min(s->windows, n - offset);
        for (int i = 0; i < count; i++) {
            int b, y, x;
            slices_window_origin(s, offset + i, &b, &y, &x);
            s->window_styles[i] = styles[b];
            float *input = s->input + (size_t)i * SLICES_PIXELS;
            float *condition = s->condition + (size_t)i * SLICES_PIXELS * 3;
            for (int r = 0; r < SLICES_WINDOW; r++) {
                size_t at = tile * b + (size_t)(y + r) * s->w + x;
                for (int c = 0; c < SLICES_WINDOW; c++) {
                    int p = r * SLICES_WINDOW + c;
                    input[p] = (float)canvas[at + c] * cin;
                    condition[p] = 1;
                    condition[p + SLICES_PIXELS] = line[at + c] ? 1 : -1;
                    condition[p + SLICES_PIXELS * 2] = slices_min(dist[at + c], 32) / 16.0f - 1;
                }
            }
        }
        s->network(s->network_context, count, s->input, s->condition, s->window_styles,
            logf(sig) / 4, s->output);
        for (int i = 0; i < count; i++) {
            int b, y, x;
            slices_window_origin(s, offset + i, &b, &y, &x);
            const float *value = s->output + (size_t)i * SLICES_PIXELS;
            for (int r = 0; r < SLICES_WINDOW; r++) {
                size_t at = tile * b + (size_t)(y + r) * s->w + x;
                for (int c = 0; c < SLICES_WINDOW; c++) {
                    int p = r * SLICES_WINDOW + c;
                    float prediction = cskip * (float)canvas[at + c] + cout * value[p];
                    double weight = s->hann[p];
                    double weighted = slices_precision(prediction * weight, s->low_precision);
                    out[at + c] = slices_precision(out[at + c] + weighted, s->low_precision);
                    s->weights[at + c] =
                        slices_precision(s->weights[at + c] + weight, s->low_precision);
                }
            }
        }
    }
    for (size_t i = 0; i < pixels; i++) {
        out[i] = slices_precision(out[i] / s->weights[i], s->low_precision);
    }
}

static void slices_sample(SliceNoise *rng, SliceNetwork network, void *context, int w, int h, int n,
    int windows, int steps, int low_precision, const u8 *lines, const u8 *dist, const i64 *style,
    u8 *out) {
    size_t count = (size_t)w * h * n;
    size_t nwin = (size_t)((w - SLICES_WINDOW) / SLICES_STRIDE + 1) *
        ((h - SLICES_WINDOW) / SLICES_STRIDE + 1) * n;
    need(nwin > 0 && nwin <= INT_MAX, "too many synthetic sampler windows");
    windows = slices_min(windows, (int)nwin);
    size_t win_pixels = (size_t)windows * SLICES_PIXELS;
    need(win_pixels < (SIZE_MAX - (8 << 20)) / 64 &&
            count < (SIZE_MAX - (8 << 20)) / 64 - win_pixels,
        "synthetic sampler arrays are too large");
    Arena mem = arena(64 * (count + win_pixels) + (8 << 20));
    SliceSample sample = {
        .network = network,
        .network_context = context,
        .w = w,
        .h = h,
        .n = n,
        .windows = windows,
        .low_precision = low_precision,
        .hann = push(&mem, SLICES_PIXELS * sizeof(double)),
        .weights = push(&mem, count * sizeof(double)),
        .input = push(&mem, win_pixels * sizeof(float)),
        .condition = push(&mem, win_pixels * 3 * sizeof(float)),
        .output = push(&mem, win_pixels * sizeof(float)),
        .window_styles = push(&mem, (size_t)windows * sizeof(i64)),
    };
    double axis[SLICES_WINDOW];
    for (int i = 0; i < SLICES_WINDOW; i++) {
        if (low_precision) {
            axis[i] = .5f * (1.0f - cosf((i + 1) * (2.0f * (float)M_PI / 257.0f)));
        } else {
            axis[i] = .5 * (1 - cos((i + 1) * (2 * M_PI / 257)));
        }
    }
    for (int y = 0; y < SLICES_WINDOW; y++) {
        for (int x = 0; x < SLICES_WINDOW; x++) {
            double weight = slices_precision(axis[y] * axis[x], low_precision);
            sample.hann[y * SLICES_WINDOW + x] = slices_precision(weight + .001, low_precision);
        }
    }
    double *sigma = push(&mem, ((size_t)steps + 1) * sizeof(*sigma));
    double max = pow(80, 1.0 / 7), min = pow(.002, 1.0 / 7);
    for (int i = 0; i < steps; i++) {
        if (low_precision) {
            sigma[i] = powf((float)max + (float)i / (steps - 1) * (float)(min - max), 7);
        } else {
            sigma[i] = pow(max + (double)i / (steps - 1) * (min - max), 7);
        }
    }
    sigma[steps] = 0;
    double *state = push(&mem, count * sizeof(double));
    double *next = push(&mem, count * sizeof(double));
    double *first = push(&mem, count * sizeof(double));
    double *second = push(&mem, count * sizeof(double));
    slices_sampler_noise(rng, state, count, low_precision);
    for (size_t p = 0; p < count; p++) {
        state[p] = slices_precision(state[p] * sigma[0], low_precision);
    }
    for (int i = 0; i < steps; i++) {
        double old = sigma[i], target = sigma[i + 1], at = old;
        if (old >= .05 && old <= 50) {
            double churn = fmin(30.0 / steps, sqrt(2) - 1);
            at = slices_precision(old * slices_precision(1 + churn, low_precision), low_precision);
            double at2 = slices_precision(at * at, low_precision);
            double old2 = slices_precision(old * old, low_precision);
            double radius = sqrt(at2 - old2) * 1.003;
            if (low_precision) {
                radius = sqrtf((float)(at2 - old2)) * 1.003f;
            }
            slices_sampler_noise(rng, next, count, low_precision);
            for (size_t p = 0; p < count; p++) {
                double noise = slices_precision(radius * next[p], low_precision);
                state[p] = slices_precision(state[p] + noise, low_precision);
            }
        }
        double delta = slices_precision(target - at, low_precision);
        slices_denoise(&sample, state, lines, dist, style, at, first);
        for (size_t p = 0; p < count; p++) {
            double difference = slices_precision(state[p] - first[p], low_precision);
            first[p] = slices_precision(difference / at, low_precision);
            double step = slices_precision(delta * first[p], low_precision);
            next[p] = slices_precision(state[p] + step, low_precision);
        }
        if (target > 0) {
            slices_denoise(&sample, next, lines, dist, style, target, second);
            for (size_t p = 0; p < count; p++) {
                double difference = slices_precision(next[p] - second[p], low_precision);
                double deriv = slices_precision(difference / target, low_precision);
                double slope = slices_precision(.5 * first[p] + .5 * deriv, low_precision);
                double step = slices_precision(delta * slope, low_precision);
                next[p] = slices_precision(state[p] + step, low_precision);
            }
        }
        double *swap = state;
        state = next;
        next = swap;
    }
    for (size_t p = 0; p < count; p++) {
        float value = fminf(fmaxf((float)state[p], -1), 1);
        out[p] = (u8)nearbyintf((value + 1) * 127.5f);
    }
    munmap(mem.base, mem.size);
}
