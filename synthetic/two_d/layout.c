typedef struct {
    double gap, gap_var, packet, inter, lens, lens_amp, lens_width, lens_inner;
    double pinch, shift, ripple, ripple_width, wave, wave_width, wave_rows, tear;
    double angle, bend, bend_width;
    int bends;
} SliceSettings;

typedef struct {
    double x, y;
} SlicePoint;

typedef struct {
    double c, s, start;
    int count;
    double *offset;
} SliceBend;

typedef struct {
    int count;
    double c, s;
    SliceBend bend[3];
} SliceFold;

typedef struct {
    int row;
    double x, left, right, height;
} SliceLens;

typedef struct {
    int row, gap;
    double left, right, side, curl, length;
} SliceTear;

static double slices_uniform(Pcg64 *rng, double low, double high) {
    return low + (high - low) * pcg64_double(rng);
}

static int slices_min(int a, int b) { return a < b ? a : b; }

static int slices_max(int a, int b) { return a > b ? a : b; }

static double slices_gamma2(Pcg64 *rng) {
    const double d = 2.0 - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x = syn_normal(rng), v = 1.0 + c * x;
        if (v > 0) {
            v = v * v * v;
            double u = pcg64_double(rng);
            if (u < 1.0 - 0.0331 * (x * x) * (x * x) ||
                log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
                return d * v;
            }
        }
    }
}

static int slices_poisson(Pcg64 *rng, double rate) {
    if (rate == 0) {
        return 0;
    }
    if (rate < 10) {
        double product = 1.0, limit = exp(-rate);
        int count = 0;
        while ((product *= pcg64_double(rng)) > limit) {
            count++;
        }
        return count;
    }
    double b = .931 + 2.53 * sqrt(rate), a = -.059 + .02483 * b;
    double inv_alpha = 1.1239 + 1.1328 / (b - 3.4), vr = .9277 - 3.6224 / (b - 2);
    for (;;) {
        double u = pcg64_double(rng) - .5, v = pcg64_double(rng), us = .5 - fabs(u);
        if (!us) {
            continue;
        }
        double k = floor((2 * a / us + b) * u + rate + .43);
        if (us >= .07 && v <= vr) {
            return (int)k;
        }
        if (k >= 0 && (us >= .013 || v <= us) &&
            log(v) + log(inv_alpha) - log(a / (us * us) + b) <=
                -rate + k * log(rate) - lgamma(k + 1)) {
            return (int)k;
        }
    }
}

static SliceSettings slices_settings(Pcg64 *rng) {
    SliceSettings s;
    s.gap = slices_uniform(rng, 14.5, 19.5);
    s.gap_var = slices_uniform(rng, .1, .25);
    s.packet = slices_uniform(rng, 5, 16);
    s.inter = slices_uniform(rng, 1.3, 2.4);
    s.lens = slices_uniform(rng, 1.2, 3.0);
    s.lens_amp = slices_uniform(rng, 32, 70);
    s.lens_width = slices_uniform(rng, 70, 200);
    s.lens_inner = slices_uniform(rng, .1, .4);
    s.pinch = slices_uniform(rng, .8, 2.0);
    s.shift = slices_uniform(rng, .2, 1.0);
    s.ripple = slices_uniform(rng, .5, 1.4);
    s.ripple_width = slices_uniform(rng, 10, 25);
    s.wave = slices_uniform(rng, 2, 9);
    s.wave_width = slices_uniform(rng, 60, 160);
    s.wave_rows = slices_uniform(rng, 4, 15);
    s.tear = slices_uniform(rng, .25, .7);
    s.angle = slices_uniform(rng, 0, M_PI);
    s.bend = slices_uniform(rng, .4, .85);
    s.bend_width = slices_uniform(rng, 70, 200);
    s.bends = (int)draw(rng, 2) + 1;
    pcg64_double(rng);
    pcg64_double(rng);
    pcg64_double(rng);
    pcg64_double(rng);
    return s;
}

static int slices_bound(int x, int n) { return slices_min(slices_max(x, 0), n - 1); }

static int slices_gaussian_weights(double sigma, double *weights, int capacity) {
    int radius = (int)(4.0 * sigma + .5);
    need(radius < capacity, "synthetic layout Gaussian is too wide");
    double sum = weights[0] = 1.0;
    for (int k = 1; k <= radius; k++) {
        weights[k] = exp(-.5 / (sigma * sigma) * k * k);
        sum += 2.0 * weights[k];
    }
    for (int k = 0; k <= radius; k++) {
        weights[k] /= sum;
    }
    return radius;
}

static void slices_filter(double *out, const double *in, int n, int stride, double sigma) {
    double weights[600];
    int radius = slices_gaussian_weights(sigma, weights, 600);
    for (int i = 0; i < n; i++) {
        double value = in[i * stride] * weights[0];
        for (int k = 1; k <= radius; k++) {
            value += (in[slices_bound(i - k, n) * stride] + in[slices_bound(i + k, n) * stride]) *
                weights[k];
        }
        out[i * stride] = value;
    }
}

static double slices_std(const double *data, size_t n) {
    double mean = 0.0, var = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean += data[i];
    }
    mean /= n;
    for (size_t i = 0; i < n; i++) {
        double d = data[i] - mean;
        var += d * d;
    }
    return sqrt(var / n);
}

static double slices_offset(const SliceBend *b, double x) {
    double index = (x - b->start) / 2;
    if (index <= 0.0) {
        return b->offset[0];
    }
    if (index >= b->count - 1) {
        return b->offset[b->count - 1];
    }
    int i = (int)index;
    return b->offset[i] + (index - i) * (b->offset[i + 1] - b->offset[i]);
}

static SliceFold slices_fold(Arena *mem, const SliceSettings *s, Pcg64 *rng, double reach) {
    SliceFold fold = {.count = s->bends + (reach > 2000), .c = cos(s->angle), .s = sin(s->angle)};
    int n = (int)ceil((2 * reach + 1) / 2);
    for (int i = 0; i < fold.count; i++) {
        double angle = s->angle + .2 * syn_normal(rng);
        double amp = s->bend * slices_uniform(rng, .6, 1) * (i ? .55 : 1);
        amp *= draw(rng, 2) ? 1 : -1;
        double width = s->bend_width * slices_uniform(rng, .7, 1.3);
        double center = slices_uniform(rng, -.25, .25) * reach / 2.5;
        syn_normal(rng);
        double kind = pcg64_double(rng);
        double *offset = push(mem, (size_t)n * sizeof(*offset));
        SliceBend *b = &fold.bend[i];
        *b = (SliceBend){cos(angle), sin(angle), -reach, n, offset};
        double std = 1;
        if (kind >= .75) {
            double *noise = malloc(n * sizeof(*noise));
            need(noise != NULL, "out of memory");
            for (int k = 0; k < n; k++) {
                noise[k] = syn_normal(rng);
            }
            slices_filter(offset, noise, n, 1, width / 2);
            std = slices_std(offset, n) + 1e-9;
            free(noise);
        }
        double sum = 0;
        for (int k = 0; k < n; k++) {
            double x = (-reach + 2.0 * k - center) / width, change;
            if (kind < .4) {
                change = amp * .5 * (1 + tanh(x));
            } else if (kind < .75) {
                change = amp * exp(-.5 * x * x);
            } else {
                change = amp * tanh(offset[k] / std * 1.1);
            }
            sum += change;
            offset[k] = 2 * sum;
        }
        double origin = slices_offset(b, 0);
        for (int k = 0; k < n; k++) {
            offset[k] -= origin;
        }
    }
    return fold;
}

static SlicePoint slices_transform(const SliceFold *f, double x, double y, int inverse) {
    if (!inverse) {
        double u = x * f->c - y * f->s;
        y = x * f->s + y * f->c;
        x = u;
    }
    for (int i = 0; i < f->count; i++) {
        const SliceBend *b = &f->bend[inverse ? f->count - 1 - i : i];
        double a = x * b->c + y * b->s, v = -x * b->s + y * b->c;
        v += (inverse ? -1 : 1) * slices_offset(b, a);
        x = a * b->c - v * b->s;
        y = a * b->s + v * b->c;
    }
    return inverse ? (SlicePoint){x * f->c + y * f->s, -x * f->s + y * f->c} : (SlicePoint){x, y};
}

static void slices_noise(
    Arena *mem, Pcg64 *rng, float *d, int rows, int n, double sy, double sx, double amplitude) {
    int cols = n / slices_max(1, (int)(sx / 4)) + 4;
    double *noise = push(mem, (size_t)rows * cols * sizeof(*noise));
    double *field = push(mem, (size_t)rows * cols * sizeof(*field));
    for (size_t i = 0; i < (size_t)rows * cols; i++) {
        noise[i] = syn_normal(rng);
    }
    for (int c = 0; c < cols; c++) {
        slices_filter(field + c, noise + c, rows, cols, sy);
    }
    for (int r = 0; r < rows; r++) {
        slices_filter(
            noise + r * cols, field + r * cols, cols, 1, sx / slices_max(1, (int)(sx / 4)));
    }
    double std = slices_std(noise, (size_t)rows * cols) + 1e-9;
    for (size_t i = 0; i < (size_t)rows * cols; i++) {
        noise[i] = noise[i] / std * amplitude;
    }
    for (int i = 0; i < n; i++) {
        double x = i == n - 1 ? cols - 1 : i * ((double)(cols - 1) / (n - 1));
        int a = slices_min((int)x, cols - 2);
        float t = (float)(x - a);
        for (int r = 0; r < rows; r++) {
            const double *row = noise + r * cols;
            d[(size_t)r * n + i] += (float)(row[a] * (1.0f - t) + row[a + 1] * t);
        }
    }
}

static int slices_lower(const float *u, int n, double x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (u[mid] < x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static double slices_beta(double a, double b) { return exp(lgamma(a) + lgamma(b) - lgamma(a + b)); }

static void *slices_grow(void *data, int n, size_t element) {
    if (n % 64 == 0) {
        void *grown = realloc(data, (size_t)(n + 64) * element);
        need(grown != NULL, "out of memory");
        data = grown;
    }
    return data;
}

static int slices_stack(
    Pcg64 *rng, const SliceSettings *s, double low, double high, double **rows, u8 **packet) {
    double *y = NULL, sum = 0;
    u8 *ends = NULL;
    int count = 0;
    while (sum < high - low + 440) {
        int n = slices_max(2, (int)nearbyint(slices_gamma2(rng) * (s->packet / 2)));
        double gap = s->gap * exp(.12 * syn_normal(rng));
        for (int i = 0; i < n; i++) {
            y = slices_grow(y, count, sizeof(*y));
            ends = slices_grow(ends, count, sizeof(*ends));
            y[count] = (sum + low) - 220;
            ends[count++] = i == n - 1;
            if (i == n - 1) {
                sum += gap * s->inter * exp(.3 * syn_normal(rng));
            } else {
                sum += fmax(9.4, gap * exp(s->gap_var * syn_normal(rng)));
            }
        }
    }
    y = slices_grow(y, count, sizeof(*y));
    y[count++] = (sum + low) - 220;
    *rows = y;
    *packet = ends;
    return count;
}

static void slices_lens_side(float *d, const float *u, int n, int i0, int i1, int rows, int row,
    int side, double factor, double decay, double spread, double x, double left, double right,
    double height, double p) {
    double area = slices_beta(.5, p + 1) / slices_beta(.5, 3.5);
    for (int k = 0; k < rows; k++) {
        double width = 1 + k / spread, amp = height * exp(-k / decay);
        float *sheet = d + (size_t)(row + k * side) * n;
        for (int i = i0; i < i1; i++) {
            double r = (u[i] - x) / ((u[i] < x ? left : right) * width);
            double shape = fmax(1 - r * r, 0);
            double profile = k ? area * pow(shape, 2.5) / width : pow(shape, p);
            sheet[i] = (float)(sheet[i] + factor * (amp * profile));
        }
    }
}

static SliceLens *slices_lenses(Pcg64 *rng, const SliceSettings *s, const u8 *ends, const float *u,
    int n, int rows, float *d, int *count) {
    SliceLens *lenses = NULL;
    *count = 0;
    for (int r = 0; r + 1 < rows; r++) {
        int inter = ends[r];
        int events = slices_poisson(rng, s->lens * (inter ? 1 : s->lens_inner) * n / 1000);
        for (int event = 0; event < events; event++) {
            double x = slices_uniform(rng, u[0], u[n - 1]);
            double left = s->lens_width * exp(.35 * syn_normal(rng));
            double right = s->lens_width * exp(.35 * syn_normal(rng));
            if (!inter) {
                left *= .6;
                right *= .6;
            }
            double height = fmin(s->lens_amp * exp(.45 * syn_normal(rng)),
                fmin(.6 * (left + right) / 2, fmin(.8 * fmin(left, right), 110)));
            height *= inter ? 1.0 : .45;
            double p = slices_uniform(rng, 1.5, 2.2), dc = .8 * syn_normal(rng);
            double ratio = slices_uniform(rng, .25, .75);
            double decay = slices_uniform(rng, 8, 25), spread = slices_uniform(rng, 3, 8);
            int up = slices_min(rows - r - 1, (int)(3 * decay) + 1);
            int down = slices_min(r + 1, (int)(3 * decay) + 1);
            double reach = 1 + slices_max(up, down) / spread, pad = 2 * fabs(dc) + 2;
            int i0 = slices_lower(u, n, x - left * reach - pad);
            int i1 = slices_lower(u, n, x + right * reach + pad);
            if (i1 <= i0) {
                continue;
            }
            slices_lens_side(
                d, u, n, i0, i1, up, r + 1, 1, ratio, decay, spread, x, left, right, height, p);
            slices_lens_side(d, u, n, i0, i1, down, r, -1, -(1 - ratio), decay, spread, x, left,
                right, height, p);
            lenses = slices_grow(lenses, *count, sizeof(*lenses));
            lenses[(*count)++] = (SliceLens){r, x, left, right, height};
        }
    }
    return lenses;
}

static void slices_distort(Arena *mem, Pcg64 *rng, const SliceSettings *s, const float *u, int n,
    const double *y, int rows, float *d) {
    float *field = push(mem, (size_t)n * sizeof(*field));
    int count = slices_poisson(rng, s->pinch * (n / 1000.0) * (rows / 100.0) * 2.5);
    for (int event = 0; event < count; event++) {
        int pinch = pcg64_double(rng) < .5;
        int reach = pinch ? 4 + (int)draw(rng, 10) : 3 + (int)draw(rng, 6);
        int first = (int)draw(rng, slices_max(1, rows - reach));
        int last = slices_min(rows - 1, first + reach);
        int pivot = pinch ? (first + last) / 2 : (pcg64_double(rng) < .5 ? first : last);
        double span = fmax(1, fmax(y[last] - y[pivot], y[pivot] - y[first]));
        double x = slices_uniform(rng, u[0], u[n - 1]);
        syn_normal(rng);
        double decay = slices_uniform(rng, 4, 14);
        double width, open = 0, amp = 0, beta = 0;
        if (pinch) {
            open = fmin(slices_uniform(rng, 0, 1), 40 / span);
            amp = 1 + open - slices_uniform(rng, .3, .5);
            width = fmax(slices_uniform(rng, 40, 120), span * amp * .607 / .45);
        } else {
            beta = fmin(slices_uniform(rng, .6, 1.2), log(1 + 90 / span));
            beta *= draw(rng, 2) ? 1 : -1;
            width = fmax(slices_uniform(rng, 50, 140), span * fabs(beta) * exp(fabs(beta)) / .45);
        }
        for (int i = 0; i < n; i++) {
            double x0 = (u[i] - (float)x) / width;
            field[i] = (float)(pinch ? open - amp * exp(-.5 * x0 * x0) : exp(beta * tanh(x0)) - 1);
        }
        int tail = (int)(4 * decay) + 1;
        for (int r = slices_max(0, first - tail); r <= slices_min(rows - 1, last + tail); r++) {
            int parent = slices_min(last, slices_max(first, r));
            float rel = (float)(y[parent] - y[pivot]);
            double scale = exp(-abs(r - parent) / decay);
            for (int i = 0; i < n; i++) {
                size_t at = (size_t)r * n + i;
                d[at] = (float)(d[at] + (rel * field[i]) * scale);
            }
        }
    }
    count = slices_poisson(rng, s->shift * (n / 1000.0) * (rows / 100.0));
    for (int event = 0; event < count; event++) {
        double row = slices_uniform(rng, 0, rows), scale = slices_uniform(rng, 2, 10);
        double amp = slices_uniform(rng, 15, 55);
        amp *= draw(rng, 2) ? 1 : -1;
        double x = slices_uniform(rng, u[0], u[n - 1]);
        double width = fmax(slices_uniform(rng, 40, 120), fabs(amp) / .9);
        syn_normal(rng);
        for (int i = 0; i < n; i++) {
            field[i] = (float)(amp * .5 * (1 + tanh((u[i] - (float)x) / width)));
        }
        for (int r = 0; r < rows; r++) {
            float weight = (float)(1 / (1 + exp(-(r - row) / scale)));
            for (int i = 0; i < n; i++) {
                d[(size_t)r * n + i] += weight * field[i];
            }
        }
    }
}

static void slices_spacing(Arena *mem, const SliceFold *fold, const float *u, int n,
    const double *row, int rows, float *field) {
    size_t size = (size_t)n * rows;
    double *initial = push(mem, size * sizeof(*initial));
    double *v = push(mem, size * sizeof(*v));
    float *spacing = push(mem, size * sizeof(*spacing));
    double *smooth = push(mem, (size_t)n * sizeof(*smooth));
    double *stretch = push(mem, (size_t)n * sizeof(*stretch));
    double *peak = push(mem, (size_t)n * sizeof(*peak));
    int anchor = 0;
    for (int r = 0; r < rows; r++) {
        if (fabs(row[r]) < fabs(row[anchor])) {
            anchor = r;
        }
        for (int i = 0; i < n; i++) {
            size_t at = (size_t)r * n + i;
            initial[at] = v[at] = row[r] + field[at];
            spacing[at] = 9;
        }
    }
    for (int pass = 0; pass < 4; pass++) {
        for (int r = 0; r < rows; r++) {
            size_t at = (size_t)r * n;
            slices_filter(smooth, v + at, n, 1, 4);
            if (pass) {
                for (int i = 0; i < n; i++) {
                    smooth[i] = (float)smooth[i];
                }
            }
            for (int i = 0; i < n; i++) {
                int left = slices_max(0, i - 1), right = slices_min(n - 1, i + 1);
                double slope = (smooth[right] - smooth[left]) / (right - left);
                SlicePoint a = slices_transform(fold, u[i], v[at + i], 0);
                double shifted = v[at + i] + .5 * slope;
                if (pass) {
                    shifted = (float)v[at + i] + .5f * (float)slope;
                }
                SlicePoint b = slices_transform(fold, u[i] + .5f, shifted, 0);
                stretch[i] = 9.0f * (float)(hypot(b.x - a.x, b.y - a.y) / .5);
            }
            for (int i = 0; i < n; i++) {
                double max = 0;
                for (int k = slices_max(0, i - 12); k <= slices_min(n - 1, i + 12); k++) {
                    max = fmax(max, stretch[k]);
                }
                peak[i] = max;
            }
            slices_filter(smooth, peak, n, 1, 6);
            for (int i = 0; i < n; i++) {
                float gap = (float)smooth[i] * 1.03f;
                spacing[at + i] = fminf(fmaxf(spacing[at + i], gap), 36);
            }
        }
        for (int i = 0; i < n; i++) {
            float position = 0;
            v[i] = 0;
            for (int r = 1; r < rows; r++) {
                size_t at = (size_t)r * n + i;
                double min = fmaxf(spacing[at], spacing[at - n]);
                double delta = (initial[at] - initial[at - n] - min) / .6;
                position += (float)(min + .6 * (fmax(delta, 0) + log1p(exp(-fabs(delta)))));
                v[at] = position;
            }
            double offset = initial[(size_t)anchor * n + i] - v[(size_t)anchor * n + i];
            for (int r = 0; r < rows; r++) {
                size_t at = (size_t)r * n + i;
                v[at] = (float)(v[at] + offset);
            }
        }
    }
    for (int r = 0; r < rows; r++) {
        size_t at = (size_t)r * n;
        slices_filter(smooth, v + at, n, 1, 2);
        for (int i = 0; i < n; i++) {
            field[at + i] = (float)smooth[i];
        }
    }
}

static SliceTear *slices_tears(Pcg64 *rng, const SliceSettings *s, const SliceLens *lenses,
    int count, const float *u, int n, const float *v, int *made) {
    SliceTear *tears = NULL;
    *made = 0;
    for (int k = 0; k < count; k++) {
        const SliceLens *lens = &lenses[k];
        if (lens->height < 20 || pcg64_double(rng) > s->tear) {
            continue;
        }
        int upper = pcg64_double(rng) < .5;
        int i0 = slices_lower(u, n, lens->x - lens->left);
        int i1 = slices_lower(u, n, lens->x + lens->right);
        if (i1 - i0 < 20) {
            continue;
        }
        const float *a = v + (size_t)lens->row * n, *b = a + n;
        int peak = i0;
        for (int i = i0 + 1; i < i1; i++) {
            if (b[i] - a[i] > b[peak] - a[peak]) {
                peak = i;
            }
        }
        float threshold = fmaxf(14, .5f * (b[peak] - a[peak]));
        if (b[peak] - a[peak] < threshold) {
            continue;
        }
        int first = peak, last = peak;
        while (first > i0 && b[first - 1] - a[first - 1] >= threshold) {
            first--;
        }
        while (last + 1 < i1 && b[last + 1] - a[last + 1] >= threshold) {
            last++;
        }
        double lo = u[first] + 3.0f, hi = u[last] - 3.0f;
        if (hi - lo < 14) {
            continue;
        }
        double left, right;
        if (pcg64_double(rng) < .3) {
            if (pcg64_double(rng) < .5) {
                left = u[0] - 10.0f;
                right = slices_uniform(rng, lo, hi);
            } else {
                left = slices_uniform(rng, lo, hi);
                right = u[n - 1] + 10.0f;
            }
        } else {
            left = slices_uniform(rng, lo, hi - 6);
            right = left + slices_uniform(rng, 6, fmax(7, hi - left));
        }
        double curl = slices_uniform(rng, .03, .2), length = slices_uniform(rng, 15, 45);
        tears = slices_grow(tears, *made, sizeof(*tears));
        tears[(*made)++] =
            (SliceTear){lens->row + upper, lens->row, left, right, upper ? -1 : 1, curl, length};
    }
    return tears;
}

static SlicePoint slices_clip(SlicePoint p, SlicePoint q, int w, int h) {
    double dx = q.x - p.x, dy = q.y - p.y, t = 1;
    if (q.x < -.5 || q.x > w - .5) {
        t = fmin(t, ((q.x < -.5 ? -.5 : w - .5) - p.x) / dx);
    }
    if (q.y < -.5 || q.y > h - .5) {
        t = fmin(t, ((q.y < -.5 ? -.5 : h - .5) - p.y) / dy);
    }
    t = fmax(t, 0);
    return (SlicePoint){p.x + t * dx, p.y + t * dy};
}

static int slices_inside(SlicePoint p, int w, int h) {
    return p.x >= -.5 && p.x <= w - .5 && p.y >= -.5 && p.y <= h - .5;
}

typedef struct {
    int x, y;
} SlicePixel;

static void slices_raster(SlicePoint *points, double *lengths, SlicePixel *pixels, int count,
    u16 *out, int w, int h, u16 id) {
    lengths[0] = 0;
    for (int i = 0; i < count; i++) {
        points[i].x = (float)points[i].x;
        points[i].y = (float)points[i].y;
        if (i) {
            float dx = (float)points[i].x - (float)points[i - 1].x;
            float dy = (float)points[i].y - (float)points[i - 1].y;
            lengths[i] = (float)lengths[i - 1] + hypotf(dx, dy);
        }
    }
    int edge = 0, n = 0;
    SlicePixel prev = {-2, -2};
    for (int i = 0; i * .3 < lengths[count - 1] + 1e-6; i++) {
        double distance = i * .3;
        while (edge + 1 < count && lengths[edge + 1] <= distance) {
            edge++;
        }
        SlicePoint p = points[edge];
        if (edge + 1 < count) {
            double step = distance - lengths[edge], total = lengths[edge + 1] - lengths[edge];
            p.x += ((points[edge + 1].x - p.x) / total) * step;
            p.y += ((points[edge + 1].y - p.y) / total) * step;
        }
        SlicePixel xy = {
            slices_bound((int)nearbyint(p.x), w), slices_bound((int)nearbyint(p.y), h)};
        if (xy.x == prev.x && xy.y == prev.y) {
            continue;
        }
        prev = xy;
        while (n >= 2 && abs(xy.x - pixels[n - 2].x) <= 1 && abs(xy.y - pixels[n - 2].y) <= 1) {
            n--;
        }
        if (n == 0 || xy.x != pixels[n - 1].x || xy.y != pixels[n - 1].y) {
            pixels[n++] = xy;
        }
    }
    for (int i = 0; i < n; i++) {
        out[(size_t)pixels[i].y * w + pixels[i].x] = id;
    }
}

static void slices_draw(Arena *mem, const SliceFold *fold, const float *u, int n, const float *v,
    int rows, const SliceTear *tears, int torn, u16 *out, int w, int h) {
    float *line = push(mem, (size_t)n * sizeof(*line));
    u8 *present = push(mem, n);
    SlicePoint *xy = push(mem, (size_t)n * sizeof(*xy));
    SlicePoint *clipped = push(mem, (size_t)(n + 2) * sizeof(*xy));
    double *lengths = push(mem, (size_t)(n + 2) * sizeof(*lengths));
    SlicePixel *pixels = push(mem, (size_t)w * h * sizeof(*pixels));
    u16 id = 0;
    for (int r = 0; r < rows; r++) {
        memcpy(line, v + (size_t)r * n, (size_t)n * sizeof(*line));
        memset(present, 1, n);
        for (int k = 0; k < torn; k++) {
            if (tears[k].row == r) {
                for (int i = 0; i < n; i++) {
                    present[i] &= !(u[i] > (float)tears[k].left && u[i] < (float)tears[k].right);
                }
            }
        }
        for (int k = 0; k < torn; k++) {
            const SliceTear *tear = &tears[k];
            if (tear->row != r) {
                continue;
            }
            double ends[] = {tear->left, tear->right};
            for (int end = 0; end < 2; end++) {
                double x = ends[end];
                if (x < u[0] || x > u[n - 1]) {
                    continue;
                }
                int at = slices_lower(u, n, x);
                const float *lower = v + (size_t)tear->gap * n, *upper = lower + n;
                double height = tear->side * tear->curl * fmax(0, upper[at] - lower[at] - 9);
                for (int i = 0; i < n; i++) {
                    double span = fmax(1 - fabs(u[i] - x) / tear->length, 0);
                    line[i] = (float)(line[i] + present[i] * height * span * span);
                }
            }
        }
        int visible = 0;
        for (int start = 0; start < n;) {
            if (!present[start]) {
                start++;
                continue;
            }
            int end = start + 1;
            while (end < n && present[end]) {
                end++;
            }
            if (end - start >= 25) {
                for (int i = start; i < end; i++) {
                    xy[i] = slices_transform(fold, u[i], line[i], 0);
                    xy[i].x += w / 2.0 - .5;
                    xy[i].y += h / 2.0 - .5;
                }
                for (int first = start; first < end;) {
                    if (!slices_inside(xy[first], w, h)) {
                        first++;
                        continue;
                    }
                    int last = first + 1;
                    while (last < end && slices_inside(xy[last], w, h)) {
                        last++;
                    }
                    int count = 0;
                    if (first > start) {
                        clipped[count++] = slices_clip(xy[first], xy[first - 1], w, h);
                    }
                    for (int i = first; i < last; i++) {
                        clipped[count++] = xy[i];
                    }
                    if (last < end) {
                        clipped[count++] = slices_clip(xy[last - 1], xy[last], w, h);
                    }
                    if (count >= 2) {
                        visible = 1;
                        slices_raster(clipped, lengths, pixels, count, out, w, h, id + 1);
                    }
                    first = last;
                }
            }
            start = end;
        }
        id += visible;
    }
}

static void slices_layout(Pcg64 *rng, int w, int h, u16 *out) {
    Arena mem = arena(GB);
    SliceSettings s = slices_settings(rng);
    int wp = w + 96, hp = h + 96, border = (wp + hp) / 2;
    double reach = 1.5 * hypot(wp, hp) + 400;
    SliceFold fold = slices_fold(&mem, &s, rng, reach);
    double umin = INFINITY, umax = -INFINITY, vmin = INFINITY, vmax = -INFINITY;
    for (int i = 0; i < border; i++) {
        double t = i * (2.0 / (hp + wp));
        double x[] = {t * wp, wp, (1 - t) * wp, 0};
        double y[] = {0, t * hp, hp, (1 - t) * hp};
        for (int k = 0; k < 4; k++) {
            SlicePoint p = slices_transform(&fold, x[k] - wp / 2.0, y[k] - hp / 2.0, 1);
            umin = fmin(umin, p.x);
            umax = fmax(umax, p.x);
            vmin = fmin(vmin, p.y);
            vmax = fmax(vmax, p.y);
        }
    }
    int n = (int)ceil((umax + 30) - (umin - 30));
    float *u = push(&mem, (size_t)n * sizeof(*u));
    for (int i = 0; i < n; i++) {
        u[i] = (float)((umin - 30) + i);
    }
    double *row;
    u8 *ends;
    int rows = slices_stack(rng, &s, vmin, vmax, &row, &ends);
    float *v = pushz(&mem, (size_t)n * rows * sizeof(*v));
    int lenses, torn;
    SliceLens *lens = slices_lenses(rng, &s, ends, u, n, rows, v, &lenses);
    slices_distort(&mem, rng, &s, u, n, row, rows, v);
    slices_noise(&mem, rng, v, rows, n, s.wave_rows, s.wave_width, s.wave);
    slices_noise(&mem, rng, v, rows, n, .8, s.ripple_width, s.ripple);
    slices_spacing(&mem, &fold, u, n, row, rows, v);
    SliceTear *tears = slices_tears(rng, &s, lens, lenses, u, n, v, &torn);
    for (int r = 0; r < rows; r++) {
        syn_normal(rng);
    }
    memset(out, 0, (size_t)w * h * sizeof(*out));
    slices_draw(&mem, &fold, u, n, v, rows, tears, torn, out, w, h);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        syn_normal(rng);
    }
    free(row);
    free(ends);
    free(lens);
    free(tears);
    munmap(mem.base, mem.size);
}
