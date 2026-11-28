enum { SLICE_PATCH = 128, SLICE_PATCH_PIXELS = SLICE_PATCH * SLICE_PATCH, DISC_BATCH = 32 };

typedef struct {
    size_t count;
    float *data;
} SliceCrops;

static int disc_rank(const int *histogram, int rank) {
    for (int i = 0; i < 256; i++) {
        rank -= histogram[i];
        if (rank < 0) {
            return i;
        }
    }
    return 255;
}

static double disc_quantile(const int *histogram, int n, double q) {
    double index = (n - 1) * q;
    int lo = disc_rank(histogram, (int)index), hi = disc_rank(histogram, (int)ceil(index));
    return lo + (index - floor(index)) * (hi - lo);
}

static void disc_patch(float *out, const u8 *crop, int h, int w, Pcg64 *rng) {
    int y = draw(rng, slices_max(h - SLICE_PATCH, 1));
    int x = draw(rng, slices_max(w - SLICE_PATCH, 1));
    const u8 *patch = crop + (size_t)y * w + x;
    int histogram[256] = {0};
    for (int y0 = 0; y0 < SLICE_PATCH; y0++) {
        for (int x0 = 0; x0 < SLICE_PATCH; x0++) {
            histogram[patch[(size_t)y0 * w + x0]]++;
        }
    }
    int n = SLICE_PATCH_PIXELS - histogram[0];
    histogram[0] = 0;
    double lo = n ? disc_quantile(histogram, n, .05) : 0;
    double hi = n ? disc_quantile(histogram, n, .95) : 1;
    double span = fmax(1, hi - lo);
    for (int y0 = 0; y0 < SLICE_PATCH; y0++) {
        for (int x0 = 0; x0 < SLICE_PATCH; x0++) {
            u8 value = patch[(size_t)y0 * w + x0];
            out[y0 * SLICE_PATCH + x0] = value ? (float)((value - lo) / span) : 0;
        }
    }
}

static void disc_patches(SliceCrops *crops, const char *path, const char *name, Pcg64 *rng) {
    char member[64];
    snprintf(member, sizeof(member), "%s.npy", name);
    int error;
    zip_t *zip = zip_open(path, ZIP_RDONLY, &error);
    need(zip != NULL, "%s: cannot open crops archive", path);
    zip_stat_t stat;
    need(zip_stat(zip, member, 0, &stat) == 0 && stat.size < SIZE_MAX - (1 << 20),
        "%s: missing or oversized %s", path, member);
    zip_close(zip);
    Arena mem = arena(stat.size + (1 << 20));
    Array a = npz_get(&mem, path, name);
    need(a.ndim == 3 && !strcmp(a.descr, "|u1") && a.shape[0] > 0 && a.shape[0] < INT_MAX &&
            a.shape[1] >= SLICE_PATCH && a.shape[1] < INT_MAX && a.shape[2] >= SLICE_PATCH &&
            a.shape[2] < INT_MAX,
        "%s: %s must be a nonempty uint8 crop stack at least %d pixels wide and tall", path, name,
        SLICE_PATCH);
    int count = (int)a.shape[0], h = (int)a.shape[1], w = (int)a.shape[2];
    size_t pixels = (size_t)h * w;
    need(
        a.size / pixels == (size_t)count && a.size % pixels == 0, "%s: truncated crop stack", path);
    size_t all = crops->count + count;
    need(all < (size_t)INT_MAX && all <= SIZE_MAX / (SLICE_PATCH_PIXELS * sizeof(float)),
        "too many discriminator crops");
    float *data = realloc(crops->data, all * SLICE_PATCH_PIXELS * sizeof(*data));
    need(data != NULL, "out of memory loading discriminator crops");
    for (int i = 0; i < count; i++) {
        disc_patch(data + (crops->count + i) * SLICE_PATCH_PIXELS,
            (const u8 *)a.data + (size_t)i * pixels, h, w, rng);
    }
    crops->data = data;
    crops->count = all;
    munmap(mem.base, mem.size);
}

typedef struct {
    float *value, *grad, *mean, *square;
    int count;
} DiscParam;

typedef struct {
    int ci, co, h, w, kernel, stride, pool;
    DiscParam weight, bias;
} DiscLayer;

typedef struct {
    float *columns, *output, *features;
    u16 *winner;
} DiscActivation;

typedef struct {
    DiscLayer stage[4];
    DiscParam weight, bias;
    int layers;
} DiscNetwork;

static DiscParam disc_init(Arena *mem, SliceNoise *noise, int count, int fan_in) {
    DiscParam p = {
        .value = push(mem, count * sizeof(float)),
        .grad = pushz(mem, count * sizeof(float)),
        .mean = pushz(mem, count * sizeof(float)),
        .square = pushz(mem, count * sizeof(float)),
        .count = count,
    };
    float bound = (float)(1 / sqrt(fan_in));
    for (int i = 0; i < count; i++) {
        p.value[i] = fmaf((float)slices_noise_uniform(noise, 1), 2 * bound, -bound);
    }
    return p;
}

static DiscNetwork disc_network(Arena *mem, SliceNoise *noise, int fidelity) {
    DiscNetwork net = {.layers = fidelity ? 3 : 4};
    const int channels[] = {fidelity ? 2 : 1, 16, 32, 64, 64};
    int height = fidelity ? 16 : SLICE_PATCH, width = fidelity ? 32 : SLICE_PATCH;
    for (int k = 0; k < net.layers; k++) {
        int ci = channels[k], co = channels[k + 1], kernel = k || fidelity ? 3 : 5;
        int fan_in = ci * kernel * kernel;
        net.stage[k] = (DiscLayer){.ci = ci,
            .co = co,
            .h = height >> k,
            .w = width >> k,
            .kernel = kernel,
            .stride = fidelity ? 1 : 2,
            .pool = fidelity && k + 1 < net.layers};
        net.stage[k].weight = disc_init(mem, noise, co * fan_in, fan_in);
        net.stage[k].bias = disc_init(mem, noise, co, fan_in);
    }
    net.weight = disc_init(mem, noise, 64, 64);
    net.bias = disc_init(mem, noise, 1, 64);
    return net;
}

static void disc_conv_columns(
    const DiscLayer *layer, const float *input, float *matrix, int transpose) {
    int h = layer->h, w = layer->w, stride = layer->stride;
    int rows = h / stride, cols = w / stride, pixels = rows * cols;
    int k = layer->kernel, pad = k / 2;
    float *gradient = transpose ? matrix : NULL;
    const float *data = input;
    if (transpose) {
        memset(gradient, 0, (size_t)layer->ci * h * w * sizeof(float));
    }
    for (int c = 0; c < layer->ci; c++) {
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                int col = (c * k + ky) * k + kx;
                for (int y = 0; y < rows; y++) {
                    int iy = y * stride + ky - pad;
                    size_t at = (size_t)col * pixels + y * cols;
                    for (int x = 0; x < cols; x++) {
                        int ix = x * stride + kx - pad;
                        int inside = iy >= 0 && iy < h && ix >= 0 && ix < w;
                        if (transpose) {
                            if (inside) {
                                gradient[((size_t)c * h + iy) * w + ix] += data[at + x];
                            }
                        } else {
                            matrix[at + x] = inside ? data[((size_t)c * h + iy) * w + ix] : 0;
                        }
                    }
                }
            }
        }
    }
}

static DiscActivation disc_conv_forward(Arena *mem, const DiscLayer *layer, const float *input) {
    int ci = layer->ci, co = layer->co, k = layer->kernel, cols = ci * k * k;
    int h = layer->h / layer->stride, w = layer->w / layer->stride, pixels = h * w;
    float *matrix = push(mem, (size_t)cols * pixels * sizeof(float));
    float *output = push(mem, (size_t)co * pixels * sizeof(float));
    disc_conv_columns(layer, input, matrix, 0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, co, pixels, cols, 1, layer->weight.value,
        cols, matrix, pixels, 0, output, pixels);
    for (int c = 0; c < co; c++) {
        float bias = layer->bias.value[c];
        for (int p = 0; p < pixels; p++) {
            size_t at = (size_t)c * pixels + p;
            output[at] = fmaxf(0, output[at] + bias);
        }
    }
    if (!layer->pool) {
        return (DiscActivation){.columns = matrix, .output = output, .features = output};
    }
    int n = co * pixels / 4;
    float *pooled = push(mem, (size_t)n * sizeof(float));
    u16 *winner = push(mem, (size_t)n * sizeof(*winner));
    size_t at = 0;
    for (int c = 0; c < co; c++) {
        for (int y = 0; y < h; y += 2) {
            for (int x = 0; x < w; x += 2) {
                const int first = (c * h + y) * w + x;
                const int pool[] = {first, first + 1, first + w, first + w + 1};
                int best = first;
                for (int i = 1; i < 4; i++) {
                    if (output[pool[i]] > output[best]) {
                        best = pool[i];
                    }
                }
                winner[at] = best;
                pooled[at++] = output[best];
            }
        }
    }
    return (DiscActivation){
        .columns = matrix, .output = output, .features = pooled, .winner = winner};
}

static float *disc_conv_backward(
    Arena *mem, DiscLayer *layer, DiscActivation activation, float *gradient, int propagate) {
    int in = layer->ci, out = layer->co, h = layer->h, w = layer->w;
    int pixels = h / layer->stride * (w / layer->stride);
    int cols = in * layer->kernel * layer->kernel;
    if (activation.winner) {
        float *expanded = pushz(mem, (size_t)out * pixels * sizeof(float));
        for (int i = 0; i < out * pixels / 4; i++) {
            expanded[activation.winner[i]] = gradient[i];
        }
        gradient = expanded;
    }
    for (int c = 0; c < out; c++) {
        float sum = 0;
        for (int p = 0; p < pixels; p++) {
            size_t at = (size_t)c * pixels + p;
            if (activation.output[at] <= 0) {
                gradient[at] = 0;
            }
            sum += gradient[at];
        }
        layer->bias.grad[c] += sum;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, out, cols, pixels, 1, gradient, pixels,
        activation.columns, pixels, 1, layer->weight.grad, cols);
    if (!propagate) {
        return NULL;
    }
    float *updated = push(mem, (size_t)in * h * w * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, cols, pixels, out, 1, layer->weight.value,
        cols, gradient, pixels, 0, activation.columns, pixels);
    disc_conv_columns(layer, activation.columns, updated, 1);
    return updated;
}

static float disc_predict(
    Arena *mem, DiscNetwork *net, const float *x, int label, int batch, int flip) {
    if (flip) {
        int h = net->stage[0].h * net->stage[0].ci, w = net->stage[0].w;
        float *reversed = push(mem, (size_t)h * w * sizeof(float));
        for (int y = 0; y < h; y++) {
            for (int i = 0; i < w; i++) {
                reversed[y * w + i] = x[y * w + w - 1 - i];
            }
        }
        x = reversed;
    }
    DiscActivation layers[4];
    for (int i = 0; i < net->layers; i++) {
        layers[i] = disc_conv_forward(mem, &net->stage[i], x);
        x = layers[i].features;
    }
    const DiscLayer *last = net->stage + net->layers - 1;
    int pixels = last->h / last->stride * (last->w / last->stride);
    float avg[64];
    float logit = net->bias.value[0];
    for (int c = 0; c < 64; c++) {
        float sum = 0;
        for (int i = 0; i < pixels; i++) {
            sum += x[c * pixels + i];
        }
        avg[c] = sum / pixels;
        logit += avg[c] * net->weight.value[c];
    }
    if (batch) {
        float derivative = (1.0f / (1.0f + expf(-logit)) - label) / batch;
        net->bias.grad[0] += derivative;
        float *gradient = push(mem, 64 * pixels * sizeof(float));
        for (int c = 0; c < 64; c++) {
            net->weight.grad[c] += avg[c] * derivative;
            float part = net->weight.value[c] * derivative / pixels;
            for (int p = 0; p < pixels; p++) {
                gradient[c * pixels + p] = part;
            }
        }
        for (int i = net->layers - 1; i >= 0; i--) {
            gradient = disc_conv_backward(mem, &net->stage[i], layers[i], gradient, i != 0);
        }
    }
    return logit;
}

static void disc_adam(DiscParam p, float step, float correction) {
    for (int i = 0; i < p.count; i++) {
        float grad = p.grad[i] + 1e-4f * p.value[i];
        p.mean[i] += .1f * (grad - p.mean[i]);
        p.square[i] = .999f * p.square[i] + .001f * grad * grad;
        p.value[i] -= step * p.mean[i] / (sqrtf(p.square[i]) / correction + 1e-8f);
        p.grad[i] = 0;
    }
}

static void disc_step(DiscNetwork *net, int step) {
    float alpha = (float)(1e-3 / (1 - pow(.9, step + 1.0)));
    float correction = (float)sqrt(1 - pow(.999, step + 1.0));
    for (int i = 0; i < net->layers; i++) {
        disc_adam(net->stage[i].weight, alpha, correction);
        disc_adam(net->stage[i].bias, alpha, correction);
    }
    disc_adam(net->weight, alpha, correction);
    disc_adam(net->bias, alpha, correction);
}

typedef struct {
    float score;
    int label;
} DiscScore;

static int disc_score_order(const void *a, const void *b) {
    float x = ((const DiscScore *)a)->score, y = ((const DiscScore *)b)->score;
    return (x > y) - (x < y);
}

static double disc_auc(DiscScore *a, int n) {
    qsort(a, n, sizeof(*a), disc_score_order);
    double pos_rank = 0;
    int positives = 0;
    for (int start = 0; start < n;) {
        int end = start + 1;
        while (end < n && a[end].score == a[start].score) {
            end++;
        }
        double rank = start + 1 + (end - start - 1) / 2.0;
        for (int i = start; i < end; i++) {
            if (a[i].label) {
                pos_rank += rank;
                positives++;
            }
        }
        start = end;
    }
    double n1 = positives, n0 = n - positives;
    return (pos_rank - n1 * (n1 + 1) / 2) / fmax(n1 * n0, 1);
}

static int disc_interval(Pcg64 *rng, int max) {
    u32 mask = (u32)max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    u32 value;
    do {
        value = pcg64_next32(rng) & mask;
    } while (value > (u32)max);
    return (int)value;
}

static const float *disc_crop(
    const SliceCrops *real, const SliceCrops *synthetic, size_t id, int *label) {
    *label = id < real->count;
    return *label ? real->data + id * SLICE_PATCH_PIXELS
                  : synthetic->data + (id - real->count) * SLICE_PATCH_PIXELS;
}

static void disc_fit(const SliceCrops *real, const SliceCrops *synthetic, int steps, u64 seed,
    double *auc, double *loss) {
    size_t total = real->count + synthetic->count;
    need(total < INT_MAX, "too many discriminator crops");
    int count = (int)total, train = (int)(.7 * total), eval = count - train;
    Arena mem = arena((size_t)count * (sizeof(int) + sizeof(DiscScore)) + (8 << 20));
    int *order = push(&mem, (size_t)count * sizeof(*order));
    DiscScore *scores = push(&mem, (size_t)eval * sizeof(*scores));
    for (int i = 0; i < count; i++) {
        order[i] = i;
    }
    Pcg64 rng = pcg64(seed);
    for (int i = count - 1; i > 0; i--) {
        int at = disc_interval(&rng, i), swap = order[i];
        order[i] = order[at];
        order[at] = swap;
    }
    SliceNoise noise = slices_noise_seed(seed);
    DiscNetwork net = disc_network(&mem, &noise, 0);
    size_t used = mem.used;
    for (int step = 0; step < steps; step++) {
        int flip = slices_noise_uniform(&noise, 1) < .5;
        for (int i = 0; i < DISC_BATCH; i++) {
            int label, item = order[draw(&rng, train)];
            const float *patch = disc_crop(real, synthetic, item, &label);
            disc_predict(&mem, &net, patch, label, DISC_BATCH, flip);
            mem.used = used;
        }
        disc_step(&net, step);
    }
    float losses[2] = {0};
    int classes[2] = {0};
    for (int i = 0; i < eval; i++) {
        int label;
        const float *patch = disc_crop(real, synthetic, order[train + i], &label);
        float score = disc_predict(&mem, &net, patch, label, 0, 0);
        scores[i] = (DiscScore){score, label};
        losses[label] += fmaxf(score, 0) - score * label + log1pf(expf(-fabsf(score)));
        classes[label]++;
        mem.used = used;
    }
    *auc = disc_auc(scores, eval);
    *loss = classes[0] && classes[1] ? .5 * (losses[0] / classes[0] + losses[1] / classes[1]) : NAN;
    munmap(mem.base, mem.size);
}

static int slices_discriminator_command(int argc, char **argv) {
    Arena args = arena((size_t)argc * sizeof(char *) + 4096);
    const char **paths = push(&args, (size_t)argc * sizeof(*paths));
    int count = 0, steps = 400, positional = 0;
    u64 seed = 0;
    for (int i = 0; i < argc; i++) {
        if (positional) {
            paths[count++] = argv[i];
            continue;
        }
        char split[32];
        const char *value;
        const char *key = slices_cli_option(argv[i], split, &value);
        if (!strcmp(key, "--")) {
            positional = 1;
        } else if (!strcmp(key, "--help") || !strcmp(key, "-h")) {
            puts("synthetic.two_d.discriminator REAL.npz SHARD.npz [...] [--steps N] [--seed N]");
            munmap(args.base, args.size);
            return 0;
        } else if (!strcmp(key, "--steps") || !strcmp(key, "--seed")) {
            if (!value) {
                need(i + 1 < argc, "%s needs a value", key);
                value = argv[++i];
            }
            if (!strcmp(key, "--steps")) {
                steps = slices_positive(value, key);
            } else {
                seed = slices_unsigned(value, key);
            }
        } else {
            need(key[0] != '-', "unknown discriminator option: %s", key);
            paths[count++] = key;
        }
    }
    need(count >= 2, "synthetic.two_d.discriminator requires REAL.npz and synthetic shards");
    SliceCrops real = {0}, synthetic = {0};
    Pcg64 rng = pcg64(seed);
    disc_patches(&real, paths[0], "crops", &rng);
    for (int i = 1; i < count; i++) {
        disc_patches(&synthetic, paths[i], "image", &rng);
    }
    double auc, loss;
    disc_fit(&real, &synthetic, steps, seed, &auc, &loss);
    printf("{\"real\": %zu, \"synthetic\": %zu, \"auc\": %.17g, \"loss\": ", real.count,
        synthetic.count, auc);
    if (isnan(loss)) {
        puts("NaN}");
    } else {
        printf("%.17g}\n", loss);
    }
    free(real.data);
    free(synthetic.data);
    munmap(args.base, args.size);
    return 0;
}
