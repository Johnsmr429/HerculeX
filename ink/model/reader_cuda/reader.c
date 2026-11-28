void *gpu_alloc(size_t n);
void gpu_put(void *to, const void *from, size_t n);
void gpu_get(void *to, const void *from, size_t n);
void gpu_measure(const u8 *tiles, const f32 *table, const f32 *conv1, const f32 *conv2, f32 *image,
    f32 *first, f32 *second, int count, int h, int w, const f32 *norm, f32 *mean, f32 *variance,
    int stage, void *workspace, size_t room);
void gpu_stem(const u8 *stack, int height, int width, int y0, int x0, int rows, int cols,
    const u16 *table, const u16 *w1, const u16 *b1, const u16 *w2, const u16 *b2,
    const u16 *attention, u16 attention_bias, u16 *image, u16 *first, u16 *second, u16 *features,
    int h, int w, void *workspace, size_t room);
void gpu_conv(const u16 *x, int batch, int in, int h, int w, const u16 *weight, int out, int k,
    int stride, u16 *y, void *workspace, size_t room);
void gpu_up(const u16 *x, int in, int h, int w, const u16 *weight, const u16 *bias, int out,
    const u16 *skip, u16 *y, f32 *acc);
void gpu_norm(const u16 *x, const u16 *bias, int channels, int batch, i64 n, const u16 *gamma,
    const u16 *beta, const u16 *residual, int nonlinear, u16 *y);
void gpu_pool(const u16 *x, int channels, int h, int w, u16 *y);
void gpu_ink(const u16 *logits, const u16 *bias, f32 *read, i64 n);
void gpu_blend(const f32 *read, const f32 *window, int y0, int x0, int th, int tw, int tx,
    f32 *total, f32 *weight, int width);
void gpu_finish(f32 *total, const f32 *weight, i64 n, f32 *host);
void gpu_zero(void *p, size_t n);

#define PLANE ((i64)TILE * TILE)

const char *lane_kernels = "ink/model/reader_cuda/reader.cu";
#define BATCH 4
#define ROOM (64 * 9 * PLANE * 2)

typedef struct {
    Tensor *tensors;
    int count;
    u16 **weights;
    Stem stem_weights;
    u16 *table, *w1, *b1, *w2, *b2, *att;
    u16 attention_bias;
    f32 *measure_table, *conv1, *conv2, *norm, *measure_image, *first_measured, *second_measured;
    u8 *tiles;
    u16 *image, *first, *second, *features, *skips[6], *a, *b, *c, *p, *q, *cat, *cols;
    f32 *acc, *read, *window, *total, *weight;
    u8 *stack, *host_tiles;
    Arena *arena;
    Reading work;
    i64 queued;
} Reader;

u16 half_bits(f32 f) {
    u32 x;
    memcpy(&x, &f, 4);
    u32 sign = x >> 16 & 0x8000, mantissa = x & 0x7fffff;
    i32 exponent = (i32)(x >> 23 & 0xff) - 112;
    if ((x & 0x7fffffff) > 0x7f800000) {
        return sign | 0x7e00;
    }
    if (exponent >= 31) {
        return sign | 0x7c00;
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000;
        int shift = 14 - exponent;
        u32 half = mantissa >> shift, rest = mantissa & ((1u << shift) - 1),
            middle = 1u << (shift - 1);
        return sign | (half + (rest > middle || (rest == middle && (half & 1))));
    }
    u32 half = sign | exponent << 10 | mantissa >> 13, rest = mantissa & 0x1fff;
    return half + (rest > 0x1000 || (rest == 0x1000 && (half & 1)));
}

u16 *upload_half(const f32 *values, i64 n) {
    u16 *bits = malloc(n * 2);
    for (i64 i = 0; i < n; i++) {
        bits[i] = half_bits(values[i]);
    }
    u16 *device = gpu_alloc(n * 2);
    gpu_put(device, bits, n * 2);
    free(bits);
    return device;
}

u16 *weight(Reader *r, const char *name) {
    Tensor *t = tensor_named(r->tensors, r->count, fmt(r->arena, "model.network.%s", name));
    i64 i = t - r->tensors;
    if (!r->weights[i]) {
        r->weights[i] = upload_half(t->data, tensor_elements(t));
    }
    return r->weights[i];
}

Reader *reader_open(Arena *a, const char *weights, i64 columns) {
    Reader *r = pushz(a, sizeof(Reader));
    reading_init(&r->work, a, columns);
    r->arena = a;
    r->tensors = pushz(a, TENSORS * sizeof(Tensor));
    r->count = load_tensors(a, weights, r->tensors);
    r->weights = pushz(a, r->count * sizeof(u16 *));
    for (int i = 0; i < r->count; i++) {
        Tensor *t = r->tensors + i;
        const char *bare = strncmp(t->name, "model.", 6) ? t->name : t->name + 6;
        if (!strncmp(bare, "network.", 8) && !strstr(bare, "all_modules")) {
            r->weights[i] = upload_half(t->data, tensor_elements(t));
        }
    }
    r->stem_weights = stem_of(a, r->tensors, r->count);
    r->attention_bias = half_bits(r->stem_weights.attention_bias);
    i64 measured = (i64)MEASURED_TILES * DEPTH * MEASURED_SIDE * MEASURED_SIDE;
    r->tiles = gpu_alloc(measured);
    r->measure_table = gpu_alloc(256 * 4);
    r->conv1 = gpu_alloc(sizeof(r->stem_weights.conv1));
    r->conv2 = gpu_alloc(sizeof(r->stem_weights.conv2));
    gpu_put(r->conv1, r->stem_weights.conv1, sizeof(r->stem_weights.conv1));
    gpu_put(r->conv2, r->stem_weights.conv2, sizeof(r->stem_weights.conv2));
    r->norm = gpu_alloc(2 * STEM * 4);
    r->measure_image = gpu_alloc(measured * 4);
    r->first_measured = gpu_alloc(measured * STEM * 4);
    r->second_measured = gpu_alloc(measured * STEM * 4);
    r->table = gpu_alloc(256 * 2);
    r->w1 = gpu_alloc(27 * STEM * 2);
    r->b1 = gpu_alloc(STEM * 2);
    r->w2 = gpu_alloc(27 * STEM * STEM * 2);
    r->b2 = gpu_alloc(STEM * 2);
    r->att = upload_half(r->stem_weights.attention, STEM);
    r->image = gpu_alloc(DEPTH * PLANE * 2);
    r->first = gpu_alloc(STEM * DEPTH * PLANE * 2);
    r->second = gpu_alloc(STEM * DEPTH * PLANE * 2);
    r->features = gpu_alloc(BATCH * 2 * STEM * PLANE * 2);
    int widths[6] = {32, 64, 128, 256, 320, 320};
    for (int s = 0; s < 6; s++) {
        r->skips[s] = gpu_alloc(BATCH * widths[s] * (PLANE >> 2 * s) * 2);
    }
    r->a = gpu_alloc(BATCH * 32 * PLANE * 2);
    r->c = gpu_alloc(BATCH * 32 * PLANE * 2);
    r->b = gpu_alloc(BATCH * 32 * PLANE * 2);
    r->p = gpu_alloc(BATCH * 32 * PLANE * 2);
    r->q = gpu_alloc(BATCH * 32 * PLANE * 2);
    r->cat = gpu_alloc(BATCH * 64 * PLANE * 2);
    r->cols = gpu_alloc(64 * 9 * PLANE * 2);
    r->acc = gpu_alloc(32 * PLANE * 4);
    r->read = gpu_alloc(BATCH * PLANE * 4);
    r->host_tiles = push(a, measured);
    r->window = gpu_alloc(PLANE * 4);
    r->stack = gpu_alloc(DEPTH * columns);
    r->total = gpu_alloc(columns * 4);
    r->weight = gpu_alloc(columns * 4);
    return r;
}

typedef struct {
    const u8 *stack;
    int height, width, sy, sx;
    i64 corners[MEASURED_TILES][2];
    Scale scale;
} Stats;

void measure_stack(Reader *r, const u8 *stack, int h, int w, Stats *out) {
    *out = (Stats){.stack = stack, .height = h, .width = w};
    measured_tiles(stack, DEPTH, h, w, out->corners, &out->sy, &out->sx);
    out->scale = raw_scale(stack, DEPTH, h, w, out->corners, out->sy, out->sx);
}

void use_stats(Reader *r, const Stats *stats) {
    int sy = stats->sy, sx = stats->sx;
    Scale s = stats->scale;
    pack_measurement(stats->stack, stats->height, stats->width, sy, sx, stats->corners,
        MEASURED_TILES, r->host_tiles);
    gpu_put(r->tiles, r->host_tiles, (i64)MEASURED_TILES * DEPTH * sy * sx);
    f32 table[256];
    u16 half_table[256];
    for (int v = 0; v < 256; v++) {
        f32 low = s.low, high = s.high, x = v;
        f32 clamped = x < low ? low : x > high ? high : x;
        table[v] = (clamped - (f32)s.centre) / (f32)s.spread;
        half_table[v] = half_bits(table[v]);
    }
    gpu_put(r->measure_table, table, sizeof(table));
    gpu_put(r->table, half_table, sizeof(half_table));
    Stem *k = &r->stem_weights;
    f32 scale1[STEM], shift1[STEM], scale2[STEM], shift2[STEM], norm[2 * STEM];
    f32 mean[STEM], variance[STEM];
    gpu_measure(r->tiles, r->measure_table, r->conv1, r->conv2, r->measure_image, r->first_measured,
        r->second_measured, MEASURED_TILES, sy, sx, r->norm, mean, variance, 1, r->cols, ROOM);
    scale_shift(k->norms[0], k->norms[1], mean, variance, scale1, shift1);
    memcpy(norm, scale1, sizeof(scale1));
    memcpy(norm + STEM, shift1, sizeof(shift1));
    gpu_put(r->norm, norm, sizeof(norm));
    gpu_measure(r->tiles, r->measure_table, r->conv1, r->conv2, r->measure_image, r->first_measured,
        r->second_measured, MEASURED_TILES, sy, sx, r->norm, mean, variance, 2, r->cols, ROOM);
    scale_shift(k->norms[2], k->norms[3], mean, variance, scale2, shift2);
    Folded f = fold(k, scale1, shift1, scale2, shift2);
    f32 w1[STEM * 27], w2[STEM * STEM * 27];
    for (int tap = 0; tap < 27; tap++) {
        for (int o = 0; o < STEM; o++) {
            w1[o * 27 + tap] = f.w1[tap * STEM + o];
            for (int i = 0; i < STEM; i++) {
                w2[(o * STEM + i) * 27 + tap] = f.w2[(tap * STEM + i) * STEM + o];
            }
        }
    }
    u16 bits[27 * STEM * STEM];
    f32 *parts[] = {w1, f.b1, w2, f.b2};
    u16 *targets[] = {r->w1, r->b1, r->w2, r->b2};
    i64 sizes[] = {27 * STEM, STEM, 27 * STEM * STEM, STEM};
    for (int p = 0; p < 4; p++) {
        for (i64 i = 0; i < sizes[p]; i++) {
            bits[i] = half_bits(parts[p][i]);
        }
        gpu_put(targets[p], bits, sizes[p] * 2);
    }
}

void fix_stem(Reader *r, const u8 *stack, int h, int w) {
    Stats stats;
    measure_stack(r, stack, h, w, &stats);
    use_stats(r, &stats);
}

u16 *conv_norm(Reader *r, const u16 *x, int batch, int in, int h, int w, const char *prefix,
    int out, int stride, int k, const u16 *residual, int nonlinear, u16 *y) {
    const char *conv = fmt(r->arena, "%s.conv.weight", prefix);
    const u16 *bias = k == 3 ? weight(r, fmt(r->arena, "%s.conv.bias", prefix)) : 0;
    gpu_conv(x, batch, in, h, w, weight(r, conv), out, k, stride, r->c, r->cols, ROOM);
    gpu_norm(r->c, bias, out, batch, (i64)(h / stride) * (w / stride),
        weight(r, fmt(r->arena, "%s.norm.weight", prefix)),
        weight(r, fmt(r->arena, "%s.norm.bias", prefix)), residual, nonlinear, y);
    return y;
}

void body_tiles(Reader *r, int batch, int ty, int tx) {
    u64 mark = r->arena->used;
    int widths[6] = {32, 64, 128, 256, 320, 320}, blocks[6] = {1, 3, 4, 6, 6, 6};
    int h = ty, w = tx, width = 32;
    u16 *x = conv_norm(
        r, r->features, batch, 32, h, w, "shared_encoder.stem.convs.0", 32, 1, 3, 0, 1, r->a);
    for (int s = 0; s < 6; s++) {
        for (int k = 0; k < blocks[s]; k++) {
            const char *block = fmt(r->arena, "shared_encoder.stages.%d.blocks.%d", s, k);
            int stride = s > 0 && k == 0 ? 2 : 1;
            const u16 *residual = x;
            if (stride == 2) {
                gpu_pool(x, width * batch, h, w, r->p);
                residual = r->p;
            }
            if (width != widths[s]) {
                residual = conv_norm(r, residual, batch, width, h / stride, w / stride,
                    fmt(r->arena, "%s.skip.1", block), widths[s], 1, 1, 0, 0, r->q);
            }
            conv_norm(r, x, batch, width, h, w, fmt(r->arena, "%s.conv1", block), widths[s], stride,
                3, 0, 1, r->b);
            h /= stride;
            w /= stride;
            x = conv_norm(r, r->b, batch, widths[s], h, w, fmt(r->arena, "%s.conv2", block),
                widths[s], 1, 3, residual, 1, r->skips[s]);
            width = widths[s];
        }
    }
    for (int s = 0; s < 5; s++) {
        int out = widths[4 - s];
        const char *up = fmt(r->arena, "shared_decoder.transpconvs.%d", s);
        const u16 *kernel = weight(r, fmt(r->arena, "%s.weight", up));
        const u16 *bias = weight(r, fmt(r->arena, "%s.bias", up));
        i64 below = (i64)h * w, above = 4 * below;
        for (int b = 0; b < batch; b++) {
            gpu_up(x + b * width * below, width, h, w, kernel, bias, out,
                r->skips[4 - s] + b * out * above, r->cat + b * 2 * out * above, r->acc);
        }
        h *= 2;
        w *= 2;
        const char *stage = fmt(r->arena, "shared_decoder.stages.%d.convs.0", s);
        x = conv_norm(
            r, r->cat, batch, 2 * out, h, w, stage, out, 1, 3, 0, 1, x == r->a ? r->b : r->a);
        width = out;
    }
    gpu_conv(x, batch, 32, h, w, weight(r, "task_heads.ink.weight"), 1, 1, 1, r->p, r->cols, ROOM);
    gpu_ink(r->p, weight(r, "task_heads.ink.bias"), r->read, (i64)batch * h * w);
    r->arena->used = mark;
}

void ink_read(Reader *r, const u8 *stack, int h, int w, f32 *out) {
    int ty, tx, count = plan_tiles(&r->work, stack, h, w, &ty, &tx);
    i64 n = (i64)h * w, plane = (i64)ty * tx;
    gpu_put(r->stack, r->work.filled, DEPTH * n);
    gpu_put(r->window, r->work.blend, plane * 4);
    gpu_zero(r->total, n * 4);
    gpu_zero(r->weight, n * 4);
    for (int k = 0; k < count; k += BATCH) {
        int batch = count - k < BATCH ? count - k : BATCH;
        for (int b = 0; b < batch; b++) {
            Place p = r->work.places[k + b];
            gpu_stem(r->stack, h, w, p.y, p.x, p.h, p.w, r->table, r->w1, r->b1, r->w2, r->b2,
                r->att, r->attention_bias, r->image, r->first, r->second,
                r->features + b * 2 * STEM * plane, ty, tx, r->cols, ROOM);
        }
        body_tiles(r, batch, ty, tx);
        for (int b = 0; b < batch; b++) {
            Place p = r->work.places[k + b];
            gpu_blend(
                r->read + b * plane, r->window, p.y, p.x, p.h, p.w, tx, r->total, r->weight, w);
        }
    }
    gpu_finish(r->total, r->weight, n, out);
}

i64 ink_queue(Reader *r, const u8 *stack, int h, int w, f32 *out) {
    ink_read(r, stack, h, w, out);
    return r->queued++;
}

int ink_ready(const Reader *r, i64 id) { return id < r->queued; }

void ink_finish_through(Reader *r, i64 id) {}
