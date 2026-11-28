#include <sys/statvfs.h>

#define TILE 512
#define STEP 384
#define DEPTH 17
#define STEM 16
#define MEASURED_SIDE 128
#define MEASURED_TILES 8
#define CANDIDATES 64
#define EVERY_NTH_VALUE 97
#define FILLING_TRIES 400
#define TENSORS 1024

static void ink_hash_file(EVP_MD_CTX *hash, const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    need(fd >= 0, "cannot open %s: %s", path, strerror(errno));
    u8 buffer[65536];
    for (;;) {
        ssize_t size = read(fd, buffer, sizeof(buffer));
        if (size < 0 && errno == EINTR) {
            continue;
        }
        need(size >= 0, "cannot read %s: %s", path, strerror(errno));
        if (size == 0) {
            break;
        }
        need(EVP_DigestUpdate(hash, buffer, size), "cannot hash %s", path);
    }
    close(fd);
}

typedef struct {
    char name[128];
    char key[16];
    i64 offset;
    int ndim;
    i64 shape[8];
    f32 *data;
} Tensor;

static i64 tensor_elements(const Tensor *tensor) {
    i64 count = 1;
    for (int d = 0; d < tensor->ndim; d++)
        count *= tensor->shape[d];
    return count;
}

typedef struct {
    int kind;
    i64 number;
    char *text;
    int count;
    void **items;
} Pickled;

enum { P_MARK = 1, P_INT, P_TEXT, P_TUPLE, P_GLOBAL, P_STORAGE, P_TENSOR, P_NONE, P_DICT };

Tensor *tensor_named(Tensor *tensors, int count, const char *name) {
    const char *bare = strncmp(name, "model.", 6) ? name : name + 6;
    for (int i = 0; i < count; i++) {
        if (!strcmp(tensors[i].name, name) || !strcmp(tensors[i].name, bare)) {
            return tensors + i;
        }
    }
    die("the checkpoint has no %s", name);
    return 0;
}

void record_tensor(Tensor *tensors, int *count, Pickled *key, Pickled *t, const char *path) {
    if (t->kind != P_TENSOR || key->kind != P_TEXT) {
        return;
    }
    Pickled *shape = t->items[2];
    Tensor *out = tensors + (*count)++;
    need(*count <= TENSORS, "%s has too many tensors", path);
    snprintf(out->name, sizeof(out->name), "%s", key->text);
    snprintf(out->key, sizeof(out->key), "%s", ((Pickled *)t->items[0])->text);
    out->offset = ((Pickled *)t->items[1])->number;
    out->data = 0;
    out->ndim = shape->count;
    need(out->ndim <= 8, "%s has a tensor of %d dimensions", path, out->ndim);
    for (int d = 0; d < shape->count; d++) {
        out->shape[d] = ((Pickled *)shape->items[d])->number;
    }
}

int load_tensors(Arena *a, const char *path, Tensor *tensors) {
    int error;
    zip_t *z = zip_open(path, ZIP_RDONLY, &error);
    need(z != 0, "cannot open %s", path);
    const char *pickle_name = 0;
    for (i64 i = 0; i < zip_get_num_entries(z, 0); i++) {
        const char *entry = zip_get_name(z, i, 0);
        if (strlen(entry) > 9 && !strcmp(entry + strlen(entry) - 9, "/data.pkl")) {
            pickle_name = entry;
        }
    }
    need(pickle_name != 0, "%s is not a torch checkpoint", path);
    int prefix = strlen(pickle_name) - 9;
    zip_stat_t st;
    zip_stat(z, pickle_name, 0, &st);
    u8 *p = push(a, st.size);
    zip_file_t *f = zip_fopen(z, pickle_name, 0);
    need(zip_fread(f, p, st.size) == (i64)st.size, "cannot read %s", pickle_name);
    zip_fclose(f);
    Pickled **stack = push(a, 4096 * sizeof(void *));
    Pickled **memo = pushz(a, 8192 * sizeof(void *));
    int top = 0, count = 0;
    u8 *end = p + st.size;
    while (p < end) {
        u8 op = *p++;
        Pickled *v = 0;
        int n = 0;
        if (op == 0x80) {
            p++;
        } else if (op == '}') {
            v = pushz(a, sizeof(Pickled));
            v->kind = P_DICT;
        } else if (op == '(') {
            v = pushz(a, sizeof(Pickled));
            v->kind = P_MARK;
        } else if (op == 'X') {
            memcpy(&n, p, 4);
            v = pushz(a, sizeof(Pickled));
            v->kind = P_TEXT;
            v->text = fmt(a, "%.*s", n, p + 4);
            p += 4 + n;
        } else if (op == 'c') {
            u8 *nl = memchr(p, '\n', end - p);
            u8 *nl2 = memchr(nl + 1, '\n', end - nl - 1);
            v = pushz(a, sizeof(Pickled));
            v->kind = P_GLOBAL;
            v->text = fmt(a, "%.*s.%.*s", (int)(nl - p), p, (int)(nl2 - nl - 1), nl + 1);
            p = nl2 + 1;
        } else if (op == 'K' || op == 'M' || op == 'J') {
            int size = op == 'K' ? 1 : op == 'M' ? 2 : 4;
            i32 number = 0;
            memcpy(&number, p, size);
            p += size;
            v = pushz(a, sizeof(Pickled));
            v->kind = P_INT;
            v->number = number;
        } else if (op == 'q' || op == 'r') {
            u32 at = 0;
            memcpy(&at, p, op == 'q' ? 1 : 4);
            p += op == 'q' ? 1 : 4;
            memo[at] = stack[top - 1];
        } else if (op == 'h' || op == 'j') {
            u32 at = 0;
            memcpy(&at, p, op == 'h' ? 1 : 4);
            p += op == 'h' ? 1 : 4;
            v = memo[at];
        } else if (op == 't' || op == ')' || op == 0x85 || op == 0x86 || op == 0x87) {
            int first = top;
            if (op == 't') {
                while (stack[first - 1]->kind != P_MARK) {
                    first--;
                }
            } else {
                first = top - (op == ')' ? 0 : op - 0x84);
            }
            v = pushz(a, sizeof(Pickled));
            v->kind = P_TUPLE;
            v->count = top - first;
            v->items = push(a, v->count * sizeof(void *) + 1);
            memcpy(v->items, stack + first, v->count * sizeof(void *));
            top = op == 't' ? first - 1 : first;
        } else if (op == 'Q') {
            Pickled *id = stack[--top];
            v = pushz(a, sizeof(Pickled));
            v->kind = P_STORAGE;
            v->text = ((Pickled *)id->items[2])->text;
        } else if (op == 0x89 || op == 0x88 || op == 'N') {
            v = pushz(a, sizeof(Pickled));
            v->kind = P_NONE;
        } else if (op == 'R') {
            Pickled *args = stack[--top], *call = stack[--top];
            v = pushz(a, sizeof(Pickled));
            v->kind = P_NONE;
            if (!strcmp(call->text, "torch._utils._rebuild_tensor_v2")) {
                v->kind = P_TENSOR;
                v->count = args->count;
                v->items = args->items;
            }
        } else if (op == 'u' || op == 'e') {
            int first = top;
            while (stack[first - 1]->kind != P_MARK) {
                first--;
            }
            for (int i = first; op == 'u' && i < top; i += 2) {
                record_tensor(tensors, &count, stack[i], stack[i + 1], path);
            }
            top = first - 1;
        } else if (op == 's') {
            record_tensor(tensors, &count, stack[top - 2], stack[top - 1], path);
            top -= 2;
        } else if (op == 'a' || op == 'b') {
            top--;
        } else if (op == ']') {
            v = pushz(a, sizeof(Pickled));
            v->kind = P_TUPLE;
        } else if (op == 'G') {
            p += 8;
            v = pushz(a, sizeof(Pickled));
            v->kind = P_INT;
        } else if (op == '.') {
            break;
        } else {
            die("%s: pickle opcode 0x%02x is not read", path, op);
        }
        if (v) {
            stack[top++] = v;
        }
    }
    for (int i = 0; i < count; i++) {
        Tensor *t = tensors + i;
        for (int j = 0; j < i && !t->data; j++) {
            if (!strcmp(tensors[j].key, t->key)) {
                t->data = tensors[j].data - tensors[j].offset + t->offset;
            }
        }
        if (t->data) {
            continue;
        }
        char *member = fmt(a, "%.*s/data/%s", prefix, pickle_name, t->key);
        zip_stat(z, member, 0, &st);
        u8 *storage = push(a, st.size);
        f = zip_fopen(z, member, 0);
        need(zip_fread(f, storage, st.size) == (i64)st.size, "cannot read %s", member);
        zip_fclose(f);
        t->data = (f32 *)storage + t->offset;
    }
    zip_close(z);
    return count;
}

int by_coverage(const void *a, const void *b) {
    const i64 *x = a, *y = b;
    return x[0] != y[0] ? (x[0] < y[0] ? 1 : -1) : (x[3] > y[3]) - (x[3] < y[3]);
}

int measured_tiles(const u8 *stack, int depth, int h, int w, i64 corners[MEASURED_TILES][2],
    int *side_y, int *side_x) {
    int sy = h < MEASURED_SIDE ? h : MEASURED_SIDE;
    int sx = w < MEASURED_SIDE ? w : MEASURED_SIDE;
    *side_y = sy;
    *side_x = sx;
    if (h == sy && w == sx) {
        memset(corners, 0, MEASURED_TILES * sizeof(*corners));
        return MEASURED_TILES;
    }
    const u8 *middle = stack + (i64)(depth / 2) * h * w;
    Pcg64 r = pcg64(h * 31 + w);
    i64 spots[CANDIDATES][4];
    for (int i = 0; i < CANDIDATES; i++) {
        spots[i][1] = draw(&r, h - sy + 1);
    }
    for (int i = 0; i < CANDIDATES; i++) {
        spots[i][2] = draw(&r, w - sx + 1);
        spots[i][3] = i;
        int previous = 0;
        while (previous < i &&
            (spots[previous][1] != spots[i][1] || spots[previous][2] != spots[i][2])) {
            previous++;
        }
        if (previous < i) {
            spots[i][0] = spots[previous][0];
            continue;
        }
        int covered = 0;
        for (int y = 0; y < sy; y++) {
            const u8 *row = middle + (spots[i][1] + y) * w + spots[i][2];
            for (int x = 0; x < sx; x++) {
                covered += row[x] != 0;
            }
        }
        spots[i][0] = covered;
    }
    qsort(spots, CANDIDATES, sizeof(*spots), by_coverage);
    for (int i = 0; i < MEASURED_TILES; i++) {
        corners[i][0] = spots[i][1];
        corners[i][1] = spots[i][2];
    }
    return MEASURED_TILES;
}

static int intensity_at(const i64 *counts, i64 rank) {
    int value = 0;
    while (rank >= counts[value]) {
        rank -= counts[value++];
    }
    return value;
}

static f64 intensity_quantile(const i64 *counts, i64 n, f64 q) {
    f64 at = q * (n - 1);
    i64 below = at;
    f64 lo = intensity_at(counts, below);
    if (below + 1 == n) {
        return lo;
    }
    f64 hi = intensity_at(counts, below + 1);
    f64 t = at - below;
    return t >= 0.5 ? hi - (hi - lo) * (1 - t) : lo + (hi - lo) * t;
}

static f64 clipped_intensity(int value, f64 low, f64 high) {
    return value < low ? low : value > high ? high : value;
}

typedef struct {
    f64 low, high, centre, spread;
} Scale;

Scale raw_scale(
    const u8 *stack, int depth, int h, int w, i64 corners[MEASURED_TILES][2], int sy, int sx) {
    i64 n = 0, counts[256] = {0};
    int frequency[MEASURED_TILES] = {0};
    for (int t = 0; t < MEASURED_TILES; t++) {
        int first = 0;
        while (corners[first][0] != corners[t][0] || corners[first][1] != corners[t][1]) {
            first++;
        }
        frequency[first]++;
    }
    for (int t = 0; t < MEASURED_TILES; t++) {
        if (!frequency[t]) {
            continue;
        }
        i64 seen = 0;
        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < sy; y++) {
                const u8 *row = stack + ((i64)z * h + corners[t][0] + y) * w + corners[t][1];
                for (int x = 0; x < sx; x++) {
                    if (row[x] && seen++ % EVERY_NTH_VALUE == 0) {
                        counts[row[x]] += frequency[t];
                        n += frequency[t];
                    }
                }
            }
        }
    }
    if (!n) {
        return (Scale){0, 255, 0, 1};
    }
    Scale s = {
        .low = intensity_quantile(counts, n, 0.01), .high = intensity_quantile(counts, n, 0.99)};
    i64 lower = (n - 1) / 2, upper = n / 2;
    f64 low = clipped_intensity(intensity_at(counts, lower), s.low, s.high);
    f64 high = clipped_intensity(intensity_at(counts, upper), s.low, s.high);
    s.centre = (low + high) / 2;
    int left = s.centre, right = left + 1;
    i64 seen = 0;
    f64 first = 0, last = 0;
    while (seen <= upper) {
        f64 a = left >= 0 ? s.centre - clipped_intensity(left, s.low, s.high) : INFINITY;
        f64 b = right < 256 ? clipped_intensity(right, s.low, s.high) - s.centre : INFINITY;
        int from_left = a <= b;
        last = from_left ? a : b;
        i64 next = seen + (from_left ? counts[left--] : counts[right++]);
        if (seen <= lower && next > lower) {
            first = last;
        }
        seen = next;
    }
    s.spread = 1.4826 * ((first + last) / 2);
    s.spread = s.spread > 1 ? s.spread : 1;
    return s;
}

int wrap(int at, int length) { return at < 0 ? at + length : at >= length ? at - length : at; }

void fill_stack(
    const u8 *stack, int depth, int h, int w, const u8 *keep, u8 *out, i32 *source, i32 *spots) {
    i64 size = (i64)h * w, valid = 0;
    for (i64 i = 0; i < size; i++) {
        valid += keep[i];
    }
    i64 remaining = size - valid;
    if (!valid || !remaining) {
        memcpy(out, stack, depth * size);
        return;
    }
    Pcg64 r = pcg64(h * 31 + w);
    int sparse = valid * 4 < size;
    i64 count = 0;
    for (i64 i = 0; i < size; i++) {
        source[i] = sparse && !keep[i] ? -1 : !sparse && !keep[i] ? 0 : i;
        if (keep[i] == sparse) {
            spots[count++] = i;
        }
    }
    for (int k = 0; k < FILLING_TRIES && remaining; k++) {
        int dy = draw(&r, h), dx = draw(&r, w), fy = draw(&r, 2), fx = draw(&r, 2);
        i64 left = 0;
        for (i64 i = 0; i < count; i++) {
            int row = spots[i] / w, col = spots[i] % w;
            if (sparse) {
                int ty = wrap((fy ? h - 1 - row : row) + dy, h);
                int tx = wrap((fx ? w - 1 - col : col) + dx, w);
                if (source[ty * w + tx] < 0) {
                    source[ty * w + tx] = spots[i];
                    remaining--;
                }
            } else {
                int from_y = wrap(row - dy, h), from_x = wrap(col - dx, w);
                i32 from = (fy ? h - 1 - from_y : from_y) * w + (fx ? w - 1 - from_x : from_x);
                if (keep[from]) {
                    source[spots[i]] = from;
                } else {
                    spots[left++] = spots[i];
                }
            }
        }
        if (!sparse) {
            count = left;
            remaining = left;
        }
    }
    if (valid >= size / 2) {
        i64 holes = 0;
        for (i64 i = 0; i < size; i++) {
            if (!keep[i]) {
                spots[holes++] = i;
            }
        }
        memcpy(out, stack, depth * size);
        for (int z = 0; z < depth; z++) {
            const u8 *layer = stack + z * size;
            u8 *filled = out + z * size;
            for (i64 i = 0; i < holes; i++) {
                i32 at = spots[i];
                filled[at] = layer[source[at]];
            }
        }
        return;
    }
    for (int z = 0; z < depth; z++) {
        const u8 *layer = stack + z * size;
        for (i64 i = 0; i < size; i++) {
            out[z * size + i] = layer[source[i] < 0 ? 0 : source[i]];
        }
    }
}

int tile_starts(int length, int tile, int *starts) {
    int last = length > tile ? length - tile : 0, n = 0;
    for (int s = 0; s <= last; s += STEP) {
        starts[n++] = s;
    }
    if (starts[n - 1] != last) {
        starts[n++] = last;
    }
    return n;
}

f64 hann(int k, int length) {
    f64 m = length + 2;
    return 0.5 + 0.5 * cos(M_PI * (2 * (k + 1) + 1 - m) / (m - 1));
}

void hann_blend(int ty, int tx, f32 *blend) {
    f64 across[TILE];
    for (int x = 0; x < tx; x++) {
        across[x] = hann(x, tx);
    }
    for (int y = 0; y < ty; y++) {
        f64 down = hann(y, ty);
        for (int x = 0; x < tx; x++) {
            blend[y * tx + x] = down * across[x];
        }
    }
}

typedef struct {
    f32 conv1[STEM * 27], conv2[STEM * STEM * 27], norms[4][STEM], attention[STEM];
    f32 attention_bias;
} Stem;

typedef struct {
    f32 w1[27 * STEM], b1[STEM], w2[27 * STEM * STEM], b2[STEM];
} Folded;

typedef struct {
    int y, x, h, w;
} Place;

typedef struct {
    u8 *keep, *filled;
    i32 *source, *spots;
    f32 blend[TILE * TILE];
    Place places[4096];
} Reading;

Stem stem_of(Arena *a, Tensor *tensors, int count) {
    Stem s;
    const char *names[] = {"features.0.weight", "features.3.weight", "features.1.weight",
        "features.1.bias", "features.4.weight", "features.4.bias", "attention_logits.weight",
        "attention_logits.bias"};
    f32 *to[] = {s.conv1, s.conv2, s.norms[0], s.norms[1], s.norms[2], s.norms[3], s.attention,
        &s.attention_bias};
    i64 sizes[] = {STEM * 27, STEM * STEM * 27, STEM, STEM, STEM, STEM, STEM, 1};
    for (int k = 0; k < 8; k++) {
        Tensor *t = tensor_named(tensors, count, fmt(a, "model.depth_fusion.%s", names[k]));
        memcpy(to[k], t->data, sizes[k] * sizeof(f32));
    }
    return s;
}

void channel_moments(const f32 *x, i64 n, f32 *mean, f32 *variance) {
    for (int c = 0; c < STEM; c++) {
        f64 sum = 0, squares = 0;
        for (i64 i = 0; i < n; i++) {
            sum += x[i * STEM + c];
        }
        f64 m = sum / n;
        for (i64 i = 0; i < n; i++) {
            f64 d = x[i * STEM + c] - m;
            squares += d * d;
        }
        mean[c] = m;
        variance[c] = squares / n;
    }
}

void scale_shift(const f32 *weight, const f32 *bias, const f32 *mean, const f32 *variance,
    f32 *scale, f32 *shift) {
    for (int c = 0; c < STEM; c++) {
        scale[c] = weight[c] * (1 / sqrtf(variance[c] + 1e-5f));
        shift[c] = bias[c] - mean[c] * scale[c];
    }
}

Folded fold(
    const Stem *s, const f32 *scale1, const f32 *shift1, const f32 *scale2, const f32 *shift2) {
    Folded f;
    for (int tap = 0; tap < 27; tap++) {
        for (int o = 0; o < STEM; o++) {
            f.w1[tap * STEM + o] = s->conv1[o * 27 + tap] * scale1[o];
            for (int i = 0; i < STEM; i++) {
                f32 weight = s->conv2[(o * STEM + i) * 27 + tap];
                f.w2[(tap * STEM + i) * STEM + o] = weight * scale2[o];
            }
        }
    }
    memcpy(f.b1, shift1, sizeof(f.b1));
    memcpy(f.b2, shift2, sizeof(f.b2));
    return f;
}

void reading_init(Reading *k, Arena *a, i64 columns) {
    k->keep = push(a, columns);
    k->filled = push(a, DEPTH * columns);
    k->source = push(a, columns * 4);
    k->spots = push(a, columns * 4);
}

void pack_measurement(
    const u8 *stack, int h, int w, int sy, int sx, const i64 (*corners)[2], int count, u8 *tiles) {
    for (int t = 0; t < count; t++) {
        for (int z = 0; z < DEPTH; z++) {
            for (int y = 0; y < sy; y++) {
                memcpy(tiles + (((i64)t * DEPTH + z) * sy + y) * sx,
                    stack + ((i64)z * h + corners[t][0] + y) * w + corners[t][1], sx);
            }
        }
    }
}

int plan_tiles(Reading *k, const u8 *stack, int h, int w, int *ty, int *tx) {
    i64 columns = (i64)h * w;
    memset(k->keep, 0, columns);
    for (int z = 0; z < DEPTH; z++) {
        for (i64 i = 0; i < columns; i++) {
            k->keep[i] |= stack[z * columns + i] != 0;
        }
    }
    fill_stack(stack, DEPTH, h, w, k->keep, k->filled, k->source, k->spots);
    *ty = (h + 31) / 32 * 32 < TILE ? (h + 31) / 32 * 32 : TILE;
    *tx = (w + 31) / 32 * 32 < TILE ? (w + 31) / 32 * 32 : TILE;
    hann_blend(*ty, *tx, k->blend);
    int ys[1024], xs[1024], count = 0;
    int ny = tile_starts(h, *ty, ys), nx = tile_starts(w, *tx, xs);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            Place p = {
                ys[j], xs[i], h - ys[j] < *ty ? h - ys[j] : *ty, w - xs[i] < *tx ? w - xs[i] : *tx};
            int covered = 0;
            for (int y = 0; y < p.h && !covered; y++) {
                covered = memchr(k->keep + (i64)(p.y + y) * w + p.x, 1, p.w) != 0;
            }
            if (covered) {
                k->places[count++] = p;
            }
        }
    }
    return count;
}

void cut_tile(Reading *k, Place p, int h, int w, int ty, int tx, u8 *block) {
    memset(block, 0, DEPTH * ty * tx);
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < p.h; y++) {
            const u8 *row = k->filled + (z * (i64)h + p.y + y) * w + p.x;
            memcpy(block + ((i64)z * ty + y) * tx, row, p.w);
        }
    }
}

void blend_add(const f32 *blend, Place p, int tx, const f32 *read, f32 *total, int width) {
    for (int y = 0; y < p.h; y++) {
        for (int x = 0; x < p.w; x++) {
            f32 part = read[y * tx + x] * blend[y * tx + x];
            total[(i64)(p.y + y) * width + p.x + x] += part;
        }
    }
}

void blend_weights(
    const f32 *blend, const Place *places, int count, int tx, f32 *weight, int h, int w) {
    memset(weight, 0, (i64)h * w * sizeof(f32));
    for (int k = 0; k < count; k++) {
        Place p = places[k];
        for (int y = 0; y < p.h; y++) {
            for (int x = 0; x < p.w; x++) {
                weight[(i64)(p.y + y) * w + p.x + x] += blend[y * tx + x];
            }
        }
    }
}

void blend_finish(f32 *total, const f32 *weight, i64 n) {
    for (i64 i = 0; i < n; i++) {
        total[i] /= weight[i] > 1e-6f ? weight[i] : 1e-6f;
    }
}

#include LANE_READER
#include "model/studio/reader.h"

#define FACE_LAYER 10
#define PAST_FACE 2
#define LOOK_SIDE 512
#define LOOKS 2
#define LOOKED_PIXELS (1 << 20)
#define NO_INK 0.25f
#define SURE 0.65f
#define KEEP_FREE (40 * GB)
#define FETCH_FREE (25 * GB)
#define STOCK_MODEL "reader-v3"

typedef struct {
    const char *name, *title, *file, *url, *sha256, *backend, *author, *source, *license;
} Published;

#define PUBLISHED_COUNT 4

const Published PUBLISHED[PUBLISHED_COUNT] = {
    {STOCK_MODEL, "Reader v3", "reader-v3.pth",
        "https://huggingface.co/domenicor046/reader-v3/resolve/"
        "aee1d64bced4eb35546348f3d276b9e2a0b99c46/reader-v3.pth",
        "ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a", "reader",
        "Domenico Russo", "https://huggingface.co/domenicor046/reader-v3", "MIT"},
    {"reader-pu-bhand", "Reader-PU", "tile-reader.pth",
        "https://huggingface.co/domenicor046/scribble-pu/resolve/"
        "5990815f6704d80bf65ddb0f9bd0d8d02fd95246/reader-pu-bhand.pth",
        "9abc43e04af97b1f575efd49cdf66230755f10f3b99618a3ef598a958cb743cc", "reader",
        "Domenico Russo", "https://huggingface.co/domenicor046/scribble-pu", "MIT"},
    {"studio", "Studio", "studio-softloss-20261023-ema.safetensors",
        "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/"
        "studio-softloss-20261023/studio-softloss-20261023-ema.safetensors",
        "96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90", "studio",
        "Priyank Aranke", "https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt",
        "CC BY-NC 4.0"},
    {"studio-jeff", "Studio, Jeff Chen post-train", "studio_00500.safetensors",
        "https://drive.usercontent.google.com/"
        "download?id=1Ko84Y-1I17VL4VL5f5iwUf1LXZn_ag2d&export=download&confirm=t",
        "474d0f4be9db7bd18a4a8c8a1a12a0252a72b31dea5bb2edce5189366ec8f2b2", "studio",
        "Jeff Chen; based on Studio by Priyank Aranke",
        "https://drive.google.com/file/d/1Ko84Y-1I17VL4VL5f5iwUf1LXZn_ag2d/view", "unspecified"},
};

const Published *published_model(const char *name) {
    for (int k = 0; k < PUBLISHED_COUNT; k++) {
        if (!strcmp(PUBLISHED[k].name, name)) {
            return PUBLISHED + k;
        }
    }
    return NULL;
}

typedef struct {
    char name[64], title[128], weights[PATH_MAX], backend[32];
    char faces[64], suffix[72];
} Model;

Model chosen_model(Arena *a) {
    Model m = {0};
    const char *name = env("HERCULEX_INK_MODEL", STOCK_MODEL);
    const Published *published = published_model(name);
    snprintf(m.name, sizeof(m.name), "%s", name);
    if (published) {
        snprintf(m.title, sizeof(m.title), "%s", published->title);
        snprintf(m.backend, sizeof(m.backend), "%s", published->backend);
        const char *state = fmt(a, "%s/weights/ink/reader-v3/tile-reader.pth", LIBRARY);
        const char *path =
            fmt(a, "%s/weights/ink/%s/%s", LIBRARY, published->name, published->file);
        snprintf(m.weights, sizeof(m.weights), "%s",
            !strcmp(name, STOCK_MODEL) && exists(state) ? state : path);
    } else {
        json_object *listed = json_read(fmt(a, "%s/weights/ink/models.json", LIBRARY)), *entry = 0;
        need(listed && json_object_object_get_ex(listed, name, &entry),
            "HERCULEX_INK_MODEL=%s is not listed in %s", name,
            fmt(a, "%s/weights/ink/models.json", LIBRARY));
        snprintf(m.title, sizeof(m.title), "%s", jstr(entry, "title"));
        snprintf(m.weights, sizeof(m.weights), "%s/weights/ink/%s", LIBRARY, jstr(entry, "file"));
        json_object *backend = json_object_object_get(entry, "backend");
        snprintf(m.backend, sizeof(m.backend), "%s",
            backend ? json_object_get_string(backend) : "reader");
        json_object_put(listed);
    }
    need(exists(m.weights), "%s: no checkpoint at %s; see ./run.sh models", m.name, m.weights);
    const char *production = !strcmp(name, STOCK_MODEL) ? "level-0" : fmt(a, "level-0-%s", name);
    snprintf(m.faces, sizeof(m.faces), "%s", env("HERCULEX_FACES", production));
    snprintf(
        m.suffix, sizeof(m.suffix), "%s", strcmp(name, STOCK_MODEL) ? fmt(a, "-%s", name) : "");
    return m;
}

void model_digest(Arena *a, const Model *m, char *out) {
    EVP_MD_CTX *c = EVP_MD_CTX_new();
    need(c && EVP_DigestInit_ex(c, EVP_sha256(), NULL), "cannot start ink model hash");
    ink_hash_file(c, m->weights);
    const char *reader_sources[] = {"ink/ink.h", "sampling/sampling.h", LANE_READER, lane_kernels};
    const char *studio_sources[] = {"ink/ink.h", "sampling/sampling.h", "ink/model/studio/model.py",
        "ink/model/studio/worker.py"};
    const char **sources = !strcmp(m->backend, "studio") ? studio_sources : reader_sources;
    for (int k = 0; k < 4; k++) {
        ink_hash_file(c, fmt(a, "%s/%s", HOME, sources[k]));
    }
    u8 d[32];
    need(EVP_DigestUpdate(c, SAMPLING, strlen(SAMPLING)) && EVP_DigestFinal_ex(c, d, NULL),
        "cannot finish ink model hash");
    EVP_MD_CTX_free(c);
    for (int i = 0; i < 6; i++) {
        snprintf(out + 2 * i, 3, "%02x", d[i]);
    }
}

int face_layer_of(const u8 *cube, int layers, int h, int w, int reversed) {
    int ny = (h + 7) / 8, nx = (w + 7) / 8, k = ny * nx;
    f32 *sample = malloc((i64)layers * k * 4), *row = malloc(k * 4), profile[64];
    u8 *good = malloc(k);
    i64 n = (i64)h * w;
    int any = 0;
    for (int j = 0; j < k; j++) {
        good[j] = 1;
    }
    for (int l = 0; l < layers; l++) {
        const u8 *layer = cube + (i64)(reversed ? layers - 1 - l : l) * n;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                f32 v = layer[(i64)y * 8 * w + x * 8];
                sample[(i64)l * k + y * nx + x] = v;
                good[y * nx + x] &= v > 0;
            }
        }
    }
    for (int j = 0; j < k; j++) {
        any |= good[j];
    }
    for (int l = 0; l < layers; l++) {
        int m = 0;
        for (int j = 0; j < k; j++) {
            if (good[j] || !any) {
                row[m++] = sample[(i64)l * k + j];
            }
        }
        profile[l] = mean_of(row, m);
    }
    int middle = layers / 2, best = middle - 2;
    f32 least = profile[best + 1] - profile[best];
    for (int i = middle - 1; i < middle + 8 && i < layers - 1; i++) {
        f32 fall = profile[i + 1] - profile[i];
        if (fall < least) {
            least = fall;
            best = i;
        }
    }
    free(sample);
    free(row);
    free(good);
    int face = best - FACE_LAYER;
    face = face < 2 ? 2 : face;
    return face < layers - DEPTH - 2 ? face : layers - DEPTH - 2;
}

void window_of(Stack s, int reversed, u8 *out) {
    int first = face_layer_of(s.cube, s.layers, s.height, s.width, reversed) + PAST_FACE;
    first = first < s.layers - DEPTH ? first : s.layers - DEPTH;
    i64 n = (i64)s.height * s.width;
    for (int l = 0; l < DEPTH; l++) {
        int at = first + l;
        memcpy(out + l * n, s.cube + (i64)(reversed ? s.layers - 1 - at : at) * n, n);
    }
}

int by_mass(const void *a, const void *b) {
    const i64 *x = a, *y = b;
    return x[0] != y[0] ? (x[0] < y[0] ? 1 : -1) : (x[1] > y[1]) - (x[1] < y[1]);
}

int fullest(const u8 *sheet, int h, int w, int spots[LOOKS][2]) {
    int side = LOOK_SIDE < h ? LOOK_SIDE : h;
    side = side < w ? side : w;
    int step = side / 2, nr = (h - side) / step + 1, nc = (w - side) / step + 1, chosen = 0;
    i64(*mass)[2] = malloc((i64)nr * nc * sizeof(*mass));
    for (int r = 0; r < nr; r++) {
        for (int c = 0; c < nc; c++) {
            i64 total = 0;
            for (int y = r * step; y < r * step + side; y++) {
                for (int x = c * step; x < c * step + side; x++) {
                    total += sheet[(i64)y * w + x] > 0;
                }
            }
            mass[r * nc + c][0] = total;
            mass[r * nc + c][1] = r * nc + c;
        }
    }
    qsort(mass, (i64)nr * nc, sizeof(*mass), by_mass);
    for (i64 k = 0; k < (i64)nr * nc && chosen < LOOKS; k++) {
        int y = mass[k][1] / nc * step, x = mass[k][1] % nc * step, apart = 1;
        for (int j = 0; j < chosen; j++) {
            apart &= abs(y - spots[j][0]) >= side || abs(x - spots[j][1]) >= side;
        }
        if (apart) {
            spots[chosen][0] = y;
            spots[chosen++][1] = x;
        }
    }
    free(mass);
    return side;
}

void fft_roots(int n, f64 *cosines, f64 *sines) {
    for (int i = 0; i < n; i++) {
        f64 angle = -2 * M_PI * i / n;
        cosines[i] = cos(angle);
        sines[i] = sin(angle);
    }
}

void fft(f64 *re, f64 *im, int n, int stride, int inverse, f64 *scratch, const f64 *cosines,
    const f64 *sines) {
    if (n & (n - 1)) {
        for (int k = 0; k < n; k++) {
            f64 sr = 0, si = 0;
            for (int t = 0; t < n; t++) {
                int at = (i64)k * t % n;
                f64 wr = cosines[at], wi = inverse ? -sines[at] : sines[at];
                sr += re[t * stride] * wr - im[t * stride] * wi;
                si += re[t * stride] * wi + im[t * stride] * wr;
            }
            scratch[2 * k] = sr;
            scratch[2 * k + 1] = si;
        }
        for (int k = 0; k < n; k++) {
            re[k * stride] = scratch[2 * k];
            im[k * stride] = scratch[2 * k + 1];
        }
        return;
    }
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            f64 tr = re[i * stride], ti = im[i * stride];
            re[i * stride] = re[j * stride];
            im[i * stride] = im[j * stride];
            re[j * stride] = tr;
            im[j * stride] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                int at = k * step;
                f64 wr = cosines[at], wi = inverse ? -sines[at] : sines[at];
                i64 a = (i64)(i + k) * stride, b = (i64)(i + k + len / 2) * stride;
                f64 xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
    }
}

typedef struct {
    int side;
    f64 *scratch, *cosines, *sines;
    void *plan;
} PhaseFFT;

static void phase_fft_rows(const PhaseFFT *transform, f64 *re, f64 *im, int rows, int inverse) {
    int side = transform->side;
#ifdef __APPLE__
    if (transform->plan) {
        apple_fft_rows(transform->plan, re, im, side, rows, inverse);
        return;
    }
#endif
    for (int y = 0; y < rows; y++) {
        i64 at = (i64)y * side;
        fft(re + at, im + at, side, 1, inverse, transform->scratch, transform->cosines,
            transform->sines);
    }
}

static void transpose_square(f64 *re, f64 *im, int side) {
    enum { BLOCK_SIDE = 16 };
    for (int a = 0; a < side; a += BLOCK_SIDE) {
        int a_end = a + BLOCK_SIDE < side ? a + BLOCK_SIDE : side;
        for (int b = a; b < side; b += BLOCK_SIDE) {
            int b_end = b + BLOCK_SIDE < side ? b + BLOCK_SIDE : side;
            for (int y = a; y < a_end; y++) {
                int first = b > y ? b : y + 1;
                for (int x = first; x < b_end; x++) {
                    i64 i = (i64)y * side + x, j = (i64)x * side + y;
                    f64 r = re[i], q = im[i];
                    re[i] = re[j];
                    im[i] = im[j];
                    re[j] = r;
                    im[j] = q;
                }
            }
        }
    }
}

void scrambled(const u8 *window, int side, Pcg64 *rng, u8 *out) {
    i64 n = (i64)side * side;
    int half = side / 2 + 1;
    f64 *phase_re = malloc((i64)side * half * 8), *phase_im = malloc((i64)side * half * 8);
    for (i64 i = 0; i < (i64)side * half; i++) {
        f64 angle = 2 * M_PI * pcg64_double(rng);
        phase_re[i] = cos(angle);
        phase_im[i] = sin(angle);
    }
    const u8 *middle = window + DEPTH / 2 * n;
    f32 *image = malloc(n * 4), *values = malloc(n * 4);
    f64 *re = malloc(n * 8), *im = malloc(n * 8), *scratch = malloc(4 * side * 8);
    PhaseFFT transform = {.side = side,
        .scratch = scratch,
        .cosines = scratch + 2 * side,
        .sines = scratch + 3 * side};
#ifdef __APPLE__
    transform.plan = apple_fft_open(side);
#endif
    if (!transform.plan) {
        fft_roots(side, transform.cosines, transform.sines);
    }
    for (int l = 0; l < DEPTH; l++) {
        const u8 *layer = window + l * n;
        i64 count = 0;
        for (i64 i = 0; i < n; i++) {
            if (middle[i] > 0) {
                values[count++] = layer[i];
            }
        }
        f32 mean = count ? mean_of(values, count) : 0;
        for (i64 i = 0; i < n; i++) {
            f32 v = layer[i];
            image[i] = middle[i] > 0 ? v - mean : 0;
            re[i] = image[i];
            im[i] = 0;
        }
        phase_fft_rows(&transform, re, im, side, 0);
        transpose_square(re, im, side);
        phase_fft_rows(&transform, re, im, half, 0);
        for (int x = 0; x < half; x++) {
            for (int y = 0; y < side; y++) {
                i64 i = (i64)x * side + y, p = (i64)y * half + x;
                f64 r = re[i] * phase_re[p] - im[i] * phase_im[p];
                im[i] = re[i] * phase_im[p] + im[i] * phase_re[p];
                re[i] = r;
            }
        }
        phase_fft_rows(&transform, re, im, half, 1);
        transpose_square(re, im, side);
        for (int y = 0; y < side; y++) {
            f64 *r = re + (i64)y * side, *q = im + (i64)y * side;
            q[0] = 0;
            if (side % 2 == 0) {
                q[side / 2] = 0;
            }
            for (int x = half; x < side; x++) {
                r[x] = r[side - x];
                q[x] = -q[side - x];
            }
        }
        phase_fft_rows(&transform, re, im, side, 1);
        for (i64 i = 0; i < n; i++) {
            f64 v = re[i] / (f64)n + mean;
            out[l * n + i] = v < 1 ? 1 : v > 255 ? 255 : v;
        }
    }
#ifdef __APPLE__
    if (transform.plan) {
        apple_fft_close(transform.plan);
    }
#endif
    free(phase_re);
    free(phase_im);
    free(image);
    free(values);
    free(re);
    free(im);
    free(scratch);
}

f64 sure_of(const f32 *read, i64 n) {
    i64 inked = 0, sure = 0;
    for (i64 i = 0; i < n; i++) {
        inked += read[i] > 0;
        sure += read[i] > 0 && read[i] >= SURE;
    }
    return inked ? (f64)sure / inked : 0;
}

typedef struct {
    char folder[PATH_MAX], out[PATH_MAX], unit[64];
    int piece, turn;
    f64 area;
} Job;

char *beside(Arena *a, const char *out, const char *suffix) {
    return fmt(a, "%.*s%s", (int)(strlen(out) - 4), out, suffix);
}

int newer(const char *path, const char *folder, Arena *a) {
    return exists(path) && mtime_ns(path) >= mtime_ns(fmt(a, "%s/meta.json", folder));
}

json_object *side_record(const char *path) {
    json_object *side = exists(path) ? json_object_from_file(path) : 0;
    if (side && !json_object_is_type(side, json_type_object)) {
        json_object_put(side);
        return 0;
    }
    return side;
}

typedef struct {
    Model model;
    char digest[16];
    char *same[256];
    int sames;
    char windows[PATH_MAX], sample[64], scan[64], base[PATH_MAX];
    f64 *line;
    int rows;
    Reader *reader;
    StudioReader *studio;
} Stage;

int read_by(Stage *g, json_object *side, int turn) {
    json_object *model = 0, *recorded = 0;
    int t = 1;
    if (json_object_object_get_ex(side, "turn", &recorded)) {
        t = json_object_get_int(recorded);
    }
    if (t != turn || !json_object_object_get_ex(side, "model", &model)) {
        return 0;
    }
    const char *name = json_object_get_string(model);
    int known = !strcmp(name, g->digest);
    for (int k = 0; k < g->sames; k++) {
        known |= !strcmp(name, g->same[k]);
    }
    return known;
}

int ink_is_current(Stage *g, Job *j, Arena *a) {
    const char *record = beside(a, j->out, "-taught-side.json");
    if (!newer(record, j->folder, a)) {
        return 0;
    }
    json_object *side = side_record(record);
    int current = side && read_by(g, side, j->turn) && exists(beside(a, j->out, "-taught.png"));
    json_object_put(side);
    return current;
}

char *kept_window(Stage *g, Job *j, Arena *a) {
    return fmt(a, "%s/%s/%s/%s/%s/piece-%04d.zst", g->windows, g->sample, g->scan, SAMPLING,
        j->unit, j->piece);
}

void write_png(Arena *a, const char *path, const u8 *grey, int w, int h) {
    u64 mark = a->used;
    Bytes b = png_encode(a, grey, w, h, 3);
    write_file(path, b.data, b.size);
    a->used = mark;
}

void keep_window(Arena *a, const char *path, const u8 *face, int h, int w) {
    mkparent(path);
    struct statvfs fs;
    if (statvfs(fmt(a, "%.*s", (int)(strrchr(path, '/') - path), path), &fs) == 0 &&
        (u64)fs.f_bavail * fs.f_frsize <= KEEP_FREE) {
        return;
    }
    u64 mark = a->used;
    i64 n = (i64)DEPTH * h * w;
    Bytes packed = zstd_pack(a, face, n, 1);
    u8 *file = push(a, 12 + packed.size);
    u32 shape[3] = {DEPTH, h, w};
    memcpy(file, shape, 12);
    memcpy(file + 12, packed.data, packed.size);
    write_file(path, file, 12 + packed.size);
    a->used = mark;
}

typedef struct {
    u8 *stack;
    int height, width, measure;
    const char *seen;
    f32 *map;
    Stats *stats;
} Read;

void write_side(Arena *a, const char *path, Recto fibres, char *const *names, const f64 *values,
    int looks, const char *written, const char *model, int turn) {
    char *text = fmt(a,
        "{\"fibres\": {\"face\": \"%s\", \"polarity\": %s, \"error\": %s, \"tiles\": %d}",
        fibres.face, fibres.tiles >= FEWEST_FIBRE_TILES ? python_float(a, fibres.polarity) : "null",
        fibres.tiles >= FEWEST_FIBRE_TILES ? python_float(a, fibres.error) : "null", fibres.tiles);
    for (int k = 0; k < looks; k++) {
        text = fmt(a, "%s, \"%s\": %s", text, names[k], python_float(a, values[k]));
    }
    text = fmt(
        a, "%s, \"written\": \"%s\", \"model\": \"%s\", \"turn\": %d}", text, written, model, turn);
    write_file(path, text, strlen(text));
}

typedef struct {
    Job *job;
    Arena arena;
    int current, h, w, count;
    Read reads[8];
    u8 *sheet;
    Recto fibres;
    char written[8];
    json_object *side;
    const char *record;
    i64 last;
} Piece;

Piece *sample_piece(Stage *g, Job *j, const Bricks *store) {
    Piece *p = calloc(1, sizeof(Piece));
    p->job = j;
    p->arena = arena(64 * GB);
    Arena *a = &p->arena;
    const char *record = beside(a, j->out, "-taught-side.json"),
               *layer0 = beside(a, j->out, "-layer0.png");
    const char *kept = kept_window(g, j, a);
    json_object *side = side_record(record);
    json_object *was = 0;
    int cached = newer(kept, j->folder, a) && exists(j->out) && exists(layer0) && side &&
        mtime_ns(kept) <= mtime_ns(record) && json_object_object_get_ex(side, "written", &was);
    json_object *turned = 0;
    if (cached && json_object_object_get_ex(side, "turn", &turned)) {
        cached = json_object_get_int(turned) == j->turn;
    } else if (cached) {
        cached = j->turn == 1;
    }
    char *written = p->written;
    snprintf(written, 8, "near");
    Recto fibres = {.face = "unknown"};
    u8 *sheet, *face;
    int h, w;
    Read *reads = p->reads;
    int count = 0;
    if (cached) {
        snprintf(written, 8, "%s", json_object_get_string(was));
        Bytes raw = read_file(a, kept);
        u32 shape[3];
        memcpy(shape, raw.data, 12);
        face = zstd_unpack(a, raw.data + 12, raw.size - 12).data;
        h = shape[1];
        w = shape[2];
        sheet = png_read(a, j->out, &w, &h);
        reads[count++] = (Read){.stack = face, .height = h, .width = w, .measure = 1, .seen = 0};
    } else {
        json_object_put(side);
        side = 0;
        Arena scratch = arena(a->size);
        Surface s = load_surface(&scratch, j->folder);
        Readable r = readable(&scratch, s, g->line, g->rows);
        Stack stack = flatten(&scratch, &r, j->turn, store, LAYERS);
        h = stack.height;
        w = stack.width;
        i64 n = (i64)h * w;
        sheet = push(a, n);
        u8 *central = push(a, n);
        across_sheet(stack, s.voxel, sheet, central);
        mkparent(j->out);
        if (ink_is_current(g, j, a)) {
            munmap(scratch.base, scratch.size);
            write_png(a, layer0, central, w, h);
            write_png(a, j->out, sheet, w, h);
            p->current = 1;
            return p;
        }
        fibres = recto_face(stack);
        snprintf(written, 8, "%s", strcmp(fibres.face, "far") ? "near" : "far");
        int far = !strcmp(written, "far");
        face = push(a, DEPTH * n);
        window_of(stack, far, face);
        i64 covered = 0;
        for (i64 i = 0; i < n; i++) {
            covered += sheet[i] > 0;
        }
        if (covered >= LOOKED_PIXELS) {
            int spots[LOOKS][2], side_length = fullest(sheet, h, w, spots);
            i64 m = (i64)side_length * side_length;
            u8 *other = push(a, DEPTH * n);
            window_of(stack, !far, other);
            munmap(scratch.base, scratch.size);
            Pcg64 rng = pcg64((u64)h * 31 + w);
            const char *names[2] = {far ? "near" : "far", written};
            u8 *faces[2] = {other, face};
            u8 *last = 0;
            for (int f = 0; f < 2; f++) {
                reads[count++] =
                    (Read){.stack = faces[f], .height = h, .width = w, .measure = 2, .seen = 0};
                for (int k = 0; k < LOOKS; k++) {
                    u8 *part = push(a, DEPTH * m);
                    for (int l = 0; l < DEPTH; l++) {
                        for (int y = 0; y < side_length; y++) {
                            memcpy(part + l * m + (i64)y * side_length,
                                faces[f] + (i64)l * n + (i64)(spots[k][0] + y) * w + spots[k][1],
                                side_length);
                        }
                    }
                    reads[count++] = (Read){.stack = part,
                        .height = side_length,
                        .width = side_length,
                        .measure = 0,
                        .seen = names[f]};
                    last = k == 0 ? part : last;
                }
            }
            u8 *noise = push(a, DEPTH * m);
            scrambled(last, side_length, &rng, noise);
            reads[count++] = (Read){.stack = noise,
                .height = side_length,
                .width = side_length,
                .measure = 0,
                .seen = "scrambled"};
            reads[count++] =
                (Read){.stack = face, .height = h, .width = w, .measure = 0, .seen = 0};
        } else {
            munmap(scratch.base, scratch.size);
            reads[count++] =
                (Read){.stack = face, .height = h, .width = w, .measure = 1, .seen = 0};
        }
        unlink(record);
        unlink(kept);
        write_png(a, layer0, central, w, h);
        write_png(a, j->out, sheet, w, h);
        if (strcmp(env("HERCULEX_KEEP_STACKS", "1"), "0") != 0) {
            keep_window(a,
                fmt(a, "%s/%s/%s/%s/%s/piece-%04d.zst", g->windows, g->sample, g->scan, SAMPLING,
                    j->unit, j->piece),
                face, h, w);
        }
    }
    for (int k = 0; k < count; k++) {
        Read *r = reads + k;
        if (r->measure && g->reader) {
            r->stats = push(a, sizeof(Stats));
            measure_stack(g->reader, r->stack, r->height, r->width, r->stats);
        }
    }
    p->h = h;
    p->w = w;
    p->count = count;
    p->sheet = sheet;
    p->fibres = fibres;
    p->side = side;
    p->record = record;
    return p;
}

void queue_piece(Stage *g, Piece *p) {
    for (int k = 0; k < p->count; k++) {
        Read *r = p->reads + k;
        if (r->measure && g->reader) {
            use_stats(g->reader, r->stats);
        }
        if (r->measure == 2) {
            continue;
        }
        r->map = push(&p->arena, (i64)r->height * r->width * 4);
        p->last = g->studio ? studio_read(g->studio, r->stack, r->height, r->width, r->map)
                            : ink_queue(g->reader, r->stack, r->height, r->width, r->map);
    }
}

const char *write_ink_piece(const Stage *g, Piece *p) {
    Arena *a = &p->arena;
    Job *j = p->job;
    Recto fibres = p->fibres;
    json_object *side = p->side;
    int h = p->h, w = p->w;
    char *names[8];
    f64 values[8], sums[8] = {0};
    int looks = 0, counts[8] = {0};
    if (side) {
        json_object *f = 0;
        if (json_object_object_get_ex(side, "fibres", &f)) {
            fibres.face = fmt(a, "%s", jstr(f, "face"));
            fibres.tiles = ji64(f, "tiles");
            fibres.polarity = json_object_get_double(jget(f, "polarity"));
            fibres.error = json_object_get_double(jget(f, "error"));
        }
        json_object_object_foreach(side, key, value) {
            int skip = !strcmp(key, "fibres") || !strcmp(key, "written") || !strcmp(key, "model") ||
                !strcmp(key, "turn");
            if (!skip && looks < 8) {
                names[looks] = fmt(a, "%s", key);
                values[looks++] = json_object_get_double(value);
            }
        }
        json_object_put(side);
    } else {
        names[looks] = "near";
        values[looks++] = 0;
        names[looks] = "far";
        values[looks++] = 0;
    }
    f32 *map = 0;
    for (int k = 0; k < p->count; k++) {
        Read *r = p->reads + k;
        if (r->measure == 2) {
            continue;
        }
        map = r->map;
        if (!r->seen) {
            continue;
        }
        int at = 0;
        while (at < looks && strcmp(names[at], r->seen) != 0) {
            at++;
        }
        if (at == looks) {
            names[looks] = (char *)r->seen;
            values[looks++] = 0;
        }
        sums[at] += sure_of(r->map, (i64)r->height * r->width);
        counts[at]++;
    }
    for (int k = 0; k < looks; k++) {
        values[k] = counts[k] ? sums[k] / counts[k] : values[k];
    }
    i64 n = (i64)h * w;
    u8 *shown = push(a, n);
    for (i64 i = 0; i < n; i++) {
        f32 scaled = (map[i] - NO_INK) / 0.5f;
        scaled = scaled < 0 ? 0 : scaled > 1 ? 1 : scaled;
        f32 grey = scaled * 255;
        shown[i] = p->sheet[i] > 0 ? (u8)grey : 0;
    }
    write_png(a, beside(a, j->out, "-taught.png"), shown, w, h);
    write_side(a, p->record, fibres, names, values, looks, p->written, g->digest, j->turn);
    return strcmp(p->written, "far") ? "near" : "far";
}

#define PIECES_AHEAD 6

typedef struct {
    Stage *g;
    Job *jobs;
    int count, next, ahead, ready_count;
    const Bricks *store;
    Piece *ready[PIECES_AHEAD];
    pthread_mutex_t lock;
    pthread_cond_t ready_signal, room;
} Samplers;

static int sampler_count(void) {
    const char *setting = env("HERCULEX_SAMPLERS", "");
    char *end;
    long count = strtol(setting, &end, 10);
    if (*end || count < 1) {
        count = cores() - 2;
        count = count < 1 ? 1 : count;
    }
    return count > PIECES_AHEAD ? PIECES_AHEAD : (int)count;
}

void *sampler_loop(void *context) {
    Samplers *s = context;
    for (;;) {
        pthread_mutex_lock(&s->lock);
        while (s->ahead >= PIECES_AHEAD && s->next < s->count) {
            pthread_cond_wait(&s->room, &s->lock);
        }
        if (s->next >= s->count) {
            pthread_mutex_unlock(&s->lock);
            return 0;
        }
        int k = s->next++;
        s->ahead++;
        pthread_mutex_unlock(&s->lock);
        Piece *p = sample_piece(s->g, s->jobs + k, s->store);
        pthread_mutex_lock(&s->lock);
        s->ready[s->ready_count++] = p;
        pthread_cond_signal(&s->ready_signal);
        pthread_mutex_unlock(&s->lock);
    }
}

typedef struct {
    Samplers *samplers;
    Piece *ready[PIECES_AHEAD];
    pthread_cond_t write_ready;
    int head, count, written, background;
    f64 began, seconds;
} InkWriter;

static void write_and_release(InkWriter *writer, Piece *piece) {
    Samplers *s = writer->samplers;
    f64 start = now();
    const char *written = piece->current ? "current" : write_ink_piece(s->g, piece);
    writer->seconds += now() - start;
    int done = ++writer->written;
    if (done % 10 == 0 || done == s->count) {
        printf("%s %d of %d {'piece': 'piece-%04d', 'unit': '%s', 'written': '%s'} %.0f s\n",
            piece->job->unit, done, s->count, piece->job->piece, piece->job->unit, written,
            now() - writer->began);
        fflush(stdout);
    }
    munmap(piece->arena.base, piece->arena.size);
    free(piece);
    pthread_mutex_lock(&s->lock);
    s->ahead--;
    pthread_cond_broadcast(&s->room);
    pthread_mutex_unlock(&s->lock);
}

static void *write_ink_pieces(void *context) {
    InkWriter *writer = context;
    Samplers *s = writer->samplers;
    for (int k = 0; k < s->count; k++) {
        pthread_mutex_lock(&s->lock);
        while (!writer->count) {
            pthread_cond_wait(&writer->write_ready, &s->lock);
        }
        Piece *p = writer->ready[writer->head];
        writer->head = (writer->head + 1) % PIECES_AHEAD;
        writer->count--;
        pthread_mutex_unlock(&s->lock);
        write_and_release(writer, p);
    }
    return NULL;
}

static void queue_ink_write(InkWriter *writer, Piece *piece) {
    if (!writer->background) {
        write_and_release(writer, piece);
        return;
    }
    Samplers *s = writer->samplers;
    pthread_mutex_lock(&s->lock);
    writer->ready[(writer->head + writer->count++) % PIECES_AHEAD] = piece;
    pthread_cond_signal(&writer->write_ready);
    pthread_mutex_unlock(&s->lock);
}

typedef struct {
    Bricks *store;
    i64 plane, *first;
    Ask *asks;
} Delivery;

void deliver_brick(void *context, Ask *ask, const u8 *body, size_t size) {
    Delivery *d = context;
    if (!body || (i64)size == d->store->size) {
        bricks_put(d->store, ask->index, body);
        return;
    }
    u8 *whole = calloc(d->store->size, 1);
    i64 begin = d->first[ask - d->asks] * d->plane;
    memcpy(whole + begin, body, size);
    bricks_put(d->store, ask->index, whole);
    free(whole);
}

int text_order(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static void close_ink_store(Bricks *store) {
    if (store->data) {
        munmap((void *)store->data, store->mapped);
    }
    pthread_mutex_destroy(&store->lock);
}

static void write_brick_plan(const char *path, char **paths, int count) {
    qsort(paths, count, sizeof(char *), text_order);
    json_object *plan = json_object_new_array_ext(count);
    need(plan != NULL, "cannot allocate a plan for %s", path);
    for (int k = 0; k < count; k++) {
        json_object *piece = json_object_new_string(paths[k]);
        need(piece && !json_object_array_add(plan, piece), "cannot add a piece to %s", path);
    }
    const char *text = json_object_to_json_string_ext(plan, JSON_C_TO_STRING_PLAIN);
    write_file(path, text, strlen(text));
    json_object_put(plan);
}

static int brick_plan_covers(json_object *plan, char **paths, int count) {
    if (!json_object_is_type(plan, json_type_array) ||
        json_object_array_length(plan) < (size_t)count) {
        return 0;
    }
    json_object *listed = json_object_new_object();
    need(listed != NULL, "cannot allocate a brick plan index");
    for (size_t i = 0; i < json_object_array_length(plan); i++) {
        json_object *piece = json_object_array_get_idx(plan, i);
        if (json_object_is_type(piece, json_type_string)) {
            need(!json_object_object_add(listed, json_object_get_string(piece), NULL),
                "cannot index a brick plan piece");
        }
    }
    int covered = 1;
    for (int k = 0; k < count && covered; k++) {
        covered = json_object_object_get_ex(listed, paths[k], NULL);
    }
    json_object_put(listed);
    return covered;
}

typedef struct {
    const Stage *stage;
    const Job *jobs;
    const Bricks *store;
    i32 *ranges;
    int count, workers;
} BrickPlanning;

static void plan_brick_ranges(void *context, i64 worker) {
    const BrickPlanning *plan = context;
    Arena a = arena(8 * GB);
    i32 *ranges = plan->ranges + 2 * worker * plan->store->count;
    for (i64 i = 0; i < plan->store->count; i++) {
        ranges[2 * i] = plan->store->brick[0];
        ranges[2 * i + 1] = -1;
    }
    for (i64 k = worker; k < plan->count; k += plan->workers) {
        Surface surface = load_surface(&a, plan->jobs[k].folder);
        Readable mesh = readable(&a, surface, plan->stage->line, plan->stage->rows);
        mark_ranges(ranges, plan->store, &mesh, &a);
        a.used = 0;
    }
    munmap(a.base, a.size);
}

static i32 *brick_ranges(
    Arena *a, const Stage *g, const Job *jobs, int count, const Bricks *store) {
    int workers = count < 4 ? count : 4;
    i64 cells = 2 * store->count;
    i32 *ranges = push(a, cells * workers * sizeof(*ranges));
    BrickPlanning plan = {.stage = g,
        .jobs = jobs,
        .store = store,
        .ranges = ranges,
        .count = count,
        .workers = workers};
    parallel(workers, workers, plan_brick_ranges, &plan);
    for (int worker = 1; worker < workers; worker++) {
        const i32 *other = ranges + cells * worker;
        for (i64 i = 0; i < cells; i += 2) {
            include(ranges + i, other[i], other[i + 1]);
        }
    }
    return ranges;
}

static int prepare_ink_store(
    Stage *g, Job *jobs, int count, const char *folder, Arena *a, int ahead, int *store_lock) {
    if (!count) {
        return 1;
    }
    u64 mark = a->used;
    f64 began = now();
    mkparent(folder);
    const char *name = strrchr(folder, '/') + 1;
    const char *lock_path = fmt(a, "%.*s.%s.fetching", (int)(name - folder), folder, name);
    int lock = lock_file(lock_path, 1, 1);
    need(lock >= 0, "cannot lock %s: %s", lock_path, strerror(errno));
    char **paths = push(a, count * sizeof(char *)), **records = push(a, count * sizeof(char *));
    int units = 0;
    for (int k = 0; k < count; k++) {
        paths[k] = jobs[k].folder;
        char *record = fmt(a, "%.*s/record.json",
            (int)(strrchr(jobs[k].folder, '/') - jobs[k].folder), jobs[k].folder);
        int seen = 0;
        for (int u = 0; u < units; u++) {
            seen |= !strcmp(records[u], record);
        }
        if (!seen) {
            records[units++] = record;
        }
    }
    qsort(records, units, sizeof(char *), text_order);
    EVP_MD_CTX *c = EVP_MD_CTX_new();
    EVP_DigestInit_ex(c, EVP_sha256(), 0);
    char *head = fmt(a, "[\"%s\", %d]", g->scan, LAYERS);
    EVP_DigestUpdate(c, head, strlen(head));
    for (int u = 0; u < units; u++) {
        if (exists(records[u])) {
            Bytes b = read_file(a, records[u]);
            EVP_DigestUpdate(c, b.data, b.size);
        }
    }
    u8 d[32];
    unsigned int dn;
    EVP_DigestFinal_ex(c, d, &dn);
    EVP_MD_CTX_free(c);
    char stamp[40];
    for (int i = 0; i < 8; i++) {
        snprintf(stamp + 2 * i, 3, "%02x", d[i]);
    }
    char *planned = fmt(a, "%s/planned.json", folder), *stamped = fmt(a, "%s/unit.json", folder);
    json_object *plan = json_read(planned);
    int stale = plan && !brick_plan_covers(plan, paths, count);
    json_object *old = json_read(stamped);
    stale |= old && strcmp(json_object_get_string(old), stamp) != 0;
    json_object_put(plan);
    json_object_put(old);
    if (stale) {
        remove_tree(folder);
    }
    char *source;
    char *url = zarr_level(a, g->sample, g->scan, 0, &source);
    if (!exists(fmt(a, "%s/meta.json", folder))) {
        json_object *zarray = http_json(fmt(a, "%s/.zarray", url));
        i64 shape[3], brick[3];
        for (int k = 0; k < 3; k++) {
            shape[k] = json_object_get_int64(json_object_array_get_idx(jget(zarray, "shape"), k));
            brick[k] = json_object_get_int64(json_object_array_get_idx(jget(zarray, "chunks"), k));
        }
        json_object_put(zarray);
        bricks_create(a, folder, shape, brick, source);
    }
    if (!exists(planned)) {
        write_brick_plan(planned, paths, count);
        char *quoted = fmt(a, "\"%s\"", stamp);
        write_file(stamped, quoted, strlen(quoted));
    }
    Bricks store;
    bricks_open(a, &store, folder);
    i64 unknown = 0;
    for (i64 i = 0; i < store.count; i++) {
        unknown += store.offsets[i] == UNKNOWN;
    }
    int ready = 1;
    if (!unknown) {
        printf("%s %s pieces %d fetched 0\n", g->sample, name, count);
        goto finish;
    }
    i32 *touched = brick_ranges(a, g, jobs, count, &store);
    Ask *asks = push(a, store.count * sizeof(Ask));
    i64 *first = push(a, store.count * 8), wanted = 0, moved = 0;
    i64 plane = store.brick[1] * store.brick[2];
    for (i64 i = 0; i < store.count; i++) {
        if (touched[2 * i + 1] < 0 || store.offsets[i] != UNKNOWN) {
            continue;
        }
        i64 z = i / (store.grid[1] * store.grid[2]), y = i / store.grid[2] % store.grid[1];
        i64 x = i % store.grid[2];
        asks[wanted] = (Ask){.index = i,
            .first = touched[2 * i] * plane,
            .last = (touched[2 * i + 1] + 1) * plane - 1};
        snprintf(asks[wanted].path, sizeof(asks[wanted].path), "%lld/%lld/%lld", (long long)z,
            (long long)y, (long long)x);
        first[wanted] = touched[2 * i];
        moved += touched[2 * i + 1] - touched[2 * i] + 1;
        wanted++;
    }
    if (ahead && wanted) {
        struct statvfs fs;
        u64 free_bytes = statvfs(folder, &fs) == 0 ? (u64)fs.f_bavail * fs.f_frsize : 0;
        if (free_bytes < FETCH_FREE || (u64)wanted > (free_bytes - FETCH_FREE) / store.size) {
            printf(
                "%s %s: fetching CT after the previous unit releases its store\n", g->sample, name);
            ready = 0;
            goto finish;
        }
    }
    Delivery delivery = {.store = &store, .plane = plane, .first = first, .asks = asks};
    f64 planned_at = now();
    i64 failed = fetch_ranges(url, asks, wanted, deliver_brick, &delivery);
    if (failed && ahead) {
        bricks_save(&store);
        ready = 0;
        goto finish;
    }
    if (failed) {
        printf("%lld bricks failed; trying them again in 60 s\n", (long long)failed);
        sleep(60);
        i64 again = 0;
        for (i64 k = 0; k < wanted; k++) {
            if (store.offsets[asks[k].index] == UNKNOWN) {
                asks[again] = asks[k];
                first[again++] = first[k];
            }
        }
        failed = fetch_ranges(url, asks, again, deliver_brick, &delivery);
    }
    bricks_save(&store);
    need(!failed, "%lld bricks of %s failed; rerun to fetch them", (long long)failed, name);
    json_object *kept = json_read(planned);
    if (kept && (int)json_object_array_length(kept) > count) {
        write_brick_plan(planned, paths, count);
    }
    json_object_put(kept);
    for (i64 i = 0; i < store.count; i++) {
        store.offsets[i] = store.offsets[i] == UNKNOWN ? ABSENT : store.offsets[i];
    }
    bricks_save(&store);
    printf("%s %s pieces %d fetched %lld bricks, %.2f GiB, %.2f s planning and %.2f s transfer\n",
        g->sample, name, count, (long long)wanted, (f64)moved * plane / GB, planned_at - began,
        now() - planned_at);
finish:
    close_ink_store(&store);
    if (ready && store_lock) {
        *store_lock = lock;
    } else {
        close(lock);
    }
    a->used = mark;
    return ready;
}

void fetch_store(Stage *g, Job *jobs, int count, const char *folder, Arena *a) {
    prepare_ink_store(g, jobs, count, folder, a, 0, NULL);
}

int job_order(const void *a, const void *b) {
    const Job *x = a, *y = b;
    int u = strcmp(x->unit, y->unit);
    if (u) {
        return u;
    }
    if (x->area != y->area) {
        return x->area < y->area ? 1 : -1;
    }
    return x->piece != y->piece ? (x->piece < y->piece ? 1 : -1)
                                : (x->turn < y->turn) - (x->turn > y->turn);
}

i64 columns_of(Arena *a, Job *j) {
    u64 mark = a->used;
    json_object *meta = json_read(fmt(a, "%s/meta.json", j->folder));
    int stride = ji64(meta, "stride");
    json_object_put(meta);
    TIFF *t = tiff_open_read(fmt(a, "%s/x.tif", j->folder));
    need(t != 0, "cannot open %s/x.tif", j->folder);
    u32 w, h;
    TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
    TIFFClose(t);
    a->used = mark;
    return ((i64)(h + 2 * MARGIN - 1) * stride + 1) * ((i64)(w + 2 * MARGIN - 1) * stride + 1);
}

char *first_line(Arena *a, char *const arguments[]) {
    int output[2];
    need(!pipe(output), "cannot open a pipe");
    pid_t child = fork();
    if (!child) {
        close(output[0]);
        dup2(output[1], STDOUT_FILENO);
        close(output[1]);
        execv(arguments[0], arguments);
        _exit(127);
    }
    close(output[1]);
    FILE *stream = fdopen(output[0], "r");
    char line[1024] = {0};
    int got = fgets(line, sizeof(line), stream) != 0;
    fclose(stream);
    int status;
    waitpid(child, &status, 0);
    if (!got || !WIFEXITED(status) || WEXITSTATUS(status)) {
        return 0;
    }
    line[strcspn(line, "\r\n")] = 0;
    return fmt(a, "%s", line);
}

void stage_open(Arena *a, Stage *g, int argc, char **argv) {
    need(argc >= 1, "usage: SCROLL [--scan ID]");
    snprintf(g->sample, sizeof(g->sample), "%s", argv[0]);
    const char *scan = flag(argc, argv, "--scan");
    scan = scan ? scan : argc == 2 && argv[1][0] != '-' ? argv[1] : 0;
    if (!scan) {
        char *chosen = first_line(
            a, (char *[]){"./run.sh", "scans.catalog.scan_choice", "chosen", argv[0], 0});
        char held[256], found[256];
        need(chosen && sscanf(chosen, "%255s %255s", held, found) == 2, "%s: no scan chosen",
            argv[0]);
        scan = fmt(a, "%s", found);
    }
    snprintf(g->scan, sizeof(g->scan), "%s", scan);
    snprintf(g->base, sizeof(g->base), "%s/unwrapping/level/%s/%s", OUTPUTS, g->sample, g->scan);
    const char *stacks = fmt(a, "%s/stacks", OUTPUTS);
    const char *windows =
        exists(fmt(a, "%s/level", stacks)) || !exists(fmt(a, "%s/level_route", stacks))
        ? fmt(a, "%s/level", stacks)
        : fmt(a, "%s/level_route", stacks);
    snprintf(g->windows, sizeof(g->windows), "%s", env("HERCULEX_WINDOWS", windows));
}

void stage_axis(Arena *a, Stage *g) {
    json_object *axis = json_read(fmt(a, "%s/sheet/axis.json", g->base));
    need(axis != 0, "%s %s: no umbilicus kept; ./run.sh level.level.umbilicus %s %s keeps one",
        g->sample, g->scan, g->sample, g->scan);
    g->rows = json_object_array_length(axis);
    g->line = push(a, g->rows * 3 * 8);
    for (int k = 0; k < g->rows; k++) {
        json_object *point = json_object_array_get_idx(axis, k);
        for (int c = 0; c < 3; c++) {
            g->line[k * 3 + c] = json_object_get_double(json_object_array_get_idx(point, c));
        }
    }
    json_object_put(axis);
}

const char *store_of(Arena *a, Stage *g, const char *unit) {
    const char *bricks = env("HERCULEX_BRICKS", "");
    return *bricks ? fmt(a, "%s/%s/ct_bricks/%s", bricks, g->sample, unit)
                   : fmt(a, "%s/cache/bricks/%s/%s", LIBRARY, g->sample, unit);
}

typedef struct {
    Stage *stage;
    Job *jobs;
    int count, ahead, ready, lock;
    pthread_t thread;
} InkFetch;

static void *fetch_ink_unit(void *argument) {
    InkFetch *fetch = argument;
    Arena a = arena(64 * GB);
    const char *folder = store_of(&a, fetch->stage, fetch->jobs[0].unit);
    fetch->ready = prepare_ink_store(
        fetch->stage, fetch->jobs, fetch->count, folder, &a, fetch->ahead, &fetch->lock);
    munmap(a.base, a.size);
    return NULL;
}

static int ink_unit_end(const Job *jobs, int first, int count) {
    int end = first + 1;
    while (end < count && !strcmp(jobs[end].unit, jobs[first].unit)) {
        end++;
    }
    return end;
}

static int start_ink_fetch(InkFetch *fetch, Stage *stage, Job *jobs, int count, int ahead) {
    *fetch = (InkFetch){.stage = stage, .jobs = jobs, .count = count, .ahead = ahead};
    return pthread_create(&fetch->thread, NULL, fetch_ink_unit, fetch) == 0;
}

int ink_command(int argc, char **argv) {
    Arena a = arena(256 * GB);
    Stage g = {.model = chosen_model(&a)};
    stage_open(&a, &g, argc, argv);
    stage_axis(&a, &g);
    model_digest(&a, &g.model, g.digest);
    Bytes same = read_file(&a, fmt(&a, "%s/ink/reading/same_output_models.txt", HOME));
    char *text = fmt(&a, "%.*s", (int)same.size, same.data), *line, *rest = text;
    while ((line = strsep(&rest, "\n"))) {
        char *word = strsep(&line, " ");
        if (word && !strcmp(word, g.digest)) {
            for (word = line ? strsep(&line, " ") : NULL; word && g.sames < 256;
                word = line ? strsep(&line, " ") : NULL) {
                if (*word) {
                    g.same[g.sames++] = word;
                }
            }
        }
    }
    json_object *layout = json_read(fmt(&a, "%s/sheet/layout.json", g.base));
    need(layout != 0, "%s %s: no layout; run the layout stage first", g.sample, g.scan);
    json_object *pieces = jget(layout, "pieces");
    int total = json_object_array_length(pieces), count = 0;
    Job *jobs = push(&a, (i64)total * sizeof(Job));
    const char *only = env("HERCULEX_UNITS", "");
    for (int k = 0; k < total; k++) {
        json_object *p = json_object_array_get_idx(pieces, k), *turn = 0;
        const char *unit = jstr(p, "unit");
        if (*only && !strstr(fmt(&a, ",%s,", only), fmt(&a, ",%s,", unit))) {
            continue;
        }
        Job *j = jobs + count++;
        snprintf(j->unit, sizeof(j->unit), "%s", unit);
        j->piece = ji64(p, "piece");
        j->area = jf64(p, "area");
        j->turn = json_object_object_get_ex(p, "turn", &turn) ? json_object_get_int(turn) : 1;
        snprintf(j->folder, sizeof(j->folder), "%s/pieces/%s/piece-%04d", g.base, unit, j->piece);
        snprintf(j->out, sizeof(j->out), "%s/faces/%s/%s/piece-%04d.png", g.base, g.model.faces,
            unit, j->piece);
    }
    qsort(jobs, count, sizeof(Job), job_order);
    int left = 0;
    i64 widest = 1;
    for (int k = 0; k < count; k++) {
        Job *j = jobs + k;
        int done = exists(j->out) && exists(beside(&a, j->out, "-layer0.png")) &&
            ink_is_current(&g, j, &a);
        if (!done) {
            jobs[left++] = *j;
            i64 columns = columns_of(&a, j);
            widest = columns > widest ? columns : widest;
        }
    }
    if (!left) {
        printf("%s %s: no ink maps to update\n", g.sample, g.scan);
        return 0;
    }
    printf("%s %s: %d pieces queued\n", g.sample, g.scan, left);
    char *started = fmt(&a, "%s/sheet/reading.started", g.base);
    char *stamp = fmt(&a, "%.6f\n", now());
    write_file(started, stamp, strlen(stamp));
    InkFetch fetch;
    int end = ink_unit_end(jobs, 0, left);
    int fetching = start_ink_fetch(&fetch, &g, jobs, end, 0);
    if (!strcmp(g.model.backend, "studio")) {
        g.studio = studio_open(&a, g.model.weights);
    } else {
        g.reader = reader_open(&a, g.model.weights, widest);
    }
    int samplers_per_unit = sampler_count();
    for (int k = 0; k < left;) {
        u64 unit_mark = a.used;
        const char *folder = store_of(&a, &g, jobs[k].unit);
        f64 began = now();
        int store_lock = -1;
        if (fetching) {
            pthread_join(fetch.thread, NULL);
        }
        if (fetching && fetch.ready) {
            store_lock = fetch.lock;
        } else {
            prepare_ink_store(&g, jobs + k, end - k, folder, &a, 0, &store_lock);
        }
        f64 ct = now() - began, waiting = 0, reading = 0, settling = 0;
        Bricks store;
        bricks_open(&a, &store, folder);
        int next = end < left ? ink_unit_end(jobs, end, left) : left;
        fetching = end < left && start_ink_fetch(&fetch, &g, jobs + end, next - end, 1);
        Samplers samplers = {.g = &g, .jobs = jobs + k, .count = end - k, .store = &store};
        pthread_mutex_init(&samplers.lock, 0);
        pthread_cond_init(&samplers.ready_signal, 0);
        pthread_cond_init(&samplers.room, 0);
        pthread_t threads[PIECES_AHEAD];
        int workers = 0;
        while (workers < samplers_per_unit && workers < samplers.count &&
            !pthread_create(threads + workers, NULL, sampler_loop, &samplers)) {
            workers++;
        }
        need(workers > 0, "%s %s: cannot start ink samplers", g.sample, jobs[k].unit);
        InkWriter writer = {
            .samplers = &samplers, .write_ready = PTHREAD_COND_INITIALIZER, .began = began};
        pthread_t writer_thread;
        writer.background = !pthread_create(&writer_thread, NULL, write_ink_pieces, &writer);
        Piece *previous = 0;
        for (int i = 0; i <= end - k; i++) {
            Piece *p = 0;
            if (i < end - k) {
                f64 start = now();
                pthread_mutex_lock(&samplers.lock);
                while (!samplers.ready_count) {
                    pthread_cond_wait(&samplers.ready_signal, &samplers.lock);
                }
                p = samplers.ready[--samplers.ready_count];
                pthread_mutex_unlock(&samplers.lock);
                f64 ready = now();
                waiting += ready - start;
                if (!p->current) {
                    queue_piece(&g, p);
                    reading += now() - ready;
                }
            }
            if (previous) {
                f64 start = now();
                if (g.reader) {
                    ink_finish_through(g.reader, previous->last);
                }
                settling += now() - start;
                queue_ink_write(&writer, previous);
            }
            previous = p;
            if (p && (p->current || g.studio || ink_ready(g.reader, p->last))) {
                queue_ink_write(&writer, p);
                previous = NULL;
            }
        }
        for (int t = 0; t < workers; t++) {
            pthread_join(threads[t], 0);
        }
        if (writer.background) {
            pthread_join(writer_thread, NULL);
        }
        pthread_cond_destroy(&writer.write_ready);
        pthread_cond_destroy(&samplers.room);
        pthread_cond_destroy(&samplers.ready_signal);
        pthread_mutex_destroy(&samplers.lock);
        close_ink_store(&store);
        if (strcmp(env("HERCULEX_KEEP_BRICKS", ""), "1") != 0) {
            remove_tree(folder);
        }
        close(store_lock);
        printf("%s %s ink %.2f s elapsed, %.2f s waiting for CT, %.2f s waiting for samples, "
               "%.2f s queueing reads, %.2f s settling reads, %.2f s writing pieces\n",
            g.sample, jobs[k].unit, now() - began, ct, waiting, reading, settling, writer.seconds);
        a.used = unit_mark;
        k = end;
        end = next;
    }
    return 0;
}

int unit_order(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

char **units_in(Arena *a, const char *folder, int *count) {
    int all = 0, n = 0;
    char **names = exists(folder) ? list_dir(a, folder, &all) : 0;
    char **units = push(a, (all + 1) * sizeof(char *));
    for (int pass = 0; pass < 2 && !n; pass++) {
        for (int k = 0; k < all; k++) {
            if (!strncmp(names[k], pass ? "band-" : "rows-", 5) && !strstr(names[k], ".building")) {
                units[n++] = names[k];
            }
        }
    }
    qsort(units, n, sizeof(char *), unit_order);
    *count = n;
    return units;
}

int unit_finished(Arena *a, const char *unit, const char *method, char **same, int sames) {
    json_object *record = json_read(fmt(a, "%s/record.json", unit)), *m = 0, *pieces = 0;
    int done = record && json_object_object_get_ex(record, "method", &m) &&
        json_object_object_get_ex(record, "pieces", &pieces) &&
        json_object_is_type(pieces, json_type_array);
    if (done) {
        const char *recorded = json_object_get_string(m);
        int known = !strcmp(recorded, method);
        for (int k = 0; k < sames; k++) {
            known |= !strcmp(recorded, same[k]);
        }
        done = known;
    }
    json_object_put(record);
    return done;
}

i64 folder_bytes(Arena *a, const char *folder) {
    int count = 0;
    char **names = exists(folder) ? list_dir(a, folder, &count) : 0;
    i64 total = 0;
    for (int k = 0; k < count; k++) {
        struct stat st;
        if (!stat(fmt(a, "%s/%s", folder, names[k]), &st) && S_ISREG(st.st_mode)) {
            total += st.st_size;
        }
    }
    return total;
}

int taken_over(Arena *a, Stage *g, const char *pieces, char **units, int count) {
    i64 newest = 0;
    for (int k = 0; k < count; k++) {
        i64 t = mtime_ns(fmt(a, "%s/%s/record.json", pieces, units[k]));
        newest = t > newest ? t : newest;
    }
    if (!count) {
        return 0;
    }
    const char *names[] = {"layout.json", "reading.started"};
    for (int k = 0; k < 2; k++) {
        const char *path = fmt(a, "%s/sheet/%s", g->base, names[k]);
        if (exists(path) && mtime_ns(path) > newest) {
            return 1;
        }
    }
    return 0;
}

int follow_fetch(int argc, char **argv) {
    Arena a = arena(64 * GB);
    Stage g = {0};
    stage_open(&a, &g, argc, argv);
    if (!exists(fmt(&a, "%s/unwrapping/nodes/%s/%s/nodes", OUTPUTS, g.sample, g.scan))) {
        printf("%s %s: no sheet nodes to fetch for\n", g.sample, g.scan);
        return 0;
    }
    stage_axis(&a, &g);
    char *method = first_line(&a, (char *[]){"./run.sh", "level.level.run_levels", "method", 0});
    need(method != 0, "the level fit names no method");
    char *same[256];
    int sames = 0;
    Bytes aliases = exists("level/level/same_output_methods.txt")
        ? read_file(&a, "level/level/same_output_methods.txt")
        : (Bytes){0};
    char *text = fmt(&a, "%.*s", (int)aliases.size, aliases.data), *word;
    for (word = strsep(&text, " \n\t"); word && sames < 256; word = strsep(&text, " \n\t")) {
        if (*word) {
            same[sames++] = word;
        }
    }
    f64 budget = strtod(env("HERCULEX_FETCH_AHEAD_GIB", "200"), NULL) * GB;
    f64 idle = strtod(env("HERCULEX_FETCH_IDLE_HOURS", "1.5"), NULL) * 3600, last_new = now();
    const char *pieces = fmt(&a, "%s/pieces", g.base);
    const char *nodes = fmt(&a, "%s/unwrapping/nodes/%s/%s/nodes", OUTPUTS, g.sample, g.scan);
    char *fetched[4096];
    int done = 0;
    for (;;) {
        u64 mark = a.used;
        int all = 0, count = 0;
        char **units = units_in(&a, pieces, &all),
             **finished = push(&a, (all + 1) * sizeof(char *));
        for (int k = 0; k < all; k++) {
            if (unit_finished(&a, fmt(&a, "%s/%s", pieces, units[k]), method, same, sames)) {
                finished[count++] = units[k];
            }
        }
        for (int k = 0; k < count; k++) {
            if (taken_over(&a, &g, pieces, finished, count)) {
                printf("%s %s: the read has taken over; it fetches from here\n", g.sample, g.scan);
                return 0;
            }
            int seen = 0;
            for (int f = 0; f < done; f++) {
                seen |= !strcmp(fetched[f], finished[k]);
            }
            if (seen) {
                continue;
            }
            i64 stored = 0;
            for (int f = 0; f < count; f++) {
                stored += folder_bytes(&a, store_of(&a, &g, finished[f]));
            }
            if (stored >= budget) {
                printf("%s %s: stores reached %g GiB; the read fetches the rest\n", g.sample,
                    g.scan, budget / GB);
                return 0;
            }
            json_object *record = json_read(fmt(&a, "%s/%s/record.json", pieces, finished[k]));
            json_object *list = jget(record, "pieces");
            int n = json_object_array_length(list), jobs_count = 0;
            Job *jobs = push(&a, (i64)n * sizeof(Job));
            for (int p = 0; p < n; p++) {
                json_object *row = json_object_array_get_idx(list, p), *written = 0;
                if (!json_object_object_get_ex(row, "written", &written) ||
                    !json_object_get_boolean(written)) {
                    continue;
                }
                Job *j = jobs + jobs_count++;
                snprintf(j->unit, sizeof(j->unit), "%s", finished[k]);
                j->piece = ji64(row, "piece");
                j->turn = 1;
                snprintf(j->folder, sizeof(j->folder), "%s/%s/piece-%04d", pieces, finished[k],
                    j->piece);
            }
            json_object_put(record);
            const char *store = store_of(&a, &g, finished[k]);
            struct statvfs fs;
            mkparent(store);
            if (statvfs(fmt(&a, "%.*s", (int)(strrchr(store, '/') - store), store), &fs) == 0 &&
                (u64)fs.f_bavail * fs.f_frsize < FETCH_FREE) {
                printf("%s %s: stopped fetching ahead, the disc is short of room\n", g.sample,
                    finished[k]);
                return 0;
            }
            f64 began = now();
            fetch_store(&g, jobs, jobs_count, store, &a);
            printf("%s %s: CT ready in %.0f s\n", g.sample, finished[k], now() - began);
            fflush(stdout);
            fetched[done++] = strdup(finished[k]);
            last_new = now();
        }
        int node_count = 0, ready = 1;
        char **wanted = units_in(&a, nodes, &node_count);
        for (int k = 0; k < node_count; k++) {
            int seen = 0;
            for (int f = 0; f < done; f++) {
                seen |= !strcmp(fetched[f], wanted[k]);
            }
            ready &= seen;
        }
        if (node_count && ready) {
            printf("%s %s: every unit's CT is ready\n", g.sample, g.scan);
            return 0;
        }
        if (now() - last_new > idle) {
            printf("%s %s: no unit finished for %g h; stopping\n", g.sample, g.scan, idle / 3600);
            return 0;
        }
        a.used = mark;
        sleep(30);
    }
}

int fetch_command(int argc, char **argv) {
    if (argc < 1 || strcmp(argv[0], "run") != 0) {
        return follow_fetch(argc, argv);
    }
    const char *scan = flag(argc, argv, "--scan");
    for (int k = 1; k < argc; k++) {
        if (argv[k][0] == '-') {
            k += strcmp(argv[k], "--band") != 0;
            continue;
        }
        char *one[3] = {argv[k], "--scan", (char *)scan};
        Arena a = arena(GB);
        const char *nodes =
            scan ? fmt(&a, "%s/unwrapping/nodes/%s/%s/nodes", OUTPUTS, argv[k], scan) : 0;
        if (nodes && !exists(nodes)) {
            continue;
        }
        int code = follow_fetch(scan ? 3 : 1, one);
        munmap(a.base, a.size);
        if (code) {
            return code;
        }
    }
    return 0;
}

int sample_command(int argc, char **argv) {
    need(argc >= 4, "usage: ./run.sh sample SCROLL SCAN UNIT PIECE [--turn -1]");
    Arena a = arena(64 * GB);
    Stage g = {0};
    char *open_args[3] = {argv[0], "--scan", argv[1]};
    stage_open(&a, &g, 3, open_args);
    stage_axis(&a, &g);
    const char *turned = flag(argc, argv, "--turn");
    Job j = {.piece = (int)strtol(argv[3], NULL, 10),
        .turn = turned ? (int)strtol(turned, NULL, 10) : 1};
    snprintf(j.unit, sizeof(j.unit), "%s", argv[2]);
    snprintf(j.folder, sizeof(j.folder), "%s/pieces/%s/piece-%04d", g.base, j.unit, j.piece);
    snprintf(j.out, sizeof(j.out), "%s/faces/sample/%s/piece-%04d.png", g.base, j.unit, j.piece);
    const char *store = store_of(&a, &g, j.unit);
    fetch_store(&g, &j, 1, store, &a);
    Bricks bricks;
    bricks_open(&a, &bricks, store);
    Surface s = load_surface(&a, j.folder);
    Readable r = readable(&a, s, g.line, g.rows);
    Stack stack = flatten(&a, &r, j.turn, &bricks, LAYERS);
    i64 n = (i64)stack.height * stack.width;
    u8 *sheet = push(&a, n), *central = push(&a, n);
    across_sheet(stack, s.voxel, sheet, central);
    mkparent(j.out);
    write_png(&a, j.out, sheet, stack.width, stack.height);
    write_png(&a, beside(&a, j.out, "-layer0.png"), central, stack.width, stack.height);
    const char *cube = beside(&a, j.out, "-stack.npy");
    npy_write(cube,
        (Array){.descr = "|u1",
            .ndim = 3,
            .shape = {stack.layers, stack.height, stack.width},
            .data = stack.cube,
            .size = stack.layers * n});
    printf("%s: %d layers of %d x %d\n", cube, stack.layers, stack.height, stack.width);
    return 0;
}

int file_sha256(Arena *a, const char *path, char *hex) {
    if (!exists(path)) {
        return 0;
    }
    u64 mark = a->used;
    Bytes b = read_file(a, path);
    sha256_hex(b.data, b.size, hex);
    a->used = mark;
    return 1;
}

void copy_checkpoint(const char *source, const char *target) {
    FILE *in = fopen(source, "rb");
    need(in != NULL, "cannot read %s", source);
    mkparent(target);
    FILE *out = fopen(target, "wb");
    need(out != NULL, "cannot write %s", target);
    u8 *buffer = malloc(8 << 20);
    need(buffer != NULL, "cannot allocate checkpoint copy buffer");
    size_t count;
    while ((count = fread(buffer, 1, 8 << 20, in))) {
        need(fwrite(buffer, 1, count, out) == count, "cannot write %s", target);
    }
    int failed = ferror(in);
    failed |= fclose(in) != 0;
    failed |= fclose(out) != 0;
    need(!failed, "cannot copy %s", source);
    free(buffer);
}

int weights_command(int argc, char **argv) {
    Arena a = arena(4 * GB);
    const char *state = fmt(&a, "%s/weights/ink/%s/tile-reader.pth", LIBRARY, STOCK_MODEL);
    if (argc == 1 && !strcmp(argv[0], "status")) {
        for (int k = 0; k < PUBLISHED_COUNT; k++) {
            const char *path =
                fmt(&a, "%s/weights/ink/%s/%s", LIBRARY, PUBLISHED[k].name, PUBLISHED[k].file);
            printf("%-36s %-8s %s\n", PUBLISHED[k].title, exists(path) ? "ready" : "missing", path);
        }
        printf("%-36s %-8s %s\n", "stock Reader v3, tile reader state",
            exists(state) ? "ready" : "missing", state);
        return 0;
    }
    int fetch = (argc == 1 || argc == 2) && !strcmp(argv[0], "fetch");
    int import = argc == 3 && !strcmp(argv[0], "import");
    need(fetch || import,
        "usage: ./run.sh weights fetch [reader-v3|reader-pu-bhand|studio|studio-jeff]|import NAME "
        "FILE|status");
    const char *name = import ? argv[1] : argc == 2 ? argv[1] : STOCK_MODEL;
    const Published *p = 0;
    for (int k = 0; k < PUBLISHED_COUNT; k++) {
        p = strcmp(PUBLISHED[k].name, name) ? p : PUBLISHED + k;
    }
    need(p != 0, "%s is not a published ink model", name);
    const char *folder = fmt(&a, "%s/weights/ink/%s", LIBRARY, p->name),
               *checkpoint = fmt(&a, "%s/%s", folder, p->file);
    char hex[65] = {0};
    if (import) {
        need(file_sha256(&a, argv[2], hex), "cannot read %s", argv[2]);
        need(!strcmp(hex, p->sha256), "%s does not match %s's sha256", argv[2], p->title);
        const char *part = fmt(&a, "%s.part", checkpoint);
        copy_checkpoint(argv[2], part);
        need(!rename(part, checkpoint), "cannot install %s", checkpoint);
        printf("%s ready: %s\n", p->title, checkpoint);
        return 0;
    }
    if (!file_sha256(&a, checkpoint, hex) || strcmp(hex, p->sha256) != 0) {
        mkdirs(folder);
        printf("fetching %s\n", p->url);
        fflush(stdout);
        const char *part = fmt(&a, "%s.part", checkpoint);
        FILE *out = fopen(part, "wb");
        need(out != 0, "cannot write %s", part);
        CURL *c = curl_easy_init();
        curl_easy_setopt(c, CURLOPT_URL, p->url);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, out);
        curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
        CURLcode result = curl_easy_perform(c);
        curl_easy_cleanup(c);
        fclose(out);
        need(result == CURLE_OK, "cannot fetch %s: %s", p->title, curl_easy_strerror(result));
        file_sha256(&a, part, hex);
        if (strcmp(hex, p->sha256) != 0) {
            unlink(part);
            die("%s download does not match its sha256; try again", p->title);
        }
        rename(part, checkpoint);
    }
    printf("%s ready: %s\n", p->title, p == PUBLISHED && exists(state) ? state : checkpoint);
    return 0;
}

int models_command(int argc, char **argv) {
    Arena a = arena(4 * GB);
    const char *listed = fmt(&a, "%s/weights/ink/models.json", LIBRARY);
    json_object *models = json_read(listed);
    if (!models) {
        models = json_object_new_object();
    }
    if ((argc == 4 || argc == 5) && !strcmp(argv[0], "add")) {
        const char *name = argv[1], *title = argv[2], *source = argv[3];
        const char *backend = argc == 5 ? argv[4] : "reader";
        need(!published_model(name), "%s is built in", name);
        need(!strcmp(backend, "reader") || !strcmp(backend, "studio"),
            "backend must be reader or studio");
        if (!strcmp(backend, "reader")) {
            Tensor *tensors = pushz(&a, TENSORS * sizeof(Tensor));
            int count = load_tensors(&a, source, tensors);
            tensor_named(tensors, count, "model.depth_fusion.features.0.weight");
            tensor_named(tensors, count, "model.network.task_heads.ink.weight");
        }
        const char *file = !strcmp(backend, "reader") ? "tile-reader.pth" : "model.safetensors";
        const char *target = fmt(&a, "%s/weights/ink/%s/%s", LIBRARY, name, file);
        mkparent(target);
        Bytes b = read_file(&a, source);
        write_file(target, b.data, b.size);
        json_object *entry = json_object_new_object();
        json_object_object_add(entry, "title", json_object_new_string(title));
        json_object_object_add(entry, "file", json_object_new_string(fmt(&a, "%s/%s", name, file)));
        json_object_object_add(entry, "backend", json_object_new_string(backend));
        json_object_object_add(models, name, entry);
        const char *text = json_object_to_json_string_ext(models, JSON_C_TO_STRING_PRETTY);
        char *line = fmt(&a, "%s\n", text);
        write_file(listed, line, strlen(line));
        printf("%s: %s, %s\n", name, title, target);
        return 0;
    }
    need(argc == 0, "usage: ./run.sh models [add NAME TITLE FILE [reader|studio]]");
    const char *chosen = env("HERCULEX_INK_MODEL", STOCK_MODEL);
    int known = 0;
    for (int k = 0; k < PUBLISHED_COUNT; k++) {
        const Published *p = PUBLISHED + k;
        const char *state = fmt(&a, "%s/weights/ink/reader-v3/tile-reader.pth", LIBRARY);
        const char *path = !strcmp(p->name, STOCK_MODEL) && exists(state)
            ? state
            : fmt(&a, "%s/weights/ink/%s/%s", LIBRARY, p->name, p->file);
        printf("%s %-24s %-40s %-8s %-8s %s\n", strcmp(chosen, p->name) ? " " : "*", p->name,
            p->title, p->backend, exists(path) ? "ready" : "missing", path);
        known |= !strcmp(chosen, p->name);
    }
    json_object_object_foreach(models, name, entry) {
        if (published_model(name)) {
            continue;
        }
        const char *path = fmt(&a, "%s/weights/ink/%s", LIBRARY, jstr(entry, "file"));
        json_object *backend = json_object_object_get(entry, "backend");
        printf("%s %-24s %-40s %-8s %-8s %s\n", strcmp(chosen, name) ? " " : "*", name,
            jstr(entry, "title"), backend ? json_object_get_string(backend) : "reader",
            exists(path) ? "ready" : "missing", path);
        known |= !strcmp(chosen, name);
    }
    need(known, "HERCULEX_INK_MODEL=%s is not one of the ink models", chosen);
    return 0;
}
