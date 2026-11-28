#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#if (defined(__APPLE__) && defined(__OBJC__)) || defined(HX_SLICES_CUDA)
#define HX_SLICES_ACCELERATOR 1
int slices_accelerator_available(void);
void *slices_accelerator_open(u64 seed);
void slices_accelerator_close(void *model);
void slices_accelerator_conv(void *model, const float *weights, const float *input, float *out,
    int ci, int co, int side, int kernel, int stride);
#ifdef HX_SLICES_CUDA
void slices_accelerator_noise(void *model, double *out, size_t n);
#endif
#endif

typedef struct {
    Arena memory;
    json_object *header;
    Bytes weights;
    void *accelerator;
} SliceModel;

typedef struct {
    float *data;
    int channels, side;
} SliceTensor;

enum { SP_MARK, SP_NONE, SP_INT, SP_TEXT, SP_GLOBAL, SP_TUPLE, SP_DICT, SP_STORAGE, SP_TENSOR };

typedef struct SlicePickle SlicePickle;
typedef struct SlicePicklePair SlicePicklePair;

struct SlicePickle {
    int kind;
    i64 number;
    const char *text;
    SlicePickle **items;
    size_t count;
    SlicePicklePair *pairs;
};

struct SlicePicklePair {
    const char *name;
    SlicePickle *value;
    SlicePicklePair *next;
};

typedef struct {
    Arena *mem;
    const char *path;
    const u8 *p, *end;
    SlicePickle **stack, **memo;
    size_t top, capacity, memo_size;
} SlicePickleReader;

static const u8 *slices_pickle_read(SlicePickleReader *r, size_t n) {
    need(n <= (size_t)(r->end - r->p), "%s: truncated checkpoint metadata", r->path);
    const u8 *data = r->p;
    r->p += n;
    return data;
}

static u64 slices_pickle_uint(SlicePickleReader *r, int n) {
    const u8 *data = slices_pickle_read(r, n);
    u64 value = 0;
    for (int i = 0; i < n; i++) {
        value |= (u64)data[i] << (8 * i);
    }
    return value;
}

static const char *slices_pickle_string(SlicePickleReader *r, size_t n, int line) {
    if (line) {
        const u8 *end = memchr(r->p, '\n', r->end - r->p);
        need(end != NULL, "%s: truncated checkpoint metadata", r->path);
        n = end - r->p;
    }
    const u8 *bytes = slices_pickle_read(r, n + line);
    need(!memchr(bytes, 0, n), "%s: invalid checkpoint string", r->path);
    char *s = push(r->mem, n + 1);
    memcpy(s, bytes, n);
    s[n] = 0;
    return s;
}

static SlicePickle *slices_pickle_object(SlicePickleReader *r, int kind) {
    SlicePickle *node = pushz(r->mem, sizeof(*node));
    node->kind = kind;
    return node;
}

static SlicePickle *slices_pickle_top(SlicePickleReader *r, size_t depth, int kind) {
    need(depth < r->top, "%s: invalid checkpoint stack", r->path);
    SlicePickle *v = r->stack[r->top - 1 - depth];
    need(kind < 0 || v->kind == kind, "%s: invalid checkpoint value", r->path);
    return v;
}

static size_t slices_pickle_mark(SlicePickleReader *r) {
    size_t at = r->top;
    while (at && r->stack[at - 1]->kind != SP_MARK) {
        at--;
    }
    need(at > 0, "%s: missing checkpoint tuple boundary", r->path);
    return at - 1;
}

static int slices_pickle_is(const SlicePickle *v, int kind, const char *name) {
    return v->kind == kind && !strcmp(v->text, name);
}

static void slices_pickle_set(SlicePickleReader *r, size_t first, size_t target) {
    SlicePickle *dict = r->stack[target];
    need(dict->kind == SP_DICT && (r->top - first) % 2 == 0, "%s: invalid checkpoint dictionary",
        r->path);
    for (size_t i = first; i < r->top; i += 2) {
        SlicePickle *key = r->stack[i];
        need(key->kind == SP_TEXT, "%s: invalid checkpoint key", r->path);
        SlicePicklePair *pair = push(r->mem, sizeof(*pair));
        *pair = (SlicePicklePair){key->text, r->stack[i + 1], dict->pairs};
        dict->pairs = pair;
    }
}

static SlicePickle *slices_pickle_reduce(SlicePickleReader *r) {
    SlicePickle *args = slices_pickle_top(r, 0, SP_TUPLE);
    SlicePickle *call = slices_pickle_top(r, 1, SP_GLOBAL);
    r->top -= 2;
    if (slices_pickle_is(call, SP_GLOBAL, "collections.OrderedDict") && !args->count) {
        return slices_pickle_object(r, SP_DICT);
    }
    need(slices_pickle_is(call, SP_GLOBAL, "torch._utils._rebuild_tensor_v2") ||
            slices_pickle_is(call, SP_GLOBAL, "torch._utils._rebuild_tensor"),
        "%s: unsupported checkpoint constructor %s", r->path, call->text);
    need(args->count >= 4 && args->items[0]->kind == SP_STORAGE && args->items[1]->kind == SP_INT &&
            args->items[2]->kind == SP_TUPLE && args->items[3]->kind == SP_TUPLE,
        "%s: invalid checkpoint tensor", r->path);
    SlicePickle *tensor = slices_pickle_object(r, SP_TENSOR);
    tensor->items = args->items;
    return tensor;
}

static SlicePickle *slices_pickle_decode(Arena *mem, const char *path, Bytes bytes) {
    SlicePickleReader r = {.mem = mem,
        .path = path,
        .p = bytes.data,
        .end = bytes.data + bytes.size,
        .capacity = bytes.size};
    r.stack = push(mem, r.capacity * sizeof(*r.stack));
    r.memo = pushz(mem, r.capacity * sizeof(*r.memo));
    need(slices_pickle_uint(&r, 1) == 0x80, "%s: missing checkpoint protocol", path);
    u64 protocol = slices_pickle_uint(&r, 1);
    need(protocol >= 2 && protocol <= 5, "%s: unsupported checkpoint protocol", path);
    while (r.p < r.end) {
        int op = slices_pickle_uint(&r, 1);
        SlicePickle *v = NULL;
        switch (op) {
        case 0x95: {
            u64 length = slices_pickle_uint(&r, 8);
            need(length <= (u64)(r.end - r.p), "%s: truncated checkpoint frame", path);
            break;
        }
        case '.':
            need(r.top == 1, "%s: incomplete checkpoint stack", path);
            return slices_pickle_top(&r, 0, SP_DICT);
        case '(': v = slices_pickle_object(&r, SP_MARK); break;
        case '}': v = slices_pickle_object(&r, SP_DICT); break;
        case 'N':
        case 0x88:
        case 0x89: v = slices_pickle_object(&r, SP_NONE); break;
        case 'G':
            slices_pickle_read(&r, 8);
            v = slices_pickle_object(&r, SP_NONE);
            break;
        case 'K':
        case 'M':
        case 'J': {
            int n = op == 'K' ? 1 : op == 'M' ? 2 : 4;
            u32 x = slices_pickle_uint(&r, n);
            v = slices_pickle_object(&r, SP_INT);
            v->number = n == 4 ? (i64)(i32)x : x;
            break;
        }
        case 0x8a: {
            u64 n = slices_pickle_uint(&r, 1);
            need(n <= 8, "%s: oversized checkpoint integer", path);
            u64 x = slices_pickle_uint(&r, n);
            if (n && n < 8 && (x >> (8 * n - 1))) {
                x |= UINT64_MAX << (8 * n);
            }
            v = slices_pickle_object(&r, SP_INT);
            memcpy(&v->number, &x, sizeof(x));
            break;
        }
        case 'X':
        case 0x8c:
        case 0x8d: {
            u64 n = slices_pickle_uint(&r, op == 'X' ? 4 : op == 0x8c ? 1 : 8);
            need(n <= (u64)(r.end - r.p), "%s: truncated checkpoint string", path);
            v = slices_pickle_object(&r, SP_TEXT);
            v->text = slices_pickle_string(&r, n, 0);
            break;
        }
        case 'c':
        case 0x93: {
            const char *module, *name;
            if (op == 'c') {
                module = slices_pickle_string(&r, 0, 1);
                name = slices_pickle_string(&r, 0, 1);
            } else {
                name = slices_pickle_top(&r, 0, SP_TEXT)->text;
                module = slices_pickle_top(&r, 1, SP_TEXT)->text;
                r.top -= 2;
            }
            v = slices_pickle_object(&r, SP_GLOBAL);
            v->text = fmt(mem, "%s.%s", module, name);
            break;
        }
        case 'q':
        case 'r':
        case 0x94: {
            u64 index = op == 0x94 ? r.memo_size : slices_pickle_uint(&r, op == 'q' ? 1 : 4);
            need(index < r.capacity, "%s: invalid checkpoint reference", path);
            r.memo_size += r.memo[index] == NULL;
            r.memo[index] = slices_pickle_top(&r, 0, -1);
            break;
        }
        case 'h':
        case 'j': {
            u64 index = slices_pickle_uint(&r, op == 'h' ? 1 : 4);
            need(index < r.capacity && r.memo[index], "%s: invalid checkpoint reference", path);
            v = r.memo[index];
            break;
        }
        case 't':
        case ')':
        case 0x85:
        case 0x86:
        case 0x87: {
            size_t count = op == ')' || op == 't' ? 0 : op - 0x84;
            need(count <= r.top, "%s: invalid checkpoint tuple", path);
            size_t first = op == 't' ? slices_pickle_mark(&r) + 1 : r.top - count;
            v = slices_pickle_object(&r, SP_TUPLE);
            v->count = r.top - first;
            if (v->count) {
                v->items = push(mem, v->count * sizeof(*v->items));
                memcpy(v->items, r.stack + first, v->count * sizeof(*v->items));
            }
            r.top = first - (op == 't');
            break;
        }
        case 'Q': {
            v = slices_pickle_top(&r, 0, SP_TUPLE);
            need(v->count == 5 && slices_pickle_is(v->items[0], SP_TEXT, "storage") &&
                    v->items[1]->kind == SP_GLOBAL && v->items[2]->kind == SP_TEXT &&
                    v->items[3]->kind == SP_TEXT && v->items[4]->kind == SP_INT,
                "%s: invalid checkpoint storage", path);
            SlicePickle *storage = slices_pickle_object(&r, SP_STORAGE);
            storage->items = v->items;
            v = storage;
            r.top--;
            break;
        }
        case 'R': v = slices_pickle_reduce(&r); break;
        case 's':
            need(r.top >= 3, "%s: invalid checkpoint dictionary", path);
            slices_pickle_set(&r, r.top - 2, r.top - 3);
            r.top -= 2;
            break;
        case 'u': {
            size_t mark = slices_pickle_mark(&r);
            need(mark, "%s: invalid checkpoint dictionary", path);
            slices_pickle_set(&r, mark + 1, mark - 1);
            r.top = mark;
            break;
        }
        case 'b':
            slices_pickle_top(&r, 0, SP_DICT);
            slices_pickle_top(&r, 1, SP_DICT);
            r.top--;
            break;
        default: die("%s: unsupported checkpoint opcode 0x%02x", path, op);
        }
        if (v) {
            need(r.top < r.capacity, "%s: oversized checkpoint stack", path);
            r.stack[r.top++] = v;
        }
    }
    die("%s: incomplete checkpoint metadata", path);
    return NULL;
}

static Bytes slices_checkpoint_member(Arena *a, zip_t *zip, const char *name, size_t limit) {
    zip_stat_t st;
    need(!zip_stat(zip, name, 0, &st) && (st.valid & ZIP_STAT_SIZE) && st.size <= limit,
        "missing or oversized checkpoint member %s", name);
    Bytes bytes = {push(a, st.size), st.size};
    zip_file_t *file = zip_fopen(zip, name, 0);
    need(file != NULL, "cannot open checkpoint member %s", name);
    size_t at = 0;
    while (at < bytes.size) {
        i64 n = zip_fread(file, bytes.data + at, bytes.size - at);
        need(n > 0, "cannot read checkpoint member %s", name);
        at += n;
    }
    need(!zip_fclose(file), "cannot close checkpoint member %s", name);
    return bytes;
}

static float slices_checkpoint_float(const u8 *data, int type) {
    if (!type) {
        float out;
        memcpy(&out, data, sizeof(out));
        return out;
    }
    if (type == 1) {
        double out;
        memcpy(&out, data, sizeof(out));
        return out;
    }
    u16 word;
    memcpy(&word, data, sizeof(word));
    if (type == 2) {
        int exponent = (word >> 10) & 31, fraction = word & 1023;
        float out = exponent == 31 ? (fraction ? NAN : INFINITY)
            : exponent             ? ldexpf(1024 + fraction, exponent - 25)
                                   : ldexpf(fraction, -24);
        return word & 32768 ? -out : out;
    }
    u32 bits = (u32)word << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void slices_checkpoint_tensor(
    SliceModel *model, Arena *mem, zip_t *zip, const char *prefix, const SlicePicklePair *pair) {
    const SlicePickle *t = pair->value;
    need(t->kind == SP_TENSOR, "%s: expected a checkpoint tensor", pair->name);
    SlicePickle **storage = t->items[0]->items;
    const char *key = storage[2]->text;
    need(*key && strlen(key) == strspn(key, "0123456789"), "%s: invalid storage key", pair->name);
    const char *formats[] = {
        "torch.FloatStorage", "torch.DoubleStorage", "torch.HalfStorage", "torch.BFloat16Storage"};
    int widths[] = {4, 8, 2, 2};
    int type = 0;
    while (type < 4 && strcmp(storage[1]->text, formats[type]) != 0) {
        type++;
    }
    need(type < 4, "%s: unsupported checkpoint type %s", pair->name, storage[1]->text);
    const size_t limit = 128u << 20;
    const char *member = fmt(mem, "%s/data/%s", prefix, key);
    Bytes raw = slices_checkpoint_member(mem, zip, member, limit);
    i64 length = storage[4]->number, start = t->items[1]->number;
    int width = widths[type];
    need(length > 0 && (u64)length == raw.size / width && raw.size % width == 0 && start >= 0 &&
            start < length,
        "%s: invalid checkpoint storage size", pair->name);
    SlicePickle *shape = t->items[2], *strides = t->items[3];
    int n = (int)shape->count;
    need(shape->count <= 8 && strides->count == shape->count, "%s: invalid checkpoint dimensions",
        pair->name);
    size_t sizes[8], steps[8], count = 1, last = start;
    int contiguous = 1;
    for (int i = n - 1; i >= 0; i--) {
        SlicePickle *dim = shape->items[i], *stride = strides->items[i];
        need(dim->kind == SP_INT && stride->kind == SP_INT && dim->number > 0 &&
                (u64)dim->number <= limit / sizeof(float) / count && stride->number >= 0 &&
                (u64)stride->number <= (u64)length,
            "%s: invalid checkpoint dimensions", pair->name);
        sizes[i] = dim->number;
        steps[i] = stride->number;
        need(!steps[i] || sizes[i] - 1 <= ((size_t)length - 1 - last) / steps[i],
            "%s: checkpoint view exceeds its storage", pair->name);
        last += (sizes[i] - 1) * steps[i];
        contiguous &= sizes[i] == 1 || steps[i] == count;
        count *= sizes[i];
    }
    float *values = push(&model->memory, count * sizeof(*values));
    if (!type && contiguous) {
        memcpy(values, raw.data + start * width, count * sizeof(*values));
    } else {
        for (size_t i = 0; i < count; i++) {
            size_t at = start + i;
            if (!contiguous) {
                at = start;
                size_t position = i;
                for (int d = n - 1; d >= 0; d--) {
                    at += (position % sizes[d]) * steps[d];
                    position /= sizes[d];
                }
            }
            values[i] = slices_checkpoint_float(raw.data + at * width, type);
        }
    }
    size_t offset = (u8 *)values - model->memory.base;
    json_object *spec = json_object_new_object(), *dims = json_object_new_array_ext(n),
                *offsets = json_object_new_array_ext(2);
    json_object_object_add(spec, "dtype", json_object_new_string("F32"));
    json_object_object_add(spec, "shape", dims);
    json_object_object_add(spec, "data_offsets", offsets);
    for (int i = 0; i < n; i++) {
        json_object_array_add(dims, json_object_new_uint64(sizes[i]));
    }
    json_object_array_add(offsets, json_object_new_uint64(offset));
    json_object_array_add(offsets, json_object_new_uint64(offset + count * sizeof(*values)));
    json_object_object_add(model->header, pair->name, spec);
}

static SliceModel slices_checkpoint_open(const char *path) {
    int error;
    zip_t *zip = zip_open(path, ZIP_RDONLY, &error);
    need(zip != NULL, "cannot open checkpoint %s", path);
    const char *pickle = NULL;
    i64 entries = zip_get_num_entries(zip, 0);
    need(entries > 0 && entries <= 8192, "%s: invalid checkpoint directory", path);
    for (i64 i = 0; i < entries; i++) {
        const char *name = zip_get_name(zip, i, 0);
        size_t n = name ? strlen(name) : 0;
        if (n > 9 && !strcmp(name + n - 9, "/data.pkl")) {
            need(!pickle, "%s: checkpoint contains multiple tensor dictionaries", path);
            pickle = name;
        }
    }
    need(pickle != NULL, "%s: checkpoint has no tensor dictionary", path);
    Arena mem = arena((256u << 20) + (16u << 20));
    const char *prefix = fmt(&mem, "%.*s", (int)(strlen(pickle) - 9), pickle);
    const char *byteorder = fmt(&mem, "%s/byteorder", prefix);
    if (zip_name_locate(zip, byteorder, 0) >= 0) {
        Bytes bytes = slices_checkpoint_member(&mem, zip, byteorder, 16);
        need(bytes.size == 6 && !memcmp(bytes.data, "little", 6),
            "%s: unsupported checkpoint byte order", path);
    }
    Bytes pickle_bytes = slices_checkpoint_member(&mem, zip, pickle, 1 << 20);
    SlicePickle *dict = slices_pickle_decode(&mem, path, pickle_bytes);
    for (const SlicePicklePair *pair = dict->pairs; pair; pair = pair->next) {
        if (!strcmp(pair->name, "model")) {
            dict = pair->value;
            break;
        }
    }
    need(
        dict->kind == SP_DICT && dict->pairs, "%s: checkpoint has no model state dictionary", path);
    SliceModel model = {.memory = arena(1u << 30), .header = json_object_new_object()};
    size_t mark = mem.used;
    int count = 0;
    for (const SlicePicklePair *pair = dict->pairs; pair; pair = pair->next) {
        need(++count <= 1024, "%s: too many checkpoint tensors", path);
        if (!json_object_object_get(model.header, pair->name)) {
            slices_checkpoint_tensor(&model, &mem, zip, prefix, pair);
        }
        mem.used = mark;
    }
    model.weights = (Bytes){model.memory.base, model.memory.used};
    munmap(mem.base, mem.size);
    zip_discard(zip);
    return model;
}

static SliceModel slices_model_open(const char *path) {
    struct stat st;
    need(stat(path, &st) == 0 && st.st_size > 8 && (u64)st.st_size < SIZE_MAX - (1 << 20),
        "%s: no readable Synthetic-Slices checkpoint", path);
    u8 signature[4];
    int input = open(path, O_RDONLY);
    need(input >= 0, "cannot open %s: %s", path, strerror(errno));
    ssize_t n = read(input, signature, sizeof(signature));
    close(input);
    need(n == sizeof(signature), "cannot read checkpoint %s", path);
    if (!memcmp(signature, "PK\003\004", sizeof(signature))) {
        return slices_checkpoint_open(path);
    }
    SliceModel model = {.memory = arena((size_t)st.st_size + (1 << 20))};
    Bytes b = read_file(&model.memory, path);
    u64 length;
    memcpy(&length, b.data, 8);
    need(length <= b.size - 8 && length < INT_MAX, "%s: invalid safetensors header", path);
    json_tokener *tok = json_tokener_new();
    need(tok != NULL, "cannot allocate safetensors parser");
    model.header = json_tokener_parse_ex(tok, (char *)b.data + 8, (int)length);
    need(model.header && json_object_get_type(model.header) == json_type_object &&
            json_tokener_get_error(tok) == json_tokener_success,
        "%s is not a safetensors checkpoint", path);
    json_tokener_free(tok);
    model.weights = (Bytes){b.data + 8 + length, b.size - 8 - length};
    return model;
}

static void slices_model_close(SliceModel *model) {
#ifdef HX_SLICES_ACCELERATOR
    if (model->accelerator) {
        slices_accelerator_close(model->accelerator);
    }
#endif
    json_object_put(model->header);
    munmap(model->memory.base, model->memory.size);
}

static const float *slices_parameter(
    const SliceModel *model, const char *layer, const char *name, const int *shape, int n) {
    char key[96];
    snprintf(key, sizeof(key), "%s.%s", layer, name);
    json_object *spec, *types, *dims, *offsets;
    need(json_object_object_get_ex(model->header, key, &spec) &&
            json_object_object_get_ex(spec, "dtype", &types) &&
            !strcmp(json_object_get_string(types), "F32") &&
            json_object_object_get_ex(spec, "shape", &dims) &&
            json_object_array_length(dims) == (size_t)n &&
            json_object_object_get_ex(spec, "data_offsets", &offsets) &&
            json_object_array_length(offsets) == 2,
        "%s: expected float32 parameter", key);
    size_t bytes = sizeof(float);
    for (int i = 0; i < n; i++) {
        need(json_object_get_int(json_object_array_get_idx(dims, i)) == shape[i],
            "%s: unexpected dimension %d", key, i);
        bytes *= shape[i];
    }
    u64 start = json_object_get_uint64(json_object_array_get_idx(offsets, 0));
    u64 end = json_object_get_uint64(json_object_array_get_idx(offsets, 1));
    need(start <= end && end <= model->weights.size && end - start == bytes &&
            ((size_t)model->weights.data + start) % _Alignof(float) == 0,
        "%s: invalid checkpoint offsets", key);
    return (const float *)(model->weights.data + start);
}

static SliceTensor slices_tensor(Arena *mem, int channels, int side) {
    return (SliceTensor){push(mem, (size_t)channels * side * side * sizeof(float)), channels, side};
}

static void slices_model_linear(const SliceModel *model, const char *layer, const float *input,
    int inputs, float *out, int outputs) {
    const float *weights = slices_parameter(model, layer, "weight", (int[]){outputs, inputs}, 2);
    const float *bias = slices_parameter(model, layer, "bias", &outputs, 1);
    memcpy(out, bias, (size_t)outputs * sizeof(*out));
    cblas_sgemv(
        CblasRowMajor, CblasNoTrans, outputs, inputs, 1, weights, inputs, input, 1, 1, out, 1);
}

static void slices_conv(Arena *mem, const SliceModel *model, const char *layer, SliceTensor input,
    SliceTensor output, int kernel, int stride) {
    int in = input.channels, out = output.channels;
    int size = input.side, side = output.side, pixels = side * side;
    const float *weights =
        slices_parameter(model, layer, "weight", (int[]){out, in, kernel, kernel}, 4);
    const float *bias = slices_parameter(model, layer, "bias", &out, 1);
#ifdef HX_SLICES_ACCELERATOR
    if (model->accelerator) {
        slices_accelerator_conv(
            model->accelerator, weights, input.data, output.data, in, out, size, kernel, stride);
    } else
#endif
        if (kernel == 1) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out, pixels, in, 1, weights, in,
            input.data, pixels, 0, output.data, pixels);
    } else {
        size_t used = mem->used;
        int cols = in * kernel * kernel, pad = kernel / 2;
        int block = slices_min(1024, pixels);
        float *matrix = push(mem, (size_t)cols * block * sizeof(*matrix));
        for (int start = 0; start < pixels; start += block) {
            int count = slices_min(block, pixels - start);
            int col = 0;
            for (int c = 0; c < in; c++) {
                for (int ky = 0; ky < kernel; ky++) {
                    for (int kx = 0; kx < kernel; kx++, col++) {
                        const float *plane = input.data + (size_t)c * size * size;
                        float *row = matrix + (size_t)col * block;
                        int remaining = count;
                        int y = start / side, x = start % side;
                        while (remaining > 0) {
                            int run = slices_min(remaining, side - x);
                            int iy = y * stride + ky - pad, ix = x * stride + kx - pad;
                            if (iy < 0 || iy >= size) {
                                memset(row, 0, (size_t)run * sizeof(*row));
                            } else if (stride == 1 && ix >= 0 && ix + run <= size) {
                                memcpy(row, plane + (size_t)iy * size + ix,
                                    (size_t)run * sizeof(*row));
                            } else {
                                for (int i = 0; i < run; i++) {
                                    int at = ix + i * stride;
                                    row[i] =
                                        at >= 0 && at < size ? plane[(size_t)iy * size + at] : 0;
                                }
                            }
                            row += run;
                            remaining -= run;
                            x = 0;
                            y++;
                        }
                    }
                }
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out, count, cols, 1, weights,
                cols, matrix, block, 0, output.data + start, pixels);
        }
        mem->used = used;
    }
    for (int c = 0; c < out; c++) {
        float *plane = output.data + (size_t)c * pixels;
        for (int i = 0; i < pixels; i++) {
            plane[i] += bias[c];
        }
    }
}

static void slices_group_norm(
    const SliceModel *model, const char *layer, SliceTensor input, SliceTensor output) {
    int channels = input.channels, pixels = input.side * input.side;
    const float *weights = slices_parameter(model, layer, "weight", &channels, 1);
    const float *bias = slices_parameter(model, layer, "bias", &channels, 1);
    int depth = channels / 32;
    for (int g = 0; g < 32; g++) {
        int first = g * depth, last = first + depth;
        const float *base = input.data + (size_t)first * pixels;
        double sum = 0, square = 0;
        for (int i = 0; i < pixels * depth; i++) {
            double value = base[i];
            sum += value;
            square += value * value;
        }
        double mean = sum / (pixels * depth);
        double var = fmax(0, square / (pixels * depth) - mean * mean);
        float inv = 1.0f / sqrtf((float)var + 1e-5f);
        for (int c = first; c < last; c++) {
            const float *in = input.data + (size_t)c * pixels;
            float *out = output.data + (size_t)c * pixels;
            float scale = weights[c] * inv, add = bias[c] - (float)mean * scale;
            for (int i = 0; i < pixels; i++) {
                out[i] = scale * in[i] + add;
            }
        }
    }
}

static void slices_silu(float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] *= 1.0f / (1.0f + expf(-data[i]));
    }
}

static void slices_attention(
    Arena *mem, const SliceModel *model, const char *layer, SliceTensor tensor) {
    char key[80];
    size_t used = mem->used;
    int c = tensor.channels, side = tensor.side, pixels = side * side, dim = c / (c / 64);
    SliceTensor norm = slices_tensor(mem, c, side);
    SliceTensor qkv = slices_tensor(mem, c * 3, side);
    snprintf(key, sizeof(key), "%s.na", layer);
    slices_group_norm(model, key, tensor, norm);
    snprintf(key, sizeof(key), "%s.qkv", layer);
    slices_conv(mem, model, key, norm, qkv, 1, 1);
    float *scores = push(mem, (size_t)pixels * pixels * sizeof(*scores));
    for (int head = 0; head < c / dim; head++) {
        const float *q = qkv.data + (size_t)head * dim * pixels;
        const float *k = q + (size_t)c * pixels;
        const float *v = k + (size_t)c * pixels;
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, pixels, pixels, dim, 1.0f / sqrtf(dim),
            q, pixels, k, pixels, 0, scores, pixels);
        for (int i = 0; i < pixels; i++) {
            float *row = scores + (size_t)i * pixels;
            float max = row[0];
            for (int j = 1; j < pixels; j++) {
                max = fmaxf(row[j], max);
            }
            float sum = 0;
            for (int j = 0; j < pixels; j++) {
                row[j] = expf(row[j] - max);
                sum += row[j];
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < pixels; j++) {
                row[j] *= inv;
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, dim, pixels, pixels, 1, v, pixels,
            scores, pixels, 0, norm.data + (size_t)head * dim * pixels, pixels);
    }
    SliceTensor attn = slices_tensor(mem, c, side);
    snprintf(key, sizeof(key), "%s.pr", layer);
    slices_conv(mem, model, key, norm, attn, 1, 1);
    for (size_t i = 0; i < (size_t)c * pixels; i++) {
        tensor.data[i] += attn.data[i];
    }
    mem->used = used;
}

static void slices_block(Arena *mem, const SliceModel *model, const char *layer, SliceTensor input,
    SliceTensor output, const float *emb, int attention) {
    size_t used = mem->used;
    int side = input.side, ci = input.channels, co = output.channels, pixels = side * side;
    SliceTensor norm = slices_tensor(mem, ci, side), conv = slices_tensor(mem, co, side);
    char key[80];
    snprintf(key, sizeof(key), "%s.n1", layer);
    slices_group_norm(model, key, input, norm);
    slices_silu(norm.data, (size_t)ci * pixels);
    snprintf(key, sizeof(key), "%s.c1", layer);
    slices_conv(mem, model, key, norm, conv, 3, 1);
    float affine[512];
    snprintf(key, sizeof(key), "%s.e", layer);
    slices_model_linear(model, key, emb, 512, affine, 2 * co);
    snprintf(key, sizeof(key), "%s.n2", layer);
    slices_group_norm(model, key, conv, conv);
    for (int c = 0; c < co; c++) {
        float *plane = conv.data + (size_t)c * pixels;
        float scale = 1 + affine[c], bias = affine[co + c];
        for (int p = 0; p < pixels; p++) {
            plane[p] = plane[p] * scale + bias;
        }
    }
    slices_silu(conv.data, (size_t)co * pixels);
    snprintf(key, sizeof(key), "%s.c2", layer);
    slices_conv(mem, model, key, conv, output, 3, 1);
    SliceTensor residual = input;
    if (ci != co) {
        residual = slices_tensor(mem, co, side);
        snprintf(key, sizeof(key), "%s.sk", layer);
        slices_conv(mem, model, key, input, residual, 1, 1);
    }
    for (size_t i = 0; i < (size_t)co * pixels; i++) {
        output.data[i] += residual.data[i];
    }
    mem->used = used;
    if (attention) {
        slices_attention(mem, model, layer, output);
    }
}

static void slices_embedding(const SliceModel *model, float log_sigma, i64 style, float *out) {
    float time[256], hidden[512];
    for (int i = 0; i < 128; i++) {
        float phase = log_sigma * 1000 * expf(-(float)log(10000) * i / 128) / 4;
        time[i] = sinf(phase);
        time[i + 128] = cosf(phase);
    }
    slices_model_linear(model, "f.0", time, 256, hidden, 512);
    slices_silu(hidden, 512);
    slices_model_linear(model, "f.2", hidden, 512, out, 512);
    const float *styles = slices_parameter(model, "sty", "weight", (int[]){11, 512}, 2);
    const float *axes = slices_parameter(model, "axe", "weight", (int[]){3, 512}, 2);
    for (int i = 0; i < 512; i++) {
        out[i] += styles[style * 512 + i];
        out[i] += axes[i];
    }
}

static void slices_forward(Arena *mem, const SliceModel *model, const float *image,
    const float *condition, i64 style, float log_sigma, float *output) {
    const int widths[] = {64, 128, 192, 256, 256};
    float emb[512];
    slices_embedding(model, log_sigma, style, emb);
    SliceTensor input = slices_tensor(mem, 4, SLICES_WINDOW);
    memcpy(input.data, image, SLICES_PIXELS * sizeof(float));
    memcpy(input.data + SLICES_PIXELS, condition, SLICES_PIXELS * 3 * sizeof(float));
    SliceTensor tensor = slices_tensor(mem, widths[0], SLICES_WINDOW);
    slices_conv(mem, model, "cin", input, tensor, 3, 1);
    SliceTensor skip[15] = {tensor};
    int count = 1, module = 0, side = SLICES_WINDOW;
    char name[40];
    for (int level = 0; level < 5; level++) {
        for (int block = 0; block < 2; block++) {
            SliceTensor next = slices_tensor(mem, widths[level], side);
            snprintf(name, sizeof(name), "down.%d", module++);
            slices_block(mem, model, name, tensor, next, emb, side == 32 || side == 16);
            tensor = skip[count++] = next;
        }
        if (level < 4) {
            side /= 2;
            SliceTensor next = slices_tensor(mem, tensor.channels, side);
            snprintf(name, sizeof(name), "down.%d", module++);
            slices_conv(mem, model, name, tensor, next, 3, 2);
            tensor = skip[count++] = next;
        }
    }
    SliceTensor mid = slices_tensor(mem, tensor.channels, side);
    slices_block(mem, model, "mid.0", tensor, mid, emb, 1);
    tensor = slices_tensor(mem, mid.channels, side);
    slices_block(mem, model, "mid.1", mid, tensor, emb, 0);
    module = 0;
    for (int level = 4; level >= 0; level--) {
        for (int block = 0; block < 3; block++) {
            SliceTensor next = slices_tensor(mem, widths[level], side);
            size_t used = mem->used;
            SliceTensor residual = skip[--count];
            SliceTensor joined = slices_tensor(mem, tensor.channels + residual.channels, side);
            size_t plane = (size_t)side * side * sizeof(float);
            memcpy(joined.data, tensor.data, plane * tensor.channels);
            memcpy(joined.data + (size_t)tensor.channels * side * side, residual.data,
                plane * residual.channels);
            snprintf(name, sizeof(name), "up.%d", module++);
            slices_block(mem, model, name, joined, next, emb, side == 32 || side == 16);
            mem->used = used;
            tensor = next;
        }
        if (level > 0) {
            side *= 2;
            SliceTensor next = slices_tensor(mem, tensor.channels, side);
            size_t used = mem->used;
            SliceTensor spread = slices_tensor(mem, tensor.channels, side);
            for (int c = 0; c < tensor.channels; c++) {
                const float *from = tensor.data + (size_t)c * tensor.side * tensor.side;
                float *to = spread.data + (size_t)c * side * side;
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        to[(size_t)y * side + x] = from[(size_t)(y / 2) * tensor.side + x / 2];
                    }
                }
            }
            module++;
            snprintf(name, sizeof(name), "up.%d", module++);
            slices_conv(mem, model, name, spread, next, 3, 1);
            mem->used = used;
            tensor = next;
        }
    }
    slices_group_norm(model, "out.0", tensor, tensor);
    slices_silu(tensor.data, (size_t)tensor.channels * SLICES_PIXELS);
    slices_conv(mem, model, "out.2", tensor, (SliceTensor){output, 1, SLICES_WINDOW}, 3, 1);
}

static void slices_model_network(void *context, int n, const float *image, const float *condition,
    const i64 *style, float log_sigma, float *out) {
    const SliceModel *model = context;
    Arena mem = arena(GB);
    for (int i = 0; i < n; i++) {
        size_t at = (size_t)i * SLICES_PIXELS;
        slices_forward(&mem, model, image + at, condition + 3 * at, style[i], log_sigma, out + at);
        mem.used = 0;
    }
    munmap(mem.base, mem.size);
}
