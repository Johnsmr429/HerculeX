#include "../base.h"
#include <dirent.h>
#include <sys/wait.h>

#include "../storage/tile_index.h"

#define PIXEL_MM 0.01
#define TILE 512

static char temp[] = "/tmp/hx-tiles.XXXXXX";
static const char *binary;

static void clean(void) {
    char *argv[] = {"rm", "-rf", temp, NULL};
    pid_t pid = fork();
    if (!pid) {
        execvp("rm", argv);
        _exit(127);
    }
    if (pid > 0)
        waitpid(pid, NULL, 0);
}

static u8 next_byte(Pcg64 *r, u32 *buffer, int *left) {
    if (!*left) {
        *buffer = pcg64_next32(r);
        *left = 3;
    } else {
        *buffer >>= 8;
        (*left)--;
    }
    return (u8)*buffer;
}

static u8 *integers(Arena *a, Pcg64 *r, int low, int high, int h, int w) {
    u8 *out = push(a, (size_t)h * w), range = (u8)(high - 1 - low);
    u32 buffer = 0;
    int left = 0;
    for (size_t i = 0; i < (size_t)h * w; i++) {
        if (range == 0xff) {
            out[i] = (u8)(low + next_byte(r, &buffer, &left));
            continue;
        }
        u8 span = range + 1;
        u16 m = (u16)(next_byte(r, &buffer, &left) * span);
        u8 leftover = (u8)m;
        if (leftover < span) {
            u8 threshold = (u8)((u8)(UINT8_MAX - range) % span);
            while (leftover < threshold) {
                m = (u16)(next_byte(r, &buffer, &left) * span);
                leftover = (u8)m;
            }
        }
        out[i] = (u8)(low + (m >> 8));
    }
    return out;
}

static void save_png(Arena *a, const char *path, const u8 *grey, int w, int h) {
    Bytes png = png_encode(a, grey, w, h, 6);
    write_file(path, png.data, png.size);
}

static json_object *laid_piece(
    Arena *a, const char *faces, int n, const u8 *pixels, int h, int w, double x_mm) {
    for (int k = 0; k < 2; k++)
        save_png(a, fmt(a, "%s/unit/piece-%04d%s.png", faces, n, k ? "-layer0" : ""), pixels, w, h);
    json_object *row = json_object_new_object();
    json_object_object_add(row, "unit", json_object_new_string("unit"));
    json_object_object_add(row, "piece", json_object_new_int(n));
    json_object_object_add(row, "area", json_object_new_double((double)h * w));
    json_object_object_add(row, "um", json_object_new_double(PIXEL_MM));
    json_object_object_add(row, "stride", json_object_new_int(1));
    json_object_object_add(row, "x_mm", json_object_new_double(x_mm));
    json_object_object_add(row, "y_mm", json_object_new_double(0.02));
    json_object_object_add(row, "width_mm", json_object_new_double(w * PIXEL_MM));
    json_object_object_add(row, "height_mm", json_object_new_double(h * PIXEL_MM));
    json_object_object_add(row, "flip_rows", json_object_new_boolean(0));
    json_object_object_add(row, "flip_cols", json_object_new_boolean(0));
    json_object_object_add(row, "group", json_object_new_int(n));
    return row;
}

static void write_layout(const char *path, json_object *pieces) {
    json_object *layout = json_object_new_object();
    json_object_object_add(layout, "pieces", pieces);
    const char *text = json_object_to_json_string_ext(layout, JSON_C_TO_STRING_PLAIN);
    write_file(path, text, strlen(text));
    json_object_put(layout);
}

static int run(char *const *args, int workers, const char *outputs) {
    pid_t pid = fork();
    need(pid >= 0, "cannot fork %s", args[1]);
    if (!pid) {
        if (outputs) {
            setenv("HERCULEX_OUTPUTS", outputs, 1);
            setenv("HERCULEX_INK_MODEL", "reader-v3", 1);
            setenv("HERCULEX_FACES", "level-0", 1);
            char count[8];
            snprintf(count, sizeof(count), "%d", workers);
            setenv("HERCULEX_COMPOSE_PROCESSES", count, 1);
        }
        execv(binary, args);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid, "lost %s", args[1]);
    return WIFEXITED(code) ? WEXITSTATUS(code) : 128;
}

static int compose(const char *outputs, const char *page, const char *layers, int workers,
    const char *one, const char *two, const char *three, const char *four, const char *five,
    const char *six) {
    char *args[] = {(char *)binary, "viewer.sheet_tiles", "scroll", "scan", "0.01", (char *)page,
        (char *)layers, (char *)one, (char *)two, (char *)three, (char *)four, (char *)five,
        (char *)six, NULL};
    return run(args, workers, outputs);
}

typedef struct {
    TileIndex index;
    char folder[PATH_MAX];
} Store;

static void store_open(Store *s, const char *folder) {
    snprintf(s->folder, sizeof(s->folder), "%s", folder);
    int directory = open(folder, O_RDONLY | O_DIRECTORY);
    need(directory >= 0 && tile_index_open(&s->index, directory) == 1, "cannot open %s", folder);
    close(directory);
}

static int store_place(
    const Store *s, const char *layer, int level, int c, int r, uint64_t *offset, uint32_t *size) {
    json_object *entry = json_object_object_get(s->index.layers, layer);
    TileSpan span;
    if (!entry || !tile_index_level(&s->index, entry, level, &span) || c < span.x || r < span.y ||
        c - span.x >= span.w || r - span.y >= span.h)
        return 0;
    size_t row = span.first + (size_t)(r - span.y) * span.w + (c - span.x);
    need(tile_index_blob(&s->index, row, offset, size), "damaged tile table");
    return *size > 0;
}

static Bytes store_tile(Arena *a, const Store *s, const char *layer, int level, int c, int r) {
    uint64_t offset;
    uint32_t size;
    if (!store_place(s, layer, level, c, r, &offset, &size))
        return (Bytes){0};
    Bytes tile = {.data = push(a, size), .size = size};
    need(transfer_at(s->index.pack, tile.data, size, offset, 0), "cannot read a tile");
    return tile;
}

static u8 *decoded(Arena *a, Bytes tile, int *w, int *h) {
    return tile.data[0] == 0x89 ? png_decode(a, tile, w, h) : jpeg_decode(a, tile, w, h);
}

static int levels_of(const Store *s, const char *layer) {
    json_object *entry = json_object_object_get(s->index.layers, layer);
    return (int)json_object_array_length(json_object_object_get(entry, "grid"));
}

typedef void (*Visit)(
    Arena *a, const Store *s, const char *layer, int level, int c, int r, void *context);

static void each_tile(Arena *a, const Store *s, const char *layer, Visit visit, void *context) {
    json_object *entry = json_object_object_get(s->index.layers, layer);
    for (int level = 0; level < levels_of(s, layer); level++) {
        TileSpan span;
        need(tile_index_level(&s->index, entry, level, &span), "bad level %d", level);
        for (int x = 0; x < span.w; x++) {
            for (int y = 0; y < span.h; y++) {
                uint64_t offset;
                uint32_t size;
                if (store_place(s, layer, level, span.x + x, span.y + y, &offset, &size))
                    visit(a, s, layer, level, span.x + x, span.y + y, context);
            }
        }
    }
}

static int by_name(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int layer_names(const Store *s, const char **names) {
    int n = 0;
    json_object_object_foreach(s->index.layers, key, value) {
        (void)value;
        names[n++] = key;
    }
    qsort(names, n, sizeof(*names), by_name);
    return n;
}

typedef struct {
    EVP_MD_CTX *pixels, *raw;
} Hashes;

static void hash_tile(
    Arena *a, const Store *s, const char *layer, int level, int c, int r, void *context) {
    Hashes *h = context;
    size_t saved = a->used;
    Bytes tile = store_tile(a, s, layer, level, c, r);
    int w, rows;
    u8 *grey = decoded(a, tile, &w, &rows);
    EVP_DigestUpdate(h->pixels, grey, (size_t)w * rows);
    if (!level)
        EVP_DigestUpdate(h->raw, tile.data, tile.size);
    a->used = saved;
}

static void hex(const u8 digest[32], char out[65]) {
    for (int i = 0; i < 32; i++)
        snprintf(out + 2 * i, 3, "%02x", digest[i]);
}

typedef struct {
    const Store *other;
    int differ;
} Same;

static void same_tile_as(
    Arena *a, const Store *s, const char *layer, int level, int c, int r, void *context) {
    Same *same = context;
    size_t saved = a->used;
    Bytes mine = store_tile(a, s, layer, level, c, r);
    Bytes theirs = store_tile(a, same->other, layer, level, c, r);
    same->differ |= mine.size != theirs.size || memcmp(mine.data, theirs.data, mine.size);
    a->used = saved;
}

static void py_text(Arena *a, char **out, const char *text) {
    *out = fmt(a, "%s\"", *out);
    for (const u8 *p = (const u8 *)text; *p;) {
        u32 code = *p;
        int extra = code < 0x80 ? 0 : code < 0xe0 ? 1 : code < 0xf0 ? 2 : 3;
        code &= extra ? 0x3f >> extra : 0x7f;
        p++;
        for (int i = 0; i < extra; i++)
            code = code << 6 | (*p++ & 0x3f);
        const char *escape = code == '"' ? "\\\""
            : code == '\\'               ? "\\\\"
            : code == '\n'               ? "\\n"
            : code == '\r'               ? "\\r"
            : code == '\t'               ? "\\t"
            : code == '\b'               ? "\\b"
            : code == '\f'               ? "\\f"
                                         : NULL;
        if (escape)
            *out = fmt(a, "%s%s", *out, escape);
        else if (code < 0x20 || (code >= 0x7f && code < 0x10000))
            *out = fmt(a, "%s\\u%04x", *out, code);
        else if (code >= 0x10000)
            *out = fmt(a, "%s\\u%04x\\u%04x", *out, 0xd800 + ((code - 0x10000) >> 10),
                0xdc00 + ((code - 0x10000) & 0x3ff));
        else
            *out = fmt(a, "%s%c", *out, (char)code);
    }
    *out = fmt(a, "%s\"", *out);
}

static void py_dump(Arena *a, char **out, json_object *value) {
    switch (json_object_get_type(value)) {
    case json_type_null: *out = fmt(a, "%snull", *out); return;
    case json_type_boolean:
        *out = fmt(a, "%s%s", *out, json_object_get_boolean(value) ? "true" : "false");
        return;
    case json_type_int:
        *out = fmt(a, "%s%lld", *out, (long long)json_object_get_int64(value));
        return;
    case json_type_double:
        *out = fmt(a, "%s%s", *out, python_float(a, json_object_get_double(value)));
        return;
    case json_type_string: py_text(a, out, json_object_get_string(value)); return;
    case json_type_array:
        *out = fmt(a, "%s[", *out);
        for (size_t i = 0; i < json_object_array_length(value); i++) {
            if (i)
                *out = fmt(a, "%s,", *out);
            py_dump(a, out, json_object_array_get_idx(value, i));
        }
        *out = fmt(a, "%s]", *out);
        return;
    case json_type_object: {
        int n = json_object_object_length(value), k = 0;
        const char **keys = push(a, (n ? n : 1) * sizeof(*keys));
        json_object_object_foreach(value, key, item) {
            (void)item;
            keys[k++] = key;
        }
        qsort(keys, n, sizeof(*keys), by_name);
        *out = fmt(a, "%s{", *out);
        for (int i = 0; i < n; i++) {
            if (i)
                *out = fmt(a, "%s,", *out);
            py_text(a, out, keys[i]);
            *out = fmt(a, "%s:", *out);
            py_dump(a, out, json_object_object_get(value, keys[i]));
        }
        *out = fmt(a, "%s}", *out);
        return;
    }
    }
}

static const char *canonical(Arena *a, json_object *value) {
    char *out = fmt(a, "%s", "");
    py_dump(a, &out, value);
    return out;
}

static const char *canonical_hash(Arena *a, json_object *value) {
    const char *text = canonical(a, value);
    char *digest = push(a, 65);
    sha256_hex(text, strlen(text), digest);
    return digest;
}

static int entries(const char *folder, const char *prefix, const char *suffix, char names[][96]) {
    DIR *d = opendir(folder);
    need(d != NULL, "cannot list %s", folder);
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t len = strlen(e->d_name), tail = strlen(suffix);
        if (e->d_name[0] == '.' || strncmp(e->d_name, prefix, strlen(prefix)) || len < tail ||
            strcmp(e->d_name + len - tail, suffix))
            continue;
        need(n < 8, "too many files in %s", folder);
        snprintf(names[n++], 96, "%s", e->d_name);
    }
    closedir(d);
    qsort(names, n, 96, (int (*)(const void *, const void *))strcmp);
    return n;
}

static void one_pack(Arena *a, const char *outputs) {
    Pcg64 r = pcg64(3);
    const char *base = fmt(a, "%s/one/unwrapping/level/scroll/scan", outputs);
    const char *faces = fmt(a, "%s/faces/level-0", base);
    const char *sheet = fmt(a, "%s/one/website/sheets/sheet", outputs);
    u8 *first = integers(a, &r, 40, 200, 700, 900), *second = integers(a, &r, 40, 200, 300, 300);
    json_object *pieces = json_object_new_array();
    json_object_array_add(pieces, laid_piece(a, faces, 0, first, 700, 900, 0.02));
    json_object_array_add(
        pieces, laid_piece(a, faces, 1, second, 300, 300, 2 * TILE * PIXEL_MM + 1.0));
    write_layout(fmt(a, "%s/sheet/layout.json", base), pieces);
    need(!compose(fmt(a, "%s/one", outputs), "sheet", "sheet,layer0,layer-3", 3, NULL, NULL, NULL,
             NULL, NULL, NULL),
        "the sheet did not compose");
    Store s;
    store_open(&s, sheet);
    json_object *pack = json_object_object_get(s.index.document, "pack");
    char names[8][96] = {{0}};
    int n = entries(sheet, "", "", names);
    need(n == 3 && !strcmp(names[0], "layers.json") &&
            !strcmp(names[1], json_object_get_string(pack)) && !strcmp(names[2], "tiles.index"),
        "a sheet keeps its voxels in one pack beside its index and layers");
    need(json_object_object_length(s.index.layers) == 2 &&
            json_object_object_get(s.index.layers, "sheet") &&
            json_object_object_get(s.index.layers, "layer0"),
        "the layers are the sheet and layer 0, and an absent layer is skipped");
    json_object *record =
        json_object_array_get_idx(json_object_from_file(fmt(a, "%s/layers.json", sheet)), 0);
    need(json_object_get_boolean(json_object_object_get(record, "raw")) &&
            json_object_get_int(json_object_object_get(record, "levels")) == levels_of(&s, "sheet"),
        "the layer record names a raw layer and every level");
    u8 *whole = push(a, (size_t)2 * TILE * 4 * TILE);
    memset(whole, 0, (size_t)2 * TILE * 4 * TILE);
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 2; row++) {
            Bytes tile = store_tile(a, &s, "sheet", 0, c, row);
            if (!tile.size)
                continue;
            int w, h;
            u8 *grey = decoded(a, tile, &w, &h);
            need(w == TILE && h == TILE, "a tile is %dx%d", w, h);
            for (int y = 0; y < TILE; y++)
                memcpy(
                    whole + (size_t)(row * TILE + y) * 4 * TILE + c * TILE, grey + y * TILE, TILE);
        }
    }
    int kept = 1, blank = 1;
    for (int y = 0; y < 700; y++)
        kept &= !memcmp(whole + (size_t)y * 4 * TILE, first + (size_t)y * 900, 900);
    for (int y = 0; y < 2 * TILE; y++)
        for (int x = 900; x < 2 * TILE; x++)
            blank &= !whole[(size_t)y * 4 * TILE + x];
    need(kept && blank, "every voxel reaches the sheet and nothing lands between pieces");
    Bytes png = store_tile(a, &s, "sheet", 0, 1, 1), jpeg = store_tile(a, &s, "sheet", 1, 0, 0);
    need(png.size > 4 && !memcmp(png.data, "\x89PNG", 4) && jpeg.size > 2 && jpeg.data[0] == 0xff &&
            jpeg.data[1] == 0xd8,
        "the first level is PNG and the next are JPEG");
    need(!store_tile(a, &s, "sheet", 0, 2, 1).size && !store_tile(a, &s, "sheet", 0, 9, 9).size,
        "empty and outside tiles are absent");
    uint64_t at[2];
    uint32_t size[2];
    need(store_place(&s, "layer0", 0, 0, 0, &at[0], &size[0]) &&
            store_place(&s, "sheet", 0, 0, 0, &at[1], &size[1]) && at[0] == at[1] &&
            size[0] == size[1],
        "identical tiles share one place in the pack");
    uint64_t total = 0;
    json_object *entry = json_object_object_get(s.index.layers, "sheet");
    for (int level = 0; level < levels_of(&s, "sheet"); level++) {
        TileSpan span;
        need(tile_index_level(&s.index, entry, level, &span), "bad level");
        for (int k = 0; k < span.w * span.h; k++) {
            uint64_t offset;
            uint32_t bytes;
            need(tile_index_blob(&s.index, span.first + k, &offset, &bytes), "bad cell");
            total += bytes;
        }
    }
    need(s.index.pack_bytes < 1.2 * total, "the pack holds little beyond the sheet's tiles");
    tile_index_close(&s.index);
}

static void resized_pieces(Arena *a, const char *outputs) {
    Pcg64 r = pcg64(194);
    const char *base = fmt(a, "%s/unwrapping/level/scroll/scan", outputs);
    const char *faces = fmt(a, "%s/faces/level-0", base);
    const char *sheets = fmt(a, "%s/website/sheets", outputs);
    const int shape[3][2] = {{643, 721}, {240, 330}, {86, 119}};
    const double x[3] = {1.3, 5.7, 85.0}, scale[3] = {1, 1.25, 0.8};
    json_object *pieces = json_object_new_array();
    for (int i = 0; i < 3; i++) {
        int h = shape[i][0], w = shape[i][1];
        u8 *pixels = integers(a, &r, 0, 256, h, w);
        for (int y = 7; y < 13; y++)
            memset(pixels + (size_t)y * w + 11, 0, 12);
        json_object *row = laid_piece(a, faces, i, pixels, h, w, x[i]);
        json_object_object_add(row, "group", json_object_new_boolean(i > 0));
        json_object_object_add(row, "flip_rows", json_object_new_boolean(i == 1));
        json_object_object_add(row, "um", json_object_new_double(PIXEL_MM * scale[i]));
        json_object_object_add(row, "width_mm", json_object_new_double(w * PIXEL_MM * scale[i]));
        json_object_object_add(row, "height_mm", json_object_new_double(h * PIXEL_MM * scale[i]));
        json_object *overlap = json_object_new_array();
        const int sides[] = {2, 1, 3, 0};
        for (int k = 0; k < 4; k++)
            json_object_array_add(overlap, json_object_new_int(sides[k]));
        json_object_object_add(row, "overlap", overlap);
        json_object_array_add(pieces, row);
        u8 *taught = push(a, (size_t)h * w);
        for (size_t k = 0; k < (size_t)h * w; k++)
            taught[k] = pixels[k] / 2;
        save_png(a, fmt(a, "%s/unit/piece-%04d-taught.png", faces, i), taught, w, h);
        if (i < 2) {
            const char *side = "{\"written\":\"far\"}";
            write_file(fmt(a, "%s/unit/piece-%04d-taught-side.json", faces, i), side, strlen(side));
        }
    }
    const char *layout = fmt(a, "%s/sheet/layout.json", base);
    write_layout(layout, pieces);
    need(!compose(outputs, "native", "sheet,taught", 3, "--written-face", "--model", "Test reader",
             "--ink-name", "ink label", NULL),
        "the resized pieces did not compose");
    const char *dst = fmt(a, "%s/native", sheets);
    Store native;
    store_open(&native, dst);
    const char *names[8];
    int n = layer_names(&native, names);
    Hashes h = {EVP_MD_CTX_new(), EVP_MD_CTX_new()};
    EVP_DigestInit_ex(h.pixels, EVP_sha256(), NULL);
    EVP_DigestInit_ex(h.raw, EVP_sha256(), NULL);
    for (int i = 0; i < n; i++)
        each_tile(a, &native, names[i], hash_tile, &h);
    u8 digest[32];
    char pixels[65], raw[65];
    EVP_DigestFinal_ex(h.pixels, digest, NULL);
    hex(digest, pixels);
    EVP_DigestFinal_ex(h.raw, digest, NULL);
    EVP_MD_CTX_free(h.pixels);
    EVP_MD_CTX_free(h.raw);
    hex(digest, raw);
    need(!strcmp(pixels, "ad738490ff16ccc17843e252f0e0b0e3aff8941770b71bfc326ec589936bf9cd"),
        "tile pixels differ from the golden: %s", pixels);
    need(!strcmp(raw, "963548911bb7e4a25ca474f58cb99ae5a783ba15df4b6c04d86c73cb94bcc623"),
        "first-level tile bytes differ from the golden: %s", raw);
    json_object *layers = json_object_from_file(fmt(a, "%s/layers.json", dst));
    need(!strcmp(json_object_get_string(
                     json_object_object_get(json_object_array_get_idx(layers, 1), "model")),
             "Test reader"),
        "the taught layer names its reader");
    json_object_put(layers);

    for (int level = 0; level < 2; level++) {
        const char *card = fmt(a, "%s/card.jpg", temp);
        char *args[] = {(char *)binary, "viewer.sheet_tiles", "--card", (char *)dst,
            fmt(a, "%d", level), "0.04", "0.08", "8.4", "6.2", (char *)card, NULL};
        need(!run(args, 1, NULL), "the card did not render at level %d", level);
        Bytes jpeg = read_file(a, card);
        size_t k = 2;
        while (k + 9 < jpeg.size &&
            !(jpeg.data[k] == 0xff && (jpeg.data[k + 1] & 0xf0) == 0xc0 &&
                jpeg.data[k + 1] != 0xc4 && jpeg.data[k + 1] != 0xcc))
            k += 2 + (jpeg.data[k + 2] << 8 | jpeg.data[k + 3]);
        need(k + 9 < jpeg.size && jpeg.data[k + 9] == 1, "a card is one grey channel");
        int w, rows, lit = 0;
        u8 *grey = jpeg_decode(a, jpeg, &w, &rows);
        for (size_t p = 0; p < (size_t)w * rows && !lit; p++)
            lit = grey[p] != 0;
        need(lit, "a card shows the sheet");
    }

    json_object *placed = json_object_from_file(fmt(a, "%s/sheet/placed.json", base));
    json_object *rows = json_object_new_array();
    json_object *expected = json_object_object_get(placed, "pieces");
    for (int i = 0; i < 105; i++) {
        for (size_t j = 0; j < json_object_array_length(expected); j++) {
            json_object *row = NULL;
            json_object_deep_copy(json_object_array_get_idx(expected, j), &row, NULL);
            int piece = json_object_get_int(json_object_object_get(row, "piece"));
            double at = json_object_get_double(json_object_object_get(row, "x_mm"));
            json_object_object_add(row, "piece", json_object_new_int(piece + 3 * i));
            json_object_object_add(row, "x_mm", json_object_new_double(at + i * 18.5));
            json_object_object_add(row, "group", json_object_new_int(i));
            json_object_object_add(row, "moved_from_joint_mm", json_object_new_double(3.5 * j));
            json_object_array_add(rows, row);
        }
    }
    json_object_put(placed);
    Bytes original = read_file(a, layout);
    write_layout(layout, rows);
    need(!compose(outputs, "viewer", "sheet,taught", 3, "--review-surface", "level", "--model",
             "Reader v3", NULL, NULL),
        "the viewer page did not compose");
    write_file(layout, original.data, original.size);
    Bytes html = read_file(a, fmt(a, "%s/viewer/index.html", sheets));
    const char *start = strstr((const char *)html.data, "const M = ");
    need(start != NULL, "the viewer page holds no metadata");
    json_tokener *tok = json_tokener_new();
    json_object *meta = json_tokener_parse_ex(tok, start + 10, (int)strlen(start + 10));
    json_tokener_free(tok);
    need(meta != NULL, "the viewer metadata does not parse");
    need(!strcmp(canonical_hash(a, json_object_object_get(meta, "shelves")),
             "5754a871c0726d15acf3ac21b15875ec5c2fd9cc06cea50a240053b59e5def8e"),
        "the shelves differ from the golden: %s",
        canonical(a, json_object_object_get(meta, "shelves")));
    need(!strcmp(canonical_hash(a, json_object_object_get(meta, "places")),
             "7a9d4ba4ff0c3307e0daece86bf4564973af2d13973c5f290165553d4b8e63d5"),
        "the places differ from the golden");
    json_object *area = json_object_object_get(meta, "area");
    need(json_object_object_length(area) == 4 &&
            json_object_get_double(json_object_object_get(area, "x")) == 0 &&
            json_object_get_double(json_object_object_get(area, "y")) == 0 &&
            json_object_get_double(json_object_object_get(area, "w")) == 418.7 &&
            json_object_get_double(json_object_object_get(area, "h")) == 138.6,
        "the sheet area is %s", canonical(a, json_object_object_get(meta, "area")));
    const char *title = json_object_get_string(json_object_object_get(meta, "title"));
    const char *prefix = "scroll \xc2\xb7 Reader v3 \xc2\xb7 level \xc2\xb7 ";
    need(!strncmp(title, prefix, strlen(prefix)), "the title is %s", title);
    need(!strcmp(json_object_get_string(json_object_object_get(meta, "scan")), "level/scroll/scan"),
        "the page names its scan");
    json_object *listed = json_object_from_file(fmt(a, "%s/viewer/layers.json", sheets));
    need(!strcmp(canonical(a, json_object_object_get(meta, "layers")), canonical(a, listed)),
        "the page lists the layers it was composed with");
    json_object_put(listed);
    json_object_put(meta);

    const char *old_pack = fmt(a, "%s/%s", dst,
        json_object_get_string(json_object_object_get(native.index.document, "pack")));
    Bytes before = read_file(a, old_pack);
    need(!compose(outputs, "native", "sheet", 1, "--written-face", NULL, NULL, NULL, NULL, NULL),
        "the sheet did not compose again");
    Store after;
    store_open(&after, dst);
    Bytes kept = read_file(a, old_pack);
    need(strcmp(json_object_get_string(json_object_object_get(after.index.document, "pack")),
             json_object_get_string(json_object_object_get(native.index.document, "pack"))) &&
            kept.size == before.size && !memcmp(kept.data, before.data, before.size),
        "a new pack leaves the previous one as it was");
    need(!strcmp(canonical(a, after.index.layers), canonical(a, native.index.layers)),
        "the layers are the same after composing again");
    Same same = {.other = &after};
    for (int i = 0; i < n; i++)
        each_tile(a, &native, names[i], same_tile_as, &same);
    need(!same.differ, "every tile is the same after composing again");

    const char *index = fmt(a, "%s/tiles.index", dst);
    Bytes contents = read_file(a, index);
    char packs[8][96] = {{0}}, later[8][96] = {{0}};
    int count = entries(dst, "tiles-", ".bin", packs);
    write_file(index, contents.data, 10);
    need(compose(outputs, "native", "sheet", 1, NULL, NULL, NULL, NULL, NULL, NULL) != 0,
        "a damaged index stops the composition");
    Bytes damaged = read_file(a, index);
    need(damaged.size == 10 && !memcmp(damaged.data, contents.data, 10),
        "a damaged index is left as it was");
    need(entries(dst, "tiles-", ".bin", later) == count &&
            !memcmp(packs, later, sizeof(packs[0]) * count),
        "a damaged index leaves the packs as they were");
    kept = read_file(a, old_pack);
    need(kept.size == before.size && !memcmp(kept.data, before.data, before.size),
        "a damaged index leaves the previous pack as it was");
    write_file(index, contents.data, contents.size);
    tile_index_close(&after.index);
    tile_index_close(&native.index);
}

int main(int argc, char **argv) {
    need(argc == 2, "usage: tile_store_e2e HERCULEX_BINARY");
    binary = argv[1];
    need(mkdtemp(temp) != NULL, "cannot make the tile store fixture");
    atexit(clean);
    Arena a = arena(1ull << 30);
    one_pack(&a, temp);
    resized_pieces(&a, fmt(&a, "%s/two", temp));
    puts("tile store e2e passed");
    return 0;
}
