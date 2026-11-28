#include <sys/file.h>
#include <sys/statvfs.h>
#include <tiffio.h>
#include <time.h>
#include <zstd.h>

typedef struct {
    char *root, *origin, *plan, *done, *unfinished;
    int server;
} Mirror;

typedef struct {
    json_object *row;
    size_t order;
} MirrorOrder;

static int mirror_suffix(const char *name, const char *suffix) {
    size_t n = strlen(name), size = strlen(suffix);
    return n >= size && !memcmp(name + n - size, suffix, size);
}

static int mirror_prefix(const char *name, const char *prefix) {
    return !strncmp(name, prefix, strlen(prefix));
}

static json_object *mirror_jsonl(const char *path, int optional) {
    FILE *file = fopen(path, "r");
    if (!file && (!optional || errno != ENOENT)) {
        return NULL;
    }
    char *line = NULL;
    size_t cap = 0;
    json_object *list = sequence();
    int ok = list != NULL;
    while (ok && file && getline(&line, &cap, file) >= 0) {
        json_object *row = json_tokener_parse(line);
        ok = row && add(list, row);
    }
    ok = ok && (!file || !ferror(file));
    if (file) {
        fclose(file);
    }
    free(line);
    if (!ok) {
        drop(list);
        list = NULL;
    }
    return list;
}

static int mirror_by_key(const void *left, const void *right) {
    const MirrorOrder *a = left, *b = right;
    int order = strcmp(text_field(a->row, "key"), text_field(b->row, "key"));
    if (order) {
        return order;
    }
    uint64_t x = json_object_get_uint64(value_of(a->row, "bytes"));
    uint64_t y = json_object_get_uint64(value_of(b->row, "bytes"));
    return (x > y) - (x < y);
}

static int mirror_by_size(const void *left, const void *right) {
    const MirrorOrder *a = left, *b = right;
    uint64_t x = json_object_get_uint64(value_of(a->row, "bytes"));
    uint64_t y = json_object_get_uint64(value_of(b->row, "bytes"));
    if (x != y) {
        return (x > y) - (x < y);
    }
    return (a->order > b->order) - (a->order < b->order);
}

static int mirror_by_rank(const void *left, const void *right) {
    const MirrorOrder *a = left, *b = right;
    int x = json_object_get_int(value_of(a->row, "rank"));
    int y = json_object_get_int(value_of(b->row, "rank"));
    return x == y ? mirror_by_size(left, right) : (x > y) - (x < y);
}

static json_object *mirror_sorted(json_object *rows, int (*compare)(const void *, const void *)) {
    size_t count = elements(rows);
    MirrorOrder *order = calloc(count ? count : 1, sizeof(*order));
    json_object *out = order ? sequence() : NULL;
    for (size_t i = 0; out && i < count; i++) {
        order[i] = (MirrorOrder){element(rows, i), i};
    }
    if (out) {
        qsort(order, count, sizeof(*order), compare);
        for (size_t i = 0; i < count; i++) {
            if (!add(out, retain(order[i].row))) {
                drop(out);
                out = NULL;
                break;
            }
        }
    }
    free(order);
    return out;
}

static int mirror_parked(const char *folder) {
    size_t n = strlen(folder);
    while (n && folder[n - 1] == '/') {
        n--;
    }
    size_t start = n;
    while (start && folder[start - 1] != '/') {
        start--;
    }
    size_t length = n - start;
    const char names[][16] = {"surface-volumes", "predictions", "spiral"};
    for (size_t i = 0; i < sizeof(names) / sizeof(*names); i++) {
        if (length == strlen(names[i]) && !memcmp(folder + start, names[i], length)) {
            return 1;
        }
    }
    return length >= 5 && !memcmp(folder + n - 5, ".zarr", 5);
}

static int mirror_walk_folder(CURL *curl, const char *base, const char *prefix, json_object *keys,
    json_object *skipped, int depth, char error[128]) {
    json_object *folders = sequence();
    InventoryListing found = {.keys = keys, .folders = folders};
    int ok = depth < 64 && folders && inventory_listing(curl, base, prefix, 1, 1000, &found, error);
    for (size_t i = 0; ok && i < elements(folders); i++) {
        json_object *entry = element(folders, i);
        const char *folder = string_value(entry);
        if (!mirror_prefix(folder, prefix) || strlen(folder) <= strlen(prefix)) {
            continue;
        }
        if (mirror_parked(folder)) {
            ok = add(skipped, retain(entry));
        } else {
            ok = mirror_walk_folder(curl, base, folder, keys, skipped, depth + 1, error);
        }
    }
    drop(folders);
    return ok;
}

typedef struct {
    const char *base;
    const char *sample;
    json_object *keys, *skip;
    int ok;
} MirrorWalk;

static void mirror_walk_sample(void *data, size_t i) {
    MirrorWalk *walk = (MirrorWalk *)data + i;
    char *prefix = NULL, error[128] = "";
    CURL *curl = curl_easy_init();
    json_object *folders = sequence();
    walk->keys = sequence();
    walk->skip = sequence();
    InventoryListing found = {.keys = walk->keys, .folders = folders};
    int ok = folders && walk->keys && walk->skip && curl &&
        asprintf(&prefix, "%s/", walk->sample) >= 0 &&
        inventory_listing(curl, walk->base, prefix, 1, 1000, &found, error);
    for (size_t j = 0; ok && j < elements(folders); j++) {
        const char *folder = string_value(element(folders, j));
        if (mirror_suffix(folder, "/segments/")) {
            InventoryListing segments = {.folders = sequence()};
            ok = segments.folders &&
                inventory_listing(curl, walk->base, folder, 1, 1000, &segments, error);
            for (size_t k = 0; ok && k < elements(segments.folders); k++) {
                const char *path = string_value(element(segments.folders, k));
                ok = mirror_walk_folder(curl, walk->base, path, walk->keys, walk->skip, 0, error);
            }
            drop(segments.folders);
        } else {
            ok = mirror_walk_folder(curl, walk->base, folder, walk->keys, walk->skip, 0, error);
        }
    }
    if (!ok) {
        fprintf(stderr, "HerculeX: %s: %s\n", walk->sample, *error ? error : "could not list");
    }
    walk->ok = ok;
    if (curl) {
        curl_easy_cleanup(curl);
    }
    free(prefix);
    drop(folders);
}

static json_object *mirror_ordered_names(json_object *map) {
    size_t count = fields(map), used = 0;
    const char **keys = calloc(count ? count : 1, sizeof(*keys));
    json_object *out = keys ? sequence() : NULL;
    if (out) {
        Cursor it = entries(map);
        Node *value;
        const char *key;
        while ((key = next_entry(&it, &value))) {
            keys[used++] = key;
        }
        qsort(keys, count, sizeof(*keys), inventory_string_order);
        for (size_t i = 0; i < count; i++) {
            if (!add(out, json_object_new_string(keys[i]))) {
                drop(out);
                out = NULL;
                break;
            }
        }
    }
    free(keys);
    return out;
}

static json_object *mirror_plan_products(const Mirror *mirror, const char *catalog, size_t *left) {
    char *inventory = path_join(catalog, "products.jsonl");
    json_object *metadata = cached_catalog(NULL), *samples = value_of(metadata, "samples");
    json_object *names = is_map(samples) ? mirror_ordered_names(samples) : NULL;
    json_object *rows = inventory ? mirror_jsonl(inventory, 0) : NULL;
    json_object *thin = mapping(), *parked = mapping();
    json_object *root = sequence(), *wanted = sequence();
    size_t count = elements(names);
    MirrorWalk *walks = names ? calloc(count ? count : 1, sizeof(*walks)) : NULL;
    char error[128] = "";
    CURL *curl = curl_easy_init();
    InventoryListing top = {.keys = root};
    int ok = rows && thin && parked && root && wanted && walks && curl &&
        inventory_listing(curl, mirror->origin, "", 1, 3, &top, error);
    for (size_t i = 0; ok && i < elements(rows); i++) {
        json_object *row = element(rows, i);
        const char *kind = text_field(row, "kind"), *path = text_field(row, "path");
        if (kind && path && (!strcmp(kind, "ink-detection") || !strcmp(kind, "alpha-render"))) {
            ok = assign(thin, path, json_object_new_boolean(1));
        }
    }
    for (size_t i = 0; ok && i < count; i++) {
        walks[i] = (MirrorWalk){.base = mirror->origin, .sample = string_value(element(names, i))};
    }
    if (ok) {
        inventory_parallel(mirror_walk_sample, walks, count, 12);
    }
    for (size_t i = 0; walks && i <= count; i++) {
        json_object *files = i ? walks[i - 1].keys : root;
        json_object *skip = i ? walks[i - 1].skip : NULL;
        ok = ok && (!i || walks[i - 1].ok);
        for (size_t j = 0; ok && j < elements(files); j++) {
            json_object *row = element(files, j);
            const char *key = text_field(row, "key");
            if (key && !mirror_suffix(key, "/") && !value_of(thin, key)) {
                ok = add(wanted, retain(row));
            }
        }
        for (size_t j = 0; ok && j < elements(skip); j++) {
            ok = assign(parked, string_value(element(skip, j)), json_object_new_boolean(1));
        }
        if (i) {
            drop(files);
            drop(skip);
        }
    }
    json_object *sorted = ok ? mirror_sorted(wanted, mirror_by_key) : NULL;
    json_object *brick_stores = sorted ? mirror_ordered_names(parked) : NULL;
    json_object *thinned = sorted ? mirror_ordered_names(thin) : NULL;
    json_object *state = mapping();
    char *sidecar = path_join(mirror->root, "left-out.json");
    *left = elements(brick_stores);
    ok = sorted && sidecar && state && brick_stores && thinned &&
        assign(state, "brick_stores", retain(brick_stores)) &&
        assign(state, "thinned", retain(thinned)) && folders(mirror->plan) &&
        inventory_write(mirror->plan, sorted, 1) && inventory_write(sidecar, state, 0);
    if (!ok) {
        fprintf(stderr, "HerculeX: %s\n", *error ? error : "could not build product mirror plan");
        drop(sorted);
        sorted = NULL;
    }
    if (curl) {
        curl_easy_cleanup(curl);
    }
    drop(brick_stores);
    drop(thinned);
    drop(state);
    drop(root);
    drop(wanted);
    drop(thin);
    drop(parked);
    drop(rows);
    drop(names);
    drop(metadata);
    free(sidecar);
    free(walks);
    free(inventory);
    return sorted;
}

static int mirror_rank(const char *path) {
    const char *labels[] = {"/datasets/seg-derived-recto-surfaces/labelsTr/",
        "/datasets/fiber-skeletons/", "/other/annotators/"};
    for (size_t i = 0; i < sizeof(labels) / sizeof(*labels); i++) {
        if (mirror_prefix(path, labels[i])) {
            return 2;
        }
    }
    const char patches[] = "/datasets/spiral_datasets/PHercParis4/";
    if (mirror_prefix(path, patches)) {
        const char *name = path + sizeof(patches) - 1;
        const char *parts[] = {
            "verified_patches", "eval_fibers", "winding_inference", "outer_shell"};
        for (size_t i = 0; i < sizeof(parts) / sizeof(*parts); i++) {
            if (mirror_prefix(name, parts[i])) {
                return 4;
            }
        }
        const char *json = strstr(name, ".json"), *slash = strchr(name, '/');
        if (json && json > name && (!slash || json < slash)) {
            return 4;
        }
    }
    if ((!mirror_prefix(path, "/full-scrolls/") && !mirror_prefix(path, "/fragments/")) ||
        (!strstr(path, "/paths/") && !strstr(path, "/working/") &&
            !strstr(path, "/multispectral/") && !strstr(path, "/volumetric-instance-labels/"))) {
        return -1;
    }
    if (strstr(path, "/Scroll5/") && strstr(path, "/working/")) {
        return -1;
    }
    const char *name = strrchr(path, '/') + 1, *suffix = strrchr(name, '.');
    if (suffix == name) {
        suffix = NULL;
    }
    const char *ignored[] = {".ppm", ".vcps", ".vcano", ".orig", ".ply", ".psd", ".mp4", ".mkv"};
    for (size_t i = 0; suffix && i < sizeof(ignored) / sizeof(*ignored); i++) {
        if (!strcasecmp(suffix, ignored[i])) {
            return -1;
        }
    }
    size_t n = strlen(name);
    if ((n >= 11 && !strcasecmp(name + n - 11, "_points.obj")) ||
        (n >= 11 && !strcasecmp(name + n - 11, "cellmap.tif"))) {
        return -1;
    }
    return !suffix ? 0 : !strcasecmp(suffix, ".obj") ? 1 : !strcasecmp(suffix, ".tif") ? 3 : 0;
}

static json_object *mirror_plan_server(const Mirror *mirror, const char *catalog) {
    char *inventory = path_join(catalog, "server.jsonl");
    json_object *source = inventory ? mirror_jsonl(inventory, 0) : NULL, *rows = sequence();
    int ok = source && rows;
    for (size_t i = 0; ok && i < elements(source); i++) {
        json_object *entry = element(source, i);
        const char *key = text_field(entry, "path");
        int rank = key ? mirror_rank(key) : -1;
        if (rank >= 0) {
            json_object *row = mapping();
            ok = row && assign(row, "key", json_object_new_string(key)) &&
                assign(row, "bytes", retain(value_of(entry, "bytes"))) &&
                assign(row, "rank", json_object_new_int(rank)) && add(rows, retain(row));
            drop(row);
        }
    }
    json_object *sorted = ok ? mirror_sorted(rows, mirror_by_rank) : NULL;
    ok = sorted && folders(mirror->plan) && inventory_write(mirror->plan, sorted, 1);
    if (!ok) {
        fprintf(stderr, "HerculeX: could not build server mirror plan\n");
        drop(sorted);
        sorted = NULL;
    }
    drop(rows);
    drop(source);
    free(inventory);
    return sorted;
}

static int mirror_safe(const char *key, int root_allowed) {
    if (*key == '/' && root_allowed) {
        key++;
    }
    if (!*key || *key == '/' || mirror_suffix(key, "/")) {
        return 0;
    }
    for (const char *at = key; *at;) {
        size_t length = strcspn(at, "/");
        if (!length || (length == 1 && at[0] == '.') ||
            (length == 2 && at[0] == '.' && at[1] == '.')) {
            return 0;
        }
        at += length;
        if (*at == '/') {
            at++;
        }
    }
    return 1;
}

static char *mirror_held(const Mirror *mirror, const char *key) {
    char *out = NULL;
    if (mirror_safe(key, mirror->server) &&
        asprintf(&out, "%s/%s%s", mirror->root, key + (*key == '/'),
            mirror_suffix(key, ".obj") ? ".zst" : "") < 0) {
        out = NULL;
    }
    return out;
}

typedef struct {
    uint32_t width, height;
    uint16_t bits, format, samples, planar, photo, orientation;
    tmsize_t stride;
} MirrorTiff;

static int mirror_tiff_settings(TIFF *file, MirrorTiff *info) {
    if (!file || !TIFFGetField(file, TIFFTAG_IMAGEWIDTH, &info->width) ||
        !TIFFGetField(file, TIFFTAG_IMAGELENGTH, &info->height)) {
        return 0;
    }
    TIFFGetFieldDefaulted(file, TIFFTAG_BITSPERSAMPLE, &info->bits);
    TIFFGetFieldDefaulted(file, TIFFTAG_SAMPLEFORMAT, &info->format);
    TIFFGetFieldDefaulted(file, TIFFTAG_SAMPLESPERPIXEL, &info->samples);
    TIFFGetFieldDefaulted(file, TIFFTAG_PLANARCONFIG, &info->planar);
    TIFFGetFieldDefaulted(file, TIFFTAG_PHOTOMETRIC, &info->photo);
    TIFFGetFieldDefaulted(file, TIFFTAG_ORIENTATION, &info->orientation);
    info->stride = TIFFScanlineSize(file);
    return info->width && info->height && info->stride > 0;
}

typedef struct {
    TIFF *file;
    const MirrorTiff *info;
    uint32_t width, height, block;
    uint16_t sample;
    size_t pixel, tile_stride;
    uint8_t *pixels, *tile;
} MirrorTiffRead;

static int mirror_tiff_read_open(MirrorTiffRead *reader, TIFF *file, const MirrorTiff *info) {
    reader->file = file;
    reader->info = info;
    reader->block = UINT32_MAX;
    if (!TIFFIsTiled(file)) {
        return 1;
    }
    tmsize_t tile_size = TIFFTileSize(file), tile_stride = TIFFTileRowSize(file);
    int ok = !(info->bits % 8) && tile_size > 0 && tile_stride > 0 &&
        TIFFGetField(file, TIFFTAG_TILEWIDTH, &reader->width) && reader->width &&
        TIFFGetField(file, TIFFTAG_TILELENGTH, &reader->height) && reader->height &&
        (size_t)info->stride <= SIZE_MAX / reader->height;
    reader->pixel = info->bits / 8 * (info->planar == PLANARCONFIG_SEPARATE ? 1 : info->samples);
    reader->tile_stride = tile_stride;
    reader->pixels = ok ? malloc((size_t)info->stride * reader->height) : NULL;
    reader->tile = reader->pixels ? malloc(tile_size) : NULL;
    return reader->tile != NULL;
}

static int mirror_tiff_row(MirrorTiffRead *reader, uint8_t *out, uint32_t row, uint16_t sample) {
    if (!reader->tile) {
        return TIFFReadScanline(reader->file, out, row, sample) >= 0;
    }
    const MirrorTiff *info = reader->info;
    uint32_t block = row / reader->height * reader->height;
    if (block != reader->block || sample != reader->sample) {
        uint32_t rows = info->height - block;
        rows = rows < reader->height ? rows : reader->height;
        for (uint32_t x = 0; x < info->width; x += reader->width) {
            if (TIFFReadTile(reader->file, reader->tile, x, block, 0, sample) < 0) {
                return 0;
            }
            uint32_t cols = info->width - x;
            cols = cols < reader->width ? cols : reader->width;
            for (uint32_t r = 0; r < rows; r++) {
                memcpy(reader->pixels + (size_t)r * info->stride + (size_t)x * reader->pixel,
                    reader->tile + r * reader->tile_stride, cols * reader->pixel);
            }
        }
        reader->block = block;
        reader->sample = sample;
    }
    memcpy(out, reader->pixels + (size_t)(row - block) * info->stride, info->stride);
    return 1;
}

static void mirror_tiff_read_close(MirrorTiffRead *reader) {
    free(reader->pixels);
    free(reader->tile);
}

static int mirror_tiff_equal(const char *one, const char *two) {
    TIFF *a = TIFFOpen(one, "r"), *b = TIFFOpen(two, "r");
    MirrorTiff x = {0}, y = {0};
    MirrorTiffRead readers[2] = {0};
    int ok = mirror_tiff_settings(a, &x) && mirror_tiff_settings(b, &y) && x.width == y.width &&
        x.height == y.height && x.bits == y.bits && x.format == y.format &&
        x.samples == y.samples && x.planar == y.planar && x.stride == y.stride &&
        TIFFNumberOfDirectories(a) == 1 && TIFFNumberOfDirectories(b) == 1 &&
        mirror_tiff_read_open(readers, a, &x) && mirror_tiff_read_open(readers + 1, b, &y);
    uint8_t *rows = ok ? malloc((size_t)x.stride * 2) : NULL;
    ok = rows != NULL;
    uint16_t planes = x.planar == PLANARCONFIG_SEPARATE ? x.samples : 1;
    for (uint16_t k = 0; ok && k < planes; k++) {
        for (uint32_t row = 0; ok && row < x.height; row++) {
            ok = mirror_tiff_row(readers, rows, row, k) &&
                mirror_tiff_row(readers + 1, rows + x.stride, row, k) &&
                !memcmp(rows, rows + x.stride, (size_t)x.stride);
        }
    }
    free(rows);
    mirror_tiff_read_close(readers);
    mirror_tiff_read_close(readers + 1);
    if (a) {
        TIFFClose(a);
    }
    if (b) {
        TIFFClose(b);
    }
    return ok;
}

static void mirror_pack_tiff(const char *path, uint64_t original) {
    FILE *probe = fopen(path, "rb");
    uint8_t magic[4] = {0};
    if (probe) {
        fread(magic, 1, sizeof(magic), probe);
        fclose(probe);
    }
    if (memcmp(magic, "II*\0", 4) != 0 && memcmp(magic, "MM\0*", 4) != 0)
        return;

    TIFF *input = TIFFOpen(path, "r");
    MirrorTiff info = {0};
    MirrorTiffRead reader = {0};
    char *temporary = NULL;
    int ok = mirror_tiff_settings(input, &info) && info.samples == 1 &&
        TIFFNumberOfDirectories(input) == 1 &&
        (info.bits == 8 || info.bits == 16 || info.bits == 32 || info.bits == 64) &&
        (info.photo == PHOTOMETRIC_MINISBLACK || info.photo == PHOTOMETRIC_MINISWHITE) &&
        mirror_tiff_read_open(&reader, input, &info) && asprintf(&temporary, "%s.tiff", path) >= 0;
    TIFF *output = ok ? TIFFOpen(temporary, "w") : NULL;
    void *row = output ? malloc(info.stride) : NULL;
    const struct {
        ttag_t tag;
        uint32_t value;
    } fields[] = {{TIFFTAG_IMAGEWIDTH, info.width}, {TIFFTAG_IMAGELENGTH, info.height},
        {TIFFTAG_BITSPERSAMPLE, info.bits}, {TIFFTAG_SAMPLEFORMAT, info.format},
        {TIFFTAG_SAMPLESPERPIXEL, info.samples}, {TIFFTAG_PLANARCONFIG, info.planar},
        {TIFFTAG_PHOTOMETRIC, info.photo}, {TIFFTAG_ORIENTATION, info.orientation},
        {TIFFTAG_COMPRESSION, COMPRESSION_ZSTD},
        {TIFFTAG_PREDICTOR,
            info.format == SAMPLEFORMAT_IEEEFP ? PREDICTOR_FLOATINGPOINT : PREDICTOR_HORIZONTAL},
        {TIFFTAG_ZSTD_LEVEL, 9}};
    ok = row != NULL;
    for (size_t i = 0; ok && i < sizeof(fields) / sizeof(*fields); i++)
        ok = TIFFSetField(output, fields[i].tag, fields[i].value);
    ok = ok && TIFFSetField(output, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(output, 0));
    for (uint32_t i = 0; ok && i < info.height; i++)
        ok = mirror_tiff_row(&reader, row, i, 0) && TIFFWriteScanline(output, row, i, 0) >= 0;
    free(row);
    mirror_tiff_read_close(&reader);
    if (input)
        TIFFClose(input);
    if (output)
        TIFFClose(output);

    struct stat held;
    if (ok && !stat(temporary, &held) && (uint64_t)held.st_size < original &&
        mirror_tiff_equal(path, temporary))
        rename(temporary, path);
    if (temporary) {
        unlink(temporary);
        free(temporary);
    }
}

typedef struct {
    FILE *file;
    ZSTD_CCtx *zstd;
    void *block;
    size_t capacity;
    uint64_t seen;
} MirrorPayload;

static int mirror_zstd(MirrorPayload *out, const void *data, size_t size, ZSTD_EndDirective end) {
    ZSTD_inBuffer input = {data, size, 0};
    size_t remaining;
    do {
        ZSTD_outBuffer buffer = {out->block, out->capacity, 0};
        remaining = ZSTD_compressStream2(out->zstd, &buffer, &input, end);
        if (ZSTD_isError(remaining) || fwrite(buffer.dst, 1, buffer.pos, out->file) != buffer.pos) {
            return 0;
        }
    } while (input.pos < input.size || (end == ZSTD_e_end && remaining));
    return 1;
}

static size_t mirror_payload(char *data, size_t size, size_t count, void *context) {
    MirrorPayload *out = context;
    size_t bytes = size * count;
    int ok = out->zstd ? mirror_zstd(out, data, bytes, ZSTD_e_continue)
                       : fwrite(data, 1, bytes, out->file) == bytes;
    if (ok) {
        out->seen += bytes;
    }
    return ok ? bytes : 0;
}

static int mirror_receive(const Mirror *mirror, const char *key, const char *path, int packing,
    uint64_t expected, uint64_t *seen, char error[128]) {
    CURL *curl = curl_easy_init();
    char *quoted = curl ? inventory_quote_path(curl, key) : NULL, *url = NULL;
    int made =
        quoted && asprintf(&url, "%s%s%s", mirror->origin, *key == '/' ? "" : "/", quoted) >= 0;
    MirrorPayload out = {.file = made ? fopen(path, "wb") : NULL};
    int ok = out.file != NULL;
    if (ok && packing && mirror_suffix(key, ".obj")) {
        out.zstd = ZSTD_createCCtx();
        out.capacity = ZSTD_CStreamOutSize();
        out.block = out.zstd ? malloc(out.capacity) : NULL;
        ok = out.block &&
            !ZSTD_isError(ZSTD_CCtx_setParameter(out.zstd, ZSTD_c_compressionLevel, 6));
    }
    curl_off_t declared = -1;
    CURLcode result = CURLE_FAILED_INIT;
    if (ok) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 120L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 120L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        if (mirror->server) {
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "herculex-inventory");
        }
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, mirror_payload);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
        result = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &declared);
        if (expected == UINT64_MAX && mirror->server && declared >= 0) {
            expected = (uint64_t)declared;
        }
        ok = result == CURLE_OK && (expected == UINT64_MAX ? !packing : out.seen == expected) &&
            (!out.zstd || mirror_zstd(&out, NULL, 0, ZSTD_e_end));
    }
    if (out.file) {
        ok = !fclose(out.file) && ok;
    }
    if (!ok) {
        if (result == CURLE_OK && expected != UINT64_MAX && out.seen != expected) {
            snprintf(error, 128, "%llu bytes of %llu", (unsigned long long)out.seen,
                (unsigned long long)expected);
        } else {
            snprintf(error, 128, "%s",
                result == CURLE_OK ? "could not store completed transfer"
                                   : curl_easy_strerror(result));
        }
        unlink(path);
    }
    *seen = out.seen;
    ZSTD_freeCCtx(out.zstd);
    free(out.block);
    free(url);
    curl_free(quoted);
    if (curl) {
        curl_easy_cleanup(curl);
    }
    return ok;
}

static json_object *mirror_done_keys(json_object *done) {
    json_object *keys = mapping();
    int ok = keys != NULL;
    for (size_t i = 0; ok && i < elements(done); i++) {
        const char *key = text_field(element(done, i), "key");
        ok = key && assign(keys, key, json_object_new_boolean(1));
    }
    if (!ok) {
        drop(keys);
        keys = NULL;
    }
    return keys;
}

static int mirror_space(const Mirror *mirror) {
    struct statvfs room;
    return statvfs(mirror->root, &room) ? -1 : (double)room.f_bavail * room.f_frsize >= 30 * 0x1p30;
}

typedef struct {
    const Mirror *mirror;
    json_object **rows;
    uint8_t *complete;
    FILE *log;
    pthread_mutex_t write;
    size_t count;
    uint64_t fetched, stored;
} MirrorQueue;

static void mirror_worker(void *context, size_t index) {
    MirrorQueue *queue = context;
    const Mirror *mirror = queue->mirror;
    json_object *row = queue->rows[index];
    const char *key = text_field(row, "key");
    char *target = key ? mirror_held(mirror, key) : NULL, *part = NULL;
    char error[128] = "invalid mirror key";
    if (!target) {
        fprintf(stderr, "failed %s %s\n", key ? key : "", error);
        return;
    }
    int space = mirror_space(mirror);
    while (space == 0 && !mirror->server) {
        printf("waiting for room on the internal disc\n");
        fflush(stdout);
        sleep(300);
        space = mirror_space(mirror);
    }
    if (space <= 0) {
        free(target);
        return;
    }
    uint64_t seen = 0;
    uint64_t expected =
        mirror->server ? UINT64_MAX : json_object_get_uint64(value_of(row, "bytes"));
    int ok = folders(target) && asprintf(&part, "%s.part", target) >= 0 &&
        mirror_receive(mirror, key, part, 1, expected, &seen, error);
    if (ok && mirror_suffix(key, ".tif") &&
        (mirror->server || strstr(key, "/mesh/") || strstr(key, "/segments/raw/"))) {
        mirror_pack_tiff(part, seen);
    }
    struct stat held;
    ok = ok && !stat(part, &held) && !rename(part, target);
    if (ok) {
        json_object *receipt = mapping();
        ok = receipt && assign(receipt, "key", json_object_new_string(key)) &&
            assign(receipt, "bytes", json_object_new_uint64(seen)) &&
            assign(receipt, "stored", json_object_new_uint64(held.st_size));
        if (ok) {
            pthread_mutex_lock(&queue->write);
            const char *line = json_object_to_json_string_ext(
                receipt, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE);
            ok =
                fputs(line, queue->log) >= 0 && fputc('\n', queue->log) >= 0 && !fflush(queue->log);
            if (ok && ++queue->count % 500 == 0) {
                printf("%zu files %.2f GB fetched %.2f GB held\n", queue->count,
                    (queue->fetched + seen) / 1e9, (queue->stored + held.st_size) / 1e9);
                fflush(stdout);
            }
            queue->fetched += seen;
            queue->stored += held.st_size;
            pthread_mutex_unlock(&queue->write);
        }
        drop(receipt);
    }
    if (!ok) {
        if (part) {
            unlink(part);
        }
        fprintf(stderr, "failed %s %.100s\n", key, error);
    }
    queue->complete[index] = ok;
    free(target);
    free(part);
}

static int mirror_heavy(const json_object *row) {
    const char *key = text_field((json_object *)row, "key");
    return key && mirror_suffix(key, ".tif") &&
        json_object_get_uint64(value_of((json_object *)row, "bytes")) > 300000000;
}

typedef struct {
    MirrorQueue *queue;
    const size_t *index;
} MirrorLane;

static void mirror_lane_file(void *context, size_t index) {
    MirrorLane *lane = context;
    mirror_worker(lane->queue, lane->index[index]);
}

typedef struct {
    MirrorLane lane;
    size_t count;
} MirrorHeavy;

static void *mirror_heavy_lane(void *context) {
    MirrorHeavy *lane = context;
    for (size_t i = 0; i < lane->count; i++) {
        mirror_lane_file(&lane->lane, i);
    }
    return NULL;
}

static int mirror_fetch(const Mirror *mirror) {
    char *lock_path = path_join(mirror->root, "fetch.lock");
    int fd = lock_path && folders(lock_path) ? open(lock_path, O_CREAT | O_WRONLY, 0666) : -1;
    free(lock_path);
    int ok = fd >= 0 && !flock(fd, LOCK_EX | LOCK_NB);
    json_object *plan = ok ? mirror_jsonl(mirror->plan, 0) : NULL;
    json_object *done = plan ? mirror_jsonl(mirror->done, 1) : NULL;
    json_object *keys = done ? mirror_done_keys(done) : NULL, *pending = sequence();
    ok = keys && pending;
    for (size_t i = 0; ok && i < elements(plan); i++) {
        json_object *row = element(plan, i);
        const char *key = text_field(row, "key");
        if (!key || !mirror_safe(key, mirror->server)) {
            ok = 0;
        } else if (!value_of(keys, key)) {
            ok = add(pending, retain(row)) && assign(keys, key, json_object_new_boolean(1));
        }
    }
    json_object *sorted = ok && !mirror->server ? mirror_sorted(pending, mirror_by_size) : NULL;
    size_t n = elements(pending);
    json_object **rows = calloc(n ? n : 1, sizeof(*rows));
    uint8_t *complete = calloc(n ? n : 1, 1);
    size_t *work = calloc(n ? n : 1, sizeof(*work)), *heavy = calloc(n ? n : 1, sizeof(*heavy));
    FILE *log = ok ? fopen(mirror->done, "a") : NULL;
    ok = ok && (mirror->server || sorted) && rows && complete && log && work && heavy;
    for (size_t i = 0; ok && i < n; i++) {
        rows[i] =
            mirror->server ? element(pending, i) : element(sorted, i % 2 ? n - 1 - i / 2 : i / 2);
    }
    MirrorQueue queue = {mirror, rows, complete, log, PTHREAD_MUTEX_INITIALIZER, 0, 0, 0};
    size_t remaining = n;
    int passes = mirror->server ? 3 : 5;
    for (int pass = 0; ok && remaining && pass < passes; pass++) {
        size_t small_count = 0, heavy_count = 0;
        for (size_t i = 0; i < n; i++) {
            if (complete[i]) {
                continue;
            }
            if (mirror->server && mirror_heavy(rows[i])) {
                heavy[heavy_count++] = i;
            } else {
                work[small_count++] = i;
            }
        }
        MirrorHeavy lane = {{&queue, heavy}, heavy_count};
        pthread_t background;
        int parallel = heavy_count && !pthread_create(&background, NULL, mirror_heavy_lane, &lane);
        for (size_t start = 0; start < small_count; start += 5000) {
            size_t batch = small_count - start;
            MirrorLane light = {&queue, work + start};
            inventory_parallel(
                mirror_lane_file, &light, batch < 5000 ? batch : 5000, mirror->server ? 8 : 12);
        }
        if (parallel) {
            pthread_join(background, NULL);
        } else {
            mirror_heavy_lane(&lane);
        }
        remaining = 0;
        for (size_t i = 0; i < n; i++) {
            remaining += !complete[i];
        }
        if (mirror_space(mirror) <= 0) {
            break;
        }
        if (!mirror->server && remaining && pass + 1 < passes) {
            sleep(120);
        }
    }
    if (log) {
        ok = !fclose(log) && ok;
    }
    pthread_mutex_destroy(&queue.write);
    json_object *left = sequence();
    ok = ok && left;
    for (size_t i = 0; ok && i < n; i++) {
        if (!complete[i]) {
            ok = add(left, retain(value_of(rows[i], "key")));
        }
    }
    if (ok) {
        ok = inventory_write(mirror->unfinished, left, 0);
        printf("finished, unfinished %zu\n", remaining);
    } else {
        fprintf(stderr, "HerculeX: could not fetch from mirror plan %s\n", mirror->plan);
    }
    drop(left);
    drop(sorted);
    drop(pending);
    drop(keys);
    drop(done);
    drop(plan);
    free(heavy);
    free(work);
    free(complete);
    free(rows);
    if (fd >= 0) {
        close(fd);
    }
    return !ok ? 1 : remaining && mirror->server ? 3 : 0;
}

static int mirror_file_equal(const char *one, const char *two) {
    FILE *a = fopen(one, "rb"), *b = fopen(two, "rb");
    int ok = a && b;
    uint8_t x[65536], y[65536];
    while (ok) {
        size_t n = fread(x, 1, sizeof(x), a);
        ok = fread(y, 1, sizeof(y), b) == n && !memcmp(x, y, n) && !ferror(a) && !ferror(b);
        if (n < sizeof(x)) {
            break;
        }
    }
    if (a) {
        fclose(a);
    }
    if (b) {
        fclose(b);
    }
    return ok;
}

static int mirror_zstd_equal(const char *packed, const char *raw) {
    FILE *input = fopen(packed, "rb"), *expected = fopen(raw, "rb");
    ZSTD_DCtx *d = input && expected ? ZSTD_createDCtx() : NULL;
    uint8_t source[65536], decoded[65536], ref[65536];
    int ok = d != NULL;
    size_t remaining = 1;
    while (ok) {
        ZSTD_inBuffer in = {source, fread(source, 1, sizeof(source), input), 0};
        size_t produced = 0;
        while (ok && (in.pos < in.size || (remaining && produced == sizeof(decoded)))) {
            ZSTD_outBuffer out = {decoded, sizeof(decoded), 0};
            remaining = ZSTD_decompressStream(d, &out, &in);
            produced = out.pos;
            ok = !ZSTD_isError(remaining) && fread(ref, 1, out.pos, expected) == out.pos &&
                !memcmp(decoded, ref, out.pos);
        }
        if (in.size < sizeof(source)) {
            break;
        }
    }
    ok = ok && !remaining && fgetc(expected) == EOF && !ferror(input) && !ferror(expected);
    ZSTD_freeDCtx(d);
    if (input) {
        fclose(input);
    }
    if (expected) {
        fclose(expected);
    }
    return ok;
}

static int mirror_compare(const Mirror *mirror, const char *key) {
    char *target = mirror_held(mirror, key), *raw = NULL;
    if (!target || access(target, R_OK) || asprintf(&raw, "%s.check.XXXXXX", target) < 0) {
        free(raw);
        free(target);
        return 0;
    }
    int fd = mkstemp(raw);
    if (fd >= 0) {
        close(fd);
    }
    char error[128];
    uint64_t seen = 0;
    int read = fd >= 0 && mirror_receive(mirror, key, raw, 0, UINT64_MAX, &seen, error);
    int same = read &&
        (mirror_suffix(key, ".obj") ? mirror_zstd_equal(target, raw)
                                    : mirror_file_equal(target, raw));
    if (read && !same && mirror_suffix(key, ".tif")) {
        same = mirror_tiff_equal(target, raw);
    }
    if (fd >= 0) {
        unlink(raw);
    }
    free(raw);
    free(target);
    return same;
}

typedef struct {
    uint32_t mt[624];
    size_t position;
} MirrorRandom;

static void mirror_seed(MirrorRandom *rng) {
    uint32_t *x = rng->mt;
    x[0] = 19650218U;
    for (size_t i = 1; i < 624; i++) {
        x[i] = 1812433253U * (x[i - 1] ^ (x[i - 1] >> 30)) + (uint32_t)i;
    }
    size_t i = 1;
    for (size_t j = 0; j < 624; j++) {
        x[i] ^= (x[i - 1] ^ (x[i - 1] >> 30)) * 1664525U;
        if (++i == 624) {
            x[0] = x[623];
            i = 1;
        }
    }
    for (size_t j = 0; j < 623; j++) {
        x[i] = (x[i] ^ ((x[i - 1] ^ (x[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
        if (++i == 624) {
            x[0] = x[623];
            i = 1;
        }
    }
    x[0] = 0x80000000U;
    rng->position = 624;
}

static uint32_t mirror_random_word(MirrorRandom *rng) {
    uint32_t *mt = rng->mt;
    if (rng->position == 624) {
        for (size_t i = 0; i < 624; i++) {
            uint32_t y = (mt[i] & 0x80000000U) | (mt[(i + 1) % 624] & 0x7fffffffU);
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
        }
        rng->position = 0;
    }
    uint32_t word = mt[rng->position++];
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680U;
    word ^= (word << 15) & 0xefc60000U;
    return word ^ (word >> 18);
}

static size_t mirror_pick(MirrorRandom *rng, size_t limit) {
    size_t count = limit, bits = 0;
    while (count) {
        count >>= 1;
        bits++;
    }
    size_t choice;
    do {
        if (bits <= 32) {
            choice = mirror_random_word(rng) >> (32 - bits);
        } else {
            uint64_t low = mirror_random_word(rng);
            uint64_t high = mirror_random_word(rng) >> (64 - bits);
            choice = (size_t)(low | (high << 32));
        }
    } while (choice >= limit);
    return choice;
}

static int mirror_verify(const Mirror *mirror, json_object *done, size_t *count, size_t *same) {
    size_t max = mirror->server ? 40 : 60, used = 0;
    uint64_t cut = mirror->server ? 100000000 : 200000000;
    const char **small = calloc(elements(done) ? elements(done) : 1, sizeof(*small));
    if (!small) {
        return 0;
    }
    for (size_t i = 0; i < elements(done); i++) {
        json_object *row = element(done, i);
        const char *key = text_field(row, "key");
        if (key && json_object_get_uint64(value_of(row, "bytes")) < cut) {
            small[used++] = key;
        }
    }
    *count = used < max ? used : max;
    size_t setsize = 21;
    if (*count > 5) {
        size_t buckets = 4;
        while (buckets < *count * 3) {
            buckets *= 4;
        }
        setsize += buckets;
    }
    MirrorRandom rng;
    mirror_seed(&rng);
    uint8_t *picked = used > setsize ? calloc(used, 1) : NULL;
    int ok = used <= setsize || picked;
    for (size_t i = 0; ok && i < *count; i++) {
        size_t at = mirror_pick(&rng, used <= setsize ? used - i : used);
        if (used <= setsize) {
            *same += mirror_compare(mirror, small[at]);
            small[at] = small[used - i - 1];
        } else {
            while (picked[at]) {
                at = mirror_pick(&rng, used);
            }
            picked[at] = 1;
            *same += mirror_compare(mirror, small[at]);
        }
    }
    free(picked);
    free(small);
    return ok;
}

static uint64_t mirror_sum(json_object *rows, const char *field) {
    uint64_t total = 0;
    for (size_t i = 0; i < elements(rows); i++) {
        total += json_object_get_uint64(value_of(element(rows, i), field));
    }
    return total;
}

static int mirror_status(const Mirror *mirror, json_object *plan, int verify, size_t left) {
    json_object *done = mirror_jsonl(mirror->done, 1);
    json_object *keys = done ? mirror_done_keys(done) : NULL;
    int ok = keys != NULL;
    size_t missing = 0, absent = 0, checked = 0, same = 0;
    for (size_t i = 0; ok && i < elements(plan); i++) {
        const char *key = text_field(element(plan, i), "key");
        ok = key != NULL;
        missing += ok && !value_of(keys, key);
    }
    if (ok) {
        Cursor cursor = entries(keys);
        Node *value;
        const char *key;
        while ((key = next_entry(&cursor, &value))) {
            char *held = mirror_held(mirror, key);
            absent += !held || access(held, F_OK);
            free(held);
        }
        printf("%zu %s %zu held %zu missing %zu absent on disc\n", elements(plan),
            mirror->server ? "wanted" : "keys published and wanted", fields(keys), missing, absent);
        printf("%.2f GB published %.2f GB held\n", mirror_sum(done, "bytes") / 1e9,
            mirror_sum(done, "stored") / 1e9);
    }
    if (ok && verify) {
        ok = mirror_verify(mirror, done, &checked, &same);
        if (ok) {
            printf("%zu of %zu identical to %s when decoded\n", same, checked,
                mirror->server ? "the server" : "published");
            if (!mirror->server) {
                printf("%zu brick stores left out\n", left);
            }
        }
    }
    drop(done);
    drop(keys);
    return ok;
}

static int mirror_cli(int argc, char **argv, int server) {
    const char *action = argc > 1 ? argv[1] : "--help";
    const char *url = argc == 4 && !strcmp(argv[2], "--root") ? argv[3]
        : argc == 2                                           ? (server ? inventory_server : bucket)
                                                              : NULL;
    int make = !strcmp(action, "plan") || (!server && !strcmp(action, "walk"));
    int fetch = !strcmp(action, "fetch"), status = !strcmp(action, "status");
    int verify = !strcmp(action, "verify");
    if (!url || (!make && !fetch && !status && !verify)) {
        const char *name = server ? "fetch_server" : "fetch_products";
        printf("Usage: ./run.sh scans.transfer.%s %s|fetch|status|verify [--root URL]\n", name,
            server ? "plan" : "walk");
        return argc == 1 || inventory_help(argc, argv) ? 0 : 2;
    }
    char *library = local_root(library_root);
    char *catalog = library ? path_join(library, "catalog") : NULL;
    Mirror mirror = {.root = library
            ? path_join(library, server ? "published/dl_ash2txt" : "published/open_data")
            : NULL,
        .origin = inventory_base(url),
        .server = server};
    if (mirror.root) {
        mirror.plan = path_join(mirror.root, "plan.jsonl");
        mirror.done = path_join(mirror.root, "done.jsonl");
        mirror.unfinished = path_join(mirror.root, "unfinished.json");
    }
    int curl = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    int result = 1;
    if (curl && catalog && mirror.origin && mirror.plan && mirror.done && mirror.unfinished) {
        if (fetch) {
            result = mirror_fetch(&mirror);
        } else {
            size_t left = 0;
            json_object *plan = !(make || verify) ? mirror_jsonl(mirror.plan, 0)
                : server                          ? mirror_plan_server(&mirror, catalog)
                                                  : mirror_plan_products(&mirror, catalog, &left);
            if (plan && make) {
                printf("%zu %s %.2f GB", elements(plan), server ? "files" : "keys",
                    mirror_sum(plan, "bytes") / 1e9);
                if (server) {
                    size_t count[5] = {0};
                    for (size_t i = 0; i < elements(plan); i++) {
                        int rank = json_object_get_int(value_of(element(plan, i), "rank"));
                        count[rank]++;
                    }
                    printf("\n[");
                    int first = 1;
                    for (size_t i = 0; i < 5; i++) {
                        if (count[i]) {
                            printf("%s(%zu, %zu)", first ? "" : ", ", i, count[i]);
                            first = 0;
                        }
                    }
                    puts("]");
                } else {
                    printf(" %zu brick stores left out\n", left);
                }
                result = 0;
            } else if (plan) {
                result = !mirror_status(&mirror, plan, verify, left);
            } else {
                fprintf(stderr, "HerculeX: no usable mirror plan %s\n", mirror.plan);
            }
            drop(plan);
        }
    }
    if (curl) {
        curl_global_cleanup();
    }
    free(mirror.unfinished);
    free(mirror.done);
    free(mirror.plan);
    free(mirror.origin);
    free(mirror.root);
    free(catalog);
    free(library);
    return result;
}

int hx_fetch_products_main(int argc, char **argv) { return mirror_cli(argc, argv, 0); }
int hx_fetch_server_main(int argc, char **argv) { return mirror_cli(argc, argv, 1); }
