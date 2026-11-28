#include <blosc.h>

enum { SPOT_RENDER_LAYERS = 67, SPOT_GROW_ROUNDS = 15, SPOT_GROW_REACH = 6, SPOT_SNAP_REACH = 15 };

static const char *spot_surface_names[] = {"original", "m7", "grown"};

static char *spot_surface_work(Arena *a, json_object *spot) {
    const char *id = site_string(spot, "spot_id");
    need(site_name(id), "invalid spot ID");
    return fmt(a, "%s/studies/shared/spot-candidates/spots/%s", LIBRARY, id);
}

typedef struct {
    const char *root, *folder;
    i64 shape[3], chunk[3], grid[3];
    size_t bytes;
} SpotPrediction;

typedef struct {
    const char *folder;
    const SpotPrediction *prediction;
} SpotPredictionFetch;

static void spot_chunk_key(const i64 grid[3], i64 ordinal, char out[64], char sep) {
    i64 z = ordinal / (grid[1] * grid[2]), y = ordinal / grid[2] % grid[1];
    i64 x = ordinal % grid[2];
    snprintf(out, 64, "%lld%c%lld%c%lld", (long long)z, sep, (long long)y, sep, (long long)x);
}

static void spot_prediction_deliver(void *context, Ask *ask, const u8 *body, size_t bytes) {
    const SpotPredictionFetch *fetch = context;
    char path[PATH_MAX], key[64];
    if (ask->index < 0) {
        snprintf(key, sizeof(key), ".zarray");
        need(body != NULL, "could not fetch m7 metadata");
    } else {
        spot_chunk_key(fetch->prediction->grid, ask->index, key, '.');
    }
    int length = snprintf(path, sizeof(path), "%s/%s", fetch->folder, key);
    need(length >= 0 && length < (int)sizeof(path), "m7 chunk path is too long");
    need(hx_write_sheet_file(path, (void *)body, body ? bytes : 0), "could not cache %s", path);
}

static void spot_zarr_shape(json_object *meta, i64 shape[3], i64 chunks[3]) {
    json_object *sizes = json_object_object_get(meta, "shape");
    json_object *tiles = json_object_object_get(meta, "chunks");
    need(site_items(sizes) == 3 && site_items(tiles) == 3, "invalid spot Zarr array shape");
    for (size_t i = 0; i < 3; i++) {
        shape[i] = json_object_get_int64(json_object_array_get_idx(sizes, i));
        chunks[i] = json_object_get_int64(json_object_array_get_idx(tiles, i));
        need(shape[i] > 0 && chunks[i] > 0 && shape[i] <= INT32_MAX && chunks[i] <= INT32_MAX,
            "invalid spot Zarr array dimension");
    }
}

static SpotPrediction spot_prediction(Arena *a, const SpotSource *sheet) {
    char *products = fmt(a, "%s/catalog/products.jsonl", LIBRARY);
    FILE *stream = fopen(products, "r");
    need(stream != NULL, "cannot open %s", products);
    char *line = NULL;
    size_t capacity = 0;
    char *root = NULL;
    while (!root && getline(&line, &capacity, stream) >= 0) {
        if (!strstr(line, "\"surface-prediction-zarr\"")) {
            continue;
        }
        json_object *row = json_tokener_parse(line);
        const char *sample = site_string(row, "sample"), *owner = site_string(row, "owner");
        const char *path = site_string(row, "path"), *bucket = site_string(row, "root");
        if (sample && !strcmp(sample, sheet->scroll) && owner && !strcmp(owner, sheet->scan) &&
            path && strstr(path, "-L0-") && bucket) {
            while (*path == '/') {
                path++;
            }
            size_t size = strlen(path);
            while (size && path[size - 1] == '/') {
                size--;
            }
            root = fmt(a, "%s/%.*s/0", bucket, (int)size, path);
        }
        json_object_put(row);
    }
    free(line);
    fclose(stream);
    need(root != NULL, "%s %s: no m7 prediction in %s", sheet->scroll, sheet->scan, products);
    const char *folder =
        fmt(a, "%s/studies/shared/spot-candidates/m7/%s/%s", LIBRARY, sheet->scroll, sheet->scan);
    mkdirs(folder);
    const char *metadata = fmt(a, "%s/.zarray", folder);
    SpotPrediction prediction = {.root = root, .folder = folder, .bytes = 1};
    if (!exists(metadata)) {
        int lock = lock_file(fmt(a, "%s/.meta.lock", folder), 1, 1);
        need(lock >= 0, "cannot lock m7 metadata");
        if (!exists(metadata)) {
            Ask ask = {.index = -1, .path = ".zarray", .first = -1};
            SpotPredictionFetch fetch = {folder, &prediction};
            need(!fetch_ranges(root, &ask, 1, spot_prediction_deliver, &fetch),
                "could not fetch %s/.zarray", root);
        }
        close(lock);
    }
    json_object *array = json_read(metadata);
    spot_zarr_shape(array, prediction.shape, prediction.chunk);
    for (int i = 0; i < 3; i++) {
        i64 size = prediction.chunk[i];
        prediction.grid[i] = (prediction.shape[i] + size - 1) / size;
        need((size_t)size <= INT_MAX / prediction.bytes, "m7 chunk is too large");
        prediction.bytes *= size;
    }
    need(prediction.grid[0] <= INT64_MAX / prediction.grid[1] / prediction.grid[2],
        "m7 chunk grid is too large");
    json_object_put(array);
    return prediction;
}

typedef struct {
    i64 chunk;
    size_t at, bytes;
} SpotSample;

static int spot_chunk_order(const void *left, const void *right) {
    const SpotSample *a = left, *b = right;
    return (a->chunk > b->chunk) - (a->chunk < b->chunk);
}

static void spot_prediction_values(
    Arena *a, const SpotPrediction *pred, SpotSample *samples, size_t count, u8 *values) {
    if (!count) {
        return;
    }
    qsort(samples, count, sizeof(*samples), spot_chunk_order);
    size_t mark = a->used, groups = 1;
    for (size_t i = 1; i < count; i++) {
        groups += samples[i].chunk != samples[i - 1].chunk;
    }
    Ask *asks = push(a, groups * sizeof(*asks));
    size_t ask_count = 0;
    char file[PATH_MAX];
    for (size_t i = 0; i < count; i++) {
        if (i && samples[i - 1].chunk == samples[i].chunk) {
            continue;
        }
        char key[64];
        i64 chunk = samples[i].chunk;
        spot_chunk_key(pred->grid, chunk, key, '.');
        int length = snprintf(file, sizeof(file), "%s/%s", pred->folder, key);
        need(length >= 0 && length < (int)sizeof(file), "m7 chunk path is too long");
        if (!exists(file)) {
            Ask *ask = &asks[ask_count++];
            *ask = (Ask){.index = chunk, .first = -1};
            spot_chunk_key(pred->grid, chunk, ask->path, '/');
        }
    }
    if (ask_count) {
        SpotPredictionFetch fetch = {pred->folder, pred};
        i64 failed = fetch_ranges(pred->root, asks, ask_count, spot_prediction_deliver, &fetch);
        need(!failed, "%lld m7 chunks could not be fetched", (long long)failed);
    }
    u8 *decoded = push(a, pred->bytes);
    size_t spare = a->used;
    for (size_t i = 0; i < count;) {
        char key[64];
        spot_chunk_key(pred->grid, samples[i].chunk, key, '.');
        Bytes compressed = read_file(a, fmt(a, "%s/%s", pred->folder, key));
        if (compressed.size) {
            size_t size = 0;
            need(blosc_cbuffer_validate(compressed.data, compressed.size, &size) == 0 &&
                    size == pred->bytes &&
                    blosc_decompress_ctx(compressed.data, decoded, size, 1) == (int)size,
                "invalid m7 chunk %s", key);
        }
        size_t j = i;
        while (j < count && samples[j].chunk == samples[i].chunk) {
            values[samples[j].at] = compressed.size && decoded[samples[j].bytes] > 0;
            j++;
        }
        a->used = spare;
        i = j;
    }
    a->used = mark;
}

static f32 spot_run_centre(const u8 *profile, int width) {
    f32 nearest = NAN;
    for (int i = 0; i < width;) {
        if (!profile[i]) {
            i++;
            continue;
        }
        int start = i++;
        while (i < width && profile[i]) {
            i++;
        }
        int middle = width / 2;
        f32 centre = (start + i - 1) * .5f - middle;
        if (!isfinite(nearest) || fabsf(centre) < fabsf(nearest)) {
            nearest = centre;
        }
    }
    if (!isfinite(nearest)) {
        return nearest;
    }
    int middle = width / 2;
    int c = fmax(1, fmin(width - 2, nearbyint(nearest + middle)));
    f32 smoothed[3] = {0};
    for (int pos = c - 1; pos <= c + 1; pos++) {
        smoothed[pos - c + 1] = (pos ? profile[pos - 1] : 0) + 2 * profile[pos] +
            (pos + 1 < width ? profile[pos + 1] : 0);
    }
    double den = smoothed[0] - 2 * smoothed[1] + smoothed[2];
    double shifted = fabs(den) > 1e-6 ? .5 * (smoothed[0] - smoothed[2]) / den : 0;
    return fabs(shifted) > .5 ? nearest : .5 * (nearest + c - middle + shifted);
}

static void spot_profiles(Arena *a, const SpotPrediction *pred, const Surface *surface,
    const f32 *normal, const u8 *only, int radius, f32 *offsets) {
    size_t mark = a->used, nodes = (size_t)surface->h * surface->w, valid = 0;
    int width = 2 * radius + 1;
    for (size_t i = 0; i < nodes; i++) {
        valid += surface->xyz[i] > 0 && (!only || only[i]);
        offsets[i] = NAN;
    }
    SpotSample *wanted = push(a, valid * width * sizeof(*wanted));
    u8 *profile = pushz(a, valid * width);
    size_t n = 0, row = 0;
    for (size_t i = 0; i < nodes; i++) {
        if (!(surface->xyz[i] > 0) || (only && !only[i])) {
            continue;
        }
        for (int d = -radius; d <= radius; d++) {
            i64 xyz[3], chunk[3], within[3];
            int inside = 1;
            for (int c = 0; c < 3; c++) {
                size_t at = (size_t)(2 - c) * nodes + i;
                f32 position = surface->xyz[at] + (f32)d * normal[at];
                if (!isfinite(position) || position <= -2 || position >= pred->shape[c] + 1) {
                    inside = 0;
                    break;
                }
                xyz[c] = nearbyintf(position);
                if (xyz[c] < 0 || xyz[c] >= pred->shape[c]) {
                    inside = 0;
                    break;
                }
                chunk[c] = xyz[c] / pred->chunk[c];
                within[c] = xyz[c] % pred->chunk[c];
            }
            if (inside) {
                i64 ordinal = (chunk[0] * pred->grid[1] + chunk[1]) * pred->grid[2] + chunk[2];
                size_t bytes =
                    (within[0] * pred->chunk[1] + within[1]) * pred->chunk[2] + within[2];
                wanted[n++] = (SpotSample){ordinal, row * width + d + radius, bytes};
            }
        }
        row++;
    }
    spot_prediction_values(a, pred, wanted, n, profile);
    row = 0;
    for (size_t i = 0; i < nodes; i++) {
        if (surface->xyz[i] > 0 && (!only || only[i])) {
            offsets[i] = spot_run_centre(profile + row * width, width);
            row++;
        }
    }
    a->used = mark;
}

static void spot_normal(Arena *a, Surface *grid, const f64 *axis, int rows, f32 *normal) {
    size_t mark = a->used, n = (size_t)grid->h * grid->w;
    f32 *values = push(a, n * sizeof(*values));
    size_t used = 0;
    for (size_t i = 0; i < n; i++) {
        if (grid->xyz[i] > 0) {
            values[used++] = grid->xyz[2 * n + i];
        }
    }
    need(used, "spot surface has no nodes");
    qsort(values, used, sizeof(*values), by_float);
    double middle = used % 2 ? values[used / 2] : (values[used / 2 - 1] + values[used / 2]) / 2.f;
    grid_normals(grid, interpolated(middle, axis, rows, 1), interpolated(middle, axis, rows, 2),
        normal, values);
    a->used = mark;
}

static void spot_normconv(Arena *a, const f32 *source, int w, int h, f32 *result) {
    size_t mark = a->used, n = (size_t)w * h;
    f32 *weight = push(a, n * sizeof(*weight)), *scratch = push(a, n * sizeof(*scratch));
    for (size_t i = 0; i < n; i++) {
        int keep = isfinite(source[i]);
        weight[i] = keep;
        result[i] = keep ? source[i] : 0;
    }
    gaussian2(result, scratch, h, w, 1);
    gaussian2(weight, scratch, h, w, 1);
    for (size_t i = 0; i < n; i++) {
        result[i] = weight[i] > 1e-3f ? result[i] / weight[i] : NAN;
    }
    a->used = mark;
}

static json_object *spot_snap(
    Arena *a, Surface *grid, const f32 *normal, const SpotPrediction *prediction) {
    size_t mark = a->used, n = (size_t)grid->h * grid->w;
    f32 *field = push(a, n * sizeof(*field)), *kept = push(a, n * sizeof(*kept));
    f32 *filled = push(a, n * sizeof(*filled)), *whole = push(a, n * sizeof(*whole));
    spot_profiles(a, prediction, grid, normal, NULL, SPOT_SNAP_REACH, field);
    size_t near = 0, seen = 0, rejected = 0, valid = 0;
    int w = grid->w, h = grid->h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t at = (size_t)y * w + x;
            kept[at] = field[at];
            valid += grid->xyz[at] > 0;
            if (!isfinite(field[at])) {
                continue;
            }
            seen++;
            near += fabsf(field[at]) <= 2.f;
            f32 values[25];
            int used = 0;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int ry = y + dy, cx = x + dx;
                    if (ry >= 0 && ry < h && cx >= 0 && cx < w) {
                        float value = field[(size_t)ry * w + cx];
                        if (isfinite(value)) {
                            values[used++] = value;
                        }
                    }
                }
            }
            qsort(values, used, sizeof(*values), by_float);
            f32 median =
                used % 2 ? values[used / 2] : (values[used / 2 - 1] + values[used / 2]) / 2.f;
            if (fabsf(field[at] - median) > 6.f) {
                kept[at] = NAN;
                rejected++;
            }
        }
    }
    spot_normconv(a, kept, w, h, filled);
    for (size_t i = 0; i < n; i++) {
        whole[i] = isfinite(kept[i]) ? kept[i] : filled[i];
    }
    spot_normconv(a, whole, w, h, kept);
    size_t changed = 0;
    for (size_t i = 0; i < n; i++) {
        if (!(grid->xyz[i] > 0)) {
            for (int c = 0; c < 3; c++) {
                grid->xyz[(size_t)c * n + i] = -1;
            }
            continue;
        }
        if (isfinite(kept[i]) && isfinite(filled[i])) {
            f32 delta = fmaxf(-SPOT_SNAP_REACH, fminf(SPOT_SNAP_REACH, kept[i]));
            for (int c = 0; c < 3; c++) {
                size_t at = (size_t)c * n + i;
                grid->xyz[at] += delta * normal[at];
            }
            field[changed++] = fabsf(delta);
        }
    }
    qsort(field, changed, sizeof(*field), by_float);
    json_object *report = json_object_new_object();
    json_object_object_add(report, "nodes", json_object_new_int64(valid));
    size_t base = valid ? valid : 1, shown = seen ? seen : 1;
    json_object_object_add(report, "with_sheet", spot_number((double)seen / base, 4));
    json_object_object_add(report, "snapped", spot_number((double)changed / base, 4));
    json_object_object_add(report, "rejected", spot_number((double)rejected / shown, 4));
    if (changed) {
        double median =
            changed % 2 ? field[changed / 2] : (field[changed / 2] + field[changed / 2 - 1]) / 2.f;
        double at = .9 * (changed - 1), between = at - floor(at);
        f32 low = field[(size_t)at], high = field[(size_t)ceil(at)];
        double p90 =
            between < .5 ? low + (high - low) * between : high - (high - low) * (1 - between);
        json_object_object_add(report, "median_move_vox", spot_number(median, 2));
        json_object_object_add(report, "p90_move_vox", spot_number(p90, 2));
    } else {
        json_object_object_add(report, "median_move_vox", NULL);
        json_object_object_add(report, "p90_move_vox", NULL);
    }
    json_object_object_add(report, "already_on", spot_number((double)near / shown, 4));
    a->used = mark;
    return report;
}

static int spot_narrow(json_object *piece, const SpotOutline *outline, int crop[4]) {
    json_object *shape = json_object_object_get(piece, "shape");
    int h = json_object_get_int(json_object_array_get_idx(shape, 0));
    int w = json_object_get_int(json_object_array_get_idx(shape, 1));
    double step = site_value(piece, "stride") * site_value(piece, "um");
    double borders[2][2] = {
        {outline->top - 3, outline->bottom + 3}, {outline->left - 3, outline->right + 3}};
    const char *places[] = {"y_mm", "x_mm"}, *flips[] = {"flip_rows", "flip_cols"};
    int sizes[] = {h, w};
    for (int d = 0; d < 2; d++) {
        int low = sizes[d], high = 0;
        int flipped = json_object_get_boolean(json_object_object_get(piece, flips[d]));
        double origin = site_value(piece, places[d]);
        for (int i = 0; i < sizes[d]; i++) {
            double mm = origin + (flipped ? sizes[d] - 1 - i : i) * step;
            if (mm >= borders[d][0] && mm <= borders[d][1]) {
                if (i < low) {
                    low = i;
                }
                high = i + 1;
            }
        }
        if (high <= low) {
            return 0;
        }
        crop[2 * d] = low;
        crop[2 * d + 1] = high;
    }
    return 1;
}

static Surface spot_surface_crop(Arena *a, Surface surface, const int crop[4]) {
    int h = crop[1] - crop[0], w = crop[3] - crop[2];
    need(h > 0 && w > 0 && crop[0] >= 0 && crop[2] >= 0 && crop[1] <= surface.h &&
            crop[3] <= surface.w,
        "invalid spot surface crop");
    size_t n = (size_t)h * w, old = (size_t)surface.h * surface.w;
    f32 *xyz = push(a, 3 * n * sizeof(*xyz));
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            memcpy(xyz + c * n + (size_t)y * w,
                surface.xyz + c * old + (size_t)(crop[0] + y) * surface.w + crop[2],
                (size_t)w * sizeof(*xyz));
        }
    }
    return (Surface){.h = h, .w = w, .stride = surface.stride, .voxel = surface.voxel, .xyz = xyz};
}

static void spot_save_surface(Arena *a, Surface grid, const char *folder, json_object *meta) {
    size_t n = (size_t)grid.h * grid.w, mark = a->used;
    double low[] = {INFINITY, INFINITY, INFINITY}, high[] = {-INFINITY, -INFINITY, -INFINITY};
    for (int c = 0; c < 3; c++) {
        f32 *plane = grid.xyz + c * n;
        for (size_t i = 0; i < n; i++) {
            if (grid.xyz[i] > 0) {
                low[c] = fmin(low[c], plane[i]);
                high[c] = fmax(high[c], plane[i]);
            }
        }
        tiff_write(fmt(a, "%s/%c.tif", folder, "xyz"[c]), plane, grid.w, grid.h);
    }
    if (isfinite(low[0])) {
        json_object *bbox = json_object_new_array_ext(2);
        for (int d = 0; d < 2; d++) {
            json_object *row = json_object_new_array_ext(3);
            for (int c = 0; c < 3; c++) {
                json_object_array_add(row, jnum(d ? high[c] : low[c]));
            }
            json_object_array_add(bbox, row);
        }
        json_object_object_add(meta, "bbox", bbox);
    }
    json_write(fmt(a, "%s/meta.json", folder), meta);
    a->used = mark;
}

static json_object *spot_crop_json(const int crop[4]) {
    json_object *result = json_object_new_array_ext(4);
    for (int i = 0; i < 4; i++) {
        json_object_array_add(result, json_object_new_int(crop[i]));
    }
    return result;
}

static void spot_read_crop(json_object *crop, int out[4]) {
    need(site_items(crop) == 4, "spot piece has an invalid crop");
    for (size_t i = 0; i < 4; i++) {
        out[i] = json_object_get_int(json_object_array_get_idx(crop, i));
    }
}

int hx_umbilicus_keep(const char *sample, const char *scan);

static json_object *spot_axis_doc(Arena *a, const SpotSource *sheet) {
    size_t length = (size_t)(strrchr(sheet->pieces, '/') - sheet->pieces);
    char *path = fmt(a, "%.*s/sheet/axis.json", (int)length, sheet->pieces);
    if (!exists(path)) {
        need(!hx_umbilicus_keep(sheet->scroll, sheet->scan), "no axis for %s %s", sheet->scroll,
            sheet->scan);
        path = fmt(
            a, "%s/unwrapping/level/%s/%s/sheet/axis.json", OUTPUTS, sheet->scroll, sheet->scan);
    }
    return json_read(path);
}

static f64 *spot_read_axis(Arena *a, json_object *axis, int *rows) {
    size_t n = site_items(axis);
    need(n && n <= INT_MAX, "spot has no usable umbilicus");
    f64 *coords = push(a, n * 3 * sizeof(*coords));
    for (size_t i = 0; i < n; i++) {
        json_object *point = json_object_array_get_idx(axis, i);
        need(site_items(point) == 3, "invalid spot umbilicus");
        for (size_t k = 0; k < 3; k++) {
            coords[i * 3 + k] = json_object_get_double(json_object_array_get_idx(point, k));
        }
    }
    *rows = n;
    return coords;
}

static const char *spot_written_side(Arena *a, const SpotSource *sheet, json_object *piece) {
    const char *unit = site_string(piece, "unit"), *slash = strrchr(sheet->pieces, '/');
    int id = site_value(piece, "piece");
    char *path = fmt(a, "%.*s/faces/level-0/%s/piece-%04d-taught-side.json",
        (int)(slash - sheet->pieces), sheet->pieces, unit, id);
    json_object *side = json_object_from_file(path);
    const char *written = site_string(side, "written"), *result = NULL;
    if (side) {
        json_object *turn = json_object_object_get(side, "turn");
        if (!turn) {
            turn = json_object_object_get(piece, "turn");
        }
        int inward =
            (!turn || json_object_get_int(turn) == 1) == (written && !strcmp(written, "near"));
        result = inward ? "inward" : "outward";
    }
    json_object_put(side);
    return result;
}

json_object *spot_surface_build(Arena *a, json_object *spot, const SpotSource *sheet) {
    size_t then = a->used;
    const char *work = spot_surface_work(a, spot);
    json_object *axis_doc = spot_axis_doc(a, sheet);
    int axis_rows;
    f64 *axis = spot_read_axis(a, axis_doc, &axis_rows);
    SpotPrediction prediction = spot_prediction(a, sheet);
    const char *key = site_string(spot, "layout");
    need(site_name(key), "spot has no saved layout");
    json_object *layout =
        json_read(fmt(a, "%s/marks/%s/layouts/%s.json", LIBRARY, sheet->key, key));
    json_object *placed = json_object_object_get(layout, "pieces");
    json_object *selected = json_object_object_get(spot, "pieces");
    json_object *records = json_object_new_array();
    SpotOutline outline = spot_outline_at(a, json_object_object_get(spot, "outline_mm"), 0);
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(selected); i++) {
        a->used = mark;
        json_object *entry = json_object_array_get_idx(selected, i);
        if (site_value(entry, "share") < .08) {
            continue;
        }
        const char *unit = site_string(entry, "unit");
        int id = site_value(entry, "piece");
        need(site_name(unit) && id >= 0, "invalid spot piece");
        json_object *p = NULL;
        for (size_t j = 0; j < site_items(placed); j++) {
            json_object *piece = json_object_array_get_idx(placed, j);
            const char *name = site_string(piece, "unit");
            if (name && !strcmp(name, unit) && site_value(piece, "piece") == id) {
                p = piece;
                break;
            }
        }
        need(p != NULL, "unknown spot piece %s/piece-%04d", unit, id);
        int crop[4];
        if (!spot_narrow(p, &outline, crop)) {
            continue;
        }
        const char *folder = fmt(a, "%s/%s/piece-%04d", sheet->pieces, unit, id);
        Surface xyz = spot_surface_crop(a, load_surface(a, folder), crop);
        size_t n = (size_t)xyz.w * xyz.h, valid = 0;
        for (size_t j = 0; j < n; j++) {
            valid += xyz.xyz[j] > 0;
        }
        if (valid < 16) {
            continue;
        }
        const char *name = fmt(a, "%s-piece-%04d", unit, id);
        json_object *meta = json_read(fmt(a, "%s/meta.json", folder));
        json_object_object_add(meta, "crop", spot_crop_json(crop));
        json_object_object_add(meta, "source", json_object_new_string(folder));
        f32 *normal = push(a, 3 * n * sizeof(*normal));
        spot_normal(a, &xyz, axis, axis_rows, normal);
        spot_save_surface(a, xyz, fmt(a, "%s/surfaces/original/%s", work, name), meta);
        json_object *report = spot_snap(a, &xyz, normal, &prediction);
        json_object_object_add(meta, "snap", json_object_get(report));
        spot_save_surface(a, xyz, fmt(a, "%s/surfaces/m7/%s", work, name), meta);
        json_object *record = json_object_new_object(), *shown = json_object_new_object();
        const char *keep[] = {
            "x_mm", "y_mm", "flip_rows", "flip_cols", "um", "stride", "shape", "area"};
        for (size_t j = 0; j < sizeof(keep) / sizeof(*keep); j++) {
            json_object_object_add(
                shown, keep[j], json_object_get(json_object_object_get(p, keep[j])));
        }
        const char *side = spot_written_side(a, sheet, p);
        json_object_object_add(record, "name", json_object_new_string(name));
        json_object_object_add(record, "unit", json_object_new_string(unit));
        json_object_object_add(record, "piece", json_object_new_int(id));
        json_object_object_add(record, "crop", spot_crop_json(crop));
        json_object_object_add(
            record, "share", json_object_get(json_object_object_get(entry, "share")));
        json_object_object_add(
            record, "winding", json_object_get(json_object_object_get(entry, "winding")));
        json_object_object_add(record, "reading_side", side ? json_object_new_string(side) : NULL);
        json_object_object_add(record, "snap", report);
        json_object_object_add(record, "layout", shown);
        json_object_array_add(records, record);
        printf("%s %s crop (%d, %d, %d, %d) snap %s\n", site_string(spot, "spot_id"), name, crop[0],
            crop[1], crop[2], crop[3],
            json_object_to_json_string_ext(report, JSON_C_TO_STRING_PLAIN));
        json_object_put(meta);
    }
    json_object_put(layout);
    json_object *doc = json_object_new_object();
    json_object_object_add(doc, "axis", axis_doc);
    json_object_object_add(doc, "pieces", records);
    json_write(fmt(a, "%s/pieces.json", work), doc);
    a->used = then;
    return doc;
}

static void spot_growth_spacing(Arena *a, Surface *wide, const u8 *recent) {
    size_t mark = a->used, count = (size_t)wide->h * wide->w;
    u8 *drop = pushz(a, count);
    int w = wide->w, h = wide->h;
    double small = .5 * wide->stride, large = 1.6 * wide->stride;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t at = (size_t)y * w + x;
            if (!recent[at] || !(wide->xyz[at] > 0)) {
                continue;
            }
            size_t near[] = {y > 0 ? at - w : at, y + 1 < h ? at + w : at, x > 0 ? at - 1 : at,
                x + 1 < w ? at + 1 : at};
            int linked = 0, fine = 1;
            for (int k = 0; fine && k < 4; k++) {
                size_t peer = near[k];
                if (peer == at || !(wide->xyz[peer] > 0)) {
                    continue;
                }
                f32 distance = 0;
                for (int c = 0; c < 3; c++) {
                    f32 diff = wide->xyz[c * count + at] - wide->xyz[c * count + peer];
                    distance += diff * diff;
                }
                distance = sqrtf(distance);
                fine &= distance >= small && distance <= large;
                linked |= fine;
            }
            drop[at] = !linked || !fine;
        }
    }
    for (size_t i = 0; i < count; i++) {
        if (drop[i]) {
            for (int c = 0; c < 3; c++) {
                wide->xyz[c * count + i] = -1;
            }
        }
    }
    a->used = mark;
}

static Surface spot_grown(Arena *a, Surface grid, int crop[4], json_object *layout,
    const SpotOutline *frame, const f64 *axis, int rows, const SpotPrediction *prediction,
    json_object **report) {
    size_t before = 0;
    for (size_t i = 0, n = (size_t)grid.h * grid.w; i < n; i++) {
        before += grid.xyz[i] > 0;
    }
    json_object *rounds = json_object_new_array_ext(SPOT_GROW_ROUNDS);
    json_object *shape = json_object_object_get(layout, "shape");
    int height = json_object_get_int(json_object_array_get_idx(shape, 0));
    int width = json_object_get_int(json_object_array_get_idx(shape, 1));
    double x0 = site_value(layout, "x_mm"), y0 = site_value(layout, "y_mm");
    double step = site_value(layout, "um") * site_value(layout, "stride");
    int flip_y = json_object_get_boolean(json_object_object_get(layout, "flip_rows"));
    int flip_x = json_object_get_boolean(json_object_object_get(layout, "flip_cols"));
    int pad = 0;
    for (int round = 0; round < SPOT_GROW_ROUNDS; round++) {
        Surface wider = extended(a, grid);
        int margin = (wider.w - grid.w) / 2;
        size_t mark = a->used, nodes = (size_t)wider.h * wider.w, added = 0;
        u8 *recent = pushz(a, nodes);
        pad += margin;
        int row0 = crop[0] - pad, col0 = crop[2] - pad;
        for (int y = 0; y < wider.h; y++) {
            int sy = y - margin, row = row0 + y;
            double py = y0 + step * (flip_y ? height - 1 - row : row);
            for (int x = 0; x < wider.w; x++) {
                size_t i = (size_t)y * wider.w + x;
                int sx = x - margin, col = col0 + x;
                int old = sy >= 0 && sy < grid.h && sx >= 0 && sx < grid.w &&
                    grid.xyz[(size_t)sy * grid.w + sx] > 0;
                if (!(wider.xyz[i] > 0) || old) {
                    continue;
                }
                double px = x0 + step * (flip_x ? width - 1 - col : col);
                if (px < frame->left - 3 || px > frame->right + 3 || py < frame->top - 3 ||
                    py > frame->bottom + 3) {
                    for (int c = 0; c < 3; c++) {
                        wider.xyz[c * nodes + i] = -1;
                    }
                    continue;
                }
                recent[i] = 1;
                added++;
            }
        }
        grid = wider;
        if (!added) {
            json_object_array_add(rounds, json_object_new_int(0));
            a->used = mark;
            break;
        }
        f32 *normal = push(a, 3 * nodes * sizeof(*normal));
        f32 *offsets = push(a, nodes * sizeof(*offsets));
        spot_normal(a, &grid, axis, rows, normal);
        spot_profiles(a, prediction, &grid, normal, recent, SPOT_GROW_REACH, offsets);
        for (size_t i = 0; i < nodes; i++) {
            if (recent[i]) {
                int valid = isfinite(offsets[i]);
                for (int c = 0; c < 3; c++) {
                    size_t at = c * nodes + i;
                    grid.xyz[at] = valid ? grid.xyz[at] + offsets[i] * normal[at] : -1;
                }
            }
        }
        for (int k = 0; k < 2; k++) {
            spot_growth_spacing(a, &grid, recent);
        }
        added = 0;
        for (size_t i = 0; i < nodes; i++) {
            added += recent[i] && grid.xyz[i] > 0;
        }
        json_object_array_add(rounds, json_object_new_int64(added));
        a->used = mark;
    }
    size_t mark = a->used;
    f32 *normal = push(a, 3 * (size_t)grid.h * grid.w * sizeof(*normal));
    spot_normal(a, &grid, axis, rows, normal);
    *report = spot_snap(a, &grid, normal, prediction);
    a->used = mark;
    int keep[] = {grid.h, 0, grid.w, 0};
    for (int y = 0; y < grid.h; y++) {
        for (int x = 0; x < grid.w; x++) {
            if (grid.xyz[(size_t)y * grid.w + x] > 0) {
                keep[0] = keep[0] < y ? keep[0] : y;
                keep[1] = keep[1] > y ? keep[1] : y + 1;
                keep[2] = keep[2] < x ? keep[2] : x;
                keep[3] = keep[3] > x ? keep[3] : x + 1;
            }
        }
    }
    int row = crop[0] - pad, col = crop[2] - pad;
    crop[0] = row + keep[0];
    crop[1] = row + keep[1];
    crop[2] = col + keep[2];
    crop[3] = col + keep[3];
    json_object_object_add(*report, "added_by_round", rounds);
    json_object_object_add(*report, "nodes_before", json_object_new_int64(before));
    return spot_surface_crop(a, grid, keep);
}

void spot_surface_grow(Arena *a, json_object *spot, json_object *work_doc) {
    size_t then = a->used;
    char *work = spot_surface_work(a, spot);
    int rows;
    f64 *axis = spot_read_axis(a, json_object_object_get(work_doc, "axis"), &rows);
    SpotSource sheet = {.scroll = site_string(spot, "scroll"), .scan = site_string(spot, "scan")};
    SpotPrediction prediction = spot_prediction(a, &sheet);
    SpotOutline frame = spot_outline_at(a, json_object_object_get(spot, "outline_mm"), 0);
    json_object *pieces = json_object_object_get(work_doc, "pieces");
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(pieces); i++) {
        a->used = mark;
        json_object *record = json_object_array_get_idx(pieces, i);
        const char *name = site_string(record, "name");
        need(site_name(name), "invalid spot surface name");
        const char *folder = fmt(a, "%s/surfaces/m7/%s", work, name);
        int crop[4];
        spot_read_crop(json_object_object_get(record, "crop"), crop);
        Surface original = load_surface(a, folder);
        json_object *report = NULL, *layout = json_object_object_get(record, "layout");
        Surface grown =
            spot_grown(a, original, crop, layout, &frame, axis, rows, &prediction, &report);
        json_object *meta = json_read(fmt(a, "%s/meta.json", folder));
        json_object_object_add(meta, "crop", spot_crop_json(crop));
        json_object_object_add(meta, "grow", json_object_get(report));
        spot_save_surface(a, grown, fmt(a, "%s/surfaces/grown/%s", work, name), meta);
        json_object *crops = json_object_object_get(record, "crops");
        if (!crops) {
            crops = json_object_new_object();
            json_object_object_add(record, "crops", crops);
        }
        json_object_object_add(crops, "grown", spot_crop_json(crop));
        json_object *summary = json_object_new_object();
        const char *keys[] = {"nodes", "nodes_before", "added_by_round", "median_move_vox"};
        for (size_t j = 0; j < sizeof(keys) / sizeof(*keys); j++) {
            json_object_object_add(
                summary, keys[j], json_object_get(json_object_object_get(report, keys[j])));
        }
        json_object_object_add(record, "grow", summary);
        printf("%s %s grown from %lld to %lld nodes\n", site_string(spot, "spot_id"), name,
            (long long)ji64(report, "nodes_before"), (long long)ji64(report, "nodes"));
        json_object_put(meta);
        json_object_put(report);
    }
    a->used = mark;
    json_write(fmt(a, "%s/pieces.json", work), work_doc);
    a->used = then;
}

typedef struct {
    Bricks *store;
    u8 *complete;
} SpotCT;

static void spot_ct_deliver(void *context, Ask *ask, const u8 *data, size_t bytes) {
    SpotCT *ct = context;
    need(!data || bytes == (size_t)ct->store->size, "incomplete CT brick %s: %zu of %lld bytes",
        ask->path, bytes, (long long)ct->store->size);
    bricks_put(ct->store, ask->index, data);
    ct->complete[ask->index] = 1;
}

static void spot_ct_store(Arena *a, const SpotSource *sheet, const char *folder, const char *spot,
    Readable *surfaces, size_t count, Bricks *store) {
    int lock = lock_file(fmt(a, "%s/.lock", folder), 1, 1);
    need(lock >= 0, "cannot lock CT for spot %s", spot);
    char *source;
    char *url = zarr_level(a, sheet->scroll, sheet->scan, 0, &source);
    char *metadata = fmt(a, "%s/meta.json", folder);
    if (!exists(metadata)) {
        json_object *array = http_json(fmt(a, "%s/.zarray", url));
        i64 shape[3], chunks[3];
        spot_zarr_shape(array, shape, chunks);
        json_object_put(array);
        bricks_create(a, folder, shape, chunks, source);
    }
    bricks_open(a, store, folder);
    size_t after = a->used;
    const char *complete_path = fmt(a, "%s/complete.npy", folder);
    u8 *complete;
    if (exists(complete_path)) {
        Array cached = npy_read(a, complete_path);
        need(cached.ndim == 3 && !memcmp(cached.shape, store->grid, sizeof(store->grid)) &&
                !strcmp(cached.descr, "|b1") && cached.size == (size_t)store->count,
            "CT completeness table does not match the spot's brick grid");
        complete = cached.data;
    } else {
        complete = pushz(a, store->count);
    }
    i32 *touched = push(a, (size_t)store->count * 2 * sizeof(*touched));
    for (i64 i = 0; i < store->count; i++) {
        touched[2 * i] = store->brick[0];
        touched[2 * i + 1] = -1;
    }
    size_t spare = a->used;
    for (size_t i = 0; i < count; i++) {
        mark_ranges_layers(touched, store, &surfaces[i], a, SPOT_RENDER_LAYERS);
        a->used = spare;
    }
    size_t needed = 0;
    for (i64 i = 0; i < store->count; i++) {
        needed += touched[2 * i + 1] >= 0 && (store->offsets[i] == UNKNOWN || !complete[i]);
    }
    Ask *asks = push(a, needed * sizeof(*asks));
    size_t at = 0;
    for (i64 i = 0; i < store->count; i++) {
        if (touched[2 * i + 1] >= 0 && (store->offsets[i] == UNKNOWN || !complete[i])) {
            Ask *ask = &asks[at++];
            *ask = (Ask){.index = i, .first = -1};
            spot_chunk_key(store->grid, i, ask->path, '/');
        }
    }
    SpotCT stage = {store, complete};
    i64 failed = needed ? fetch_ranges(url, asks, needed, spot_ct_deliver, &stage) : 0;
    bricks_save(store);
    npy_write(complete_path,
        (Array){.descr = "|b1",
            .ndim = 3,
            .shape = {store->grid[0], store->grid[1], store->grid[2]},
            .data = complete,
            .size = store->count});
    need(!failed, "%lld CT bricks failed", (long long)failed);
    bricks_map(store);
    printf("%s %s CT bricks fetched %zu\n", sheet->scroll, spot, needed);
    close(lock);
    a->used = after;
}

void spot_surface_render(
    Arena *a, json_object *spot, const SpotSource *sheet, json_object *work_doc) {
    size_t then = a->used;
    const char *work = spot_surface_work(a, spot);
    int rows;
    f64 *axis = spot_read_axis(a, json_object_object_get(work_doc, "axis"), &rows);
    json_object *records = json_object_object_get(work_doc, "pieces");
    size_t max_jobs = 3 * site_items(records), jobs = 0;
    Readable *readable_spots = push(a, max_jobs * sizeof(*readable_spots));
    const char **outputs = push(a, max_jobs * sizeof(*outputs));
    for (int type = 0; type < 3; type++) {
        const char *surface = spot_surface_names[type];
        for (size_t i = 0; i < site_items(records); i++) {
            json_object *record = json_object_array_get_idx(records, i);
            const char *name = site_string(record, "name");
            need(site_name(name), "invalid spot surface name");
            size_t mark = a->used;
            const char *folder = fmt(a, "%s/surfaces/%s/%s", work, surface, name);
            const char *out = fmt(a, "%s/stacks/%s/%s.npy", work, surface, name);
            if (!exists(folder) || exists(out)) {
                a->used = mark;
                continue;
            }
            Surface mesh = load_surface(a, folder);
            readable_spots[jobs] = readable_cells(a, mesh, axis, rows, !type);
            outputs[jobs++] = out;
        }
    }
    if (jobs) {
        Bricks store;
        spot_ct_store(a, sheet, fmt(a, "%s/ct", work), site_string(spot, "spot_id"), readable_spots,
            jobs, &store);
        size_t mark = a->used;
        for (size_t i = 0; i < jobs; i++) {
            if (exists(outputs[i])) {
                continue;
            }
            Stack stack = flatten(a, &readable_spots[i], -1, &store, SPOT_RENDER_LAYERS);
            npy_write(outputs[i],
                (Array){.descr = "|u1",
                    .ndim = 3,
                    .shape = {stack.layers, stack.height, stack.width},
                    .data = stack.cube,
                    .size = (size_t)stack.layers * stack.height * stack.width});
            printf(
                "%s rendered (%d, %d, %d)\n", outputs[i], stack.layers, stack.height, stack.width);
            a->used = mark;
        }
        if (store.data) {
            munmap((void *)store.data, store.mapped);
        }
        pthread_mutex_destroy(&store.lock);
    }
    a->used = then;
}
