typedef struct {
    const char *key, *scroll, *scan, *route, *page, *pieces;
} SpotSource;

typedef struct {
    SpotSource *rows;
    size_t count, capacity;
} SpotSources;

static int spot_local_key(const char *key) {
    for (const char *p = key; p && *p;) {
        size_t part = strcspn(p, "/");
        if (!part ||
            strspn(p, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-") !=
                part ||
            (part == 1 && p[0] == '.') || (part == 2 && !strncmp(p, "..", 2))) {
            return 0;
        }
        if (!p[part]) {
            return 1;
        }
        p += part + 1;
    }
    return 0;
}

static void spot_source_add(SpotSources *sources, SpotSource source) {
    if (sources->count == sources->capacity) {
        size_t capacity = sources->capacity ? sources->capacity * 2 : 64;
        SpotSource *rows = realloc(sources->rows, capacity * sizeof(*rows));
        need(rows != NULL, "could not allocate spot sources");
        sources->rows = rows;
        sources->capacity = capacity;
    }
    sources->rows[sources->count++] = source;
}

static SpotSources spot_sources(Arena *a, const char *marks) {
    SpotSources result = {0};
    const char *registry = fmt(a, "%s/sheets.json", marks);
    json_object *listed = json_object_from_file(registry);
    need(!exists(registry) || json_object_is_type(listed, json_type_array), "cannot read %s",
        registry);
    for (size_t i = 0; i < site_items(listed); i++) {
        json_object *row = json_object_array_get_idx(listed, i);
        const char *key = site_string(row, "key"), *scroll = site_string(row, "scroll");
        const char *scan = site_string(row, "scan"), *route = site_string(row, "route");
        const char *page = site_string(row, "page"), *pieces = site_string(row, "pieces");
        need(spot_local_key(key) && scroll && scan && route && page && pieces,
            "invalid spot sheet entry in %s", registry);
        spot_source_add(&result,
            (SpotSource){fmt(a, "%s", key), fmt(a, "%s", scroll), fmt(a, "%s", scan),
                fmt(a, "%s", route), fmt(a, "%s", page),
                *pieces == '/' ? fmt(a, "%s", pieces) : fmt(a, "%s/%s", OUTPUTS, pieces)});
    }
    json_object_put(listed);
    size_t registered = result.count;
    const char *root = fmt(a, "%s/unwrapping/level", OUTPUTS);
    int scroll_count;
    char **scrolls = list_dir(a, root, &scroll_count);
    for (int i = 0; i < scroll_count; i++) {
        int scan_count;
        char **scans = list_dir(a, fmt(a, "%s/%s", root, scrolls[i]), &scan_count);
        for (int j = 0; j < scan_count; j++) {
            size_t mark = a->used;
            const char *scroll = scrolls[i], *scan = scans[j];
            const char *base = fmt(a, "%s/%s/%s", root, scroll, scan);
            json_object *receipt = json_object_from_file(fmt(a, "%s/sheet/review.json", base));
            const char *href = site_string(receipt, "page");
            const char *key = fmt(a, "level_route/%s/%s", scroll, scan);
            int known = 0;
            for (size_t k = 0; k < registered; k++) {
                known |= !strcmp(result.rows[k].key, key);
            }
            const char *page = href && !strncmp(href, "/sheets/", 8) ? href + 8 : NULL;
            size_t chars = page ? strcspn(page, "/") : 0;
            if (!known && chars && !strcmp(page + chars, "/index.html")) {
                const char *name = fmt(a, "%.*s", (int)chars, page);
                if (site_name(name) &&
                    exists(fmt(a, "%s/website/sheets/%s/index.html", OUTPUTS, name))) {
                    spot_source_add(&result,
                        (SpotSource){key, scroll, scan, "level", name, fmt(a, "%s/pieces", base)});
                    mark = a->used;
                }
            }
            json_object_put(receipt);
            a->used = mark;
        }
    }
    const char *routes = fmt(a, "%s/level_route", marks);
    int route_count;
    char **route_scrolls = list_dir(a, routes, &route_count);
    for (int i = 0; i < route_count; i++) {
        int scan_count;
        char **scans = list_dir(a, fmt(a, "%s/%s", routes, route_scrolls[i]), &scan_count);
        for (int j = 0; j < scan_count; j++) {
            const char *scroll = route_scrolls[i], *scan = scans[j];
            int page_count;
            char **pages = list_dir(a, fmt(a, "%s/%s/%s/pages", routes, scroll, scan), &page_count);
            for (int k = 0; k < page_count; k++) {
                int own = 0;
                char *sheet = site_page_sheet(a, pages[k], scroll, scan, &own);
                size_t length = sheet ? strlen(sheet) : 0;
                const char *pieces = length > 6 && !strcmp(sheet + length - 6, "/sheet")
                    ? fmt(a, "%.*spieces", (int)(length - 5), sheet)
                    : fmt(a, "%s/website/sheets/%s/pieces", OUTPUTS, pages[k]);
                spot_source_add(&result,
                    (SpotSource){fmt(a, "level_route/%s/%s/pages/%s", scroll, scan, pages[k]),
                        scroll, scan, "level", pages[k], pieces});
            }
        }
    }
    return result;
}

static void spot_check_logs(
    Arena *a, const SpotSources *sources, const char *root, const char *key, json_object *unknown) {
    size_t mark = a->used;
    const char *folder = *key ? fmt(a, "%s/%s", root, key) : root;
    char *path = fmt(a, "%s/marks.jsonl", folder);
    if (site_size(path) >= 0) {
        int known = 0;
        for (size_t i = 0; i < sources->count; i++) {
            known |= !strcmp(sources->rows[i].key, key);
        }
        if (!known) {
            json_object_array_add(unknown, json_object_new_string(key));
        }
    }
    int count;
    char **names = list_dir(a, folder, &count);
    size_t saved = a->used;
    for (int i = 0; i < count; i++) {
        struct stat info;
        if (!lstat(fmt(a, "%s/%s", folder, names[i]), &info) && S_ISDIR(info.st_mode)) {
            const char *child = *key ? fmt(a, "%s/%s", key, names[i]) : names[i];
            spot_check_logs(a, sources, root, child, unknown);
        }
        a->used = saved;
    }
    a->used = mark;
}

typedef struct {
    double x, y;
} SpotPoint;

typedef struct {
    SpotPoint *points;
    size_t count;
    double left, right, top, bottom, area, x, y;
    int clicked;
} SpotOutline;

static SpotOutline spot_outline_at(Arena *a, json_object *points, int clicks) {
    size_t n = site_items(points);
    need(n, "spot mark has no points");
    SpotOutline p = {.points = push(a, (n < 4 ? 4 : n) * sizeof(SpotPoint)),
        .count = n,
        .left = INFINITY,
        .right = -INFINITY,
        .top = INFINITY,
        .bottom = -INFINITY};
    for (size_t i = 0; i < n; i++) {
        json_object *point = json_object_array_get_idx(points, i);
        need(site_items(point) == 2, "spot mark has a malformed point");
        double x = json_object_get_double(json_object_array_get_idx(point, 0));
        double y = json_object_get_double(json_object_array_get_idx(point, 1));
        need(isfinite(x) && isfinite(y), "spot mark has a non-finite point");
        p.points[i] = (SpotPoint){x, y};
        p.left = fmin(p.left, x);
        p.right = fmax(p.right, x);
        p.top = fmin(p.top, y);
        p.bottom = fmax(p.bottom, y);
    }
    if (clicks && fmax(p.right - p.left, p.bottom - p.top) < 2) {
        double x = (p.left + p.right) / 2, y = (p.top + p.bottom) / 2;
        p.left = x - 5;
        p.right = x + 5;
        p.top = y - 5;
        p.bottom = y + 5;
        p.points[0] = (SpotPoint){p.left, p.top};
        p.points[1] = (SpotPoint){p.right, p.top};
        p.points[2] = (SpotPoint){p.right, p.bottom};
        p.points[3] = (SpotPoint){p.left, p.bottom};
        p.count = 4;
        p.clicked = 1;
    }
    double past = 0, next = 0;
    for (size_t i = 0; i < p.count; i++) {
        SpotPoint x = p.points[i], y = p.points[(i + 1) % p.count];
        p.x += x.x;
        p.y += x.y;
        past += x.x * y.y;
        next += x.y * y.x;
    }
    p.x /= p.count;
    p.y /= p.count;
    p.area = fabs(past - next) / 2;
    return p;
}

static SpotOutline spot_outline(Arena *a, json_object *mark) {
    return spot_outline_at(a, json_object_object_get(mark, "points"), 1);
}

static int spot_inside(const SpotOutline *p, double x, double y) {
    int inside = 0;
    for (size_t i = 0, j = p->count - 1; i < p->count; j = i++) {
        SpotPoint a = p->points[j], b = p->points[i];
        if ((a.y > y) != (b.y > y) && x < a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y)) {
            inside ^= 1;
        }
    }
    return inside;
}

typedef struct {
    f32 xyz[3];
} SpotVoxel;

typedef struct {
    json_object *piece, *meta;
    float *xyz[3];
    double x, y, area, step;
    int width, height, id, flips[2], overlap[4];
    size_t order, cells, first, start;
} SpotMesh;

typedef struct {
    SpotVoxel point;
    size_t mesh;
} SpotClaim;

typedef struct {
    const char *error;
    json_object *pieces, *windings;
    SpotVoxel *sample;
    size_t n, samples, expected;
    double step, voxel, centre[3], low[3], high[3];
    int matches;
} SpotLocated;

static int spot_mesh_area(const void *left, const void *right) {
    const SpotMesh *a = left, *b = right;
    if (a->area != b->area) {
        return a->area > b->area ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static int spot_mesh_cells(const void *left, const void *right) {
    const SpotMesh *a = *(SpotMesh *const *)left, *b = *(SpotMesh *const *)right;
    if (a->cells != b->cells) {
        return a->cells > b->cells ? -1 : 1;
    }
    return (a->first > b->first) - (a->first < b->first);
}

static const char *spot_mesh_read(Arena *a, SpotMesh *mesh, const char *base) {
    const char *name = site_string(mesh->piece, "unit");
    if (!site_name(name) || mesh->id < 0) {
        return "invalid placed piece";
    }
    const char *folder = fmt(a, "%s/%s/piece-%04d", base, name, mesh->id);
    for (int axis = 0; axis < 3; axis++) {
        const char *file = fmt(a, "%s/%c.tif", folder, "xyz"[axis]);
        if (!exists(file)) {
            return fmt(a, "%s: %s", file, strerror(ENOENT));
        }
        TIFF *tiff = TIFFOpen(file, "r");
        if (!tiff) {
            return fmt(a, "cannot read %s", file);
        }
        u32 w = 0, h = 0;
        u16 bits = 0, type = 0, channels = 0;
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &h);
        TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
        TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &type);
        TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
        if (w != (unsigned)mesh->width || h != (unsigned)mesh->height) {
            TIFFClose(tiff);
            return fmt(a, "%s: stored grid (%u, %u) differs from saved layout (%d, %d)", folder, h,
                w, mesh->height, mesh->width);
        }
        int valid = bits == 32 && type == SAMPLEFORMAT_IEEEFP && channels == 1 &&
            TIFFScanlineSize64(tiff) == (u64)w * 4;
        f32 *pixels = valid ? push(a, (size_t)w * h * sizeof(*pixels)) : NULL;
        for (u32 row = 0; valid && row < h; row++) {
            valid = TIFFReadScanline(tiff, pixels + (size_t)row * w, row, 0) == 1;
        }
        TIFFClose(tiff);
        if (!valid) {
            return fmt(a, "cannot read float32 geometry in %s", file);
        }
        mesh->xyz[axis] = pixels;
    }
    const char *path = fmt(a, "%s/meta.json", folder);
    mesh->meta = json_object_from_file(path);
    return mesh->meta ? NULL : fmt(a, "cannot read %s", path);
}

static int spot_value_order(const void *left, const void *right) {
    double a = json_object_get_double(*(json_object *const *)left);
    double b = json_object_get_double(*(json_object *const *)right);
    return (a > b) - (a < b);
}

static const char *spot_group_text(json_object *group) {
    if (!group) {
        return "None";
    }
    if (json_object_is_type(group, json_type_boolean)) {
        return json_object_get_boolean(group) ? "True" : "False";
    }
    return json_object_get_string(group);
}

static void spot_mesh_summary(
    Arena *a, SpotMesh *meshes, size_t count, SpotClaim *claims, SpotLocated *found) {
    SpotMesh **held = push(a, count * sizeof(*held));
    size_t used = 0, begin = 0;
    for (size_t i = 0; i < count; i++) {
        if (meshes[i].cells) {
            held[used++] = &meshes[i];
        }
    }
    qsort(held, used, sizeof(*held), spot_mesh_cells);
    json_object **windings = push(a, used * sizeof(*windings));
    size_t winding_count = 0;
    for (size_t i = 0; i < used; i++) {
        SpotMesh *mesh = held[i];
        mesh->start = begin;
        begin += mesh->cells;
        json_object *row = json_object_new_object(), *piece = mesh->piece, *meta = mesh->meta;
        json_object *winding = json_object_object_get(meta, "winding");
        json_object *level = json_object_object_get(piece, "level");
        json_object *turn = json_object_object_get(piece, "turn");
        json_object *area = json_object_object_get(meta, "area_mm2");
        double width = site_value(piece, "width_mm");
        json_object *written = json_object_object_get(meta, "width_mm");
        int matches = !written || fabs(json_object_get_double(written) - width) < 1e-3;
        json_object_object_add(row, "unit", json_object_get(json_object_object_get(piece, "unit")));
        json_object_object_add(row, "piece", json_object_new_int(mesh->id));
        json_object_object_add(row, "cells", json_object_new_int64(mesh->cells));
        json_object_object_add(row, "share",
            jnum(decimal_round((double)mesh->cells / (found->expected ? found->expected : 1), 4)));
        json_object_object_add(row, "winding", json_object_get(winding));
        json_object_object_add(row, "level", json_object_get(level));
        json_object_object_add(row, "group",
            json_object_new_string(spot_group_text(json_object_object_get(piece, "group"))));
        json_object_object_add(row, "turn", turn ? json_object_get(turn) : json_object_new_int(1));
        json_object_object_add(row, "area_mm2",
            jnum(decimal_round(area ? json_object_get_double(area) : mesh->area, 2)));
        json_object_object_add(row, "matches_layout", json_object_new_boolean(matches));
        json_object_array_add(found->pieces, row);
        found->matches &= matches;
        if (winding) {
            windings[winding_count++] = winding;
        }
    }
    qsort(windings, winding_count, sizeof(*windings), spot_value_order);
    for (size_t i = 0; i < winding_count; i++) {
        if (!i || !json_object_equal(windings[i], windings[i - 1])) {
            json_object_array_add(found->windings, json_object_get(windings[i]));
        }
    }
    if (!found->n) {
        return;
    }
    SpotVoxel *points = push(a, found->n * sizeof(*points));
    for (size_t i = 0; i < found->n; i++) {
        points[meshes[claims[i].mesh].start++] = claims[i].point;
    }
    size_t stride = found->n > 4000 ? found->n / 4000 : 1;
    found->samples = 1 + (found->n - 1) / stride;
    found->sample = malloc(found->samples * sizeof(*found->sample));
    need(found->sample != NULL, "cannot allocate spot samples");
    f32 sum[3] = {0};
    for (size_t i = 0; i < found->n; i++) {
        SpotVoxel p = points[i];
        for (int d = 0; d < 3; d++) {
            sum[d] += p.xyz[d];
            if (!i || p.xyz[d] < found->low[d]) {
                found->low[d] = p.xyz[d];
            }
            if (!i || p.xyz[d] > found->high[d]) {
                found->high[d] = p.xyz[d];
            }
        }
        if (i % stride == 0) {
            found->sample[i / stride] = p;
        }
    }
    for (int d = 0; d < 3; d++) {
        found->centre[d] = sum[d] / (f32)found->n;
    }
}

static SpotLocated spot_locate(Arena *a, const char *marks, const SpotSource *source,
    const char *digest_key, const SpotOutline *outline) {
    SpotLocated found = {
        .pieces = json_object_new_array(), .windings = json_object_new_array(), .matches = 1};
    need(site_name(digest_key), "spot mark has an invalid layout key");
    char *layout = fmt(a, "%s/%s/layouts/%s.json", marks, source->key, digest_key);
    json_object *doc = json_object_from_file(layout);
    json_object *rows = json_object_object_get(doc, "pieces");
    size_t total = site_items(rows);
    if (!total) {
        found.error = !doc ? fmt(a, "%s: %s", layout, strerror(ENOENT))
                           : fmt(a, "empty saved layout %s", digest_key);
        json_object_put(doc);
        return found;
    }
    SpotMesh *mesh = pushz(a, total * sizeof(*mesh));
    json_object *first = json_object_array_get_idx(rows, 0);
    found.voxel = decimal_round(site_value(first, "um") * 1000, 3);
    found.step = site_value(first, "um") * site_value(first, "stride");
    size_t count = 0;
    for (size_t i = 0; i < total; i++) {
        json_object *piece = json_object_array_get_idx(rows, i);
        double x = site_value(piece, "x_mm"), y = site_value(piece, "y_mm");
        double w = site_value(piece, "width_mm"), h = site_value(piece, "height_mm");
        if (!(x < outline->right + 1 && x + w > outline->left - 1 && y < outline->bottom + 1 &&
                y + h > outline->top - 1)) {
            continue;
        }
        double step = site_value(piece, "um") * site_value(piece, "stride");
        if (!count) {
            found.step = step;
        }
        json_object *shape = json_object_object_get(piece, "shape");
        if (!isfinite(step) || !(step > 0) || site_items(shape) != 2) {
            found.error = "invalid placed piece shape or spacing";
            break;
        }
        SpotMesh *item = &mesh[count++];
        *item = (SpotMesh){.piece = piece,
            .x = x,
            .y = y,
            .step = step,
            .width = json_object_get_int(json_object_array_get_idx(shape, 1)),
            .height = json_object_get_int(json_object_array_get_idx(shape, 0)),
            .id = site_value(piece, "piece"),
            .area = site_value(piece, "area"),
            .order = i,
            .flips = {json_object_get_boolean(json_object_object_get(piece, "flip_rows")),
                json_object_get_boolean(json_object_object_get(piece, "flip_cols"))}};
        json_object *overlap = json_object_object_get(piece, "overlap");
        for (size_t j = 0; j < site_items(overlap) && j < 4; j++) {
            item->overlap[j] = json_object_get_int(json_object_array_get_idx(overlap, j));
        }
    }
    qsort(mesh, count, sizeof(*mesh), spot_mesh_area);
    for (size_t i = 0; !found.error && i < count; i++) {
        found.error = spot_mesh_read(a, &mesh[i], source->pieces);
    }
    double step = found.step, left = floor(outline->left / step), top = floor(outline->top / step);
    double cols = ceil(outline->right / step) - left + 1;
    double rows_count = ceil(outline->bottom / step) - top + 1;
    if (!found.error &&
        (!(step > 0) || !(cols >= 1 && rows_count >= 1) || cols * rows_count > 100000000 ||
            fabs(left) >= 0x1p52 || fabs(top) >= 0x1p52)) {
        found.error = "spot outline does not fit the saved node grid";
    }
    if (!found.error) {
        size_t w = cols, h = rows_count, cells = w * h;
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                found.expected += spot_inside(outline, (left + x) * step, (top + y) * step);
            }
        }
        u8 *present = pushz(a, (cells + 7) / 8);
        SpotClaim *claims = push(a, cells * sizeof(*claims));
        for (int margins = 0; margins < 2; margins++) {
            for (size_t i = 0; i < count; i++) {
                SpotMesh *piece = &mesh[i];
                int trim[4];
                for (int k = 0; k < 4; k++)
                    trim[k] = margins || piece->overlap[k] < 0 ? 0 : piece->overlap[k];
                int row_end = piece->height - trim[1], col_end = piece->width - trim[3];
                int y0 = piece->flips[0] * (piece->height - 1), dy = 1 - 2 * piece->flips[0];
                int x0 = piece->flips[1] * (piece->width - 1), dx = 1 - 2 * piece->flips[1];
                for (int r = trim[0]; r < row_end; r++) {
                    double y = piece->y + piece->step * (y0 + dy * r);
                    if (y < outline->top || y > outline->bottom) {
                        continue;
                    }
                    for (int c = trim[2]; c < col_end; c++) {
                        size_t index = (size_t)r * piece->width + c;
                        if (!(piece->xyz[0][index] > 0)) {
                            continue;
                        }
                        double x = piece->x + piece->step * (x0 + dx * c);
                        if (x < outline->left || x > outline->right ||
                            !spot_inside(outline, x, y)) {
                            continue;
                        }
                        size_t key = (size_t)(nearbyint(y / step) - top) * w +
                            (size_t)(nearbyint(x / step) - left);
                        if (key < cells && !(present[key / 8] & (1 << (key % 8)))) {
                            present[key / 8] |= 1 << (key % 8);
                            if (!piece->cells++) {
                                piece->first = found.n;
                            }
                            claims[found.n++] = (SpotClaim){.mesh = i,
                                .point = {{piece->xyz[0][index], piece->xyz[1][index],
                                    piece->xyz[2][index]}}};
                        }
                    }
                }
            }
        }
        spot_mesh_summary(a, mesh, count, claims, &found);
    }
    for (size_t i = 0; i < count; i++) {
        json_object_put(mesh[i].meta);
    }
    json_object_put(doc);
    return found;
}

static json_object *spot_number(double value, int digits) {
    return isfinite(value) ? jnum(decimal_round(value, digits)) : NULL;
}

static json_object *spot_vector(const double values[3], int available) {
    json_object *row = json_object_new_array_ext(3);
    for (int i = 0; i < 3; i++) {
        json_object_array_add(
            row, available ? spot_number(values[i], 1) : json_object_new_double(NAN));
    }
    return row;
}

static json_object *spot_catalogue_row(Arena *a, const SpotSource *source, const SpotOutline *p,
    json_object *mark, SpotLocated *where) {
    const char *id = site_string(mark, "id"), *at = site_string(mark, "time");
    const char *layer = site_string(mark, "layer"), *key = site_string(mark, "layout");
    const char *since = env("HERCULEX_SPOTS_SINCE", "");
    need(id && at && key, "incomplete spot mark");
    int location = !where->error;
    int target = strncmp(at, since, 10) >= 0;
    json_object *outline = json_object_new_array_ext(p->count);
    for (size_t i = 0; i < p->count; i++) {
        json_object *pair = json_object_new_array_ext(2);
        json_object_array_add(pair, jnum(nearbyint(p->points[i].x * 1000) / 1000));
        json_object_array_add(pair, jnum(nearbyint(p->points[i].y * 1000) / 1000));
        json_object_array_add(outline, pair);
    }
    json_object *row = json_object_new_object();
    json_object_object_add(row, "spot_id", json_object_new_string(fmt(a, "%.8s", id)));
    json_object_object_add(row, "mark_id", json_object_get(json_object_object_get(mark, "id")));
    json_object_object_add(row, "sheet_key", json_object_new_string(source->key));
    json_object_object_add(row, "page", json_object_new_string(source->page));
    json_object_object_add(row, "scroll", json_object_new_string(source->scroll));
    json_object_object_add(row, "scan", json_object_new_string(source->scan));
    json_object_object_add(row, "route", json_object_new_string(source->route));
    json_object_object_add(
        row, "labeller", json_object_new_string(target ? "target-day" : "earlier"));
    json_object_object_add(row, "target", json_object_new_boolean(target && location));
    json_object_object_add(row, "location_available", json_object_new_boolean(location));
    json_object_object_add(
        row, "location_error", json_object_new_string(location ? "" : where->error));
    json_object_object_add(row, "marked_at", json_object_new_string(at));
    json_object_object_add(row, "layer", json_object_new_string(layer ? layer : ""));
    json_object_object_add(row, "layout", json_object_new_string(key));
    json_object_object_add(row, "clicked", json_object_new_boolean(p->clicked));
    json_object_object_add(row, "outline_mm", outline);
    json_object_object_add(row, "x_mm", spot_number(p->x, 2));
    json_object_object_add(row, "y_mm", spot_number(p->y, 2));
    json_object_object_add(row, "width_mm", spot_number(p->right - p->left, 2));
    json_object_object_add(row, "height_mm", spot_number(p->bottom - p->top, 2));
    json_object_object_add(row, "area_cm2", spot_number(p->area / 100, 3));
    json_object_object_add(row, "cells_inside", json_object_new_int64(where->expected));
    json_object_object_add(row, "cells_covered", json_object_new_int64(where->n));
    json_object_object_add(row, "covered_share",
        spot_number((double)where->n / (where->expected ? where->expected : 1), 4));
    json_object_object_add(row, "pieces", where->pieces);
    json_object_object_add(row, "n_pieces", json_object_new_int64(site_items(where->pieces)));
    json_object *main = json_object_array_get_idx(where->pieces, 0);
    json_object_object_add(row, "main_piece",
        json_object_new_string(main
                ? fmt(a, "%s/piece-%04d", site_string(main, "unit"), (int)site_value(main, "piece"))
                : ""));
    json_object_object_add(row, "main_share", jnum(site_value(main, "share")));
    json_object_object_add(row, "windings", where->windings);
    json_object_object_add(
        row, "layout_matches", json_object_new_boolean(location && where->matches));
    json_object_object_add(row, "centre_xyz", spot_vector(where->centre, where->n != 0));
    json_object_object_add(row, "box_low_xyz", spot_vector(where->low, where->n != 0));
    json_object_object_add(row, "box_high_xyz", spot_vector(where->high, where->n != 0));
    json_object_object_add(
        row, "node_mm", location ? spot_number(where->step, 5) : json_object_new_double(NAN));
    json_object_object_add(
        row, "voxel_um", location ? spot_number(where->voxel, 3) : json_object_new_double(NAN));
    return row;
}

typedef struct {
    SpotVoxel *points;
    size_t count;
} SpotTree;

static void spot_tree(SpotVoxel *points, size_t count, int axis) {
    if (count < 2) {
        return;
    }
    size_t at = count / 2;
    i64 lo = 0, hi = count - 1;
    while (lo < hi) {
        i64 i = lo, j = hi;
        f32 split = points[lo + (hi - lo) / 2].xyz[axis];
        do {
            while (points[i].xyz[axis] < split) {
                i++;
            }
            while (points[j].xyz[axis] > split) {
                j--;
            }
            if (i <= j) {
                SpotVoxel swap = points[i];
                points[i++] = points[j];
                points[j--] = swap;
            }
        } while (i <= j);
        if ((i64)at <= j) {
            hi = j;
        } else if ((i64)at >= i) {
            lo = i;
        } else {
            break;
        }
    }
    int next = (axis + 1) % 3;
    spot_tree(points, at, next);
    spot_tree(points + at + 1, count - at - 1, next);
}

static int spot_reached(
    const SpotVoxel *points, size_t count, SpotVoxel point, int axis, double distance) {
    if (!count) {
        return 0;
    }
    size_t at = count / 2;
    double delta[3];
    for (int i = 0; i < 3; i++) {
        delta[i] = (double)points[at].xyz[i] - point.xyz[i];
    }
    if (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] < distance) {
        return 1;
    }
    int next = (axis + 1) % 3;
    double plane = delta[axis];
    const SpotVoxel *sides[] = {points, points + at + 1};
    const size_t sizes[] = {at, count - at - 1};
    int near = !(plane > 0), far = 1 - near;
    return spot_reached(sides[near], sizes[near], point, next, distance) ||
        (plane * plane < distance && spot_reached(sides[far], sizes[far], point, next, distance));
}

static int spot_related(SpotTree a, SpotTree b, double reach) {
    size_t hit = 0, limit = (a.count + 3) / 4;
    double distance = reach * reach;
    for (size_t i = 0; i < a.count && hit + a.count - i >= limit; i++) {
        if (spot_reached(b.points, b.count, a.points[i], 0, distance) && ++hit >= limit) {
            return 1;
        }
    }
    return 0;
}

static size_t spot_place_root(size_t *parent, size_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

static void spot_places(Arena *a, json_object *rows, SpotTree *points) {
    size_t n = site_items(rows);
    size_t *parent = push(a, n * sizeof(*parent)), *circles = pushz(a, n * sizeof(*circles));
    const char **names = pushz(a, n * sizeof(*names));
    for (size_t i = 0; i < n; i++) {
        parent[i] = i;
        spot_tree(points[i].points, points[i].count, 0);
    }
    for (size_t i = 0; i < n; i++) {
        if (!points[i].count) {
            continue;
        }
        json_object *left = json_object_array_get_idx(rows, i);
        const char *scroll = site_string(left, "scroll"), *scan = site_string(left, "scan");
        json_object *xyz = json_object_object_get(left, "centre_xyz");
        double reach = 300 / site_value(left, "voxel_um");
        for (size_t j = i + 1; j < n; j++) {
            json_object *right = json_object_array_get_idx(rows, j);
            if (!points[j].count || strcmp(scroll, site_string(right, "scroll")) != 0 ||
                strcmp(scan, site_string(right, "scan")) != 0) {
                continue;
            }
            size_t x = spot_place_root(parent, i), y = spot_place_root(parent, j);
            if (x == y) {
                continue;
            }
            json_object *centre = json_object_object_get(right, "centre_xyz");
            double distance = 0;
            for (size_t d = 0; d < 3; d++) {
                double delta = json_object_get_double(json_object_array_get_idx(xyz, d)) -
                    json_object_get_double(json_object_array_get_idx(centre, d));
                distance += delta * delta;
            }
            if (distance <= 80 * 80 * reach * reach &&
                (spot_related(points[i], points[j], reach) ||
                    spot_related(points[j], points[i], reach))) {
                parent[x] = y;
            }
        }
    }
    json_object *serials = json_object_new_object();
    for (size_t i = 0; i < n; i++) {
        size_t root = spot_place_root(parent, i);
        json_object *row = json_object_array_get_idx(rows, i);
        if (!names[root]) {
            const char *scroll = site_string(row, "scroll");
            int serial = json_object_get_int(json_object_object_get(serials, scroll)) + 1;
            json_object_object_add(serials, scroll, json_object_new_int(serial));
            names[root] = fmt(a, "%s-place-%03d", scroll, serial);
        }
        circles[root]++;
        json_object_object_add(row, "place", json_object_new_string(names[root]));
    }
    for (size_t i = 0; i < n; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        json_object_object_add(
            row, "place_circles", json_object_new_int64(circles[spot_place_root(parent, i)]));
    }
    json_object_put(serials);
}

static int spot_row_order(const void *left, const void *right) {
    json_object *a = *(json_object *const *)left, *b = *(json_object *const *)right;
    int order = strcmp(site_string(a, "scroll"), site_string(b, "scroll"));
    return order ? order : strcmp(site_string(a, "page"), site_string(b, "page"));
}

static int spot_id_order(const void *left, const void *right) {
    json_object *a = *(json_object *const *)left, *b = *(json_object *const *)right;
    return strcmp(json_object_get_string(a), json_object_get_string(b));
}

static void spot_catalogue_report(Arena *a, json_object *rows, json_object *gone) {
    size_t count = site_items(rows), wanted = 0;
    json_object **targets = push(a, count * sizeof(*targets));
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        if (json_object_get_boolean(json_object_object_get(row, "target"))) {
            targets[wanted++] = row;
        }
    }
    printf("%zu standing circles, %zu of them targets; %zu no longer standing: %s\n", count, wanted,
        site_items(gone), json_object_to_json_string_ext(gone, JSON_C_TO_STRING_PLAIN));
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *error = site_string(row, "location_error");
        if (error && *error) {
            printf("%s location unavailable: %s\n", site_string(row, "spot_id"), error);
        }
    }
    qsort(targets, wanted, sizeof(*targets), spot_row_order);
    json_object *places = json_object_new_object();
    for (size_t left = 0, right; left < wanted; left = right) {
        const char *scroll = site_string(targets[left], "scroll");
        const char *page = site_string(targets[left], "page");
        json_object *in_sheet = json_object_new_object();
        size_t matched = 0;
        double area = 0, covered = 0;
        for (right = left; right < wanted && !spot_row_order(&targets[left], &targets[right]);
            right++) {
            json_object *row = targets[right];
            const char *place = site_string(row, "place");
            json_object_object_add(in_sheet, place, NULL);
            json_object_object_add(places, place, NULL);
            matched += json_object_get_boolean(json_object_object_get(row, "layout_matches"));
            area += site_value(row, "area_cm2");
            covered += site_value(row, "covered_share");
        }
        printf("  %-11s %-42s circles %3zu  places %3d  area %6.1f cm²  covered %.2f  "
               "layout ok %zu/%zu\n",
            scroll, page, right - left, json_object_object_length(in_sheet), area,
            covered / (right - left), matched, right - left);
        json_object_put(in_sheet);
    }
    printf("places with target circles: %d\ntable %s\n", json_object_object_length(places),
        spot_table_path(a, "spots"));
    json_object_put(places);
}

int hx_spot_catalogue(int argc, char **argv) {
    need(argc == 0, "usage: ./run.sh spots catalogue");
    Arena a = arena(16 * GB);
    const char *marks = fmt(&a, "%s/marks", LIBRARY);
    SpotSources sources = spot_sources(&a, marks);
    json_object *unknown = json_object_new_array();
    spot_check_logs(&a, &sources, marks, "", unknown);
    need(!site_items(unknown), "marks logs with no sheet entry in %s/marks/sheets.json: %s",
        LIBRARY, json_object_to_json_string_ext(unknown, JSON_C_TO_STRING_PLAIN));
    json_object_put(unknown);
    json_object *rows = json_object_new_array(), *keys = json_object_new_object();
    SpotTree *samples = NULL;
    size_t count = 0, capacity = 0;
    size_t mark = a.used;
    for (size_t i = 0; i < sources.count; i++) {
        const SpotSource *source = sources.rows + i;
        json_object *ignored;
        if (json_object_object_get_ex(keys, source->key, &ignored)) {
            continue;
        }
        json_object_object_add(keys, source->key, NULL);
        const char *log = fmt(&a, "%s/%s/marks.jsonl", marks, source->key);
        json_object *active = site_standing(log);
        need(active != NULL, "cannot read %s", log);
        a.used = mark;
        for (size_t j = 0; j < site_items(active); j++) {
            json_object *entry = json_object_array_get_idx(active, j);
            const char *kind = site_string(entry, "kind");
            if (!kind || strcmp(kind, "spot") != 0) {
                continue;
            }
            SpotOutline outline = spot_outline(&a, entry);
            const char *digest_key = site_string(entry, "layout");
            json_object *best = NULL;
            int available = -1;
            size_t covered = 0;
            SpotTree tree = {0};
            size_t hold = a.used;
            for (size_t k = i; k < sources.count; k++) {
                const SpotSource *candidate = sources.rows + k;
                if (strcmp(candidate->key, source->key) != 0) {
                    continue;
                }
                SpotLocated located = spot_locate(&a, marks, candidate, digest_key, &outline);
                int usable = !located.error;
                if (usable > available || (usable == available && located.n > covered)) {
                    available = usable;
                    covered = located.n;
                    json_object_put(best);
                    free(tree.points);
                    best = spot_catalogue_row(&a, candidate, &outline, entry, &located);
                    tree = (SpotTree){located.sample, located.samples};
                } else {
                    json_object_put(located.pieces);
                    json_object_put(located.windings);
                    free(located.sample);
                }
                a.used = hold;
            }
            if (count == capacity) {
                capacity = capacity ? capacity * 2 : 64;
                SpotTree *next = realloc(samples, capacity * sizeof(*next));
                need(next != NULL, "cannot allocate spot samples");
                samples = next;
            }
            samples[count++] = tree;
            json_object_array_add(rows, best);
            a.used = mark;
        }
        json_object_put(active);
    }
    json_object_put(keys);
    free(sources.rows);
    spot_places(&a, rows, samples);
    for (size_t i = 0; i < count; i++) {
        free(samples[i].points);
    }
    free(samples);
    json_object *old = spot_rows(&a, "spots"), *gone = json_object_new_array();
    json_object *previous = json_object_new_object();
    for (size_t i = 0; i < site_items(old); i++) {
        json_object *row = json_object_array_get_idx(old, i);
        const char *id = site_string(row, "spot_id");
        if (id) {
            json_object_object_add(previous, id, json_object_get(row));
        }
    }
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        const char *id = site_string(row, "spot_id");
        json_object *last = json_object_object_get(previous, id);
        const char *status = site_string(last, "status");
        json_object_object_add(
            row, "status", json_object_new_string(status ? status : "catalogued"));
        json_object_object_del(previous, id);
    }
    json_object_object_foreach(previous, id, value) {
        (void)value;
        json_object_array_add(gone, json_object_new_string(id));
    }
    json_object_array_sort(gone, spot_id_order);
    spot_write(&a, "spots", rows);
    spot_forget(&a, gone);
    spot_catalogue_report(&a, rows, gone);
    json_object_put(previous);
    json_object_put(old);
    json_object_put(gone);
    json_object_put(rows);
    munmap(a.base, a.size);
    return 0;
}
