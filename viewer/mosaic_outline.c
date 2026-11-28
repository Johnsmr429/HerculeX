typedef struct {
    double area;
    size_t slot;
} OutlineGroup;

static int outline_group_order(const void *x, const void *y) {
    const OutlineGroup *a = x, *b = y;
    if (a->area != b->area) {
        return a->area > b->area ? -1 : 1;
    }
    return (a->slot > b->slot) - (a->slot < b->slot);
}

typedef struct {
    int w, h;
    u8 *mask;
} OutlineMask;

static size_t outline_fill(OutlineMask mask, u32 *queue, size_t first, u8 source, u8 mark) {
    size_t count = 1;
    queue[0] = first;
    mask.mask[first] = mark;
    for (size_t i = 0; i < count; i++) {
        size_t at = queue[i];
        int x = at % mask.w, y = at / mask.w;
        size_t near[] = {at - (x > 0), at + (x + 1 < mask.w), y > 0 ? at - mask.w : at,
            y + 1 < mask.h ? at + mask.w : at};
        for (int k = 0; k < 4; k++) {
            size_t to = near[k];
            if (mask.mask[to] == source) {
                mask.mask[to] = mark;
                queue[count++] = to;
            }
        }
    }
    return count;
}

static size_t outline_close(Arena *a, OutlineMask *grid, double cell) {
    int w = grid->w, h = grid->h;
    size_t count = (size_t)w * h;
    u8 *in = grid->mask, *out = pushz(a, count);
    for (int pass = 0; pass < 6; pass++) {
        int erode = pass >= 3;
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                size_t i = (size_t)y * w + x;
                out[i] = erode ? (in[i] & in[i - 1] & in[i + 1] & in[i - w] & in[i + w])
                               : (in[i] | in[i - 1] | in[i + 1] | in[i - w] | in[i + w]);
            }
        }
        u8 *swap = in;
        in = out;
        out = swap;
    }
    grid->mask = in;
    u32 *queue = push(a, count * sizeof(*queue));
    outline_fill(*grid, queue, 0, 0, 2);
    for (size_t i = 0; i < count; i++) {
        in[i] = in[i] != 2;
    }
    size_t retained = 0;
    for (size_t i = 0; i < count; i++) {
        if (in[i] != 1) {
            continue;
        }
        size_t area = outline_fill(*grid, queue, i, 1, 2);
        if (area * cell * cell >= 100) {
            retained += area;
        } else {
            for (size_t j = 0; j < area; j++) {
                in[queue[j]] = 0;
            }
        }
    }
    return retained;
}

typedef struct {
    u32 from, to;
} OutlineSegment;

static size_t outline_segments(OutlineMask mask, OutlineSegment *segments) {
    const u8 edge[][4] = {{0}, {0, 1}, {2, 0}, {2, 1}, {1, 3}, {0, 3}, {2, 0, 1, 3}, {2, 3}, {3, 2},
        {0, 1, 3, 2}, {3, 0}, {3, 1}, {1, 2}, {0, 2}, {1, 0}, {0}};
    size_t count = 0;
    int w = mask.w, h = mask.h;
    for (int y = 0; y < h - 1; y++) {
        for (int x = 0; x < w - 1; x++) {
            size_t at = (size_t)y * w + x;
            const u8 *p = mask.mask + at;
            int code = !!p[0] | (!!p[1] << 1) | (!!p[w] << 2) | (!!p[w + 1] << 3);
            if (!code || code == 15) {
                continue;
            }
            int lines = code == 6 || code == 9 ? 2 : 1;
            u32 points[] = {2 * at, 2 * at + 1, 2 * at + 3, 2 * (at + w)};
            for (int i = 0; i < lines; i++) {
                if (segments) {
                    segments[count] =
                        (OutlineSegment){points[edge[code][i * 2]], points[edge[code][i * 2 + 1]]};
                }
                count++;
            }
        }
    }
    return count;
}

typedef struct {
    double x, y;
} OutlinePoint;

static void outline_simplify(OutlinePoint *points, size_t count, u8 *keep, u32 *stack) {
    size_t used = 2;
    stack[0] = 0;
    stack[1] = count - 1;
    keep[0] = keep[count - 1] = 1;
    while (used) {
        size_t last = stack[--used], first = stack[--used];
        double x0 = points[first].x, y0 = points[first].y;
        double x1 = points[last].x, y1 = points[last].y;
        double dx = x1 - x0, dy = y1 - y0, angle = -atan2(dy, dx);
        double sn = sin(angle), cs = cos(angle), origin = x0 * sn + y0 * cs;
        double worst = .5;
        size_t pivot = 0;
        for (size_t i = first + 1; i < last; i++) {
            double x = points[i].x, y = points[i].y;
            double a = (x - x0) * dx + (y - y0) * dy;
            double b = (x1 - x) * dx + (y1 - y) * dy;
            double range = a > 0 && b > 0 ? fabs(y * cs + x * sn - origin)
                                          : sqrt(fmin((x - x0) * (x - x0) + (y - y0) * (y - y0),
                                                (x - x1) * (x - x1) + (y - y1) * (y - y1)));
            if (range > worst) {
                worst = range;
                pivot = i;
            }
        }
        if (pivot) {
            keep[pivot] = 1;
            stack[used++] = pivot;
            stack[used++] = last;
            stack[used++] = first;
            stack[used++] = pivot;
        }
    }
}

static json_object *outline_contours(Arena *a, OutlineMask mask, const char *name, double x0,
    double y0, double cell, size_t *point_count) {
    json_object *lines = json_object_new_array();
    size_t count = outline_segments(mask, NULL);
    if (!count) {
        return lines;
    }
    need(count < UINT32_MAX, "mosaic has too many outline segments");
    OutlineSegment *segments = push(a, count * sizeof(*segments));
    outline_segments(mask, segments);
    u32 *outgoing = pushz(a, (size_t)mask.w * mask.h * 2 * sizeof(*outgoing));
    for (size_t i = 0; i < count; i++) {
        outgoing[segments[i].from] = i + 1;
    }
    u8 *visited = pushz(a, count);
    OutlinePoint *vertices = push(a, (count + 1) * sizeof(*vertices));
    u8 *kept = push(a, count + 1);
    u32 *stack = push(a, (count + 1) * 2 * sizeof(*stack));
    for (size_t i = 0; i < count; i++) {
        if (visited[i]) {
            continue;
        }
        size_t last = 0, n = 0;
        size_t next = i;
        do {
            need(next < count && !visited[next], "could not join mosaic contour");
            visited[next] = 1;
            last = last < next ? next : last;
            n++;
            next = outgoing[segments[next].to] - (size_t)1;
        } while (next != i);
        next = outgoing[segments[last].to] - (size_t)1;
        for (size_t j = 0; j < n; j++) {
            u32 node = segments[next].from;
            size_t at = node / 2, down = at / mask.w;
            vertices[j] =
                (OutlinePoint){at % mask.w + (node & 1 ? 0 : .5), down + (node & 1 ? .5 : 0)};
            next = outgoing[segments[next].to] - (size_t)1;
        }
        vertices[n] = vertices[0];
        memset(kept, 0, n + 1);
        outline_simplify(vertices, n + 1, kept, stack);
        json_object *points = json_object_new_array();
        for (size_t j = 0; j <= n; j++) {
            if (kept[j]) {
                json_object *pair = json_object_new_array_ext(2);
                json_object_array_add(
                    pair, jnum(decimal_round(x0 + (vertices[j].x - 5) * cell, 3)));
                json_object_array_add(
                    pair, jnum(decimal_round(y0 + (vertices[j].y - 5) * cell, 3)));
                json_object_array_add(points, pair);
            }
        }
        if (site_items(points) < 4) {
            json_object_put(points);
        } else {
            *point_count += site_items(points);
            json_object *line = json_object_new_object();
            json_object_object_add(line, "name", json_object_new_string(name));
            json_object_object_add(line, "points", points);
            json_object_array_add(lines, line);
        }
    }
    return lines;
}

int hx_mosaic_outline_command(int argc, char **argv) {
    need((argc == 3 || argc == 4) && site_name(argv[0]) && site_name(argv[1]) && site_name(argv[2]),
        "usage: ./run.sh viewer.mosaic_outline PAGE SCROLL SCAN [RANK]");
    char *end = NULL;
    long rank = argc == 4 ? strtol(argv[3], &end, 10) : 1;
    need(rank >= 1 && (!end || !*end), "invalid mosaic rank");
    Arena a = arena(4 * GB);
    char *base = fmt(&a, "%s/unwrapping/level/%s/%s", OUTPUTS, argv[1], argv[2]);
    json_object *doc = json_object_from_file(fmt(&a, "%s/sheet/placed.json", base));
    json_object *pieces = json_object_object_get(doc, "pieces"), *slots = json_object_new_object();
    size_t count = site_items(pieces), groups = 0;
    OutlineGroup *order = pushz(&a, (count + 1) * sizeof(*order));
    int *group = push(&a, count * sizeof(*group));
    for (size_t i = 0; i < count; i++) {
        json_object *piece = json_object_array_get_idx(pieces, i);
        int slot = hx_mosaic_slot(slots, piece);
        need(slot > 0 && (size_t)slot <= count, "invalid mosaic piece");
        group[i] = slot;
        groups = fmax(groups, slot);
        order[slot - 1].area += site_value(piece, "area");
        order[slot - 1].slot = slot;
    }
    json_object_put(slots);
    need((size_t)rank <= groups, "no mosaic at rank %ld", rank);
    qsort(order, groups, sizeof(*order), outline_group_order);
    int slot = order[rank - 1].slot;
    double area = order[rank - 1].area / 100, x0 = INFINITY, y0 = INFINITY, cell = 0;
    for (size_t i = 0; i < count; i++) {
        if (group[i] == slot) {
            json_object *piece = json_object_array_get_idx(pieces, i);
            if (!cell) {
                cell = site_value(piece, "stride") * site_value(piece, "um");
            }
            x0 = fmin(x0, site_value(piece, "x_mm"));
            y0 = fmin(y0, site_value(piece, "y_mm"));
        }
    }
    need(cell > 0 && isfinite(cell), "mosaic has an invalid cell size");
    int w = 0, h = 0;
    for (size_t i = 0; i < count; i++) {
        if (group[i] == slot) {
            json_object *piece = json_object_array_get_idx(pieces, i);
            json_object *shape = json_object_object_get(piece, "shape");
            need(site_items(shape) == 2, "mosaic piece has no shape");
            double x = nearbyint((site_value(piece, "x_mm") - x0) / cell);
            double y = nearbyint((site_value(piece, "y_mm") - y0) / cell);
            double cols = x + json_object_get_double(json_object_array_get_idx(shape, 1));
            double rows = y + json_object_get_double(json_object_array_get_idx(shape, 0));
            need(cols > 0 && cols < INT_MAX - 10 && rows > 0 && rows < INT_MAX - 10,
                "mosaic extent does not fit");
            w = fmax(w, cols + 10);
            h = fmax(h, rows + 10);
        }
    }
    size_t cells = (size_t)w * h;
    need(cells < UINT32_MAX / 2 && cells <= a.size / 16, "mosaic extent is too large");
    OutlineMask mask = {.w = w, .h = h, .mask = pushz(&a, cells)};
    size_t mark = a.used;
    for (size_t i = 0; i < count; i++) {
        if (group[i] != slot) {
            continue;
        }
        a.used = mark;
        json_object *piece = json_object_array_get_idx(pieces, i);
        const char *unit = site_string(piece, "unit");
        need(site_name(unit), "mosaic piece has an invalid unit");
        char *path =
            fmt(&a, "%s/pieces/%s/piece-%04d/z.tif", base, unit, (int)site_value(piece, "piece"));
        int width, height;
        f32 *z = tiff_read(&a, path, &width, &height);
        json_object *shape = json_object_object_get(piece, "shape");
        need(width == json_object_get_int(json_object_array_get_idx(shape, 1)) &&
                height == json_object_get_int(json_object_array_get_idx(shape, 0)),
            "mosaic piece shape changed at %s", path);
        json_object *overlap = json_object_object_get(piece, "overlap");
        int crop[4] = {0};
        for (size_t j = 0; j < 4 && j < site_items(overlap); j++) {
            crop[j] = json_object_get_int(json_object_array_get_idx(overlap, j));
            need(crop[j] >= 0 && crop[j] <= (j < 2 ? height : width),
                "mosaic piece has an invalid overlap");
        }
        int flip_y = json_object_get_boolean(json_object_object_get(piece, "flip_rows"));
        int flip_x = json_object_get_boolean(json_object_object_get(piece, "flip_cols"));
        int row0 = 5 + (int)nearbyint((site_value(piece, "y_mm") - y0) / cell);
        int col0 = 5 + (int)nearbyint((site_value(piece, "x_mm") - x0) / cell);
        for (int y = crop[0]; y < height - crop[1]; y++) {
            int row = row0 + (flip_y ? height - 1 - y : y);
            for (int x = crop[2]; x < width - crop[3]; x++) {
                int col = col0 + (flip_x ? width - 1 - x : x);
                if (z[(size_t)y * width + x] > 0) {
                    mask.mask[(size_t)row * w + col] = 1;
                }
            }
        }
    }
    a.used = mark;
    size_t retained = outline_close(&a, &mask, cell);
    a.used = mark;
    const char *name = rank == 1 ? fmt(&a, "largest mosaic · %.0f cm²", area)
                                 : fmt(&a, "mosaic %ld · %.0f cm²", rank, area);
    size_t points = 0;
    json_object *lines = outline_contours(&a, mask, name, x0, y0, cell, &points);
    json_object *extra = json_object_new_object();
    json_object_object_add(extra, "outlines", lines);
    const char *folder = fmt(&a, "%s/website/sheets", OUTPUTS);
    int lock = lock_file(fmt(&a, "%s/.publish.lock", folder), 1, 1);
    char *template = read_page_text(fmt(&a, "%s/viewer/assets/sheet_viewer.html", HOME));
    char *page = fmt(&a, "%s/%s/index.html", folder, argv[0]);
    need(lock >= 0 && template && hx_refresh_fields(page, template, extra) >= 0,
        "cannot update mosaic outline on %s", argv[0]);
    json_object *report = json_object_new_object();
    json_object_object_add(report, "page", json_object_new_string(argv[0]));
    json_object_object_add(report, "outline", json_object_new_string(name));
    json_object_object_add(
        report, "traced_cm2", jnum(decimal_round(retained * cell * cell / 100, 1)));
    json_object_object_add(report, "contours", json_object_new_int64(site_items(lines)));
    json_object_object_add(report, "points", json_object_new_int64(points));
    puts(json_object_to_json_string_ext(report, JSON_C_TO_STRING_PLAIN));
    json_object_put(report);
    json_object_put(extra);
    json_object_put(doc);
    free(template);
    close(lock);
    munmap(a.base, a.size);
    return 0;
}
