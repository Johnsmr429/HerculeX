typedef struct {
    double share, x, y;
    int agreed;
} SiteProposal;

typedef struct {
    SiteProposal *rows;
    size_t count, capacity;
} SiteProposals;

static int site_proposal_order(const void *left, const void *right) {
    const SiteProposal *a = left, *b = right;
    if (a->agreed != b->agreed) {
        return b->agreed - a->agreed;
    }
    const double x[] = {a->share, a->x, a->y}, y[] = {b->share, b->x, b->y};
    for (int i = 0; i < 3; i++) {
        if (x[i] != y[i]) {
            return x[i] > y[i] ? -1 : 1;
        }
    }
    return 0;
}

static int site_proposal_piece(Arena *a, json_object *piece, const u8 *grey, const u8 *scan,
    int image_w, int image_h, int agreed, SiteProposals *windows) {
    double mm = site_value(piece, "um");
    double pixels = nearbyint(.15 / mm);
    if (!(pixels >= 1 && pixels <= INT_MAX)) {
        return 0;
    }
    int step = pixels;
    int w = image_w / step, h = image_h / step, side = 133, jump = 33;
    if (w < side || h < side) {
        return 1;
    }
    int pitch = w + 1;
    size_t grid = (size_t)pitch * (h + 1);
    double *strong = pushz(a, grid * sizeof(*strong)), *mass = pushz(a, grid * sizeof(*mass));
    uint32_t *clear = pushz(a, grid * sizeof(*clear)), *cover = pushz(a, grid * sizeof(*cover));
    int flip_rows = json_object_get_boolean(json_object_object_get(piece, "flip_rows"));
    int flip_cols = json_object_get_boolean(json_object_object_get(piece, "flip_cols"));
    int full = 1;
    size_t samples = (size_t)step * step;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t held = 0, ink = 0;
            for (int dy = 0; dy < step; dy++) {
                int row = y * step + dy;
                if (flip_rows) {
                    row = image_h - 1 - row;
                }
                for (int dx = 0; dx < step; dx++) {
                    int col = x * step + dx;
                    if (flip_cols) {
                        col = image_w - 1 - col;
                    }
                    size_t at = (size_t)row * image_w + col;
                    held += scan[at] != 0;
                    ink += grey[at] >= 235;
                }
            }
            size_t at = (size_t)(y + 1) * pitch + x + 1;
            int empty = held * 2 <= samples;
            clear[at] = clear[at - 1] + empty;
            strong[at] = (double)ink / samples;
            full &= !empty;
        }
    }
    const int rim[] = {6, 6, 6, 5, 5, 4, 2};
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int inner = full ? (((double)x * x + (double)(y + 1) * (y + 1)) * .15 * .15 > 1) : 1;
            for (int dy = -6; inner && !full && dy <= 6; dy++) {
                int line = y + dy;
                if (line < 0 || line >= h) {
                    continue;
                }
                int radius = rim[abs(dy)];
                int left = x > radius ? x - radius : 0;
                int right = x + radius + 1 < w ? x + radius + 1 : w;
                const uint32_t *row = clear + (size_t)(line + 1) * pitch;
                inner = row[left] == row[right];
            }
            size_t at = (size_t)(y + 1) * pitch + x + 1;
            cover[at] = cover[at - pitch] + inner;
            mass[at] = mass[at - pitch] + (inner ? strong[at] : 0);
        }
    }
    for (int y = 1; y <= h; y++) {
        for (int x = 1; x <= w; x++) {
            size_t at = (size_t)y * pitch + x;
            cover[at] += cover[at - 1];
            mass[at] += mass[at - 1];
        }
    }
    double margin = 2 * mm * site_value(piece, "stride");
    double x0 = site_value(piece, "x_mm") - margin, y0 = site_value(piece, "y_mm") - margin;
    for (int y = 0; y <= h - side; y += jump) {
        for (int x = 0; x <= w - side; x += jump) {
            size_t tl = (size_t)y * pitch + x, tr = tl + side;
            size_t bl = tl + (size_t)side * pitch, br = bl + side;
            int area = (int)cover[br] - (int)cover[tr] - (int)cover[bl] + (int)cover[tl];
            if (area * 2 < side * side) {
                continue;
            }
            double share = (mass[br] - mass[tr] - mass[bl] + mass[tl]) / area;
            if (share > .25) {
                continue;
            }
            if (windows->count == windows->capacity) {
                size_t capacity = windows->capacity ? 2 * windows->capacity : 256;
                SiteProposal *next = realloc(windows->rows, capacity * sizeof(*next));
                if (!next) {
                    return 0;
                }
                windows->rows = next;
                windows->capacity = capacity;
            }
            windows->rows[windows->count++] =
                (SiteProposal){share, x0 + x * .15, y0 + y * .15, agreed};
        }
    }
    return 1;
}

static int site_window_proposals(
    Arena *a, const char *scroll, const char *scan, json_object **boxes_out) {
    char *base = fmt(a, "%s/unwrapping/level/%s/%s", OUTPUTS, scroll, scan);
    char *folder = fmt(a, "%s/faces/level-0", base);
    json_object *layout = json_object_from_file(fmt(a, "%s/sheet/layout.json", base));
    json_object *pieces = json_object_object_get(layout, "pieces");
    int taught = 0, ok = json_object_is_type(pieces, json_type_array);
    glob_t paths = {0};
    if (ok) {
        int found = glob(fmt(a, "%s/*/piece-*-taught.png", folder), 0, NULL, &paths);
        taught = found == 0;
        globfree(&paths);
    }
    ok = ok && hx_sheet_orient(pieces, folder, taught ? "taught-side" : "side");
    SiteProposals candidates = {0};
    size_t mark = a->used;
    for (size_t i = 0; ok && i < site_items(pieces); i++) {
        a->used = mark;
        json_object *piece = json_object_array_get_idx(pieces, i);
        const char *unit = site_string(piece, "unit");
        if (!site_name(unit) || site_value(piece, "width_mm") < 10 ||
            site_value(piece, "height_mm") < 10) {
            continue;
        }
        char *stem = fmt(a, "%s/%s/piece-%04d", folder, unit, (int)site_value(piece, "piece"));
        char *ink_path = fmt(a, "%s-%s.png", stem, taught ? "taught" : "ink");
        char *scan_path = fmt(a, "%s.png", stem);
        if (!exists(ink_path) || !exists(scan_path)) {
            continue;
        }
        int agreed = 0;
        if (taught) {
            json_object *meta = json_object_from_file(fmt(a, "%s-taught-side.json", stem));
            const char *written = site_string(meta, "written");
            int far = written && !strcmp(written, "far");
            double here = site_value(meta, far ? "far" : "near");
            double other = site_value(meta, far ? "near" : "far");
            int reads =
                written && here >= .1 && other <= .4 * here && site_value(meta, "scrambled") <= .01;
            const char *fibre = site_string(json_object_object_get(meta, "fibres"), "face");
            agreed = reads && fibre && !strcmp(fibre, written);
            json_object_put(meta);
            if (!reads) {
                continue;
            }
        }
        int w, h, sw, sh;
        u8 *ink = hx_sheet_png(ink_path, &w, &h), *raw = hx_sheet_png(scan_path, &sw, &sh);
        ok = ink && raw && w == sw && h == sh &&
            site_proposal_piece(a, piece, ink, raw, w, h, agreed, &candidates);
        free(ink);
        free(raw);
    }
    a->used = mark;
    json_object *boxes = json_object_new_array();
    if (candidates.count) {
        qsort(candidates.rows, candidates.count, sizeof(*candidates.rows), site_proposal_order);
    }
    double x[40], y[40];
    for (size_t i = 0, used = 0; ok && used < 40 && i < candidates.count; i++) {
        SiteProposal row = candidates.rows[i];
        int separate = 1;
        for (size_t j = 0; separate && j < used; j++) {
            separate = fabs(x[j] - row.x) >= 20 || fabs(y[j] - row.y) >= 20;
        }
        if (separate) {
            x[used] = decimal_round(row.x, 2);
            y[used] = decimal_round(row.y, 2);
            json_object *box = site_rectangle(x[used], y[used], 20, 20);
            json_object_object_add(box, "share", jnum(decimal_round(row.share, 4)));
            json_object_object_add(box, "fibres_agree", json_object_new_boolean(row.agreed));
            json_object_array_add(boxes, box);
            used++;
        }
    }
    if (ok) {
        json_object *report = json_object_new_object();
        json_object_object_add(report, "boxes", json_object_get(boxes));
        json_object_object_add(report, "windows", json_object_new_int64(candidates.count));
        size_t length;
        const char *value = json_object_to_json_string_length(
            report, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_NOSLASHESCAPE, &length);
        ok = hx_write_sheet_file(fmt(a, "%s/sheet/proposals.json", base), (void *)value, length);
        json_object_put(report);
        printf("%s windows %zu\n", scroll, candidates.count);
    }
    free(candidates.rows);
    json_object_put(layout);
    if (ok) {
        *boxes_out = boxes;
    } else {
        json_object_put(boxes);
    }
    return ok;
}

int hx_window_proposals_command(int argc, char **argv) {
    need(argc == 2 && site_name(argv[0]) && site_name(argv[1]),
        "usage: ./run.sh viewer.window_proposals SCROLL SCAN");
    Arena a = arena(4 * GB);
    json_object *boxes = NULL;
    int ok = site_window_proposals(&a, argv[0], argv[1], &boxes);
    json_object_put(boxes);
    munmap(a.base, a.size);
    need(ok, "%s %s: cannot propose windows", argv[0], argv[1]);
    return 0;
}
