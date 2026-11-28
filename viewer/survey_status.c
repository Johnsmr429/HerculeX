static double site_value(json_object *entry, const char *key) {
    return json_object_get_double(json_object_object_get(entry, key));
}

static off_t site_size(const char *path) {
    struct stat info;
    return !stat(path, &info) && S_ISREG(info.st_mode) ? info.st_size : -1;
}

static json_object *site_snapshot(
    Arena *a, const char *scroll, const char *scan, const char *faces, int use_review) {
    char *base = fmt(a, "%s/unwrapping/level/%s/%s", OUTPUTS, scroll, scan);
    char *layout = fmt(a, "%s/sheet/layout.json", base);
    char *review = fmt(a, "%s/sheet/review.json", base);
    i64 stamp = mtime_ns(layout);
    if (use_review && stamp && mtime_ns(review) >= stamp) {
        json_object *doc = json_object_from_file(review);
        json_object *coverage = json_object_get(json_object_object_get(doc, "coverage"));
        json_object_put(doc);
        if (coverage) {
            return coverage;
        }
    }
    json_object *doc = json_object_from_file(layout);
    json_object *pieces = json_object_object_get(doc, "pieces");
    if (!json_object_is_type(pieces, json_type_array)) {
        json_object_put(doc);
        return NULL;
    }
    json_object *groups = json_object_new_object();
    size_t count = json_object_array_length(pieces);
    double *areas = pushz(a, (count + 1) * sizeof(*areas));
    i64 raw_bytes = 0, ink_bytes = 0, raw_pixels = 0, laid_pixels = 0;
    size_t raw = 0, ink = 0, sides = 0, far = 0, turned = 0;
    double laid_area = 0, raw_area = 0, ink_area = 0;
    for (size_t i = 0; i < count; i++) {
        size_t mark = a->used;
        json_object *piece = json_object_array_get_idx(pieces, i);
        const char *unit = site_string(piece, "unit");
        int slot = hx_mosaic_slot(groups, piece);
        if (!site_name(unit) || slot < 1 || (size_t)slot > count) {
            json_object_put(groups);
            json_object_put(doc);
            return NULL;
        }
        double area = site_value(piece, "area");
        areas[slot] += area;
        laid_area += area;
        json_object *turn = json_object_object_get(piece, "turn");
        turned += turn && json_object_get_int(turn) < 0;
        json_object *shape = json_object_object_get(piece, "shape");
        i64 pixels = 0;
        if (json_object_is_type(shape, json_type_array) && json_object_array_length(shape) >= 2) {
            i64 stride = site_value(piece, "stride");
            i64 h = json_object_get_int64(json_object_array_get_idx(shape, 0));
            i64 w = json_object_get_int64(json_object_array_get_idx(shape, 1));
            pixels = ((h + 3) * stride + 1) * ((w + 3) * stride + 1);
        }
        laid_pixels += pixels;
        char *stem =
            fmt(a, "%s/faces/%s/%s/piece-%04d", base, faces, unit, (int)site_value(piece, "piece"));
        off_t bytes = site_size(fmt(a, "%s-layer0.png", stem));
        if (bytes >= 0) {
            raw++;
            raw_area += area;
            raw_bytes += bytes;
            raw_pixels += pixels;
        }
        bytes = site_size(fmt(a, "%s-taught.png", stem));
        if (bytes >= 0) {
            ink++;
            ink_area += area;
            ink_bytes += bytes;
        }
        char *side_path = fmt(a, "%s-side.json", stem);
        if (exists(side_path)) {
            json_object *side = json_object_from_file(side_path);
            const char *written = site_string(side, "written");
            sides++;
            far += written && !strcmp(written, "far");
            json_object_put(side);
        }
        a->used = mark;
    }
    int group_count = json_object_object_length(groups), large = 0;
    double large_area = 0, largest = 0, unwrapped = 0;
    for (int i = 1; i <= group_count; i++) {
        double area = areas[i] / 100;
        largest = fmax(largest, area);
        if (area >= 4) {
            large++;
            large_area += area;
        }
    }
    json_object_put(groups);
    json_object *voxel = NULL, *level = NULL;
    int units_count;
    char **units = list_dir(a, fmt(a, "%s/pieces", base), &units_count);
    int first = 1;
    for (int pass = 0; first && pass < 2; pass++) {
        const char *prefix = pass ? "band-" : "rows-";
        for (int i = 0; i < units_count; i++) {
            if (strncmp(units[i], prefix, 5) != 0) {
                continue;
            }
            char *path = fmt(a, "%s/pieces/%s/record.json", base, units[i]);
            json_object *unit = json_object_from_file(path);
            if (unit) {
                unwrapped += site_value(unit, "area_cm2");
                if (first) {
                    voxel = json_object_get(json_object_object_get(unit, "voxel_um"));
                    level = json_object_get(json_object_object_get(unit, "level"));
                    first = 0;
                }
            }
            json_object_put(unit);
        }
    }
    json_object *report = json_object_new_object(), *record = json_object_object_get(doc, "record");
    json_object *mosaics = json_object_object_get(record, "mosaics");
    json_object_object_add(report, "sample", json_object_new_string(scroll));
    json_object_object_add(report, "volume", json_object_new_string(scan));
    json_object_object_add(
        report, "updated", json_object_new_string(utc_stamp(a, "%Y-%m-%dT%H:%M:%S+00:00")));
    json_object_object_add(report, "voxel_um", voxel);
    json_object_object_add(report, "geometry_level", level);
    json_object_object_add(report, "unwrapped_cm2", jnum(unwrapped));
    json_object_object_add(report, "laid_cm2", jnum(laid_area / 100));
    json_object_object_add(report, "pieces", json_object_new_int64(count));
    json_object_object_add(report, "logical_mosaics",
        mosaics ? json_object_get(mosaics) : json_object_new_int64(count));
    json_object_object_add(report, "display_mosaics", json_object_new_int(group_count));
    json_object_object_add(
        report, "joints", json_object_get(json_object_object_get(record, "joints")));
    json_object_object_add(
        report, "relocated_pieces", json_object_get(json_object_object_get(record, "moved")));
    json_object_object_add(report, "largest_cm2", jnum(largest));
    json_object_object_add(report, "large_mosaics", json_object_new_int(large));
    json_object_object_add(report, "large_cm2", jnum(large_area));
    json_object_object_add(report, "raw_pieces", json_object_new_int64(raw));
    json_object_object_add(report, "ink_pieces", json_object_new_int64(ink));
    json_object_object_add(report, "raw_cm2", jnum(raw_area / 100));
    json_object_object_add(report, "ink_cm2", jnum(ink_area / 100));
    json_object_object_add(report, "raw_png_bytes", json_object_new_int64(raw_bytes));
    json_object_object_add(report, "ink_png_bytes", json_object_new_int64(ink_bytes));
    json_object_object_add(report, "raw_rendered_pixels", json_object_new_int64(raw_pixels));
    json_object_object_add(report, "sides_checked", json_object_new_int64(sides));
    json_object_object_add(report, "written_on_far_side", json_object_new_int64(far));
    json_object_object_add(report, "turned_to_their_sheet", json_object_new_int64(turned));
    json_object_object_add(report, "layout_pixels", json_object_new_int64(laid_pixels));
    json_object_object_add(report, "caveat",
        json_object_new_string(
            "Areas follow the current mesh layout. Global scroll order and wrong-wrap rate are "
            "unverified; ink maps are predictions."));
    json_object_put(doc);
    return report;
}

int hx_survey_status_command(int argc, char **argv) {
    need(argc == 2 && site_name(argv[0]) && site_name(argv[1]),
        "usage: ./run.sh viewer.survey_status SCROLL SCAN");
    Arena a = arena(64 << 20);
    json_object *report = site_snapshot(&a, argv[0], argv[1], "level-0", 1);
    need(report != NULL, "no layout for %s %s", argv[0], argv[1]);
    puts(json_object_to_json_string_ext(
        report, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_NOSLASHESCAPE));
    json_object_put(report);
    munmap(a.base, a.size);
    return 0;
}
