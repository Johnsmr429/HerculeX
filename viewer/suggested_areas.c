static json_object *site_rectangle(double x, double y, double w, double h) {
    json_object *box = json_object_new_object();
    json_object_object_add(box, "x", jnum(x));
    json_object_object_add(box, "y", jnum(y));
    json_object_object_add(box, "w", jnum(w));
    json_object_object_add(box, "h", jnum(h));
    return box;
}

static json_object *site_suggestions(Arena *a, const char *scroll, const char *scan) {
    json_object *found = json_object_new_array();
    if (strcmp(scroll, "PHerc0211") != 0 || strcmp(scan, "20250821151803") != 0) {
        return found;
    }
    char *scored_path = fmt(a, "%s/records/ink/blind-pairs.json", HOME);
    json_object *scored_doc = json_object_from_file(scored_path);
    const char *base = fmt(a, "%s/unwrapping/level/%s/%s", OUTPUTS, scroll, scan);
    json_object *placed_doc = json_object_from_file(fmt(a, "%s/sheet/placed-taught.json", base));
    json_object *placed = json_object_object_get(placed_doc, "pieces");
    json_object *ranked = json_object_object_get(scored_doc, "ranked");
    size_t ranks = site_items(ranked), pieces = site_items(placed);
    const int order[] = {29, 21, 17, 14, 25};
    for (size_t i = 0; i < sizeof(order) / sizeof(*order); i++) {
        json_object *pair = NULL, *piece = NULL;
        for (size_t j = 0; j < ranks; j++) {
            json_object *row = json_object_array_get_idx(ranked, j);
            if (site_value(row, "number") == order[i]) {
                pair = row;
                break;
            }
        }
        const char *unit = site_string(pair, "unit");
        int id = site_value(pair, "piece");
        for (size_t j = 0, count = site_name(unit) ? pieces : 0; j < count; j++) {
            json_object *row = json_object_array_get_idx(placed, j);
            const char *name = site_string(row, "unit");
            if (name && !strcmp(name, unit) && site_value(row, "piece") == id) {
                piece = row;
                break;
            }
        }
        if (!piece) {
            continue;
        }
        char *path = fmt(a, "%s/faces/level-0/%s/piece-%04d-taught.png", base, unit, id);
        png_image image = {.version = PNG_IMAGE_VERSION};
        if (!png_image_begin_read_from_file(&image, path)) {
            continue;
        }
        json_object *corner = json_object_object_get(pair, "corner");
        if (site_items(corner) == 2) {
            double y = json_object_get_double(json_object_array_get_idx(corner, 0));
            double x = json_object_get_double(json_object_array_get_idx(corner, 1));
            double square[4];
            hx_sheet_suggested_square(square, site_value(piece, "x_mm"), site_value(piece, "y_mm"),
                site_value(piece, "um"), site_value(piece, "stride"), x, y, image.width,
                image.height, json_object_get_boolean(json_object_object_get(piece, "flip_cols")),
                json_object_get_boolean(json_object_object_get(piece, "flip_rows")));
            json_object *box = site_rectangle(square[0], square[1], square[2], square[3]);
            char *label = fmt(a, "%zu · %s piece %d", i + 1, unit, id);
            json_object_object_add(box, "label", json_object_new_string(label));
            json_object_array_add(found, box);
        }
        png_image_free(&image);
    }
    json_object_put(placed_doc);
    json_object_put(scored_doc);
    return found;
}
