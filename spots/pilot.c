static json_object *spot_find(Arena *a, const char *id) {
    json_object *rows = spot_rows(a, "spots"), *found = NULL;
    for (size_t i = 0; i < site_items(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        if (!strcmp(id, jstr(row, "spot_id"))) {
            found = json_object_get(row);
            break;
        }
    }
    json_object_put(rows);
    need(found != NULL, "no spot %s in the catalogue; run ./run.sh spots.catalogue", id);
    return found;
}

static void spot_record_surfaces(Arena *a, json_object *spot, json_object *document, int grown) {
    int count = grown ? 1 : 2, first = grown ? 2 : 0;
    json_object *held = json_object_new_array_ext(count);
    json_object *records = jget(document, "pieces"), *pieces = json_object_new_array();
    json_object *detail = json_object_new_object();
    const char *work = spot_work(a, spot);
    const char *snapshot = grown ? "grow" : "snap";
    i64 nodes = 0, before = 0;
    double changed = 0;
    for (size_t i = 0; i < site_items(records); i++) {
        json_object *r = json_object_array_get_idx(records, i), *snap = jget(r, snapshot);
        const char *piece = spot_piece(r);
        i64 size = ji64(snap, "nodes");
        json_object_array_add(pieces, json_object_new_string(piece));
        json_object_object_add(detail, piece, json_object_get(snap));
        nodes += size;
        if (grown)
            before += ji64(snap, "nodes_before");
        else
            changed += jf64(snap, "snapped") * size;
    }
    const char *snap_text = json_object_to_json_string_ext(detail, JSON_C_TO_STRING_SPACED);
    for (int s = first; s < first + count; s++) {
        const char *surface = spot_surface_names[s];
        json_object *row = json_object_new_object();
        json_object_object_add(row, "spot_id", json_object_get(jget(spot, "spot_id")));
        json_object_object_add(row, "surface", json_object_new_string(surface));
        json_object_object_add(row, "pieces", json_object_get(pieces));
        json_object_object_add(row, "nodes", json_object_new_int64(nodes));
        if (grown) {
            json_object_object_add(row, "nodes_before", json_object_new_int64(before));
        } else {
            json_object_object_add(
                row, "snapped_share", spot_round(s ? changed / fmax(1, nodes) : 0, 10000));
        }
        json_object_object_add(row, "snap", json_object_new_string(s ? snap_text : ""));
        json_object_object_add(row, "margin_mm", jnum(3));
        json_object_object_add(row, "layers", json_object_new_int(SPOT_RENDER_LAYERS));
        json_object_object_add(
            row, "folder", json_object_new_string(fmt(a, "%s/surfaces/%s", work, surface)));
        json_object_array_add(held, row);
    }
    json_object_put(detail);
    json_object_put(pieces);
    spot_write(a, "surfaces", held);
    json_object_put(held);
}

int spot_pilot_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh spots.pilot SPOT [build] [grow] [render] [read] [images]";
    if (!argc || !strcmp(argv[0], "-h") || !strcmp(argv[0], "--help")) {
        puts(usage);
        return argc == 0;
    }
    Arena a = arena(64 * GB);
    const char *id = argv[0];
    json_object *spot = spot_find(&a, id), *document = NULL;
    const char *key = jstr(spot, "sheet_key");
    SpotSources sources = spot_sources(&a, fmt(&a, "%s/marks", LIBRARY));
    const SpotSource *sheet = NULL;
    for (size_t i = 0; i < sources.count && !sheet; i++) {
        if (!strcmp(key, sources.rows[i].key)) {
            sheet = sources.rows + i;
        }
    }
    need(sheet != NULL, "no sheet %s for spot %s", key, id);
    const char *source = fmt(&a, "%s/pieces.json", spot_work(&a, spot));
    const char *stages[] = {"build", "grow", "render", "read", "images"};
    u8 *wanted = push(&a, (size_t)(argc > 6 ? argc - 1 : 5));
    int n = 0;
    for (int i = 1; i < argc; i++) {
        for (int j = 0; j < 5; j++) {
            if (!strcmp(argv[i], stages[j])) {
                wanted[n++] = j;
            }
        }
    }
    if (!n) {
        for (; n < 5; n++) {
            wanted[n] = n;
        }
    }
    for (int i = 0; i < n; i++) {
        size_t mark = a.used;
        double began = now();
        int stage = wanted[i];
        if (!stage) {
            json_object_put(document);
            document = spot_surface_build(&a, spot, sheet);
            spot_record_surfaces(&a, spot, document, 0);
        } else {
            if (!document) {
                document = json_read(source);
                need(document != NULL, "spot %s needs to be built first", id);
            }
            if (stage == 1) {
                spot_surface_grow(&a, spot, document);
                spot_record_surfaces(&a, spot, document, 1);
            } else if (stage == 2) {
                spot_surface_render(&a, spot, sheet, document);
            } else if (stage == 3) {
                spot_sweep(&a, spot, document);
            } else {
                spot_images(&a, spot, document);
            }
        }
        json_object_object_add(spot, "status", json_object_new_string(stages[stage]));
        json_object *updated = json_object_new_array_ext(1);
        json_object_array_add(updated, json_object_get(spot));
        spot_write(&a, "spots", updated);
        json_object_put(updated);
        printf("%s %s done in %.1f s\n", id, stages[stage], now() - began);
        fflush(stdout);
        a.used = mark;
    }
    json_object_put(document);
    json_object_put(spot);
    return 0;
}
