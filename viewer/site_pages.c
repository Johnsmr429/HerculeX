typedef struct {
    const char *scroll, *scan, *tag, *page;
    json_object *voxel, *builds;
    double fallback_um;
} SiteScan;

static int site_scan_order(const void *left, const void *right) {
    const SiteScan *a = left, *b = right;
    int name = strcmp(a->scroll, b->scroll);
    if (name) {
        return name;
    }
    double x = json_object_get_double(a->voxel), y = json_object_get_double(b->voxel);
    if (x != y) {
        return x > y ? -1 : 1;
    }
    return 0;
}

static json_object *site_scans(Arena *a, SiteScan **rows, size_t *count) {
    json_object *catalog = json_object_from_file(fmt(a, "%s/catalog/metadata.json", LIBRARY));
    json_object *samples = json_object_object_get(catalog, "samples");
    char *route = fmt(a, "%s/unwrapping/level", OUTPUTS);
    int scroll_count;
    char **scrolls = list_dir(a, route, &scroll_count);
    size_t first = a->used;
    for (int i = 0; i < scroll_count; i++) {
        const char *scroll = scrolls[i];
        int scan_count;
        size_t mark = a->used;
        char **scans = list_dir(a, fmt(a, "%s/%s", route, scroll), &scan_count);
        for (int j = 0; j < scan_count; j++) {
            size_t item = a->used;
            if (exists(fmt(a, "%s/%s/%s/sheet/layout.json", route, scroll, scans[j]))) {
                (*count)++;
            }
            a->used = item;
        }
        a->used = mark;
    }
    a->used = first;
    *rows = pushz(a, (*count + 1) * sizeof(**rows));
    size_t used = 0;
    for (int i = 0; i < scroll_count; i++) {
        const char *scroll = scrolls[i];
        int scan_count;
        char **scans = list_dir(a, fmt(a, "%s/%s", route, scroll), &scan_count);
        json_object *sample = json_object_object_get(samples, scroll);
        json_object *volumes = json_object_object_get(sample, "volumes");
        for (int j = 0; j < scan_count && used < *count; j++) {
            const char *scan = scans[j];
            size_t mark = a->used;
            if (!exists(fmt(a, "%s/%s/%s/sheet/layout.json", route, scroll, scan))) {
                a->used = mark;
                continue;
            }
            json_object *volume = json_object_object_get(volumes, scan);
            json_object *properties = json_object_object_get(volume, "properties");
            double energy =
                strcmp(scan, "20241024131838") ? site_value(properties, "energy_keV") : 0;
            const char *tag = energy ? fmt(a, "%s-%.0fkeV-level", scroll, energy)
                                     : fmt(a, "%s-%s-level", scroll, scan);
            (*rows)[used++] = (SiteScan){.scroll = scroll,
                .scan = scan,
                .tag = tag,
                .page = fmt(a, "sheet-%s", tag),
                .voxel = json_object_object_get(properties, "pixel_size_um")};
        }
    }
    *count = used;
    return catalog;
}

static int site_status_rows(Arena *a, SiteScan *scans, size_t count, json_object **status) {
    int laid = 0, read = 0;
    double area = 0, ink = 0;
    size_t mark = a->used;
    for (size_t i = 0; i < count;) {
        size_t selected = i++;
        while (i < count && !strcmp(scans[i].scroll, scans[selected].scroll)) {
            if (exists(fmt(a, "%s/website/sheets/%s/index.html", OUTPUTS, scans[i].page))) {
                selected = i;
            }
            i++;
            a->used = mark;
        }
        json_object *coverage =
            site_snapshot(a, scans[selected].scroll, scans[selected].scan, "level-0", 1);
        a->used = mark;
        if (!coverage) {
            return 0;
        }
        double written = site_value(coverage, "ink_cm2");
        ink += written;
        area += site_value(coverage, "laid_cm2");
        laid++;
        read += written > 0;
        json_object_put(coverage);
    }
    json_object *result = json_object_new_object();
    json_object_object_add(result, "scrolls", json_object_new_int(laid));
    json_object_object_add(result, "read_scrolls", json_object_new_int(read));
    json_object_object_add(result, "laid_cm2", jnum(area));
    json_object_object_add(result, "ink_cm2", jnum(ink));
    *status = result;
    return 1;
}

static int site_start_status(Arena *a, json_object **status) {
    struct stat root;
    if (stat(OUTPUTS, &root) || !S_ISDIR(root.st_mode)) {
        return 0;
    }
    size_t count = 0;
    SiteScan *scans;
    json_object *catalog = site_scans(a, &scans, &count);
    int ok = site_status_rows(a, scans, count, status);
    json_object_put(catalog);
    return ok;
}

static const char *site_surface(json_object *registered, const char *page) {
    for (size_t i = 0; i < site_items(registered); i++) {
        json_object *row = json_object_array_get_idx(registered, i);
        const char *name = site_string(row, "page"), *surface = site_string(row, "surface");
        if (name && surface && *surface && !strcmp(name, page)) {
            return surface;
        }
    }
    return strstr(page, "-continuous-frame") ? "frame" : "level";
}

static const char *site_fit(json_object *registered, const char *page) {
    for (size_t i = 0; i < site_items(registered); i++) {
        json_object *row = json_object_array_get_idx(registered, i);
        const char *name = site_string(row, "page"), *fit = site_string(row, "fit");
        if (name && fit && *fit && !strcmp(name, page)) {
            return fit;
        }
    }
    return strstr(page, "-trace") || strstr(page, "-v1.5") ? "trace 1.5" : "level 1.1";
}

static int site_prize_has(json_object *prizes, const char *name, const char *scroll) {
    json_object *list = json_object_object_get(prizes, name);
    for (size_t i = 0; i < site_items(list); i++) {
        const char *one = json_object_get_string(json_object_array_get_idx(list, i));
        if (one && !strcmp(one, scroll)) {
            return 1;
        }
    }
    return 0;
}

static const char *site_sample(Arena *a, const char *page) {
    for (const char *p = page; p && *p; p = strchr(p, '-')) {
        if (*p == '-') {
            p++;
        }
        if (!strncmp(p, "PHerc", 5)) {
            return fmt(a, "%.*s", (int)strcspn(p, "-"), p);
        }
    }
    return NULL;
}

typedef struct {
    const char *page, *scroll, *title;
    json_object *entry;
    i64 stamp;
    size_t order;
    double um;
} SiteCard;

static int site_card_order(const void *left, const void *right) {
    const SiteCard *a = left, *b = right;
    if (a->stamp != b->stamp) {
        return a->stamp > b->stamp ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static SiteCard site_build(Arena *a, const char *base, const char *page, const char *scroll,
    json_object *registered, size_t order) {
    SiteCard found = {0};
    char *folder = fmt(a, "%s/%s", base, page);
    if (site_size(fmt(a, "%s/index.html", folder)) < 0) {
        return found;
    }
    json_object *layers = json_object_from_file(fmt(a, "%s/layers.json", folder));
    const char *names[] = {"taught", "new-ink", "ink"};
    json_object *picked = NULL;
    for (size_t i = 0; !picked && i < 3; i++) {
        for (size_t j = 0; j < site_items(layers); j++) {
            json_object *layer = json_object_array_get_idx(layers, j);
            const char *name = site_string(layer, "path");
            if (name && !strcmp(name, names[i])) {
                picked = layer;
                break;
            }
        }
    }
    json_object *scaled = picked;
    for (size_t i = 0; (!scaled || !(site_value(scaled, "mm") > 0)) && i < site_items(layers);
        i++) {
        json_object *layer = json_object_array_get_idx(layers, i);
        if (site_value(layer, "mm") > 0) {
            scaled = layer;
        }
    }
    const char *model = site_string(picked, "model"), *surface = site_surface(registered, page);
    const char *fit = site_fit(registered, page);
    double um = 1000 * site_value(scaled, "mm");
    if (!(um > 0) || !isfinite(um)) {
        um = 0;
    }
    time_t stamp = json_object_get_int64(json_object_object_get(picked, "stamp"));
    struct tm utc;
    char date[64] = {0};
    if (stamp && gmtime_r(&stamp, &utc)) {
        strftime(date, sizeof(date), "%-d %B %Y", &utc);
    }
    json_object *entry = json_object_new_object();
    json_object_object_add(entry, "href", json_object_new_string(fmt(a, "%s/index.html", page)));
    json_object_object_add(entry, "model", model ? json_object_new_string(model) : NULL);
    json_object_object_add(entry, "surface", json_object_new_string(surface));
    json_object_object_add(entry, "fit", json_object_new_string(fit));
    json_object_object_add(entry, "date", json_object_new_string(date));
    json_object_object_add(entry, "stamp", json_object_new_int64(stamp));
    char *title = fmt(a, "%s%s%s · %s%s%s", scroll, model && *model ? " · " : "",
        model ? model : "", fit, *date ? " · " : "", date);
    found = (SiteCard){.page = page,
        .scroll = scroll,
        .entry = entry,
        .stamp = stamp,
        .order = order,
        .um = um,
        .title = title};
    json_object_put(layers);
    return found;
}

static json_object *site_catalogue(
    Arena *a, SiteScan *rows, size_t count, SiteCard *cards, size_t built, json_object *prizes) {
    SiteScan *all = push(a, (count + built + 1) * sizeof(*all));
    memcpy(all, rows, count * sizeof(*rows));
    for (size_t i = 0; i < count; i++) {
        all[i].builds = json_object_new_array();
    }
    for (size_t i = 0; i < built; i++) {
        SiteCard *card = cards + i;
        SiteScan *best = NULL;
        size_t longest = 0;
        for (size_t j = 0; j < count; j++) {
            SiteScan *row = all + j;
            if (strcmp(row->scroll, card->scroll) != 0) {
                continue;
            }
            if (!best) {
                best = row;
            }
            size_t length = row->page ? strlen(row->page) : 0;
            if (length > longest && !strncmp(card->page, row->page, length) &&
                (!card->page[length] || card->page[length] == '-')) {
                best = row;
                longest = length;
            }
        }
        if (!best) {
            best = all + count++;
            *best = (SiteScan){.scroll = card->scroll,
                .tag = card->page,
                .builds = json_object_new_array(),
                .fallback_um = card->um};
        } else if (!best->voxel && card->um > 0) {
            best->fallback_um = card->um;
        }
        json_object_array_add(best->builds, json_object_get(card->entry));
    }
    for (size_t i = 1; i < count; i++) {
        SiteScan moved = all[i];
        size_t j = i;
        while (j && site_scan_order(all + j - 1, &moved) > 0) {
            all[j] = all[j - 1];
            j--;
        }
        all[j] = moved;
    }
    json_object *result = json_object_new_array();
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_new_object();
        json_object_object_add(row, "scroll", json_object_new_string(all[i].scroll));
        json_object_object_add(row, "um",
            all[i].voxel ? json_object_get(all[i].voxel)
                         : (all[i].fallback_um > 0 ? jnum(all[i].fallback_um) : NULL));
        json_object_object_add(row, "tag", json_object_new_string(all[i].tag));
        json_object_object_add(
            row, "fpl", json_object_new_boolean(site_prize_has(prizes, "fpl", all[i].scroll)));
        json_object_object_add(
            row, "gp", json_object_new_boolean(site_prize_has(prizes, "gp", all[i].scroll)));
        json_object_object_add(row, "builds", all[i].builds);
        json_object_array_add(result, row);
    }
    return result;
}

static int site_home_report(
    Arena *a, const char *base, json_object *registered, int show, int status_file) {
    size_t count = 0, added = site_items(registered), built = 0;
    SiteScan *scans;
    json_object *catalog = site_scans(a, &scans, &count);
    SiteCard *cards = push(a, (count + added + 1) * sizeof(*cards));
    json_object *listed = json_object_new_object();
    for (size_t i = 0; i < count + added; i++) {
        int known = i < count;
        json_object *row = known ? NULL : json_object_array_get_idx(registered, i - count);
        const char *page = known ? scans[i].page : site_string(row, "page");
        const char *scroll = known ? scans[i].scroll : site_string(row, "scroll");
        if (site_name(page) && site_name(scroll) && !json_object_object_get(listed, page)) {
            SiteCard one = site_build(a, base, page, scroll, registered, i);
            if (one.entry) {
                cards[built++] = one;
                json_object_object_add(listed, page, json_object_new_boolean(1));
            }
        }
    }
    json_object_put(listed);
    qsort(cards, built, sizeof(*cards), site_card_order);
    json_object *prizes = json_object_from_file(fmt(a, "%s/data/catalog/prizes.json", HOME));
    if (!json_object_is_type(prizes, json_type_object)) {
        json_object_put(prizes);
        prizes = json_object_new_object();
    }
    json_object *status = NULL, *rows = site_catalogue(a, scans, count, cards, built, prizes);
    char *source = read_page_text(fmt(a, "%s/viewer/assets/start_page.html", HOME));
    char *viewer = read_page_text(fmt(a, "%s/viewer/assets/sheet_viewer.html", HOME));
    char *html = NULL;
    size_t bytes = 0;
    FILE *out = source && viewer && site_status_rows(a, scans, count, &status)
        ? open_memstream(&html, &bytes)
        : NULL;
    int ok = out != NULL;
    if (out) {
        const char *status_text = json_object_to_json_string_ext(status, JSON_C_TO_STRING_PLAIN);
        const char *rows_text = json_object_to_json_string_ext(
            rows, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE);
        const char *cursor = source;
        while (*cursor && ok) {
            if (!strncmp(cursor, "__STATUS__", 10)) {
                ok = fputs(status_text, out) >= 0;
                cursor += 10;
            } else if (!strncmp(cursor, "__SCROLLS__", 11)) {
                ok = fputs(rows_text, out) >= 0;
                cursor += 11;
            } else if (!strncmp(cursor, "__ATLAS__", 9)) {
                const char *pages[] = {"index", "lines", "published"};
                const char *names[] = {"Alphabet", "Lines", "Published"};
                for (size_t i = 0; ok && i < 3; i++) {
                    if (exists(fmt(a, "%s/website/atlas/%s.html", OUTPUTS, pages[i]))) {
                        ok = fprintf(out, "<a href=\"../atlas/%s.html\">%s</a>", pages[i],
                                 names[i]) >= 0;
                    }
                }
                if (ok && exists(fmt(a, "%s/synthetic/index.html", base))) {
                    ok = fputs("<a href=\"synthetic/index.html\">Synthetic</a>", out) >= 0;
                }
                cursor += 9;
            } else {
                ok = fputc(*cursor++, out) != EOF;
            }
        }
        ok = !fclose(out) && ok && hx_site_folders(base) &&
            hx_write_sheet_file(fmt(a, "%s/index.html", base), html, bytes);
    }
    if (ok && status_file) {
        size_t length;
        const char *value = json_object_to_json_string_length(
            status, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_NOSLASHESCAPE, &length);
        ok = hx_write_sheet_file(fmt(a, "%s/status.json", base), (void *)value, length);
    }
    int pages = 0;
    char **names = ok ? list_dir(a, base, &pages) : NULL;
    size_t mark = a->used;
    for (int i = 0; ok && i < pages; i++) {
        const char *name = names[i], *title = NULL;
        for (size_t j = 0; j < built; j++) {
            if (!strcmp(name, cards[j].page)) {
                title = cards[j].title;
                break;
            }
        }
        const char *ending = strstr(name, "-continuous-frame");
        SiteCard extra = {0};
        if (!title && !strncmp(name, "scroll-", 7) && ending && !ending[17]) {
            const char *scroll = site_sample(a, name);
            if (scroll) {
                extra = site_build(a, base, name, scroll, registered, 0);
                title = extra.title;
            }
        }
        int refreshed = hx_refresh_title(fmt(a, "%s/%s/index.html", base, name), viewer, title);
        json_object_put(extra.entry);
        ok = refreshed >= 0;
        a->used = mark;
    }
    for (size_t i = 0; i < built; i++) {
        if (show && ok) {
            puts(cards[i].title);
        }
        json_object_put(cards[i].entry);
    }
    free(html);
    free(source);
    free(viewer);
    json_object_put(status);
    json_object_put(rows);
    json_object_put(prizes);
    json_object_put(catalog);
    return ok;
}

static int site_home(Arena *a, const char *base, json_object *registered, int show) {
    return site_home_report(a, base, registered, show, 0);
}

static json_object *site_registered(Arena *a, const char *base) {
    char *path = fmt(a, "%s/cards.json", base);
    json_object *entries = exists(path) ? json_object_from_file(path) : json_object_new_array();
    if (!json_object_is_type(entries, json_type_array)) {
        json_object_put(entries);
        return NULL;
    }
    return entries;
}

static int site_card_image(Arena *a, const char *folder) {
    char *target = fmt(a, "%s/card.jpg", folder);
    if (site_size(target) >= 0) {
        return 1;
    }
    TileIndex index = {.pack = -1};
    int directory = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int opened = directory >= 0 ? tile_index_open(&index, directory) : 0;
    if (directory >= 0) {
        close(directory);
    }
    const char *names[] = {"layer0", "taught-scan", "sheet"};
    json_object *layers = json_object_from_file(fmt(a, "%s/layers.json", folder));
    double mm = 0;
    const char *name = NULL;
    for (size_t i = 0; opened == 1 && !name && i < 3; i++) {
        for (size_t j = 0; j < site_items(layers); j++) {
            json_object *row = json_object_array_get_idx(layers, j);
            const char *path = site_string(row, "path");
            if (path && !strcmp(path, names[i])) {
                name = names[i];
                mm = site_value(row, "mm");
                break;
            }
        }
    }
    TileSpan span;
    json_object *layer = name ? json_object_object_get(index.layers, name) : NULL;
    int valid = layer && mm > 0 && tile_index_level(&index, layer, 0, &span);
    int left = INT_MAX, top = INT_MAX, right = INT_MIN, bottom = INT_MIN;
    for (size_t i = 0; valid && i < (size_t)span.w * span.h; i++) {
        uint32_t size;
        uint64_t offset;
        valid = tile_index_blob(&index, span.first + i, &offset, &size);
        if (valid && size) {
            int x = span.x + (int)(i % span.w), y = span.y + (int)(i / span.w);
            left = x < left ? x : left;
            top = y < top ? y : top;
            right = x >= right ? x + 1 : right;
            bottom = y >= bottom ? y + 1 : bottom;
        }
    }
    valid = valid && right > left && bottom > top;
    int ok = 0;
    if (valid) {
        double scale = mm * 512, width = ((double)right - left) * scale;
        double extent[] = {left * scale, top * scale, width, ((double)bottom - top) * scale};
        ok = hx_sheet_card(folder, width > 8 * 1800 * mm ? 3 : 0, extent, target);
    }
    json_object_put(layers);
    tile_index_close(&index);
    return ok;
}

static int site_save_cards(Arena *a, const char *base, json_object *entries, const char *page,
    const char *scroll, const char *surface, const char *fit, int remove) {
    json_object *next = json_object_new_array();
    for (size_t i = 0; i < site_items(entries); i++) {
        json_object *row = json_object_array_get_idx(entries, i);
        const char *name = site_string(row, "page");
        if (!name || strcmp(name, page) != 0) {
            json_object_array_add(next, json_object_get(row));
        }
    }
    if (!remove) {
        json_object *row = json_object_new_object();
        json_object_object_add(row, "page", json_object_new_string(page));
        json_object_object_add(row, "scroll", json_object_new_string(scroll));
        if (surface && *surface) {
            json_object_object_add(row, "surface", json_object_new_string(surface));
        }
        if (fit && *fit) {
            json_object_object_add(row, "fit", json_object_new_string(fit));
        }
        json_object_array_add(next, row);
    }
    const char *value = json_object_to_json_string_ext(
        next, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_NOSLASHESCAPE);
    char *line = fmt(a, "%s\n", value);
    int ok = hx_write_sheet_file(fmt(a, "%s/cards.json", base), line, strlen(line)) &&
        site_home(a, base, next, 1);
    json_object_put(next);
    return ok;
}

int card_command(int argc, char **argv) {
    const char *page = argc ? argv[0] : NULL, *scroll = NULL, *surface = NULL, *fit = NULL;
    int remove = 0, valid = site_name(page);
    for (int i = 1; i < argc && valid; i++) {
        if (!strcmp(argv[i], "--remove")) {
            remove = 1;
        } else if (!strcmp(argv[i], "--surface") && i + 1 < argc) {
            surface = argv[++i];
        } else if (!strcmp(argv[i], "--fit") && i + 1 < argc) {
            fit = argv[++i];
        } else if (!scroll && site_name(argv[i]) && strncmp(argv[i], "--", 2) != 0) {
            scroll = argv[i];
        } else {
            valid = 0;
        }
    }
    need(valid, "usage: ./run.sh card PAGE [SCROLL] [--surface WORD] [--fit WORD] [--remove]");
    Arena a = arena(64 << 20);
    if (!scroll && !remove) {
        scroll = site_sample(&a, page);
        need(scroll != NULL, "no scroll named in %s, give its scroll after the page", page);
    }
    char *base = fmt(&a, "%s/website/sheets", OUTPUTS), *folder = fmt(&a, "%s/%s", base, page);
    need(remove || site_size(fmt(&a, "%s/index.html", folder)) >= 0, "no published page at %s",
        folder);
    int lock = lock_file(fmt(&a, "%s/.publish.lock", base), 1, 1);
    need(lock >= 0, "cannot lock %s", base);
    json_object *entries = site_registered(&a, base);
    int ok = entries && (remove || site_card_image(&a, folder)) &&
        site_save_cards(&a, base, entries, page, scroll, surface, fit, remove);
    json_object_put(entries);
    close(lock);
    munmap(a.base, a.size);
    need(ok, "cannot update the start-page card for %s", page);
    return 0;
}

int hx_sheet_tiles_command(int argc, char **argv) {
    char *args[argc + 2];
    args[0] = "viewer.sheet_tiles";
    memcpy(args + 1, argv, (size_t)argc * sizeof(*args));
    args[argc + 1] = NULL;
    return hx_sheet_tiles_main(argc + 1, args);
}

int hx_refresh_viewers_command(int argc, char **argv) {
    char *args[argc + 2];
    args[0] = "viewer.refresh_viewers";
    memcpy(args + 1, argv, (size_t)argc * sizeof(*args));
    args[argc + 1] = NULL;
    return hx_refresh_viewers_main(argc + 1, args);
}

static int site_review_pack(const char *folder, char pack[64], uint64_t *size) {
    TileIndex index = {.pack = -1};
    int directory = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int ok = directory >= 0 && tile_index_open(&index, directory) == 1;
    if (directory >= 0) {
        close(directory);
    }
    const char *layers[] = {"layer0", "sheet", "taught"};
    for (size_t i = 0; ok && i < 3; i++) {
        TileSpan span;
        json_object *layer = json_object_object_get(index.layers, layers[i]);
        ok = tile_index_level(&index, layer, 0, &span);
    }
    if (ok) {
        ok = snprintf(pack, 64, "%s", index.name) < 64;
        *size = index.pack_bytes;
    }
    tile_index_close(&index);
    return ok;
}

static int site_current_model(const Stage *stage, const char *model_name) {
    if (model_name && !strcmp(stage->digest, model_name)) {
        return 1;
    }
    for (int i = 0; model_name && i < stage->sames; i++) {
        if (!strcmp(stage->same[i], model_name)) {
            return 1;
        }
    }
    return 0;
}

static const char *site_lanes[][2] = {
    {"ink/model/reader_apple/reader.m", "ink/model/reader_apple/reader.metal"},
    {"ink/model/reader_cuda/reader.c", "ink/model/reader_cuda/reader.cu"},
};

static void site_lane_digest(Arena *a, const Model *m, int lane, char *out) {
    const char *sources[] = {
        "ink/ink.h", "sampling/sampling.h", site_lanes[lane][0], site_lanes[lane][1]};
    size_t mark = a->used;
    *out = 0;
    if (exists(fmt(a, "%s/%s", HOME, sources[2])) && exists(fmt(a, "%s/%s", HOME, sources[3]))) {
        EVP_MD_CTX *c = EVP_MD_CTX_new();
        need(c && EVP_DigestInit_ex(c, EVP_sha256(), NULL), "cannot start ink model hash");
        ink_hash_file(c, m->weights);
        for (int k = 0; k < 4; k++) {
            ink_hash_file(c, fmt(a, "%s/%s", HOME, sources[k]));
        }
        u8 d[32];
        need(EVP_DigestUpdate(c, SAMPLING, strlen(SAMPLING)) && EVP_DigestFinal_ex(c, d, NULL),
            "cannot finish ink model hash");
        EVP_MD_CTX_free(c);
        for (int i = 0; i < 6; i++) {
            snprintf(out + 2 * i, 3, "%02x", d[i]);
        }
    }
    a->used = mark;
}

static json_object *site_kept_review(Arena *a, const Stage *stage, const char *receipt,
    const char *layout, const char *folder, const char *href, const char *scan) {
    if (mtime_ns(receipt) < mtime_ns(layout)) {
        return NULL;
    }
    json_object *saved = json_object_from_file(receipt);
    const char *recorded = site_string(saved, "model_digest"), *page = site_string(saved, "page");
    const char *existing = site_string(saved, "pack");
    json_object *size = json_object_object_get(saved, "pack_bytes");
    json_object *coverage = json_object_object_get(saved, "coverage");
    char pack[64];
    uint64_t bytes;
    int ok = site_current_model(stage, recorded) && page && !strcmp(page, href) &&
        json_object_is_type(size, json_type_int) &&
        json_object_is_type(coverage, json_type_object) && site_review_pack(folder, pack, &bytes) &&
        (!existing || !strcmp(existing, pack)) && json_object_get_uint64(size) == bytes;
    char *html = ok ? read_page_text(fmt(a, "%s/index.html", folder)) : NULL;
    ok = html && strstr(html, scan);
    free(html);
    if (!ok) {
        json_object_put(saved);
        saved = NULL;
    }
    return saved;
}

int publish_command(int argc, char **argv) {
    const char *url = env("HERCULEX_SITE", "http://127.0.0.1:8765");
    int current = argc == 3 && !strcmp(argv[0], "--current");
    if (current) {
        argc--;
        argv++;
    }
    int valid = argc == 2 || (argc == 4 && !strcmp(argv[2], "--url") && *argv[3]);
    need(valid && site_name(argv[0]) && site_name(argv[1]),
        "usage: ./run.sh publish [--current] SCROLL SCAN [--url URL]");
    if (argc == 4) {
        url = argv[3];
    }
    Arena a = arena(4 * GB);
    char *site_url = fmt(&a, "%s", url);
    size_t end = strlen(site_url);
    while (end && site_url[end - 1] == '/') {
        site_url[--end] = 0;
    }
    const char *scroll = argv[0], *scan = argv[1];
    Stage stage = {.model = chosen_model(&a)};
    const char *production = fmt(&a, "level-0%s", stage.model.suffix);
    need(!strcmp(stage.model.faces, production) && !*env("HERCULEX_UNITS", ""),
        "whole-scroll review requires every unit in the scan's production face folder");
    size_t mark = a.used;
    model_digest(&a, &stage.model, stage.digest);
    a.used = mark;
    int here = strcmp(LANE_READER, site_lanes[0][0]) != 0;
    char lanes[2][16];
    if (!strcmp(stage.model.backend, "studio")) {
        model_digest(&a, &stage.model, lanes[0]);
        memcpy(lanes[1], lanes[0], sizeof(lanes[0]));
    } else {
        for (int k = 0; k < 2; k++) {
            site_lane_digest(&a, &stage.model, k, lanes[k]);
        }
    }
    need(!strcmp(lanes[here], stage.digest), "the composer's lane digest is not model_digest");
    char *other = lanes[!here];
    if (*other) {
        stage.same[stage.sames++] = other;
    }
    char *same = (char *)slurp(&a, fmt(&a, "%s/ink/reading/same_output_models.txt", HOME)).data;
    char *line;
    while ((line = strsep(&same, "\n"))) {
        char *current = strsep(&line, " \t");
        if (strcmp(current, stage.digest) != 0 && (!*other || strcmp(current, other) != 0)) {
            continue;
        }
        for (char *older = strsep(&line, " \t"); older && stage.sames < 256;
            older = strsep(&line, " \t")) {
            if (*older) {
                stage.same[stage.sames++] = older;
            }
        }
    }
    char *base = fmt(&a, "%s/unwrapping/level/%s/%s", OUTPUTS, scroll, scan);
    char *root = fmt(&a, "%s/website/sheets", OUTPUTS);
    char *layout_path = fmt(&a, "%s/sheet/layout.json", base);
    json_object *layout = json_object_from_file(layout_path);
    json_object *pieces = json_object_object_get(layout, "pieces");
    size_t count = site_items(pieces);
    need(count, "%s %s: no layout to publish", scroll, scan);
    size_t scans_count = 0;
    SiteScan *scans;
    json_object *catalog = site_scans(&a, &scans, &scans_count);
    const char *tag = NULL;
    for (size_t i = 0; i < scans_count; i++) {
        if (!strcmp(scans[i].scroll, scroll) && !strcmp(scans[i].scan, scan)) {
            tag = scans[i].tag;
            break;
        }
    }
    json_object_put(catalog);
    need(tag != NULL, "%s %s: no layout to publish", scroll, scan);
    char *name = fmt(&a, "sheet-%s%s", tag, stage.model.suffix);
    char *folder = fmt(&a, "%s/%s", root, name);
    char *href = fmt(&a, "/sheets/%s/index.html", name);
    char *scan_key = fmt(&a, "level/%s/%s", scroll, scan);
    char *receipt = fmt(&a, "%s/sheet/review%s.json", base, stage.model.suffix);
    if (current) {
        json_object *saved =
            site_kept_review(&a, &stage, receipt, layout_path, folder, href, scan_key);
        int stale = !saved;
        json_object_put(saved);
        json_object_put(layout);
        munmap(a.base, a.size);
        return stale;
    }
    int lock = lock_file(fmt(&a, "%s/.publish.lock", root), 1, 1);
    need(lock >= 0, "cannot lock %s", root);
    json_object *entries = site_registered(&a, root);
    need(entries != NULL, "cannot read %s/cards.json", root);
    json_object *saved = site_kept_review(&a, &stage, receipt, layout_path, folder, href, scan_key);
    const char *face_root = fmt(&a, "%s/faces/%s", base, production);
    if (!saved) {
        size_t pending = 0;
        for (size_t i = 0; i < count; i++) {
            mark = a.used;
            json_object *piece = json_object_array_get_idx(pieces, i);
            const char *unit = site_string(piece, "unit");
            int id = site_value(piece, "piece");
            need(site_name(unit) && id >= 0, "%s: invalid piece in %s", scan_key, layout_path);
            Job job = {.piece = id, .turn = 1};
            json_object *turn = json_object_object_get(piece, "turn");
            if (turn) {
                job.turn = json_object_get_int(turn);
            }
            int out_len =
                snprintf(job.out, sizeof(job.out), "%s/%s/piece-%04d.png", face_root, unit, id);
            int source_len =
                snprintf(job.folder, sizeof(job.folder), "%s/pieces/%s/piece-%04d", base, unit, id);
            need(out_len > 0 && out_len < (int)sizeof(job.out) && source_len > 0 &&
                    source_len < (int)sizeof(job.folder),
                "%s: piece path too long", scan_key);
            pending += !exists(job.out) || !exists(beside(&a, job.out, "-layer0.png")) ||
                !ink_is_current(&stage, &job, &a);
            a.used = mark;
        }
        need(!pending, "%s: %zu pieces still need ink", scan_key, pending);
        json_object *coverage = site_snapshot(&a, scroll, scan, production, 0);
        need(coverage && (size_t)site_value(coverage, "ink_pieces") == count,
            "%s: ink maps missing from the whole-scroll layout", scan_key);
        double mm = site_value(coverage, "voxel_um") / 1000;
        need(mm > 0, "%s: geometry records have no voxel size", scan_key);
        char *args[] = {"viewer.sheet_tiles", (char *)scroll, (char *)scan, fmt(&a, "%.17g", mm),
            name, "layer0,sheet,taught", "--model", stage.model.title, "--ink-name", "ink",
            "--written-face", "--review-surface", (char *)site_surface(entries, name), NULL};
        need(!hx_sheet_tiles_main(13, args), "%s: could not publish tiles", scan_key);
        char pack[64];
        uint64_t bytes;
        need(site_review_pack(folder, pack, &bytes), "%s: could not open published pack", scan_key);
        saved = json_object_new_object();
        json_object_object_add(saved, "page", json_object_new_string(href));
        json_object_object_add(saved, "weights", json_object_new_string(stage.model.weights));
        json_object_object_add(saved, "model_digest", json_object_new_string(stage.digest));
        json_object_object_add(saved, "pack", json_object_new_string(pack));
        json_object_object_add(saved, "pack_bytes", json_object_new_uint64(bytes));
        json_object_object_add(saved, "coverage", coverage);
        size_t len;
        const char *json = json_object_to_json_string_length(
            saved, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_NOSLASHESCAPE, &len);
        need(hx_write_sheet_file(receipt, (void *)json, len), "cannot write %s", receipt);
    }
    int registered = 0;
    for (size_t i = 0; i < site_items(entries); i++) {
        const char *entry = site_string(json_object_array_get_idx(entries, i), "page");
        registered |= entry && !strcmp(entry, name);
    }
    int ok = *stage.model.suffix && !registered
        ? site_save_cards(&a, root, entries, name, scroll, NULL, NULL, 0)
        : site_home(&a, root, entries, 0);
    need(ok, "%s: could not update the start page", scan_key);
    char *check[] = {"--verify", folder, scan_key, site_url, NULL};
    need(!hx_check_review(4, check), "%s: could not verify the review page", scan_key);
    const char *suffixes[] = {".png", "-layer0.png", "-taught.png"};
    for (size_t i = 0; i < count; i++) {
        mark = a.used;
        json_object *piece = json_object_array_get_idx(pieces, i);
        const char *unit = site_string(piece, "unit");
        int id = site_value(piece, "piece");
        if (site_name(unit) && id >= 0) {
            for (size_t j = 0; j < 3; j++) {
                char *path = fmt(&a, "%s/%s/piece-%04d%s", face_root, unit, id, suffixes[j]);
                need(!unlink(path) || errno == ENOENT, "cannot remove %s: %s", path,
                    strerror(errno));
            }
        }
        a.used = mark;
    }
    json_object *coverage = json_object_object_get(saved, "coverage");
    printf("%s: %lld pieces, %.2f GB of review tiles\n", href,
        (long long)site_value(coverage, "ink_pieces"), site_value(saved, "pack_bytes") / 1e9);
    json_object_put(saved);
    json_object_put(entries);
    json_object_put(layout);
    close(lock);
    munmap(a.base, a.size);
    return 0;
}

static json_object *site_read_pieces(
    Arena *a, const char *base, const char *faces, const char *units) {
    json_object *layout = json_object_from_file(fmt(a, "%s/sheet/layout.json", base));
    json_object *pieces = json_object_object_get(layout, "pieces");
    json_object *found = json_object_new_array();
    const char *filter = units && *units ? fmt(a, ",%s,", units) : NULL;
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(pieces); i++) {
        json_object *piece = json_object_array_get_idx(pieces, i);
        const char *unit = site_string(piece, "unit");
        int id = site_value(piece, "piece");
        if (site_name(unit) && id >= 0 && (!filter || strstr(filter, fmt(a, ",%s,", unit))) &&
            exists(fmt(a, "%s/faces/%s/%s/piece-%04d.png", base, faces, unit, id))) {
            json_object_array_add(found, json_object_get(piece));
        }
        a->used = mark;
    }
    json_object_put(layout);
    return found;
}

static int site_part_page(int argc, char **argv) {
    need((argc == 4 || argc == 5) && site_name(argv[0]) && site_name(argv[2]) && site_name(argv[3]),
        "usage: ./run.sh viewer.site_pages part NAME TITLE SCROLL SCAN [UNITS]");
    Arena a = arena(64 << 20);
    const char *page = argv[0], *title = argv[1], *scroll = argv[2], *scan = argv[3];
    const char *units = argc == 5 ? argv[4] : NULL;
    const char *model = env("HERCULEX_INK_MODEL", "reader-v3");
    const char *default_faces =
        !strcmp(model, "reader-v3") ? "level-0" : fmt(&a, "level-0-%s", model);
    const char *faces = env("HERCULEX_FACES", default_faces);
    char *base = fmt(&a, "%s/unwrapping/level/%s/%s", OUTPUTS, scroll, scan);
    char *folder = fmt(&a, "%s/website/sheets/%s", OUTPUTS, page);
    json_object *pieces = site_read_pieces(&a, base, faces, units);
    need(site_items(pieces), "no read pieces of %s in faces/%s",
        units && *units ? units : "any unit", faces);
    json_object *layers = json_object_from_file(fmt(&a, "%s/layers.json", folder));
    int ok = site_items(layers) &&
        hx_sheet_page(pieces, layers, folder, title, fmt(&a, "part/%s", page), -1);
    json_object_put(layers);
    json_object_put(pieces);
    munmap(a.base, a.size);
    need(ok, "cannot publish part page %s", page);
    return 0;
}

static int site_bulk_sheet(Arena *a, const char *root, const SiteScan *scan) {
    char *base = fmt(a, "%s/unwrapping/level/%s/%s", OUTPUTS, scan->scroll, scan->scan);
    char *folder = fmt(a, "%s/%s", root, scan->page);
    char *largest = fmt(a, "%s/largest-area-%s", root, scan->tag);
    char *target = fmt(a, "%s/proposals-%s", root, scan->tag);
    char *largest_meta = fmt(a, "%s/layers.json", largest);
    char *whole_meta = fmt(a, "%s/layers.json", folder);
    int draw_largest = exists(largest_meta);
    int draw_whole = exists(whole_meta) && !exists(fmt(a, "%s/sheet/review.json", base));
    if (!draw_largest && !draw_whole) {
        return 1;
    }
    json_object *visible = site_read_pieces(a, base, "level-0", NULL);
    if (!site_items(visible)) {
        json_object_put(visible);
        return 1;
    }
    json_object *catalog = json_object_from_file(fmt(a, "%s/catalog/metadata.json", LIBRARY));
    json_object *samples = json_object_object_get(catalog, "samples");
    json_object *volumes =
        json_object_object_get(json_object_object_get(samples, scan->scroll), "volumes");
    json_object *volume = json_object_object_get(volumes, scan->scan);
    double energy = strcmp(scan->scan, "20241024131838")
        ? site_value(json_object_object_get(volume, "properties"), "energy_keV")
        : 0;
    json_object_put(catalog);
    const char *title = energy ? fmt(a, "%s · %.0f keV", scan->scroll, energy)
                               : fmt(a, "%s · %s", scan->scroll, scan->scan);
    const char *key = fmt(a, "level/%s/%s", scan->scroll, scan->scan);
    json_object *layout = json_object_from_file(fmt(a, "%s/sheet/layout.json", base));
    json_object *pieces = json_object_object_get(layout, "pieces");
    json_object *layers = json_object_from_file(draw_whole ? whole_meta : largest_meta);
    json_object *suggested = draw_whole ? site_suggestions(a, scan->scroll, scan->scan) : NULL;
    json_object *meta =
        hx_sheet_metadata(pieces, visible, layers, title, key, NULL, NULL, suggested);
    json_object *places = json_object_object_get(meta, "places");
    json_object *first = site_items(places) ? json_object_array_get_idx(places, 0) : NULL;
    double extent[] = {site_value(first, "x"), site_value(first, "y"), site_value(first, "w"),
        site_value(first, "h")};
    int ok = first != NULL;
    if (ok && draw_largest) {
        json_object *raw = json_object_from_file(largest_meta);
        json_object *page = hx_sheet_metadata(
            NULL, NULL, raw, fmt(a, "%s · largest area", title), key, extent, NULL, NULL);
        ok = hx_sheet_render(page, largest) &&
            hx_sheet_card(largest, 0, extent, fmt(a, "%s/card.jpg", largest));
        json_object_put(page);
        json_object_put(raw);
    }
    if (ok && draw_whole) {
        ok = hx_sheet_render(meta, folder) &&
            hx_sheet_card(folder, 3, extent, fmt(a, "%s/card.jpg", folder));
        size_t mark = a->used;
        json_object *boxes = NULL;
        ok = ok && site_window_proposals(a, scan->scroll, scan->scan, &boxes);
        a->used = mark;
        json_object *outlined = json_object_new_array();
        for (size_t i = 0; ok && i < site_items(boxes); i++) {
            json_object *box = json_object_array_get_idx(boxes, i);
            double xywh[] = {site_value(box, "x"), site_value(box, "y"), site_value(box, "w"),
                site_value(box, "h")};
            if (!i) {
                ok = hx_site_folders(target) &&
                    hx_sheet_card(folder, 0, xywh, fmt(a, "%s/card.jpg", target));
            }
            json_object_array_add(outlined, site_rectangle(xywh[0], xywh[1], xywh[2], xywh[3]));
        }
        json_object *linked = json_object_new_array();
        for (size_t i = 0; ok && i < site_items(layers); i++) {
            json_object *copy = NULL;
            json_object *layer = json_object_array_get_idx(layers, i);
            const char *path = site_string(layer, "path");
            ok = path && !json_object_deep_copy(layer, &copy, NULL);
            if (ok) {
                json_object_object_add(
                    copy, "path", json_object_new_string(fmt(a, "../%s/%s", scan->page, path)));
                json_object_array_add(linked, copy);
            }
        }
        const char *label = fmt(a, "%s · proposed", title);
        json_object *page = ok
            ? hx_sheet_metadata(pieces, NULL, linked, label, key, NULL, outlined, suggested)
            : NULL;
        ok = ok && hx_site_folders(target) && hx_sheet_render(page, target);
        json_object_put(page);
        json_object_put(linked);
        json_object_put(outlined);
        json_object_put(boxes);
    }
    json_object_put(meta);
    json_object_put(suggested);
    json_object_put(layers);
    json_object_put(layout);
    json_object_put(visible);
    return ok;
}

int hx_site_pages_command(int argc, char **argv) {
    if (argc && !strcmp(argv[0], "card")) {
        return card_command(argc - 1, argv + 1);
    }
    if (argc && !strcmp(argv[0], "part")) {
        return site_part_page(argc - 1, argv + 1);
    }
    int home = argc == 1 && !strcmp(argv[0], "home");
    need(!argc || home, "usage: ./run.sh viewer.site_pages [card|part|home] ...");
    Arena a = arena(64 << 20);
    const char *base = fmt(&a, "%s/website/sheets", OUTPUTS);
    int lock = lock_file(fmt(&a, "%s/.publish.lock", base), 1, 1);
    need(lock >= 0, "cannot lock %s", base);
    if (home) {
        json_object *entries = site_registered(&a, base);
        int ok = entries && site_home(&a, base, entries, 0);
        json_object_put(entries);
        close(lock);
        munmap(a.base, a.size);
        need(ok, "cannot write the start page");
        return 0;
    }
    size_t count = 0;
    SiteScan *scans;
    json_object *catalog = site_scans(&a, &scans, &count);
    if (!count) {
        puts("No laid-out scans to publish.");
        json_object_put(catalog);
        close(lock);
        munmap(a.base, a.size);
        return 0;
    }
    json_object *entries = site_registered(&a, base);
    int ok = entries != NULL;
    size_t mark = a.used;
    for (size_t i = 0; ok && i < count; i++) {
        ok = site_bulk_sheet(&a, base, scans + i);
        a.used = mark;
    }
    ok = ok && site_home_report(&a, base, entries, 1, 1);
    json_object_put(entries);
    json_object_put(catalog);
    close(lock);
    munmap(a.base, a.size);
    need(ok, "cannot write the start page");
    return 0;
}
