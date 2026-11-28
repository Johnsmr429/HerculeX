typedef struct {
    json_object *row;
    size_t order;
} AtlasRow;

static const char *atlas_text(json_object *row, const char *name) {
    const char *value = json_object_get_string(json_object_object_get(row, name));
    return value ? value : "";
}

static int atlas_compare_double(double a, double b) { return (a > b) - (a < b); }

static int atlas_compare_segment(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int x = !strcmp(atlas_text(a->row, "kind"), "ink");
    int y = !strcmp(atlas_text(b->row, "kind"), "ink");
    if (x != y) {
        return y - x;
    }
    int rows = atlas_compare_double(site_value(b->row, "rows"), site_value(a->row, "rows"));
    return rows ? rows : (a->order > b->order) - (a->order < b->order);
}

static int atlas_compare_position(json_object *a, json_object *b) {
    double ae = site_value(a, "energy"), be = site_value(b, "energy");
    int a_other = ae != 77 && ae != 78, b_other = be != 77 && be != 78;
    if (a_other != b_other) {
        return a_other - b_other;
    }
    int a_fine = site_value(a, "um") < 2.3, b_fine = site_value(b, "um") < 2.3;
    if (a_fine != b_fine) {
        return a_fine - b_fine;
    }
    int sample = strcmp(atlas_text(a, "sample"), atlas_text(b, "sample"));
    return sample ? sample : strcmp(atlas_text(a, "segment"), atlas_text(b, "segment"));
}

static int atlas_compare_cards(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int position = atlas_compare_position(a->row, b->row);
    if (position) {
        return position;
    }
    int id = atlas_compare_double(site_value(a->row, "id"), site_value(b->row, "id"));
    return id ? id : (a->order > b->order) - (a->order < b->order);
}

static int atlas_compare_windows(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int position = atlas_compare_position(a->row, b->row);
    if (position) {
        return position;
    }
    int written =
        atlas_compare_double(site_value(b->row, "letters"), site_value(a->row, "letters"));
    return written ? written : (a->order > b->order) - (a->order < b->order);
}

static json_object *atlas_load_rows(
    Arena *a, const char *work, const char *name, int (*compare)(const void *, const void *)) {
    char *path = fmt(a, "%s/%s.json", work, name);
    json_object *input = exists(path) ? json_read(path) : json_object_new_array();
    need(json_object_is_type(input, json_type_array), "%s must hold an array", path);
    size_t count = site_items(input);
    if (!compare || count < 2) {
        return input;
    }
    size_t mark = a->used;
    AtlasRow *sorted = push(a, count * sizeof(*sorted));
    for (size_t i = 0; i < count; i++) {
        sorted[i] = (AtlasRow){json_object_array_get_idx(input, i), i};
    }
    qsort(sorted, count, sizeof(*sorted), compare);
    json_object *result = json_object_new_array_ext(count);
    for (size_t i = 0; i < count; i++) {
        json_object_array_add(result, json_object_get(sorted[i].row));
    }
    json_object_put(input);
    a->used = mark;
    return result;
}

static const char *atlas_file[] = {"index.html", "lines.html", "published.html"};
static const char *atlas_title[] = {"Alphabet", "Lines", "Published"};

static void atlas_gallery(FILE *out, const char *name, const char *grid, const char *heading) {
    fprintf(out, "<section id=\"%s\">", name);
    if (*heading) {
        fprintf(out, "<h2>%s</h2>", heading);
    }
    fprintf(out,
        "<div class=\"bar\"><button class=\"back\">←</button>"
        "<button class=\"next\">→</button><span class=\"status\"></span></div>"
        "<div class=\"%s gallery\"></div></section>",
        grid);
}

static void atlas_link_image(FILE *out, const char *link, const char *image) {
    fprintf(out, "<a href=\"%s\"><img src=\"%s\" alt=\"\"></a>", link, image);
}

static int atlas_number_order(const void *left, const void *right) {
    return atlas_compare_double(*(const double *)left, *(const double *)right);
}

static double atlas_median(double *numbers, size_t count) {
    if (!count) {
        return 0;
    }
    qsort(numbers, count, sizeof(*numbers), atlas_number_order);
    return count % 2 ? numbers[count / 2] : .5 * (numbers[count / 2 - 1] + numbers[count / 2]);
}

static int atlas_summary_order(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int sample = strcmp(atlas_text(a->row, "sample"), atlas_text(b->row, "sample"));
    if (sample) {
        return sample;
    }
    int energy = atlas_compare_double(site_value(a->row, "energy"), site_value(b->row, "energy"));
    if (energy) {
        return energy;
    }
    return strcmp(atlas_text(a->row, "case"), atlas_text(b->row, "case"));
}

static void atlas_summary(Arena *a, FILE *out, json_object *cards) {
    size_t count = site_items(cards), rewind = a->used;
    AtlasRow *rows = push(a, count * sizeof(*rows));
    for (size_t i = 0; i < count; i++) {
        rows[i] = (AtlasRow){json_object_array_get_idx(cards, i), i};
    }
    qsort(rows, count, sizeof(*rows), atlas_summary_order);
    double *height = push(a, count * sizeof(*height)), *width = push(a, count * sizeof(*width));
    double *spacing = push(a, count * sizeof(*spacing));
    fputs("<table><tr><th>Scroll</th><th class=n>Energy</th><th class=n>Letters</th>"
          "<th class=n>Height mm</th><th class=n>Width mm</th><th class=n>Line mm</th></tr>",
        out);
    for (size_t i = 0; i < count;) {
        const char *sample = atlas_text(rows[i].row, "sample");
        double energy = site_value(rows[i].row, "energy");
        size_t last = i;
        while (last < count && !strcmp(sample, atlas_text(rows[last].row, "sample")) &&
            site_value(rows[last].row, "energy") == energy) {
            height[last - i] = site_value(rows[last].row, "h");
            width[last - i] = site_value(rows[last].row, "w");
            last++;
        }
        size_t used = 0;
        for (size_t start = i; start < last;) {
            const char *name = atlas_text(rows[start].row, "case");
            size_t end = start + 1;
            while (end < last && !strcmp(name, atlas_text(rows[end].row, "case"))) {
                end++;
            }
            for (size_t j = start; j < end; j++) {
                double x = site_value(rows[j].row, "x_mm"), y = site_value(rows[j].row, "y_mm");
                double gap = INFINITY;
                for (size_t k = start; k < end; k++) {
                    double x2 = site_value(rows[k].row, "x_mm"),
                           y2 = site_value(rows[k].row, "y_mm");
                    if (fabs(x2 - x) < 4 && y2 > y + 2.5 && y2 < y + 14) {
                        gap = fmin(gap, y2 - y);
                    }
                }
                if (isfinite(gap)) {
                    spacing[used++] = gap;
                }
            }
            start = end;
        }
        fprintf(out,
            "<tr><td>%s</td><td class=n>%s keV</td><td class=n>%zu</td>"
            "<td class=n>%.1f</td><td class=n>%.1f</td><td class=n>%.1f</td></tr>",
            sample, atlas_text(rows[i].row, "energy"), last - i, atlas_median(height, last - i),
            atlas_median(width, last - i), atlas_median(spacing, used));
        i = last;
    }
    fputs("</table>", out);
    a->used = rewind;
}

static void atlas_alphabet(FILE *out, json_object *rows) {
    fputs("<table class=\"alphabet\" id=\"alphabet\">", out);
    for (size_t i = 0; i < site_items(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        fprintf(out, "<tr><td class=\"glyph\">%s<small>%s</small></td><td class=\"models\">",
            atlas_text(row, "capital"), atlas_text(row, "small"));
        json_object *models = json_object_object_get(row, "models");
        for (size_t j = 0; j < site_items(models); j++) {
            fprintf(out, "<img src=\"%s\" alt=\"\">",
                json_object_get_string(json_object_array_get_idx(models, j)));
        }
        fprintf(out, "</td><td class=n>%s</td><td><div class=\"examples\">",
            atlas_text(row, "matched"));
        json_object *examples = json_object_object_get(row, "examples");
        for (size_t j = 0; j < site_items(examples); j++) {
            json_object *example = json_object_array_get_idx(examples, j);
            const char *file = atlas_text(example, "file");
            if (*file) {
                fprintf(out, "<div class=\"example %s\">", atlas_text(example, "state"));
                atlas_link_image(out, file, file);
                fputs("</div>", out);
            }
        }
        fputs("</div></td></tr>", out);
    }
    fputs("</table>", out);
}

static void atlas_infrared(FILE *out, json_object *fragments) {
    fputs("<section id=\"infrared\">", out);
    for (size_t i = 0; i < site_items(fragments); i++) {
        json_object *row = json_object_array_get_idx(fragments, i);
        fprintf(out, "<h2>Infrared · %s · %s</h2><div class=\"grid three\">",
            atlas_text(row, "name"), atlas_text(row, "sample"));
        const char *names[] = {"ir", "face", "labels"};
        for (size_t j = 0; j < 3; j++) {
            const char *image = atlas_text(row, names[j]);
            if (*image) {
                atlas_link_image(out, image, image);
            }
        }
        fputs("</div><div class=\"grid wide\">", out);
        json_object *windows = json_object_object_get(row, "windows");
        for (size_t j = 0; j < site_items(windows); j++) {
            const char *image = json_object_get_string(json_object_array_get_idx(windows, j));
            atlas_link_image(out, image, image);
        }
        fputs("</div>", out);
    }
    fputs("</section>", out);
}

static const char *atlas_scripts[] = {
    "gallery(\"letters\",ATLAS.cards,[{label:\"Scroll\",get:i=>i.sample},"
    "{label:\"Energy\",get:i=>i.energy+\" keV\"},{label:\"Letter\",get:i=>i.letter},"
    "{label:\"Voxel\",get:i=>i.um+\" µm\"}],i=>`<figure><a href=\"${i.file}\">"
    "<img src=\"${i.file}\" alt=\"\"></a><figcaption>${i.sample} · ${i.energy} keV · "
    "${i.um} µm${i.letter?\" · \"+i.letter:\"\"}</figcaption></figure>`)",
    "gallery(\"scans\",ATLAS.compare,[{label:\"Scroll\",get:i=>i.sample},"
    "{label:\"Segment\",get:i=>i.segment}],i=>`<figure><a href=\"${i.file}\">"
    "<img src=\"${i.file}\" alt=\"\"></a><figcaption>${i.sample} · "
    "${i.scans}</figcaption></figure>`)\n"
    "gallery(\"lines\",ATLAS.windows,[{label:\"Scroll\",get:i=>i.sample},"
    "{label:\"Energy\",get:i=>i.energy+\" keV\"},{label:\"Segment\",get:i=>i.segment}],"
    "i=>`<figure><a href=\"${i.file}\"><img src=\"${i.file}\" alt=\"\"></a>"
    "<figcaption>${i.sample} · ${i.energy} keV · ${i.um} µm</figcaption></figure>`)",
    "gallery(\"published\",ATLAS.segments,[{label:\"Scroll\",get:i=>i.sample},"
    "{label:\"Kind\",get:i=>i.kind},{label:\"Scan\",get:i=>i.scan},"
    "{label:\"Recipe\",get:i=>i.recipe}],i=>`<figure><a href=\"${i.full||i.file}\">"
    "<img src=\"${i.file}\" alt=\"\"></a><figcaption>${i.sample} · "
    "${i.scan}</figcaption></figure>`)"};

static void atlas_page(Arena *a, const char *folder, int page, json_object *cards,
    json_object *segments, json_object *fragments, json_object *alphabet) {
    char *data = NULL;
    size_t size = 0;
    FILE *out = open_memstream(&data, &size);
    need(out != NULL, "cannot open an atlas page");
    fprintf(out,
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>%s</title><link rel=\"stylesheet\" href=\"atlas.css\"></head>"
        "<body><nav><a href=\"../sheets/index.html\">HerculeX</a>",
        atlas_title[page]);
    for (int i = 0; i < 3; i++) {
        fprintf(out, "<a href=\"%s\"%s>%s</a>", atlas_file[i], i == page ? " class=here" : "",
            atlas_title[i]);
    }
    fputs("</nav><main>", out);
    if (page == 0) {
        fputs("<a href=\"#alphabet\"><img class=\"strip\" src=\"alphabet/strip.png\" alt=\"\"></a>"
              "<div class=\"grid two\">",
            out);
        if (site_items(fragments)) {
            atlas_link_image(out, "lines.html#infrared",
                atlas_text(json_object_array_get_idx(fragments, 0), "ir"));
        }
        json_object *best = NULL;
        for (size_t i = 0; i < site_items(segments); i++) {
            json_object *row = json_object_array_get_idx(segments, i);
            if (!best || site_value(row, "rows") > site_value(best, "rows")) {
                best = row;
            }
        }
        if (best) {
            atlas_link_image(out, "published.html", atlas_text(best, "file"));
        }
        fputs("</div>", out);
        atlas_summary(a, out, cards);
        atlas_alphabet(out, alphabet);
        atlas_gallery(out, "letters", "grid", "Letters");
    } else if (page == 1) {
        atlas_infrared(out, fragments);
        atlas_gallery(out, "scans", "grid wide", "Scans");
        atlas_gallery(out, "lines", "grid wide", "Lines");
    } else {
        atlas_gallery(out, "published", "grid", "");
    }
    fprintf(out,
        "</main><script src=\"data.js\"></script><script src=\"atlas.js\"></script>"
        "<script>%s</script></body></html>",
        atlas_scripts[page]);
    need(!fclose(out) && hx_write_sheet_file(fmt(a, "%s/%s", folder, atlas_file[page]), data, size),
        "cannot write atlas page %s", atlas_file[page]);
    free(data);
}

static const char *atlas_work(Arena *a) {
    return env("HERCULEX_ATLAS_WORK", fmt(a, "%s/website/atlas-work", OUTPUTS));
}

int hx_atlas_pages_command(int argc, char **argv) {
    need(argc == 1 && !strcmp(argv[0], "pages"), "usage: ./run.sh viewer.atlas_build pages");
    Arena a = arena(128 << 20);
    char *folder = fmt(&a, "%s/website/atlas", OUTPUTS);
    const char *work = atlas_work(&a);
    mkdirs(folder);
    const char *assets[] = {"atlas.css", "atlas.js"};
    for (size_t i = 0; i < 2; i++) {
        Bytes asset = read_file(&a, fmt(&a, "%s/viewer/assets/%s", HOME, assets[i]));
        need(hx_write_sheet_file(fmt(&a, "%s/%s", folder, assets[i]), asset.data, asset.size),
            "cannot copy %s", assets[i]);
    }
    json_object *cards = atlas_load_rows(&a, work, "cards", atlas_compare_cards);
    json_object *windows = atlas_load_rows(&a, work, "windows", atlas_compare_windows);
    json_object *segments = atlas_load_rows(&a, work, "segments", atlas_compare_segment);
    json_object *compare = atlas_load_rows(&a, work, "compare", NULL);
    json_object *fragments = atlas_load_rows(&a, work, "fragments", NULL);
    json_object *alphabet = json_read(fmt(&a, "%s/alphabet.json", work));
    json_object *data = json_object_new_object();
    json_object_object_add(data, "cards", cards);
    json_object_object_add(data, "windows", windows);
    json_object_object_add(data, "segments", segments);
    json_object_object_add(data, "compare", compare);
    char *script = fmt(&a, "window.ATLAS=%s;",
        json_object_to_json_string_ext(
            data, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE));
    need(hx_write_sheet_file(fmt(&a, "%s/data.js", folder), script, strlen(script)),
        "cannot write atlas data");
    for (int i = 0; i < 3; i++) {
        atlas_page(
            &a, folder, i, cards, segments, fragments, json_object_object_get(alphabet, "table"));
    }
    int count;
    char **files = list_dir(&a, folder, &count);
    for (int i = 0; i < count; i++) {
        const char *suffix = strrchr(files[i], '.');
        if (suffix && !strcmp(suffix, ".html") && strcmp(files[i], atlas_file[0]) != 0 &&
            strcmp(files[i], atlas_file[1]) != 0 && strcmp(files[i], atlas_file[2]) != 0) {
            const char *path = fmt(&a, "%s/%s", folder, files[i]);
            need(!unlink(path), "cannot remove outdated atlas page %s", path);
        }
    }
    json_object_put(alphabet);
    json_object_put(data);
    json_object_put(fragments);
    munmap(a.base, a.size);
    return 0;
}

static int atlas_wrap(int at, int size) {
    int relative = at % size;
    return relative < 0 ? relative + size : relative;
}

static u8 *atlas_mask(Arena *a, const char *folder, const char *name, json_object *row,
    double lateral, int step, int *height, int *wide) {
    json_object *box = jget(row, "box");
    need(site_items(box) == 4, "labelled case %s has an invalid box", folder);
    int b[4];
    for (int i = 0; i < 4; i++) {
        b[i] = json_object_get_int(json_object_array_get_idx(box, i));
    }
    int h = b[1] - b[0], w = b[3] - b[2];
    need(w > 0 && h > 0 && (size_t)w * h <= 16 * GB, "invalid mask shape in %s", folder);
    Array bits = npy_read(a, fmt(a, "%s/%s.npy", folder, name));
    need(!strcmp(bits.descr, "|u1") && bits.size >= ((size_t)w * h + 7) / 8,
        "invalid ink mask in %s", folder);
    *height = 1 + (h - 1) / step;
    *wide = 1 + (w - 1) / step;
    int ny = *height, nx = *wide, shift[2] = {0};
    json_object *reg = json_read(fmt(a, "%s/registration.json", folder));
    json_object *offset = json_object_object_get(reg, "offset_um");
    if (offset) {
        need(site_items(offset) == 2, "invalid mask registration in %s", folder);
        for (int d = 0; d < 2; d++) {
            double um = json_object_get_double(json_object_array_get_idx(offset, d));
            shift[d] = (int)fmod(nearbyint(um / (lateral * step)), d ? nx : ny);
        }
    }
    json_object_put(reg);
    u8 *ink = push(a, (size_t)ny * nx);
    const u8 *packed = bits.data;
    for (int y = 0; y < ny; y++) {
        size_t row = (size_t)atlas_wrap(y - shift[0], ny) * step * w;
        for (int x = 0; x < nx; x++) {
            size_t at = row + atlas_wrap(x - shift[1], nx) * step;
            ink[(size_t)y * nx + x] = (packed[at / 8] >> (7 - at % 8)) & 1;
        }
    }
    return ink;
}

typedef struct {
    int y0, y1, x0, x1;
    size_t count;
} AtlasRegion;

static AtlasRegion atlas_component(
    const u8 *ink, u32 *labels, u32 *queue, u32 label, size_t at, int w, int h) {
    AtlasRegion box = {.y0 = h, .x0 = w, .count = 1};
    labels[at] = label;
    queue[0] = at;
    for (size_t j = 0; j < box.count; j++) {
        size_t i = queue[j];
        int y = i / w, x = i % w;
        box.y0 = y < box.y0 ? y : box.y0;
        box.y1 = y >= box.y1 ? y + 1 : box.y1;
        box.x0 = x < box.x0 ? x : box.x0;
        box.x1 = x >= box.x1 ? x + 1 : box.x1;
        for (int dy = -1; dy <= 1; dy++) {
            int py = y + dy;
            if (py < 0 || py >= h) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int px = x + dx;
                if (px < 0 || px >= w) {
                    continue;
                }
                size_t dest = (size_t)py * w + px;
                if (ink[dest] && !labels[dest]) {
                    labels[dest] = label;
                    queue[box.count++] = dest;
                }
            }
        }
    }
    return box;
}

static double atlas_lateral(json_object *row) {
    json_object *side = json_object_object_get(row, "lateral_um");
    return side ? json_object_get_double(side) : site_value(row, "voxel_um");
}

static void atlas_bodies(Arena *a, const char *work) {
    const char *cases = fmt(a, "%s/datasets/cases", LIBRARY);
    FILE *manifest = fopen(fmt(a, "%s/manifest.jsonl", cases), "r");
    need(manifest != NULL, "no labelled cases in %s", cases);
    const char *out = fmt(a, "%s/bodies", work);
    mkdirs(out);
    size_t then = a->used, capacity = 0;
    char *line = NULL;
    while (getline(&line, &capacity, manifest) >= 0) {
        a->used = then;
        json_object *row = json_tokener_parse(line);
        const char *id = site_string(row, "case_id");
        need(site_name(id), "invalid labelled case in %s/manifest.jsonl", cases);
        char *path = fmt(a, "%s/%s.json", out, id);
        if (exists(path)) {
            json_object_put(row);
            continue;
        }
        double start = now();
        double um = atlas_lateral(row);
        need(um > 0 && isfinite(um) && 19 / um <= INT_MAX, "%s has invalid voxel size", id);
        int step = fmax(1, nearbyint(19 / um)), h, w;
        const char *folder = fmt(a, "%s/%s", cases, id);
        const u8 *ink = atlas_mask(a, folder, "ink", row, um, step, &h, &w);
        double mm = um * step / 1000;
        size_t size = (size_t)h * w;
        need(size < UINT32_MAX, "labelled mask is too large in %s", folder);
        u32 *labels = pushz(a, size * sizeof(*labels)), *queue = push(a, size * sizeof(*queue));
        u32 label = 0;
        json_object *regions = json_object_new_array();
        for (size_t at = 0; at < size; at++) {
            if (!ink[at] || labels[at]) {
                continue;
            }
            AtlasRegion box = atlas_component(ink, labels, queue, ++label, at, w, h);
            double area = box.count * (mm * mm);
            if (area >= .08 && area <= 30) {
                json_object *body = json_object_new_object();
                json_object_object_add(body, "id", json_object_new_int64(label));
                json_object_object_add(body, "y0", json_object_new_int(box.y0));
                json_object_object_add(body, "y1", json_object_new_int(box.y1));
                json_object_object_add(body, "x0", json_object_new_int(box.x0));
                json_object_object_add(body, "x1", json_object_new_int(box.x1));
                json_object_object_add(body, "mm2", jnum(decimal_round(area, 3)));
                json_object_object_add(
                    body, "h_mm", jnum(decimal_round((box.y1 - box.y0) * mm, 2)));
                json_object_object_add(
                    body, "w_mm", jnum(decimal_round((box.x1 - box.x0) * mm, 2)));
                json_object_array_add(regions, body);
            }
        }
        Array matrix = {.descr = "<u4",
            .ndim = 2,
            .shape = {h, w},
            .data = labels,
            .size = size * sizeof(*labels)};
        char *target = fmt(a, "%s/%s.npz", out, id);
        char *part = fmt(a, "%s.%ld.part", target, (long)getpid());
        int error;
        Bytes member = npy_bytes(a, matrix);
        zip_t *zip = zip_open(part, ZIP_CREATE | ZIP_TRUNCATE, &error);
        need(zip != NULL, "could not write %s", target);
        zip_source_t *contents = zip_source_buffer(zip, member.data, member.size, 0);
        zip_int64_t index = contents ? zip_file_add(zip, "labels.npy", contents, 0) : -1;
        need(index >= 0 && !zip_set_file_compression(zip, index, ZIP_CM_DEFLATE, 6) &&
                !zip_close(zip) && !rename(part, target),
            "could not compress %s", target);
        json_object *report = json_object_new_object();
        json_object_object_add(report, "step", json_object_new_int(step));
        json_object_object_add(report, "rows", regions);
        json_write(path, report);
        printf("%s %zu %.1f\n", id, site_items(regions), now() - start);
        json_object_put(report);
        json_object_put(row);
    }
    free(line);
    need(!ferror(manifest), "could not read labelled cases in %s", cases);
    fclose(manifest);
}

typedef struct {
    size_t count;
    struct jpeg_decompress_struct codec;
    u8 *pixels;
    int w, h;
} AtlasJpeg;

static int atlas_decode(Arena *a, AtlasJpeg *result, Bytes data) {
    struct jpeg_decompress_struct *codec = &result->codec;
    SiteJpegError error;
    codec->err = jpeg_std_error(&error.manager);
    error.manager.error_exit = site_jpeg_error;
    if (setjmp(error.jump)) {
        return 0;
    }
    jpeg_create_decompress(codec);
    jpeg_mem_src(codec, data.data, data.size);
    if (jpeg_read_header(codec, TRUE) != JPEG_HEADER_OK || codec->image_width > INT_MAX ||
        codec->image_height > INT_MAX) {
        return 0;
    }
    codec->out_color_space = codec->jpeg_color_space == JCS_GRAYSCALE ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_start_decompress(codec);
    result->w = codec->output_width;
    result->h = codec->output_height;
    result->count = (size_t)result->w * result->h;
    result->pixels = push(a, result->count);
    u8 *rgb = codec->output_components == 3 ? push(a, (size_t)result->w * 3) : NULL;
    while (codec->output_scanline < codec->output_height) {
        unsigned y = codec->output_scanline;
        JSAMPROW grey = result->pixels + (size_t)y * result->w, row = rgb ? rgb : grey;
        if (jpeg_read_scanlines(codec, &row, 1) != 1) {
            return 0;
        }
        if (rgb) {
            for (int x = 0; x < result->w; x++) {
                grey[x] =
                    (19595 * rgb[3 * x] + 38470 * rgb[3 * x + 1] + 7471 * rgb[3 * x + 2] + 32768) >>
                    16;
            }
        }
    }
    return jpeg_finish_decompress(codec);
}

static AtlasJpeg atlas_jpeg(Arena *a, const char *path) {
    AtlasJpeg decoded = {0};
    int ok = atlas_decode(a, &decoded, read_file(a, path));
    decoded.codec.err = NULL;
    jpeg_destroy_decompress(&decoded.codec);
    need(ok, "cannot read JPEG %s", path);
    return decoded;
}

static u8 *atlas_thumbnail(Arena *a, AtlasJpeg input, int *w, int *h) {
    int limit = 2400;
    *w = input.w;
    *h = input.h;
    if (*w <= limit && *h <= limit) {
        return input.pixels;
    }
    double aspect = (double)input.w / input.h;
    if (input.w <= input.h) {
        double width = limit * aspect;
        double first = floor(width), last = ceil(width);
        *w = fmax(1, fabs(aspect - first / limit) <= fabs(aspect - last / limit) ? first : last);
        *h = limit;
    } else {
        double height = limit / aspect, first = floor(height), last = ceil(height);
        double a = first == 0 ? 0 : fabs(aspect - limit / first);
        double b = fabs(aspect - limit / last);
        *h = fmax(1, a <= b ? first : last);
        *w = limit;
    }
    int dx = (double)input.w / *w / 2, dy = (double)input.h / *h / 2;
    dx = dx > 1 ? dx : 1;
    dy = dy > 1 ? dy : 1;
    int nx = (input.w + dx - 1) / dx, ny = (input.h + dy - 1) / dy;
    const u8 *source = input.pixels;
    if (dx > 1 || dy > 1) {
        u8 *reduced = push(a, (size_t)nx * ny);
        for (int y = 0; y < ny; y++) {
            int bottom = (y + 1) * dy < input.h ? (y + 1) * dy : input.h;
            for (int x = 0; x < nx; x++) {
                int right = (x + 1) * dx < input.w ? (x + 1) * dx : input.w;
                int samples = (bottom - y * dy) * (right - x * dx);
                uint64_t sum = samples / 2;
                for (int row = y * dy; row < bottom; row++) {
                    for (int col = x * dx; col < right; col++) {
                        sum += input.pixels[(size_t)row * input.w + col];
                    }
                }
                reduced[(size_t)y * nx + x] = sum / samples;
            }
        }
        source = reduced;
    }
    u8 *output = push(a, (size_t)*w * *h);
    need(hx_site_resize_cubic(
             source, nx, ny, (double)input.w / dx, (double)input.h / dy, output, *w, *h),
        "cannot reduce atlas thumbnail");
    return output;
}

typedef struct {
    const char *file, *sample, *segment, *leaf;
} AtlasMap;

static AtlasMap atlas_map(Arena *a, const char *root, const char *path) {
    char *relative = fmt(a, "%s", path + strlen(root) + 1), *p = relative;
    char *sample = strsep(&p, "/");
    strsep(&p, "/");
    char *segment = strsep(&p, "/");
    const char *leaf = strrchr(path, '/');
    leaf = leaf ? leaf + 1 : path;
    size_t stem = strlen(leaf) - 4, bytes = stem > 46 ? 46 : stem;
    const char *file =
        fmt(a, "%s-%.*s-%.*s.jpg", sample, 32, segment, (int)bytes, leaf + stem - bytes);
    return (AtlasMap){file, sample, segment, leaf};
}

#include <regex.h>

static const char *atlas_capture(Arena *a, const regex_t *pattern, const char *text) {
    regmatch_t found;
    return !regexec(pattern, text, 1, &found, 0)
        ? fmt(a, "%.*s", (int)(found.rm_eo - found.rm_so), text + found.rm_so)
        : "";
}

static const char *atlas_recipe(Arena *a, const regex_t *prefix, const char *text) {
    regmatch_t offset;
    const char *begin = text;
    while (!regexec(prefix, begin, 1, &offset, 0)) {
        const char *recipe = begin + offset.rm_eo;
        begin = begin + offset.rm_so + 1;
        if (*recipe < 'a' || *recipe > 'z') {
            continue;
        }
        for (const char *end = strchr(recipe, '-'); end; end = strchr(end + 1, '-')) {
            if (!strncmp(end + 1, "tile", 4) || !strncmp(end + 1, "alpha", 5) ||
                (end[1] == 'l' && isdigit((unsigned char)end[2]))) {
                return fmt(a, "%.*s", (int)(end - recipe), recipe);
            }
        }
    }
    return "";
}

static void atlas_segment_maps(Arena *a, const char *work) {
    const char *root = fmt(a, "%s/published/open_data", LIBRARY);
    const char *folder = fmt(a, "%s/website/atlas/segments", OUTPUTS);
    const char *sizes = fmt(a, "%s/segment-sizes", work);
    mkdirs(folder);
    mkdirs(sizes);
    regex_t voxel, down, scan, recipe;
    need(!regcomp(&voxel, "[0-9]+\\.[0-9]+um", REG_EXTENDED) &&
            !regcomp(&down, "-ds[0-9]+", REG_EXTENDED) &&
            !regcomp(&scan, "[0-9]+\\.[0-9]+um[^-]*(-[0-9.]+m)?-[0-9]+keV", REG_EXTENDED) &&
            !regcomp(&recipe, "[0-9]{14}-(L[0-9]-)?([0-9]{14}-)?", REG_EXTENDED),
        "cannot initialize segment name parser");
    json_object *rows = json_object_new_array();
    for (int kind = 0; kind < 2; kind++) {
        glob_t found = {0};
        int result = glob(fmt(a, "%s/*/segments/*/%s/downsampled/*.jpg", root,
                              kind ? "surface-renders" : "ink-detection"),
            0, NULL, &found);
        need(!result || result == GLOB_NOMATCH, "cannot list published segment maps");
        size_t mark = a->used;
        for (size_t i = 0; i < found.gl_pathc; i++) {
            a->used = mark;
            const char *path = found.gl_pathv[i];
            AtlasMap map = atlas_map(a, root, path);
            char *target = fmt(a, "%s/%s", folder, map.file);
            char *facts = fmt(a, "%s/%s.json", sizes, map.file);
            int w = 0, h = 0, nx = 0, ny = 0;
            double bright;
            if (!exists(target)) {
                AtlasJpeg image = atlas_jpeg(a, path);
                w = image.w;
                h = image.h;
                u8 *small = atlas_thumbnail(a, image, &nx, &ny);
                size_t total = (size_t)nx * ny, lit = 0;
                for (size_t j = 0; j < total; j++) {
                    lit += small[j] > 128;
                }
                bright = decimal_round((double)lit / total, 4);
                need(hx_site_save_jpeg_channels(target, small, nx, ny, 74, 1), "cannot write %s",
                    target);
                char *meta = fmt(a, "{\"size\":[%d,%d],\"small\":[%d,%d],\"bright\":%.4f}", w, h,
                    nx, ny, bright);
                need(hx_write_sheet_file(facts, meta, strlen(meta)), "cannot write %s", facts);
            } else {
                json_object *info = json_read(facts);
                need(info != NULL, "no saved dimensions in %s", facts);
                json_object *dims = jget(info, "size"), *small = jget(info, "small");
                w = json_object_get_int(json_object_array_get_idx(dims, 0));
                h = json_object_get_int(json_object_array_get_idx(dims, 1));
                nx = json_object_get_int(json_object_array_get_idx(small, 0));
                ny = json_object_get_int(json_object_array_get_idx(small, 1));
                bright = jf64(info, "bright");
                json_object_put(info);
            }
            const char *pitch = atlas_capture(a, &voxel, map.leaf);
            const char *factor = atlas_capture(a, &down, map.leaf);
            double um = *pitch && *factor ? strtod(pitch, NULL) * strtol(factor + 3, NULL, 10) : 0;
            json_object *row = json_object_new_object();
            json_object_object_add(row, "sample", json_object_new_string(map.sample));
            json_object_object_add(row, "segment", json_object_new_string(map.segment));
            json_object_object_add(row, "kind", json_object_new_string(kind ? "render" : "ink"));
            json_object_object_add(
                row, "file", json_object_new_string(fmt(a, "segments/%s", map.file)));
            json_object_object_add(
                row, "scan", json_object_new_string(atlas_capture(a, &scan, map.leaf)));
            json_object_object_add(
                row, "recipe", json_object_new_string(atlas_recipe(a, &recipe, map.leaf)));
            json_object_object_add(
                row, "mm_w", um ? json_object_new_int64(nearbyint(um * w / 1000)) : NULL);
            json_object_object_add(
                row, "mm_h", um ? json_object_new_int64(nearbyint(um * h / 1000)) : NULL);
            json_object_object_add(
                row, "mm_per_px", um ? jnum(decimal_round(um * w / nx / 1000, 4)) : NULL);
            json_object_object_add(row, "w", json_object_new_int(nx));
            json_object_object_add(row, "h", json_object_new_int(ny));
            json_object_object_add(row, "bright", jnum(bright));
            json_object_array_add(rows, row);
            printf("%zu %.70s\n", site_items(rows), map.file);
        }
        a->used = mark;
        globfree(&found);
    }
    regfree(&recipe);
    regfree(&scan);
    regfree(&down);
    regfree(&voxel);
    json_write(fmt(a, "%s/segments.json", work), rows);
    json_object_put(rows);
}

static double atlas_row_strength(Arena *a, AtlasJpeg image, double mm) {
    size_t mark = a->used;
    int longest = image.w > image.h ? image.w : image.h;
    double *profile = pushz(a, (size_t)(image.w + image.h) * sizeof(*profile));
    double *smooth = push(a, (size_t)longest * sizeof(*smooth));
    double *power = push(a, (size_t)longest * sizeof(*power));
    double sigma = 14 / mm, radius_d = floor(4 * sigma + .5);
    need(radius_d >= 0 && radius_d < INT_MAX / 2, "invalid scale for atlas line strength");
    int radius = radius_d;
    double *weights = push(a, (size_t)(radius + 1) * sizeof(*weights));
    double sum = 1;
    weights[0] = 1;
    for (int i = 1; i <= radius; i++) {
        weights[i] = exp(-.5 / (sigma * sigma) * (double)i * i);
        sum += 2 * weights[i];
    }
    for (int i = 0; i <= radius; i++) {
        weights[i] /= sum;
    }
    for (int y = 0; y < image.h; y++) {
        for (int x = 0; x < image.w; x++) {
            int ink = image.pixels[(size_t)y * image.w + x] > 128;
            profile[x] += ink;
            profile[image.w + y] += ink;
        }
    }
    double best = 0;
    for (int axis = 0; axis < 2; axis++) {
        int count = axis ? image.h : image.w, sites = axis ? image.w : image.h;
        double *sample = profile + (axis ? image.w : 0);
        for (int i = 0; i < count; i++) {
            sample[i] /= sites;
        }
        int first = fmax(1, ceil(count * mm / 14));
        int nyquist = count / 2, last = fmin(nyquist, floor(count * mm / 3.5));
        if (last - first + 1 < 4) {
            continue;
        }
        for (int i = 0; i < count; i++) {
            double smoothed = sample[i] * weights[0];
            for (int d = radius; d > 0; d--) {
                smoothed += (sample[reflected(i - d, count)] + sample[reflected(i + d, count)]) *
                    weights[d];
            }
            smooth[i] = (sample[i] - smoothed) * (.5 - .5 * cos(2 * M_PI * i / (count - 1)));
        }
        double peak = 0;
        size_t used = 0;
        for (int k = first; k <= last; k++) {
            double re = 0, im = 0;
            for (int i = 0; i < count; i++) {
                double angle = 2 * M_PI * ((int64_t)k * i % count) / count;
                re += smooth[i] * cos(angle);
                im += smooth[i] * sin(angle);
            }
            power[used] = re * re + im * im;
            peak = fmax(peak, power[used++]);
        }
        best = fmax(best, peak / fmax(1e-12, atlas_median(power, used)));
    }
    a->used = mark;
    return best;
}

static int atlas_compare_score(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int order = atlas_compare_double(site_value(b->row, "rows"), site_value(a->row, "rows"));
    return order ? order : (a->order > b->order) - (a->order < b->order);
}

static int atlas_score_report_order(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int score = atlas_compare_double(site_value(b->row, "rows"), site_value(a->row, "rows"));
    if (score)
        return score;
    int scroll = strcmp(atlas_text(b->row, "sample"), atlas_text(a->row, "sample"));
    const char *segment = atlas_text(a->row, "segment");
    size_t end = strnlen(segment, 20);
    return scroll ? scroll : strncmp(atlas_text(b->row, "segment"), segment, end + (end < 20));
}

static void atlas_segment_scores(Arena *a, const char *work) {
    const char *root = fmt(a, "%s/published/open_data", LIBRARY);
    const char *atlas = fmt(a, "%s/website/atlas", OUTPUTS);
    const char *file = fmt(a, "%s/segments.json", work);
    json_object *rows = json_read(file);
    need(json_object_is_type(rows, json_type_array), "no segment maps at %s", file);
    size_t count = site_items(rows), used = 0;
    AtlasRow *sorted = push(a, count * sizeof(*sorted));
    size_t mark = a->used;
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        if (!strcmp(atlas_text(row, "kind"), "ink")) {
            double mm = site_value(row, "mm_per_px");
            if (mm > 0 && isfinite(mm)) {
                const char *path = fmt(a, "%s/%s", atlas, atlas_text(row, "file"));
                double strength = atlas_row_strength(a, atlas_jpeg(a, path), mm);
                json_object_object_add(row, "rows", jnum(decimal_round(strength, 1)));
                a->used = mark;
            }
            sorted[used++] = (AtlasRow){row, i};
        }
    }
    qsort(sorted, used, sizeof(*sorted), atlas_compare_score);
    glob_t sources = {0};
    int listed =
        glob(fmt(a, "%s/*/segments/*/ink-detection/downsampled/*.jpg", root), 0, NULL, &sources);
    need(!listed || listed == GLOB_NOMATCH, "could not list published maps");
    json_object *by_name = json_object_new_object();
    mark = a->used;
    for (size_t i = 0; i < sources.gl_pathc; i++) {
        AtlasMap map = atlas_map(a, root, sources.gl_pathv[i]);
        json_object_object_add(by_name, map.file, json_object_new_string(sources.gl_pathv[i]));
        a->used = mark;
    }
    for (size_t i = 0; i < used && i < 24; i++) {
        json_object *row = sorted[i].row;
        const char *path = atlas_text(row, "file"), *name = strrchr(path, '/');
        name = name ? name + 1 : path;
        const char *local = fmt(a, "segments/full-%s", name);
        const char *target = fmt(a, "%s/%s", atlas, local);
        json_object_object_add(row, "full", json_object_new_string(local));
        if (!exists(target)) {
            const char *original = site_string(by_name, name);
            need(original != NULL, "no published map for %s", name);
            Bytes data = read_file(a, original);
            need(hx_write_sheet_file(target, data.data, data.size), "could not write %s", target);
        }
        a->used = mark;
    }
    for (size_t i = 0; i < count; i++) {
        sorted[i] = (AtlasRow){json_object_array_get_idx(rows, i), i};
    }
    qsort(sorted, count, sizeof(*sorted), atlas_score_report_order);
    putchar('[');
    for (size_t i = 0; i < count && i < 8; i++) {
        json_object *row = sorted[i].row, *score = json_object_object_get(row, "rows");
        printf("%s(%s, '%s', '%.20s')", i ? ", " : "",
            score ? json_object_to_json_string_ext(score, JSON_C_TO_STRING_PLAIN) : "0",
            atlas_text(row, "sample"), atlas_text(row, "segment"));
    }
    puts("]");
    json_write(file, rows);
    json_object_put(by_name);
    json_object_put(rows);
    globfree(&sources);
}
static int atlas_render_order(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int finest_a = strstr(atlas_text(a->row, "case_id"), "1.129um") != NULL;
    int finest_b = strstr(atlas_text(b->row, "case_id"), "1.129um") != NULL;
    if (finest_a != finest_b) {
        return finest_a - finest_b;
    }
    double a_um = atlas_lateral(a->row), b_um = atlas_lateral(b->row);
    int order = atlas_compare_double(site_value(a->row, "supervised_pixels") * (a_um * a_um),
        site_value(b->row, "supervised_pixels") * (b_um * b_um));
    return order ? order : (a->order > b->order) - (a->order < b->order);
}

static json_object *atlas_records(Arena *a) {
    char *path = fmt(a, "%s/datasets/cases/manifest.jsonl", LIBRARY);
    FILE *manifest = fopen(path, "r");
    need(manifest != NULL, "no labelled case manifest at %s", path);
    char *line = NULL;
    size_t capacity = 0;
    json_object *rows = json_object_new_array();
    while (getline(&line, &capacity, manifest) >= 0) {
        json_object *row = json_tokener_parse(line);
        need(site_name(site_string(row, "case_id")), "invalid labelled case in %s", path);
        json_object_array_add(rows, row);
    }
    free(line);
    need(!ferror(manifest), "could not read %s", path);
    fclose(manifest);
    return rows;
}

static void atlas_expand(u8 *mask, u32 *queue, int w, int h, int rounds) {
    size_t count = 0;
    for (size_t i = 0, size = (size_t)w * h; i < size; i++) {
        if (mask[i]) {
            queue[count++] = i;
        }
    }
    for (size_t start = 0; rounds > 0 && start < count; rounds--) {
        size_t end = count;
        for (size_t j = start; j < end; j++) {
            size_t i = queue[j];
            int y = i / w, x = i % w;
            size_t near[] = {
                x ? i - 1 : i, x + 1 < w ? i + 1 : i, y ? i - w : i, y + 1 < h ? i + w : i};
            for (int k = 0; k < 4; k++) {
                size_t at = near[k];
                if (!mask[at]) {
                    mask[at] = 1;
                    queue[count++] = at;
                }
            }
        }
        start = end;
    }
}

typedef struct {
    const Bricks *ct;
    u8 *pixels;
    int y0, x0, width, reach;
} AtlasRender;

static void atlas_render_row(void *opaque, i64 row) {
    AtlasRender *view = opaque;
    const Bricks *ct = view->ct;
    int y = view->y0 + row, w = view->width, x0 = view->x0;
    int z0 = ct->shape[0] / 2 - view->reach, z1 = ct->shape[0] / 2 + view->reach;
    u8 *pixels = view->pixels + row * w;
    for (int z = z0; z <= z1; z++) {
        for (int x = 0; x < w; x++) {
            u8 value = (u8)voxel_at(ct, z, y, x0 + x);
            if (value > pixels[x]) {
                pixels[x] = value;
            }
        }
    }
}

static double atlas_percentile(const size_t *bins, size_t count, double fraction) {
    double place = fraction * (count - 1), part = place - floor(place);
    size_t lo = place, hi = ceil(place), total = 0;
    int first = -1;
    for (int i = 0; i < 256; i++) {
        total += bins[i];
        if (first < 0 && total > lo) {
            first = i;
        }
        if (total > hi) {
            return part < .5 ? first + part * (i - first) : i - (1 - part) * (i - first);
        }
    }
    return 255;
}

static void atlas_render_tile(Arena *a, const char *folder, const Bricks *ct, int reach, int y,
    int x, int h, int w, int local_y, int local_x, json_object *tiles) {
    size_t mark = a->used;
    u8 *pixels = pushz(a, (size_t)w * h);
    AtlasRender view = {ct, pixels, y, x, w, reach};
    parallel(h, 4, atlas_render_row, &view);
    size_t bins[256] = {0}, count = 0, visible = 0;
    for (size_t i = 0, n = (size_t)w * h; i < n; i++) {
        u8 value = pixels[i];
        if (value && visible++ % 97 == 0) {
            bins[value]++;
            count++;
        }
    }
    double low = count ? atlas_percentile(bins, count, .005) : 0;
    double high = count ? atlas_percentile(bins, count, .995) : 255;
    char *file = fmt(a, "tile-%zu.png", site_items(tiles));
    char *path = fmt(a, "%s/%s", folder, file);
    Bytes out = png_encode(a, pixels, w, h, 6);
    need(hx_write_sheet_file(path, out.data, out.size), "cannot write %s", path);
    json_object *tile = json_object_new_object();
    json_object_object_add(tile, "file", json_object_new_string(file));
    json_object_object_add(tile, "y", json_object_new_int(local_y));
    json_object_object_add(tile, "x", json_object_new_int(local_x));
    json_object_object_add(tile, "h", json_object_new_int(h));
    json_object_object_add(tile, "w", json_object_new_int(w));
    json_object_object_add(tile, "low", jnum(low));
    json_object_object_add(tile, "high", jnum(fmax(high, low + 1)));
    json_object_array_add(tiles, tile);
    a->used = mark;
}

static void atlas_renders(Arena *a, const char *work) {
    json_object *rows = atlas_records(a);
    size_t count = site_items(rows);
    AtlasRow *order = push(a, count * sizeof(*order));
    for (size_t i = 0; i < count; i++) {
        order[i] = (AtlasRow){json_object_array_get_idx(rows, i), i};
    }
    qsort(order, count, sizeof(*order), atlas_render_order);
    const char *limit = env("HERCULEX_ATLAS_SECONDS", "1e9");
    char *end;
    double limit_seconds = strtod(limit, &end);
    need(end != limit && !*end && !isnan(limit_seconds), "invalid HERCULEX_ATLAS_SECONDS");
    double began = now();
    size_t mark = a->used;
    for (size_t i = 0; i < count; i++) {
        a->used = mark;
        json_object *record = order[i].row;
        const char *id = site_string(record, "case_id");
        const char *target = fmt(a, "%s/faces/%s", work, id);
        const char *receipt = fmt(a, "%s/tiles.json", target);
        if (exists(receipt)) {
            continue;
        }
        if (now() - began > limit_seconds) {
            puts("time up");
            break;
        }
        double start = now(), um = atlas_lateral(record), voxel = site_value(record, "voxel_um");
        need(um > 0 && voxel > 0 && isfinite(um) && isfinite(voxel),
            "%s has invalid atlas voxel size", id);
        const char *case_path = fmt(a, "%s/datasets/cases/%s", LIBRARY, id);
        const char *store_path = fmt(a, "%s/layers", case_path);
        Bricks store;
        bricks_open(a, &store, store_path);
        int h, w, step = 16;
        u8 *visible = atlas_mask(a, case_path, "supervised", record, um, step, &h, &w);
        size_t cells = (size_t)w * h;
        need(cells < UINT32_MAX, "%s supervised mask is too large", id);
        u32 *labels = pushz(a, cells * sizeof(*labels)), *queue = push(a, cells * sizeof(*queue));
        atlas_expand(visible, queue, w, h, fmax(1, (int)(3000 / (um * step))));
        json_object *box = jget(record, "box"), *tiles = json_object_new_array();
        int y0 = json_object_get_int(json_object_array_get_idx(box, 0));
        int x0 = json_object_get_int(json_object_array_get_idx(box, 2));
        int reach = sheet_layers(voxel) / 2;
        u32 component = 0;
        mkdirs(target);
        for (size_t j = 0; j < cells; j++) {
            if (visible[j] && !labels[j]) {
                AtlasRegion region = atlas_component(visible, labels, queue, ++component, j, w, h);
                for (int y = region.y0 * step; y < region.y1 * step; y += 4096) {
                    int bottom = (y + 4096 < region.y1 * step ? y + 4096 : region.y1 * step);
                    int x = region.x0 * step, columns = (region.x1 - region.x0) * step;
                    atlas_render_tile(
                        a, target, &store, reach, y + y0, x + x0, bottom - y, columns, y, x, tiles);
                }
            }
        }
        json_write(receipt, tiles);
        printf("%s %zu tiles %.0f s\n", id, site_items(tiles), now() - start);
        json_object_put(tiles);
        if (store.data) {
            munmap((void *)store.data, store.mapped);
        }
        pthread_mutex_destroy(&store.lock);
    }
    json_object_put(rows);
}
typedef struct {
    double x, y;
} AtlasPoint;

typedef struct {
    int kind, count;
    AtlasPoint points[7];
} AtlasStroke;

typedef struct {
    int count;
    AtlasStroke strokes[3];
} AtlasModel;

typedef struct {
    const char *name, *capital, *small, *note;
    double width;
    int holes[2], models;
    AtlasModel variants[2];
} AtlasLetter;

static const AtlasLetter atlas_letters[] = {
    {"Alpha", "Α", "α", "pointed, the cross-bar often slanted or looped", 0.9, {0, 1}, 2,
        {
            {2,
                {
                    {0, 3, {{0, 1}, {0.5, 0}, {1, 1}}},
                    {0, 2, {{0.22, 0.62}, {0.78, 0.62}}},
                }},
            {2,
                {
                    {0, 3, {{0, 1}, {0.5, 0}, {0.9, 1}}},
                    {0, 4, {{0.1, 0.75}, {0.45, 0.45}, {0.62, 0.8}, {0.05, 0.8}}},
                }},
        }},
    {"Beta", "Β", "β", "", 0.9, {1, 2}, 1,
        {
            {3,
                {
                    {0, 2, {{0.1, 0}, {0.1, 1}}},
                    {1, 4, {{0.1, 0.02}, {0.65, 0.1}, {0.6, 0.42}, {0.1, 0.48}}},
                    {1, 4, {{0.1, 0.48}, {0.8, 0.58}, {0.75, 0.95}, {0.1, 1}}},
                }},
        }},
    {"Gamma", "Γ", "γ", "", 0.7, {0, 0}, 1,
        {
            {1,
                {
                    {0, 3, {{0.2, 1}, {0.2, 0}, {0.85, 0.02}}},
                }},
        }},
    {"Delta", "Δ", "δ", "", 0.9, {1, 1}, 1,
        {
            {1,
                {
                    {0, 4, {{0.5, 0}, {0, 1}, {1, 1}, {0.5, 0}}},
                }},
        }},
    {"Epsilon", "Ε", "ε", "lunate: a rounded Є, not the angular Ε", 0.8, {0, 0}, 1,
        {
            {2,
                {
                    {1, 5, {{0.85, 0.1}, {0.35, 0.03}, {0.05, 0.5}, {0.35, 0.97}, {0.85, 0.9}}},
                    {0, 2, {{0.05, 0.5}, {0.7, 0.5}}},
                }},
        }},
    {"Zeta", "Ζ", "ζ", "", 0.9, {0, 0}, 1,
        {
            {1,
                {
                    {0, 4, {{0.1, 0}, {0.9, 0}, {0.1, 1}, {0.9, 1}}},
                }},
        }},
    {"Eta", "Η", "η", "an H", 0.9, {0, 0}, 1,
        {
            {3,
                {
                    {0, 2, {{0.1, 0}, {0.1, 1}}},
                    {0, 2, {{0.9, 0}, {0.9, 1}}},
                    {0, 2, {{0.1, 0.5}, {0.9, 0.5}}},
                }},
        }},
    {"Theta", "Θ", "θ", "an oval with a bar, often touching both sides", 0.85, {2, 1}, 1,
        {
            {2,
                {
                    {2, 2, {{0.5, 0.5}, {0.45, 0.5}}},
                    {0, 2, {{0.1, 0.5}, {0.9, 0.5}}},
                }},
        }},
    {"Iota", "Ι", "ι", "a plain stroke; the commonest confusion", 0.12, {0, 0}, 1,
        {
            {1,
                {
                    {0, 2, {{0.5, 0}, {0.5, 1}}},
                }},
        }},
    {"Kappa", "Κ", "κ", "", 0.9, {0, 0}, 1,
        {
            {2,
                {
                    {0, 2, {{0.15, 0}, {0.15, 1}}},
                    {0, 3, {{0.85, 0.05}, {0.15, 0.55}, {0.9, 1}}},
                }},
        }},
    {"Lambda", "Λ", "λ", "", 0.9, {0, 0}, 2,
        {
            {1,
                {
                    {0, 3, {{0, 1}, {0.55, 0}, {1, 1}}},
                }},
            {2,
                {
                    {0, 2, {{0.05, 1}, {0.42, 0.42}}},
                    {0, 2, {{0.2, 0}, {1, 1}}},
                }},
        }},
    {"Mu", "Μ", "μ", "splayed legs, often curved", 1.15, {0, 0}, 2,
        {
            {1,
                {
                    {0, 5, {{0, 1}, {0.15, 0}, {0.5, 0.75}, {0.85, 0}, {1, 1}}},
                }},
            {1,
                {
                    {1, 7,
                        {{0, 1}, {0.1, 0.2}, {0.25, 0.1}, {0.5, 0.7}, {0.75, 0.1}, {0.9, 0.2},
                            {1, 1}}},
                }},
        }},
    {"Nu", "Ν", "ν", "an N, the diagonal sometimes shallow, easily taken for Η", 0.9, {0, 0}, 1,
        {
            {1,
                {
                    {0, 4, {{0.1, 1}, {0.1, 0}, {0.9, 1}, {0.9, 0}}},
                }},
        }},
    {"Xi", "Ξ", "ξ", "three bars, sometimes joined", 0.9, {0, 0}, 1,
        {
            {3,
                {
                    {0, 2, {{0.1, 0}, {0.9, 0}}},
                    {0, 2, {{0.2, 0.5}, {0.8, 0.5}}},
                    {0, 2, {{0.1, 1}, {0.9, 1}}},
                }},
        }},
    {"Omicron", "Ο", "ο", "", 1.0, {1, 1}, 1,
        {
            {1,
                {
                    {2, 2, {{0.5, 0.5}, {0.5, 0.5}}},
                }},
        }},
    {"Pi", "Π", "π", "top bar overhangs both legs", 0.9, {0, 0}, 1,
        {
            {3,
                {
                    {0, 2, {{0, 0.03}, {1, 0.03}}},
                    {0, 2, {{0.2, 0.03}, {0.2, 1}}},
                    {0, 2, {{0.8, 0.03}, {0.8, 1}}},
                }},
        }},
    {"Rho", "Ρ", "ρ", "tall stem running below the line, small bowl", 0.55, {1, 0}, 1,
        {
            {2,
                {
                    {0, 2, {{0.25, 0}, {0.25, 1.35}}},
                    {1, 4, {{0.25, 0.02}, {0.75, 0.05}, {0.8, 0.4}, {0.25, 0.55}}},
                }},
        }},
    {"Sigma", "Σ", "σ", "lunate: a C, never the Σ of mathematics", 0.8, {0, 0}, 2,
        {
            {1,
                {
                    {1, 5, {{0.85, 0.1}, {0.35, 0.03}, {0.05, 0.5}, {0.35, 0.97}, {0.85, 0.9}}},
                }},
            {1,
                {
                    {0, 5, {{0.9, 0}, {0.1, 0}, {0.5, 0.5}, {0.1, 1}, {0.9, 1}}},
                }},
        }},
    {"Tau", "Τ", "τ", "", 0.85, {0, 0}, 1,
        {
            {2,
                {
                    {0, 2, {{0, 0.02}, {1, 0.02}}},
                    {0, 2, {{0.5, 0.02}, {0.5, 1}}},
                }},
        }},
    {"Upsilon", "Υ", "υ", "a Y, the stem on the line", 0.95, {0, 0}, 2,
        {
            {2,
                {
                    {0, 3, {{0, 0}, {0.5, 0.5}, {1, 0}}},
                    {0, 2, {{0.5, 0.5}, {0.5, 1}}},
                }},
            {2,
                {
                    {1, 5, {{0, 0}, {0.25, 0.4}, {0.5, 0.55}, {0.75, 0.4}, {1, 0}}},
                    {0, 2, {{0.5, 0.55}, {0.5, 1}}},
                }},
        }},
    {"Phi", "Φ", "φ", "small oval, stem far above and below the line", 0.85, {2, 1}, 1,
        {
            {2,
                {
                    {2, 2, {{0.5, 0.5}, {0.45, 0.3}}},
                    {0, 2, {{0.5, -0.1}, {0.5, 1.1}}},
                }},
        }},
    {"Chi", "Χ", "χ", "", 0.9, {0, 0}, 1,
        {
            {2,
                {
                    {0, 2, {{0, 0}, {1, 1}}},
                    {0, 2, {{1, 0}, {0, 1}}},
                }},
        }},
    {"Psi", "Ψ", "ψ", "stem above and below the line", 1.0, {0, 0}, 1,
        {
            {2,
                {
                    {0, 2, {{0.5, -0.05}, {0.5, 1.05}}},
                    {1, 5, {{0.02, 0.1}, {0.15, 0.5}, {0.5, 0.65}, {0.85, 0.5}, {0.98, 0.1}}},
                }},
        }},
    {"Omega", "Ω", "ω",
        "rounded, ω-shaped and wide; the capital Ω is a later "
        "form",
        1.2, {0, 0}, 2,
        {
            {1,
                {
                    {1, 7,
                        {{0.15, 0.05}, {0, 0.6}, {0.22, 1}, {0.5, 0.55}, {0.78, 1}, {1, 0.6},
                            {0.85, 0.05}}},
                }},
            {1,
                {
                    {1, 7,
                        {{0.1, 1}, {0.3, 0.9}, {0.08, 0.5}, {0.5, 0}, {0.92, 0.5}, {0.7, 0.9},
                            {0.9, 1}}},
                }},
        }},
};

enum { ATLAS_GRID = 64, ATLAS_PIXELS = 64 * 64, ATLAS_LETTERS = 24 };

static int atlas_energy(json_object *record) {
    const char *scan = atlas_text(record, "scan"), *end = strstr(scan, "keV"), *start = end;
    while (start && start > scan && isdigit((unsigned char)start[-1])) {
        start--;
    }
    long energy = start && start < end ? strtol(start, NULL, 10) : 0;
    return energy <= INT_MAX ? (int)energy : 0;
}

static const char *atlas_published(Arena *a, json_object *record) {
    const char *scan = atlas_text(record, "scan"), *volume = strstr(scan, "volume-");
    if (!volume) {
        return NULL;
    }
    volume += 7;
    size_t length = strspn(volume, "0123456789");
    if (!length) {
        return NULL;
    }
    const char *match = fmt(a, "%.*s", (int)length, volume);
    const char *pat =
        fmt(a, "%s/published/open_data/%s/segments/%s*/ink-detection/downsampled/*.jpg", LIBRARY,
            atlas_text(record, "sample"), atlas_text(record, "segment"));
    glob_t result = {0};
    int found = glob(pat, 0, NULL, &result);
    need(!found || found == GLOB_NOMATCH, "cannot list published maps for %s", scan);
    const char *best = NULL, *name = NULL;
    for (size_t i = 0; i < result.gl_pathc; i++) {
        const char *path = result.gl_pathv[i], *base = strrchr(path, '/') + 1;
        if (!strstr(base, match) || strstr(base, "alpha")) {
            continue;
        }
        int rank = strstr(base, "new_canon") != NULL;
        int prior = name && strstr(name, "new_canon");
        if (!name || rank > prior || (rank == prior && strcmp(base, name) < 0)) {
            best = path;
            name = base;
        }
    }
    const char *path = best ? fmt(a, "%s", best) : NULL;
    globfree(&result);
    return path;
}

static int atlas_stroke_points(const AtlasStroke *stroke, AtlasPoint *p) {
    const AtlasPoint *q = stroke->points;
    if (!stroke->kind) {
        memcpy(p, q, stroke->count * sizeof(*p));
        return stroke->count;
    }
    if (stroke->kind == 2) {
        for (int i = 0; i < 40; i++) {
            double at = i == 39 ? 2 * M_PI : i * (2 * M_PI / 39);
            p[i] = (AtlasPoint){q[0].x + q[1].x * cos(at), q[0].y + q[1].y * sin(at)};
        }
        return 40;
    }
    int count = 0;
    for (int i = 0; i + 1 < stroke->count; i++) {
        AtlasPoint a = q[i ? i - 1 : i], b = q[i], c = q[i + 1];
        AtlasPoint d = q[i + 2 < stroke->count ? i + 2 : i + 1];
        for (int j = 0; j < 12; j++) {
            double x = j * (1.0 / 12), y = x * x, z = pow(x, 3);
            p[count++] = (AtlasPoint){.5 *
                    (2 * b.x + (c.x - a.x) * x + (2 * a.x - 5 * b.x + 4 * c.x - d.x) * y +
                        (3 * b.x - a.x - 3 * c.x + d.x) * z),
                .5 *
                    (2 * b.y + (c.y - a.y) * x + (2 * a.y - 5 * b.y + 4 * c.y - d.y) * y +
                        (3 * b.y - a.y - 3 * c.y + d.y) * z)};
        }
    }
    p[count++] = q[stroke->count - 1];
    return count;
}

static void atlas_line(u8 *image, int w, int h, int x, int y, int tx, int ty, u8 value) {
    int dx = abs(tx - x), dy = -abs(ty - y), sx = x < tx ? 1 : -1, sy = y < ty ? 1 : -1;
    int error = dx + dy;
    for (;;) {
        if (x >= 0 && x < w && y >= 0 && y < h) {
            image[(size_t)y * w + x] = value;
        }
        if (x == tx && y == ty) {
            break;
        }
        int both = 2 * error;
        if (both >= dy) {
            error += dy;
            x += sx;
        }
        if (both <= dx) {
            error += dx;
            y += sy;
        }
    }
}

static void atlas_template(
    const AtlasLetter *letter, int variant, double shear, double turn, double stretch, u8 *line) {
    const AtlasModel *model = &letter->variants[variant];
    AtlasPoint points[3][73], low = {INFINITY, INFINITY}, high = {-INFINITY, -INFINITY};
    int count[3];
    double angle = turn * M_PI / 180, sn = sin(angle), cs = cos(angle);
    for (int i = 0; i < model->count; i++) {
        count[i] = atlas_stroke_points(&model->strokes[i], points[i]);
        for (int j = 0; j < count[i]; j++) {
            AtlasPoint p = points[i][j];
            double y = p.y - .5, x = (p.x - .5) * letter->width * stretch + shear * y;
            p = (AtlasPoint){x * cs - y * sn, x * sn + y * cs};
            low.x = fmin(low.x, p.x);
            low.y = fmin(low.y, p.y);
            high.x = fmax(high.x, p.x);
            high.y = fmax(high.y, p.y);
            points[i][j] = p;
        }
    }
    double scale = 56 / fmax(high.x - low.x, high.y - low.y);
    AtlasPoint middle = {.5 * (low.x + high.x), .5 * (low.y + high.y)};
    memset(line, 0, ATLAS_PIXELS);
    for (int i = 0; i < model->count; i++) {
        int x = 0, y = 0;
        for (int j = 0; j < count[i]; j++) {
            int next_x = (points[i][j].x - middle.x) * scale + 32;
            int next_y = (points[i][j].y - middle.y) * scale + 32;
            if (j) {
                atlas_line(line, 64, 64, x, y, next_x, next_y, 1);
            }
            x = next_x;
            y = next_y;
        }
    }
}

static void atlas_normal(
    Arena *a, const u32 *labels, int stride, AtlasRegion box, u32 id, u8 *result) {
    size_t mark = a->used;
    int width = box.x1 - box.x0, height = box.y1 - box.y0;
    memset(result, 0, ATLAS_PIXELS);
    if (width <= 0 || height <= 0) {
        return;
    }
    u8 *pixels = push(a, (size_t)width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            pixels[(size_t)y * width + x] =
                labels[(size_t)(box.y0 + y) * stride + box.x0 + x] == id ? 255 : 0;
        }
    }
    double scale = 56.0 / (height > width ? height : width);
    int h = fmax(1, nearbyint(height * scale)), w = fmax(1, nearbyint(width * scale));
    u8 tmp[ATLAS_PIXELS];
    need(hx_site_resize_grey(pixels, width, height, tmp, w, h, 1), "cannot resize letter mask");
    int top = (64 - h) / 2, left = (64 - w) / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            result[(y + top) * 64 + left + x] = tmp[y * w + x] > 127;
        }
    }
    a->used = mark;
}

static void atlas_thin(u8 *m) {
    static const int offsets[] = {-ATLAS_GRID, 1 - ATLAS_GRID, 1, ATLAS_GRID + 1, ATLAS_GRID,
        ATLAS_GRID - 1, -1, -ATLAS_GRID - 1};
    u8 removed[ATLAS_PIXELS];
    int changed;
    do {
        changed = 0;
        for (int phase = 0; phase < 2; phase++) {
            int edge = 2 + 4 * phase;
            memset(removed, 0, sizeof(removed));
            for (int y = 1; y < 63; y++) {
                for (int x = 1; x < 63; x++) {
                    int at = y * 64 + x;
                    if (!m[at]) {
                        continue;
                    }
                    int n[8], total = 0, turns = 0, previous = m[at + offsets[7]];
                    for (int i = 0; i < 8; i++) {
                        n[i] = m[at + offsets[i]];
                        total += n[i];
                        turns += !previous && n[i];
                        previous = n[i];
                    }
                    int keep =
                        n[edge] && n[(edge + 2) % 8] && (n[(edge + 4) % 8] || n[(edge + 6) % 8]);
                    removed[at] = total >= 2 && total <= 6 && turns == 1 && !keep;
                    changed |= removed[at];
                }
            }
            for (int i = 0; i < ATLAS_PIXELS; i++) {
                m[i] &= !removed[i];
            }
        }
    } while (changed);
}

typedef struct {
    f32 distance[ATLAS_PIXELS];
    uint16_t cells[ATLAS_PIXELS];
    int count, letter, body;
} AtlasTemplate;

static void atlas_distance(const u8 *line, f32 *dist, int clamp) {
    int fy[ATLAS_PIXELS], fx[ATLAS_PIXELS];
    nearest_good(line, 64, 64, fy, fx);
    for (int i = 0; i < ATLAS_PIXELS; i++) {
        int dy = i / 64 - (fy[i] < 0 ? -1 : fy[i]), dx = i % 64 - (fx[i] < 0 ? 0 : fx[i]);
        int square = dx * dx + dy * dy;
        dist[i] = clamp && square > 36 ? 6.f : sqrtf(square);
    }
}

static void atlas_make_template(AtlasTemplate *t, u8 *line, int letter, int body) {
    t->count = 0;
    t->letter = letter;
    t->body = body;
    for (int i = 0; i < ATLAS_PIXELS; i++) {
        if (line[i]) {
            t->cells[t->count++] = i;
        }
    }
    atlas_distance(line, t->distance, 1);
}

static void atlas_shape_template(AtlasTemplate *out, const u8 *picture, int letter, int body) {
    u8 line[ATLAS_PIXELS], spare[ATLAS_PIXELS];
    memcpy(line, picture, sizeof(line));
    grow(line, spare, 64, 64, 0, 0);
    grow(line, spare, 64, 64, 0, 1);
    atlas_thin(line);
    atlas_make_template(out, line, letter, body);
}

static int atlas_holes(const u8 *picture) {
    u8 filled[ATLAS_PIXELS];
    int queue[ATLAS_PIXELS], holes = 0;
    memcpy(filled, picture, sizeof(filled));
    fill_holes(filled, 64, 64, queue);
    for (int i = 0; i < ATLAS_PIXELS; i++) {
        filled[i] &= !picture[i];
    }
    for (int i = 0; i < ATLAS_PIXELS; i++) {
        if (!filled[i]) {
            continue;
        }
        int n = 1;
        queue[0] = i;
        filled[i] = 0;
        for (int j = 0; j < n; j++) {
            int at = queue[j];
            int near[] = {at - 1, at + 1, at - 64, at + 64};
            for (int k = 0; k < 4; k++) {
                if (filled[near[k]]) {
                    filled[near[k]] = 0;
                    queue[n++] = near[k];
                }
            }
        }
        holes += n >= 12;
    }
    return holes;
}

static AtlasRegion atlas_box(json_object *row) {
    return (AtlasRegion){
        .y0 = ji64(row, "y0"), .y1 = ji64(row, "y1"), .x0 = ji64(row, "x0"), .x1 = ji64(row, "x1")};
}

typedef struct {
    const char *id;
    json_object *row;
    u8 picture[ATLAS_PIXELS];
    int eye;
} AtlasBody;

static AtlasRow *atlas_primary(Arena *a, json_object *rows, size_t *count) {
    size_t n = site_items(rows);
    AtlasRow *kept = push(a, n * sizeof(*kept));
    json_object *keys = json_object_new_object();
    size_t mark = a->used;
    *count = 0;
    for (size_t i = 0; i < n; i++) {
        json_object *record = json_object_array_get_idx(rows, i), *previous;
        const char *id = atlas_text(record, "case_id");
        const char *key =
            fmt(a, "%s/%s", atlas_text(record, "sample"), atlas_text(record, "segment"));
        int energy = atlas_energy(record);
        int rank = (!atlas_published(a, record) ? 4 : 0) + (energy != 77 && energy != 78 ? 2 : 0) +
            (strstr(id, "1.129um") ? 1 : 0);
        if (json_object_object_get_ex(keys, key, &previous)) {
            size_t at = json_object_get_int64(previous);
            if (rank < (int)kept[at].order) {
                kept[at] = (AtlasRow){record, rank};
            }
        } else {
            json_object_object_add(keys, key, json_object_new_int64(*count));
            kept[(*count)++] = (AtlasRow){record, rank};
        }
        a->used = mark;
    }
    json_object_put(keys);
    return kept;
}

static void atlas_match_letters(Arena *a, const char *work) {
    double start = now();
    json_object *records = atlas_records(a);
    size_t count;
    AtlasRow *cases = atlas_primary(a, records, &count);
    json_object *reads = json_read(fmt(a, "%s/viewer/assets/atlas_eye_reads.json", HOME));
    json_object *eyes = json_object_new_object();
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(reads); i++) {
        json_object *row = json_object_array_get_idx(reads, i);
        const char *name = json_object_get_string(json_object_array_get_idx(row, 0));
        const char *letter = json_object_get_string(json_object_array_get_idx(row, 2));
        int body = json_object_get_int(json_object_array_get_idx(row, 1));
        for (int k = 0; k < ATLAS_LETTERS; k++) {
            if (!strcmp(atlas_letters[k].capital, letter)) {
                json_object_object_add(eyes, fmt(a, "%s/%d", name, body), json_object_new_int(k));
                break;
            }
        }
        a->used = mark;
    }
    json_object_put(reads);
    json_object **reports = push(a, count * sizeof(*reports));
    size_t total = 0, learned = 0;
    for (size_t i = 0; i < count; i++) {
        const char *case_id = atlas_text(cases[i].row, "case_id");
        json_object *doc = json_read(fmt(a, "%s/bodies/%s.json", work, case_id));
        reports[i] = doc;
        total += site_items(jget(doc, "rows"));
    }
    AtlasBody *bodies = push(a, total * sizeof(*bodies));
    mark = a->used;
    for (size_t i = 0, item = 0; i < count; i++) {
        const char *case_id = atlas_text(cases[i].row, "case_id");
        Array grid = npz_get(a, fmt(a, "%s/bodies/%s.npz", work, case_id), "labels");
        json_object *rows = jget(reports[i], "rows");
        size_t holds = a->used;
        for (size_t j = 0; j < site_items(rows); j++) {
            json_object *row = json_object_array_get_idx(rows, j), *eye;
            int id = ji64(row, "id");
            AtlasBody *one = bodies + item++;
            one->id = case_id;
            one->row = row;
            one->eye = json_object_object_get_ex(eyes, fmt(a, "%s/%d", case_id, id), &eye)
                ? json_object_get_int(eye)
                : -1;
            learned += one->eye >= 0;
            atlas_normal(a, grid.data, grid.shape[1], atlas_box(row), id, one->picture);
            a->used = holds;
        }
        a->used = mark;
    }
    AtlasTemplate *templates = push(a, (2 * ATLAS_LETTERS * 27 + learned) * sizeof(*templates));
    size_t used = 0;
    for (int k = 0; k < ATLAS_LETTERS; k++) {
        const AtlasLetter *letter = atlas_letters + k;
        for (int v = 0; v < letter->models; v++) {
            for (int s = -1; s <= 1; s++) {
                for (int t = -1; t <= 1; t++) {
                    for (int w = -1; w <= 1; w++) {
                        u8 line[ATLAS_PIXELS];
                        atlas_template(letter, v, .15 * s, 8 * t, 1 + .15 * w, line);
                        atlas_make_template(templates + used++, line, k, -1);
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < total; i++) {
        if (bodies[i].eye >= 0) {
            atlas_shape_template(templates + used++, bodies[i].picture, bodies[i].eye, i);
        }
    }
    json_object *matched = json_object_new_array_ext(total);
    size_t agree = 0;
    for (size_t i = 0; i < total; i++) {
        AtlasBody *body = bodies + i;
        AtlasTemplate own;
        atlas_shape_template(&own, body->picture, body->eye, i);
        double scores[ATLAS_LETTERS] = {0};
        if (own.count >= 8) {
            for (size_t j = 0; j < used; j++) {
                const AtlasTemplate *want = templates + j;
                if (want->body == (int)i) {
                    continue;
                }
                f32 near = 0, far = 0;
                for (int k = 0; k < own.count; k++) {
                    near += want->distance[own.cells[k]];
                }
                for (int k = 0; k < want->count; k++) {
                    far += own.distance[want->cells[k]];
                }
                f32 score = 1.0f - .5f * (near / own.count + far / fmaxf(1, want->count)) / 6.f;
                scores[want->letter] = fmax(scores[want->letter], score);
            }
            int holes = atlas_holes(body->picture);
            for (int k = 0; k < ATLAS_LETTERS; k++) {
                const int *expected = atlas_letters[k].holes;
                scores[k] -= .06 * fmin(abs(holes - expected[0]), abs(holes - expected[1]));
            }
        }
        int first = 0, second = 1;
        if (scores[1] > scores[0]) {
            first = 1;
            second = 0;
        }
        for (int k = 2; k < ATLAS_LETTERS; k++) {
            if (scores[k] > scores[first]) {
                second = first;
                first = k;
            } else if (scores[k] > scores[second]) {
                second = k;
            }
        }
        agree += body->eye == first;
        u8 blank[ATLAS_PIXELS];
        f32 dist[ATLAS_PIXELS];
        for (int k = 0; k < ATLAS_PIXELS; k++) {
            blank[k] = !body->picture[k];
        }
        atlas_distance(blank, dist, 0);
        f32 thickest = 0;
        for (int k = 0; k < ATLAS_PIXELS; k++) {
            thickest = fmaxf(thickest, dist[k]);
        }
        int blob = thickest / 56.0 >= .17 ||
            site_value(body->row, "w_mm") >= 1.7 * site_value(body->row, "h_mm");
        json_object *out = json_object_new_object();
        json_object_object_add(out, "case", json_object_new_string(body->id));
        json_object_object_add(out, "id", json_object_get(jget(body->row, "id")));
        json_object_object_add(out, "best", json_object_new_int(first));
        json_object_object_add(out, "score", jnum(decimal_round(scores[first], 4)));
        json_object_object_add(out, "second", json_object_new_int(second));
        json_object_object_add(
            out, "margin", jnum(decimal_round(scores[first] - scores[second], 4)));
        json_object_object_add(out, "eye", body->eye < 0 ? NULL : json_object_new_int(body->eye));
        for (const char **key = (const char *[]){"h_mm", "w_mm", "mm2", NULL}; *key; key++) {
            json_object_object_add(out, *key, json_object_get(jget(body->row, *key)));
        }
        json_object_object_add(out, "blob", json_object_new_boolean(blob));
        json_object_array_add(matched, out);
    }
    json_object *result = json_object_new_object();
    double agreement = learned ? (double)agree / learned : NAN;
    json_object_object_add(result, "bodies", matched);
    json_object_object_add(result, "checked", json_object_new_int64(learned));
    json_object_object_add(result, "agreement",
        isfinite(agreement) ? jnum(decimal_round(agreement, 4)) : json_object_new_double(NAN));
    json_write(fmt(a, "%s/matches.json", work), result);
    printf("%zu bodies, %zu read by eye, matcher agrees on %.3f %.0f s\n", total, learned,
        agreement, now() - start);
    for (size_t i = 0; i < count; i++) {
        json_object_put(reports[i]);
    }
    json_object_put(result);
    json_object_put(eyes);
    json_object_put(records);
}

#include <zlib.h>

typedef struct {
    int w, h, channels;
    u8 *pixels;
} AtlasImage;

typedef struct {
    int x, y;
    AtlasImage picture;
} AtlasTile;

typedef struct {
    json_object *record, *body, *rows;
    Array labels;
    AtlasTile *tiles;
    size_t count;
    int step, y, x, borrowed;
    double cell, ratio;
    double pub_cell, pub_dy, pub_dx;
    int pub_y, pub_x;
    AtlasImage published;
} AtlasView;

typedef struct {
    double dy, dx, score;
} AtlasShift;

typedef struct {
    int offset, x, y, w, h;
} AtlasGlyph;

static const AtlasGlyph atlas_glyphs[] = {
    {0, -2, 2, 33, 12},
    {396, 0, 4, 29, 8},
    {628, -2, 2, 33, 12},
    {1024, 0, 4, 29, 8},
    {1256, 0, 5, 8, 9},
    {1328, 0, 5, 8, 9},
    {1400, 0, 5, 6, 9},
    {1454, 0, 5, 8, 9},
    {1526, 0, 5, 7, 9},
    {1589, 0, 5, 7, 9},
    {1652, 0, 5, 9, 9},
    {1733, 0, 5, 9, 9},
    {1814, 0, 5, 4, 9},
    {1850, 0, 5, 7, 9},
    {1913, 0, 5, 8, 9},
    {1985, 0, 5, 12, 9},
    {2093, 0, 5, 9, 9},
    {2174, 0, 5, 7, 9},
    {2237, 0, 5, 9, 9},
    {2318, 0, 5, 9, 9},
    {2399, 0, 5, 7, 9},
    {2462, 0, 5, 6, 9},
    {2516, 0, 5, 7, 9},
    {2579, 0, 5, 7, 9},
    {2642, 0, 4, 11, 11},
    {2763, 0, 5, 7, 9},
    {2826, 0, 5, 11, 9},
    {2925, 0, 5, 9, 9},
};
enum { ATLAS_GLYPH_BYTES = 3006 };
static const char atlas_glyph_masks[] =
    "eNqlVgtQlFUU/lgeG0gUako+Rq18hZj2kCxSfFABPjZUwFHTnJzEdNSZNEdNR03R0Sw0cXR8G41jjERqUvk"
    "I8LEJ4qNt0GJASHRxgV1ll2VX5HbO/X8eY/tvznRn9p7v/ue7j/+c7579ASSWun/tCM3mv01QO6xN2M5+UQ"
    "yMuUzWKWy1B6a1oDAi1EnCdeiqRXNrbEHFAcBtldBFeGrTgLkqoZtHwn7aI8Eb4Ucg3C7RIs+EHOCMCg1aB"
    "KsKP/VIOA7kqnBotSfCciCi4CGBmgUYU+IuPO00591uOFfajDLbAUJoR1nAixMtTsGLyHVaUatTSCzaojYz"
    "W5z43075ypozBxkp/425nTTf5pISsx+0/LomhVCmpTk/NepZWpprJqzR0pyviqx9tTRXrMKpWpr7XoUfa0h"
    "qZDMcp0GYqaIb3TU019Mpwf1InZbmonNqGu8c6A0tzbUmyouo/sPpQXOtq3rUXNv+X7Ly4hReZ7bu6XHZK3"
    "/7sXnZdCOW7RepG79hKd0aGlUXBIys0uksk4Hde4C9x6C38Ss86JhQGwjoLSkZG3nO56dwTjRZVhI0Jfu+U"
    "R/DFiiaw1b3lo3ibKJpZ8J47NMt+7Tif9chbfuMPNB8YT/aA8gn3kmYpvKyj9MXJcpX860okPbt4oYBbPcv"
    "yUolE3jvuaSbPlTLCxFkjwKyFwOH0tHe1QswVPvPukgcvS1us6KLhXwK8cBfXfpRQ77V8OzjeQ9DPuFF3MH"
    "8YEM+9xGOcA7lhbU8mFPyBPVdbDE8yOQYI65G3vWNcvchuDSJBjvWqqvu2+QRSO4mrxwZk53rqPvI5KrO6A"
    "IsuTta3+tUcUBQ3UR6GupMGCz4MMhbF38PlUL0zNo6pEnK4OyK0MZ3yIa5h2F3WaR/H+NPQMCGigeVaVSJL"
    "yUq22j9eGfi6I59zcPUoiAyhjvdybnm7lAO17UbgWxmnNspg9fDmiwnTrzfW87dxmEz/eXDJ3mtxBbF9su1"
    "+9I5ipWDYy10xJEl8K8hpe5aT7+DCLC+QlmvCzSUEsfPkrSf0470b5HrHCFXqT9FKxqzGR86W0g4pYEC0tk"
    "6m7Hh6DJg2Z5kid8r9fEtj1Swf1XM+CtQMDYdzklpxv3d90MkLoqn8O0AJpwnW9UVePMPIlTspdpSQmC+a6"
    "AC4nfmKGBMmD1GAVhVMEIBweY0BWB2A4Eiirzvb4WIkUnu67nA7CLXYs+uGXJeNJBCeT/4LLBA5t3kH1jDe"
    "W/vHDu4Scn7Eso7tyNpj+S9kysSuzjvF47SWZaWuyvTgpVbKcQy78p8DNCizHzRVD6PHX6xzljpP7FZmpyV"
    "ZPQTGwaS0p2Xxz560YqSuB/Fp/n5mVurePCqg68Nku0vUf901RTqszjdC8tDKf2O1wnKbxPjn/IlMoEBSj1"
    "8fNPPxSVlD4dp4UB3L4yqDcGR7Tj5Gf1nO0bXPsVbWlcwM1rwAIOUeaoJd6n5GVaqgoT8elH9HVeqWbYpk8"
    "x9ttojEFJtwGgz1ahMxNX5SJhcg+llkHB4E8Y5lKdJZnSwj5cwi/5vF9VMmmAO+8pMJQ2JRQ3CcrCzssnwm"
    "4ooyunipmf4/r4aEfUvYFR991/W84FNt7lSxYktTMrPrKNbMaVMn30AwbeS0NcdufwsPU877et6Hh/k0mfc"
    "+/Rx0A+zjYDxw7ZQ1xQuYcF0oDSJod5KQV9aFj3f2CPjvI5kNN/kEhU7OijHYq7aZuapwHC1wbKXauEsc2y"
    "73scv+gRaxtPTJ63xUQ/17D+xPNYhiYc3DxChDM7PxdV5HFt7V7x4fTL6FyeDvheuIVWILf8AzpTz5Q==";

typedef struct {
    u8 data[ATLAS_GLYPH_BYTES];
    int ready;
} AtlasFont;

static void atlas_glyph_on(
    AtlasFont *font, AtlasImage out, int glyph, int x, int y, const u8 color[3]) {
    if (!font->ready) {
        u8 encoded[sizeof(atlas_glyph_masks)];
        size_t chars = sizeof(atlas_glyph_masks) - 1;
        int size = EVP_DecodeBlock(encoded, (const u8 *)atlas_glyph_masks, chars);
        size -= (atlas_glyph_masks[chars - 1] == '=') + (atlas_glyph_masks[chars - 2] == '=');
        uLongf bytes = sizeof(font->data);
        need(size >= 0 && uncompress(font->data, &bytes, encoded, size) == Z_OK &&
                bytes == sizeof(font->data),
            "cannot load atlas glyphs");
        font->ready = 1;
    }
    AtlasGlyph g = atlas_glyphs[glyph];
    for (int row = 0; row < g.h; row++) {
        int py = y + row + g.y;
        if (py < 0 || py >= out.h) {
            continue;
        }
        for (int col = 0; col < g.w; col++) {
            int px = x + col + g.x;
            if (px < 0 || px >= out.w) {
                continue;
            }
            int alpha = font->data[g.offset + row * g.w + col];
            u8 *pixel = out.pixels + ((size_t)py * out.w + px) * out.channels;
            for (int c = 0; c < out.channels; c++) {
                pixel[c] = ((255 - alpha) * pixel[c] + alpha * color[c] + 127) / 255;
            }
        }
    }
}

static AtlasImage atlas_image(Arena *a, int w, int h, int channels) {
    need(w >= 0 && h >= 0 && (size_t)w * h * channels < 8 * GB, "atlas image is too large");
    return (AtlasImage){w, h, channels, pushz(a, (size_t)w * h * channels)};
}

static void atlas_rect(AtlasImage out, int left, int top, int right, int bottom, u8 grey) {
    for (int y = top < 0 ? 0 : top; y <= bottom && y < out.h; y++) {
        for (int x = left < 0 ? 0 : left; x <= right && x < out.w; x++) {
            memset(out.pixels + ((size_t)y * out.w + x) * out.channels, grey, out.channels);
        }
    }
}

static void atlas_bar(AtlasFont *font, AtlasImage out, double per_px, int mm) {
    int length = nearbyint(mm / per_px), y = out.h - 12;
    atlas_rect(out, 7, y - 1, 9 + length, y + 4, 0);
    atlas_rect(out, 8, y, 8 + length, y + 3, 255);
    const u8 white[] = {255, 255, 255}, black[] = {0, 0, 0};
    int label = mm == 1 ? 0 : 2;
    atlas_glyph_on(font, out, label, 10, y - 15, black);
    atlas_glyph_on(font, out, label + 1, 10, y - 15, white);
}

static AtlasImage atlas_grey(Arena *a, const char *path) {
    int w, h;
    u8 *pixels = hx_sheet_png(path, &w, &h);
    need(pixels != NULL, "cannot read atlas PNG %s", path);
    AtlasImage out = atlas_image(a, w, h, 1);
    memcpy(out.pixels, pixels, (size_t)w * h);
    free(pixels);
    return out;
}

static AtlasImage atlas_resize(Arena *a, AtlasImage image, int w, int h, int linear) {
    AtlasImage out = atlas_image(a, w, h, 1);
    if (image.w && image.h && w && h) {
        need(hx_site_resize_grey(image.pixels, image.w, image.h, out.pixels, w, h, linear),
            "cannot resize atlas picture");
    }
    return out;
}

static AtlasImage atlas_crop(Arena *a, AtlasImage image, int y0, int y1, int x0, int x1) {
    AtlasImage out = atlas_image(a, x1 - x0, y1 - y0, image.channels);
    int l = x0 < 0 ? 0 : x0, t = y0 < 0 ? 0 : y0;
    int r = x1 < image.w ? x1 : image.w, b = y1 < image.h ? y1 : image.h;
    for (int y = t; l < r && y < b; y++) {
        memcpy(out.pixels + ((size_t)(y - y0) * out.w + l - x0) * image.channels,
            image.pixels + ((size_t)y * image.w + l) * image.channels,
            (size_t)(r - l) * image.channels);
    }
    return out;
}

static AtlasImage atlas_rgb(Arena *a, AtlasImage grey) {
    AtlasImage out = atlas_image(a, grey.w, grey.h, 3);
    for (size_t i = 0, n = (size_t)grey.w * grey.h; i < n; i++) {
        memset(out.pixels + 3 * i, grey.pixels[i], 3);
    }
    return out;
}

static void atlas_paste(AtlasImage canvas, AtlasImage image, int x, int y) {
    need(image.channels == canvas.channels && x >= 0 && y >= 0 && x + image.w <= canvas.w &&
            y + image.h <= canvas.h,
        "atlas panel does not fit");
    for (int row = 0; row < image.h; row++) {
        memcpy(canvas.pixels + ((size_t)(row + y) * canvas.w + x) * canvas.channels,
            image.pixels + (size_t)row * image.w * image.channels,
            (size_t)image.w * image.channels);
    }
}

static AtlasImage atlas_outline(Arena *a, AtlasImage face, AtlasImage label) {
    AtlasImage out = atlas_rgb(a, face);
    if (!label.w || !label.h) {
        return out;
    }
    size_t mark = a->used;
    u8 *mask = push(a, (size_t)face.w * face.h);
    double sy = (double)label.h / face.h, sx = (double)label.w / face.w, y = .5 * sy;
    for (int row = 0; row < face.h; row++, y += sy) {
        double x = .5 * sx;
        for (int col = 0; col < face.w; col++, x += sx) {
            mask[(size_t)row * face.w + col] = label.pixels[(size_t)(int)y * label.w + (int)x];
        }
    }
    for (int row = 0; row < face.h; row++) {
        for (int col = 0; col < face.w; col++) {
            size_t at = (size_t)row * face.w + col;
            if (mask[at] &&
                (row == 0 || col == 0 || row + 1 == face.h || col + 1 == face.w || !mask[at - 1] ||
                    !mask[at + 1] || !mask[at - face.w] || !mask[at + face.w])) {
                out.pixels[3 * at] = 255;
                out.pixels[3 * at + 1] = 196;
                out.pixels[3 * at + 2] = 0;
            }
        }
    }
    a->used = mark;
    return out;
}

typedef struct {
    int w, h;
    double cell;
    f32 *values;
} AtlasCoarse;

static AtlasCoarse atlas_coarse(Arena *a, AtlasView *view) {
    int stride = view->labels.shape[1], rows = view->labels.shape[0];
    int step = fmax(1, nearbyint(.15 / (view->step * view->cell)));
    int h = rows / step, w = stride / step;
    f32 *values = push(a, (size_t)h * w * sizeof(*values));
    const u32 *labels = view->labels.data;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum = 0;
            for (int ry = y * step; ry < (y + 1) * step; ry++) {
                for (int cx = x * step; cx < (x + 1) * step; cx++) {
                    sum += labels[(size_t)ry * stride + cx] != 0;
                }
            }
            values[(size_t)y * w + x] = (double)sum / ((double)step * step);
        }
    }
    return (AtlasCoarse){w, h, step * view->step * view->cell, values};
}

static void atlas_fft(double *r, double *i, int w, int h, int inverse, double *scratch) {
    int limit = w > h ? w : h;
    double *cosines = scratch + 2 * limit, *sines = cosines + limit;
    fft_roots(w, cosines, sines);
    for (int y = 0; y < h; y++) {
        fft(r + (size_t)y * w, i + (size_t)y * w, w, 1, inverse, scratch, cosines, sines);
    }
    if (w != h) {
        fft_roots(h, cosines, sines);
    }
    for (int x = 0; x < w; x++) {
        fft(r + x, i + x, h, w, inverse, scratch, cosines, sines);
    }
}

static AtlasShift atlas_shift(Arena *a, AtlasView *left, AtlasView *right) {
    size_t mark = a->used;
    AtlasCoarse one = atlas_coarse(a, left), two = atlas_coarse(a, right);
    int w2 = nearbyint(two.w * (two.cell / one.cell));
    int h2 = nearbyint(two.h * (two.cell / one.cell));
    if (!one.w || !one.h || !two.w || !two.h || !w2 || !h2) {
        a->used = mark;
        return (AtlasShift){0};
    }
    int w = 1, h = 1, cols = one.w + w2, rows = one.h + h2;
    while (w < cols) {
        w *= 2;
    }
    while (h < rows) {
        h *= 2;
    }
    size_t n = (size_t)w * h;
    double *ar = pushz(a, n * sizeof(*ar)), *ai = pushz(a, n * sizeof(*ai));
    double *br = pushz(a, n * sizeof(*br)), *bi = pushz(a, n * sizeof(*bi));
    double *scratch = push(a, 4 * (w > h ? w : h) * sizeof(*scratch));
    f32 *power_a = push(a, (size_t)one.w * one.h * sizeof(*power_a));
    f32 *power_b = push(a, (size_t)w2 * h2 * sizeof(*power_b));
    for (int y = 0; y < one.h; y++) {
        for (int x = 0; x < one.w; x++) {
            size_t at = (size_t)y * one.w + x;
            f32 v = one.values[at];
            ar[(size_t)y * w + x] = v;
            power_a[at] = v * v;
        }
    }
    double sx = w2 > 1 ? (double)(two.w - 1) / (w2 - 1) : 0;
    double sy = h2 > 1 ? (double)(two.h - 1) / (h2 - 1) : 0;
    for (int y = 0; y < h2; y++) {
        double py = fmin(y * sy, two.h - 1);
        int yy = py, y1 = yy + (yy + 1 < two.h);
        double fy = py - yy;
        for (int x = 0; x < w2; x++) {
            double px = fmin(x * sx, two.w - 1);
            int xx = px, x1 = xx + (xx + 1 < two.w);
            double fx = px - xx;
            const f32 *top = two.values + (size_t)yy * two.w;
            const f32 *bottom = two.values + (size_t)y1 * two.w;
            f32 v = (1 - fy) * ((1 - fx) * top[xx] + fx * top[x1]) +
                fy * ((1 - fx) * bottom[xx] + fx * bottom[x1]);
            br[(size_t)y * w + x] = v;
            power_b[(size_t)y * w2 + x] = v * v;
        }
    }
    double strength =
        sqrt(pairwise(power_a, (i64)one.w * one.h)) * sqrt(pairwise(power_b, (i64)w2 * h2));
    atlas_fft(ar, ai, w, h, 0, scratch);
    atlas_fft(br, bi, w, h, 0, scratch);
    for (size_t i = 0; i < n; i++) {
        double re = br[i] * ar[i] + bi[i] * ai[i];
        bi[i] = bi[i] * ar[i] - br[i] * ai[i];
        br[i] = re;
    }
    atlas_fft(br, bi, w, h, 1, scratch);
    double best = -INFINITY;
    int dx = 0, dy = 0;
    for (int r = 0; r < rows; r++) {
        int y = r > rows / 2 ? r - rows : r;
        int at = r < h2 ? r : r - rows + h;
        for (int c = 0; c < cols; c++) {
            int x = c > cols / 2 ? c - cols : c;
            double value = br[(size_t)at * w + (c < w2 ? c : c - cols + w)] / n;
            if (value > best + 1e-10) {
                best = value;
                dx = x;
                dy = y;
            }
        }
    }
    AtlasShift result = {dy * one.cell, dx * one.cell, best / fmax(strength, 1e-6)};
    a->used = mark;
    return result;
}

static AtlasView atlas_view(Arena *a, const char *work, json_object *record, json_object *others) {
    const char *id = atlas_text(record, "case_id");
    json_object *box = jget(record, "box");
    AtlasView view = {.record = record,
        .cell = atlas_lateral(record) / 1000,
        .x = json_object_get_int(json_object_array_get_idx(box, 2)),
        .y = json_object_get_int(json_object_array_get_idx(box, 0))};
    view.body = json_read(fmt(a, "%s/bodies/%s.json", work, id));
    view.rows = jget(view.body, "rows");
    view.step = ji64(view.body, "step");
    view.labels = npz_get(a, fmt(a, "%s/bodies/%s.npz", work, id), "labels");
    const char *dir = fmt(a, "%s/faces/%s", work, id);
    json_object *tiles = json_read(fmt(a, "%s/tiles.json", dir));
    view.count = site_items(tiles);
    view.tiles = push(a, view.count * sizeof(*view.tiles));
    for (size_t i = 0; i < view.count; i++) {
        json_object *tile = json_object_array_get_idx(tiles, i);
        const char *path = fmt(a, "%s/%s", dir, atlas_text(tile, "file"));
        AtlasImage image = atlas_grey(a, path);
        float low = site_value(tile, "low");
        float scale = 255.0 / (site_value(tile, "high") - site_value(tile, "low"));
        for (size_t j = 0, n = (size_t)image.w * image.h; j < n; j++) {
            image.pixels[j] = fminf(255, fmaxf(0, (image.pixels[j] - low) * scale));
        }
        view.tiles[i] = (AtlasTile){ji64(tile, "x"), ji64(tile, "y"), image};
    }
    json_object_put(tiles);
    const char *map = atlas_published(a, record);
    json_object *published = record;
    if (!map) {
        for (size_t i = 0; !map && i < site_items(others); i++) {
            json_object *one = json_object_array_get_idx(others, i);
            int energy = atlas_energy(one);
            if ((energy != 77 && energy != 78) || strstr(atlas_text(one, "case_id"), "1.129um") ||
                strcmp(atlas_text(one, "sample"), atlas_text(record, "sample")) != 0 ||
                strcmp(atlas_text(one, "segment"), atlas_text(record, "segment")) != 0) {
                continue;
            }
            map = atlas_published(a, one);
            if (map) {
                published = one;
                view.borrowed = 1;
                size_t mark = a->used;
                AtlasView ref = {.cell = atlas_lateral(one) / 1000};
                const char *case_id = atlas_text(one, "case_id");
                ref.body = json_read(fmt(a, "%s/bodies/%s.json", work, case_id));
                ref.step = ji64(ref.body, "step");
                ref.labels = npz_get(a, fmt(a, "%s/bodies/%s.npz", work, case_id), "labels");
                AtlasShift moved = atlas_shift(a, &view, &ref);
                view.pub_dy = moved.dy;
                view.pub_dx = moved.dx;
                json_object_put(ref.body);
                a->used = mark;
            }
        }
    }
    if (map) {
        AtlasJpeg image = atlas_jpeg(a, map);
        view.published = (AtlasImage){image.w, image.h, 1, image.pixels};
        json_object *shape = jget(published, "shape"), *origin = jget(published, "box");
        view.ratio = json_object_get_double(json_object_array_get_idx(shape, 1)) / image.h;
        view.pub_cell = atlas_lateral(published) / 1000;
        view.pub_y = json_object_get_int(json_object_array_get_idx(origin, 0));
        view.pub_x = json_object_get_int(json_object_array_get_idx(origin, 2));
    }
    return view;
}

static AtlasImage atlas_face(Arena *a, AtlasView *view, AtlasRegion area) {
    AtlasImage out = atlas_image(a, area.x1 - area.x0, area.y1 - area.y0, 1);
    for (size_t i = 0; i < view->count; i++) {
        AtlasTile *tile = view->tiles + i;
        int left = area.x0 > tile->x ? area.x0 : tile->x;
        int top = area.y0 > tile->y ? area.y0 : tile->y;
        int right = area.x1 < tile->x + tile->picture.w ? area.x1 : tile->x + tile->picture.w;
        int bottom = area.y1 < tile->y + tile->picture.h ? area.y1 : tile->y + tile->picture.h;
        for (int y = top; left < right && y < bottom; y++) {
            memcpy(out.pixels + (size_t)(y - area.y0) * out.w + left - area.x0,
                tile->picture.pixels + (size_t)(y - tile->y) * tile->picture.w + left - tile->x,
                right - left);
        }
    }
    return out;
}

static int atlas_floor_div(int value, int div) { return value / div - (value % div < 0); }

static AtlasImage atlas_labels(Arena *a, AtlasView *view, AtlasRegion area, int body) {
    int step = view->step;
    int x0 = atlas_floor_div(area.x0, step), y0 = atlas_floor_div(area.y0, step);
    int x1 = atlas_floor_div(area.x1, step), y1 = atlas_floor_div(area.y1, step);
    int rw = view->labels.shape[1], rh = view->labels.shape[0];
    const u32 *labels = view->labels.data;
    int width = body >= 0 ? x1 - x0 : (area.x1 - area.x0) / step;
    int height = body >= 0 ? y1 - y0 : (area.y1 - area.y0) / step;
    int fx = fmax(0, x0), fy = fmax(0, y0), ex = fmin(rw, x1), ey = fmin(rh, y1);
    width = fmax(width, fmax(ex, fx) - x0);
    height = fmax(height, fmax(ey, fy) - y0);
    AtlasImage label = atlas_image(a, width, height, 1);
    for (int y = fy; y < ey; y++) {
        for (int x = fx; x < ex; x++) {
            u32 value = labels[(size_t)y * rw + x];
            label.pixels[(size_t)(y - y0) * width + x - x0] =
                (body >= 0 ? value == (u32)body : value > 0) ? 255 : 0;
        }
    }
    return label;
}

static AtlasImage atlas_ink_map(Arena *a, AtlasView *view, AtlasRegion area) {
    if (!view->published.pixels) {
        return (AtlasImage){0};
    }
    double x0_mm = area.x0, x1_mm = area.x1, y0_mm = area.y0, y1_mm = area.y1;
    if (view->borrowed) {
        x0_mm = (x0_mm * view->cell + view->pub_dx) / view->pub_cell;
        x1_mm = (x1_mm * view->cell + view->pub_dx) / view->pub_cell;
        y0_mm = (y0_mm * view->cell + view->pub_dy) / view->pub_cell;
        y1_mm = (y1_mm * view->cell + view->pub_dy) / view->pub_cell;
    }
    int y0 = (view->pub_y + y0_mm) / view->ratio, y1 = (view->pub_y + y1_mm) / view->ratio;
    int x0 = (view->pub_x + x0_mm) / view->ratio, x1 = (view->pub_x + x1_mm) / view->ratio;
    return atlas_crop(a, view->published, y0, y1, x0, x1);
}

static void atlas_panels(
    Arena *a, AtlasFont *font, AtlasView *view, AtlasRegion area, int w, int h, const char *path) {
    size_t mark = a->used;
    AtlasImage raw = atlas_resize(a, atlas_face(a, view, area), w, h, 0);
    AtlasImage left = atlas_rgb(a, raw);
    AtlasImage middle = atlas_outline(a, raw, atlas_labels(a, view, area, -1));
    AtlasImage map = atlas_resize(a, atlas_ink_map(a, view, area), w, h, 0);
    AtlasImage out = atlas_image(a, 3 * w + 4, h, 3);
    double mm = (area.x1 - area.x0) * view->cell;
    atlas_bar(font, left, mm / w, mm < 15 ? 1 : 5);
    atlas_paste(out, left, 0, 0);
    atlas_paste(out, middle, w + 2, 0);
    atlas_paste(out, atlas_rgb(a, map), 2 * w + 4, 0);
    need(hx_site_save_jpeg(path, out.pixels, out.w, out.h, 90), "cannot write %s", path);
    a->used = mark;
}

static json_object *atlas_view_entry(AtlasView *view) {
    json_object *row = json_object_new_object();
    json_object_object_add(row, "case", json_object_get(jget(view->record, "case_id")));
    json_object_object_add(row, "sample", json_object_get(jget(view->record, "sample")));
    json_object_object_add(row, "segment", json_object_get(jget(view->record, "segment")));
    json_object_object_add(row, "energy", json_object_new_int(atlas_energy(view->record)));
    json_object_object_add(row, "um", jnum(decimal_round(jf64(view->record, "voxel_um"), 2)));
    return row;
}

static json_object *atlas_glyph(json_object *match, const char *key) {
    json_object *field = json_object_object_get(match, key);
    int index = json_object_get_int(field);
    return field && index >= 0 && index < ATLAS_LETTERS
        ? json_object_new_string(atlas_letters[index].capital)
        : NULL;
}

static void atlas_letter_cards(Arena *a, const char *work) {
    json_object *records = atlas_records(a), *matches = json_read(fmt(a, "%s/matches.json", work));
    json_object *bodies = jget(matches, "bodies"), *reported = json_object_new_array();
    json_object *by_key = json_object_new_object();
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(bodies); i++) {
        json_object *match = json_object_array_get_idx(bodies, i);
        json_object_object_add(by_key,
            fmt(a, "%s/%lld", jstr(match, "case"), (long long)ji64(match, "id")),
            match ? json_object_get(match) : NULL);
        a->used = mark;
    }
    const char *atlas = fmt(a, "%s/website/atlas/letters", OUTPUTS);
    mkdirs(atlas);
    mark = a->used;
    AtlasFont font = {0};
    for (size_t i = 0; i < site_items(records); i++) {
        a->used = mark;
        json_object *record = json_object_array_get_idx(records, i);
        const char *name = jstr(record, "case_id"), *folder = fmt(a, "%s/%s", atlas, name);
        mkdirs(folder);
        AtlasView view = atlas_view(a, work, record, records);
        size_t again = a->used;
        for (size_t j = 0; j < site_items(view.rows); j++) {
            a->used = again;
            json_object *body = json_object_array_get_idx(view.rows, j);
            AtlasRegion box = atlas_box(body);
            int id = ji64(body, "id");
            int y = (box.y0 + box.y1) * view.step / 2;
            int x = (box.x0 + box.x1) * view.step / 2;
            double side = fmax(9, 1.35 * fmax(site_value(body, "h_mm"), site_value(body, "w_mm")));
            int half = side / view.cell / 2;
            const char *file = fmt(a, "%05d.jpg", id), *path = fmt(a, "%s/%s", folder, file);
            if (!exists(path)) {
                atlas_panels(a, &font, &view,
                    (AtlasRegion){y - half, y + half, x - half, x + half, 0}, 420, 420, path);
            }
            json_object *match = json_object_object_get(by_key, fmt(a, "%s/%d", name, id));
            json_object *row = atlas_view_entry(&view);
            json_object_object_add(row, "id", json_object_get(jget(body, "id")));
            json_object_object_add(row, "mm2", json_object_get(jget(body, "mm2")));
            json_object_object_add(row, "h", json_object_get(jget(body, "h_mm")));
            json_object_object_add(row, "w", json_object_get(jget(body, "w_mm")));
            json_object_object_add(row, "y_mm", jnum(decimal_round(y * view.cell, 2)));
            json_object_object_add(row, "x_mm", jnum(decimal_round(x * view.cell, 2)));
            json_object_object_add(
                row, "file", json_object_new_string(fmt(a, "letters/%s/%s", name, file)));
            json_object_object_add(row, "side", jnum(decimal_round(2 * half * view.cell, 1)));
            json_object_object_add(row, "letter", atlas_glyph(match, "best"));
            json_object_object_add(
                row, "score", json_object_get(json_object_object_get(match, "score")));
            json_object_object_add(row, "eye", atlas_glyph(match, "eye"));
            json_object_object_add(
                row, "published", json_object_new_boolean(view.published.pixels != NULL));
            json_object_array_add(reported, row);
        }
        printf("%s %zu\n", name, site_items(view.rows));
        json_object_put(view.body);
    }
    json_write(fmt(a, "%s/cards.json", work), reported);
    json_object_put(by_key);
    json_object_put(reported);
    json_object_put(matches);
    json_object_put(records);
}

static json_object *atlas_region_json(AtlasRegion region) {
    json_object *row = json_object_new_array_ext(4);
    int coords[] = {region.y0, region.y1, region.x0, region.x1};
    for (int i = 0; i < 4; i++) {
        json_object_array_add(row, json_object_new_int(coords[i]));
    }
    return row;
}

static AtlasRegion atlas_region_value(json_object *row) {
    need(site_items(row) == 4, "atlas region must have four coordinates");
    int v[4];
    for (int i = 0; i < 4; i++) {
        v[i] = json_object_get_int(json_object_array_get_idx(row, i));
    }
    return (AtlasRegion){v[0], v[1], v[2], v[3], 0};
}

static void atlas_letter_windows(Arena *a, const char *work) {
    json_object *records = atlas_records(a), *reported = json_object_new_array();
    const char *atlas = fmt(a, "%s/website/atlas/windows", OUTPUTS);
    mkdirs(atlas);
    AtlasFont font = {0};
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(records); i++) {
        a->used = mark;
        json_object *record = json_object_array_get_idx(records, i);
        AtlasView view = atlas_view(a, work, record, records);
        size_t count = site_items(view.rows);
        if (!count) {
            json_object_put(view.body);
            continue;
        }
        const char *case_id = jstr(record, "case_id");
        int hw = 15 / view.cell, hh = 9 / view.cell;
        AtlasPoint *centres = push(a, count * sizeof(*centres));
        u8 *left = push(a, count);
        memset(left, 1, count);
        for (size_t j = 0; j < count; j++) {
            AtlasRegion region = atlas_box(json_object_array_get_idx(view.rows, j));
            centres[j] = (AtlasPoint){(region.x0 + region.x1) * view.step / 2.0,
                (region.y0 + region.y1) * view.step / 2.0};
        }
        size_t seen = 0;
        int k = 0;
        size_t again = a->used;
        while (seen < count && k < 12) {
            a->used = again;
            size_t best = 0, most = 0;
            for (size_t j = 0; j < count; j++) {
                size_t held = 0;
                for (size_t step = 0; step < count; step++) {
                    held += left[step] && fabs(centres[j].x - centres[step].x) < hw &&
                        fabs(centres[j].y - centres[step].y) < hh;
                }
                if (held > most) {
                    most = held;
                    best = j;
                }
            }
            if (!most) {
                break;
            }
            AtlasPoint sum = {0};
            for (size_t j = 0; j < count; j++) {
                if (left[j] && fabs(centres[best].x - centres[j].x) < hw &&
                    fabs(centres[best].y - centres[j].y) < hh) {
                    left[j] = 0;
                    sum.x += centres[j].x;
                    sum.y += centres[j].y;
                }
            }
            seen += most;
            if (most < 2) {
                continue;
            }
            int x = (int)(sum.x / most), y = (int)(sum.y / most);
            AtlasRegion area = {y - hh, y + hh, x - hw, x + hw, 0};
            const char *name = fmt(a, "%s-%02d.jpg", case_id, k++);
            const char *path = fmt(a, "%s/%s", atlas, name);
            if (!exists(path)) {
                atlas_panels(a, &font, &view, area, 1000, 600, path);
            }
            json_object *row = atlas_view_entry(&view);
            json_object_object_add(row, "file", json_object_new_string(fmt(a, "windows/%s", name)));
            json_object_object_add(row, "letters", json_object_new_int64(most));
            json_object_object_add(
                row, "published", json_object_new_boolean(view.published.pixels != NULL));
            json_object_object_add(row, "box", atlas_region_json(area));
            json_object_array_add(reported, row);
        }
        printf("%s %d\n", case_id, k);
        json_object_put(view.body);
    }
    json_write(fmt(a, "%s/windows.json", work), reported);
    json_object_put(reported);
    json_object_put(records);
}

static int atlas_fragment_order(const void *left, const void *right) {
    json_object *a = *(json_object *const *)left, *b = *(json_object *const *)right;
    return atlas_compare_double(site_value(b, "mm_h") * site_value(b, "mm_w"),
        site_value(a, "mm_h") * site_value(a, "mm_w"));
}

static f32 *atlas_density(Arena *a, AtlasImage mask, int width) {
    int w = mask.w, h = mask.h, offset = width / 2;
    size_t n = (size_t)w * h;
    f32 *density = push(a, n * sizeof(*density));
    size_t mark = a->used;
    f32 *vertical = push(a, n * sizeof(*vertical));
    for (int x = 0; x < w; x++) {
        double total = 0;
        for (int y = 0; y < width; y++) {
            total += mask.pixels[reflected(y - offset, h) * w + x] != 0;
        }
        for (int y = 0; y < h; y++) {
            vertical[(size_t)y * w + x] = total / width;
            total += (mask.pixels[reflected(y + width - offset, h) * w + x] != 0) -
                (mask.pixels[reflected(y - offset, h) * w + x] != 0);
        }
    }
    for (int y = 0; y < h; y++) {
        const f32 *line = vertical + (size_t)y * w;
        double total = 0;
        for (int x = 0; x < width; x++) {
            total += line[reflected(x - offset, w)];
        }
        for (int x = 0; x < w; x++) {
            density[(size_t)y * w + x] = total / width;
            total +=
                (double)line[reflected(x + width - offset, w)] - line[reflected(x - offset, w)];
        }
    }
    a->used = mark;
    return density;
}

static void atlas_fragments(Arena *a, const char *work) {
    json_object *records = atlas_records(a), *reported = json_object_new_array();
    const char *atlas = fmt(a, "%s/website/atlas/fragments", OUTPUTS);
    mkdirs(atlas);
    AtlasFont font = {0};
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(records); i++) {
        a->used = mark;
        json_object *record = json_object_array_get_idx(records, i);
        if (strcmp(atlas_text(record, "truth"), "infrared") != 0) {
            continue;
        }
        AtlasView view = atlas_view(a, work, record, NULL);
        const char *name = jstr(record, "segment");
        const char *dir = fmt(a, "%s/datasets/fragments/%s", LIBRARY, name);
        AtlasImage ir = atlas_grey(a, fmt(a, "%s/ir.png", dir));
        AtlasImage mask = atlas_grey(a, fmt(a, "%s/mask.png", dir));
        AtlasRegion box = atlas_region_value(jget(record, "box"));
        int w = box.x1 - box.x0, h = box.y1 - box.y0;
        AtlasImage shown = atlas_crop(a, ir, box.y0, box.y1, box.x0, box.x1);
        mask = atlas_crop(a, mask, box.y0, box.y1, box.x0, box.x1);
        for (size_t j = 0, n = (size_t)h * w; j < n; j++) {
            if (!mask.pixels[j]) {
                shown.pixels[j] = 0;
            }
        }
        AtlasRegion extent = {0, h, 0, w, 0};
        AtlasImage face = atlas_face(a, &view, extent), labels = atlas_labels(a, &view, extent, -1);
        double scale = fmax(h, w) / 2400;
        int cols = w / scale, rows = h / scale;
        json_object *item = json_object_new_object(), *windows = json_object_new_array();
        json_object_object_add(item, "name", json_object_new_string(name));
        json_object_object_add(item, "sample", json_object_get(jget(record, "sample")));
        json_object_object_add(item, "mm_h", jnum(decimal_round(h * view.cell, 1)));
        json_object_object_add(item, "mm_w", jnum(decimal_round(w * view.cell, 1)));
        json_object_object_add(item, "windows", windows);
        AtlasImage small_ir = atlas_resize(a, shown, cols, rows, 0);
        size_t again = a->used;
        const char *kinds[] = {"ir", "face", "labels"};
        for (int j = 0; j < 3; j++) {
            a->used = again;
            AtlasImage panel = j == 2
                ? atlas_outline(a, small_ir, labels)
                : atlas_rgb(a, j == 1 ? atlas_resize(a, face, cols, rows, 0) : small_ir);
            if (j < 2) {
                atlas_bar(&font, panel, scale * view.cell, 5);
            }
            const char *file = fmt(a, "%s-%s.jpg", name, kinds[j]);
            const char *path = fmt(a, "%s/%s", atlas, file);
            need(hx_site_save_jpeg(path, panel.pixels, panel.w, panel.h, 92), "cannot write %s",
                path);
            json_object_object_add(
                item, kinds[j], json_object_new_string(fmt(a, "fragments/%s", file)));
        }
        a->used = again;
        int side = 10 / view.cell, half = side / 2, reach = side / view.step;
        f32 *density = labels.w && labels.h ? atlas_density(a, labels, fmax(1, reach)) : NULL;
        again = a->used;
        for (int k = 0; density && k < 8; k++) {
            a->used = again;
            size_t first = 0, n = (size_t)labels.w * labels.h;
            for (size_t j = 1; j < n; j++) {
                if (density[j] > density[first]) {
                    first = j;
                }
            }
            if (density[first] <= .01) {
                break;
            }
            int x = first % labels.w, y = first / labels.w;
            int top = (int)fmax(0, y * view.step - half) / 3 * 3;
            int left = (int)fmax(0, x * view.step - half) / 3 * 3;
            int bottom = top + side / 3 * 3, right = left + side / 3 * 3;
            int stop_y = fmin(y + reach, labels.h), stop_x = fmin(x + reach, labels.w);
            for (int yy = fmax(y - reach, 0); yy < stop_y; yy++) {
                for (int xx = fmax(x - reach, 0); xx < stop_x; xx++) {
                    density[(size_t)yy * labels.w + xx] = 0;
                }
            }
            int by = bottom < h ? bottom : h, rx = right < w ? right : w;
            if (by - top < half || rx - left < half) {
                continue;
            }
            AtlasRegion area = {top, bottom, left, right, 0};
            AtlasImage photo =
                atlas_resize(a, atlas_crop(a, shown, top, by, left, rx), 800, 800, 0);
            AtlasImage raw = atlas_resize(a, atlas_face(a, &view, area), 800, 800, 0);
            AtlasImage edged = atlas_outline(a, photo, atlas_labels(a, &view, area, -1));
            AtlasImage front = atlas_rgb(a, photo), out = atlas_image(a, 2404, 800, 3);
            atlas_bar(&font, front, 10.0 / 800, 1);
            atlas_paste(out, front, 0, 0);
            atlas_paste(out, atlas_rgb(a, raw), 802, 0);
            atlas_paste(out, edged, 1604, 0);
            const char *file = fmt(a, "%s-window-%d.jpg", name, k);
            const char *path = fmt(a, "%s/%s", atlas, file);
            need(hx_site_save_jpeg(path, out.pixels, out.w, out.h, 92), "cannot write %s", path);
            json_object_array_add(windows, json_object_new_string(fmt(a, "fragments/%s", file)));
        }
        printf("%s %zu\n", name, site_items(windows));
        json_object_array_add(reported, item);
        json_object_put(view.body);
    }
    json_object_array_sort(reported, atlas_fragment_order);
    json_write(fmt(a, "%s/fragments.json", work), reported);
    json_object_put(reported);
    json_object_put(records);
}

static int atlas_scan_order(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int energy = atlas_energy(b->row) - atlas_energy(a->row);
    int voxel =
        atlas_compare_double(site_value(a->row, "voxel_um"), site_value(b->row, "voxel_um"));
    return energy ? energy : voxel ? voxel : (a->order > b->order) - (a->order < b->order);
}

static void atlas_compare(Arena *a, const char *work) {
    json_object *records = atlas_records(a);
    json_object *windows = atlas_load_rows(a, work, "windows", NULL);
    json_object *reported = json_object_new_array(), *groups = json_object_new_object();
    const char *atlas = fmt(a, "%s/website/atlas/compare", OUTPUTS);
    mkdirs(atlas);
    AtlasFont font = {0};
    size_t mark = a->used;
    for (size_t i = 0; i < site_items(records); i++) {
        a->used = mark;
        json_object *record = json_object_array_get_idx(records, i);
        const char *key = fmt(a, "%s/%s", jstr(record, "sample"), jstr(record, "segment"));
        json_object *group = json_object_object_get(groups, key);
        if (!group) {
            group = json_object_new_array();
            json_object_object_add(groups, key, group);
        }
        json_object_array_add(group, json_object_get(record));
    }
    json_object_object_foreach(groups, key, scans) {
        (void)key;
        a->used = mark;
        size_t count = site_items(scans);
        int low = 0, high = 0;
        AtlasRow *sorted = push(a, count * sizeof(*sorted));
        for (size_t i = 0; i < count; i++) {
            json_object *row = json_object_array_get_idx(scans, i);
            int energy = atlas_energy(row);
            low |= energy == 77 || energy == 78;
            high |= energy == 111 || energy == 113 || energy == 116;
            sorted[i] = (AtlasRow){row, i};
        }
        if (!low || !high) {
            continue;
        }
        qsort(sorted, count, sizeof(*sorted), atlas_scan_order);
        AtlasView *views = push(a, count * sizeof(*views)), *first = NULL;
        AtlasShift *shifts = push(a, count * sizeof(*shifts));
        const char *names = "";
        size_t n = 0;
        for (size_t i = 0; i < count; i++) {
            json_object *record = sorted[i].row;
            const char *id = jstr(record, "case_id");
            if (strstr(id, "4.317um") || strstr(id, "2.215um")) {
                continue;
            }
            int energy = atlas_energy(record);
            if (!first && (energy == 77 || energy == 78) && !strstr(id, "1.129um")) {
                first = views + n;
            }
            views[n] = atlas_view(a, work, record, NULL);
            names = fmt(a, "%s%s%d keV", names, n ? " · " : "", energy);
            n++;
        }
        need(first != NULL, "comparison has no reference scan");
        for (size_t i = 0; i < n; i++) {
            shifts[i] = atlas_shift(a, first, views + i);
        }
        const char *sample = jstr(first->record, "sample"),
                   *segment = jstr(first->record, "segment");
        const char *case_id = jstr(first->record, "case_id");
        size_t again = a->used;
        int found = 0;
        for (size_t i = 0; found < 6 && i < site_items(windows); i++) {
            a->used = again;
            json_object *window = json_object_array_get_idx(windows, i);
            if (strcmp(atlas_text(window, "case"), case_id) != 0) {
                continue;
            }
            AtlasRegion area = atlas_region_value(jget(window, "box"));
            AtlasImage out = atlas_image(a, (int)n * 802 - 2, 480, 3);
            size_t hold = a->used;
            for (size_t j = 0; j < n; j++) {
                a->used = hold;
                AtlasView *view = views + j;
                int y = (area.y0 * first->cell + shifts[j].dy) / view->cell;
                int x = (area.x0 * first->cell + shifts[j].dx) / view->cell;
                int h = (area.y1 - area.y0) * first->cell / view->cell;
                int w = (area.x1 - area.x0) * first->cell / view->cell;
                AtlasRegion at = {y, y + h, x, x + w, 0};
                AtlasImage result =
                    atlas_outline(a, atlas_resize(a, atlas_face(a, view, at), 800, 480, 0),
                        atlas_labels(a, view, at, -1));
                if (!j) {
                    atlas_bar(&font, result, (area.x1 - area.x0) * first->cell / 800, 5);
                }
                atlas_paste(out, result, (int)j * 802, 0);
            }
            a->used = hold;
            const char *file = fmt(a, "%s-%s-%02d.jpg", sample, segment, found++);
            const char *path = fmt(a, "%s/%s", atlas, file);
            need(hx_site_save_jpeg(path, out.pixels, out.w, out.h, 90), "cannot write %s", path);
            json_object *row = json_object_new_object();
            json_object_object_add(row, "sample", json_object_new_string(sample));
            json_object_object_add(row, "segment", json_object_new_string(segment));
            json_object_object_add(row, "file", json_object_new_string(fmt(a, "compare/%s", file)));
            json_object_object_add(row, "scans", json_object_new_string(names));
            json_object_array_add(reported, row);
        }
        printf("%s %s %d {", sample, segment, found);
        for (size_t i = 0; i < n; i++) {
            const char *name = jstr(views[i].record, "case_id");
            size_t len = strlen(name);
            printf(
                "%s'%s': %.2f", i ? ", " : "", name + (len > 22 ? len - 22 : 0), shifts[i].score);
            json_object_put(views[i].body);
        }
        puts("}");
    }
    json_write(fmt(a, "%s/compare.json", work), reported);
    json_object_put(groups);
    json_object_put(reported);
    json_object_put(windows);
    json_object_put(records);
}

static void atlas_write_png(Arena *a, const char *path, AtlasImage image) {
    size_t mark = a->used;
    png_image png = {.version = PNG_IMAGE_VERSION,
        .width = image.w,
        .height = image.h,
        .format = image.channels == 1 ? PNG_FORMAT_GRAY : PNG_FORMAT_RGB};
    png_alloc_size_t n = 0;
    need(png_image_write_to_memory(&png, NULL, &n, 0, image.pixels, 0, NULL),
        "cannot size atlas PNG");
    u8 *bytes = push(a, n);
    need(png_image_write_to_memory(&png, bytes, &n, 0, image.pixels, 0, NULL) &&
            hx_write_sheet_file(path, bytes, n),
        "cannot write %s", path);
    a->used = mark;
}

static void atlas_examples_panels(Arena *a, AtlasView *view, json_object *body, AtlasImage ir,
    AtlasImage *result, const char **identified) {
    AtlasRegion box = atlas_box(body);
    int x = (box.x0 + box.x1) * view->step / 2, y = (box.y0 + box.y1) * view->step / 2;
    int half = 1.3 * fmax(jf64(body, "h_mm"), jf64(body, "w_mm")) / view->cell / 2;
    AtlasRegion area = {y - half, y + half, x - half, x + half, 0};
    int id = ji64(body, "id");
    AtlasImage first = atlas_rgb(a, atlas_resize(a, atlas_labels(a, view, area, id), 200, 200, 1));
    AtlasImage source;
    if (ir.pixels) {
        int y0 = fmax(0, area.y0), x0 = fmax(0, area.x0);
        int y1 = fmin(ir.h, area.y1), x1 = fmin(ir.w, area.x1);
        source = y1 > y0 && x1 > x0 ? atlas_crop(a, ir, y0, y1, x0, x1) : (AtlasImage){0};
    } else {
        source = atlas_ink_map(a, view, area);
    }
    *identified = ir.pixels ? "infrared photograph"
        : source.pixels     ? "published ML map"
                            : "no published map";
    AtlasImage second = atlas_rgb(a, atlas_resize(a, source, 200, 200, 0));
    if (!source.w || !source.h) {
        for (size_t i = 0; i < 200 * 200; i++) {
            second.pixels[i * 3] = second.pixels[i * 3 + 1] = 24;
            second.pixels[i * 3 + 2] = 26;
        }
    }
    AtlasImage third = atlas_rgb(a, atlas_resize(a, atlas_face(a, view, area), 200, 200, 0));
    *result = atlas_image(a, 604, 200, 3);
    atlas_paste(*result, first, 0, 0);
    atlas_paste(*result, second, 202, 0);
    atlas_paste(*result, third, 404, 0);
}

static int atlas_match_order(const void *left, const void *right) {
    const AtlasRow *a = left, *b = right;
    int score = atlas_compare_double(site_value(b->row, "score"), site_value(a->row, "score"));
    return score ? score : (a->order > b->order) - (a->order < b->order);
}

static size_t atlas_spread(
    AtlasRow *sorted, size_t count, int letter, int mode, json_object **out, size_t limit) {
    json_object *groups = json_object_new_object();
    for (size_t i = 0; i < count; i++) {
        json_object *row = sorted[i].row, *eye = json_object_object_get(row, "eye");
        int best = ji64(row, "best");
        int picked = mode == 2
            ? !eye && best == letter && site_value(row, "margin") > .02 &&
                !json_object_get_boolean(json_object_object_get(row, "blob"))
            : eye && json_object_get_int(eye) == letter && ((best == letter) == (mode == 0));
        if (picked) {
            const char *scroll = atlas_text(row, "sample");
            json_object *items = json_object_object_get(groups, scroll);
            if (!items) {
                items = json_object_new_array();
                json_object_object_add(groups, scroll, items);
            }
            json_object_array_add(items, json_object_get(row));
        }
    }
    size_t used = 0, n = 1;
    for (size_t i = 0; used < limit && n; i++) {
        n = 0;
        json_object_object_foreach(groups, scroll, items) {
            (void)scroll;
            if (i < site_items(items) && used < limit) {
                out[used++] = json_object_array_get_idx(items, i);
                n++;
            }
        }
    }
    json_object_put(groups);
    return used;
}

static json_object *atlas_body_by_id(json_object *rows, int id) {
    for (size_t i = 0; i < site_items(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        if (ji64(row, "id") == id) {
            return row;
        }
    }
    die("atlas letter body %d was not found", id);
    return NULL;
}

static void atlas_alphabet_examples(Arena *a, const char *work) {
    json_object *records = atlas_records(a), *by_case = json_object_new_object();
    json_object *found = json_read(fmt(a, "%s/matches.json", work)),
                *bodies = jget(found, "bodies");
    json_object *table = json_object_new_array_ext(ATLAS_LETTERS),
                *wanted = json_object_new_object();
    const char *atlas = fmt(a, "%s/website/atlas", OUTPUTS);
    const char *folder = fmt(a, "%s/alphabet", atlas);
    mkdirs(folder);
    for (size_t i = 0; i < site_items(records); i++) {
        json_object *record = json_object_array_get_idx(records, i);
        json_object_object_add(by_case, jstr(record, "case_id"), json_object_get(record));
    }
    size_t count = site_items(bodies);
    AtlasRow *sorted = push(a, count * sizeof(*sorted));
    for (size_t i = 0; i < count; i++) {
        json_object *body = json_object_array_get_idx(bodies, i);
        json_object *record = json_object_object_get(by_case, jstr(body, "case"));
        json_object_object_add(body, "sample", json_object_get(jget(record, "sample")));
        json_object_object_add(body, "segment", json_object_get(jget(record, "segment")));
        json_object_object_add(body, "energy", json_object_new_int(atlas_energy(record)));
        sorted[i] = (AtlasRow){body, i};
    }
    qsort(sorted, count, sizeof(*sorted), atlas_match_order);
    AtlasImage strip = atlas_image(a, 12 * 92, 220, 3);
    AtlasImage sprite = atlas_grey(a, fmt(a, "%s/viewer/assets/atlas_bookhand.png", HOME));
    size_t mark = a->used;
    int model_index = 0;
    for (int k = 0; k < ATLAS_LETTERS; k++) {
        a->used = mark;
        const AtlasLetter *letter = atlas_letters + k;
        json_object *selected[20];
        size_t confirmed = atlas_spread(sorted, count, k, 0, selected, 14);
        if (!confirmed) {
            confirmed = atlas_spread(sorted, count, k, 1, selected, 14);
        }
        size_t extra =
            confirmed < 7 ? atlas_spread(sorted, count, k, 2, selected + confirmed, 6) : 0;
        size_t matched = 0, eyes = 0;
        for (size_t i = 0; i < count; i++) {
            json_object *row = sorted[i].row, *eye = json_object_object_get(row, "eye");
            matched += ji64(row, "best") == k;
            eyes += eye && json_object_get_int(eye) == k;
        }
        json_object *entry = json_object_new_object(), *models = json_object_new_array();
        json_object *examples = json_object_new_array();
        json_object_object_add(entry, "name", json_object_new_string(letter->name));
        json_object_object_add(entry, "capital", json_object_new_string(letter->capital));
        json_object_object_add(entry, "small", json_object_new_string(letter->small));
        json_object_object_add(entry, "note", json_object_new_string(letter->note));
        json_object_object_add(entry, "models", models);
        json_object_object_add(entry, "matched", json_object_new_int64(matched));
        json_object_object_add(entry, "eye", json_object_new_int64(eyes));
        json_object_object_add(entry, "examples", examples);
        for (size_t i = 0; i < confirmed + extra; i++) {
            json_object *row = selected[i];
            const char *case_id = jstr(row, "case");
            json_object *items = json_object_object_get(wanted, case_id);
            if (!items) {
                items = json_object_new_array();
                json_object_object_add(wanted, case_id, items);
            }
            json_object_object_add(
                row, "state", json_object_new_string(i < confirmed ? "confirmed" : "unconfirmed"));
            json_object_array_add(examples, json_object_get(row));
            json_object_array_add(items, json_object_get(row));
        }
        AtlasImage first = {0};
        for (int i = 0; i < letter->models; i++) {
            int y = (model_index / 8) * 120, x = (model_index % 8) * 120;
            AtlasImage model = atlas_crop(a, sprite, y, y + 120, x, x + 120);
            model_index++;
            const char *name = fmt(a, "alphabet/model-%s-%d.png", letter->name, i);
            atlas_write_png(a, fmt(a, "%s/%s", atlas, name), model);
            json_object_array_add(models, json_object_new_string(name));
            if (!i) {
                first = model;
            }
        }
        if (confirmed) {
            json_object *chosen = selected[0];
            const char *id = jstr(chosen, "case");
            json_object *doc = json_read(fmt(a, "%s/bodies/%s.json", work, id));
            json_object *body = atlas_body_by_id(jget(doc, "rows"), ji64(chosen, "id"));
            Array labels = npz_get(a, fmt(a, "%s/bodies/%s.npz", work, id), "labels");
            first = atlas_image(a, 64, 64, 1);
            atlas_normal(
                a, labels.data, labels.shape[1], atlas_box(body), ji64(body, "id"), first.pixels);
            for (int i = 0; i < ATLAS_PIXELS; i++) {
                first.pixels[i] *= 255;
            }
            json_object_put(doc);
        }
        atlas_paste(
            strip, atlas_rgb(a, atlas_resize(a, first, 90, 90, 0)), (k % 12) * 92, (k / 12) * 110);
        json_object_array_add(table, entry);
    }
    AtlasFont font = {0};
    const u8 capital[] = {200, 198, 190};
    a->used = mark;
    for (int k = 0; k < ATLAS_LETTERS; k++) {
        atlas_glyph_on(&font, strip, k + 4, (k % 12) * 92 + 39, (k / 12) * 110 + 92, capital);
    }
    atlas_write_png(a, fmt(a, "%s/strip.png", folder), strip);
    json_object_object_foreach(wanted, case_id, items) {
        a->used = mark;
        json_object *record = json_object_object_get(by_case, case_id);
        AtlasView view = atlas_view(a, work, record, NULL);
        AtlasImage ir = {0};
        if (!strcmp(atlas_text(record, "truth"), "infrared")) {
            const char *path =
                fmt(a, "%s/datasets/fragments/%s/ir.png", LIBRARY, jstr(record, "segment"));
            AtlasImage all = atlas_grey(a, path);
            AtlasRegion region = atlas_region_value(jget(record, "box"));
            ir = atlas_crop(a, all, region.y0, region.y1, region.x0, region.x1);
        }
        size_t again = a->used;
        for (size_t i = 0; i < site_items(items); i++) {
            a->used = again;
            json_object *item = json_object_array_get_idx(items, i);
            int id = ji64(item, "id");
            json_object *body = atlas_body_by_id(view.rows, id);
            AtlasImage image;
            const char *source;
            atlas_examples_panels(a, &view, body, ir, &image, &source);
            size_t lit = 0;
            for (int y = 0; y < 200; y++) {
                for (int x = 202 * 3; x < 604 * 3; x++) {
                    lit += image.pixels[y * 604 * 3 + x];
                }
            }
            if (lit < 34 * 402 * 3 * 200) {
                continue;
            }
            const char *file = fmt(a, "alphabet/%s-%05d.jpg", case_id, id);
            const char *path = fmt(a, "%s/%s", atlas, file);
            need(hx_site_save_jpeg(path, image.pixels, image.w, image.h, 92), "cannot write %s",
                path);
            json_object_object_add(item, "file", json_object_new_string(file));
            json_object_object_add(item, "source", json_object_new_string(source));
        }
        printf("%s %zu\n", case_id, site_items(items));
        json_object_put(view.body);
    }
    json_object *report = json_object_new_object();
    json_object_object_add(report, "table", table);
    json_object_object_add(report, "checked", json_object_get(jget(found, "checked")));
    json_object_object_add(report, "agreement", json_object_get(jget(found, "agreement")));
    json_object_object_add(report, "bodies", json_object_new_int64(count));
    json_write(fmt(a, "%s/alphabet.json", work), report);
    json_object_put(wanted);
    json_object_put(report);
    json_object_put(found);
    json_object_put(by_case);
    json_object_put(records);
}
int hx_atlas_command(int argc, char **argv) {
    const char *usage =
        "usage: ./run.sh viewer.atlas_build everything|pages|bodies|match_letters|renders|"
        "segment_maps|segment_scores|letter_cards|letter_windows|fragments|compare|"
        "alphabet_examples";
    need(argc == 1, "%s", usage);
    if (!strcmp(argv[0], "pages")) {
        return hx_atlas_pages_command(argc, argv);
    }
    static const struct {
        const char *name;
        void (*run)(Arena *, const char *);
    } stages[] = {
        {"bodies", atlas_bodies},
        {"match_letters", atlas_match_letters},
        {"renders", atlas_renders},
        {"segment_maps", atlas_segment_maps},
        {"segment_scores", atlas_segment_scores},
        {"letter_cards", atlas_letter_cards},
        {"letter_windows", atlas_letter_windows},
        {"fragments", atlas_fragments},
        {"compare", atlas_compare},
        {"alphabet_examples", atlas_alphabet_examples},
    };
    Arena a = arena(128 * GB);
    const char *work = atlas_work(&a);
    size_t mark = a.used;
    int all = !strcmp(argv[0], "everything"), ran = 0;
    for (size_t i = 0; i < sizeof(stages) / sizeof(*stages); i++) {
        if (all || !strcmp(argv[0], stages[i].name)) {
            stages[i].run(&a, work);
            a.used = mark;
            ran = 1;
        }
    }
    munmap(a.base, a.size);
    need(ran, "%s", usage);
    if (all) {
        char *args[] = {"pages", NULL};
        return hx_atlas_pages_command(1, args);
    }
    return 0;
}
