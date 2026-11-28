typedef struct {
    long number;
    const char *title, *path;
} SitePicture;

static int site_picture_order(const void *a, const void *b) {
    const SitePicture *left = a, *right = b;
    if (left->number != right->number) {
        return left->number < right->number ? -1 : 1;
    }
    int title = strcmp(left->title, right->title);
    return title ? title : strcmp(left->path, right->path);
}

static const char *site_html_escape(Arena *a, const char *text) {
    size_t length = strlen(text);
    char *escaped = push(a, length * 6 + 1), *out = escaped;
    for (size_t i = 0; i < length; i++) {
        const char *value = NULL;
        switch (text[i]) {
        case '&': value = "&amp;"; break;
        case '<': value = "&lt;"; break;
        case '>': value = "&gt;"; break;
        case '"': value = "&quot;"; break;
        case '\'': value = "&#x27;"; break;
        default: break;
        }
        if (value) {
            size_t n = strlen(value);
            memcpy(out, value, n);
            out += n;
        } else {
            *out++ = text[i];
        }
    }
    *out = 0;
    return escaped;
}

static const char *site_panel_title(Arena *a, const char *page, const char *target) {
    size_t mark = a->used;
    Bytes old = slurp(a, target);
    const char *start = old.data ? strstr((char *)old.data, "<title>") : NULL;
    const char *end = start ? strstr(start + 7, "</title>") : NULL;
    if (!end || memchr(start, '\n', end - start)) {
        a->used = mark;
        return site_html_escape(a, page);
    }
    return fmt(a, "%.*s", (int)(end - (start + 7)), start + 7);
}

static const char *site_title_plain(Arena *a, const char *escaped) {
    const char *names[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&#x27;", "&#39;", "&apos;"};
    const char originals[] = "&<>\"'''";
    char *text = push(a, strlen(escaped) + 1), *out = text;
    while (*escaped) {
        int found = 0;
        for (size_t i = 0; *escaped == '&' && i < sizeof(names) / sizeof(*names); i++) {
            size_t n = strlen(names[i]);
            if (!strncmp(escaped, names[i], n)) {
                *out++ = originals[i];
                escaped += n;
                found = 1;
                break;
            }
        }
        if (!found) {
            *out++ = *escaped++;
        }
    }
    *out = 0;
    return text;
}

static json_object *site_panel_places(Arena *a, const char *folder) {
    int count;
    char **names = list_dir(a, folder, &count);
    json_object *places = json_object_new_array();
    for (int i = 0; i < count; i++) {
        size_t mark = a->used;
        int files;
        char **found = list_dir(a, fmt(a, "%s/%s", folder, names[i]), &files);
        SitePicture *order = push(a, (size_t)files * sizeof(*order));
        size_t used = 0;
        for (int j = 0; j < files; j++) {
            char *end;
            const char *name = found[j], *suffix = strrchr(name, '.');
            long number = strtol(name, &end, 10);
            if (isdigit((unsigned char)*name) && end && *end == '-' && suffix && end + 1 < suffix &&
                (!strcmp(suffix, ".jpg") || !strcmp(suffix, ".png"))) {
                char *text = fmt(a, "%.*s", (int)(suffix - end - 1), end + 1);
                for (char *c = text; *c; c++) {
                    if (*c == '-') {
                        *c = ' ';
                    }
                }
                order[used++] = (SitePicture){
                    .number = number, .title = text, .path = fmt(a, "%s/%s", names[i], name)};
            }
        }
        if (used) {
            qsort(order, used, sizeof(*order), site_picture_order);
            json_object *place = json_object_new_object(), *pictures = json_object_new_array();
            json_object_object_add(place, "name", json_object_new_string(names[i]));
            for (size_t j = 0; j < used; j++) {
                json_object *picture = json_object_new_object();
                json_object_object_add(picture, "title", json_object_new_string(order[j].title));
                json_object_object_add(picture, "src", json_object_new_string(order[j].path));
                json_object_array_add(pictures, picture);
            }
            json_object_object_add(place, "pictures", pictures);
            json_object_array_add(places, place);
        }
        a->used = mark;
    }
    return places;
}

static int site_panel_write(
    const char *target, const char *source, const char *title, json_object *meta) {
    const char *text = json_object_to_json_string_ext(meta, JSON_C_TO_STRING_PLAIN);
    char *page = NULL;
    size_t bytes = 0;
    FILE *out = open_memstream(&page, &bytes);
    if (!out) {
        return 0;
    }
    const char *cursor = source;
    int ok = 1;
    while (*cursor && ok) {
        if (!strncmp(cursor, "__TITLE__", 9)) {
            ok = fputs(title, out) >= 0;
            cursor += 9;
        } else if (!strncmp(cursor, "__META__", 8)) {
            ok = fputs(text, out) >= 0;
            cursor += 8;
        } else {
            ok = fputc(*cursor++, out) != EOF;
        }
    }
    ok = !fclose(out) && ok && hx_write_sheet_file(target, page, bytes);
    free(page);
    return ok;
}

int hx_panel_pages_command(int argc, char **argv) {
    Arena a = arena(64 << 20);
    const char *asset = fmt(&a, "%s/viewer/assets/panel_page.html", HOME);
    char *template = read_page_text(asset);
    need(template != NULL, "could not read %s", asset);
    size_t start = a.used;
    for (int i = 0; i < argc; i++) {
        const char *name = argv[i];
        need(site_name(name), "invalid panel page %s", name);
        char *folder = fmt(&a, "%s/website/sheets/%s", OUTPUTS, name);
        char *target = fmt(&a, "%s/index.html", folder);
        const char *title = site_panel_title(&a, name, target);
        json_object *meta = json_object_new_object();
        json_object_object_add(meta, "title", json_object_new_string(site_title_plain(&a, title)));
        json_object_object_add(meta, "back", json_object_new_string("../index.html"));
        json_object_object_add(meta, "places", site_panel_places(&a, folder));
        need(site_panel_write(target, template, title, meta), "cannot publish panel %s", name);
        json_object_put(meta);
        printf("page %s\n", target);
        a.used = start;
    }
    free(template);
    munmap(a.base, a.size);
    return 0;
}
