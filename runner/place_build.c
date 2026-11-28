#ifndef HX_UNITY
const char hx_usage[] = "Place imported builds and keep the builds and viewers they replace.";
#endif

static void place_say(const char *format, ...) {
    char stamp[32];
    time_t now = time(NULL);
    struct tm local;
    localtime_r(&now, &local);
    strftime(stamp, sizeof(stamp), "%H:%M:%S", &local);
    printf("%s ", stamp);
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    puts("");
    fflush(stdout);
}

static void place_copy(Arena *a, const char *from, const char *to) {
    size_t saved = a->used;
    Bytes b = read_file(a, from);
    struct stat st;
    need(stat(from, &st) == 0, "cannot stat %s", from);
    write_file(to, b.data, b.size);
#ifdef __APPLE__
    struct timespec times[] = {st.st_atimespec, st.st_mtimespec};
#else
    struct timespec times[] = {st.st_atim, st.st_mtim};
#endif
    need(chmod(to, st.st_mode & 0777) == 0 && utimensat(AT_FDCWD, to, times, 0) == 0,
        "cannot keep file mode and timestamps on %s: %s", to, strerror(errno));
    a->used = saved;
}

static void place_move(const char *from, const char *to) {
    struct stat st;
    need(
        lstat(to, &st) && errno == ENOENT, "destination already exists or is inaccessible: %s", to);
    mkparent(to);
    need(!rename(from, to), "cannot place %s in %s: %s", from, to, strerror(errno));
}

static void place_part(Arena *a, const char *page, const char *name, const char *layout) {
    size_t saved = a->used;
    const char *path = fmt(a, "%s/index.html", page);
    Bytes old = read_file(a, path);
    char *meta = (char *)old.data;
    while ((meta = strstr(meta, "const M"))) {
        meta += strlen("const M");
        meta += strspn(meta, " \t\n\r");
        if (*meta == '=') {
            break;
        }
    }
    need(meta != NULL, "no metadata in archived viewer %s", page);
    meta++;
    meta += strspn(meta, " \t\n\r");
    json_tokener *parser = json_tokener_new();
    need(parser != NULL, "cannot allocate viewer metadata parser");
    size_t prefix = (size_t)(meta - (char *)old.data);
    json_object *o = json_tokener_parse_ex(parser, meta, (int)(old.size - prefix));
    size_t old_bytes = json_tokener_get_parse_end(parser);
    need(json_tokener_get_error(parser) == json_tokener_success &&
            json_object_is_type(o, json_type_object),
        "invalid metadata in %s", path);
    json_tokener_free(parser);
    json_object_object_add(o, "scan", json_object_new_string(fmt(a, "part/%s", name)));
    size_t n;
    const char *text = json_object_to_json_string_length(o, JSON_C_TO_STRING_NOSLASHESCAPE, &n);
    size_t size = old.size - old_bytes + n;
    u8 *html = push(a, size);
    memcpy(html, old.data, prefix);
    memcpy(html + prefix, text, n);
    memcpy(html + prefix + n, old.data + prefix + old_bytes, old.size - prefix - old_bytes);
    place_copy(a, layout, fmt(a, "%s/placed.json", page));
    write_file(path, html, size);
    json_object_put(o);
    a->used = saved;
}

static const char *place_archive(
    Arena *a, const char *sheets, const char *build, const char *scroll, const char *scan) {
    const char *layout = fmt(a, "%s/sheet/layout.json", build);
    struct stat st;
    need(!stat(exists(layout) ? layout : build, &st), "cannot date build %s", build);
    struct tm local;
    localtime_r(&st.st_mtime, &local);
    char date[32];
    strftime(date, sizeof(date), "%Y-%m-%d", &local);
    const char *target =
        fmt(a, "%s/archive/%s-build/unwrapping/level/%s/%s", OUTPUTS, date, scroll, scan);
    const char *key = fmt(a, "level/%s/%s", scroll, scan);
    glob_t g = {0};
    glob(fmt(a, "%s/sheet-%s-*-level", sheets, scroll), 0, NULL, &g);
    const char *from = NULL, *name = NULL;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        json_object *meta = hx_viewer_metadata(fmt(a, "%s/index.html", g.gl_pathv[i]));
        if (meta && !strcmp(key, board_text(meta, "scan"))) {
            need(!from, "multiple published viewers for %s", key);
            from = fmt(a, "%s", g.gl_pathv[i]);
            name = fmt(a, "%s-%s", disk_base(from), date);
        }
        json_object_put(meta);
    }
    globfree(&g);
    place_move(build, target);
    if (from) {
        const char *archived = fmt(a, "%s/%s", sheets, name);
        place_move(from, archived);
        const char *placed = fmt(a, "%s/sheet/placed.json", target);
        place_part(
            a, archived, name, exists(placed) ? placed : fmt(a, "%s/sheet/layout.json", target));
        place_say("%s %s: the build of %s archived, its page kept as %s", scroll, scan, date, name);
    } else {
        place_say("%s %s: the build of %s archived", scroll, scan, date);
    }
    return name;
}

static void place_card(Arena *a, const char *sheets, const char *page, const char *scroll) {
    json_object *entries = site_registered(a, sheets);
    const char *folder = fmt(a, "%s/%s", sheets, page);
    int ok = entries && site_card_image(a, folder) &&
        site_save_cards(a, sheets, entries, page, scroll, NULL, NULL, 0);
    json_object_put(entries);
    need(ok, "cannot register page %s", page);
}

int place_build_command(int argc, char **argv) {
    if (!argc || (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")))) {
        puts("runner.place_build STAGING SCROLL [SCROLL ...] [--model NAME]");
        return !argc;
    }
    const char *staging = argv[0], *model = NULL;
    char **scrolls = calloc(argc, sizeof(*scrolls));
    need(scrolls != NULL, "cannot allocate scrolls to place");
    int count = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) {
            i++;
            need(i < argc && site_name(argv[i]), "invalid --model name");
            model = argv[i];
        } else {
            need(site_name(argv[i]), "invalid scroll name %s", argv[i]);
            scrolls[count++] = argv[i];
        }
    }
    need(count > 0, "runner.place_build needs one or more scrolls");
    Arena a = arena(128 << 20);
    const char *sheets = fmt(&a, "%s/website/sheets", OUTPUTS);
    mkdirs(sheets);
    int fd = lock_file(fmt(&a, "%s/.publish.lock", sheets), 1, 1);
    need(fd >= 0, "cannot hold the publish lock");
    for (int i = 0; i < count; i++) {
        const char *scroll = scrolls[i],
                   *source = fmt(&a, "%s/unwrapping/level/%s", staging, scroll);
        int n;
        need(exists(source), "no staged builds for %s", scroll);
        char **scans = list_dir(&a, source, &n);
        for (int s = 0; s < n; s++) {
            const char *scan = scans[s], *build = fmt(&a, "%s/%s", source, scan);
            struct stat st;
            if (stat(build, &st) || !S_ISDIR(st.st_mode)) {
                continue;
            }
            const char *into = fmt(&a, "%s/unwrapping/level/%s/%s", OUTPUTS, scroll, scan);
            if (model) {
                const char *receipt = fmt(&a, "%s/sheet/review.json", build);
                if (exists(receipt)) {
                    need(exists(fmt(&a, "%s/sheet", into)), "no live sheet at %s", into);
                    place_copy(&a, receipt, fmt(&a, "%s/sheet/review-%s.json", into, model));
                }
                continue;
            }
            if (exists(into)) {
                const char *archived = place_archive(&a, sheets, into, scroll, scan);
                if (archived) {
                    place_card(&a, sheets, archived, scroll);
                }
            }
            place_move(build, into);
            const char *nodes = fmt(&a, "%s/nodes", into);
            if (lstat(nodes, &st) && errno == ENOENT &&
                exists(fmt(&a, "%s/unwrapping/nodes/%s/%s/nodes", OUTPUTS, scroll, scan))) {
                need(!symlink(fmt(&a, "../../../nodes/%s/%s/nodes", scroll, scan), nodes),
                    "cannot link %s: %s", nodes, strerror(errno));
            }
        }
        glob_t pages = {0};
        glob(fmt(&a, "%s/website/sheets/sheet-%s-*", staging, scroll), 0, NULL, &pages);
        for (size_t p = 0; p < pages.gl_pathc; p++) {
            const char *base = disk_base(pages.gl_pathv[p]);
            const char *name = model ? fmt(&a, "%s-%s", base, model) : base;
            place_move(pages.gl_pathv[p], fmt(&a, "%s/%s", sheets, name));
            if (model) {
                place_card(&a, sheets, name, scroll);
            }
            place_say("%s: page %s placed", scroll, name);
        }
        globfree(&pages);
    }
    json_object *registered = site_registered(&a, sheets);
    need(site_home(&a, sheets, registered, 0), "cannot rewrite start page");
    json_object_put(registered);
    place_say("start page rewritten");
    close(fd);
    munmap(a.base, a.size);
    free(scrolls);
    return 0;
}
