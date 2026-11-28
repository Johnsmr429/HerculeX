const char hx_usage_sheet[] = "Lay out ready pieces, read their ink, and publish the scroll page.";

static double modified(const char *path) {
    struct stat status;
    if (stat(path, &status) || !S_ISREG(status.st_mode)) {
        return -1;
    }
#ifdef __APPLE__
    struct timespec stamp = status.st_mtimespec;
#else
    struct timespec stamp = status.st_mtim;
#endif
    return stamp.tv_sec + stamp.tv_nsec * 1e-9;
}

static int unit_paths(const char *pieces, glob_t *units) {
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/rows-*", pieces) < 0)
        return 0;
    int found = glob(pattern, 0, NULL, units) == 0 && units->gl_pathc > 0;
    free(pattern);
    if (found)
        return 1;
    globfree(units);
    memset(units, 0, sizeof(*units));
    if (asprintf(&pattern, "%s/band-*", pieces) < 0)
        return 0;
    found = glob(pattern, 0, NULL, units) == 0 && units->gl_pathc > 0;
    free(pattern);
    return found;
}

static const char *name_of(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static int has_record(const char *folder) {
    char *path = path_join(folder, "record.json");
    int found = path && regular(path);
    free(path);
    return found;
}

static int same_rows(const char *base, glob_t *units) {
    if (strncmp(name_of(units->gl_pathv[0]), "rows-", 5) != 0) {
        return 1;
    }
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/nodes/rows-*", base) < 0)
        return 0;
    glob_t nodes = {0};
    int found = glob(pattern, 0, NULL, &nodes) == 0;
    free(pattern);
    int same = found && nodes.gl_pathc == units->gl_pathc;
    for (size_t i = 0; same && i < nodes.gl_pathc; i++) {
        same = !strcmp(name_of(units->gl_pathv[i]), name_of(nodes.gl_pathv[i]));
    }
    globfree(&nodes);
    return same;
}

static int sheet_ready(const char *base, glob_t *units) {
    char *pieces = path_join(base, "pieces");
    int ok = pieces && unit_paths(pieces, units);
    for (size_t i = 0; ok && i < units->gl_pathc; i++)
        ok = has_record(units->gl_pathv[i]);
    ok = ok && same_rows(base, units);
    int framed = ok && units->gl_pathc > 1 && !strncmp(name_of(units->gl_pathv[0]), "rows-", 5);
    if (framed) {
        char *joints = path_join(pieces, "joints");
        char *frame = path_join(pieces, "frame");
        ok = joints && frame && has_record(joints) && has_record(frame);
        char *record = frame ? path_join(frame, "record.json") : NULL;
        double frame_time = record ? modified(record) : -1;
        free(record);
        for (size_t i = 0; ok && i < units->gl_pathc; i++) {
            record = path_join(units->gl_pathv[i], "record.json");
            ok = record && modified(record) <= frame_time;
            free(record);
        }
        free(frame);
        free(joints);
    }
    free(pieces);
    return ok;
}

static int layout_needed(const char *base, glob_t *units, int *framed) {
    *framed = units->gl_pathc > 1 && !strncmp(name_of(units->gl_pathv[0]), "rows-", 5);
    char *layout = NULL;
    if (asprintf(&layout, "%s/sheet/layout.json", base) < 0)
        return 0;
    double product = modified(layout), newest = -1;
    for (size_t i = 0; i < units->gl_pathc; i++) {
        char *record = path_join(units->gl_pathv[i], "record.json");
        double stamp = record ? modified(record) : -1;
        if (stamp > newest)
            newest = stamp;
        free(record);
    }
    int pieces_changed = newest > product;
    if (pieces_changed) {
        char *joints = NULL;
        if (asprintf(&joints, "%s/sheet/joints.json", base) >= 0)
            unlink(joints);
        free(joints);
    }
    int needed = pieces_changed;
    if (!needed && *framed) {
        char *record = NULL;
        if (asprintf(&record, "%s/pieces/frame/record.json", base) >= 0) {
            needed = modified(record) > product;
        }
        free(record);
    }
    free(layout);
    return needed;
}

static size_t pieces_in(const char *base) {
    char *layout = NULL;
    if (asprintf(&layout, "%s/sheet/layout.json", base) < 0)
        return 0;
    json_object *document = json_object_from_file(layout);
    free(layout);
    json_object *pieces = json_object_object_get(document, "pieces");
    size_t count =
        json_object_is_type(pieces, json_type_array) ? json_object_array_length(pieces) : 0;
    json_object_put(document);
    return count;
}

static int laid_out(const char *base, const char *sample, const char *scan, glob_t *units) {
    int framed;
    if (!layout_needed(base, units, &framed))
        return 0;
    char *arguments[7] = {"./run.sh",
        framed ? "level.level.scroll_layout" : "level.layout.mosaic_layout", (char *)sample,
        (char *)scan, NULL, NULL, NULL};
    if (!strncmp(name_of(units->gl_pathv[0]), "band-", 5))
        arguments[4] = "band";
    return run_stage_span(arguments, "layout", scan);
}

static int run_scan(
    const char *base, const char *sample, const char *scan, const char *mode, glob_t *units) {
    printf("%s %s %s (level)\n", mode, sample, scan);
    fflush(stdout);
    int code = laid_out(base, sample, scan, units);
    if (code || !strcmp(mode, "layout"))
        return code;
    size_t pieces = pieces_in(base);
    if (!pieces) {
        printf("%s %s: %s\n", sample, scan,
            !strcmp(mode, "ink") ? "no laid-out pieces to read" : "no pieces to publish");
        return 0;
    }
    char *read[] = {"./run.sh", "ink", (char *)sample, (char *)scan, NULL};
    int for_page = !strcmp(mode, "sheet");
    const char *for_publication = getenv("HERCULEX_FOR_PUBLICATION");
    int saved = 0;
    if (for_page || (for_publication && !strcmp(for_publication, "1"))) {
        char *current[] = {"./run.sh", "publish", "--current", (char *)sample, (char *)scan, NULL};
        saved = run(current) == 0;
    }
    code = saved ? 0 : run(read);
    if (code || !for_page)
        return code;
    const char *site = getenv("HERCULEX_SITE");
    char *publish[] = {"./run.sh", "runner.review_page", (char *)sample, (char *)scan, "--url",
        (char *)(site && *site ? site : "http://127.0.0.1:8765"), NULL};
    return run_stage_span(publish, "publish", scan);
}

static void sheet_usage(FILE *out) {
    fputs("Usage: ./run.sh runner.sheet_stage {layout|ink|sheet} [SCROLL ...] [--scan ID]\n", out);
}

static int sheet_run(
    Roots *roots, const char *mode, char **samples, int count, const char *wanted_scan) {
    setenv("MallocLargeCache", "0", 1);
    char *pattern = NULL;
    int *ready_samples = calloc(count ? (size_t)count : 1, sizeof(*ready_samples));
    if (!ready_samples || asprintf(&pattern, "%s/*/*/pieces", roots->route) < 0) {
        free(ready_samples);
        return 1;
    }
    glob_t scans = {0};
    int found = glob(pattern, 0, NULL, &scans) == 0;
    free(pattern);
    int code = 0;
    if (found) {
        for (size_t i = 0; i < scans.gl_pathc; i++) {
            char *pieces = scans.gl_pathv[i];
            if (!directory(pieces))
                continue;
            char *base = strndup(pieces, strlen(pieces) - strlen("/pieces"));
            char *slash = base ? strrchr(base, '/') : NULL;
            if (!slash) {
                free(base);
                continue;
            }
            char *scan = strdup(slash + 1);
            *slash = 0;
            char *sample = strdup(name_of(base));
            *slash = '/';
            if (!scan || !sample) {
                free(sample);
                free(scan);
                free(base);
                code = 1;
                break;
            }
            if (!selected(sample, samples, count) ||
                (wanted_scan && strcmp(scan, wanted_scan) != 0)) {
                free(sample);
                free(scan);
                free(base);
                continue;
            }
            glob_t units = {0};
            if (!sheet_ready(base, &units)) {
                globfree(&units);
                free(sample);
                free(scan);
                free(base);
                continue;
            }
            for (int k = 0; k < count; k++)
                ready_samples[k] |= !strcmp(sample, samples[k]);
            code = run_scan(base, sample, scan, mode, &units);
            globfree(&units);
            free(sample);
            free(scan);
            free(base);
            if (code)
                break;
        }
    }
    globfree(&scans);
    if (!code) {
        for (int i = 0; i < count; i++) {
            if (!ready_samples[i]) {
                fprintf(stderr,
                    "HerculeX: no ready pieces in level for %s%s%s%s. Check pipeline status and "
                    "finish the pieces stage for that scan.\n",
                    samples[i], wanted_scan ? " (" : "", wanted_scan ? wanted_scan : "",
                    wanted_scan ? ")" : "");
                code = 1;
            }
        }
    }
    free(ready_samples);
    return code;
}

int hx_sheet_main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        sheet_usage(argc < 2 ? stderr : stdout);
        return argc < 2;
    }
    const char *mode = argv[1];
    if (strcmp(mode, "layout") != 0 && strcmp(mode, "ink") != 0 && strcmp(mode, "sheet") != 0) {
        sheet_usage(stderr);
        return 2;
    }
    char **samples = calloc((size_t)argc, sizeof(*samples));
    int count = 0;
    const char *wanted_scan = NULL;
    for (int i = 2; samples && i < argc; i++) {
        if (!strcmp(argv[i], "--scan") && i + 1 < argc)
            wanted_scan = argv[++i];
        else if (!strcmp(argv[i], "--since") && i + 1 < argc)
            i++;
        else if (strcmp(argv[i], "--band") != 0)
            samples[count++] = argv[i];
    }
    if (!samples)
        return 1;
    Roots roots;
    if (!route_open(&roots)) {
        free(samples);
        return 1;
    }
    int code = sheet_run(&roots, mode, samples, count, wanted_scan);
    release_roots(&roots);
    free(samples);
    return code;
}
