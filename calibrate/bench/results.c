#ifndef HX_UNITY
const char hx_usage[] = "List, inspect or compare the bench's recorded metrics.";
#endif

typedef struct {
    const char *name, *system, *kind, *method, *metric, *about;
} BenchColumn;

static const BenchColumn bench_columns[] = {
    {"L-W", "network", "winding", NULL, "share_frames",
        "network W: share of truth on its frame's modal offset"},
    {"L-Wcov", "network", "winding", NULL, "coverage",
        "network W: share of the box's truth with a valid field"},
    {"L-adj", "network", "winding", NULL, "adjacent_exact",
        "network W: radially adjacent truth wraps one apart in W"},
    {"L-cm2", "network", "surface", NULL, "correct_cm2",
        "network sheets: cm2 on their dominant wrap"},
    {"L-wrong", "network", "surface", NULL, "wrong_rate", "network sheets: wrong-wrap area rate"},
    {"L-big", "network", "surface", NULL, "largest_correct_cm2",
        "network sheets: largest connected correct cm2"},
    {"gF", "v1", "winding", "grown-field", "share_frames",
        "v1 grown field kept whole, at nearest node, whole band"},
    {"lP-wr", "v1", "surface", "level-pieces", "wrong_rate",
        "v1 level pieces (ink calibration): wrong-wrap rate"},
    {"lP-big", "v1", "surface", "level-pieces", "largest_correct_cm2",
        "v1 level pieces: largest correct cm2"},
    {"gP-wr", "v1", "surface", "grown-pieces", "wrong_rate",
        "v1 grown pieces (ink calibration): wrong rate"},
    {"gP-big", "v1", "surface", "grown-pieces", "largest_correct_cm2",
        "v1 grown pieces: largest correct cm2"},
    {"pP-wr", "v1", "surface", "grown-production", "wrong_rate",
        "v1 production grown pieces: wrong rate"},
    {"pP-big", "v1", "surface", "grown-production", "largest_correct_cm2",
        "v1 production grown: largest cm2"},
};

static const BenchColumn bench_recorded[] = {
    {"ws-wr", "v1", "recorded", "wrap-study-patches", "usable.wrong_rate",
        "wrap study: wrong rate in pieces >=1 cm2"},
    {"ws-cm2", "v1", "recorded", "wrap-study-patches", "usable.correct_cm2",
        "wrap study: truth cm2 judged right in pieces >=1 cm2"},
    {"ws-big", "v1", "recorded", "wrap-study-patches", "largest_cm2",
        "wrap study: largest piece cm2 (claimed, not checked)"},
    {"rr-lvl", "v1", "recorded", "ruled-roll-level-route", "modal_share",
        "ruled-roll level route: one-offset share (slab)"},
};

static const char *const bench_slabs[][2] = {
    {"roll-frame-planes-", "ruled-roll-p50-frame-planes-"},
    {"level-field", "ruled-roll-level-frame"},
    {"grown-field", "ruled-roll-grown-field"},
};

static const char bench_footing[] =
    "Footing. Table 1 is one footing: the bench's truth (names where wrap names exist, else v1's "
    "groups) and the\n"
    "bench's metrics, for the winding network and for v1's stored outputs alike; only these "
    "columns compare the two. Its winding\n"
    "numbers count every truth vertex in the box, pressed ones included; its surface numbers are "
    "area on truth, the modal\n"
    "wrap per 4-connected part. Table 2 puts slab and whole-band fields on v1's own footing "
    "(groups truth, resolved\n"
    "vertices of the largest group, as ideas/2026-11-12-ruled-roll quick/error_map) beside v1's "
    "recorded number: they\n"
    "reproduce v1, they are not comparable with table 1. Table 3 is v1's records as written: the "
    "wrap study re-grew its\n"
    "own pieces and counted truth vertices judged per piece and group with pressed truth left out; "
    "none of it compares\n"
    "with any bench column.\n";

static const char *bench_split(Arena *a, json_object *manifest, const char *control) {
    json_object *item = json_object_object_get(jget(manifest, "controls"), control);
    if (item) {
        control = jstr(item, "scan");
    }
    const char *first = strchr(control, '-');
    const char *end = first ? strchr(first + 1, '-') : NULL;
    const char *name = end ? fmt(a, "%.*s", (int)(end - control), control) : control;
    json_object *scan = json_object_object_get(jget(manifest, "scans"), name);
    if (!scan) {
        return "excluded";
    }
    json_object *zones = jget(scan, "zones");
    int dev = 0, train = 1;
    for (int i = 0; i < board_count(zones); i++) {
        const char *zone = board_item(json_object_array_get_idx(zones, i), 2);
        dev |= !strcmp(zone, "dev");
        train &= !strcmp(zone, "train") || !strcmp(zone, "val");
    }
    return train ? "train" : dev ? "mixed" : "holdout-A";
}

static const char *bench_truth_source(Arena *a, const char *control) {
    const char *names = fmt(a, "%s/controls/sheets/%s/truth.npz", LIBRARY, control);
    const char *group = fmt(a, "%s/calibration/geometry_wrap/truth/%s.npz", LIBRARY, control);
    int groups = exists(group);
    if (!exists(names)) {
        return groups ? "groups" : "-";
    }
    if (!groups) {
        return "names";
    }
    size_t saved = a->used;
    Array x = npz_get(a, names, "names");
    int width = (int)strtol(x.descr + 2, NULL, 10), named = 0;
    char *text = push(a, width + 1);
    for (i64 i = 0; i < elements(x) && !named; i++) {
        for (int j = 0; j < width; j++) {
            text[j] = x.descr[1] == 'U' ? ((u32 *)x.data)[i * width + j]
                                        : ((const char *)x.data)[i * width + j];
        }
        text[width] = 0;
        for (char *at = text; (at = strstr(at, "-w")); at += 2) {
            size_t digits = strspn(at + 2, "0123456789");
            if (digits && at[2 + digits] == '_') {
                named = 1;
                break;
            }
        }
    }
    a->used = saved;
    return named ? "names" : "groups";
}

static i64 bench_latest(Arena *a, Board *b, const char *control, const char *system,
    const char *kind, const char *method, const char **truth) {
    const char *q = "SELECT id,parameters FROM runs WHERE control=%Q AND system=%Q AND kind=%Q";
    const char *order = " ORDER BY created DESC,id DESC LIMIT 1";
    sqlite3_stmt *s;
    if (method && bench_file(method, "*")) {
        const char *prefix = fmt(a, "%.*s%%", (int)strlen(method) - 1, method);
        const char *prepared = fmt(a, "%s AND method LIKE %%Q AND method NOT LIKE %%Q%s", q, order);
        s = board_query(b, prepared, control, system, kind, prefix, "%@%");
    } else if (method) {
        s = board_query(b, fmt(a, "%s AND method=%%Q%s", q, order), control, system, kind, method);
    } else {
        s = board_query(b, fmt(a, "%s%s", q, order), control, system, kind);
    }
    i64 id = 0;
    if (sqlite3_step(s) == SQLITE_ROW) {
        id = sqlite3_column_int64(s, 0);
        if (truth) {
            const char *params = (const char *)sqlite3_column_text(s, 1);
            json_object *o = params ? json_tokener_parse(params) : NULL;
            const char *value = board_text(o, "truth");
            *truth = *value ? fmt(a, "%s", value) : "?";
            json_object_put(o);
        }
    }
    sqlite3_finalize(s);
    return id;
}

static double bench_metric(Board *b, i64 run, const char *key) {
    if (!run) {
        return NAN;
    }
    sqlite3_stmt *q =
        board_query(b, "SELECT value FROM metrics WHERE run=%lld AND key=%Q", (long long)run, key);
    double value = sqlite3_step(q) == SQLITE_ROW && sqlite3_column_type(q, 0) != SQLITE_NULL
        ? sqlite3_column_double(q, 0)
        : NAN;
    sqlite3_finalize(q);
    return value;
}

typedef struct {
    const char **values;
    int n, cap, cols;
    const char *const *head;
} BenchTable;

static void bench_row(BenchTable *t, const char **cells) {
    if (t->n == t->cap) {
        t->cap = t->cap ? 2 * t->cap : 16;
        void *grown = realloc(t->values, t->cap * t->cols * sizeof(*t->values));
        need(grown != NULL, "cannot allocate bench report rows");
        t->values = grown;
    }
    memcpy(t->values + (size_t)t->n++ * t->cols, cells, t->cols * sizeof(*cells));
}

static void bench_table(BenchTable *t) {
    int width[t->cols];
    for (int k = 0; k < t->cols; k++) {
        width[k] = strlen(t->head[k]);
        for (int i = 0; i < t->n; i++) {
            width[k] = fmax(width[k], strlen(t->values[i * t->cols + k]));
        }
    }
    for (int i = -1; i < t->n; i++) {
        for (int k = 0; k < t->cols; k++) {
            const char *value = i < 0 ? t->head[k] : t->values[i * t->cols + k];
            printf("%s%-*s", k ? "  " : "", width[k], value);
        }
        puts("");
    }
    free(t->values);
}

static const char *bench_cell(Arena *a, double value) {
    return isfinite(value) ? fmt(a, fabs(value) < 10 ? "%.3f" : "%.1f", value) : "-";
}

static void bench_slab_rows(Arena *a, Board *b, const char *control, BenchTable *table) {
    int n = sizeof(bench_slabs) / sizeof(*bench_slabs);
    for (int i = 0; i < n; i++) {
        const char *computed = bench_slabs[i][0], *recorded = bench_slabs[i][1];
        sqlite3_stmt *q = board_query(b,
            "SELECT DISTINCT method FROM runs WHERE control=%Q "
            "AND system='v1' AND kind='winding' AND method LIKE %Q ORDER BY method",
            control, fmt(a, "%s%%", computed));
        while (sqlite3_step(q) == SQLITE_ROW) {
            const char *name = fmt(a, "%s", sqlite3_column_text(q, 0)), *truth = "None";
            i64 mine = bench_latest(a, b, control, "v1", "winding", name, &truth);
            int groups = bench_file(name, "@groups") || !strcmp(truth, "groups");
            const char *base = strchr(name, '@');
            size_t length = base ? (size_t)(base - name) : strlen(name);
            const char *target = bench_file(computed, "-")
                ? fmt(a, "%s%.*s", recorded, (int)(length - strlen(computed)),
                      name + strlen(computed))
                : recorded;
            i64 theirs = bench_latest(a, b, control, "v1", "recorded", target, NULL);
            double v1 = groups ? bench_metric(b, mine, "v1_share_largest_frame_resolved") : NAN;
            double old = groups ? bench_metric(b, theirs, "modal_share") : NAN;
            const char *row[] = {control, name, truth,
                bench_cell(a, bench_metric(b, mine, "share_frames")),
                bench_cell(a, bench_metric(b, mine, "coverage")), bench_cell(a, v1),
                bench_cell(a, old),
                isfinite(v1) && isfinite(old) ? fmt(a, "%+.4f", v1 - old) : "-"};
            bench_row(table, row);
        }
        sqlite3_finalize(q);
    }
}

static void bench_report(
    Arena *a, Board *b, const char *split, const char *network, int groups, int sources) {
    json_object *manifest = json_read(fmt(a, "%s/calibrate/bench/controls.json", HOME));
    need(manifest != NULL, "cannot read bench controls");
    glob_t paths = {0};
    glob(fmt(a, "%s/controls/sheets/*/record.json", LIBRARY), 0, NULL, &paths);
    enum {
        cols = sizeof(bench_columns) / sizeof(*bench_columns),
        recorded = sizeof(bench_recorded) / sizeof(*bench_recorded)
    };
    const char *first[3 + cols], *last[1 + recorded];
    first[0] = last[0] = "control";
    first[1] = "split";
    first[2] = "truth";
    for (int i = 0; i < cols; i++) {
        first[i + 3] = bench_columns[i].name;
    }
    for (int i = 0; i < recorded; i++) {
        last[i + 1] = bench_recorded[i].name;
    }
    const char *slabs[] = {
        "control", "v1 field", "truth", "share", "coverage", "v1-footing", "recorded", "diff"};
    BenchTable whole = {.cols = 3 + cols, .head = first};
    BenchTable part = {.cols = 8, .head = slabs};
    BenchTable history = {.cols = 1 + recorded, .head = last};
    for (size_t i = 0; i < paths.gl_pathc; i++) {
        const char *path = paths.gl_pathv[i], *slash = strrchr(path, '/');
        char *parent = fmt(a, "%.*s", (int)(slash - path), path);
        const char *name = bench_basename(parent), *band = bench_split(a, manifest, name);
        if (strcmp(split, "all") != 0 && strcmp(split, band) != 0) {
            continue;
        }
        const char *row[3 + cols], *old[1 + recorded];
        row[0] = old[0] = name;
        row[1] = band;
        row[2] = bench_truth_source(a, name);
        for (int k = 0; k < cols; k++) {
            const BenchColumn *c = &bench_columns[k];
            int v1 = !strcmp(c->system, "v1");
            i64 run = 0;
            if (v1 && groups) {
                run = bench_latest(a, b, name, "v1", c->kind, fmt(a, "%s@groups", c->method), NULL);
            }
            if (!run) {
                run = bench_latest(a, b, name, c->system, c->kind, v1 ? c->method : network, NULL);
            }
            row[k + 3] = bench_cell(a, bench_metric(b, run, c->metric));
        }
        bench_row(&whole, row);
        bench_slab_rows(a, b, name, &part);
        for (int k = 0; k < recorded; k++) {
            const BenchColumn *c = &bench_recorded[k];
            i64 run = bench_latest(a, b, name, c->system, c->kind, c->method, NULL);
            old[k + 1] = bench_cell(a, bench_metric(b, run, c->metric));
        }
        bench_row(&history, old);
    }
    globfree(&paths);
    json_object_put(manifest);
    puts(bench_footing);
    puts("Table 1. Bench footing, whole control (network and v1 comparable)");
    bench_table(&whole);
    for (int i = 0; i < cols; i++) {
        const BenchColumn *c = &bench_columns[i];
        printf("  %-7s %s [%s %s %s %s]\n", c->name, c->about, c->system, c->kind,
            c->method ? c->method : "newest", c->metric);
    }
    puts("\nTable 2. v1 fields: bench footing, v1's footing reproduced, "
         "v1's record (groups truth only)");
    bench_table(&part);
    puts("\nTable 3. v1 records as written (v1's footing; not comparable with tables 1 and 2)");
    bench_table(&history);
    for (int i = 0; i < recorded; i++) {
        const BenchColumn *c = &bench_recorded[i];
        printf("  %-7s %s [%s %s]\n", c->name, c->about, c->method, c->metric);
    }
    if (sources) {
        sqlite3_stmt *q = board_query(b,
            "SELECT DISTINCT method,source FROM runs "
            "WHERE source IS NOT NULL ORDER BY method,source");
        while (sqlite3_step(q) == SQLITE_ROW) {
            printf("source of %s: %s\n", sqlite3_column_text(q, 0), sqlite3_column_text(q, 1));
        }
        sqlite3_finalize(q);
    }
}

int bench_results_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.bench.results report|runs|show ...");
        return !argc;
    }
    Arena a = arena(128 << 20);
    Board b = bench_database(&a);
    if (!strcmp(argv[0], "report")) {
        const char *split = "all", *network = NULL;
        int groups = 0, sources = 0;
        for (int i = 1; i < argc; i++) {
            const char *option = argv[i];
            if (!strcmp(option, "--sources")) {
                sources = 1;
                continue;
            }
            need(++i < argc, "%s needs a value", option);
            if (!strcmp(option, "--split")) {
                split = argv[i];
                need(!strcmp(split, "all") || !strcmp(split, "dev") || !strcmp(split, "holdout"),
                    "unknown split %s", split);
            } else if (!strcmp(option, "--network")) {
                network = argv[i];
            } else if (!strcmp(option, "--v1-truth")) {
                need(!strcmp(argv[i], "groups") || !strcmp(argv[i], "default"),
                    "unknown v1 truth %s", argv[i]);
                groups = !strcmp(argv[i], "groups");
            } else {
                die("unknown bench report option %s", option);
            }
        }
        bench_report(&a, &b, split, network, groups, sources);
    } else if (!strcmp(argv[0], "runs")) {
        need(argc == 1 || (argc == 3 && !strcmp(argv[1], "--control")),
            "calibrate.bench.results runs [--control CONTROL]");
        const char *base = "SELECT id,created,control,system,kind,method,commit_id FROM runs";
        sqlite3_stmt *q = argc == 3
            ? board_query(&b, "%s WHERE control=%Q ORDER BY id", base, argv[2])
            : board_query(&b, "%s ORDER BY id", base);
        while (sqlite3_step(q) == SQLITE_ROW) {
            for (int i = 0; i < 7; i++) {
                const char *value = (const char *)sqlite3_column_text(q, i);
                printf("%s%s", i ? " | " : "", value ? value : "None");
            }
            puts("");
        }
        sqlite3_finalize(q);
    } else if (!strcmp(argv[0], "show")) {
        need(argc == 2, "calibrate.bench.results show RUN");
        long long run = board_int(argv[1]);
        sqlite3_stmt *q = board_query(&b,
            "SELECT id,unit,created,control,split,system,method,kind,"
            "commit_id,parameters,source FROM runs WHERE id=%lld",
            run);
        need(sqlite3_step(q) == SQLITE_ROW, "no run %lld", run);
        const char *keys[] = {"id", "unit", "created", "control", "split", "system", "method",
            "kind", "commit", "parameters", "source"};
        for (int i = 0; i < 11; i++) {
            const char *value = (const char *)sqlite3_column_text(q, i);
            printf("%s: %s\n", keys[i], value ? value : "None");
        }
        sqlite3_finalize(q);
        q = board_query(&b, "SELECT key,value FROM metrics WHERE run=%lld ORDER BY key", run);
        while (sqlite3_step(q) == SQLITE_ROW) {
            printf("  %s = %g\n", sqlite3_column_text(q, 0), sqlite3_column_double(q, 1));
        }
        sqlite3_finalize(q);
    } else {
        die("unknown bench results command %s", argv[0]);
    }
    sqlite3_close(b.db);
    munmap(a.base, a.size);
    return 0;
}
