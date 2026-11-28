#ifndef HX_UNITY
const char hx_usage[] = "Check score bounds using perturbed surfaces made from a control.";
#endif

static const char *const bench_cases[] = {
    "oracle-w", "oracle-int", "oracle-local", "shift80", "offby1", "switch"};

typedef struct {
    int index;
    const char *metric;
    double low, high;
} BenchBound;

static const BenchBound bench_bounds[] = {
    {0, "truth_recall", 0.999, 1},
    {0, "winding.right_share", 0.999, 1},
    {0, "area_cm2.wrong", 0, 0.01},
    {1, "truth_recall", 0.999, 1},
    {1, "winding.right_share", 0.999, 1},
    {1, "area_cm2.wrong", 0, 0.01},
    {2, "truth_recall", 0.999, 1},
    {2, "winding.right_share", 0.999, 1},
    {2, "area_cm2.wrong", 0, 0.01},
    {3, "truth_recall", 0, 0.6},
    {4, "winding.right_share", 0.3, 0.75},
    {5, "winding.right_share", 0.2, 0.8},
    {5, "wrong_join_share", 0.0001, 1},
};

static void bench_check_mesh(
    Arena *a, BenchPieces *p, BenchTruth *t, BenchSurface *s, const char *folder, int mode) {
    size_t saved = a->used;
    int h = s->h, w = s->w;
    i64 n = (i64)h * w;
    f32 *plane = push(a, n * sizeof(*plane));
    const char axes[] = "xyz";
    int wrap = board_integer(s->meta, "wrap_name");
    for (int d = 0; d < 3; d++) {
        for (i64 i = 0; i < n; i++) {
            i32 at = s->index[i];
            const f32 *point = s->grid + 3 * i;
            double value = at >= 0 ? point[d] : -1;
            if (mode == 3 && at >= 0) {
                const f32 *normal = p->normals + 3 * (s->first + at);
                if (!bench_finite(normal)) {
                    value = -1;
                } else {
                    double z = point[2];
                    double x = interpolated(z, t->knots, t->nknots, 1) - point[0];
                    double y = interpolated(z, t->knots, t->nknots, 2) - point[1];
                    double sign = x * normal[0] + y * normal[1] < 0 ? -1 : 1;
                    value += sign * normal[d] * 80 / t->voxel;
                }
            }
            plane[i] = value;
        }
        tiff_write(fmt(a, "%s/%c.tif", folder, axes[d]), plane, w, h);
    }
    if (mode != 1 && mode != 2) {
        need(s->winding != NULL, "mesh %s has no winding image", s->name);
        for (i64 i = 0; i < n; i++) {
            plane[i] = s->winding[i] + (mode == 4 && (wrap % 2 != 0));
        }
        tiff_write(fmt(a, "%s/w.tif", folder), plane, w, h);
    }
    const char *meta = mode == 2 ? "{}" : fmt(a, "{\"winding\": %d}", wrap);
    write_file(fmt(a, "%s/meta.json", folder), meta, strlen(meta));
    a->used = saved;
}

static void bench_check_switch(Arena *a, BenchPieces *p, BenchTruth *t, const char *root) {
    need(p->n >= 2, "the bench switch check needs at least two meshes");
    int first = 0, second = 1;
    for (int i = 1; i < p->n; i++) {
        if (p->surfaces[i].n > p->surfaces[first].n) {
            first = i;
        }
    }
    BenchSurface *x = p->surfaces + first;
    int wrap = board_integer(x->meta, "wrap_name"), found = 0;
    for (int i = 0; i < p->n; i++) {
        if (board_integer(p->surfaces[i].meta, "wrap_name") == wrap + 1) {
            found = 1;
            second = i;
            break;
        }
    }
    if (!found) {
        second = first ? 0 : 1;
        for (int i = 0; i < p->n; i++) {
            if (i != first && p->surfaces[i].n > p->surfaces[second].n) {
                second = i;
            }
        }
    }
    BenchSurface *y = p->surfaces + second;
    int h = x->h < y->h ? x->h : y->h, w = x->w < y->w ? x->w : y->w;
    i64 valid_y = 0, left = 0, least = INT64_MAX;
    int half = 0;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            valid_y += y->index[(i64)r * y->w + c] >= 0;
        }
    }
    for (int c = 0; c < w; c++) {
        for (int r = 0; r < h; r++) {
            left += x->index[(i64)r * x->w + c] >= 0;
        }
        i64 difference = llabs(left - valid_y);
        if (difference < least) {
            least = difference;
            half = c;
        }
        for (int r = 0; r < h; r++) {
            valid_y -= y->index[(i64)r * y->w + c] >= 0;
        }
    }
    i64 n = (i64)h * w;
    f32 *grid = push(a, n * 3 * sizeof(*grid)), *wind = push(a, n * sizeof(*wind));
    i32 *mask = push(a, n * sizeof(*mask));
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            i64 dst = (i64)r * w + c;
            BenchSurface *src = c < half ? x : y;
            i64 at = (i64)r * src->w + c;
            mask[dst] = src->index[at];
            memcpy(grid + 3 * dst, src->grid + 3 * at, 3 * sizeof(*grid));
            wind[dst] = x->winding[(i64)r * x->w + c];
        }
    }
    BenchSurface mixed = {
        .grid = grid, .index = mask, .h = h, .w = w, .winding = wind, .meta = x->meta};
    bench_check_mesh(a, p, t, &mixed, fmt(a, "%s/switch/%s", root, x->name), 5);
}

static void bench_check_make(Arena *a, const BenchOptions *o, const char *root) {
    size_t saved = a->used;
    const char *input = fmt(a, "%s/bench/controls/%s", OUTPUTS, o->control);
    BenchTruth t = bench_truth(a, input);
    json_object *source = json_object_new_array();
    json_object_array_add(source, json_object_new_string(fmt(a, "%s/meshes", input)));
    json_object *paths = bench_paths(a, source);
    BenchPieces p = bench_pieces(a, paths, &t, o);
    for (int i = 0; i < p.n; i++) {
        BenchSurface *piece = &p.surfaces[i];
        for (int c = 0; c < 5; c++) {
            bench_check_mesh(
                a, &p, &t, piece, fmt(a, "%s/%s/%s", root, bench_cases[c], piece->name), c);
        }
    }
    bench_check_switch(a, &p, &t, root);
    for (int i = 0; i < p.n; i++) {
        json_object_put(p.surfaces[i].meta);
    }
    json_object_put(t.record);
    json_object_put(paths);
    json_object_put(source);
    a->used = saved;
}

int bench_score_check_command(int argc, char **argv) {
    const char *control = NULL;
    int keep = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("calibrate.bench.score_check [CONTROL] [--keep]");
            return 0;
        }
        if (!strcmp(argv[i], "--keep")) {
            keep = 1;
        } else {
            need(!control && site_name(argv[i]), "invalid score check argument %s", argv[i]);
            control = argv[i];
        }
    }
    Arena a = arena(64 * GB);
    BenchOptions o = {.control = control ? control : "PHerc0139-dev",
        .zones = 1,
        .low = -INFINITY,
        .high = INFINITY};
    const char *scratch = env("HERCULEX_SCRATCH", fmt(&a, "%s/runtime/scratch", HOME));
    char *root = fmt(&a, "%s/score-check/%s.XXXXXXXX", scratch, o.control);
    mkparent(root);
    need(mkdtemp(root) != NULL, "cannot create score check folder");
    bench_check_make(&a, &o, root);
    json_object *zones = json_object_new_array();
    json_object_array_add(zones, json_object_new_string("dev"));
    int failed = 0;
    for (int c = 0; c < 6; c++) {
        size_t saved = a.used;
        o.label = fmt(&a, "check-%s", bench_cases[c]);
        o.folder = fmt(&a, "%s/score-%s", root, bench_cases[c]);
        json_object *source = json_object_new_array();
        json_object_array_add(
            source, json_object_new_string(fmt(&a, "%s/%s", root, bench_cases[c])));
        json_object *paths = bench_paths(&a, source);
        json_object *summary = bench_score(&a, &o, paths, zones);
        json_object *row = json_object_new_object();
        json_object_object_add(row, "case", json_object_new_string(bench_cases[c]));
        for (size_t i = 0; i < sizeof(bench_bounds) / sizeof(*bench_bounds); i++) {
            const BenchBound *b = &bench_bounds[i];
            if (b->index != c) {
                continue;
            }
            const char *dot = strchr(b->metric, '.');
            json_object *value = dot
                ? jget(jget(summary, fmt(&a, "%.*s", (int)(dot - b->metric), b->metric)), dot + 1)
                : jget(summary, b->metric);
            json_object_object_add(row, b->metric, json_object_get(value));
            double v = json_object_get_double(value);
            if (!value || !(v >= b->low && v <= b->high)) {
                fprintf(stderr, "%s: %s = %s outside [%g, %g]\n", bench_cases[c], b->metric,
                    value ? json_object_get_string(value) : "None", b->low, b->high);
                failed++;
            }
        }
        puts(board_json(row));
        fflush(stdout);
        json_object_put(row);
        json_object_put(summary);
        json_object_put(paths);
        json_object_put(source);
        a.used = saved;
    }
    if (keep) {
        printf("kept score check in %s\n", root);
    } else {
        remove_tree(root);
    }
    if (!failed) {
        puts("all bounds hold");
    }
    json_object_put(zones);
    munmap(a.base, a.size);
    return !!failed;
}
