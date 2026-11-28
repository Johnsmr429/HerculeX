#ifndef HX_UNITY
const char hx_usage[] = "Build, list or verify frozen benchmark controls and source digests.";
#endif

static int bench_hash_of(const char *path, char hex[65]) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    need(context && EVP_DigestInit_ex(context, EVP_sha256(), NULL), "cannot hash %s", path);
    char data[65536];
    int ok = 1;
    for (;;) {
        ssize_t read_bytes = read(fd, data, sizeof(data));
        if (read_bytes < 0 && errno == EINTR) {
            continue;
        }
        if (read_bytes <= 0) {
            ok = read_bytes == 0;
            break;
        }
        need(EVP_DigestUpdate(context, data, read_bytes), "cannot hash %s", path);
    }
    close(fd);
    u8 digest[32];
    need(EVP_DigestFinal_ex(context, digest, NULL), "cannot finish hash of %s", path);
    EVP_MD_CTX_free(context);
    if (!ok) {
        return 0;
    }
    for (int i = 0; i < 32; i++) {
        snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    }
    return 1;
}

static int bench_hash(const char *path, const char *wanted) {
    char hex[65];
    return bench_hash_of(path, hex) && !strcmp(hex, wanted);
}

static json_object *bench_verify(Arena *a, const char *package, const char *name) {
    char *folder = fmt(a, "%s/%s", package, name);
    json_object *record = json_read(fmt(a, "%s/control.json", folder));
    need(record != NULL, "no frozen control %s", folder);
    json_object *files = jget(record, "files"), *sources = jget(record, "sources");
    json_object *bad = json_object_new_array(), *extra = json_object_new_array();
    json_object *changed = json_object_new_array();
    {
        json_object_object_foreach(files, file, hash) {
            size_t saved = a->used;
            if (!bench_hash(fmt(a, "%s/%s", folder, file), json_object_get_string(hash))) {
                json_object_array_add(bad, json_object_new_string(file));
            }
            a->used = saved;
        }
    }
    char *roots[] = {folder, NULL};
    FTS *tree = fts_open(roots, FTS_LOGICAL | FTS_NOCHDIR, NULL);
    need(tree != NULL, "cannot list frozen package %s", folder);
    size_t prefix = strlen(folder) + 1;
    for (FTSENT *p; (p = fts_read(tree));) {
        if (p->fts_info == FTS_F && strcmp(p->fts_name, "control.json") != 0) {
            const char *relative = p->fts_path + prefix;
            if (!json_object_object_get_ex(files, relative, NULL)) {
                json_object_array_add(extra, json_object_new_string(relative));
            }
        }
    }
    fts_close(tree);
    {
        json_object_object_foreach(sources, segment, entry) {
            const char *base = jstr(entry, "folder");
            json_object *hashes = jget(entry, "sha256");
            json_object_object_foreach(hashes, file, hash) {
                size_t saved = a->used;
                if (!bench_hash(fmt(a, "%s/%s", base, file), json_object_get_string(hash))) {
                    json_object_array_add(
                        changed, json_object_new_string(fmt(a, "%s/%s", segment, file)));
                }
                a->used = saved;
            }
        }
    }
    json_object *out = json_object_new_object();
    json_object_object_add(out, "name", json_object_new_string(name));
    json_object_object_add(out, "files", json_object_new_int(json_object_object_length(files)));
    json_object_object_add(out, "bad", bad);
    json_object_object_add(out, "extra", extra);
    json_object_object_add(out, "sources_changed", changed);
    json_object_put(record);
    return out;
}

int bench_package_inspect_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("calibrate.bench.package list|verify [CONTROL ...]");
        return !argc;
    }
    Arena a = arena(32 << 20);
    const char *root = fmt(&a, "%s/bench/controls", OUTPUTS);
    if (!strcmp(argv[0], "list")) {
        need(argc == 1, "calibrate.bench.package list");
        json_object *manifest = json_read(fmt(&a, "%s/calibrate/bench/controls.json", HOME));
        need(manifest != NULL, "cannot read bench controls");
        json_object *entries = jget(manifest, "controls");
        json_object_object_foreach(entries, name, spec) {
            json_object *z = jget(spec, "z");
            i64 first = json_object_get_int64(json_object_array_get_idx(z, 0));
            i64 last = json_object_get_int64(json_object_array_get_idx(z, 1));
            int built = exists(fmt(&a, "%s/%s/control.json", root, name));
            printf("%-16s %-28s z %6lld-%-6lld %s\n", name, jstr(spec, "scan"), (long long)first,
                (long long)last, built ? "built" : "-");
        }
        json_object_put(manifest);
    } else if (!strcmp(argv[0], "verify")) {
        if (argc > 1) {
            for (int i = 1; i < argc; i++) {
                size_t saved = a.used;
                need(site_name(argv[i]), "invalid control name %s", argv[i]);
                json_object *out = bench_verify(&a, root, argv[i]);
                puts(board_json(out));
                json_object_put(out);
                a.used = saved;
            }
        } else {
            glob_t g = {0};
            glob(fmt(&a, "%s/*/control.json", root), 0, NULL, &g);
            for (size_t i = 0; i < g.gl_pathc; i++) {
                size_t saved = a.used;
                const char *path = g.gl_pathv[i], *slash = strrchr(path, '/');
                const char *name = bench_basename(fmt(&a, "%.*s", (int)(slash - path), path));
                json_object *out = bench_verify(&a, root, name);
                puts(board_json(out));
                json_object_put(out);
                a.used = saved;
            }
            globfree(&g);
        }
    } else {
        die("unknown bench package option %s", argv[0]);
    }
    munmap(a.base, a.size);
    return 0;
}

typedef struct {
    BenchSurface raw;
    const char *source, *segment;
    i64 inside, first;
    int lo, hi, wrap, end, frame;
    double offset, dropped, starts, ends;
    double *phi, *radius;
    i32 *truth;
} PackageGrid;

typedef struct {
    const char *name, *sample, *scan, *axis_source;
    json_object *definition, *entry, *sources, *votes;
    PackageGrid *grids;
    int grids_n, sense, knot_count, step, named, anchor;
    double *knots, voxel, low, high, spacing;
    i64 count;
} Package;

static int package_wrap(const char *name, int *last) {
    *last = -1;
    for (const char *p = name; (p = strstr(p, "-w")); p += 2) {
        if (!isdigit((unsigned char)p[2])) {
            continue;
        }
        char *end;
        int a = strtol(p + 2, &end, 10), b = a;
        if (*end == '-' && isdigit((unsigned char)end[1])) {
            b = strtol(end + 1, &end, 10);
        }
        if (!*end || *end == '_') {
            *last = b;
            return a;
        }
    }
    return -1;
}

static int package_name_order(const void *a, const void *b) {
    const PackageGrid *x = a, *y = b;
    return strcmp(x->segment, y->segment);
}

static Package package_open(Arena *a, json_object *manifest, const char *name) {
    Package p = {.name = name};
    p.definition = json_object_object_get(jget(manifest, "controls"), name);
    need(p.definition != NULL, "unknown bench control %s", name);
    p.entry = jget(jget(manifest, "scans"), jstr(p.definition, "scan"));
    p.sample = jstr(p.entry, "sample");
    p.scan = jstr(p.entry, "scan");
    p.voxel = jf64(p.entry, "voxel_um");
    json_object *z = jget(p.definition, "z");
    p.low = json_object_get_double(json_object_array_get_idx(z, 0));
    p.high = json_object_get_double(json_object_array_get_idx(z, 1));
    p.step = ji64(p.definition, "crossing_step");
    need(p.step > 0 && p.voxel > 0 && p.low < p.high, "invalid zone for bench control %s", name);
    const char *root[] = {
        fmt(a, "%s/cache/open_data", OUTPUTS), fmt(a, "%s/published/open_data", LIBRARY)};
    json_object *found = json_object_new_object();
    for (int i = 0; i < 2; i++) {
        glob_t files = {0};
        glob(fmt(a, "%s/%s/segments/*/mesh/*-on-%s-*um.tifxyz", root[i], p.sample, p.scan), 0, NULL,
            &files);
        for (size_t f = 0; f < files.gl_pathc; f++) {
            const char *path = files.gl_pathv[f], *tail = bench_basename(path);
            struct stat st;
            if (stat(fmt(a, "%s/z.tif", path), &st) || st.st_size <= 0) {
                continue;
            }
            const char *mesh = fmt(a, "%.*s", (int)(tail - path - 1), path);
            const char *parent = bench_basename(mesh);
            const char *segment = bench_basename(fmt(a, "%.*s", (int)(parent - mesh - 1), mesh));
            if (!json_object_object_get_ex(found, segment, NULL)) {
                json_object_object_add(found, segment, json_object_new_string(path));
            }
        }
        globfree(&files);
    }
    int n = json_object_object_length(found);
    PackageGrid *every = pushz(a, n * sizeof(*every));
    p.grids = every;
    int take = 0;
    json_object_object_foreach(found, segment, path) {
        every[take].source = fmt(a, "%s", json_object_get_string(path));
        every[take++].segment = fmt(a, "%s", segment);
    }
    json_object_put(found);
    qsort(every, n, sizeof(*every), package_name_order);
    p.sources = json_object_new_object();
    BenchOptions source = {.ignore_winding = 1};
    Arena scratch = arena(16ull << 30);
    for (int i = 0; i < n; i++) {
        PackageGrid g = every[i];
        size_t saved = a->used;
        scratch.used = 0;
        g.raw = bench_load_piece(&scratch, g.source, &source);
        int h = g.raw.h, w = g.raw.w;
        g.lo = h;
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                i64 at = (i64)r * w + c;
                if (g.raw.index[at] < 0) {
                    continue;
                }
                double z = g.raw.grid[3 * at + 2];
                g.inside += z >= p.low && z < p.high;
                if (z >= p.low - 512 && z < p.high + 512) {
                    g.lo = r < g.lo ? r : g.lo;
                    g.hi = r + 1;
                }
            }
        }
        json_object *record = json_object_new_object(), *hashes = json_object_new_object();
        json_object_object_add(record, "folder", json_object_new_string(g.source));
        json_object_object_add(record, "sha256", hashes);
        json_object_object_add(record, "in_range", json_object_new_int64(g.inside));
        json_object_object_add(p.sources, g.segment, record);
        const char *parts[] = {"x.tif", "y.tif", "z.tif", "meta.json"};
        for (int d = 0; d < 4; d++) {
            char hex[65];
            if (bench_hash_of(fmt(a, "%s/%s", g.source, parts[d]), hex)) {
                json_object_object_add(hashes, parts[d], json_object_new_string(hex));
            }
        }
        if (g.inside) {
            i64 n = h * (i64)w;
            const f32 *grid = g.raw.grid;
            const i32 *index = g.raw.index;
            f32 *kept = push(a, n * 3 * sizeof(*kept));
            g.raw.grid = kept;
            g.raw.index = push(a, n * sizeof(*index));
            memcpy(kept, grid, n * 3 * sizeof(*kept));
            memcpy(g.raw.index, index, n * sizeof(*index));
            g.wrap = package_wrap(g.segment, &g.end);
            p.named += g.wrap >= 0;
            g.first = p.count;
            p.count += g.inside;
            p.grids[p.grids_n++] = g;
        } else {
            json_object_put(g.raw.meta);
            a->used = saved;
        }
    }
    munmap(scratch.base, scratch.size);
    p.named = p.named >= 2;
    need(p.count > 0 && p.grids_n <= INT16_MAX, "no mesh vertex of %s %s in z [%g, %g)", p.sample,
        p.scan, p.low, p.high);
    return p;
}

static void package_normal(const BenchSurface *s, int y, int x, int top, int bottom, double n[3]) {
    i64 at = (i64)y * s->w + x;
    if (s->index[at] < 0) {
        n[0] = n[1] = n[2] = NAN;
        return;
    }
    i64 around[] = {y > top ? at - s->w : at, y + 1 < bottom ? at + s->w : at, x ? at - 1 : at,
        x + 1 < s->w ? at + 1 : at};
    for (int i = 0; i < 4; i++) {
        if (s->index[around[i]] < 0) {
            around[i] = at;
        }
    }
    double v[2][3];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            v[i][j] = around[2 * i] == at && around[2 * i + 1] == at
                ? NAN
                : (double)s->grid[3 * around[2 * i + 1] + j] - s->grid[3 * around[2 * i] + j];
        }
    }
    bench_cross3(v[0], v[1], n);
    double length = bench_norm3(n);
    for (int i = 0; i < 3; i++) {
        n[i] = length > 0 ? n[i] / length : NAN;
    }
}

static double package_quantile(double *values, i64 n, double q) {
    need(n > 0, "cannot find a quantile of an empty mesh");
    qsort(values, n, sizeof(*values), bench_double_order);
    double at = (n - 1) * q;
    i64 i = floor(at), j = ceil(at);
    return values[i] + (values[j] - values[i]) * (at - i);
}

static void package_knot(Package *p, double z, double x, double y) {
    double *v = p->knots + 3 * p->knot_count++;
    v[0] = z;
    v[1] = x;
    v[2] = y;
}

static int package_point_order(const void *a, const void *b) {
    double x = ((const double *)a)[0], y = ((const double *)b)[0];
    return (x > y) - (x < y);
}

static int package_legacy_axis(Arena *a, Package *p) {
    glob_t axes = {0};
    glob(fmt(a, "%s/calibration/geometry_wrap/truth/%s-%s-rows-*.npz", LIBRARY, p->sample, p->scan),
        0, NULL, &axes);
    double *data = NULL;
    int n = 0;
    for (size_t i = 0; i < axes.gl_pathc; i++) {
        const char *path = axes.gl_pathv[i];
        if (bench_file(path, "tier-a.npz")) {
            continue;
        }
        size_t saved = a->used;
        Array knots = npz_get(a, path, "knots");
        need(knots.ndim == 2 && knots.shape[1] == 3, "invalid v1 axis in %s", path);
        int count = knots.shape[0];
        double scale = exp2(bench_number(npz_get(a, path, "level"), 0));
        double *grown = realloc(data, ((size_t)n + count) * 3 * sizeof(*data));
        if (!grown)
            die("cannot keep legacy package axis");
        data = grown;
        for (int j = 0; j < 3 * count; j++) {
            data[3 * n + j] = bench_number(knots, j) * scale + 0.5 * (scale - 1);
        }
        n += count;
        a->used = saved;
    }
    globfree(&axes);
    if (!n) {
        free(data);
        return 0;
    }
    qsort(data, n, 3 * sizeof(*data), package_point_order);
    p->knots = push(a, 3 * n * sizeof(*p->knots));
    for (int i = 0; i < n; i++) {
        if (!i || nearbyint(data[3 * i]) != nearbyint(data[3 * (i - 1)])) {
            package_knot(p, data[3 * i], data[3 * i + 1], data[3 * i + 2]);
        }
    }
    free(data);
    p->axis_source = "v1 wrap truth knots (all bands of the scan)";
    return 1;
}

typedef struct {
    int mesh, row, centre;
} PackageCoreRow;

static int package_core_order(const void *aa, const void *bb) {
    const PackageCoreRow *a = aa, *b = bb;
    return a->centre != b->centre ? a->centre - b->centre : a->mesh - b->mesh;
}

typedef struct {
    double x;
    int sign;
} PackageRay;

static int package_ray_order(const void *aa, const void *bb) {
    double a = ((const PackageRay *)aa)->x, b = ((const PackageRay *)bb)->x;
    return (a > b) - (a < b);
}

static int package_core_one(Arena *a, Package *p, const PackageCoreRow *rows, int n) {
    size_t saved = a->used;
    i64 size = 0;
    for (int i = 0; i < n; i++) {
        size += p->grids[rows[i].mesh].raw.w;
    }
    f32 *pts = push(a, 3 * size * sizeof(*pts));
    i32(*lines)[2] = push(a, size * sizeof(*lines));
    i32 count = 0, segments = 0;
    double lowx = INFINITY, lowy = INFINITY, highx = -INFINITY, highy = -INFINITY;
    for (int i = 0; i < n; i++) {
        BenchSurface *s = &p->grids[rows[i].mesh].raw;
        i64 base = (i64)rows[i].row * s->w;
        for (int x = 0; x < s->w;) {
            if (s->index[base + x] < 0) {
                x++;
                continue;
            }
            int end = x + 1;
            while (end < s->w && s->index[base + end] >= 0) {
                end++;
            }
            if (end - x >= 2) {
                lines[segments][0] = count;
                int step = end - x > 2 ? 2 : 1;
                for (int k = x; k < end; k += step) {
                    const f32 *point = s->grid + 3 * (base + k);
                    pts[3LL * count] = point[0];
                    pts[3LL * count + 1] = point[1];
                    pts[3LL * count++ + 2] = 0;
                    lowx = fmin(lowx, point[0]);
                    lowy = fmin(lowy, point[1]);
                    highx = fmax(highx, point[0]);
                    highy = fmax(highy, point[1]);
                }
                lines[segments++][1] = count;
            }
            x = end;
        }
    }
    int width = count ? ceil((highx - lowx) / 16) : 0;
    int height = count ? ceil((highy - lowy) / 16) : 0;
    if (width < 2 || height < 2) {
        a->used = saved;
        return 0;
    }
    double *field = push(a, (i64)width * height * sizeof(*field)), most = 0;
    PackageRay *rays = push(a, count * sizeof(*rays));
    for (int y = 0; y < height; y++) {
        double gy = lowy + 16 * y;
        int r = 0, winding = 0;
        for (int l = 0; l < segments; l++) {
            for (int i = lines[l][0]; i + 1 < lines[l][1]; i++) {
                const f32 *from = pts + 3LL * i, *to = from + 3;
                int up = from[1] <= gy && to[1] > gy;
                int down = to[1] <= gy && from[1] > gy;
                if (up || down) {
                    double x = from[0] +
                        (gy - from[1]) / ((double)to[1] - from[1]) * ((double)to[0] - from[0]);
                    rays[r++] = (PackageRay){x, up ? 1 : -1};
                    winding += up ? 1 : -1;
                }
            }
        }
        qsort(rays, r, sizeof(*rays), package_ray_order);
        for (int x = 0, crossed = 0; x < width; x++) {
            double gx = lowx + 16 * x;
            while (crossed < r && rays[crossed].x <= gx) {
                winding -= rays[crossed++].sign;
            }
            double turns = winding;
            for (int l = 0; l < segments; l++) {
                const f32 *first = pts + 3LL * lines[l][0];
                const f32 *last = pts + 3LL * (lines[l][1] - 1);
                double start = atan2(first[1] - gy, first[0] - gx);
                double end = atan2(last[1] - gy, last[0] - gx);
                start += start < 0 ? 2 * M_PI : 0;
                end += end < 0 ? 2 * M_PI : 0;
                turns += (end - start) / (2 * M_PI);
            }
            field[y * (i64)width + x] = fabs(turns);
            most = fmax(most, fabs(turns));
        }
    }
    if (most < 2) {
        a->used = saved;
        return 0;
    }
    BenchTree tree = bench_tree(a, pts, count);
    double distance = -1, x0 = 0, y0 = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (field[y * (i64)width + x] < most - 0.5) {
                continue;
            }
            double gx = lowx + 16 * x, gy = lowy + 16 * y;
            f32 centre[] = {gx, gy, 0};
            BenchCandidate near[BENCH_NEIGHBORS];
            int used = 0;
            bench_candidates(&tree, 0, centre, INFINITY, near, &used);
            if (used && near[0].squared > distance) {
                distance = near[0].squared;
                x0 = gx;
                y0 = gy;
            }
        }
    }
    if (distance >= 0) {
        package_knot(p, rows[0].centre, x0, y0);
    }
    a->used = saved;
    return distance >= 0;
}

static int package_core_axis(Arena *a, Package *p) {
    PackageCoreRow *rows = NULL;
    int n = 0, cap = 0;
    for (int i = 0; i < p->grids_n; i++) {
        BenchSurface *s = &p->grids[i].raw;
        size_t saved = a->used;
        double *row_z = push(a, s->h * sizeof(*row_z)), *z = push(a, s->w * sizeof(*z));
        double last = -INFINITY;
        for (int r = 0; r < s->h; r++) {
            int used = 0;
            for (int x = 0; x < s->w; x++) {
                i64 at = (i64)r * s->w + x;
                if (s->index[at] >= 0) {
                    z[used++] = s->grid[3 * at + 2];
                }
            }
            row_z[r] = used >= 2 ? bench_median(z, used, 0) : NAN;
            last = fmax(last, row_z[r]);
        }
        for (int centre = 128; centre < last + 128; centre += 256) {
            int at = -1;
            double gap = INFINITY;
            for (int j = 0; j < s->h; j++) {
                double d = fabs(row_z[j] - centre);
                if (d < gap) {
                    gap = d;
                    at = j;
                }
            }
            if (at >= 0 && gap <= 32) {
                if (n == cap) {
                    cap = cap ? 2 * cap : 128;
                    PackageCoreRow *grown = realloc(rows, cap * sizeof(*rows));
                    if (!grown)
                        die("cannot retain package core rows");
                    rows = grown;
                }
                rows[n++] = (PackageCoreRow){i, at, centre};
            }
        }
        a->used = saved;
    }
    if (!n) {
        free(rows);
        return 0;
    }
    qsort(rows, n, sizeof(*rows), package_core_order);
    size_t saved = a->used;
    p->knots = push(a, n * 3 * sizeof(*p->knots));
    for (int i = 0; i < n;) {
        int end = i + 1;
        while (end < n && rows[end].centre == rows[i].centre) {
            end++;
        }
        package_core_one(a, p, rows + i, end - i);
        i = end;
    }
    free(rows);
    if (p->knot_count < 3) {
        a->used = saved;
        p->knot_count = 0;
        return 0;
    }
    p->axis_source = "inside the innermost traced turn: the point the truth rows "
                     "wind around most, 256 slices per knot";
    return 1;
}

static double package_turn(double angle) {
    double folded = angle - 2 * M_PI * floor((angle + M_PI) / (2 * M_PI));
    return folded == -M_PI && angle > 0 ? M_PI : folded;
}

static int package_ring(const BenchSurface *s, int row, double *zs, double out[3]) {
    i64 start = (i64)row * s->w;
    double xmean = 0, ymean = 0;
    int n = 0;
    for (int i = 0; i < s->w; i++) {
        if (s->index[start + i] >= 0) {
            const f32 *p = s->grid + 3 * (start + i);
            xmean += p[0];
            ymean += p[1];
            zs[n++] = p[2];
        }
    }
    if (n < 50) {
        return 0;
    }
    xmean /= n;
    ymean /= n;
    double xx = 0, xy = 0, yy = 0, ax = 0, ay = 0;
    for (int i = 0; i < s->w; i++) {
        if (s->index[start + i] >= 0) {
            const f32 *p = s->grid + 3 * (start + i);
            double x = p[0] - xmean, y = p[1] - ymean, q = x * x + y * y;
            xx += x * x;
            xy += x * y;
            yy += y * y;
            ax += q * x * 0.5;
            ay += q * y * 0.5;
        }
    }
    double determinant = xx * yy - xy * xy;
    if (!(determinant > 0)) {
        return 0;
    }
    double x0 = xmean + (ax * yy - ay * xy) / determinant;
    double y0 = ymean + (ay * xx - ax * xy) / determinant;
    double lo = INFINITY, hi = -INFINITY, previous = 0, phi = 0;
    for (int i = 0; i < s->w; i++) {
        if (s->index[start + i] >= 0) {
            const f32 *p = s->grid + 3 * (start + i);
            double angle = atan2(p[1] - y0, p[0] - x0);
            phi = isfinite(lo) ? phi + package_turn(angle - previous) : angle;
            previous = angle;
            lo = fmin(lo, phi);
            hi = fmax(hi, phi);
        }
    }
    if (hi - lo < 1.8 * M_PI) {
        return 0;
    }
    out[0] = bench_median(zs, n, 0);
    out[1] = x0;
    out[2] = y0;
    return 1;
}

static int package_ring_axis(Arena *a, Package *p) {
    size_t saved = a->used;
    int n = 0, rows = 0;
    for (int i = 0; i < p->grids_n; i++) {
        rows += p->grids[i].raw.h;
    }
    p->knots = push(a, rows * 3 * sizeof(*p->knots));
    size_t keep = a->used;
    double (*fit)[3] = push(a, rows * sizeof(*fit));
    for (int i = 0; i < p->grids_n; i++) {
        const BenchSurface *s = &p->grids[i].raw;
        size_t mark = a->used;
        double *z = push(a, s->w * sizeof(*z));
        for (int y = 0; y < s->h; y++) {
            n += package_ring(s, y, z, fit[n]);
        }
        a->used = mark;
    }
    qsort(fit, n, sizeof(*fit), package_point_order);
    double *xs = push(a, n * sizeof(*xs)), *ys = push(a, n * sizeof(*ys));
    for (int i = 0; i < n;) {
        double bin = floor(fit[i][0] / 256) * 256;
        int count = 0;
        while (i < n && fit[i][0] < bin + 256) {
            xs[count] = fit[i][1];
            ys[count++] = fit[i++][2];
        }
        if (count >= 5) {
            package_knot(p, bin + 128, bench_median(xs, count, 0), bench_median(ys, count, 0));
        }
    }
    if (p->knot_count < 3) {
        p->knot_count = 0;
        a->used = saved;
        return 0;
    }
    a->used = keep;
    p->axis_source = "median centre of circles fitted to closed truth rings, 256 slices per knot";
    return 1;
}

typedef struct {
    double z, x, y, nx, ny;
} PackageAxisPoint;

static int package_axis_point_order(const void *aa, const void *bb) {
    double a = ((const PackageAxisPoint *)aa)->z, b = ((const PackageAxisPoint *)bb)->z;
    return (a > b) - (a < b);
}

static int package_fitted_axis(Arena *a, Package *p) {
    p->knots = push(a, 12 * 3 * sizeof(*p->knots));
    size_t saved = a->used;
    PackageAxisPoint *points = push(a, p->count * sizeof(*points));
    i64 n = 0;
    for (int i = 0; i < p->grids_n; i++) {
        PackageGrid *g = p->grids + i;
        BenchSurface *s = &g->raw;
        for (int r = g->lo; r < g->hi; r++) {
            for (int x = 0; x < s->w; x++) {
                i64 at = (i64)r * s->w + x;
                const f32 *v = s->grid + 3 * at;
                if (s->index[at] < 0 || v[2] < p->low || v[2] >= p->high) {
                    continue;
                }
                double normal[3];
                package_normal(s, r, x, g->lo, g->hi, normal);
                double radius = hypot(normal[0], normal[1]);
                if (radius > 0.7) {
                    points[n++] = (PackageAxisPoint){
                        v[2], v[0], v[1], normal[0] / radius, normal[1] / radius};
                }
            }
        }
    }
    if (!n) {
        a->used = saved;
        return 0;
    }
    qsort(points, n, sizeof(*points), package_axis_point_order);
    double edges[13];
    for (int i = 0; i < 13; i++) {
        double v = (n - 1) * (i / 12.0);
        i64 first = floor(v), last = ceil(v);
        edges[i] = points[first].z + (v - first) * (points[last].z - points[first].z);
    }
    double *weights = push(a, n * sizeof(*weights)), *residual = push(a, n * sizeof(*residual));
    for (int i = 0; i < 12; i++) {
        i64 first = 0, last = n;
        while (first < n && points[first].z < edges[i]) {
            first++;
        }
        while (last > first && points[last - 1].z > edges[i + 1]) {
            last--;
        }
        i64 count = last - first;
        if (count < 500) {
            continue;
        }
        for (i64 j = 0; j < count; j++) {
            weights[j] = 1;
        }
        double x = 0, y = 0;
        int ok = 1;
        for (int k = 0; k < 12; k++) {
            double xx = 0, xy = 0, yy = 0, bx = 0, by = 0;
            for (i64 j = 0; j < count; j++) {
                const PackageAxisPoint *q = points + first + j;
                double w = weights[j], dx = q->nx, dy = q->ny;
                double ax = 1 - dx * dx, ay = 1 - dy * dy, cross = -dx * dy;
                xx += w * ax;
                yy += w * ay;
                xy += w * cross;
                bx += w * (ax * q->x + cross * q->y);
                by += w * (cross * q->x + ay * q->y);
            }
            double determinant = xx * yy - xy * xy;
            if (!(determinant > 0)) {
                ok = 0;
                break;
            }
            x = (yy * bx - xy * by) / determinant;
            y = (xx * by - xy * bx) / determinant;
            for (i64 j = 0; j < count; j++) {
                const PackageAxisPoint *q = points + first + j;
                residual[j] = fabs((x - q->x) * q->ny - (y - q->y) * q->nx);
            }
            double scale = fmax(1, 2 * bench_median(residual, count, 0));
            for (i64 j = 0; j < count; j++) {
                const PackageAxisPoint *q = points + first + j;
                double miss = ((x - q->x) * q->ny - (y - q->y) * q->nx) / scale;
                weights[j] = 1 / (1 + miss * miss);
            }
        }
        if (ok) {
            package_knot(p, (edges[i] + edges[i + 1]) / 2, x, y);
        }
    }
    a->used = saved;
    p->axis_source = "fitted to the truth normals";
    return p->knot_count > 0;
}

typedef struct {
    int vertices, root;
    i64 span;
    double turns;
} PackageRun;

typedef struct {
    i32 a, b;
    double turns;
} PackageJoin;

static int package_root(PackageRun *run, int i) {
    while (i != run[i].root) {
        i = run[i].root;
    }
    return i;
}

static void package_angle(const Package *p, const f32 *x, double *theta, double *radius) {
    double cx = interpolated(x[2], p->knots, p->knot_count, 1);
    double cy = interpolated(x[2], p->knots, p->knot_count, 2);
    *theta = atan2(x[1] - cy, x[0] - cx);
    *radius = hypot(x[0] - cx, x[1] - cy);
}

static void package_unwrap(Arena *a, Package *p, PackageGrid *g) {
    BenchSurface *s = &g->raw;
    int h = s->h, w = s->w, n = 0;
    i64 size = (i64)h * w, crop = (i64)(g->hi - g->lo) * w;
    g->phi = push(a, crop * sizeof(*g->phi));
    g->radius = push(a, crop * sizeof(*g->radius));
    size_t saved = a->used;
    i32 *piece = push(a, size * sizeof(*piece));
    double *phi = push(a, size * sizeof(*phi)), *radius = push(a, size * sizeof(*radius));
    PackageRun *run = push(a, s->n * sizeof(*run));
    for (int y = 0; y < h; y++) {
        int previous_piece = -1;
        double previous = 0;
        for (int x = 0; x < w; x++) {
            i64 i = (i64)y * w + x;
            if (s->index[i] < 0) {
                piece[i] = -1;
                phi[i] = radius[i] = NAN;
                previous_piece = -1;
                continue;
            }
            double theta;
            package_angle(p, s->grid + 3 * i, &theta, &radius[i]);
            if (previous_piece < 0) {
                previous_piece = n;
                run[n] = (PackageRun){.vertices = 0, .root = n, .span = 0, .turns = NAN};
                n++;
                phi[i] = theta;
            } else {
                phi[i] = phi[i - 1] + package_turn(theta - previous);
            }
            previous = theta;
            piece[i] = previous_piece;
            run[previous_piece].vertices++;
        }
    }
    PackageJoin *join = push(a, 2 * n * sizeof(*join));
    i32 links = 0;
    for (int y = 0; y + 1 < h; y++) {
        i64 left = (i64)y * w, right = left + w;
        for (int x = 0; x < w;) {
            int pa = piece[left + x], pb = piece[right + x];
            if (pa < 0 || pb < 0) {
                x++;
                continue;
            }
            int changed = x + 1, best = x;
            double reach = fmin(radius[left + x], radius[right + x]);
            while (changed < w && piece[left + changed] == pa && piece[right + changed] == pb) {
                double r = fmin(radius[left + changed], radius[right + changed]);
                if (r > reach) {
                    reach = r;
                    best = changed;
                }
                changed++;
            }
            double theta_a, theta_b, unused;
            package_angle(p, s->grid + 3 * (left + best), &theta_a, &unused);
            package_angle(p, s->grid + 3 * (right + best), &theta_b, &unused);
            join[links++] = (PackageJoin){pa, pb,
                nearbyint((phi[left + best] + package_turn(theta_b - theta_a) - phi[right + best]) /
                    (2 * M_PI))};
            int aroot = package_root(run, pa), broot = package_root(run, pb);
            if (aroot < broot) {
                run[broot].root = aroot;
            } else {
                run[aroot].root = broot;
            }
            x = changed;
        }
    }
    int keep = 0, seed = -1;
    for (int i = 0; i < n; i++) {
        int root = package_root(run, i);
        run[i].root = root;
        run[root].span += run[i].vertices;
    }
    for (int i = 0; i < n; i++) {
        if (run[i].span > run[keep].span) {
            keep = i;
        }
    }
    i32 *out = pushz(a, (n + 1) * sizeof(*out)), *in = pushz(a, (n + 1) * sizeof(*in));
    for (int i = 0; i < n; i++) {
        if (run[i].root == keep && (seed < 0 || run[i].vertices > run[seed].vertices)) {
            seed = i;
        }
    }
    for (int i = 0; i < links; i++) {
        out[join[i].a + 1]++;
        in[join[i].b + 1]++;
    }
    for (int i = 1; i <= n; i++) {
        out[i] += out[i - 1];
        in[i] += in[i - 1];
    }
    i32 *ordered = push(a, 2 * links * sizeof(*ordered));
    i32 *next = push(a, (n + 1) * sizeof(*next));
    memcpy(next, out, (n + 1) * sizeof(*out));
    for (int i = 0; i < links; i++) {
        ordered[next[join[i].a]++] = i;
    }
    memcpy(next, in, (n + 1) * sizeof(*in));
    for (int i = 0; i < links; i++) {
        ordered[links + next[join[i].b]++] = i;
    }
    i32 *queue = next;
    queue[0] = seed;
    run[seed].turns = 0;
    int tail = 1;
    for (int i = 0; i < tail; i++) {
        int at = queue[i];
        for (int d = 0; d < 2; d++) {
            i32 *used = d ? in : out;
            for (int e = used[at]; e < used[at + 1]; e++) {
                PackageJoin link = join[ordered[d * links + e]];
                int there = d ? link.a : link.b;
                if (!isfinite(run[there].turns)) {
                    run[there].turns = run[at].turns + (d ? -link.turns : link.turns);
                    queue[tail++] = there;
                }
            }
        }
    }
    g->dropped = 1 - run[keep].span / (double)s->n;
    double *lo = push(a, h * sizeof(*lo)), *hi = push(a, h * sizeof(*hi));
    double low = INFINITY, high = -INFINITY;
    int rows = 0;
    for (int y = 0; y < h; y++) {
        double first = INFINITY, last = -INFINITY;
        int count = 0;
        for (int x = 0; x < w; x++) {
            i64 i = (i64)y * w + x;
            int at = piece[i];
            double value = at >= 0 ? phi[i] + run[at].turns * (2 * M_PI) : NAN;
            if (isfinite(value)) {
                first = fmin(first, value);
                last = fmax(last, value);
                count++;
            }
            if (y >= g->lo && y < g->hi) {
                i64 kept = i - g->lo * (i64)w;
                g->phi[kept] = value;
                g->radius[kept] = radius[i];
            }
        }
        low = fmin(low, first);
        high = fmax(high, last);
        if (count >= 2) {
            lo[rows] = first / (2 * M_PI);
            hi[rows++] = last / (2 * M_PI);
        }
    }
    g->starts = rows ? package_quantile(lo, rows, 0.05) : low / (2 * M_PI);
    g->ends = rows ? package_quantile(hi, rows, 0.95) : high / (2 * M_PI);
    a->used = saved;
}

static double package_radial_median(double *sample, const double *phi, const double *radius, int n,
    double low, double high, int up) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (up ? phi[i] >= low && phi[i] < high : phi[i] > low && phi[i] <= high) {
            sample[count++] = radius[i];
        }
    }
    return bench_median(sample, count, NAN);
}

static void package_handedness(Arena *a, Package *p) {
    double correlation = 0;
    int votes = 0, rows = 0;
    const double window = 0.05 * 2 * M_PI;
    for (int i = 0; i < p->grids_n; i++) {
        PackageGrid *g = &p->grids[i];
        int w = g->raw.w;
        size_t saved = a->used;
        double *sample = push(a, w * sizeof(*sample));
        double sumx = 0, sumy = 0, sqx = 0, sqy = 0, xy = 0;
        double first = INFINITY, last = -INFINITY;
        i64 count = 0;
        for (int y = g->lo; y < g->hi; y++) {
            const double *phi = g->phi + (y - g->lo) * (i64)w;
            const double *radius = g->radius + (y - g->lo) * (i64)w;
            double low = INFINITY, high = -INFINITY;
            int n = 0;
            for (int x = 0; x < w; x++) {
                if (isfinite(phi[x])) {
                    low = fmin(low, phi[x]);
                    high = fmax(high, phi[x]);
                    sumx += phi[x];
                    sumy += radius[x];
                    sqx += phi[x] * phi[x];
                    sqy += radius[x] * radius[x];
                    xy += phi[x] * radius[x];
                    n++;
                }
            }
            count += n;
            first = fmin(first, low);
            last = fmax(last, high);
            if (n < 20 || high - low < 2 * M_PI + window) {
                continue;
            }
            double med[4];
            med[0] = package_radial_median(sample, phi, radius, w, low, low + window, 1);
            med[1] = package_radial_median(
                sample, phi, radius, w, low + 2 * M_PI, low + 2 * M_PI + window, 1);
            med[2] = package_radial_median(
                sample, phi, radius, w, high - 2 * M_PI - window, high - 2 * M_PI, 0);
            med[3] = package_radial_median(sample, phi, radius, w, high - window, high, 0);
            for (int d = 0; d < 4; d += 2) {
                if (isfinite(med[d]) && isfinite(med[d + 1])) {
                    votes += (med[d + 1] > med[d]) - (med[d + 1] < med[d]);
                    rows++;
                }
            }
        }
        if (count > 100 && last - first > 0.5) {
            double vx = sqx - sumx * sumx / count, vy = sqy - sumy * sumy / count;
            correlation += count * (xy - sumx * sumy / count) / sqrt(vx * vy);
        }
        a->used = saved;
    }
    p->sense = rows ? votes >= 0 ? 1 : -1 : correlation >= 0 ? 1 : -1;
    p->votes = json_object_new_object();
    json_object_object_add(p->votes, "rows", json_object_new_int(rows));
    if (rows) {
        json_object_object_add(p->votes, "net", json_object_new_int(votes));
    }
}

typedef struct {
    f32 *points, *normal, *area;
    double *winding;
    i16 *mesh;
    i32 *row, *col, *frame;
    u8 *zone, *flags;
} PackageVertices;

static double package_edge_squared(const f32 *x, const f32 *y) {
    double a = (double)x[0] - y[0], b = (double)x[1] - y[1], c = (double)x[2] - y[2];
    return a * a + b * b + c * c;
}

static int package_zone(json_object *zones, double z) {
    int zone = 255;
    for (int j = 0; j < board_count(zones); j++) {
        json_object *band = json_object_array_get_idx(zones, j);
        double lo = json_object_get_double(json_object_array_get_idx(band, 0));
        double hi = json_object_get_double(json_object_array_get_idx(band, 1));
        if (z >= lo && z < hi) {
            zone = bench_zone_of(board_item(band, 2));
        }
    }
    return zone;
}

static PackageVertices package_vertices(Arena *a, Package *p) {
    i64 n = p->count, at = 0, reserved = 0;
    need(n <= INT32_MAX && p->grids_n <= INT16_MAX, "control is too large");
    PackageVertices v = {.points = push(a, n * 3 * sizeof(f32)),
        .normal = push(a, n * 3 * sizeof(f32)),
        .area = push(a, n * sizeof(f32)),
        .winding = push(a, n * sizeof(double)),
        .mesh = push(a, n * sizeof(i16)),
        .row = push(a, n * sizeof(i32)),
        .col = push(a, n * sizeof(i32)),
        .frame = push(a, n * sizeof(i32)),
        .zone = push(a, n),
        .flags = push(a, n)};
    memset(v.flags, 0, n);
    memset(v.area, 0, n * sizeof(f32));
    json_object *zones = jget(p->entry, "zones");
    for (int id = 0; id < p->grids_n; id++) {
        PackageGrid *g = &p->grids[id];
        i64 size = (g->hi - g->lo) * (i64)g->raw.w;
        g->truth = push(a, size * sizeof(*g->truth));
        memset(g->truth, 0xff, size * sizeof(*g->truth));
        reserved += size;
    }
    size_t saved = a->used;
    double *distances = push(a, reserved * sizeof(*distances));
    i64 edges = 0;
    for (int id = 0; id < p->grids_n; id++) {
        PackageGrid *g = &p->grids[id];
        BenchSurface *s = &g->raw;
        int w = s->w;
        g->first = at;
        for (int r = g->lo; r < g->hi; r++) {
            for (int c = 0; c < w; c++) {
                i64 index = r * (i64)w + c, trim = (r - g->lo) * (i64)w + c;
                const f32 *point = s->grid + index * 3;
                if (s->index[index] < 0) {
                    continue;
                }
                if (c > 0 && s->index[index - 1] >= 0) {
                    distances[edges++] = sqrt(package_edge_squared(point, point - 3));
                }
                if (point[2] < p->low || point[2] >= p->high) {
                    continue;
                }
                double normal[3];
                package_normal(s, r, c, g->lo, g->hi, normal);
                double x = interpolated(point[2], p->knots, p->knot_count, 1) - point[0];
                double y = interpolated(point[2], p->knots, p->knot_count, 2) - point[1];
                int sign = normal[0] * x + normal[1] * y < 0 ? -1 : 1;
                for (int d = 0; d < 3; d++) {
                    v.normal[3 * at + d] = sign * normal[d];
                    v.points[3 * at + d] = point[d];
                }
                v.winding[at] = p->sense * g->phi[trim] / (2 * M_PI);
                v.zone[at] = package_zone(zones, point[2]);
                v.mesh[at] = id;
                v.row[at] = r;
                v.col[at] = c;
                g->truth[trim] = at++;
            }
        }
    }
    p->spacing = bench_median(distances, edges, NAN) * p->voxel;
    a->used = saved;
    need(at == n, "control vertex census changed");
    return v;
}

static void package_areas(Arena *a, const Package *p, PackageVertices *v) {
    double scale = p->voxel * 0.001;
    scale = 0.125 * scale * scale;
    for (int m = 0; m < p->grids_n; m++) {
        const PackageGrid *g = &p->grids[m];
        const BenchSurface *s = &g->raw;
        int w = s->w, h = g->hi - g->lo;
        size_t saved = a->used;
        double *area = push(a, g->inside * sizeof(*area));
        memset(area, 0, g->inside * sizeof(*area));
        for (int r = 0; r + 1 < h; r++) {
            for (int c = 0; c + 1 < w; c++) {
                i64 offset = r * (i64)w + c, span = offset + g->lo * (i64)w;
                if (s->index[span] < 0 || s->index[span + 1] < 0 || s->index[span + w] < 0 ||
                    s->index[span + w + 1] < 0) {
                    continue;
                }
                double x[3], y[3], cross[3];
                for (int d = 0; d < 3; d++) {
                    x[d] = (double)s->grid[3 * (span + w + 1) + d] - s->grid[3 * span + d];
                    y[d] = (double)s->grid[3 * (span + w) + d] - s->grid[3 * (span + 1) + d];
                }
                bench_cross3(x, y, cross);
                double contribution = bench_norm3(cross) * scale;
                i32 index[4] = {g->truth[offset], g->truth[offset + 1], g->truth[offset + w],
                    g->truth[offset + w + 1]};
                for (int i = 0; i < 4; i++) {
                    if (index[i] >= 0) {
                        area[index[i] - g->first] += contribution;
                    }
                }
            }
        }
        for (i64 i = 0; i < g->inside; i++) {
            v->area[g->first + i] = area[i];
        }
        a->used = saved;
    }
}

typedef struct {
    const Package *package;
    PackageVertices *vertices;
    BenchTree tree;
    const int *rank;
} PackageNeighbors;

static double package_projection(const f32 *p, const f32 *q, const f32 *n) {
    return ((double)q[0] - p[0]) * n[0] + ((double)q[1] - p[1]) * n[1] +
        ((double)q[2] - p[2]) * n[2];
}

static int package_good_normal(const f32 *normal) {
    return isfinite(normal[0]) && isfinite(normal[1]) && isfinite(normal[2]);
}

static void package_flags_range(void *opaque, i64 first, i64 last) {
    PackageNeighbors *neighbors = opaque;
    const Package *p = neighbors->package;
    PackageVertices *v = neighbors->vertices;
    double lateral = 0.75 * p->spacing / p->voxel;
    double touching = 40 / p->voxel, pressed = 80 / p->voxel, radial = 400 / p->voxel;
    double lateral_squared = lateral * lateral, near = pressed * pressed + lateral_squared;
    for (i64 i = first; i < last; i++) {
        const f32 *point = v->points + 3 * i, *normal = v->normal + 3 * i;
        if (!isfinite(v->winding[i]) || !package_good_normal(normal)) {
            continue;
        }
        BenchCandidate hit[64];
        int used = 0, a = v->mesh[i], mine = p->grids[a].wrap;
        const BenchTree *tree = &neighbors->tree;
        bench_candidates_limit(tree, 0, point, near, hit, &used, 24);
        unsigned flags = 0;
        for (int j = 0; j < used; j++) {
            i32 ix = hit[j].index;
            int b = v->mesh[ix];
            if (a == b || !isfinite(v->winding[ix])) {
                continue;
            }
            double along = package_projection(point, v->points + 3 * (i64)ix, normal);
            double apart = fabs(along);
            if (hit[j].squared - along * along > lateral_squared) {
                continue;
            }
            int same_frame = v->frame[ix] == v->frame[i], theirs = p->grids[b].wrap;
            double step = fabs(v->winding[ix] - v->winding[i]);
            int contact = apart <= touching, same = same_frame && contact && step < 0.5;
            if (contact && (!same_frame || step < 0.5) && neighbors->rank[b] < neighbors->rank[a]) {
                flags |= 1;
            }
            if (apart <= pressed && (!same_frame || step >= 0.5)) {
                flags |= BENCH_PRESSED | (contact ? BENCH_TOUCHING : 0);
            }
            if (same) {
                i64 offset = mine >= 0 && theirs >= 0 ? (i64)theirs - mine : 0;
                flags |= offset == 1 ? 4 : offset == -1 ? 8 : 16;
            }
            if (same_frame && contact && step >= 1.5) {
                flags |= 32;
            }
        }
        used = 0;
        bench_candidates_limit(tree, 0, point, radial * radial, hit, &used, 64);
        for (int j = 0; j < used; j++) {
            i32 ix = hit[j].index;
            if (a == v->mesh[ix] || !isfinite(v->winding[ix]) || v->frame[ix] != v->frame[i]) {
                continue;
            }
            double out = -package_projection(point, v->points + 3 * (i64)ix, normal);
            if (out > pressed && out >= 0.8 * sqrt(hit[j].squared)) {
                if (v->winding[ix] - v->winding[i] < 0.5) {
                    flags |= 64;
                }
                break;
            }
        }
        v->flags[i] = flags;
    }
}

static Array package_names(Arena *a, const Package *p) {
    int width = 1;
    for (int m = 0; m < p->grids_n; m++) {
        const char *name = p->grids[m].segment;
        int len = 0;
        while (*name) {
            bench_utf8(&name);
            len++;
        }
        if (len > width) {
            width = len;
        }
    }
    i64 size = p->grids_n * (i64)width * sizeof(u32);
    u32 *names = pushz(a, size);
    for (int m = 0; m < p->grids_n; m++) {
        const char *name = p->grids[m].segment;
        for (int j = 0; *name; j++) {
            names[m * (i64)width + j] = bench_utf8(&name);
        }
    }
    Array out = {.ndim = 1, .shape = {p->grids_n}, .data = names, .size = size};
    snprintf(out.descr, sizeof(out.descr), "<U%d", width);
    return out;
}

static void package_vertex_archive(
    Arena *a, const char *folder, Package *p, const PackageVertices *v, Array names) {
    i64 n = p->count, sense = p->sense;
    size_t saved = a->used;
    i64 *wraps = push(a, p->grids_n * sizeof(*wraps));
    for (int m = 0; m < p->grids_n; m++) {
        wraps[m] = p->grids[m].wrap;
    }
    const char *keys[] = {"points", "normals", "mesh", "row", "col", "winding", "frame", "area_mm2",
        "zone", "flags", "names", "mesh_wrap", "sense", "voxel_um", "spacing_um", "knots"};
    Array arrays[] = {
        {.descr = "<f4", .ndim = 2, .shape = {n, 3}, .data = v->points, .size = n * 12},
        {.descr = "<f4", .ndim = 2, .shape = {n, 3}, .data = v->normal, .size = n * 12},
        {.descr = "<i2", .ndim = 1, .shape = {n}, .data = v->mesh, .size = n * 2},
        {.descr = "<i4", .ndim = 1, .shape = {n}, .data = v->row, .size = n * 4},
        {.descr = "<i4", .ndim = 1, .shape = {n}, .data = v->col, .size = n * 4},
        {.descr = "<f8", .ndim = 1, .shape = {n}, .data = v->winding, .size = n * 8},
        {.descr = "<i4", .ndim = 1, .shape = {n}, .data = v->frame, .size = n * 4},
        {.descr = "<f4", .ndim = 1, .shape = {n}, .data = v->area, .size = n * 4},
        {.descr = "|u1", .ndim = 1, .shape = {n}, .data = v->zone, .size = n},
        {.descr = "|u1", .ndim = 1, .shape = {n}, .data = v->flags, .size = n},
        names,
        {.descr = "<i8",
            .ndim = 1,
            .shape = {p->grids_n},
            .data = wraps,
            .size = p->grids_n * (i64)sizeof(*wraps)},
        {.descr = "<i8", .data = &sense, .size = sizeof(sense)},
        {.descr = "<f8", .data = &p->voxel, .size = sizeof(p->voxel)},
        {.descr = "<f8", .data = &p->spacing, .size = sizeof(p->spacing)},
        {.descr = "<f8",
            .ndim = 2,
            .shape = {p->knot_count, 3},
            .data = p->knots,
            .size = p->knot_count * 3LL * sizeof(*p->knots)},
    };
    npz_write_codec(
        fmt(a, "%s/truth.npz", folder), sizeof(keys) / sizeof(*keys), keys, arrays, ZIP_CM_DEFLATE);
    a->used = saved;
}

typedef struct {
    f32 z, xy[2], row, normal[3];
    double winding;
    i32 col, vertex_row, edge;
    i16 mesh;
    u8 flags, zone;
} PackageCrossing;

static int package_crossing_order(const void *left, const void *right) {
    const PackageCrossing *a = left, *b = right;
    if (a->z != b->z) {
        return a->z < b->z ? -1 : 1;
    }
    if (a->mesh != b->mesh) {
        return a->mesh - b->mesh;
    }
    if (a->col != b->col) {
        return a->col - b->col;
    }
    return a->edge - b->edge;
}

static void package_normal_toward(const Package *p, const f32 *point, double normal[3]) {
    double x = interpolated(point[2], p->knots, p->knot_count, 1) - point[0];
    double y = interpolated(point[2], p->knots, p->knot_count, 2) - point[1];
    if (x * normal[0] + y * normal[1] < 0) {
        for (int d = 0; d < 3; d++) {
            normal[d] = -normal[d];
        }
    }
}

static i64 package_crossing_grid(
    const Package *p, const PackageVertices *v, int m, PackageCrossing *cross) {
    const PackageGrid *g = &p->grids[m];
    const BenchSurface *s = &g->raw;
    json_object *zones = jget(p->entry, "zones");
    i64 count = 0, max_plane = floor((p->high - 1 - p->low) / p->step);
    for (int r = g->lo; r + 1 < g->hi; r++) {
        for (int c = 0; c < s->w; c++) {
            i64 i = r * (i64)s->w + c, k = (r - g->lo) * (i64)s->w + c;
            if (s->index[i] < 0 || s->index[i + s->w] < 0 || !isfinite(g->phi[k]) ||
                !isfinite(g->phi[k + s->w]) || !isfinite(g->offset)) {
                continue;
            }
            const f32 *a = s->grid + 3 * i, *b = a + 3LL * s->w;
            double za = a[2], zb = b[2];
            if (za == zb) {
                continue;
            }
            i64 first = fmax(0, ceil((fmin(za, zb) - p->low) / p->step));
            i64 last = fmin(max_plane, ceil((fmax(za, zb) - p->low) / p->step) - 1);
            if (last < first) {
                continue;
            }
            if (!cross) {
                count += last - first + 1;
                continue;
            }
            double na[3], nb[3];
            package_normal(s, r, c, g->lo, g->hi, na);
            package_normal(s, r + 1, c, g->lo, g->hi, nb);
            package_normal_toward(p, a, na);
            package_normal_toward(p, b, nb);
            double w0 = p->sense * g->phi[k] / (2 * M_PI) + g->offset + p->anchor;
            double w1 = p->sense * g->phi[k + s->w] / (2 * M_PI) + g->offset + p->anchor;
            for (i64 plane = first; plane <= last; plane++) {
                double z = p->low + plane * p->step, t = (z - za) / (zb - za);
                double x = a[0] + t * ((double)b[0] - a[0]);
                double y = a[1] + t * ((double)b[1] - a[1]);
                double n[3];
                for (int d = 0; d < 3; d++) {
                    n[d] = na[d] + t * (nb[d] - na[d]);
                }
                double length = bench_norm3(n);
                double ax = interpolated(z, p->knots, p->knot_count, 1) - x;
                double ay = interpolated(z, p->knots, p->knot_count, 2) - y;
                if (n[0] * ax + n[1] * ay < 0) {
                    length = -length;
                }
                int nearest = t >= 0.5;
                i32 vertex = g->truth[k + nearest * s->w];
                cross[count++] = (PackageCrossing){.z = z,
                    .xy = {x, y},
                    .row = (f32)(r - g->lo + t) + g->lo,
                    .col = c,
                    .mesh = m,
                    .vertex_row = r + nearest,
                    .edge = r,
                    .winding = w0 + t * (w1 - w0),
                    .normal = {n[0] / length, n[1] / length, n[2] / length},
                    .flags = vertex < 0 ? 0 : v->flags[vertex],
                    .zone = package_zone(zones, z)};
            }
        }
    }
    return count;
}

static i64 package_crossing_archive(Arena *a, const char *folder, const Package *p,
    const PackageVertices *v, Array names, i64 *zones) {
    size_t saved = a->used;
    i64 count = 0, step = p->step;
    for (int m = 0; m < p->grids_n; m++) {
        count += package_crossing_grid(p, v, m, NULL);
    }
    PackageCrossing *crossing = push(a, count * sizeof(*crossing));
    i64 at = 0;
    for (int m = 0; m < p->grids_n; m++) {
        at += package_crossing_grid(p, v, m, crossing + at);
    }
    need(at == count, "control crossing census changed");
    qsort(crossing, count, sizeof(*crossing), package_crossing_order);
    f32 *z = push(a, count * 4), *xy = push(a, count * 8), *row = push(a, count * 4),
        *normals = push(a, count * 12);
    double *winding = push(a, count * 8);
    i32 *col = push(a, count * 4), *vertex_row = push(a, count * 4);
    i16 *mesh = push(a, count * 2);
    u8 *flags = push(a, count), *zone = push(a, count);
    for (i64 i = 0; i < count; i++) {
        const PackageCrossing *c = &crossing[i];
        z[i] = c->z;
        memcpy(xy + 2 * i, c->xy, sizeof(c->xy));
        row[i] = c->row;
        col[i] = c->col;
        winding[i] = c->winding;
        memcpy(normals + 3 * i, c->normal, sizeof(c->normal));
        vertex_row[i] = c->vertex_row;
        mesh[i] = c->mesh;
        flags[i] = c->flags;
        zone[i] = c->zone;
        zones[c->zone]++;
    }
    const char *keys[] = {"z", "xy", "row", "col", "winding", "normal", "vertex_row", "mesh",
        "flags", "zone", "names", "step"};
    Array arrays[] = {
        {.descr = "<f4", .ndim = 1, .shape = {count}, .data = z, .size = count * 4},
        {.descr = "<f4", .ndim = 2, .shape = {count, 2}, .data = xy, .size = count * 8},
        {.descr = "<f4", .ndim = 1, .shape = {count}, .data = row, .size = count * 4},
        {.descr = "<i4", .ndim = 1, .shape = {count}, .data = col, .size = count * 4},
        {.descr = "<f8", .ndim = 1, .shape = {count}, .data = winding, .size = count * 8},
        {.descr = "<f4", .ndim = 2, .shape = {count, 3}, .data = normals, .size = count * 12},
        {.descr = "<i4", .ndim = 1, .shape = {count}, .data = vertex_row, .size = count * 4},
        {.descr = "<i2", .ndim = 1, .shape = {count}, .data = mesh, .size = count * 2},
        {.descr = "|u1", .ndim = 1, .shape = {count}, .data = flags, .size = count},
        {.descr = "|u1", .ndim = 1, .shape = {count}, .data = zone, .size = count},
        names,
        {.descr = "<i8", .data = &step, .size = sizeof(step)},
    };
    npz_write_codec(fmt(a, "%s/crossings.npz", folder), sizeof(keys) / sizeof(*keys), keys, arrays,
        ZIP_CM_DEFLATE);
    a->used = saved;
    return count;
}

static json_object *package_pair(double a, double b) {
    json_object *out = json_object_new_array();
    json_object_array_add(out, isfinite(a) ? jnum(a) : json_object_new_double(a));
    json_object_array_add(out, isfinite(b) ? jnum(b) : json_object_new_double(b));
    return out;
}

static void package_mesh(
    Arena *a, const char *root, const Package *p, const PackageVertices *v, const PackageGrid *g) {
    size_t saved = a->used;
    int r0 = g->hi, r1 = g->lo, c0 = g->raw.w, c1 = 0;
    double low = INFINITY, high = -INFINITY;
    for (i64 i = g->first; i < g->first + g->inside; i++) {
        r0 = v->row[i] < r0 ? v->row[i] : r0;
        r1 = v->row[i] > r1 ? v->row[i] : r1;
        c0 = v->col[i] < c0 ? v->col[i] : c0;
        c1 = v->col[i] > c1 ? v->col[i] : c1;
        low = fmin(low, v->winding[i]);
        high = fmax(high, v->winding[i]);
    }
    r0 = r0 > g->lo ? r0 - 1 : r0;
    c0 = c0 > 0 ? c0 - 1 : 0;
    r1 = r1 + 2 < g->hi ? r1 + 2 : g->hi;
    c1 = c1 + 2 < g->raw.w ? c1 + 2 : g->raw.w;
    int h = r1 - r0, w = c1 - c0;
    const char *folder = fmt(a, "%s/meshes/%s.tifxyz", root, g->segment);
    mkdirs(folder);
    f32 *plane = push(a, h * (i64)w * sizeof(*plane));
    for (int d = 0; d < 4; d++) {
        for (int y = r0; y < r1; y++) {
            for (int x = c0; x < c1; x++) {
                i64 offset = (y - g->lo) * (i64)g->raw.w + x;
                i32 ix = g->truth[offset];
                double value;
                if (d == 3) {
                    value = ix < 0 ? NAN : v->winding[ix];
                } else {
                    value = ix < 0 ? -1 : v->points[3 * (i64)ix + d];
                }
                plane[(y - r0) * (i64)w + x - c0] = value;
            }
        }
        tiff_write(fmt(a, "%s/%c.tif", folder, "xyzw"[d]), plane, w, h);
    }
    json_object *meta = json_object_new_object();
    json_object_object_add(meta, "format", json_object_new_string("tifxyz"));
    json_object_object_add(meta, "type", json_object_new_string("seg"));
    json_object *scale = json_object_object_get(g->raw.meta, "scale");
    json_object_object_add(
        meta, "scale", scale ? json_object_get(scale) : package_pair(0.05, 0.05));
    json_object_object_add(meta, "segment", json_object_new_string(g->segment));
    json_object_object_add(meta, "source", json_object_new_string(g->source));
    json_object_object_add(meta, "wrap_name", json_object_new_int(g->wrap));
    json_object_object_add(meta, "frame", json_object_new_int(g->frame));
    json_object_object_add(meta, "grid_origin_row_col", package_pair(r0, c0));
    json_object_object_add(meta, "z_range", package_pair(p->low, p->high));
    json_object_object_add(meta, "winding_range",
        package_pair(round(low * 10000) / 10000, round(high * 10000) / 10000));
    json_object_object_add(meta, "note",
        json_object_new_string(
            "x, y, z level-0 voxels (-1 outside the control's z range); w.tif continuous winding"));
    json_write(fmt(a, "%s/meta.json", folder), meta);
    json_object_put(meta);
    a->used = saved;
}

typedef struct {
    int a, b, delta, chunk;
    i64 count;
} PackageVote;

typedef struct {
    PackageVote *slot;
    i64 capacity, count;
} PackageVotes;

static u64 package_vote_hash(int a, int b, int d) {
    return ((u64)(u32)a * 0x9e3779b185ebca87ull ^ (u64)(u32)b * 0xc2b2ae3d27d4eb4full ^
               (u64)(u32)d * 0x165667b19e3779f9ull) >>
        16;
}

static PackageVote *package_vote_slot(PackageVotes *v, int a, int b, int d) {
    u64 hash = package_vote_hash(a, b, d);
    i64 at = hash & (v->capacity - 1);
    while (
        v->slot[at].count && (v->slot[at].a != a || v->slot[at].b != b || v->slot[at].delta != d)) {
        at = (at + 1) & (v->capacity - 1);
    }
    return v->slot + at;
}

static void package_vote_add(PackageVotes *v, int a, int b, int d, int chunk, i64 count) {
    if (v->count * 2 >= v->capacity) {
        PackageVotes old = *v;
        v->capacity = old.capacity ? 2 * old.capacity : 128;
        v->slot = calloc(v->capacity, sizeof(*v->slot));
        need(v->slot != NULL, "cannot allocate bench vote counts");
        for (i64 i = 0; i < old.capacity; i++) {
            PackageVote e = old.slot[i];
            if (e.count) {
                *package_vote_slot(v, e.a, e.b, e.delta) = e;
            }
        }
        free(old.slot);
    }
    PackageVote *e = package_vote_slot(v, a, b, d);
    if (e->count) {
        e->count += count;
    } else {
        *e = (PackageVote){a, b, d, chunk, count};
        v->count++;
    }
}

static int package_vote_order(const void *left, const void *right) {
    const PackageVote *a = left, *b = right;
    if (a->chunk != b->chunk) {
        return a->chunk - b->chunk;
    }
    if (a->a != b->a) {
        return a->a - b->a;
    }
    if (a->b != b->b) {
        return a->b - b->b;
    }
    return (a->delta > b->delta) - (a->delta < b->delta);
}

static void package_votes_compact(PackageVotes *v) {
    i64 n = 0;
    for (i64 i = 0; i < v->capacity; i++) {
        if (v->slot[i].count) {
            v->slot[n++] = v->slot[i];
        }
    }
    qsort(v->slot, n, sizeof(*v->slot), package_vote_order);
}

static void package_choose(Arena *a, Pcg64 *rng, i32 *pick, i64 usable, i64 count) {
    if (usable > 10000 && count > usable / 50) {
        i64 first = usable - count;
        for (i64 i = usable - 1; i > 0 && i >= first; i--) {
            i64 j = draw(rng, i + 1);
            i32 tmp = pick[i];
            pick[i] = pick[j];
            pick[j] = tmp;
        }
        memmove(pick, pick + first, count * sizeof(*pick));
        return;
    }
    size_t saved = a->used;
    i32 *other = push(a, count * sizeof(*other));
    i64 cap = 1;
    while (cap < count * 2) {
        cap *= 2;
    }
    i32 *keys = push(a, cap * sizeof(*keys));
    memset(keys, 0xff, cap * sizeof(*keys));
    for (i64 i = usable - count, k = 0; i < usable; i++, k++) {
        i32 j = draw(rng, i + 1);
        i64 at = j & (cap - 1);
        while (keys[at] >= 0 && keys[at] != j) {
            at = (at + 1) & (cap - 1);
        }
        if (keys[at] == j) {
            j = i;
            at = j & (cap - 1);
            while (keys[at] >= 0) {
                at = (at + 1) & (cap - 1);
            }
        }
        keys[at] = j;
        other[k] = pick[j];
    }
    for (i64 i = count - 1; i > 0; i--) {
        i64 j = draw(rng, i + 1);
        i32 tmp = other[i];
        other[i] = other[j];
        other[j] = tmp;
    }
    memcpy(pick, other, count * sizeof(*pick));
    a->used = saved;
    return;
}

static void package_pair_votes(Arena *a, const PackageNeighbors *neighbors, PackageVotes votes[3]) {
    const Package *p = neighbors->package;
    const PackageVertices *v = neighbors->vertices;
    size_t saved = a->used;
    i32 *pick = push(a, p->count * sizeof(*pick));
    i64 usable = 0;
    for (i64 i = 0; i < p->count; i++) {
        if (isfinite(v->winding[i]) && package_good_normal(v->normal + 3 * i)) {
            pick[usable++] = i;
        }
    }
    i64 count = usable < 300000 ? usable : 300000;
    Pcg64 rng = pcg64(0);
    package_choose(a, &rng, pick, usable, count);
    double lateral = 0.75 * p->spacing / p->voxel;
    double touching = 40 / p->voxel, same = 16 / p->voxel;
    double pressed = 80 / p->voxel, radial = 400 / p->voxel;
    for (i64 i = 0; i < count; i++) {
        i64 index = pick[i];
        const f32 *point = v->points + 3 * index, *normal = v->normal + 3 * index;
        BenchCandidate hit[64];
        int used = 0, id = v->mesh[index], outward = 0, chunk = i / 100000;
        bench_candidates_limit(&neighbors->tree, 0, point, radial * radial, hit, &used, 64);
        for (int j = 0; j < used; j++) {
            i32 ix = hit[j].index;
            int other = v->mesh[ix];
            if (id == other || !isfinite(v->winding[ix])) {
                continue;
            }
            double along = package_projection(point, v->points + 3 * (i64)ix, normal);
            int delta = nearbyint(v->winding[ix] - v->winding[index]);
            if (fabs(along) <= touching && hit[j].squared - along * along <= lateral * lateral) {
                package_vote_add(&votes[0], id, other, delta, chunk, 1);
                if (fabs(along) <= same) {
                    package_vote_add(&votes[2], id, other, delta, chunk, 1);
                }
            }
            if (!outward && -along > pressed && -along >= 0.8 * sqrt(hit[j].squared)) {
                package_vote_add(&votes[1], id, other, delta, chunk, 1);
                outward = 1;
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        package_votes_compact(&votes[i]);
    }
    a->used = saved;
}

typedef struct {
    json_object *chain, *links, *checks, *name_offsets;
} PackageAlignment;

static double package_round(double value, int places) {
    double scale = places == 4 ? 10000 : places == 3 ? 1000 : places == 2 ? 100 : 10;
    return nearbyint(value * scale) / scale;
}

static void package_ratio(json_object *record, const char *key, i64 part, i64 count) {
    json_object_object_add(
        record, key, count ? jnum(package_round((double)part / count, 4)) : NULL);
}

static int package_frame_find(const int *parent, const int *offset, int i, int *total) {
    *total = 0;
    while (i != parent[i]) {
        *total += offset[i];
        i = parent[i];
    }
    return i;
}

typedef struct {
    int a, b, order, mode;
    i64 count, peak;
} PackageLink;

static int package_link_order(const void *left, const void *right) {
    const PackageLink *a = left, *b = right;
    if (a->count != b->count) {
        return a->count > b->count ? -1 : 1;
    }
    return a->order - b->order;
}

static void package_link_frames(
    Arena *a, Package *p, const PackageVotes *surface, json_object *record) {
    size_t saved = a->used;
    int n = p->grids_n;
    PackageVotes votes = {0}, pairs = {0};
    for (i64 i = 0; i < surface->count; i++) {
        PackageVote e = surface->slot[i];
        PackageGrid x = p->grids[e.a], y = p->grids[e.b];
        if (x.frame == y.frame) {
            continue;
        }
        int delta = e.delta + y.offset - x.offset;
        if (x.frame < y.frame) {
            package_vote_add(&votes, x.frame, y.frame, delta, i, e.count);
            package_vote_add(&pairs, x.frame, y.frame, 0, i, e.count);
        } else {
            package_vote_add(&votes, y.frame, x.frame, -delta, i, e.count);
            package_vote_add(&pairs, y.frame, x.frame, 0, i, e.count);
        }
    }
    PackageLink *links = push(a, pairs.count * sizeof(*links));
    i64 count = 0;
    for (i64 i = 0; i < pairs.capacity; i++) {
        PackageVote e = pairs.slot[i];
        if (e.count) {
            links[count++] = (PackageLink){.a = e.a, .b = e.b, .order = e.chunk, .count = e.count};
        }
    }
    package_votes_compact(&votes);
    for (i64 i = 0; i < votes.count; i++) {
        PackageVote e = votes.slot[i];
        for (i64 j = 0; j < count; j++) {
            PackageLink *x = links + j;
            if (x->a == e.a && x->b == e.b && e.count > x->peak) {
                x->peak = e.count;
                x->mode = e.delta;
                break;
            }
        }
    }
    free(votes.slot);
    free(pairs.slot);
    qsort(links, count, sizeof(*links), package_link_order);
    int *parent = push(a, n * sizeof(*parent));
    int *offset = pushz(a, n * sizeof(*offset)), *ids = push(a, n * sizeof(*ids));
    memset(ids, 0xff, n * sizeof(*ids));
    for (int i = 0; i < n; i++) {
        parent[i] = i;
    }
    for (i64 i = 0; i < count; i++) {
        PackageLink e = links[i];
        int step = -e.mode;
        json_object *item = json_object_new_object();
        json_object_array_add(record, item);
        json_object_object_add(item, "frames", package_pair(e.a, e.b));
        json_object_object_add(item, "votes", json_object_new_int64(e.count));
        package_ratio(item, "purity", e.peak, e.count);
        json_object_object_add(item, "offset", json_object_new_int(step));
        const char *joined = "no";
        if (e.count >= 200 && (double)e.peak / e.count >= 0.9) {
            int dx, dy;
            int x = package_frame_find(parent, offset, e.a, &dx);
            int y = package_frame_find(parent, offset, e.b, &dy);
            joined = "yes";
            if (x == y) {
                joined = "already";
                json_object_object_add(
                    item, "consistent", json_object_new_boolean(dy - dx == step));
            } else if (x < y) {
                parent[y] = x;
                offset[y] = step - dy + dx;
            } else {
                parent[x] = y;
                offset[x] = -step + dy - dx;
            }
        }
        json_object_object_add(item, "joined", json_object_new_string(joined));
    }
    for (int m = 0; m < n; m++) {
        int shift;
        int root = package_frame_find(parent, offset, p->grids[m].frame, &shift);
        p->grids[m].frame = root;
        p->grids[m].offset += shift;
        ids[root] = 0;
    }
    for (int i = 0, next = 0; i < n; i++) {
        if (ids[i] == 0) {
            ids[i] = next++;
        }
    }
    for (int m = 0; m < n; m++) {
        p->grids[m].frame = ids[p->grids[m].frame];
    }
    a->used = saved;
}

static void package_chain(Arena *a, Package *p, const PackageVotes *radial, json_object *out) {
    int n = p->grids_n, runs = 0;
    const char *mode = json_object_get_string(json_object_object_get(p->definition, "winding"));
    json_object *links = json_object_new_array();
    json_object_object_add(out, "links", links);
    size_t saved = a->used;
    if (!p->named) {
        for (int m = 0; m < n; m++) {
            p->grids[m].frame = m;
        }
    } else if (mode && !strcmp(mode, "names")) {
        runs = 1;
        json_object_object_add(
            out, "from", json_object_new_string("names: each mesh anchored by its own name"));
        int next = 1;
        for (int m = 0; m < n; m++) {
            PackageGrid *g = &p->grids[m];
            if (g->wrap < 0) {
                g->frame = next++;
            } else {
                g->frame = 0;
                g->offset = g->wrap - floor(p->sense > 0 ? g->starts : -g->ends);
            }
        }
    } else {
        int *order = push(a, n * sizeof(*order));
        int used = 0;
        for (int m = 0; m < n; m++) {
            PackageGrid *g = &p->grids[m];
            g->frame = -1;
            if (g->wrap >= 0) {
                int j = used++;
                while (j > 0 && p->grids[order[j - 1]].wrap > g->wrap) {
                    order[j] = order[j - 1];
                    j--;
                }
                order[j] = m;
            }
        }
        for (int i = 0; i < used; i++) {
            int m = order[i];
            PackageGrid *g = &p->grids[m];
            if (g->frame < 0) {
                g->frame = runs++;
            }
            int next = -1, winning_delta = 0;
            i64 winning_count = 0, winning_peak = 0;
            for (int j = 0; j < used; j++) {
                int k = order[j];
                PackageGrid *candidate = &p->grids[k];
                if (candidate->wrap != g->end + 1) {
                    continue;
                }
                if (next >= 0 && p->grids[next].frame < 0 && candidate->frame >= 0) {
                    continue;
                }
                i64 count = 0, peak = 0;
                int delta = 0;
                for (i64 x = 0; x < radial->count; x++) {
                    PackageVote v = radial->slot[x];
                    if (v.a == m && v.b == k) {
                        count += v.count;
                        if (v.count > peak) {
                            peak = v.count;
                            delta = v.delta;
                        }
                    }
                }
                if (next < 0 || (p->grids[next].frame >= 0 && candidate->frame < 0) ||
                    count > winning_count) {
                    next = k;
                    winning_count = count;
                    winning_peak = peak;
                    winning_delta = delta;
                }
            }
            if (next >= 0) {
                PackageGrid *candidate = &p->grids[next];
                json_object *item = json_object_new_object();
                json_object_array_add(links, item);
                json_object_object_add(item, "from", json_object_new_int(g->wrap));
                json_object_object_add(item, "to", json_object_new_int(candidate->wrap));
                json_object_object_add(item, "votes", json_object_new_int64(winning_count));
                if (winning_count) {
                    candidate->offset = g->offset + 1 - winning_delta;
                    candidate->frame = g->frame;
                    package_ratio(item, "purity", winning_peak, winning_count);
                    json_object_object_add(item, "step", json_object_new_int(1 - winning_delta));
                }
            }
        }
        int next = 0;
        for (int m = 0; m < n; m++) {
            if (p->grids[m].frame >= next) {
                next = p->grids[m].frame + 1;
            }
        }
        for (int m = 0; m < n; m++) {
            if (p->grids[m].frame < 0) {
                p->grids[m].frame = next++;
            }
        }
    }
    json_object_object_add(out, "runs", json_object_new_int(runs));
    a->used = saved;
}

static PackageAlignment package_align(
    Arena *a, Package *p, PackageVertices *v, const PackageVotes votes[3]) {
    PackageAlignment out = {.chain = json_object_new_object(),
        .links = json_object_new_array(),
        .checks = json_object_new_object(),
        .name_offsets = json_object_new_object()};
    package_chain(a, p, &votes[1], out.chain);
    package_link_frames(a, p, &votes[2], out.links);
    size_t saved = a->used;
    i64 most = 0;
    for (int m = 0; m < p->grids_n; m++) {
        if (p->grids[m].inside > most) {
            most = p->grids[m].inside;
        }
    }
    double *values = push(a, most * sizeof(*values));
    i32 *starts = push(a, p->grids_n * sizeof(*starts));
    PackageVotes anchors = {0};
    for (int m = 0; m < p->grids_n; m++) {
        PackageGrid *g = p->grids + m;
        i64 valid = 0;
        starts[m] = INT32_MIN;
        for (i64 i = g->first; i < g->first + g->inside; i++) {
            v->winding[i] += g->offset;
            v->frame[i] = isfinite(v->winding[i]) ? g->frame : -1;
            if (p->named && g->wrap >= 0 && v->frame[i] >= 0) {
                values[valid++] = v->winding[i];
            }
        }
        if (valid) {
            starts[m] = g->wrap - floor(package_quantile(values, valid, 0.001));
            package_vote_add(&anchors, 0, 0, starts[m], m, 1);
        }
    }
    package_votes_compact(&anchors);
    i64 count = 0;
    for (i64 i = 0; i < anchors.count; i++) {
        PackageVote v = anchors.slot[i];
        if (v.count > count) {
            count = v.count;
            p->anchor = v.delta;
        }
    }
    free(anchors.slot);
    for (i64 i = 0; i < p->count; i++) {
        v->winding[i] += p->anchor;
    }
    for (int m = 0; m < p->grids_n; m++) {
        if (starts[m] != INT32_MIN && starts[m] != p->anchor) {
            json_object_object_add(
                out.name_offsets, p->grids[m].segment, json_object_new_int(starts[m] - p->anchor));
        }
    }
    i64 contact[4] = {0}, radial[4] = {0};
    for (int kind = 0; kind < 2; kind++) {
        for (i64 i = 0; i < votes[kind].count; i++) {
            PackageVote e = votes[kind].slot[i];
            const PackageGrid *x = p->grids + e.a, *y = p->grids + e.b;
            if (x->frame != y->frame) {
                continue;
            }
            int delta = e.delta + y->offset - x->offset;
            if (kind) {
                radial[0] += e.count;
                radial[delta <= 0 ? 3 : delta > 1 ? 2 : 1] += e.count;
            } else {
                contact[0] += e.count;
                contact[delta == 0 ? 1 : abs(delta) == 1 ? 2 : 3] += e.count;
            }
        }
    }
    json_object_object_add(out.checks, "radial_pairs", json_object_new_int64(radial[0]));
    package_ratio(out.checks, "radial_step_1", radial[1], radial[0]);
    package_ratio(out.checks, "radial_step_2_or_more", radial[2], radial[0]);
    package_ratio(out.checks, "radial_step_0_or_less", radial[3], radial[0]);
    json_object_object_add(out.checks, "contact_pairs", json_object_new_int64(contact[0]));
    package_ratio(out.checks, "contact_same_winding", contact[1], contact[0]);
    package_ratio(out.checks, "contact_one_apart", contact[2], contact[0]);
    package_ratio(out.checks, "contact_two_or_more", contact[3], contact[0]);
    a->used = saved;
    return out;
}

static void package_flags_chunk(void *context, i64 chunk) {
    const PackageNeighbors *n = context;
    i64 first = chunk * 256, end = first + 256;
    package_flags_range(context, first, end < n->package->count ? end : n->package->count);
}

static PackageAlignment package_measure(Arena *a, Package *p, PackageVertices *v) {
    size_t saved = a->used;
    int *rank = pushz(a, p->grids_n * sizeof(*rank));
    for (int i = 0; i < p->grids_n; i++) {
        for (int j = 0; j < p->grids_n; j++) {
            i64 x = p->grids[i].inside, y = p->grids[j].inside;
            rank[i] += y > x || (y == x && j < i);
        }
    }
    PackageNeighbors n = {
        .package = p, .vertices = v, .tree = bench_tree(a, v->points, p->count), .rank = rank};
    PackageVotes votes[3] = {{0}};
    if (p->grids_n > 1) {
        package_pair_votes(a, &n, votes);
    }
    PackageAlignment alignment = package_align(a, p, v, votes);
    for (int i = 0; i < 3; i++) {
        free(votes[i].slot);
    }
    if (p->grids_n > 1) {
        parallel((p->count + 255) / 256, 4, package_flags_chunk, &n);
    }
    a->used = saved;
    return alignment;
}

static json_object *package_json_knots(const Package *p) {
    json_object *out = json_object_new_array_ext(p->knot_count);
    for (int i = 0; i < p->knot_count; i++) {
        json_object *knot = json_object_new_array_ext(3);
        for (int d = 0; d < 3; d++) {
            json_object_array_add(knot, json_object_new_double(p->knots[3 * i + d]));
        }
        json_object_array_add(out, knot);
    }
    return out;
}

static json_object *package_file_hashes(char *folder) {
    char *roots[] = {folder, NULL};
    FTS *tree = fts_open(roots, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
    need(tree != NULL, "cannot list new package %s", folder);
    size_t prefix = strlen(folder) + 1;
    json_object *out = json_object_new_object();
    for (FTSENT *p; (p = fts_read(tree));) {
        if (p->fts_info == FTS_F) {
            char hex[65];
            need(bench_hash_of(p->fts_path, hex), "cannot hash %s", p->fts_path);
            json_object_object_add(out, p->fts_path + prefix, json_object_new_string(hex));
        }
    }
    fts_close(tree);
    return out;
}

static json_object *package_record(const Package *p, const PackageVertices *v,
    const PackageAlignment *alignment, const i64 *crossing_counts, double elapsed) {
    json_object *out = json_object_new_object(), *dropped = json_object_new_object(),
                *turns = json_object_new_object(), *frames = json_object_new_object(),
                *zones = json_object_new_object(), *flags = json_object_new_object();
    char head[128], dirty[128], revision[136];
    bench_git(head, 0);
    bench_git(dirty, 1);
    snprintf(revision, sizeof(revision), "%s%s", *head ? head : "unknown", *dirty ? "+dirty" : "");
    char built[32];
    time_t wall = time(NULL);
    struct tm local;
    localtime_r(&wall, &local);
    strftime(built, sizeof(built), "%Y-%m-%d %H:%M:%S", &local);
    json_object_object_add(out, "name", json_object_new_string(p->name));
    json_object_object_add(out, "version", json_object_new_int(5));
    json_object_object_add(out, "built", json_object_new_string(built));
    json_object_object_add(out, "seconds", json_object_new_double(package_round(elapsed, 1)));
    json_object_object_add(out, "code", json_object_new_string(revision));
    json_object *definition = json_object_new_object();
    json_object_object_foreach(p->definition, key, value) {
        json_object_object_add(definition, key, json_object_get(value));
    }
    json_object_object_add(definition, "name", json_object_new_string(p->name));
    json_object_object_add(out, "definition", definition);
    json_object_object_add(out, "scan", json_object_get(p->entry));
    json_object_object_add(out, "sources", json_object_get(p->sources));
    json_object_object_add(out, "meshes_in_range", json_object_new_int(p->grids_n));
    json_object_object_add(out, "named", json_object_new_boolean(p->named));
    json_object_object_add(out, "sense", json_object_new_int(p->sense));
    json_object_object_add(out, "sense_votes", json_object_get(p->votes));
    const char *sense_source = ji64(p->votes, "rows")
        ? "rows passing their start angle again"
        : "angle-radius correlation (no row spans a turn)";
    json_object_object_add(out, "sense_source", json_object_new_string(sense_source));
    json_object_object_add(out, "axis_source", json_object_new_string(p->axis_source));
    json_object_object_add(out, "spacing_um", json_object_new_double(package_round(p->spacing, 2)));
    json_object_object_add(out, "unwrap_dropped_share", dropped);
    json_object_object_add(out, "turns_in_range", turns);
    json_object_object_add(out, "chain", json_object_get(alignment->chain));
    json_object_object_add(out, "frame_links", json_object_get(alignment->links));
    json_object_object_add(out, "frames", frames);
    json_object_object_add(out, "anchor", json_object_new_int(p->anchor));
    json_object_object_add(out, "name_start_offsets", json_object_get(alignment->name_offsets));
    json_object_object_add(out, "checks", json_object_get(alignment->checks));
    json_object_object_add(out, "zones", zones);
    json_object_object_add(out, "flags", flags);
    for (int m = 0; m < p->grids_n; m++) {
        const PackageGrid *g = &p->grids[m];
        if (g->dropped) {
            json_object_object_add(
                dropped, g->segment, json_object_new_double(package_round(g->dropped, 4)));
        }
        double low = INFINITY, high = -INFINITY;
        for (i64 i = g->first; i < g->first + g->inside; i++) {
            low = fmin(low, v->winding[i]);
            high = fmax(high, v->winding[i]);
        }
        if (isfinite(low)) {
            json_object_object_add(
                turns, g->segment, json_object_new_double(package_round(high - low, 3)));
        }
        json_object_object_add(frames, g->segment, json_object_new_int(g->frame));
    }
    const char *flag_names[] = {"duplicate", "pressed", "overlap_next", "overlap_previous",
        "overlap_other", "conflict", "order", "touching"};
    i64 bits[8] = {0}, census[256] = {0}, kept[256] = {0}, pressured[256] = {0};
    i64 unjudged[256] = {0}, touched[256] = {0};
    double areas[256] = {0};
    for (i64 i = 0; i < p->count; i++) {
        int z = v->zone[i], f = v->flags[i];
        census[z]++;
        for (int d = 0; d < 8; d++) {
            bits[d] += (f >> d) & 1;
        }
        if (!(f & 1)) {
            kept[z]++;
            areas[z] += v->area[i];
            pressured[z] += (f & 2) != 0;
            unjudged[z] += (f & (32 | 64)) != 0;
            touched[z] += (f & 128) != 0;
        }
    }
    for (int i = 0; i < 8; i++) {
        json_object_object_add(flags, flag_names[i], json_object_new_int64(bits[i]));
    }
    for (size_t z = 0; z < sizeof(bench_zones) / sizeof(*bench_zones); z++) {
        if (!census[z]) {
            continue;
        }
        json_object *item = json_object_new_object();
        json_object_object_add(zones, bench_zones[z], item);
        json_object_object_add(item, "vertices", json_object_new_int64(census[z]));
        json_object_object_add(
            item, "area_cm2", json_object_new_double(package_round(areas[z] / 100, 2)));
        package_ratio(item, "pressed_share", pressured[z], kept[z]);
        package_ratio(item, "unjudged_share", unjudged[z], kept[z]);
        package_ratio(item, "touching_share", touched[z], kept[z]);
        json_object_object_add(item, "crossings", json_object_new_int64(crossing_counts[z]));
    }
    return out;
}

int bench_package_build_command(int argc, char **argv) {
    need(argc >= 1, "calibrate.bench.package build CONTROL [...]");
    Arena a = arena(96ull << 30);
    json_object *manifest = json_read(fmt(&a, "%s/calibrate/bench/controls.json", HOME));
    need(manifest != NULL, "cannot read bench controls");
    const char *root = fmt(&a, "%s/bench/controls", OUTPUTS);
    mkdirs(root);
    size_t base = a.used;
    for (int i = 0; i < argc; i++) {
        double started = now();
        const char *name = argv[i];
        const char *out = fmt(&a, "%s/%s", root, name);
        need(!exists(out), "%s exists; a package is frozen once built", out);
        Package p = package_open(&a, manifest, name);
        need(package_core_axis(&a, &p) || package_ring_axis(&a, &p) ||
                package_legacy_axis(&a, &p) || package_fitted_axis(&a, &p),
            "cannot fit control axis");
        for (int m = 0; m < p.grids_n; m++) {
            package_unwrap(&a, &p, p.grids + m);
        }
        package_handedness(&a, &p);
        PackageVertices vertices = package_vertices(&a, &p);
        PackageAlignment alignment = package_measure(&a, &p, &vertices);
        package_areas(&a, &p, &vertices);
        char *stage = fmt(&a, "%s/.package.XXXXXXXX", root);
        need(mkdtemp(stage) != NULL, "cannot stage bench control: %s", strerror(errno));
        Array names = package_names(&a, &p);
        package_vertex_archive(&a, stage, &p, &vertices, names);
        i64 crossing_counts[256] = {0};
        package_crossing_archive(&a, stage, &p, &vertices, names, crossing_counts);
        for (int m = 0; m < p.grids_n; m++) {
            package_mesh(&a, stage, &p, &vertices, p.grids + m);
            json_object_put(p.grids[m].raw.meta);
        }
        json_object *axis = json_object_new_object();
        json_object_object_add(axis, "knots_z_x_y", package_json_knots(&p));
        json_object_object_add(axis, "source", json_object_new_string(p.axis_source));
        json_write(fmt(&a, "%s/axis.json", stage), axis);
        json_object_put(axis);
        json_object *record =
            package_record(&p, &vertices, &alignment, crossing_counts, now() - started);
        json_object_object_add(record, "files", package_file_hashes(stage));
        json_write(fmt(&a, "%s/control.json", stage), record);
        need(rename(stage, out) == 0, "cannot publish %s: %s", out, strerror(errno));
        json_object *report = json_object_new_object();
        const char *keys[] = {
            "name", "seconds", "meshes_in_range", "named", "sense", "checks", "zones"};
        for (size_t k = 0; k < sizeof(keys) / sizeof(*keys); k++) {
            json_object_object_add(report, keys[k], json_object_get(jget(record, keys[k])));
        }
        puts(board_json(report));
        json_object_put(report);
        json_object_put(record);
        json_object_put(p.sources);
        json_object_put(p.votes);
        json_object_put(alignment.chain);
        json_object_put(alignment.links);
        json_object_put(alignment.checks);
        json_object_put(alignment.name_offsets);
        a.used = base;
    }
    json_object_put(manifest);
    munmap(a.base, a.size);
    return 0;
}
