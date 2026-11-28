enum {
    FIDELITY_DEPTH = 16,
    FIDELITY_CUT = 32,
    FIDELITY_PATCH = 2 * FIDELITY_DEPTH * FIDELITY_CUT,
    FIDELITY_REACH = 24,
    FIDELITY_SNAP = 6,
    FIDELITY_PROFILE = 2 * FIDELITY_REACH + 1,
    FIDELITY_REAL = 120,
    FIDELITY_FRAGMENTS = 3,
    FIDELITY_STEPS = 600,
    FIDELITY_BATCH = 64
};

typedef struct {
    double profile[FIDELITY_PROFILE], inward, outward;
    float cut[FIDELITY_PATCH];
    int block;
    u8 injected, second;
} FidelityRow;

typedef struct {
    FidelityRow *data;
    size_t count, capacity;
    double narrow[17], broad[33];
} FidelityRows;

static void fidelity_gaussian(double *kernel, int sigma) {
    int radius = sigma * 4;
    for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] = exp(-.5 / (sigma * sigma) * (i * i));
    }
    double sum = syn_sum(kernel, 2 * radius + 1);
    for (int i = 0; i <= 2 * radius; i++) {
        kernel[i] /= sum;
    }
}

static double fidelity_smooth(
    const double *profile, int n, int at, const double *kernel, int radius) {
    double sum = 0;
    for (int j = -radius; j <= radius; j++) {
        int i = at + j;
        i = i < 0 ? -1 - i : i >= n ? 2 * n - 1 - i : i;
        sum += profile[i] * kernel[j + radius];
    }
    return sum;
}

static float fidelity_sample(const u8 *ct, SlabPoint point) {
    for (int i = 0; i < 3; i++) {
        point.v[i] = fmax(0, fmin(fragment_dims[i] - 1, point.v[i]));
    }
    return fragment_ct_linear(ct, point);
}

static void fidelity_measure(FidelityRows *rows, const u8 *ct, const SlabPoint *points,
    const SlabPoint *normals, size_t count, int block, int injected) {
    enum { RADIUS = FIDELITY_REACH + FIDELITY_SNAP, RAW = 2 * RADIUS + 1 };
    need(count <= rows->capacity - rows->count, "too many fidelity samples");
    for (size_t i = 0; i < count; i++) {
        SlabPoint p = points[i], n = normals[i];
        double raw[RAW];
        for (int j = -RADIUS; j <= RADIUS; j++) {
            SlabPoint at;
            for (int k = 0; k < 3; k++) {
                at.v[k] = p.v[k] + .5 * j * n.v[k];
            }
            raw[j + RADIUS] = fidelity_sample(ct, at);
        }
        float peak = -INFINITY;
        int snap = 0;
        for (int j = -FIDELITY_SNAP; j <= FIDELITY_SNAP; j++) {
            float value = fidelity_smooth(raw, RAW, j + RADIUS, rows->narrow, 8);
            if (value > peak) {
                peak = value;
                snap = j;
            }
        }
        for (int k = 0; k < 3; k++) {
            p.v[k] += .5 * snap * n.v[k];
        }
        FidelityRow *row = rows->data + rows->count++;
        row->block = block;
        row->injected = injected;
        memcpy(row->profile, raw + FIDELITY_SNAP + snap, sizeof(row->profile));
        row->second =
            fragment_sheet_rule(row->profile, FIDELITY_PROFILE, .5, &row->inward, &row->outward);
        double width = fmax(hypot(n.v[1], n.v[2]), 1e-9);
        SlabPoint eu = {{0, -n.v[2] / width, n.v[1] / width}};
        SlabPoint ev = {{n.v[1] * eu.v[2] - n.v[2] * eu.v[1], -n.v[0] * eu.v[2], n.v[0] * eu.v[1]}};
        size_t offset = 0;
        for (int side = 0; side < 2; side++) {
            SlabPoint along = side ? ev : eu;
            for (int y = 0; y < FIDELITY_DEPTH; y++) {
                for (int x = 0; x < FIDELITY_CUT; x++) {
                    SlabPoint at;
                    for (int k = 0; k < 3; k++) {
                        at.v[k] = p.v[k] + (y - 7.5) * n.v[k] + (x - 15.5) * along.v[k];
                    }
                    row->cut[offset++] = fidelity_sample(ct, at);
                }
            }
        }
    }
}

static void fidelity_describe(
    const FidelityRows *rows, const FidelityRow *row, double air, double *result) {
    const double *profile = row->profile;
    const int mid = FIDELITY_REACH;
    double peak = fmax(profile[mid], fmax(profile[mid - 1], profile[mid + 1]));
    double half = air + .5 * (peak - air);
    int left = 0, right = 0;
    for (int k = 0; k <= mid; k++) {
        if (profile[mid - k] < half) {
            left = k;
            break;
        }
    }
    for (int k = 0; k <= mid; k++) {
        if (profile[mid + k] < half) {
            right = k;
            break;
        }
    }
    double residual[18], mean = 0, variance = 0;
    for (int i = 0, k = 0; i < FIDELITY_PROFILE; i++) {
        int offset = abs(i - mid);
        if (offset >= 12 && offset <= 20) {
            residual[k] =
                profile[i] - fidelity_smooth(profile, FIDELITY_PROFILE, i, rows->broad, 16);
            mean += residual[k++];
        }
    }
    mean /= 18;
    for (int k = 0; k < 18; k++) {
        double d = residual[k] - mean;
        variance += d * d;
    }
    result[0] = peak;
    result[1] = .5 * (left + right);
    result[2] = .5 * (profile[mid - 10] + profile[mid + 10]);
    result[3] = sqrt(variance / 18);
}

static json_object *fidelity_discriminate(
    Arena *a, const FidelityRows *rows, const u32 *ids, size_t count, u64 seed, int control) {
    size_t mark = a->used;
    int *group = push(a, count * sizeof(*group));
    int groups = 0, prev = -1;
    u8 *label = push(a, count);
    Pcg64 fake = pcg64(seed + 1);
    float *patch = push(a, count * FIDELITY_PATCH * sizeof(*patch));
    for (size_t i = 0; i < count; i++) {
        const FidelityRow *r = rows->data + ids[i];
        if (r->block != prev) {
            prev = r->block;
            groups++;
        }
        group[i] = groups - 1;
        label[i] = control ? draw(&fake, 2) : !r->injected;
        for (int j = 0; j < FIDELITY_PATCH; j++) {
            patch[i * FIDELITY_PATCH + j] = r->cut[j] / 255.0f;
        }
    }
    int train_groups = nearbyint(.7 * groups);
    if (groups == train_groups || !train_groups) {
        a->used = mark;
        return NULL;
    }
    u8 *train = pushz(a, groups);
    u32 *picks = push(a, (size_t)train_groups * sizeof(*picks));
    Pcg64 rng = pcg64(seed);
    syn_sample_nodes(a, &rng, groups, train_groups, picks);
    for (int i = 0; i < train_groups; i++) {
        train[picks[i]] = 1;
    }
    u32 *class[2] = {push(a, count * sizeof(u32)), push(a, count * sizeof(u32))};
    size_t amount[2] = {0}, held[2] = {0};
    for (size_t i = 0; i < count; i++) {
        int y = label[i];
        if (train[group[i]]) {
            class[y][amount[y]++] = i;
        } else {
            held[y]++;
        }
    }
    if (!amount[0] || !amount[1] || !held[0] || !held[1]) {
        a->used = mark;
        return NULL;
    }
    size_t tests = held[0] + held[1], scores = 0;
    DiscScore *predictions = push(a, tests * sizeof(*predictions));
    SliceNoise noise = slices_noise_seed(seed);
    DiscNetwork net = disc_network(a, &noise, 1);
    size_t scratch = a->used;
    for (int step = 0; step < FIDELITY_STEPS; step++) {
        u32 batch[FIDELITY_BATCH];
        for (int k = 0; k < FIDELITY_BATCH; k++) {
            int y = k < FIDELITY_BATCH / 2;
            batch[k] = class[y][draw(&rng, amount[y])];
        }
        int flip = pcg64_double(&rng) < .5;
        for (int k = 0; k < FIDELITY_BATCH; k++) {
            u32 i = batch[k];
            disc_predict(a, &net, patch + i * FIDELITY_PATCH, label[i], FIDELITY_BATCH, flip);
            a->used = scratch;
        }
        disc_step(&net, step);
    }
    for (size_t i = 0; i < count; i++) {
        if (!train[group[i]]) {
            float score = disc_predict(a, &net, patch + i * FIDELITY_PATCH, label[i], 0, 0);
            predictions[scores++] = (DiscScore){.score = score, .label = label[i]};
            a->used = scratch;
        }
    }
    json_object *out = json_object_new_object();
    syn_add(out, "auc", decimal_round(disc_auc(predictions, scores), 3));
    json_object_object_add(out, "train", json_object_new_int64(count - tests));
    json_object_object_add(out, "test", json_object_new_int64(tests));
    a->used = mark;
    return out;
}

static void fidelity_real(Arena *a, FidelityRows *rows, const InjectionFragments *lib,
    const float *normals, const FragmentBlock *block, SlabPoint origin, Pcg64 *rng, int group) {
    const int shape[] = {FRAGMENT_Z, FRAGMENT_Y, FRAGMENT_X};
    const float *xyz = lib->scan.points.data;
    u32 *eligible = push(a, lib->kept * sizeof(*eligible));
    u32 count = 0;
    for (size_t i = 0; i < lib->kept; i++) {
        u32 j = lib->keep[i];
        SlabPoint zyx;
        int keep = 1;
        for (int k = 0; k < 3; k++) {
            zyx.v[k] = (double)xyz[3ull * j + 2 - k] - origin.v[2 - k];
            keep &= zyx.v[k] >= 44 && zyx.v[k] <= shape[k] - 44;
        }
        for (int p = 0; keep && p < block->count; p++) {
            const FragmentPlacement *placed = block->placed + p;
            double squared = 0, reach = placed->frag.reach + 44;
            for (int k = 0; k < 3; k++) {
                double d = zyx.v[k] - placed->site.centre.v[k];
                squared += d * d;
            }
            keep = squared > reach * reach;
        }
        if (keep) {
            eligible[count++] = j;
        }
    }
    u32 chosen[FIDELITY_REAL];
    int n = count < FIDELITY_REAL ? (int)count : FIDELITY_REAL;
    if (count > FIDELITY_REAL) {
        syn_sample_nodes(a, rng, count, FIDELITY_REAL, chosen);
    }
    SlabPoint points[FIDELITY_REAL], normal[FIDELITY_REAL];
    for (int i = 0; i < n; i++) {
        u32 j = eligible[count > FIDELITY_REAL ? chosen[i] : i];
        const float *p = xyz + 3ull * j, *v = normals + 3ull * j;
        SlabPoint axis = slab_axis(lib->axis, lib->knots, p[2]);
        double dot = v[0] * (axis.v[1] - p[0]) + v[1] * (axis.v[2] - p[1]);
        double norm = sqrt((double)v[0] * v[0] + (double)v[1] * v[1] + (double)v[2] * v[2]);
        norm = dot < 0 ? -norm : norm;
        for (int k = 0; k < 3; k++) {
            points[i].v[k] = p[2 - k] - origin.v[2 - k];
            normal[i].v[k] = v[2 - k] / norm;
        }
    }
    fidelity_measure(rows, block->ct, points, normal, n, group, 0);
}

static double fidelity_ks(const double *a, size_t nr, const double *b, size_t ns) {
    double distance = 0;
    for (size_t i = 0, j = 0; i < nr || j < ns;) {
        double value = fmin(i < nr ? a[i] : INFINITY, j < ns ? b[j] : INFINITY);
        while (i < nr && a[i] <= value) {
            i++;
        }
        while (j < ns && b[j] <= value) {
            j++;
        }
        distance = fmax(distance, fabs((double)i / nr - (double)j / ns));
    }
    return distance;
}

static json_object *fidelity_report(Arena *a, const FidelityRows *rows, const char *scroll,
    double air, int blocks, int fragments, u64 seed) {
    size_t n = rows->count, count[2] = {0}, rejected[2] = {0}, taken = 0;
    u32 *single = push(a, n * sizeof(*single)), *real = push(a, n * sizeof(*real));
    double mean[2][FIDELITY_PROFILE] = {{0}};
    double *metric[2][4];
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 4; j++) {
            metric[k][j] = push(a, n * sizeof(double));
        }
    }
    for (size_t i = 0; i < n; i++) {
        const FidelityRow *r = rows->data + i;
        int k = r->injected;
        if (r->second || r->inward < 10 || r->outward < 10) {
            rejected[k]++;
            continue;
        }
        single[taken++] = i;
        if (!k) {
            real[count[k]] = i;
        }
        double stats[4];
        fidelity_describe(rows, r, air, stats);
        for (int j = 0; j < FIDELITY_PROFILE; j++) {
            mean[k][j] += r->profile[j];
        }
        for (int j = 0; j < 4; j++) {
            metric[k][j][count[k]] = stats[j];
        }
        count[k]++;
    }
    need(count[0] && count[1], "fidelity needs isolated real and injected points");
    json_object *out = json_object_new_object(), *measures = json_object_new_object();
    json_object_object_add(out, "scroll", json_object_new_string(scroll));
    json_object_object_add(out, "blocks", json_object_new_int(blocks));
    json_object_object_add(out, "fragments", json_object_new_int(fragments));
    json_object *points = json_object_new_object(), *excluded = json_object_new_object();
    const char *classes[] = {"real", "injected"};
    for (int k = 0; k < 2; k++) {
        json_object_object_add(points, classes[k], json_object_new_int64(count[k]));
        json_object_object_add(excluded, classes[k], json_object_new_int64(rejected[k]));
    }
    json_object_object_add(out, "points", points);
    json_object_object_add(out, "points_not_single", excluded);
    json_object *offsets = json_object_new_array();
    json_object *profile[] = {json_object_new_array(), json_object_new_array()};
    for (int i = 0; i < FIDELITY_PROFILE; i += 2) {
        json_object_array_add(offsets, json_object_new_int((i - FIDELITY_REACH) / 2));
        for (int k = 0; k < 2; k++) {
            json_object_array_add(profile[k], jnum(decimal_round(mean[k][i] / count[k], 1)));
        }
    }
    json_object_object_add(out, "profile_offsets", offsets);
    json_object_object_add(out, "profile_real", profile[0]);
    json_object_object_add(out, "profile_injected", profile[1]);
    const char *keys[] = {"peak", "width", "flank", "outer_noise"};
    for (int j = 0; j < 4; j++) {
        json_object *m = json_object_new_object();
        for (int k = 0; k < 2; k++) {
            double *stats = metric[k][j];
            qsort(stats, count[k], sizeof(*stats), syn_double_order);
            double median = realism_percentile(stats, count[k], .5);
            syn_add(m, fmt(a, "%s_median", classes[k]), decimal_round(median, 2));
            json_object *iqr = json_object_new_array_ext(2);
            for (int q = 1; q <= 3; q += 2) {
                double value = realism_percentile(stats, count[k], q / 4.0);
                json_object_array_add(iqr, jnum(decimal_round(value, 2)));
            }
            json_object_object_add(m, fmt(a, "%s_iqr", classes[k]), iqr);
        }
        double ks = fidelity_ks(metric[0][j], count[0], metric[1][j], count[1]);
        syn_add(m, "ks", decimal_round(ks, 3));
        json_object_object_add(measures, keys[j], m);
    }
    json_object_object_add(out, "measures", measures);
    json_object_object_add(
        out, "discriminator", fidelity_discriminate(a, rows, single, taken, seed, 0));
    json_object_object_add(
        out, "control_real_vs_real", fidelity_discriminate(a, rows, real, count[0], seed, 1));
    return out;
}

static void fidelity_archive(Arena *a, const FidelityRows *rows, const char *path) {
    size_t n = rows->count;
    double *profile = push(a, n * FIDELITY_PROFILE * sizeof(double));
    double *in = push(a, n * sizeof(double)), *out = push(a, n * sizeof(double));
    u8 *second = push(a, n), *injected = push(a, n);
    float *cut = push(a, n * FIDELITY_PATCH * sizeof(float));
    i64 *block = push(a, n * sizeof(i64));
    for (size_t i = 0; i < n; i++) {
        const FidelityRow *r = rows->data + i;
        memcpy(profile + i * FIDELITY_PROFILE, r->profile, sizeof(r->profile));
        memcpy(cut + i * FIDELITY_PATCH, r->cut, sizeof(r->cut));
        in[i] = r->inward;
        out[i] = r->outward;
        second[i] = r->second;
        injected[i] = r->injected;
        block[i] = r->block;
    }
    Array features[] = {{.descr = "<f8",
                            .ndim = 2,
                            .shape = {n, FIDELITY_PROFILE},
                            .size = n * FIDELITY_PROFILE * sizeof(double),
                            .data = profile},
        {.descr = "<f8", .ndim = 1, .shape = {n}, .size = n * sizeof(double), .data = in},
        {.descr = "<f8", .ndim = 1, .shape = {n}, .size = n * sizeof(double), .data = out},
        {.descr = "|b1", .ndim = 1, .shape = {n}, .size = n, .data = second},
        {.descr = "<f4",
            .ndim = 4,
            .shape = {n, 2, FIDELITY_DEPTH, FIDELITY_CUT},
            .size = n * FIDELITY_PATCH * sizeof(float),
            .data = cut},
        {.descr = "|b1", .ndim = 1, .shape = {n}, .size = n, .data = injected},
        {.descr = "<i8", .ndim = 1, .shape = {n}, .size = n * sizeof(i64), .data = block}};
    const char *keys[] = {"profiles", "room_in", "room_out", "second", "cuts", "injected", "block"};
    npz_write_codec(path, 7, keys, features, ZIP_CM_DEFLATE);
}

static int injection_fidelity_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.injection.fidelity check SCROLL "
                        "[--blocks N] [--seed N] [--out OUT.json] [--rows OUT.npz]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(argc >= 2 && !strcmp(argv[0], "check"), "%s", usage);
    int blocks = 24;
    u64 seed = 0;
    const char *path = NULL, *archive = NULL;
    for (int i = 2; i < argc; i++) {
        char split[32];
        const char *value;
        const char *key = slices_cli_option(argv[i], split, &value);
        if (!value) {
            need(i + 1 < argc, "%s", usage);
            value = argv[++i];
        }
        if (!strcmp(key, "--blocks")) {
            blocks = slices_positive(value, key);
        } else if (!strcmp(key, "--seed")) {
            seed = slices_unsigned(value, key);
        } else if (!strcmp(key, "--out")) {
            path = value;
        } else if (!strcmp(key, "--rows")) {
            archive = value;
        } else {
            die("%s", usage);
        }
    }
    double start = now();
    Arena a = arena(64 * GB);
    const char *scroll = argv[1], *package = fmt(&a, "%s-train", scroll);
    InjectionFragments lib = fragment_open(&a, scroll, package);
    Array norms = npz_get(&a, fmt(&a, "%s/truth.npz", lib.scan.folder), "normals");
    need(norms.ndim == 2 && norms.shape[1] == 3 && norms.shape[0] == lib.scan.points.shape[0] &&
            !strcmp(norms.descr, "<f4") && norms.shape[0] <= UINT32_MAX,
        "%s: invalid truth normals", package);
    size_t capacity = (size_t)blocks * (FIDELITY_REAL + FIDELITY_FRAGMENTS * 64 * 64);
    need(capacity <= UINT32_MAX && blocks <= INT_MAX / 40, "too many fidelity blocks");
    FidelityRows rows = {.data = push(&a, capacity * sizeof(*rows.data)), .capacity = capacity};
    fidelity_gaussian(rows.narrow, 2);
    fidelity_gaussian(rows.broad, 4);
    size_t mark = a.used;
    int tries = 40 * blocks, kept = 0, fragments = 0;
    Pcg64 rng = pcg64(seed);
    for (int i = 0; i < blocks; i++) {
        SlabPoint origin;
        u8 *real = fragment_real_block(&a, &lib, &rng, &tries, &origin);
        if (!real) {
            break;
        }
        FragmentBlock injected =
            fragment_inject_block(&a, &lib, real, origin, FIDELITY_FRAGMENTS, &rng);
        if (!injected.count) {
            a.used = mark;
            continue;
        }
        size_t before = rows.count;
        for (int j = 0; j < injected.count; j++) {
            SlabPoint *points, *normals;
            size_t n = fragment_mid_surface(&a, injected.placed + j, 4, 18, &points, &normals);
            fidelity_measure(&rows, injected.ct, points, normals, n, i, 1);
        }
        fidelity_real(&a, &rows, &lib, norms.data, &injected, origin, &rng, i);
        kept += rows.count != before;
        fragments += injected.count;
        printf("block %d: %d fragments\n", i, injected.count);
        fflush(stdout);
        fragment_dispose(&injected);
        a.used = mark;
    }
    double air = jf64(jget(lib.summary, "air"), "level");
    fragment_close(&lib);
    json_object *report = fidelity_report(&a, &rows, scroll, air, kept, fragments, seed);
    json_object_object_add(report, "seconds", json_object_new_int64(nearbyint(now() - start)));
    if (path) {
        json_write(path, report);
    }
    if (archive) {
        fidelity_archive(&a, &rows, archive);
    }
    puts(json_object_to_json_string_ext(report, JSON_C_TO_STRING_PRETTY));
    json_object_put(report);
    munmap(a.base, a.size);
    return 0;
}
