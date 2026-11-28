#include <ctype.h>

#ifndef HX_UNITY
const char hx_usage[] = "Fit synthetic scroll settings to real crops and training geometry.";
#endif

enum { SYN_TUNE_N = sizeof(syn_tuning) / sizeof(*syn_tuning), SYN_TUNE_VIEWS = 3 };

typedef struct {
    const char *name;
    double voxel, pitch[5], pressed;
    int target;
} SynTuneProfile;

static const SynTuneProfile syn_tune_profiles[] = {
    {"PHerc0139", 9.362, {45, 113, 171, 261, 509}, .31, 0},
    {"PHerc1667", 7.91, {71, 149, 232, 358, 676}, .20, 0},
    {"PHerc0172", 7.91, {44, 118, 175, 266, 548}, .17, 0},
    {"PHercParis4", 7.91, {55, 133, 209, 336, 759}, .35, 0},
    {"9.362", 9.362, {0}, 0, 0},
    {"7.91", 7.91, {0}, 0, 0},
    {"PHerc0211", 9.362, {0}, 0, 1},
    {"PHerc0125", 9.362, {0}, 0, 1},
    {"PHerc0826", 9.362, {0}, 0, 1},
    {"PHerc1545", 9.362, {0}, 0, 1},
    {"PHerc0846A", 9.362, {0}, 0, 1},
    {"PHerc0175A", 8.64, {0}, 0, 1},
    {"PHerc1218", 8.64, {0}, 0, 1},
    {"PHerc0268", 8.64, {0}, 0, 1},
};

static const char *syn_tune_keys[] = {"xy", "zx", "xy0"};

typedef struct {
    double *rows[SYN_TUNE_VIEWS];
    size_t count[SYN_TUNE_VIEWS];
} SynTuneRows;

typedef struct {
    json_object *entry;
    double score;
    int index;
} SynTuneCandidate;

static const SynTuneProfile *syn_tune_profile(const char *name) {
    for (size_t i = 0; i < sizeof(syn_tune_profiles) / sizeof(*syn_tune_profiles); i++) {
        if (!strcmp(name, syn_tune_profiles[i].name)) {
            return &syn_tune_profiles[i];
        }
    }
    die("unknown tuning profile %s", name);
    return NULL;
}

static double syn_tune_value(int i, double x) {
    SynTuning t = syn_tuning[i];
    if (t.log_scale) {
        return exp(log(t.lo) + x * (log(t.hi) - log(t.lo)));
    }
    return t.lo + x * (t.hi - t.lo);
}

static double syn_tune_position(int i, double x) {
    SynTuning t = syn_tuning[i];
    if (t.log_scale) {
        return syn_clip((log(fmax(x, 1e-12)) - log(t.lo)) / (log(t.hi) - log(t.lo)), 0, 1);
    }
    return syn_clip((x - t.lo) / (t.hi - t.lo), 0, 1);
}

static json_object *syn_tune_spec(
    Arena *a, json_object *settings, double voxel, int oval, i64 seed) {
    json_object *input = json_tokener_parse(
        "{\"margin_um\":300,\"roll\":{\"core_um\":1500,\"delamination_um\":[5,30]},"
        "\"flows\":[{\"scale_um\":6000,\"amp_um\":120},"
        "{\"scale_um\":2500,\"amp_um\":60,\"z_scale_um\":1000}]}");
    json_object *spec = syn_complete(a, input);
    json_object_put(input);
    json_object_object_add(
        spec, "family", json_object_new_string(oval ? "tune/oval" : "tune/round"));
    json_object_object_add(spec, "seed", json_object_new_int64(seed));
    syn_add(spec, "voxel_um", voxel);
    syn_add_pair(spec, "band_um", 0, REALISM_SIDE * 2 * voxel + 300);
    json_object *crush = jget(jget(spec, "shape"), "crush");
    syn_add(crush, "kappa", oval ? .6 : 1);
    syn_add(crush, "kappa_core", oval ? .4 : .9);
    for (int i = 0; i < SYN_TUNE_N; i++) {
        SynTuning t = syn_tuning[i];
        json_object *value = json_object_object_get(settings, t.name);
        if (!value) {
            continue;
        }
        json_object *section = jget(spec, t.container);
        if (!strcmp(t.container, "flows")) {
            json_object_object_add(
                json_object_array_get_idx(section, t.pos), t.key, json_object_get(value));
        } else if (t.pos < 0) {
            json_object_object_add(section, t.key, json_object_get(value));
        } else {
            json_object_array_put_idx(jget(section, t.key), t.pos, json_object_get(value));
        }
    }
    json_object *roll = jget(spec, "roll"), *delam = jget(roll, "delamination_um");
    double low = json_object_get_double(json_object_array_get_idx(delam, 0));
    double high = json_object_get_double(json_object_array_get_idx(delam, 1));
    json_object_array_put_idx(delam, 0, jnum(fmin(low, high)));
    double pitch = jf64(roll, "thickness_um") + jf64(roll, "gap_um");
    syn_add(roll, "turns", syn_clip(10500 / pitch, 15, 60));
    return spec;
}

static double syn_tune_geometry(
    Arena *a, const SynTuneProfile *profile, const SynScroll *scroll, json_object *spec) {
    if (!profile->pitch[0]) {
        return 0;
    }
    json_object *out = json_object_new_object();
    syn_sheet_checks(a, scroll, spec, 30000, out);
    json_object *pitches = json_object_object_get(out, "radial_pitch_um");
    double result = 25;
    if (pitches) {
        double d = (jf64(out, "pressed") - profile->pressed) / .1;
        result = d * d;
        for (int i = 0; i < 5; i++) {
            double v = json_object_get_double(json_object_array_get_idx(pitches, i));
            double r = log(v / profile->pitch[i]) / .2;
            result += r * r / 5;
        }
    }
    json_object_put(out);
    return result;
}

static SynTuneRows syn_tune_real(Arena *a, const char *root, const SynTuneProfile *profile) {
    SynTuneRows real = {0};
    const char *look = isdigit((unsigned char)profile->name[0]) ? NULL : profile->name;
    for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
        const char *path = syn_realism_path(a, root, look, profile->voxel, v != 2, v == 1);
        if (exists(path)) {
            real.count[v] =
                syn_realism_real(a, root, look, profile->voxel, v != 2, v == 1, &real.rows[v]);
        }
    }
    return real;
}

static SynTuneRows syn_tune_allocate(Arena *a, const SynTuneRows *real, int crops) {
    SynTuneRows rows = {0};
    for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
        if (real->count[v]) {
            rows.rows[v] = push(a, 2 * (size_t)crops * REALISM_FEATURES * sizeof(double));
        }
    }
    need(rows.rows[0] || rows.rows[1] || rows.rows[2], "no cached realism views");
    return rows;
}

static SliceCrops syn_tune_real_patches(Arena *a, const SynTuneProfile *profile) {
    const int count = 160;
    int group = profile->voxel == 7.91;
    size_t maximum = count * (group ? 3 : 1);
    SliceCrops real = {.data = push(a, (size_t)count * SLICE_PATCH_PIXELS * sizeof(float))};
    size_t mark = a->used;
    const char **names = push(a, maximum * sizeof(*names));
    u8 *images = push(a, maximum * REALISM_N);
    size_t found = realism_real_crops(a, group, count, 1, 0, images, names, 0);
    Pcg64 rng = pcg64(0);
    for (size_t i = 0; i < found; i++) {
        if (!strcmp(profile->name, names[i])) {
            disc_patch(real.data + real.count++ * SLICE_PATCH_PIXELS, images + i * REALISM_N,
                REALISM_SIDE, REALISM_SIDE, &rng);
        }
    }
    need(real.count, "no cached crops for %s", profile->name);
    a->used = mark;
    return real;
}

static double syn_tune_evaluate(Arena *a, const SynTuneProfile *profile, const SynTuneRows *real,
    SynTuneRows *rows, json_object *settings, int crops, i64 seed, int workers,
    const SliceCrops *patches, json_object **cnn) {
    size_t mark = a->used;
    memset(rows->count, 0, sizeof(rows->count));
    u8 *images = push(a, (size_t)crops * REALISM_N);
    SliceCrops synthetic = {0};
    if (patches) {
        synthetic.data = push(a, 2 * (size_t)crops * SLICE_PATCH_PIXELS * sizeof(float));
    }
    Pcg64 rng = pcg64(seed);
    SynBox window = {-5500, 2000, 5500, 13000};
    double geometry = 0;
    for (int shape = 0; shape < 2; shape++) {
        json_object *spec = syn_tune_spec(a, settings, profile->voxel, shape, seed + shape);
        size_t frame = a->used;
        SynScroll scroll;
        int level1 = real->count[0] || real->count[1] || profile->pitch[0];
        syn_scroll(a, &scroll, spec, &window, (1 + level1) * profile->voxel, workers);
        geometry += .5 * syn_tune_geometry(a, profile, &scroll, spec);
        for (int level = level1; level >= 0; level--) {
            if (!level && !real->count[2]) {
                continue;
            }
            double voxel = (level + 1) * profile->voxel;
            if (!level && level1) {
                size_t n1 = SYN_NOISE_1 * SYN_NOISE_1 * SYN_NOISE_1;
                size_t n2 = SYN_NOISE_2 * SYN_NOISE_2 * SYN_NOISE_2;
                float *noise = push(a, (n1 + n2) * sizeof(*noise));
                scroll.kernel.noise = syn_scroll_noise(a, spec, voxel, noise, noise + n1);
                scroll.kernel.noise1 = noise;
                scroll.kernel.noise2 = noise + n1;
            }
            for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
                if (!real->count[v] || (v != 2) != level) {
                    continue;
                }
                size_t count = syn_realism_extract(a, &scroll, v ? crops / 2 : crops, seed + shape,
                    voxel, v == 1, images, workers);
                for (size_t k = 0; k < count; k++) {
                    const u8 *image = images + k * REALISM_N;
                    realism_features(a, image, rows->rows[v] + rows->count[v]++ * REALISM_FEATURES);
                    if (patches && !v) {
                        float *dst = synthetic.data + synthetic.count++ * SLICE_PATCH_PIXELS;
                        disc_patch(dst, image, REALISM_SIDE, REALISM_SIDE, &rng);
                    }
                }
            }
        }
        a->used = frame;
        json_object_put(spec);
    }
    double distance = 0;
    int views = 0;
    *cnn = NULL;
    for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
        if (!real->count[v]) {
            continue;
        }
        if (!rows->count[v]) {
            a->used = mark;
            return INFINITY;
        }
        double score;
        json_object_put(realism_compare(
            a, real->rows[v], real->count[v], rows->rows[v], rows->count[v], &score));
        distance += score;
        views++;
    }
    double result = geometry + distance / views;
    if (patches && synthetic.count) {
        double auc, loss;
        disc_fit(patches, &synthetic, 300, seed, &auc, &loss);
        *cnn = json_object_new_object();
        syn_add(*cnn, "auc", auc);
        syn_add(*cnn, "loss", loss);
        result += 3 * fmax(0, 1 - loss / log(2));
    }
    a->used = mark;
    return result;
}

static int syn_tune_order(const void *a, const void *b) {
    const SynTuneCandidate *x = a, *y = b;
    if (x->score != y->score) {
        return x->score > y->score ? 1 : -1;
    }
    return (x->index > y->index) - (x->index < y->index);
}

static json_object *syn_tune_report(Arena *a, const SynTuneProfile *profile,
    const SynTuneRows *real, const SynTuneRows *rows, json_object *settings, double score) {
    json_object *out = json_object_new_object(), *shape = json_object_new_object(),
                *all = json_object_new_object(), *differences = json_object_new_object();
    json_object_object_add(out, "profile", json_object_new_string(profile->name));
    syn_add(out, "score", nearbyint(1e4 * score) / 1e4);
    json_object_object_add(out, "auc_shape", shape);
    json_object_object_add(out, "auc_all", all);
    json_object_object_add(out, "settings", json_object_get(settings));
    json_object_object_add(out, "differences", differences);
    for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
        if (!real->count[v]) {
            continue;
        }
        double *r = real->rows[v], *s = rows->rows[v], unused;
        size_t nr = real->count[v], ns = rows->count[v];
        syn_add(shape, syn_tune_keys[v],
            nearbyint(1000 * realism_separability(a, r, nr, s, ns, 5)) / 1000);
        syn_add(all, syn_tune_keys[v],
            nearbyint(1000 * realism_separability(a, r, nr, s, ns, 0)) / 1000);
        json_object_object_add(
            differences, syn_tune_keys[v], realism_compare(a, r, nr, s, ns, &unused));
    }
    return out;
}

static void syn_tune_save(Arena *a, const char *folder, const char *profile, json_object *look) {
    const char *path = fmt(a, "%s/look-%s.json", folder, profile);
    int lock = lock_file(fmt(a, "%s.lock", path), 1, 1);
    need(lock >= 0, "cannot lock %s", path);
    json_write(path, look);
    close(lock);
}

static void syn_tune_append(const char *path, json_object *entry) {
    FILE *f = fopen(path, "a");
    need(f != NULL, "cannot append %s: %s", path, strerror(errno));
    need(!flock(fileno(f), LOCK_EX), "cannot lock %s: %s", path, strerror(errno));
    need(fputs(json_object_to_json_string_ext(entry, JSON_C_TO_STRING_PLAIN), f) >= 0 &&
            fputc('\n', f) != EOF,
        "cannot append %s: %s", path, strerror(errno));
    need(!fclose(f), "cannot close %s: %s", path, strerror(errno));
}

static void syn_tune_search(Arena *a, const char *root, const char *folder,
    const SynTuneProfile *profile, int population, int elite, int rounds, int crops, int workers,
    const char *start, int use_cnn) {
    SynTuneRows real = syn_tune_real(a, root, profile);
    SynTuneRows candidate = syn_tune_allocate(a, &real, crops);
    SynTuneRows best = syn_tune_allocate(a, &real, crops);
    SliceCrops patches = {0};
    if (use_cnn) {
        need(profile->pitch[0], "--cnn requires a training scroll");
        need(real.count[0], "--cnn requires cached level-1 xy features");
        patches = syn_tune_real_patches(a, profile);
    }
    Pcg64 rng = pcg64(0);
    double mean[SYN_TUNE_N], spread[SYN_TUNE_N], anchor[SYN_TUNE_N];
    json_object *starting = NULL;
    if (start) {
        const char *slash = strrchr(start, '/');
        const char *path = slash
            ? fmt(a, "%s/%.*s/look-%s.json", root, (int)(slash - start), start, slash + 1)
            : fmt(a, "%s/look-%s.json", root, start);
        starting = json_read(path);
        need(starting != NULL, "no starting look at %s", path);
    }
    for (int k = 0; k < SYN_TUNE_N; k++) {
        json_object *v = starting
            ? json_object_object_get(jget(starting, "settings"), syn_tuning[k].name)
            : NULL;
        anchor[k] = mean[k] = v ? syn_tune_position(k, json_object_get_double(v)) : .5;
        spread[k] = starting ? .15 : .3;
    }
    json_object_put(starting);
    double (*xs)[SYN_TUNE_N] = push(a, (size_t)population * sizeof(*xs));
    SynTuneCandidate *order = push(a, (size_t)population * sizeof(*order));
    const char *trail = fmt(a, "%s/tune-%s.jsonl", folder, profile->name);
    double best_score = INFINITY;
    json_object *best_settings = NULL, *best_cnn = NULL;
    for (int round = 0; round < rounds; round++) {
        for (int i = 0; i < population; i++) {
            double started = now(), prior = 0;
            json_object *settings = json_object_new_object(), *entry = json_object_new_object();
            json_object *cnn;
            for (int k = 0; k < SYN_TUNE_N; k++) {
                double x = xs[i][k] = syn_clip(mean[k] + spread[k] * syn_normal(&rng), 0, 1);
                syn_add(settings, syn_tuning[k].name, syn_tune_value(k, x));
                if (start && !profile->pitch[0] && k >= 21 && k <= 30) {
                    double shift = (x - anchor[k]) / .2;
                    prior += shift * shift / 10;
                }
            }
            double score = syn_tune_evaluate(a, profile, &real, &candidate, settings, crops,
                               (i64)round * 1000 + i, workers, use_cnn ? &patches : NULL, &cnn) +
                prior;
            if (score < best_score) {
                best_score = score;
                json_object_put(best_settings);
                json_object_put(best_cnn);
                best_settings = json_object_get(settings);
                best_cnn = json_object_get(cnn);
                for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
                    best.count[v] = candidate.count[v];
                    if (best.count[v]) {
                        memcpy(best.rows[v], candidate.rows[v],
                            best.count[v] * REALISM_FEATURES * sizeof(double));
                    }
                }
            }
            order[i] = (SynTuneCandidate){.score = score, .index = i};
            json_object_object_add(entry, "round", json_object_new_int(round));
            json_object_object_add(entry, "score", json_object_new_double(score));
            json_object_object_add(entry, "settings", settings);
            json_object_object_add(entry, "cnn", cnn);
            syn_tune_append(trail, entry);
            json_object_put(entry);
            printf("round %d candidate %d: %.3f (%.0f s)\n", round, i, score, now() - started);
            fflush(stdout);
        }
        qsort(order, population, sizeof(*order), syn_tune_order);
        double top_score = 0;
        for (int i = 0; i < elite; i++) {
            top_score += order[i].score / elite;
        }
        for (int k = 0; k < SYN_TUNE_N; k++) {
            double average = 0, variance = 0;
            for (int i = 0; i < elite; i++) {
                average += xs[order[i].index][k] / elite;
            }
            for (int i = 0; i < elite; i++) {
                double d = xs[order[i].index][k] - average;
                variance += d * d / elite;
            }
            mean[k] = average;
            spread[k] = fmax(.03, sqrt(variance));
        }
        printf("round %d: best %.3f, elite mean %.3f\n", round, order[0].score, top_score);
        fflush(stdout);
    }
    need(best_settings != NULL, "no completed tuning candidate for %s", profile->name);
    json_object *look = syn_tune_report(a, profile, &real, &best, best_settings, best_score);
    json_object *elite_spread = json_object_new_object();
    for (int k = 0; k < SYN_TUNE_N; k++) {
        double high = syn_tune_value(k, fmin(1, mean[k] + spread[k]));
        double low = syn_tune_value(k, fmax(0, mean[k] - spread[k]));
        syn_add(elite_spread, syn_tuning[k].name, nearbyint(5000 * (high - low)) / 1e4);
    }
    json_object_object_add(look, "elite_spread", elite_spread);
    json_object_object_add(look, "cnn", best_cnn);
    syn_tune_save(a, folder, profile->name, look);
    puts(json_object_to_json_string_ext(look, JSON_C_TO_STRING_PRETTY));
    json_object_put(look);
    json_object_put(best_settings);
}

static int syn_tune_round_order(const void *a, const void *b) {
    const SynTuneCandidate *x = a, *y = b;
    i64 rx = ji64(x->entry, "round"), ry = ji64(y->entry, "round");
    if (rx != ry) {
        return rx > ry ? -1 : 1;
    }
    return syn_tune_order(a, b);
}

static void syn_tune_finish(Arena *a, const char *root, const char *folder,
    const SynTuneProfile *profile, int elite, int crops, int workers) {
    const char *path = fmt(a, "%s/tune-%s.jsonl", folder, profile->name);
    FILE *f = fopen(path, "r");
    need(f != NULL, "cannot read %s: %s", path, strerror(errno));
    need(!flock(fileno(f), LOCK_SH), "cannot lock %s: %s", path, strerror(errno));
    SynTuneCandidate *entries = NULL;
    int count = 0, capacity = 0;
    char *line = NULL;
    size_t size = 0;
    json_object *best = NULL;
    double best_score = INFINITY;
    while (getline(&line, &size, f) >= 0) {
        json_object *entry = json_tokener_parse(line);
        need(entry != NULL, "cannot parse tuning candidate in %s", path);
        double score = jf64(entry, "score");
        if (!isfinite(score)) {
            json_object_put(entry);
            continue;
        }
        if (count == capacity) {
            capacity = capacity ? 2 * capacity : 64;
            SynTuneCandidate *grown = realloc(entries, (size_t)capacity * sizeof(*entries));
            need(grown != NULL, "cannot allocate tuning candidates");
            entries = grown;
        }
        if (score < best_score) {
            best_score = score;
            best = entry;
        }
        entries[count] = (SynTuneCandidate){entry, score, count};
        count++;
    }
    need(!ferror(f), "cannot read %s: %s", path, strerror(errno));
    fclose(f);
    free(line);
    need(count >= elite, "no round with %d completed tuning candidates in %s", elite, path);
    qsort(entries, count, sizeof(*entries), syn_tune_round_order);
    int first = 0, end = 0;
    i64 last = 0;
    while (first < count) {
        last = ji64(entries[first].entry, "round");
        end = first + 1;
        while (end < count && ji64(entries[end].entry, "round") == last) {
            end++;
        }
        if (end - first >= elite) {
            break;
        }
        first = end;
    }
    need(first < count, "no round with %d completed tuning candidates in %s", elite, path);
    SynTuneRows real = syn_tune_real(a, root, profile), rows = syn_tune_allocate(a, &real, crops);
    json_object *settings = jget(best, "settings"), *cnn;
    double score =
        syn_tune_evaluate(a, profile, &real, &rows, settings, crops, 99, workers, NULL, &cnn);
    need(isfinite(score), "cannot rescore the best tuning candidate for %s", profile->name);
    json_object *look = syn_tune_report(a, profile, &real, &rows, settings, score);
    syn_add(look, "trail_score", nearbyint(1e4 * best_score) / 1e4);
    json_object_object_add(look, "rounds", json_object_new_int64(last + 1));
    json_object *spread = json_object_new_object();
    for (int k = 0; k < SYN_TUNE_N; k++) {
        const char *name = syn_tuning[k].name;
        double mean = 0, variance = 0;
        int present = 0;
        for (int i = first; i < first + elite; i++) {
            double v = syn_optional(jget(entries[i].entry, "settings"), name, NAN);
            if (isfinite(v)) {
                present++;
                double d = v - mean;
                mean += d / present;
                variance += d * (v - mean);
            }
        }
        if (present) {
            syn_add(spread, name, nearbyint(1e4 * sqrt(variance / present)) / 1e4);
        }
    }
    json_object_object_add(look, "elite_spread", spread);
    syn_tune_save(a, folder, profile->name, look);
    const char *keys[] = {"profile", "score", "trail_score", "rounds", "auc_shape"};
    json_object *summary = json_object_new_object();
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++) {
        json_object_object_add(summary, keys[i], json_object_get(jget(look, keys[i])));
    }
    puts(json_object_to_json_string_ext(summary, JSON_C_TO_STRING_SPACED));
    json_object_put(summary);
    json_object_put(look);
    for (int i = 0; i < count; i++) {
        json_object_put(entries[i].entry);
    }
    free(entries);
}

static void syn_tune_publish(Arena *a, const char *root, const char *folder, const char *path,
    const char *const *names, int count) {
    int lock = lock_file(fmt(a, "%s.lock", path), 1, 1);
    need(lock >= 0, "cannot lock %s", path);
    json_object *looks = json_read(path);
    if (!looks) {
        looks = json_tokener_parse("{\"version\":1,\"profiles\":{}}");
    }
    json_object *profiles = jget(looks, "profiles");
    for (int i = 0; i < count; i++) {
        size_t mark = a->used;
        const SynTuneProfile *profile = syn_tune_profile(names[i]);
        json_object *look = json_read(fmt(a, "%s/look-%s.json", folder, profile->name));
        if (!look) {
            printf("%s: no tuned look, left out\n", profile->name);
            a->used = mark;
            continue;
        }
        SynTuneRows real = syn_tune_real(a, root, profile);
        json_object *row = json_object_new_object(), *crops = json_object_new_object();
        json_object_object_add(row, "settings", json_object_get(jget(look, "settings")));
        json_object_object_add(row, "spread", json_object_get(jget(look, "elite_spread")));
        json_object_object_add(row, "score", json_object_get(jget(look, "score")));
        json_object_object_add(row, "auc_shape", json_object_get(jget(look, "auc_shape")));
        syn_add(row, "voxel_um", profile->voxel);
        json_object_object_add(
            row, "group", json_object_new_string(profile->voxel == 7.91 ? "54keV" : "113keV"));
        json_object_object_add(row, "target", json_object_new_boolean(profile->target));
        for (int v = 0; v < SYN_TUNE_VIEWS; v++) {
            if (real.count[v]) {
                json_object_object_add(
                    crops, syn_tune_keys[v], json_object_new_int64(real.count[v]));
            }
        }
        json_object_object_add(row, "crops", crops);
        json_object_object_add(profiles, profile->name, row);
        json_object_put(look);
        a->used = mark;
    }
    json_write(path, looks);
    close(lock);
    puts(json_object_to_json_string_ext(looks, JSON_C_TO_STRING_PRETTY));
    json_object_put(looks);
}

static int syn_tune_command(int argc, char **argv) {
    const char *usage =
        "Usage: ./run.sh synthetic.three_d.tune PROFILE [--population N] [--elite N] "
        "[--rounds N] [--crops N] [--start LOOK] [--cnn] [--round NAME]\n"
        "       ./run.sh synthetic.three_d.tune {finish|publish} "
        "[--profiles PROFILE ...] [--round NAME]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(argc > 0, "%s", usage);
    int population = 24, elite = 6, rounds = 8, crops = 16, cnn = 0;
    int count = 4;
    const char *defaults[] = {"PHerc0139", "PHerc1667", "PHerc0172", "PHercParis4"};
    const char *const *names = defaults;
    const char *start = NULL, *name = NULL;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--profiles")) {
            names = (const char *const *)(argv + i + 1);
            count = 0;
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
                count++;
                i++;
            }
        } else if (!strcmp(arg, "--cnn")) {
            cnn = 1;
        } else if (!strcmp(arg, "--start") || !strcmp(arg, "--round")) {
            need(i + 1 < argc, "%s", usage);
            if (!strcmp(arg, "--start")) {
                start = argv[++i];
            } else {
                name = argv[++i];
            }
        } else {
            int *value = NULL;
            if (!strcmp(arg, "--population")) {
                value = &population;
            } else if (!strcmp(arg, "--elite")) {
                value = &elite;
            } else if (!strcmp(arg, "--rounds")) {
                value = &rounds;
            } else if (!strcmp(arg, "--crops")) {
                value = &crops;
            }
            need(value && i + 1 < argc, "%s", usage);
            char *end;
            long parsed = strtol(argv[++i], &end, 10);
            need(*argv[i] && !*end && parsed > 0 && parsed <= INT_MAX / 20, "%s", usage);
            *value = parsed;
        }
    }
    Arena a = arena(64 * GB);
    const char *root = fmt(&a, "%s/synthetic/realism", OUTPUTS);
    const char *folder = name ? fmt(&a, "%s/%s", root, name) : root;
    mkdirs(folder);
    char *end;
    long threads = strtol(env("HERCULEX_SYNTHETIC_THREADS", "0"), &end, 10);
    int workers = threads > 0 && threads <= INT_MAX && !*end ? threads : fmax(1, cores() - 2);
    if (!strcmp(argv[0], "publish")) {
        const char *path = name
            ? fmt(&a, "%s/looks.json", folder)
            : env("HERCULEX_SYNTHETIC_LOOKS", fmt(&a, "%s/synthetic/three_d/looks.json", HOME));
        syn_tune_publish(&a, root, folder, path, names, count);
    } else if (!strcmp(argv[0], "finish")) {
        for (int i = 0; i < count; i++) {
            size_t mark = a.used;
            syn_tune_finish(&a, root, folder, syn_tune_profile(names[i]), elite, crops, workers);
            a.used = mark;
        }
    } else {
        syn_tune_search(&a, root, folder, syn_tune_profile(argv[0]), population,
            elite < population ? elite : population, rounds, crops, workers, start, cnn);
    }
    return 0;
}
