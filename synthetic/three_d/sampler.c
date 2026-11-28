typedef struct {
    char name[48];
    int shape, condition, seen;
    i64 step, seed;
    double score, stale, prob, loss[8];
} SynFamily;

typedef struct {
    const char *name;
    i64 seed;
} SynBandKey;

typedef struct {
    int count;
    SynFamily rows[(SYN_DENTED + 1) * (SYN_CRACKED + 1)];
} SynFamilies;

static SynFamilies syn_pool(int held, i64 seed) {
    SynFamilies pool = {0};
    for (int i = 0; i <= SYN_DENTED; i++) {
        for (int j = 0; j <= SYN_CRACKED; j++) {
            if (held >= 0 && syn_held_out(i, j) != held) {
                continue;
            }
            SynFamily *f = pool.rows + pool.count++;
            snprintf(f->name, sizeof(f->name), "%s/%s", syn_shapes[i], syn_conditions[j]);
            f->shape = i;
            f->condition = j;
            f->seed = seed;
            f->step = -1;
        }
    }
    return pool;
}

static SynFamily *syn_family_find(SynFamilies *pool, const char *name) {
    for (int k = 0; name && k < pool->count; k++) {
        if (!strcmp(pool->rows[k].name, name)) {
            return pool->rows + k;
        }
    }
    return NULL;
}

static double syn_hardness(json_object *row) {
    json_object *accuracy = json_object_object_get(row, "global_accuracy");
    return accuracy ? 1 - json_object_get_double(accuracy) : syn_optional(row, "loss", NAN);
}

static size_t syn_priorities(SynFamilies *pool, const char *log) {
    FILE *file = fopen(log, "r");
    char *line = NULL;
    size_t capacity = 0, records = 0;
    i64 current = 0;
    while (file && getline(&line, &capacity, file) >= 0) {
        json_object *row = json_tokener_parse(line);
        if (!row) {
            continue;
        }
        records++;
        const char *name = json_object_get_string(json_object_object_get(row, "family"));
        SynFamily *family = syn_family_find(pool, name);
        double cost = syn_hardness(row);
        int held = json_object_get_boolean(json_object_object_get(row, "held_out"));
        if (family && !held && isfinite(cost)) {
            family->loss[family->seen++ % 8] = cost;
            i64 step = json_object_get_int64(json_object_object_get(row, "step"));
            family->step = step;
            if (step > current) {
                current = step;
            }
        }
        json_object_put(row);
    }
    if (file) {
        fclose(file);
    }
    free(line);
    double top = -INFINITY, stale = 0;
    for (int i = 0; i < pool->count; i++) {
        SynFamily *f = pool->rows + i;
        if (f->seen) {
            int n = f->seen < 8 ? f->seen : 8;
            double ordered[8];
            for (int j = 0; j < n; j++) {
                ordered[j] = f->loss[(f->seen - n + j) % 8];
            }
            f->score = syn_sum(ordered, n) / n;
            top = fmax(top, f->score);
        }
        f->stale = current - f->step;
        stale += f->stale;
    }
    if (!isfinite(top)) {
        top = 1;
    }
    for (int i = 0; i < pool->count; i++) {
        if (!pool->rows[i].seen) {
            pool->rows[i].score = 2 * top + 1;
        }
    }
    double rewards[(SYN_DENTED + 1) * (SYN_CRACKED + 1)];
    for (int i = 0; i < pool->count; i++) {
        double rank = 0.5;
        for (int j = 0; j < pool->count; j++) {
            rank += pool->rows[j].score > pool->rows[i].score ? 1
                : pool->rows[j].score == pool->rows[i].score  ? 0.5
                                                              : 0;
        }
        rewards[i] = 1 / rank;
    }
    double total = syn_sum(rewards, pool->count), floor = 0.1 / pool->count;
    for (int i = 0; i < pool->count; i++) {
        SynFamily *f = pool->rows + i;
        double age = stale > 0 ? f->stale / stale : 1.0 / pool->count;
        f->prob = 0.9 * (0.8 * (rewards[i] / total) + 0.2 * age) + floor;
    }
    return records;
}

static SynBandKey *syn_sampler_plan(
    Arena *a, const char *bands, const char *log, int count, i64 seed_base) {
    SynFamilies pool = syn_pool(0, seed_base - 1);
    size_t total = syn_priorities(&pool, log);
    FILE *file = fopen(fmt(a, "%s/index.jsonl", bands), "r");
    char *line = NULL;
    size_t capacity = 0;
    while (file && getline(&line, &capacity, file) >= 0) {
        json_object *row = json_tokener_parse(line);
        need(row != NULL, "invalid band index row");
        SynFamily *f = syn_family_find(&pool, jstr(row, "family"));
        i64 seed = ji64(row, "seed");
        if (f && seed > f->seed) {
            f->seed = seed;
        }
        json_object_put(row);
    }
    free(line);
    if (file) {
        fclose(file);
    }
    double probabilities[(SYN_DENTED + 1) * (SYN_CRACKED + 1)];
    for (int i = 0; i < pool.count; i++) {
        probabilities[i] = pool.rows[i].prob;
    }
    double sum = syn_sum(probabilities, pool.count), cdf = 0;
    for (int i = 0; i < pool.count; i++) {
        cdf += probabilities[i] / sum;
        probabilities[i] = cdf;
    }
    Pcg64 rng = pcg64(total);
    SynBandKey *picks = push(a, (size_t)count * sizeof(*picks));
    const char *names[(SYN_DENTED + 1) * (SYN_CRACKED + 1)] = {0};
    for (int i = 0; i < count; i++) {
        double val = pcg64_double(&rng) * cdf;
        int k = 0;
        while (k + 1 < pool.count && val >= probabilities[k]) {
            k++;
        }
        if (!names[k]) {
            names[k] = fmt(a, "%s", pool.rows[k].name);
        }
        picks[i] = (SynBandKey){.name = names[k], .seed = ++pool.rows[k].seed};
    }
    return picks;
}

static int syn_priority_order(const void *left, const void *right) {
    const SynFamily *a = left, *b = right;
    return (a->prob < b->prob) - (a->prob > b->prob);
}

typedef struct {
    json_object *row;
    double cost;
    size_t order;
} SynHardBand;

static int syn_hard_order(const void *left, const void *right) {
    const SynHardBand *a = left, *b = right;
    if (a->cost != b->cost) {
        return (a->cost < b->cost) - (a->cost > b->cost);
    }
    return (a->order > b->order) - (a->order < b->order);
}

static json_object *syn_mutate(Arena *a, json_object *spec, Pcg64 *rng) {
    const char *edits[] = {"narrower", "contact", "crack", "wrinkle", "haze", "noise", "plies"};
    const int first = draw(rng, 6), draw2 = draw(rng, 7);
    int picks[] = {first, draw2 == first ? 6 : draw2};
    if (!draw(rng, 2)) {
        int swap = picks[0];
        picks[0] = picks[1];
        picks[1] = swap;
    }
    json_object *child = syn_complete(a, spec), *ancestor = jget(child, "lineage");
    json_object *roll = jget(child, "roll"), *sheet = jget(child, "sheet"), *ct = jget(child, "ct");
    json_object *history = jget(ancestor, "edits");
    for (int i = 0; i < 2; i++) {
        int edit = picks[i];
        json_object_array_add(history, json_object_new_string(edits[edit]));
        if (edit == 0) {
            syn_add(roll, "gap_um", jf64(roll, "gap_um") * syn_u(rng, 0.6, 0.85));
        } else if (edit == 1) {
            syn_add(roll, "contact", fmin(0.7, jf64(roll, "contact") + syn_u(rng, 0.1, 0.25)));
        } else if (edit == 2) {
            json_object *cracks = syn_family_cracks(a, rng, child);
            json_object_array_add(
                jget(child, "cracks"), json_object_get(json_object_array_get_idx(cracks, 0)));
            json_object_put(cracks);
        } else if (edit == 3) {
            json_object *wrinkle = json_object_new_object();
            syn_add(wrinkle, "scale_um", syn_u(rng, 500, 1200));
            syn_add(wrinkle, "amp_um", syn_u(rng, 20, 60));
            syn_add(wrinkle, "z_scale_um", syn_u(rng, 600, 2000));
            json_object_array_add(jget(child, "flows"), wrinkle);
        } else if (edit == 4) {
            syn_add(sheet, "fog_um", jf64(sheet, "fog_um") + syn_u(rng, 4, 10));
        } else if (edit == 5) {
            syn_add(ct, "noise", jf64(ct, "noise") * syn_u(rng, 1.15, 1.4));
        } else {
            syn_add(roll, "delamination",
                fmin(0.9, jf64(roll, "delamination") + syn_u(rng, 0.15, 0.3)));
        }
    }
    char digest[13];
    syn_spec_digest(a, spec, digest);
    json_object_object_add(ancestor, "parent", json_object_new_string(digest));
    return child;
}

static int syn_sampler_command(int argc, char **argv) {
    const char *usage = "Usage: ./run.sh synthetic.three_d.sampler "
                        "{show LOG|plan LOG COUNT [--seed-base S]|mutants LOG COUNT OUT}";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    Arena a = arena(GB);
    if (argc == 2 && !strcmp(argv[0], "show")) {
        SynFamilies pool = syn_pool(0, 0);
        syn_priorities(&pool, argv[1]);
        for (int i = 1; i < pool.count; i++) {
            SynFamily f = pool.rows[i];
            int j = i;
            while (j > 0 && syn_priority_order(&f, &pool.rows[j - 1]) < 0) {
                pool.rows[j] = pool.rows[j - 1];
                j--;
            }
            pool.rows[j] = f;
        }
        for (int k = 0; k < pool.count; k++) {
            SynFamily *f = pool.rows + k;
            printf("%-24s score %8.4f stale %6.0f p %.4f\n", f->name, f->score, f->stale, f->prob);
        }
        return 0;
    }
    int mutate = argc == 4 && !strcmp(argv[0], "mutants");
    need(mutate ||
            ((argc == 3 || (argc == 5 && !strcmp(argv[3], "--seed-base"))) &&
                !strcmp(argv[0], "plan")),
        "%s", usage);
    char *end;
    long count = strtol(argv[2], &end, 10);
    need(!*end && count >= 0 && count <= INT_MAX, "%s", usage);
    const char *bands = fmt(&a, "%s/synthetic/bands", OUTPUTS);
    if (!mutate) {
        i64 seed = argc == 5 ? strtoll(argv[4], &end, 10) : 1000;
        need(!*end, "%s", usage);
        SynBandKey *picks = syn_sampler_plan(&a, bands, argv[1], count, seed);
        for (long i = 0; i < count; i++) {
            printf("%s%s:%lld", i ? " " : "", picks[i].name, (long long)picks[i].seed);
        }
        puts("");
        return 0;
    }
    FILE *file = fopen(argv[1], "r");
    char *line = NULL;
    size_t bytes = 0, used = 0, cap = 64;
    SynHardBand *rows = malloc(cap * sizeof(*rows));
    need(rows != NULL, "cannot allocate synthetic log");
    while (file && getline(&line, &bytes, file) >= 0) {
        json_object *row = json_tokener_parse(line);
        const char *name = json_object_get_string(json_object_object_get(row, "band"));
        double cost = syn_hardness(row);
        if (!name || !*name || !isfinite(cost) ||
            json_object_get_boolean(json_object_object_get(row, "held_out"))) {
            json_object_put(row);
            continue;
        }
        if (used == cap) {
            cap *= 2;
            void *grown = realloc(rows, cap * sizeof(*rows));
            need(grown != NULL, "cannot grow synthetic log");
            rows = grown;
        }
        rows[used] = (SynHardBand){.row = row, .cost = cost, .order = used};
        used++;
    }
    if (file) {
        fclose(file);
    }
    free(line);
    if (used) {
        qsort(rows, used, sizeof(*rows), syn_hard_order);
        Pcg64 rng = pcg64(used);
        size_t top = used >= 4 ? used / 4 : 1;
        mkdirs(argv[3]);
        for (long i = 0; i < count; i++) {
            const char *name = jstr(rows[i % top].row, "band");
            size_t saved = a.used;
            json_object *spec = json_read(fmt(&a, "%s/%s/spec.json", bands, name));
            if (spec) {
                json_object *child = syn_mutate(&a, spec, &rng);
                int n = json_object_array_length(jget(jget(child, "lineage"), "edits"));
                char *written = fmt(&a, "%s-m%d-%ld", name, n, i);
                json_write(fmt(&a, "%s/%s.json", argv[3], written), child);
                puts(written);
                json_object_put(child);
                json_object_put(spec);
            }
            a.used = saved;
        }
    }
    for (size_t i = 0; i < used; i++) {
        json_object_put(rows[i].row);
    }
    free(rows);
    return 0;
}
