static void syn_sample_nodes(Arena *a, Pcg64 *rng, u32 available, u32 count, u32 *out) {
    size_t mark = a->used;
    if (available > 10000 && count > available / 50) {
        u32 *permutation = push(a, (size_t)available * sizeof(*permutation));
        for (u32 i = 0; i < available; i++) {
            permutation[i] = i;
        }
        u32 low = available - count;
        for (u32 i = available - 1; i > 0 && i >= low; i--) {
            u32 j = draw(rng, (u64)i + 1), value = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = value;
        }
        memcpy(out, permutation + low, (size_t)count * sizeof(*out));
    } else {
        size_t length = 1;
        while (length <= (size_t)count * 2) {
            length <<= 1;
        }
        u32 *used = push(a, length * sizeof(*used));
        memset(used, 255, length * sizeof(*used));
        size_t mask = length - 1;
        for (u32 i = 0, j = available - count; i < count; i++, j++) {
            u32 pick = draw(rng, (u64)j + 1);
            size_t at = pick & mask;
            while (used[at] != UINT32_MAX && used[at] != pick) {
                at = (at + 1) & mask;
            }
            if (used[at] == pick) {
                pick = j;
                at = pick & mask;
                while (used[at] != UINT32_MAX) {
                    at = (at + 1) & mask;
                }
            }
            used[at] = out[i] = pick;
        }
        for (u32 i = count; i > 1;) {
            --i;
            u32 j = draw(rng, (u64)i + 1), value = out[i];
            out[i] = out[j];
            out[j] = value;
        }
    }
    a->used = mark;
}

static size_t syn_node_realism(Arena *a, const char *control, double *output, size_t stride) {
    size_t mark = a->used, n = 0;
    void *raw = NULL;
    i32 *cells = NULL;
    need(!hx_synthetic_raw_nodes(control, &raw, &cells, &n) && n && n <= UINT32_MAX,
        "cannot read synthetic realism nodes at %s", control);
    RealismNode *nodes = raw;
    u32 count = n < 400000 ? (u32)n : 400000;
    i32 max = 0;
    for (size_t i = 0; i < n; i++) {
        if (cells[i] > max) {
            max = cells[i];
        }
    }
    i32 *per = pushz(a, ((size_t)max + 1) * sizeof(*per));
    for (size_t i = 0; i < n; i++) {
        per[cells[i]]++;
    }
    u32 *picks = push(a, (size_t)count * sizeof(*picks));
    Pcg64 rng = pcg64(0);
    syn_sample_nodes(a, &rng, n, count, picks);
    for (u32 k = 0; k < count; k++) {
        u32 i = picks[k];
        RealismNode p = nodes[i];
        double norm = sqrt(p.nx * p.nx + p.ny * p.ny + p.nz * p.nz);
        output[k] = p.prominence;
        output[stride + k] = p.width;
        output[2 * stride + k] = p.coherence;
        output[3 * stride + k] = (double)abs(p.nz) / fmax(norm, 1e-9);
        output[4 * stride + k] = per[cells[i]];
    }
    free(cells);
    free(raw);
    a->used = mark;
    return count;
}

static int syn_realism_nodes(int argc, char **argv) {
    const char *profile = argv[0];
    Arena a = arena(64 * GB);
    const char *controls[] = {"PHerc0139-20250728140407-rows-0007-0019",
        "PHerc1667-20231117161658-rows-0008-0020", "PHerc0172-20241024131839-rows-0010-0022",
        "PHercParis4-20230205180739-rows-0010-0022"};
    int v = !strcmp(profile, "9.362") ? 0 : !strcmp(profile, "7.91") ? 1 : -1;
    need(v >= 0, "unknown scanner profile %s", profile);
    int real_count = v == 0 ? 1 : 3, syn_count = argc - 1;
    size_t stride[] = {(size_t)400000 * real_count, (size_t)400000 * syn_count};
    double *values[] = {
        push(&a, 5 * stride[0] * sizeof(double)), push(&a, 5 * stride[1] * sizeof(double))};
    size_t count[] = {0, 0};
    for (int k = 0; k < real_count; k++) {
        const char *p = fmt(&a, "%s/controls/sheets/%s", LIBRARY, controls[v + k]);
        count[0] += syn_node_realism(&a, p, values[0] + count[0], stride[0]);
    }
    json_object *bands = json_object_new_array();
    for (int k = 0; k < syn_count; k++) {
        const char *p = fmt(&a, "%s/synthetic/bands/%s/control", OUTPUTS, argv[k + 1]);
        count[1] += syn_node_realism(&a, p, values[1] + count[1], stride[1]);
        json_object_array_add(bands, json_object_new_string(argv[k + 1]));
    }
    json_object *out = json_object_new_object(), *summary = json_object_new_object();
    const char *keys[] = {"prominence", "width", "coherence", "tilt", "per_cell"};
    for (int k = 0; k < 5; k++) {
        double *real = values[0] + k * stride[0], *synthetic = values[1] + k * stride[1];
        qsort(real, count[0], sizeof(*real), syn_double_order);
        qsort(synthetic, count[1], sizeof(*synthetic), syn_double_order);
        double ks = 0;
        for (size_t r = 0, s = 0; r < count[0] || s < count[1];) {
            double x =
                fmin(r < count[0] ? real[r] : INFINITY, s < count[1] ? synthetic[s] : INFINITY);
            while (r < count[0] && real[r] <= x) {
                r++;
            }
            while (s < count[1] && synthetic[s] <= x) {
                s++;
            }
            ks = fmax(ks, fabs((double)r / count[0] - (double)s / count[1]));
        }
        json_object *value = json_object_new_object();
        syn_add(value, "real_p50", realism_percentile(real, count[0], 0.5));
        syn_add(value, "synthetic_p50", realism_percentile(synthetic, count[1], 0.5));
        syn_add(value, "ks", nearbyint(ks * 1000) / 1000);
        json_object_object_add(summary, keys[k], value);
    }
    json_object_object_add(out, "profile", json_object_new_string(profile));
    json_object_object_add(out, "bands", bands);
    json_object_object_add(out, "nodes", summary);
    puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_PRETTY));
    json_object_put(out);
    return 0;
}

static size_t syn_realism_extract(Arena *a, const SynScroll *scroll, int count, i64 seed,
    double voxel, int across, u8 *output, int workers) {
    size_t mark = a->used;
    SynDeform d = scroll->deform;
    const size_t n = 20000;
    double *y = push(a, n * sizeof(*y)), *x = push(a, n * sizeof(*x));
    double half = 0.5 * REALISM_SIDE * voxel;
    double pad = fmin(d.box.y1 - d.box.y0, d.box.x1 - d.box.x0) > 4 * half ? half : 0;
    Pcg64 rng = pcg64(seed);
    for (size_t i = 0; i < n; i++) {
        y[i] = syn_u(&rng, d.box.y0 + pad, d.box.y1 - pad);
    }
    for (size_t i = 0; i < n; i++) {
        x[i] = syn_u(&rng, d.box.x0 + pad, d.box.x1 - pad);
    }
    size_t found = 0, tries = 0;
    for (size_t i = 0; i < n && found < (size_t)count && tries < 20 * (size_t)count; i++) {
        SynTruthPoint t = syn_truth_at(&scroll->kernel, d.mid, y[i], x[i]);
        if (!t.mask || !(fabs(t.nearest) <= t.thickness / 2)) {
            continue;
        }
        tries++;
        u8 *crop = output + found * REALISM_N;
        double corner[] = {d.mid - (across ? half : 0), y[i] - (across ? 0 : half), x[i] - half};
        hx_syn_render(&scroll->kernel, across ? REALISM_SIDE : 1, across ? 1 : REALISM_SIDE,
            REALISM_SIDE, corner, voxel, crop, workers);
        size_t inside = 0;
        for (int k = 0; k < REALISM_N; k++) {
            inside += crop[k] != 0;
        }
        found += (double)inside / REALISM_N >= 0.85;
    }
    a->used = mark;
    return found;
}

static size_t syn_realism_crops(Arena *a, json_object *spec, int count, i64 seed, double voxel,
    int across, u8 *output, int workers) {
    size_t mark = a->used;
    SynScroll scroll;
    syn_scroll(a, &scroll, spec, NULL, voxel, workers);
    size_t found = syn_realism_extract(a, &scroll, count, seed, voxel, across, output, workers);
    a->used = mark;
    return found;
}

static int syn_realism_tag(const Array scans, size_t i, const char *scroll) {
    int utf32 = scans.descr[1] == 'U';
    size_t width = strtoul(scans.descr + 2, NULL, 10), n = strlen(scroll);
    if (n > width) {
        return 0;
    }
    if (!utf32) {
        const char *str = (const char *)scans.data + i * width;
        return !strncmp(str, scroll, n) && (width == n || str[n] == 0);
    }
    const u32 *s = (const u32 *)scans.data + i * width;
    for (size_t k = 0; k < n; k++) {
        if (s[k] != (u8)scroll[k]) {
            return 0;
        }
    }
    return width == n || s[n] == 0;
}

static const char *syn_realism_path(
    Arena *a, const char *root, const char *look, double voxel, int level, int across) {
    const char *suffix = fmt(a, "%s%s", across ? "-zx" : "", level ? "" : "-L0");
    int target = look && strcmp(look, "PHerc0139") != 0 && strcmp(look, "PHerc1667") != 0 &&
        strcmp(look, "PHerc0172") != 0 && strcmp(look, "PHercParis4") != 0;
    const char *profile =
        (look ? !strcmp(look, "PHerc0139") : fabs(voxel - 9.362) < 1e-3) ? "9.362" : "7.91";
    return target ? fmt(a, "%s/target-%s%s.npz", root, look, suffix)
                  : fmt(a, "%s/real-%s%s.npz", root, profile, suffix);
}

static size_t syn_realism_real(Arena *a, const char *root, const char *look, double voxel,
    int level, int across, double **real) {
    const char *path = syn_realism_path(a, root, look, voxel, level, across);
    Array features = npz_get(a, path, "features");
    need(features.ndim == 2 && features.shape[1] == REALISM_FEATURES &&
            !strcmp(features.descr, "<f8"),
        "invalid real feature matrix in %s", path);
    *real = features.data;
    if (!look || !strncmp(strrchr(path, '/') + 1, "target-", 7)) {
        return features.shape[0];
    }
    Array scans = npz_get(a, path, "scans");
    need(scans.shape[0] == features.shape[0], "inconsistent real feature matrix in %s", path);
    size_t count = 0;
    for (i64 k = 0; k < features.shape[0]; k++) {
        if (syn_realism_tag(scans, k, look)) {
            memmove(*real + count++ * REALISM_FEATURES, *real + k * REALISM_FEATURES,
                REALISM_FEATURES * sizeof(double));
        }
    }
    return count;
}

static int syn_realism_command(int argc, char **argv) {
    const char *usage =
        "Usage: ./run.sh synthetic.three_d.realism {nodes PROFILE BAND ...|"
        "family FAMILY ... [--seeds N] [--crops N] [--look NAME] [--level 0|1] [--plane xy|zx]}";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[0], "nodes")) {
        return syn_realism_nodes(argc - 1, argv + 1);
    }
    need(argc >= 2 && !strcmp(argv[0], "family"), "%s", usage);
    Arena a = arena(64 * GB);
    const char *root = fmt(&a, "%s/synthetic/realism", OUTPUTS), *look = NULL;
    int seeds = 3, crops = 20, level = 1, across = 0, n = 0;
    const char **names = push(&a, (size_t)argc * sizeof(*names));
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--plane") && i + 1 < argc) {
            const char *plane = argv[++i];
            need(!strcmp(plane, "xy") || !strcmp(plane, "zx"), "%s", usage);
            across = !strcmp(plane, "zx");
        } else if (!strcmp(arg, "--look") && i + 1 < argc) {
            look = argv[++i];
        } else if (!strcmp(arg, "--level") && i + 1 < argc) {
            const char *text = argv[++i];
            need(!strcmp(text, "0") || !strcmp(text, "1"), "%s", usage);
            level = text[0] - '0';
        } else if ((!strcmp(arg, "--seeds") || !strcmp(arg, "--crops")) && i + 1 < argc) {
            char *end;
            long count = strtol(argv[++i], &end, 10);
            need(!*end && count > 0 && count <= INT_MAX, "%s", usage);
            if (!strcmp(arg, "--seeds")) {
                seeds = count;
            } else {
                crops = count;
            }
        } else {
            need(arg[0] != '-', "%s", usage);
            names[n++] = arg;
        }
    }
    need(n > 0, "%s", usage);
    mkdirs(root);
    int workers = (int)strtol(env("HERCULEX_SYNTHETIC_THREADS", "0"), NULL, 10);
    if (workers <= 0) {
        workers = fmax(1, cores() - 2);
    }
    double *sampled = push(&a, (size_t)seeds * crops * REALISM_FEATURES * sizeof(*sampled));
    u8 *images = push(&a, (size_t)crops * REALISM_N);
    const char *report = fmt(&a, "%s/families.jsonl", root);
    for (int j = 0; j < n; j++) {
        size_t mark = a.used, count = 0;
        double first_voxel = 0;
        for (int i = 1; i <= seeds; i++) {
            json_object *spec = syn_family_draw(&a, names[j], i, 0, look);
            double voxel = jf64(spec, "voxel_um");
            if (i == 1) {
                first_voxel = voxel;
            }
            voxel = ldexp(voxel, level);
            if (across) {
                double z =
                    json_object_get_double(json_object_array_get_idx(jget(spec, "band_um"), 0));
                syn_add_pair(spec, "band_um", z, z + REALISM_SIDE * voxel + 300);
            }
            size_t got = syn_realism_crops(&a, spec, crops, i, voxel, across, images, workers);
            for (size_t k = 0; k < got; k++) {
                realism_features(&a, images + k * REALISM_N, sampled + count++ * REALISM_FEATURES);
            }
            json_object_put(spec);
        }
        json_object *out = json_object_new_object();
        json_object_object_add(out, "family", json_object_new_string(names[j]));
        if (!count) {
            json_object_object_add(
                out, "error", json_object_new_string("no crops inside the mask"));
        } else {
            double *real, distance;
            size_t rows = syn_realism_real(&a, root, look, first_voxel, level, across, &real);
            need(rows != 0, "no real features for %s", look ? look : names[j]);
            json_object *diff = realism_compare(&a, real, rows, sampled, count, &distance);
            json_object_object_add(out, "look", look ? json_object_new_string(look) : NULL);
            json_object_object_add(out, "level", json_object_new_int(level));
            json_object_object_add(out, "plane", json_object_new_string(across ? "zx" : "xy"));
            syn_add(out, "distance", nearbyint(distance * 1000) / 1000);
            json_object_object_add(out, "crops", json_object_new_int64(count));
            syn_add(out, "auc_all",
                nearbyint(realism_separability(&a, real, rows, sampled, count, 0) * 1000) / 1000);
            syn_add(out, "auc_shape",
                nearbyint(realism_separability(&a, real, rows, sampled, count, 5) * 1000) / 1000);
            json_object_object_add(out, "differences", diff);
            int fd = open(report, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
            need(fd >= 0, "cannot open %s", report);
            const char *line =
                fmt(&a, "%s\n", json_object_to_json_string_ext(out, JSON_C_TO_STRING_PLAIN));
            write_all(fd, line, strlen(line), report);
            close(fd);
        }
        puts(json_object_to_json_string_ext(out, JSON_C_TO_STRING_PLAIN));
        fflush(stdout);
        json_object_put(out);
        a.used = mark;
    }
    return 0;
}

static int syn_discriminator_run(int argc, char **argv, int injected) {
    const char *usage = injected
        ? "Usage: ./run.sh synthetic.three_d.inject realism [--look SCROLL] "
          "[--families FAMILY ...] [--crops N] [--steps N]"
        : "Usage: ./run.sh synthetic.three_d.discriminator LOOK "
          "[--families FAMILY ...] [--crops N] [--steps N] [--tag TAG]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    need(injected || argc > 0, "%s", usage);
    const char *look = injected ? "PHerc0139" : argv[0], *tag = "";
    const char *defaults[] = {"round/normal", "oval/normal", "oval/pressed"};
    const char **families = defaults;
    int nf = 3, crops = 160, steps = 400;
    for (int i = !injected; i < argc; i++) {
        const char *option = argv[i];
        need(i + 1 < argc, "%s", usage);
        if (!strcmp(option, "--families")) {
            families = (const char **)(argv + i + 1);
            nf = 0;
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
                nf++;
                i++;
            }
        } else if (injected && !strcmp(option, "--look")) {
            look = argv[++i];
        } else if (!injected && !strcmp(option, "--tag")) {
            tag = argv[++i];
        } else if (!strcmp(option, "--steps") || !strcmp(option, "--crops")) {
            char *end;
            long n = strtol(argv[++i], &end, 10);
            need(!*end && n >= 0 && n < INT_MAX / 90, "%s", usage);
            if (!strcmp(option, "--steps")) {
                steps = n;
            } else {
                crops = n;
            }
        } else {
            die("%s", usage);
        }
    }
    need(crops > 0 && nf > 0, "%s", usage);
    int profile = -1;
    for (size_t i = 0; i < sizeof(realism_training) / sizeof(*realism_training); i++) {
        if (!strcmp(look, realism_training[i].sample)) {
            profile = i != 0;
        }
    }
    need(profile >= 0, "unknown training scroll %s", look);
    Arena a = arena(64 * GB);
    size_t maximum = (size_t)crops * (profile ? 3 : 1);
    const char **names = push(&a, maximum * sizeof(*names));
    SliceCrops real = {.data = push(&a, (size_t)crops * SLICE_PATCH_PIXELS * sizeof(float))};
    size_t mark = a.used;
    u8 *images = push(&a, maximum * REALISM_N);
    size_t found = realism_real_crops(&a, profile, crops, 1, 0, images, names, 0);
    Pcg64 rng = pcg64(0);
    for (size_t i = 0; i < found; i++) {
        if (!strcmp(look, names[i])) {
            disc_patch(real.data + real.count++ * SLICE_PATCH_PIXELS, images + i * REALISM_N,
                REALISM_SIDE, REALISM_SIDE, &rng);
        }
    }
    need(real.count, "no cached crops for %s", look);
    a.used = mark;
    size_t share = real.count / (3 * (size_t)nf);
    int per = share > 1 ? (int)share : 1;
    SliceCrops synthetic = {
        .data = push(&a, (size_t)per * 3 * nf * SLICE_PATCH_PIXELS * sizeof(float))};
    images = push(&a, (size_t)per * REALISM_N);
    int workers = (int)strtol(env("HERCULEX_SYNTHETIC_THREADS", "0"), NULL, 10);
    if (workers <= 0) {
        workers = fmax(1, cores() - 2);
    }
#ifdef __APPLE__
    SynInjectLibrary library;
    if (injected) {
        library = syn_inject_library(&a, look);
    }
#endif
    json_object *family_names = json_object_new_array();
    for (int i = 0; i < nf; i++) {
        json_object_array_add(family_names, json_object_new_string(families[i]));
        for (int seed = 21; seed <= 23; seed++) {
            mark = a.used;
            json_object *spec = syn_family_draw(&a, families[i], seed, 0, look);
            size_t count;
#ifdef __APPLE__
            if (injected) {
                count = syn_inject_crops(&a, &library, spec, look, per, seed, images);
            } else
#endif
            {
                double voxel = 2 * jf64(spec, "voxel_um");
                count = syn_realism_crops(&a, spec, per, seed, voxel, 0, images, workers);
            }
            for (size_t k = 0; k < count; k++) {
                disc_patch(synthetic.data + synthetic.count++ * SLICE_PATCH_PIXELS,
                    images + k * REALISM_N, REALISM_SIDE, REALISM_SIDE, &rng);
            }
            json_object_put(spec);
            a.used = mark;
        }
    }
    need(synthetic.count, "no synthetic crops for %s", look);
    double auc, loss;
    disc_fit(&real, &synthetic, steps, 0, &auc, &loss);
    json_object *out = json_object_new_object();
    json_object_object_add(out, "look", json_object_new_string(look));
    if (!injected) {
        const char *looks =
            env("HERCULEX_SYNTHETIC_LOOKS", fmt(&a, "%s/synthetic/three_d/looks.json", HOME));
        json_object_object_add(out, "tag", json_object_new_string(tag));
        json_object_object_add(out, "looks_file", json_object_new_string(looks));
    }
    json_object_object_add(out, "real", json_object_new_int64(real.count));
    json_object_object_add(
        out, injected ? "injected" : "synthetic", json_object_new_int64(synthetic.count));
    syn_add(out, "auc", nearbyint(auc * 1000) / 1000);
    syn_add(out, "loss", nearbyint(loss * 10000) / 10000);
    json_object_object_add(out, "families", family_names);
    const char *line = fmt(&a, "%s\n", json_object_to_json_string_ext(out, JSON_C_TO_STRING_PLAIN));
    if (!injected) {
        const char *root = fmt(&a, "%s/synthetic/realism", OUTPUTS);
        mkdirs(root);
        const char *report = fmt(&a, "%s/discriminator.jsonl", root);
        int fd = open(report, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
        need(fd >= 0, "cannot open %s", report);
        write_all(fd, line, strlen(line), report);
        close(fd);
    }
    fputs(line, stdout);
    json_object_put(out);
    munmap(a.base, a.size);
    return 0;
}

static int syn_discriminator_command(int argc, char **argv) {
    return syn_discriminator_run(argc, argv, 0);
}

#ifdef __APPLE__
static int syn_injected_realism_command(int argc, char **argv) {
    return syn_discriminator_run(argc, argv, 1);
}
#endif
