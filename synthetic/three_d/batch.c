#include <glob.h>

static int syn_train_hard(SynFamily f) {
    return f.condition == SYN_PRESSED || f.condition == SYN_FUSED ||
        (f.shape >= SYN_U_FOLD && f.shape <= SYN_S_CORE);
}

static int syn_name_order(const void *left, const void *right) {
    const SynFamily *a = left, *b = right;
    return strcmp(a->name, b->name);
}

static void syn_batch_one(Arena *a, SynBandKey key, const char *root, const char *name,
    const char *input, const char *look, int slices, const char *device, int held) {
    size_t mark = a->used;
    if (!name) {
        char *flat = fmt(a, "%s", key.name);
        for (char *c = flat; *c; c++) {
            if (*c == '/') {
                *c = '-';
            }
        }
        name = fmt(a, "%s-s%lld", flat, (long long)key.seed);
    }
    const char *folder = fmt(a, "%s/%s", root, name);
    if (exists(fmt(a, "%s/truth.json", folder))) {
        if (!input) {
            printf("%s: done before\n", name);
        }
        a->used = mark;
        return;
    }
    double start = now();
    const char *staged = input ? input : fmt(a, "%s/.spec.%ld.json", folder, (long)getpid());
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        if (!input) {
            json_object *spec = syn_family_draw(a, key.name, key.seed, slices, look);
            json_write(staged, spec);
            json_object_put(spec);
        }
        int quiet = open("/dev/null", O_WRONLY);
        if (quiet >= 0) {
            dup2(quiet, STDOUT_FILENO);
            close(quiet);
        }
        execl("./run.sh", "./run.sh", "synthetic.three_d.band", "build", staged, "--name", name,
            "--root", root, "--device", device, NULL);
        _exit(127);
    }
    need(pid > 0, "cannot start synthetic band %s", name);
    int status;
    pid_t ended;
    do {
        ended = waitpid(pid, &status, 0);
    } while (ended < 0 && errno == EINTR);
    if (!input) {
        unlink(staged);
    }
    if (ended < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
        printf("%s: failed\n", name);
        a->used = mark;
        return;
    }
    json_object *band = json_read(fmt(a, "%s/band.json", folder)),
                *entry = json_object_new_object();
    if (input) {
        json_object *spec = json_read(input);
        json_object_object_add(entry, "seed", json_object_get(jget(spec, "seed")));
        json_object_put(spec);
    } else {
        json_object_object_add(entry, "seed", json_object_new_int64(key.seed));
    }
    json_object_object_add(entry, "name", json_object_new_string(name));
    json_object_object_add(entry, "held_out", json_object_new_boolean(held));
    const char *fields[] = {"family", "look", "digest", "nodes", "seconds"};
    for (int k = 0; k < 5; k++) {
        json_object_object_add(entry, fields[k], json_object_get(jget(band, fields[k])));
    }
    json_object *truth = jget(band, "truth");
    const char *metrics[] = {"pieces", "judged_area_cm2", "judged_terms_right"};
    for (int k = 0; k < 3; k++) {
        json_object_object_add(entry, metrics[k], json_object_get(jget(truth, metrics[k])));
    }
    const char *index = fmt(a, "%s/index.jsonl", root);
    int fd = open(index, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    need(fd >= 0, "cannot open %s", index);
    const char *line =
        fmt(a, "%s\n", json_object_to_json_string_ext(entry, JSON_C_TO_STRING_PLAIN));
    write_all(fd, line, strlen(line), index);
    close(fd);
    printf("%s: %s", name, line);
    if (!input) {
        printf("%s: %.0f s\n", name, now() - start);
    }
    fflush(stdout);
    json_object_put(entry);
    json_object_put(band);
    a->used = mark;
}

static int syn_batch_command(int argc, char **argv) {
    const char *usage =
        "Usage: ./run.sh synthetic.three_d.batch [FAMILY:SEED ...] "
        "[--first N] [--seed S] [--slices N] [--device cpu|gpu] [--part K/N] "
        "[--look SCROLL] [--root DIR] [--held-out N] [--plan LOG COUNT] [--specs DIR] [--index]";
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts(usage);
        return 0;
    }
    Arena a = arena(GB);
    const char *bands = fmt(&a, "%s/synthetic/bands", OUTPUTS), *root = bands;
    const char *look = NULL, *specs = NULL, *plan = NULL;
#ifdef __APPLE__
    const char *device = "gpu";
#else
    const char *device = "cpu";
#endif
    int first = 0, slices = 0, held = 0, count = 0, list = 0, k = 1, parts = 1, planned = 0;
    i64 seed = 1;
    SynBandKey *given = push(&a, (size_t)argc * sizeof(*given));
    for (int i = 0; i < argc; i++) {
        const char *arg = argv[i], *next = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(arg, "--index")) {
            list = 1;
        } else if (!strcmp(arg, "--plan") && i + 2 < argc) {
            plan = argv[++i];
            planned = (int)strtol(argv[++i], NULL, 10);
        } else if (!strcmp(arg, "--part") && next) {
            char *slash = NULL, *end = NULL;
            k = (int)strtol(next, &slash, 10);
            int split = slash != next && *slash == '/';
            parts = split ? (int)strtol(slash + 1, &end, 10) : 0;
            need(split && end != slash + 1 && !*end && k >= 1 && k <= parts,
                "invalid synthetic batch part %s", next);
            i++;
        } else if (next && !strcmp(arg, "--specs")) {
            specs = argv[++i];
        } else if (next && !strcmp(arg, "--look")) {
            look = argv[++i];
        } else if (next && !strcmp(arg, "--root")) {
            root = argv[++i];
        } else if (next && !strcmp(arg, "--device")) {
            device = argv[++i];
        } else if (next && !strcmp(arg, "--first")) {
            first = (int)strtol(argv[++i], NULL, 10);
        } else if (next && !strcmp(arg, "--slices")) {
            slices = (int)strtol(argv[++i], NULL, 10);
        } else if (next && !strcmp(arg, "--seed")) {
            seed = strtoll(argv[++i], NULL, 10);
        } else if (next && !strcmp(arg, "--held-out")) {
            held = (int)strtol(argv[++i], NULL, 10);
        } else {
            const char *sep = strrchr(arg, ':');
            need(sep && sep[1], "%s", usage);
            char *end;
            i64 value = strtoll(sep + 1, &end, 10);
            need(!*end, "%s", usage);
            given[count++] =
                (SynBandKey){.name = fmt(&a, "%.*s", (int)(sep - arg), arg), .seed = value};
        }
    }
    need(first >= 0 && planned >= 0 && held >= 0 && slices >= 0 &&
            (!strcmp(device, "gpu") || !strcmp(device, "cpu")),
        "%s", usage);
    SynFamilies reserved = syn_pool(1, 0);
    if (specs) {
        glob_t files = {0};
        glob(fmt(&a, "%s/*.json", specs), 0, NULL, &files);
        for (size_t i = 0; i < files.gl_pathc; i++) {
            const char *file = files.gl_pathv[i], *last = strrchr(file, '/') + 1;
            const char *name = fmt(&a, "%.*s", (int)strlen(last) - 5, last);
            json_object *spec = json_read(file);
            if (spec) {
                SynBandKey key = {.name = jstr(spec, "family"), .seed = ji64(spec, "seed")};
                syn_batch_one(&a, key, bands, name, file, NULL, 0, device,
                    syn_family_find(&reserved, key.name) != NULL);
                json_object_put(spec);
            }
        }
        globfree(&files);
        return 0;
    }
    if (list) {
        Bytes data = read_file(&a, fmt(&a, "%s/index.jsonl", bands));
        if (data.size) {
            fwrite(data.data, 1, data.size, stdout);
        }
        return 0;
    }
    size_t total = (size_t)count + first + planned + (size_t)held * reserved.count;
    SynBandKey *picks = push(&a, total * sizeof(*picks));
    memcpy(picks, given, (size_t)count * sizeof(*picks));
    size_t pos = count;
    SynFamilies training = syn_pool(0, 0);
    const char *early[] = {"flattened/pressed", "pancake/pressed", "oval/pressed", "pancake/fused",
        "u-fold/pressed", "crease/pressed", "u-fold/normal", "crease/normal", "round/normal"};
    const char *hard[147], *rest[147];
    int h = 0, n = 0;
    for (int i = 0; i < (int)(sizeof(early) / sizeof(*early)); i++) {
        hard[h++] = early[i];
    }
    for (int i = 0; i < training.count; i++) {
        SynFamily *f = training.rows + i;
        int have = 0;
        for (int j = 0; j < h; j++) {
            have |= !strcmp(hard[j], f->name);
        }
        if (have) {
            continue;
        }
        if (syn_train_hard(*f)) {
            hard[h++] = f->name;
        } else {
            rest[n++] = f->name;
        }
    }
    int round = 2 * h + n;
    for (int i = 0; i < first; i++) {
        int j = i % round;
        picks[pos++] = (SynBandKey){.name = j < 2 * h ? hard[j % h] : rest[j - 2 * h],
            .seed = seed + 2 * (i / round) + (j >= h && j < 2 * h)};
    }
    if (planned) {
        SynBandKey *rows = syn_sampler_plan(&a, bands, plan, planned, 1000);
        memcpy(picks + pos, rows, (size_t)planned * sizeof(*picks));
        pos += planned;
    }
    if (held) {
        root = fmt(&a, "%s/synthetic/held-out", OUTPUTS);
        qsort(reserved.rows, reserved.count, sizeof(*reserved.rows), syn_name_order);
        for (int i = 1; i <= held; i++) {
            for (int j = 0; j < reserved.count; j++) {
                picks[pos++] = (SynBandKey){.name = reserved.rows[j].name, .seed = i};
            }
        }
    }
    mkdirs(root);
    for (size_t i = (size_t)k - 1; i < pos; i += parts) {
        syn_batch_one(&a, picks[i], root, NULL, NULL, look, slices, device,
            syn_family_find(&reserved, picks[i].name) != NULL);
    }
    return 0;
}
