#ifndef HX_UNITY
const char hx_usage[] =
    "Generate labelled synthetic CT tiles with the Synthetic-Slices diffusion model.";
#endif

static const char *const slice_styles[] = {
    "PHerc0211",
    "PHerc1545",
    "PHerc1203",
    "PHerc0175A",
    "PHerc0358",
    "PHerc0813",
    "PHerc0826",
    "PHerc0191",
    "PHerc0306B",
    "Paris4",
};

static void slices_distances(Arena *mem, const u16 *labels, u8 *lines, u8 *out, int w, int h) {
    size_t used = mem->used;
    int *dx = push(mem, (size_t)w * h * sizeof(*dx));
    int *rows = push(mem, (size_t)h * sizeof(*rows));
    double *breaks = push(mem, (size_t)(h + 1) * sizeof(*breaks));
    for (int y = 0; y < h; y++) {
        size_t at = (size_t)y * w;
        int gap = INT_MAX;
        for (int x = 0; x < w; x++) {
            lines[at + x] = labels[at + x] != 0;
            gap = lines[at + x] ? 0 : gap + (gap < INT_MAX);
            dx[at + x] = gap;
        }
        gap = INT_MAX;
        for (int x = w - 1; x >= 0; x--) {
            gap = lines[at + x] ? 0 : gap + (gap < INT_MAX);
            dx[at + x] = slices_min(dx[at + x], gap);
        }
    }
    for (int x = 0; x < w; x++) {
        int next = 0;
        for (int y = 0; y < h; y++) {
            int gap = dx[(size_t)y * w + x];
            if (gap == INT_MAX) {
                continue;
            }
            double start = -INFINITY;
            while (next > 0) {
                int last = rows[next - 1], last_gap = dx[(size_t)last * w + x];
                start = ((double)gap * gap - (double)last_gap * last_gap + (double)y * y -
                            (double)last * last) /
                    (2.0 * (y - last));
                if (start > breaks[next - 1]) {
                    break;
                }
                next--;
            }
            rows[next] = y;
            breaks[next++] = start;
        }
        breaks[next] = INFINITY;
        int nearest = 0;
        for (int y = 0; y < h; y++) {
            while (nearest + 1 < next && breaks[nearest + 1] < y) {
                nearest++;
            }
            int ry = next ? rows[nearest] : -1;
            int gap = next ? dx[(size_t)ry * w + x] : x;
            double distance = nearbyint(hypot(y - ry, gap));
            out[(size_t)y * w + x] = (u8)fmin(distance, 255);
        }
    }
    mem->used = used;
}

static const char *slices_cli_option(const char *arg, char key[32], const char **value) {
    *value = NULL;
    const char *equal = strncmp(arg, "--", 2) == 0 ? strchr(arg, '=') : NULL;
    if (!equal) {
        return arg;
    }
    size_t n = (size_t)(equal - arg);
    need(n < 32, "unknown synthetic slice option: %s", arg);
    memcpy(key, arg, n);
    key[n] = 0;
    *value = equal + 1;
    return key;
}

static u64 slices_unsigned(const char *value, const char *key) {
    char *end;
    errno = 0;
    u64 result = strtoull(value, &end, 10);
    need(*value >= '0' && *value <= '9' && !*end && !errno, "invalid %s: %s", key, value);
    return result;
}

static int slices_positive(const char *value, const char *key) {
    u64 result = slices_unsigned(value, key);
    need(result && result < INT_MAX, "invalid %s: %s", key, value);
    return (int)result;
}

static int slices_style(const char *name) {
    if (!strcmp(name, "random")) {
        return -1;
    }
    for (int i = 0; i < (int)(sizeof(slice_styles) / sizeof(*slice_styles)); i++) {
        if (!strcmp(name, slice_styles[i])) {
            return i;
        }
    }
    die("unknown synthetic slice style: %s", name);
    return 0;
}

static const char slice_source[] = "https://github.com/vincentl065/Synthetic-Slices";
static const char slice_revision[] = "e09a22d1944c9df03d94d29331a894ec5944299d";
static const char slice_repo[] = "vincentl065/Synthetic-Slices";
static const char slice_weight_revision[] = "3f645a40b684a613ace83cfba82422c3e88619a1";
static const char slice_weight_file[] = "synthetic_slices_ema_6655.safetensors";

typedef struct {
    const char *out, *weights, *device;
    int size, count, batch, windows, steps, style;
    u64 seed;
} SliceOptions;

typedef enum { SLICES_LAYOUT, SLICES_GENERATE_NATIVE, SLICES_GENERATE_CPU } SliceCommand;

#if defined(__APPLE__) && defined(__OBJC__)
static const char slice_native_device[] = "mps";
static const char slice_device_options[] = "auto|cpu|mps";
#elif defined(HX_SLICES_CUDA)
static const char slice_native_device[] = "cuda";
static const char slice_device_options[] = "auto|cpu|cuda";
#else
static const char slice_native_device[] = "cpu";
static const char slice_device_options[] = "auto|cpu";
#endif

static int slices_options(SliceOptions *o, int argc, char **argv, SliceCommand command) {
    int generate = command != SLICES_LAYOUT;
    int cpu_only = command == SLICES_GENERATE_CPU;
    *o = (SliceOptions){
        .size = 256,
        .count = 1,
        .batch = 4,
        .windows = 4,
        .steps = 32,
        .style = -1,
        .device = command == SLICES_GENERATE_NATIVE ? "auto" : "cpu",
    };
    for (int i = 0; i < argc; i++) {
        char split[32];
        const char *value;
        const char *key = slices_cli_option(argv[i], split, &value);
        if (!strcmp(key, "--help") || !strcmp(key, "-h")) {
            printf("%s --out DIR [--n N] [--size N] [--seed N] "
                   "[--style random|SCROLL] [--batch N]",
                command == SLICES_GENERATE_NATIVE ? "synthetic.two_d.generate.native"
                    : cpu_only                    ? "synthetic.two_d.generate.cpu"
                                                  : "synthetic.two_d.layout");
            if (generate) {
                printf(" [--device %s] [--windows N] [--steps N] [--weights FILE]",
                    cpu_only ? "auto|cpu" : slice_device_options);
            }
            putchar('\n');
            return 0;
        }
        if (!value) {
            need(i + 1 < argc, "%s needs a value", key);
            value = argv[++i];
        }
        if (!strcmp(key, "--out")) {
            o->out = value;
        } else if (!strcmp(key, "--n")) {
            o->count = slices_positive(value, key);
        } else if (!strcmp(key, "--size")) {
            o->size = slices_positive(value, key);
        } else if (!strcmp(key, "--seed")) {
            o->seed = slices_unsigned(value, key);
        } else if (!strcmp(key, "--style")) {
            o->style = slices_style(value);
        } else if (!strcmp(key, "--batch")) {
            o->batch = slices_positive(value, key);
        } else if (generate && !strcmp(key, "--weights")) {
            o->weights = value;
        } else if (generate && !strcmp(key, "--windows")) {
            o->windows = slices_positive(value, key);
        } else if (generate && !strcmp(key, "--steps")) {
            o->steps = slices_positive(value, key);
        } else if (generate && !strcmp(key, "--device")) {
            o->device = value;
        } else {
            die("unknown synthetic slice option: %s", key);
        }
    }
    need(o->out != NULL, "synthetic slices require --out");
    need(o->size >= 256 && o->size % 128 == 0, "--size must be a multiple of 128 and at least 256");
    need(o->size < INT_MAX / 4, "--size is too large");
    need(o->steps >= 2, "--steps must be at least 2");
    if (!strcmp(o->device, "auto")) {
        o->device = "cpu";
#ifdef HX_SLICES_ACCELERATOR
        if (!cpu_only && slices_accelerator_available()) {
            o->device = slice_native_device;
        }
#endif
    }
    int cpu = !strcmp(o->device, "cpu");
    need(!cpu_only || cpu, "synthetic.two_d.generate.cpu requires --device cpu");
    need(cpu || !strcmp(o->device, slice_native_device), "device must be one of %s",
        slice_device_options);
    need(strlen(o->out) + 32 < PATH_MAX, "output directory name is too long");
    need(!exists(o->out), "output directory already exists: %s", o->out);
    return 1;
}

static const char *slices_released_weights(Arena *mem) {
    const char *cache = fmt(mem, "%s/weights/synthetic-slices", LIBRARY);
    const char *snapshot = fmt(mem, "%s/models--vincentl065--Synthetic-Slices/snapshots/%s/%s",
        cache, slice_weight_revision, slice_weight_file);
    if (exists(snapshot)) {
        return snapshot;
    }
    const char *dest = fmt(mem, "%s/%s/%s", cache, slice_weight_revision, slice_weight_file);
    if (exists(dest)) {
        return dest;
    }
    const char *url = fmt(mem, "https://huggingface.co/%s/resolve/%s/%s", slice_repo,
        slice_weight_revision, slice_weight_file);
    char *part = fmt(mem, "%s.XXXXXX", dest);
    mkparent(part);
    int fd = mkstemp(part);
    need(fd >= 0, "cannot create checkpoint cache in %s: %s", cache, strerror(errno));
    FILE *file = fdopen(fd, "wb");
    CURL *curl = curl_easy_init();
    need(curl && file, "cannot open checkpoint download");
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 4096L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);
    CURLcode result = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    int closed = fclose(file);
    if (result != CURLE_OK || closed || rename(part, dest)) {
        unlink(part);
        die("cannot cache Synthetic-Slices checkpoint: %s",
            result == CURLE_OK ? strerror(errno) : curl_easy_strerror(result));
    }
    return dest;
}

static void slices_manifest(Arena *mem, const SliceOptions *o) {
    json_object *metadata = json_object_new_object(), *styles = json_object_new_array();
    for (size_t i = 0; i < sizeof(slice_styles) / sizeof(*slice_styles); i++) {
        json_object_array_add(styles, json_object_new_string(slice_styles[i]));
    }
    const char *weights =
        o->weights ? o->weights : fmt(mem, "%s/%s", slice_repo, slice_weight_file);
    json_object_object_add(metadata, "source", json_object_new_string(slice_source));
    json_object_object_add(metadata, "source_revision", json_object_new_string(slice_revision));
    json_object_object_add(metadata, "weights", json_object_new_string(weights));
    json_object_object_add(metadata, "weights_revision",
        o->weights ? NULL : json_object_new_string(slice_weight_revision));
    json_object_object_add(metadata, "styles", styles);
    json_object_object_add(metadata, "style_requested",
        json_object_new_string(o->style < 0 ? "random" : slice_styles[o->style]));
    json_object_object_add(metadata, "seed", json_object_new_uint64(o->seed));
    json_object_object_add(metadata, "device", json_object_new_string(o->device));
    json_object_object_add(metadata, "voxel_um_nominal", jnum(9.4));
    json_object_object_add(metadata, "tile_size", json_object_new_int(o->size));
    json_object_object_add(metadata, "tiles_requested", json_object_new_int(o->count));
    json_object_object_add(metadata, "batch", json_object_new_int(o->batch));
    json_object_object_add(metadata, "windows", json_object_new_int(o->windows));
    json_object_object_add(metadata, "steps", json_object_new_int(o->steps));
    json_object_object_add(metadata, "layout_license", json_object_new_string("MIT"));
    json_object_object_add(metadata, "released_weights_and_images_license",
        o->weights ? NULL : json_object_new_string("CC-BY-NC-4.0"));
    json_write(fmt(mem, "%s/manifest.json", o->out), metadata);
    json_object_put(metadata);
}

static int slices_generate_tiles(int argc, char **argv, SliceCommand command) {
    SliceOptions o;
    if (!slices_options(&o, argc, argv, command)) {
        return 0;
    }
    int generate = command != SLICES_LAYOUT;
    SliceModel model = {0};
    if (generate) {
        Arena paths = arena(1 << 20);
        model = slices_model_open(o.weights ? o.weights : slices_released_weights(&paths));
        munmap(paths.base, paths.size);
#ifdef HX_SLICES_ACCELERATOR
        if (strcmp(o.device, "cpu") != 0) {
            model.accelerator = slices_accelerator_open(o.seed);
        }
#endif
    }
    mkparent(o.out);
    need(mkdir(o.out, 0775) == 0, "cannot create %s: %s", o.out, strerror(errno));

    int batch = slices_min(o.batch, o.count), size = o.size;
    size_t tile = (size_t)size * size, scale = (4 + generate) * (size_t)batch + 5;
    need(tile < (SIZE_MAX - (1 << 20)) / scale, "synthetic layout is too large");
    Arena mem = arena(tile * scale + (1 << 20));
    if (generate) {
        slices_manifest(&mem, &o);
    }
    u16 *labels = push(&mem, tile * batch * sizeof(*labels));
    u8 *lines = push(&mem, tile * batch), *dist = push(&mem, tile * batch);
    u8 *images = generate ? push(&mem, tile * batch) : NULL;
    i64 *styles = push(&mem, (size_t)batch * sizeof(*styles));
    size_t used = mem.used;
    Pcg64 rng = pcg64(o.seed);
    SliceNoise noise = slices_noise_seed(o.seed);
#ifdef HX_SLICES_CUDA
    if (model.accelerator) {
        noise.accelerator = model.accelerator;
        noise.accelerated_fill = slices_accelerator_noise;
    }
#endif
    int low_precision = !strcmp(o.device, "mps");
    for (int done = 0, shard = 0; done < o.count; shard++) {
        int n = slices_min(batch, o.count - done);
        for (int i = 0; i < n; i++) {
            slices_layout(&rng, size, size, labels + tile * i);
            styles[i] = o.style < 0 ? (i64)draw(&rng, 9) : o.style;
            slices_distances(
                &mem, labels + tile * i, lines + tile * i, dist + tile * i, size, size);
        }
        if (generate) {
            slices_sample(&noise, slices_model_network, &model, size, size, n, o.windows, o.steps,
                low_precision, lines, dist, styles, images);
        }
        const char *names[] = {"image", "lines", "sheet_id", "dist", "style", "seed"};
        Array arrays[] = {
            {.descr = "|u1", .ndim = 3, .shape = {n, size, size}, .data = images, .size = tile * n},
            {.descr = "|u1", .ndim = 3, .shape = {n, size, size}, .data = lines, .size = tile * n},
            {.descr = "<u2",
                .ndim = 3,
                .shape = {n, size, size},
                .data = labels,
                .size = tile * n * 2},
            {.descr = "|u1", .ndim = 3, .shape = {n, size, size}, .data = dist, .size = tile * n},
            {.descr = "<i8", .ndim = 1, .shape = {n}, .data = styles, .size = (size_t)n * 8},
            {.descr = "<u8", .ndim = 0, .data = &o.seed, .size = sizeof(o.seed)},
        };
        const char *path = fmt(&mem, "%s/shard_%04d.npz", o.out, shard);
        int first = !generate;
        npz_write_codec(path, (int)(sizeof(arrays) / sizeof(*arrays)) - first, names + first,
            arrays + first, ZIP_CM_DEFLATE);
        done += n;
        printf("%d/%d tiles · %s\n", done, o.count, path);
        fflush(stdout);
        mem.used = used;
    }
    if (generate) {
        slices_model_close(&model);
    }
    munmap(mem.base, mem.size);
    return 0;
}

static int slices_layout_command(int argc, char **argv) {
    return slices_generate_tiles(argc, argv, SLICES_LAYOUT);
}

static int slices_generate_command(int argc, char **argv) {
    return slices_generate_tiles(argc, argv, SLICES_GENERATE_CPU);
}

static int slices_generate_native_command(int argc, char **argv) {
    return slices_generate_tiles(argc, argv, SLICES_GENERATE_NATIVE);
}
