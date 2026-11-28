enum { SPOT_LAYERS = 67, SPOT_MIDDLE = 33, SPOT_REACH = 15, SPOT_DEPTHS = 31 };

static const char *spot_surfaces[] = {"grown", "m7", "original"};
static const char *spot_sides[] = {"inward", "outward"};

typedef struct {
    const char *name, *weights;
} SpotModel;

static Bytes spot_map(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0 && errno == ENOENT) {
        return (Bytes){0};
    }
    struct stat st;
    need(fd >= 0 && !fstat(fd, &st) && st.st_size >= 10, "cannot read %s", path);
    u8 *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    need(data != MAP_FAILED, "cannot map %s", path);
    return (Bytes){.data = data, .size = (size_t)st.st_size};
}

static const char *spot_work(Arena *a, json_object *spot) {
    const char *id = jstr(spot, "spot_id");
    need(id && site_name(id), "invalid spot ID");
    return fmt(a, "%s/studies/shared/spot-candidates/spots/%s", LIBRARY, id);
}

static const char *spot_piece(json_object *record) {
    const char *name = jstr(record, "name");
    need(name && site_name(name), "invalid piece name in spot receipt");
    return name;
}

static Array spot_stack(Bytes b, const char *path) {
    Array stack = npy_parse(b, path);
    need(stack.ndim == 3 && stack.shape[0] == SPOT_LAYERS && stack.shape[1] > 0 &&
            stack.shape[2] > 0 && stack.shape[1] <= INT_MAX / stack.shape[2] &&
            stack.size == (size_t)elements(stack) && !strcmp(stack.descr, "|u1"),
        "%s is not a uint8 67-layer spot stack", path);
    return stack;
}

static void spot_read_model(
    Arena *a, const char *work, const SpotModel *model, int stock, json_object *pieces) {
    size_t count = json_object_array_length(pieces);
    i64 max_pixels = 0;
    for (int s = 0; s < (stock ? 3 : 2); s++) {
        for (size_t i = 0; i < count; i++) {
            size_t mark = a->used;
            const char *name = spot_piece(json_object_array_get_idx(pieces, i));
            const char *out =
                fmt(a, "%s/reads/%s/%s/%s.npy", work, spot_surfaces[s], model->name, name);
            const char *path = fmt(a, "%s/stacks/%s/%s.npy", work, spot_surfaces[s], name);
            Bytes mapped = exists(out) ? (Bytes){0} : spot_map(path);
            if (mapped.data) {
                Array stack = spot_stack(mapped, path);
                i64 n = stack.shape[1] * stack.shape[2];
                if (n > max_pixels) {
                    max_pixels = n;
                }
                munmap(mapped.data, mapped.size);
            }
            a->used = mark;
        }
    }
    if (!max_pixels) {
        return;
    }
    need(exists(model->weights), "%s has no checkpoint at %s", model->name, model->weights);
    Reader *reader = reader_open(a, model->weights, max_pixels);
    u8 *slab = push(a, DEPTH * max_pixels);
    f32 *probability = push(a, max_pixels * sizeof(*probability));
    _Float16 *encoded = push(a, max_pixels * sizeof(*encoded));
    for (int s = 0; s < (stock ? 3 : 2); s++) {
        for (size_t i = 0; i < count; i++) {
            size_t mark = a->used;
            const char *name = spot_piece(json_object_array_get_idx(pieces, i));
            const char *surface = spot_surfaces[s];
            const char *out = fmt(a, "%s/reads/%s/%s/%s.npy", work, surface, model->name, name);
            const char *path = fmt(a, "%s/stacks/%s/%s.npy", work, surface, name);
            Bytes mapped = exists(out) ? (Bytes){0} : spot_map(path);
            if (!mapped.data) {
                a->used = mark;
                continue;
            }
            double began = now();
            Array stack = spot_stack(mapped, path);
            int h = (int)stack.shape[1], w = (int)stack.shape[2];
            i64 columns = (i64)h * w;
            need(columns <= max_pixels, "spot stack changed while reading %s", path);
            char *part = fmt(a, "%s.part.XXXXXXXX", out);
            mkparent(part);
            int fd = mkstemp(part);
            need(fd >= 0, "cannot stage %s", out);
            Bytes header = npy_bytes(a,
                (Array){
                    .descr = "<f2", .ndim = 4, .shape = {2, SPOT_DEPTHS, h, w}, .data = encoded});
            write_all(fd, header.data, header.size, part);
            const u8 *pixels = stack.data;
            for (int side = 0; side < 2; side++) {
                for (int depth = -SPOT_REACH; depth <= SPOT_REACH; depth++) {
                    const u8 *window = pixels + (SPOT_MIDDLE + depth - DEPTH / 2) * columns;
                    if (!side) {
                        for (int z = 0; z < DEPTH; z++) {
                            memcpy(slab + z * columns, window + (DEPTH - 1 - z) * columns, columns);
                        }
                        window = slab;
                    }
                    fix_stem(reader, window, h, w);
                    ink_read(reader, window, h, w, probability);
                    for (i64 k = 0; k < columns; k++) {
                        encoded[k] = probability[k];
                    }
                    write_all(fd, encoded, columns * sizeof(*encoded), part);
                }
            }
            need(!close(fd) && !rename(part, out), "cannot finish %s", out);
            munmap(mapped.data, mapped.size);
            printf("%s %s %s read (2, %d, %d, %d) %.1f s\n", surface, model->name, name,
                SPOT_DEPTHS, h, w, now() - began);
            a->used = mark;
        }
    }
}

static void spot_sweep(Arena *a, json_object *spot, json_object *document) {
    const char *work = spot_work(a, spot);
    const char *root = fmt(a, "%s/weights/ink", LIBRARY);
    json_object *listed = json_object_from_file(fmt(a, "%s/models.json", root));
    int count = 1 + (listed ? json_object_object_length(listed) : 0);
    SpotModel *models = push(a, (i64)count * sizeof(*models));
    models[0] = (SpotModel){
        .name = STOCK_MODEL, .weights = fmt(a, "%s/%s/tile-reader.pth", root, STOCK_MODEL)};
    int n = 1;
    if (listed) {
        json_object_object_foreach(listed, name, entry) {
            const char *file = jstr(entry, "file");
            need(site_name(name) && file, "invalid spot ink model %s", name);
            if (strcmp(name, STOCK_MODEL) != 0) {
                const char *path = *file == '/' ? file : fmt(a, "%s/%s", root, file);
                models[n++] = (SpotModel){.name = name, .weights = path};
            }
        }
    }
    json_object *pieces = jget(document, "pieces");
    need(json_object_is_type(pieces, json_type_array), "the spot has no built pieces");
    for (int i = 0; i < n; i++) {
        fflush(NULL);
        pid_t pid = fork();
        need(pid >= 0, "cannot start the spot reader");
        if (!pid) {
            Arena reader = arena(32 * GB);
            spot_read_model(&reader, work, models + i, i == 0, pieces);
            fflush(stdout);
            _exit(0);
        }
        int status;
        pid_t result;
        do {
            result = waitpid(pid, &status, 0);
        } while (result < 0 && errno == EINTR);
        need(result == pid && WIFEXITED(status) && !WEXITSTATUS(status), "spot read failed for %s",
            models[i].name);
    }
    json_object *recipe = json_object_new_object(), *depths = json_object_new_array();
    json_object *sides = json_object_new_array_ext(2), *read_by = json_object_new_object();
    for (int d = -SPOT_REACH; d <= SPOT_REACH; d++) {
        json_object_array_add(depths, json_object_new_int(d));
    }
    for (int i = 0; i < 2; i++) {
        json_object_array_add(sides, json_object_new_string(spot_sides[i]));
    }
    for (int i = 0; i < 3; i++) {
        json_object *names = json_object_new_array_ext(i < 2 ? n : 1);
        for (int m = 0; m < (i < 2 ? n : 1); m++) {
            json_object_array_add(names, json_object_new_string(models[m].name));
        }
        json_object_object_add(read_by, spot_surfaces[i], names);
    }
    json_object_object_add(recipe, "depths", depths);
    json_object_object_add(recipe, "sides", sides);
    json_object_object_add(recipe, "window", json_object_new_int(DEPTH));
    json_object_object_add(recipe, "middle", json_object_new_int(SPOT_MIDDLE));
    json_object_object_add(recipe, "layers", json_object_new_int(SPOT_LAYERS));
    json_object_object_add(recipe, "read_by", read_by);
    json_write(fmt(a, "%s/reads/recipe.json", work), recipe);
    json_object_put(recipe);
    json_object_put(listed);
}
