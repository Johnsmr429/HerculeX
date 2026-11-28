#define _GNU_SOURCE
#include <glob.h>
#include <openssl/evp.h>
#include <sys/wait.h>

#define HX_FETCH_VOLUME_CLI
#define HX_FETCH_VOLUME_LIBRARY
#include "../../scans/transfer/fetch_volume.c"
#include "../../storage/data_roots.c"
#include "../../sampling/touched_planes.c"

const char hx_usage[] = "Prefetch the CT planes of each complete piece unit while its scan fits.";

enum { LAYERS = 31, POLL_SECONDS = 30 };
enum { LOW_SPACE = 1, TAKEN_OVER, UNIT_CHANGED };

float *hx_readable_surface(const char *folder, int stride, double voxel, int *height, int *width);

typedef struct {
    char *pieces, *nodes, *sheet, *bricks;
    const char *sample, *scan;
    json_object *seen;
    double budget, idle;
    int connections, while_fitting;
    const char *method;
    char *url, *source;
} Fetch;

typedef struct {
    glob_t paths;
    struct timespec newest;
    size_t ready;
} Units;

static const char *basename_of(const char *path) {
    const char *name = strrchr(path, '/');
    return name ? name + 1 : path;
}

static json_object *document_at(const char *folder, const char *name) {
    char *path = path_join(folder, name);
    json_object *result = path ? json_object_from_file(path) : NULL;
    free(path);
    return result;
}

static const char *field(json_object *row, const char *name) {
    json_object *value = json_object_object_get(row, name);
    return json_object_is_type(value, json_type_string) ? json_object_get_string(value) : NULL;
}

static struct timespec modified(const char *folder, const char *name) {
    char *path = path_join(folder, name);
    struct stat info;
    int ok = path && !stat(path, &info) && S_ISREG(info.st_mode);
    free(path);
    if (!ok) {
        return (struct timespec){0};
    }
#ifdef __APPLE__
    return info.st_mtimespec;
#else
    return info.st_mtim;
#endif
}

static int later(struct timespec a, struct timespec b) {
    return a.tv_sec > b.tv_sec || (a.tv_sec == b.tv_sec && a.tv_nsec > b.tv_nsec);
}

static int units_of(const char *folder, Units *set) {
    *set = (Units){0};
    char *pattern = path_join(folder, "rows-*");
    int status = pattern ? glob(pattern, 0, NULL, &set->paths) : GLOB_NOSPACE;
    free(pattern);
    if (status == GLOB_NOMATCH) {
        globfree(&set->paths);
        set->paths = (glob_t){0};
        pattern = path_join(folder, "band-*");
        status = pattern ? glob(pattern, 0, NULL, &set->paths) : GLOB_NOSPACE;
        free(pattern);
    }
    return status == 0 || status == GLOB_NOMATCH;
}

static int published(const char *unit) {
    size_t length = strlen(unit);
    return length < 9 || strcmp(unit + length - 9, ".building") != 0;
}

static int current(const Fetch *fetch, const char *method) {
    if (!method) {
        return 0;
    }
    if (!strcmp(method, fetch->method)) {
        return 1;
    }
    FILE *aliases = fopen("level/level/same_output_methods.txt", "r");
    if (!aliases) {
        return 0;
    }
    char alias[32];
    int found = 0;
    while (!found && fscanf(aliases, "%31s", alias) == 1) {
        found = !strcmp(alias, method);
    }
    fclose(aliases);
    return found;
}

static json_object *finished(const Fetch *fetch, const char *unit) {
    json_object *record = published(unit) ? document_at(unit, "record.json") : NULL;
    if (record && current(fetch, field(record, "method")) &&
        json_object_is_type(json_object_object_get(record, "pieces"), json_type_array)) {
        return record;
    }
    json_object_put(record);
    return NULL;
}

static void latest(const Fetch *fetch, Units *pieces) {
    for (size_t i = 0; i < pieces->paths.gl_pathc; i++) {
        const char *path = pieces->paths.gl_pathv[i];
        json_object *record = finished(fetch, path);
        if (record) {
            struct timespec date = modified(path, "record.json");
            if (later(date, pieces->newest)) {
                pieces->newest = date;
            }
            pieces->ready++;
        }
        json_object_put(record);
    }
}

static int taken_over(const Fetch *fetch, const Units *pieces) {
    return pieces->ready &&
        (later(modified(fetch->sheet, "layout.json"), pieces->newest) ||
            later(modified(fetch->sheet, "reading.started"), pieces->newest));
}

static int all_nodes(const Fetch *fetch, int seen) {
    Units nodes;
    int all = units_of(fetch->nodes, &nodes), any = 0;
    for (size_t i = 0; all && i < nodes.paths.gl_pathc; i++) {
        const char *path = nodes.paths.gl_pathv[i];
        if (!published(path) || !modified(path, "record.json").tv_sec) {
            continue;
        }
        any = 1;
        const char *name = basename_of(path);
        if (seen) {
            all = json_object_object_get(fetch->seen, name) != NULL;
        } else {
            char *piece = path_join(fetch->pieces, name);
            json_object *record = piece ? finished(fetch, piece) : NULL;
            all = record != NULL;
            json_object_put(record);
            free(piece);
        }
    }
    globfree(&nodes.paths);
    return all && any;
}

static double stored_bytes(const Fetch *fetch, const Units *pieces) {
    double total = 0;
    for (size_t i = 0; i < pieces->paths.gl_pathc; i++) {
        const char *unit = pieces->paths.gl_pathv[i];
        json_object *record = finished(fetch, unit);
        if (!record) {
            continue;
        }
        json_object_put(record);
        char *folder = path_join(fetch->bricks, basename_of(unit));
        char *pattern = folder ? path_join(folder, "*") : NULL;
        glob_t files = {0};
        if (pattern && !glob(pattern, 0, NULL, &files)) {
            for (size_t k = 0; k < files.gl_pathc; k++) {
                struct stat info;
                if (!stat(files.gl_pathv[k], &info) && S_ISREG(info.st_mode)) {
                    total += info.st_size;
                }
            }
        }
        globfree(&files);
        free(pattern);
        free(folder);
    }
    return total;
}

static int command_output(char *const arguments[], pid_t *child) {
    int channel[2];
    if (pipe(channel)) {
        return -1;
    }
    *child = fork();
    if (!*child) {
        close(channel[0]);
        if (dup2(channel[1], STDOUT_FILENO) < 0) {
            _exit(1);
        }
        close(channel[1]);
        execv(arguments[0], arguments);
        _exit(127);
    }
    close(channel[1]);
    if (*child < 0) {
        close(channel[0]);
        return -1;
    }
    return channel[0];
}

static int wait_success(pid_t child) {
    int status = 0;
    pid_t ended;
    do {
        ended = waitpid(child, &status, 0);
    } while (ended < 0 && errno == EINTR);
    return ended == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

static json_object *catalog(const char *query, const char *sample, const char *scan) {
    char *arguments[] = {"./run.sh", "scans.catalog.scan_choice", (char *)query, "--json",
        (char *)sample, (char *)scan, NULL};
    pid_t child;
    int fd = command_output(arguments, &child);
    if (fd < 0) {
        return NULL;
    }
    json_object *rows = json_object_from_fd(fd);
    close(fd);
    if (!wait_success(child) || !json_object_is_type(rows, json_type_array)) {
        json_object_put(rows);
        return NULL;
    }
    return rows;
}

static int source_of(Fetch *fetch) {
    if (fetch->url && fetch->source) {
        return 1;
    }
    json_object *rows = catalog("volume", fetch->sample, fetch->scan);
    json_object *row = rows ? json_object_array_get_idx(rows, 0) : NULL;
    const char *root = field(json_object_object_get(row, "roots"), "ome-zarr");
    const char *path = field(json_object_object_get(row, "products"), "ome-zarr");
    char *clean = path ? trimmed_path(path) : NULL;
    char *url = NULL, *source = NULL;
    int ok = root && clean && asprintf(&url, "%s/%s/0", root, clean) >= 0 &&
        asprintf(&source, "%s/0", path) >= 0;
    free(clean);
    json_object_put(rows);
    if (!ok) {
        free(url);
        free(source);
        return 0;
    }
    fetch->url = url;
    fetch->source = source;
    return 1;
}

static int stamp_of(const char *scan, const char *unit, char *stamp) {
    char *file = path_join(unit, "record.json");
    FILE *record = file ? fopen(file, "rb") : NULL;
    free(file);
    char prefix[128];
    int prefix_bytes = snprintf(prefix, sizeof(prefix), "[\"%s\", %d]", scan, LAYERS);
    EVP_MD_CTX *digest = record ? EVP_MD_CTX_new() : NULL;
    int ok = digest && prefix_bytes > 0 && (size_t)prefix_bytes < sizeof(prefix) &&
        EVP_DigestInit_ex(digest, EVP_sha256(), NULL) &&
        EVP_DigestUpdate(digest, prefix, (size_t)prefix_bytes);
    char buffer[8192];
    while (ok) {
        size_t count = fread(buffer, 1, sizeof(buffer), record);
        if (!count) {
            break;
        }
        ok = EVP_DigestUpdate(digest, buffer, count);
    }
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    ok = ok && !ferror(record) && EVP_DigestFinal_ex(digest, bytes, &size) && size >= 8;
    if (ok) {
        for (int i = 0; i < 8; i++) {
            snprintf(stamp + 2 * i, 3, "%02x", bytes[i]);
        }
    }
    EVP_MD_CTX_free(digest);
    if (record) {
        fclose(record);
    }
    return ok;
}

static int atomic_document(const char *folder, const char *name, json_object *doc) {
    char *path = path_join(folder, name), *part = NULL;
    int ok = path && asprintf(&part, "%s.%ld.part", path, (long)getpid()) >= 0 &&
        !json_object_to_file_ext(part, doc, JSON_C_TO_STRING_PLAIN) && !rename(part, path);
    if (!ok && part) {
        unlink(part);
    }
    free(path);
    free(part);
    return ok;
}

static int string_order(const void *left, const void *right) {
    return strcmp(json_object_get_string(*(json_object *const *)left),
        json_object_get_string(*(json_object *const *)right));
}

static json_object *pieces_of(const char *unit, json_object *record) {
    json_object *pieces = json_object_object_get(record, "pieces");
    json_object *paths = json_object_new_array();
    size_t count = json_object_array_length(pieces);
    for (size_t i = 0; paths && i < count; i++) {
        json_object *piece = json_object_array_get_idx(pieces, i);
        if (!json_object_get_boolean(json_object_object_get(piece, "written"))) {
            continue;
        }
        json_object *id = json_object_object_get(piece, "piece");
        char *path = NULL;
        if (!json_object_is_type(id, json_type_int) ||
            asprintf(&path, "%s/piece-%04lld", unit, (long long)json_object_get_int64(id)) < 0) {
            json_object_put(paths);
            return NULL;
        }
        json_object_array_add(paths, json_object_new_string(path));
        free(path);
    }
    if (paths) {
        json_object_array_sort(paths, string_order);
    }
    return paths;
}

static int planned_paths(const char *folder, json_object *wanted, const char *stamp) {
    json_object *planned = document_at(folder, "planned.json");
    json_object *previous = document_at(folder, "unit.json");
    int valid = json_object_is_type(planned, json_type_array) &&
        json_object_is_type(previous, json_type_string) &&
        !strcmp(json_object_get_string(previous), stamp);
    size_t parts = valid ? json_object_array_length(planned) : 0;
    for (size_t i = 0, j = 0; valid && i < json_object_array_length(wanted); i++) {
        const char *path = json_object_get_string(json_object_array_get_idx(wanted, i));
        int match = 0;
        for (; !match && j < parts; j++) {
            json_object *entry = json_object_array_get_idx(planned, j);
            match = json_object_is_type(entry, json_type_string) &&
                !strcmp(path, json_object_get_string(entry));
        }
        valid = match;
    }
    json_object_put(planned);
    json_object_put(previous);
    return valid;
}

static int surface_ranges(Touched *touched, int32_t *geometry, json_object *paths) {
    size_t count = json_object_array_length(paths);
    for (size_t i = 0; i < count; i++) {
        const char *path = json_object_get_string(json_object_array_get_idx(paths, i));
        json_object *meta = document_at(path, "meta.json");
        double voxel = json_object_get_double(json_object_object_get(meta, "voxel_um"));
        int stride = json_object_get_int(json_object_object_get(meta, "stride"));
        json_object_put(meta);
        int h, w;
        float *surface = stride > 0 && isfinite(voxel) && voxel > 0
            ? hx_readable_surface(path, stride, voxel, &h, &w)
            : NULL;
        int ok = surface &&
            !mark_surface(
                touched, geometry, surface, surface + 3 * (size_t)h * w, h, w, stride, LAYERS);
        free(surface);
        if (!ok) {
            fprintf(stderr, "HerculeX: cannot plan CT planes for %s\n", path);
            return 0;
        }
    }
    return 1;
}

static int fill_store(Fetch *fetch, HxBrickStore *store, json_object *wanted) {
    int64_t bricked = store->count;
    if ((uint64_t)bricked > SIZE_MAX / sizeof(Key)) {
        return -1;
    }
    Range *ranges = malloc((size_t)bricked * sizeof(*ranges));
    int32_t geometry[6];
    int ok = ranges != NULL;
    for (int i = 0; ok && i < 3; i++) {
        ok = store->grid[i] <= INT32_MAX && store->brick[i] <= INT32_MAX;
        geometry[i] = store->grid[i];
        geometry[i + 3] = store->brick[i];
    }
    for (int64_t i = 0; ok && i < bricked; i++) {
        ranges[i] = (Range){.first = geometry[3], .last = -1};
    }
    Touched touched = {.ranges = ranges};
    ok = ok && surface_ranges(&touched, geometry, wanted);
    int64_t count = 0;
    for (int64_t i = 0; ok && i < bricked; i++) {
        count += ranges[i].last >= 0 && store->offsets[i] == HX_BRICK_UNKNOWN;
    }
    Key *keys = ok && count ? malloc((size_t)count * sizeof(*keys)) : NULL;
    ok = ok && (!count || keys);
    const int64_t plane = store->brick[1] * store->brick[2];
    const int64_t row = store->grid[1] * store->grid[2];
    for (int64_t i = 0, k = 0; ok && count && i < bricked; i++) {
        if (ranges[i].last >= 0 && store->offsets[i] == HX_BRICK_UNKNOWN) {
            keys[k++] = (Key){.z = i / row,
                .y = i / store->grid[2] % store->grid[1],
                .x = i % store->grid[2],
                .start = ranges[i].first * plane,
                .end = (ranges[i].last + 1) * plane - 1};
        }
    }
    free(ranges);
    Transfer transfer = {.store = store,
        .base = fetch->url,
        .keys = keys,
        .count = (size_t)count,
        .space_floor = 25ULL << 30,
        .write_lock = PTHREAD_MUTEX_INITIALIZER};
    int status = ok ? transfer_keys(&transfer, fetch->connections) : -1;
    free(keys);
    if (!status) {
        for (int64_t i = 0; i < bricked; i++) {
            if (store->offsets[i] == HX_BRICK_UNKNOWN) {
                store->offsets[i] = HX_BRICK_ABSENT;
            }
        }
        status = hx_brick_save(store);
    }
    return status;
}

static int fetch_unit(Fetch *fetch, const char *unit, const char *folder, json_object *record) {
    char stamp[17];
    json_object *wanted = pieces_of(unit, record);
    if (wanted && !json_object_array_length(wanted)) {
        json_object_put(wanted);
        return 0;
    }
    if (!wanted || !stamp_of(fetch->scan, unit, stamp)) {
        json_object_put(wanted);
        return -1;
    }
    struct stat info;
    int exists = !lstat(folder, &info);
    int valid = exists && S_ISDIR(info.st_mode) && planned_paths(folder, wanted, stamp);
    json_object *meta = NULL;
    int64_t shape[3], chunks[3];
    if (!exists && errno == ENOENT && source_of(fetch)) {
        meta = array_metadata(fetch->url);
        if (supported(meta, shape, chunks)) {
            valid = 1;
        }
    }
    HxBrickStore store;
    int status = valid ? hx_brick_open(&store, folder, exists ? NULL : shape,
                             exists ? NULL : chunks, fetch->source)
                       : -1;
    if (!status) {
        if (!exists) {
            json_object *version = json_object_new_string(stamp);
            if (!version || !atomic_document(folder, "unit.json", version) ||
                !atomic_document(folder, "planned.json", wanted)) {
                status = -1;
            }
            json_object_put(version);
        }
        int pending = 0;
        for (int64_t i = 0; !pending && i < store.count; i++) {
            pending = store.offsets[i] == HX_BRICK_UNKNOWN;
        }
        if (!status && pending) {
            status = source_of(fetch) && store.source && !strcmp(store.source, fetch->source)
                ? fill_store(fetch, &store, wanted)
                : -1;
        }
        hx_brick_close(&store);
    }
    json_object_put(meta);
    json_object_put(wanted);
    return status;
}

static int locked_fetch(Fetch *fetch, const char *unit) {
    const char *name = basename_of(unit);
    char *folder = path_join(fetch->bricks, name), *lock = NULL;
    int fd = folder && asprintf(&lock, "%s/.%s.fetching", fetch->bricks, name) >= 0 &&
            !make_directory(fetch->bricks)
        ? open(lock, O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0666)
        : -1;
    free(lock);
    int status = -1;
    if (fd >= 0 && !flock(fd, LOCK_EX)) {
        Units latest_pieces;
        if (units_of(fetch->pieces, &latest_pieces)) {
            latest(fetch, &latest_pieces);
            if (taken_over(fetch, &latest_pieces)) {
                status = TAKEN_OVER;
            } else {
                json_object *record = finished(fetch, unit);
                status = record ? fetch_unit(fetch, unit, folder, record) : UNIT_CHANGED;
                json_object_put(record);
            }
        }
        globfree(&latest_pieces.paths);
    }
    if (fd >= 0) {
        close(fd);
    }
    free(folder);
    return status;
}

static double wall_time(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

static void finish_event(const char *store, double began) {
    const char *path = getenv("HERCULEX_EVENTS");
    if (!path || !*path) {
        return;
    }
    json_object *name = json_object_new_string(store);
    char line[512];
    int n = name
        ? snprintf(line, sizeof(line),
              "{\"kind\":\"fetch\",\"store\":%s,\"start\":%.3f,\"end\":%.3f,\"pid\":%ld}\n",
              json_object_to_json_string(name), began, wall_time(), (long)getpid())
        : -1;
    json_object_put(name);
    if (n < 1 || (size_t)n >= sizeof(line)) {
        return;
    }
    int fd = open(path, O_APPEND | O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) {
        ssize_t sent;
        do {
            sent = write(fd, line, (size_t)n);
        } while (sent < 0 && errno == EINTR);
        close(fd);
    }
}

static void follow(Fetch *fetch) {
    double last_new = monotonic_seconds();
    for (;;) {
        Units pieces;
        if (!units_of(fetch->pieces, &pieces)) {
            globfree(&pieces.paths);
            break;
        }
        latest(fetch, &pieces);
        const char *stop = taken_over(fetch, &pieces) ? "the read has taken over"
            : fetch->while_fitting && pieces.ready && all_nodes(fetch, 0) ? "fitting finished"
                                                                          : NULL;
        for (size_t i = 0; !stop && i < pieces.paths.gl_pathc; i++) {
            const char *unit = pieces.paths.gl_pathv[i], *name = basename_of(unit);
            if (json_object_object_get(fetch->seen, name)) {
                continue;
            }
            json_object *record = finished(fetch, unit);
            if (!record) {
                continue;
            }
            json_object_put(record);
            double began = monotonic_seconds();
            if (stored_bytes(fetch, &pieces) >= fetch->budget) {
                stop = "store budget reached";
            } else {
                double began_event = wall_time();
                int status = locked_fetch(fetch, unit);
                finish_event(name, began_event);
                if (status == LOW_SPACE || status == TAKEN_OVER) {
                    stop = status == LOW_SPACE ? "disc room is short" : "the read has taken over";
                } else if (status != UNIT_CHANGED) {
                    printf("%s %s: %s in %.0f s\n", fetch->sample, name,
                        status ? "left CT to the reader" : "CT ready", monotonic_seconds() - began);
                    json_object_object_add(fetch->seen, name, json_object_new_boolean(1));
                    last_new = monotonic_seconds();
                }
            }
        }
        globfree(&pieces.paths);
        if (stop) {
            printf("%s %s: %s; the reader fetches from here\n", fetch->sample, fetch->scan, stop);
            return;
        }
        if (all_nodes(fetch, 1)) {
            printf("%s %s: every unit's CT fetch was attempted\n", fetch->sample, fetch->scan);
            return;
        }
        if (monotonic_seconds() - last_new >= fetch->idle) {
            printf("%s %s: no unit finished for %g h; stopping\n", fetch->sample, fetch->scan,
                fetch->idle / 3600);
            return;
        }
        sleep(POLL_SECONDS);
    }
}

static double number_setting(const char *name, double fallback) {
    const char *value = getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char *end;
    double number = strtod(value, &end);
    return *end || !isfinite(number) ? fallback : number;
}

static int enabled(const char *name) {
    const char *value = getenv(name);
    return value && !strcmp(value, "1");
}

static int component(const char *name) {
    size_t n = name ? strlen(name) : 0;
    return n && n < 80 && strcmp(name, ".") != 0 && strcmp(name, "..") != 0 &&
        strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") == n;
}

static int scan(const char *sample, const char *id, const char *method, int find_nodes) {
    if (!component(sample) || !component(id)) {
        return 1;
    }
    char *outputs = local_root(outputs_root), *library = local_root(library_root);
    char *real = outputs ? realpath(outputs, NULL) : NULL;
    const char *root = real ? real : outputs;
    const char *override = getenv("HERCULEX_BRICKS");
    if (override && !*override) {
        override = NULL;
    }
    Fetch fetch = {.sample = sample,
        .scan = id,
        .seen = json_object_new_object(),
        .budget = number_setting("HERCULEX_FETCH_AHEAD_GIB", 200) * 0x1p30,
        .idle = number_setting("HERCULEX_FETCH_IDLE_HOURS", 1.5) * 3600,
        .connections = (int)fmax(1, fmin(64, number_setting("HERCULEX_FETCH_CONNECTIONS", 32))),
        .while_fitting =
            enabled("HERCULEX_PREFETCH_WHILE_FITTING") && !enabled("HERCULEX_KEEP_BRICKS"),
        .method = method};
    int ok = root && library && fetch.seen &&
        asprintf(&fetch.pieces, "%s/unwrapping/level/%s/%s/pieces", root, sample, id) >= 0 &&
        asprintf(&fetch.sheet, "%s/unwrapping/level/%s/%s/sheet", root, sample, id) >= 0 &&
        asprintf(&fetch.nodes, "%s/unwrapping/nodes/%s/%s/nodes", root, sample, id) >= 0;
    if (ok) {
        ok = override ? asprintf(&fetch.bricks, "%s/%s/ct_bricks", override, sample) >= 0
                      : asprintf(&fetch.bricks, "%s/cache/bricks/%s", library, sample) >= 0;
    }
    Units nodes = {0};
    int present = !find_nodes;
    if (ok && find_nodes && units_of(fetch.nodes, &nodes)) {
        for (size_t i = 0; !present && i < nodes.paths.gl_pathc; i++) {
            const char *unit = nodes.paths.gl_pathv[i];
            present = published(unit) && modified(unit, "record.json").tv_sec != 0;
        }
    }
    globfree(&nodes.paths);
    if (ok && present) {
        follow(&fetch);
    }
    json_object_put(fetch.seen);
    free(fetch.pieces);
    free(fetch.nodes);
    free(fetch.sheet);
    free(fetch.bricks);
    free(fetch.url);
    free(fetch.source);
    free(library);
    free(outputs);
    free(real);
    return !ok;
}

static int run(int argc, char **argv, const char *method) {
    const char *id = NULL;
    char **samples = calloc((size_t)argc + 1, sizeof(*samples));
    if (!samples) {
        return 1;
    }
    int count = 0, status = 0;
    for (int i = 0; !status && i < argc; i++) {
        if (!strcmp(argv[i], "--scan") && i + 1 < argc) {
            id = argv[++i];
        } else if (!strcmp(argv[i], "--since") && i + 1 < argc) {
            i++;
        } else if (strcmp(argv[i], "--band") != 0) {
            if (argv[i][0] == '-' || !component(argv[i])) {
                status = 2;
            } else {
                samples[count++] = argv[i];
            }
        }
    }
    if (!status && id) {
        for (int i = 0; !status && i < count; i++) {
            status = scan(samples[i], id, method, 1);
        }
    } else if (!status) {
        for (int i = 0; !status && i < (count ? count : 1); i++) {
            json_object *rows = catalog("chosen", samples[i], NULL);
            if (!rows) {
                status = 1;
            }
            size_t size = rows ? json_object_array_length(rows) : 0;
            for (size_t j = 0; !status && j < size; j++) {
                json_object *row = json_object_array_get_idx(rows, j);
                status = scan(field(row, "sample"), field(row, "volume"), method, 1);
            }
            json_object_put(rows);
        }
    }
    free(samples);
    return status;
}

static int level_method(char *method, int size) {
    char *arguments[] = {"./run.sh", "level.level.run_levels", "method", NULL};
    pid_t child;
    int fd = command_output(arguments, &child);
    if (fd < 0) {
        return 0;
    }
    FILE *output = fdopen(fd, "r");
    int available = output && fgets(method, size, output);
    if (output) {
        fclose(output);
    } else {
        close(fd);
    }
    if (!wait_success(child) || !available) {
        return 0;
    }
    method[strcspn(method, "\r\n")] = 0;
    return component(method);
}

static int prefetch_main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        puts("Usage: ./run.sh ink.reading.unit_fetch SCROLL SCAN\n"
             "       ./run.sh ink.reading.unit_fetch run [SCROLL ...] [--scan SCAN]");
        return argc == 2 ? 0 : 2;
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT)) {
        return 1;
    }
    char method[81];
    if (!level_method(method, sizeof(method))) {
        fputs("HerculeX: cannot obtain the level method for CT prefetch\n", stderr);
        curl_global_cleanup();
        return 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    int status = !strcmp(argv[1], "run") ? run(argc - 2, argv + 2, method)
        : argc == 3                      ? scan(argv[1], argv[2], method, 0)
                                         : 2;
    curl_global_cleanup();
    return status;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }
    return !strcmp(argv[1], "scans.transfer.fetch_volume") ? fetch_volume_main(argc - 1, argv + 1)
                                                           : prefetch_main(argc - 1, argv + 1);
}
