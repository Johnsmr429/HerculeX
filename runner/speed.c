#ifndef HX_UNITY
const char hx_usage[] = "Time fixed workloads and compare them with recent benchmark records.";
#endif

#include <netinet/in.h>
#include <sys/socket.h>

static double speed_cpu(void) {
    struct rusage u;
    need(getrusage(RUSAGE_SELF, &u) == 0, "cannot read benchmark CPU time");
    return u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 + u.ru_stime.tv_sec +
        u.ru_stime.tv_usec * 1e-6;
}

static void speed_elapsed(json_object *row, double began) {
    json_object_object_add(row, "seconds", jnum(now() - began));
}

static json_object *speed_solver(Arena *a) {
    enum { COUNT = 200000, TERMS = 1000000 };
    i64 hx_level_minimize(i64, i64, const i32 *, const i32 *, const i32 *, const i32 *, const i32 *,
        i32 *, i32, i64 *);
    i32 *p = push(a, TERMS * sizeof(*p)), *q = push(a, TERMS * sizeof(*q));
    i32 *k = push(a, TERMS * sizeof(*k)), *up = push(a, TERMS * sizeof(*up));
    i32 *down = push(a, TERMS * sizeof(*down)), *x = pushz(a, COUNT * sizeof(*x));
    Pcg64 rng = pcg64(0);
    for (int i = 0; i < TERMS; i++) {
        p[i] = draw(&rng, COUNT);
    }
    for (int i = 0; i < TERMS; i++) {
        q[i] = (p[i] + 1 + draw(&rng, 49)) % COUNT;
    }
    for (int i = 0; i < TERMS; i++) {
        k[i] = (int)draw(&rng, 5) - 2;
    }
    for (int i = 0; i < 2 * TERMS; i++) {
        (i < TERMS ? up : down)[i % TERMS] = draw(&rng, 63) + 1;
    }
    i64 log[3] = {0};
    double start = now();
    need(hx_level_minimize(COUNT, TERMS, p, q, k, up, down, x, 1, log) >= 0,
        "benchmark exact level solve failed");
    json_object *row = json_object_new_object();
    speed_elapsed(row, start);
    json_object_object_add(row, "terms", json_object_new_int(TERMS));
    return row;
}

static const char *speed_temporary(Arena *a, const char *kind) {
    const char *scratch = env("HERCULEX_SCRATCH", fmt(a, "%s/runtime/scratch", HOME));
    mkdirs(scratch);
    char *folder = fmt(a, "%s/speed-%s.XXXXXXXX", scratch, kind);
    need(mkdtemp(folder) != NULL, "cannot open benchmark scratch %s: %s", folder, strerror(errno));
    return folder;
}

static json_object *speed_bricks(Arena *a) {
    const char *folder = speed_temporary(a, "bricks");
    enum { SIDE = 128, COUNT = 512 };
    const i64 shape[] = {1024, 1024, 1024}, brick[] = {SIDE, SIDE, SIDE};
    Bricks b = {0};
    bricks_create(a, folder, shape, brick, NULL);
    bricks_open(a, &b, folder);
    Pcg64 rng = pcg64(0);
    u8 *data = push(a, b.size);
    int buffered = 0;
    u32 packed = 0;
    for (i64 i = 0; i < b.size; i++) {
        u16 value;
        do {
            if (!buffered) {
                packed = pcg64_next32(&rng);
                buffered = 4;
            }
            value = (u16)(packed & 255) * 255;
            packed >>= 8;
            buffered--;
        } while (!(value & 255));
        data[i] = value >> 8;
    }
    int order[COUNT];
    rng = pcg64(1);
    for (int i = 0; i < COUNT; i++) {
        order[i] = i;
    }
    for (int i = COUNT - 1; i > 0; i--) {
        u32 mask = i, j;
        for (int shift = 1; shift < 32; shift *= 2) {
            mask |= mask >> shift;
        }
        do {
            j = pcg64_next32(&rng) & mask;
        } while (j > (u32)i);
        int move = order[i];
        order[i] = order[j];
        order[j] = move;
    }
    double start = now();
    for (int i = 0; i < COUNT; i++) {
        bricks_put(&b, i, data);
    }
    bricks_save(&b);
    double wrote = now();
    bricks_map(&b);
    i64 checksum = 0;
    for (int i = 0; i < COUNT; i++) {
        checksum += b.data[b.offsets[order[i]]];
    }
    double read = now();
    munmap((void *)b.data, b.mapped);
    pthread_mutex_destroy(&b.lock);
    remove_tree(folder);
    json_object *row = json_object_new_object();
    json_object_object_add(row, "seconds", jnum(read - start));
    json_object_object_add(row, "write_s", jnum(round((wrote - start) * 1000) / 1000));
    json_object_object_add(row, "read_s", jnum(round((read - wrote) * 1000) / 1000));
    json_object_object_add(row, "sum", json_object_new_int64(checksum));
    return row;
}

static json_object *speed_level(Arena *a) {
    const char *folder =
        fmt(a, "%s/controls/sheets/PHerc1667-20231117161658-rows-0008-0020", LIBRARY);
    json_object *record = runner_record(folder);
    need(record != NULL, "no frozen benchmark control at %s", folder);
    const char *sample = jstr(record, "sample"), *scan = jstr(record, "volume");
    const char *path = fmt(a, "%s/unwrapping/level/%s/%s/sheet/curl.json", OUTPUTS, sample, scan);
    json_object *curl = json_read(path);
    need(curl != NULL, "no curl for benchmark control %s %s", sample, scan);
    int direction = ji64(curl, "curl");
    need(direction == 1 || direction == -1, "invalid curl for benchmark control %s %s", sample,
        scan);
    json_object_put(curl);
    int pipefd[2];
    need(pipe(pipefd) == 0, "cannot inspect the native level output");
    fflush(NULL);
    double started = now();
    pid_t pid = fork();
    need(pid >= 0, "cannot start native level benchmark");
    if (!pid) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        char *args[] = {"run_levels", "solve", (char *)folder, direction > 0 ? "1" : "-1"};
        exit(hx_levels_main(4, args));
    }
    close(pipefd[1]);
    FILE *out = fdopen(pipefd[0], "r");
    need(out != NULL, "cannot read native level stats");
    size_t cap = 0;
    char *line = NULL;
    json_object *stats = NULL;
    while (getline(&line, &cap, out) >= 0) {
        if (*line == '{') {
            json_object_put(stats);
            stats = json_tokener_parse(line);
        }
    }
    fclose(out);
    free(line);
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        need(errno == EINTR, "cannot wait for level benchmark");
    }
    need(WIFEXITED(status) && WEXITSTATUS(status) == 0 && stats != NULL,
        "native level benchmark failed");
    json_object *result = json_object_new_object();
    speed_elapsed(result, started);
    json_object_object_add(result, "nodes", json_object_new_int64(ji64(stats, "nodes")));
    json_object_put(stats);
    json_object_put(record);
    return result;
}

enum { SPEED_CONNECTIONS = 32, SPEED_BODY = 2 << 20, SPEED_REQUESTS = 400 };

static int speed_send(int fd, const void *body, size_t size) {
    size_t sent = 0;
    while (sent < size) {
        ssize_t n = send(fd, (const char *)body + sent, size - sent, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return 0;
        }
        sent += n;
    }
    return 1;
}

typedef struct {
    int socket;
    const u8 *payload;
    char url[80];
    _Atomic i64 moved;
} SpeedServer;

static void *speed_serve(void *context) {
    SpeedServer *s = context;
    static const char reply[] = "HTTP/1.1 200 OK\r\nContent-Length: 2097152\r\n\r\n";
    for (;;) {
        int fd = accept(s->socket, NULL, NULL);
        if (fd < 0 && errno == EINTR) {
            continue;
        }
        if (fd < 0) {
            return NULL;
        }
        char request[4096];
        size_t read = 0;
        while (read < sizeof(request) - 1) {
            ssize_t n = recv(fd, request + read, sizeof(request) - 1 - read, 0);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                break;
            }
            read += n;
            request[read] = 0;
            if (strstr(request, "\r\n\r\n")) {
                if (!speed_send(fd, reply, sizeof(reply) - 1) ||
                    !speed_send(fd, s->payload, SPEED_BODY)) {
                    break;
                }
                read = 0;
            }
        }
        close(fd);
    }
}

static void speed_deliver(void *context, Ask *ask, const u8 *data, size_t size) {
    SpeedServer *s = context;
    atomic_fetch_add(&s->moved, size);
}

static void speed_easy(void *context, i64 index) {
    SpeedServer *s = context;
    char url[112];
    snprintf(
        url, sizeof(url), "%s/0/%lld/%lld", s->url, (long long)index / 20, (long long)index % 20);
    CurlBody body = {0};
    CURL *c = curl_easy_init();
    need(c != NULL, "cannot start benchmark HTTP client");
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, body_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 15L);
    need(curl_easy_perform(c) == CURLE_OK, "benchmark HTTP client failed");
    atomic_fetch_add(&s->moved, body.size);
    curl_easy_cleanup(c);
    free(body.data);
}

static json_object *speed_fetch(Arena *a, int threaded) {
    SpeedServer server = {
        .socket = socket(AF_INET, SOCK_STREAM, 0), .payload = pushz(a, SPEED_BODY)};
    need(server.socket >= 0, "cannot open the benchmark server socket");
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    need(bind(server.socket, (struct sockaddr *)&address, sizeof(address)) == 0 &&
            listen(server.socket, SPEED_CONNECTIONS * 2) == 0,
        "cannot start local benchmark server");
    socklen_t n = sizeof(address);
    need(getsockname(server.socket, (struct sockaddr *)&address, &n) == 0,
        "cannot read benchmark server location");
    snprintf(server.url, sizeof(server.url), "http://127.0.0.1:%d/b", ntohs(address.sin_port));
    pthread_t pool[SPEED_CONNECTIONS];
    struct sigaction saved, ignored = {.sa_handler = SIG_IGN};
    sigaction(SIGPIPE, &ignored, &saved);
    for (int i = 0; i < SPEED_CONNECTIONS; i++) {
        need(pthread_create(&pool[i], NULL, speed_serve, &server) == 0,
            "cannot start a benchmark server worker");
    }
    Ask *asks = pushz(a, SPEED_REQUESTS * sizeof(*asks));
    for (int i = 0; i < SPEED_REQUESTS; i++) {
        asks[i].first = -1;
        snprintf(asks[i].path, sizeof(asks[i].path), "0/%d/%d", i / 20, i % 20);
    }
    need(curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK, "cannot initialize libcurl");
    double cpu = speed_cpu(), began = now();
    if (threaded) {
        parallel(SPEED_REQUESTS, SPEED_CONNECTIONS, speed_easy, &server);
    } else {
        need(!fetch_ranges(server.url, asks, SPEED_REQUESTS, speed_deliver, &server),
            "benchmark fetch incomplete");
    }
    double elapsed = now() - began, cpu_spent = speed_cpu() - cpu;
    shutdown(server.socket, SHUT_RDWR);
    close(server.socket);
    for (int i = 0; i < SPEED_CONNECTIONS; i++) {
        pthread_join(pool[i], NULL);
    }
    sigaction(SIGPIPE, &saved, NULL);
    need(server.moved == (i64)SPEED_REQUESTS * SPEED_BODY, "benchmark transfer size changed");
    json_object *row = json_object_new_object();
    json_object_object_add(row, "seconds", jnum(elapsed));
    json_object_object_add(row, "mb_s", jnum(round(server.moved / 1e5 / elapsed) / 10));
    json_object_object_add(row, "cpu_s", jnum(round(cpu_spent * 100) / 100));
    return row;
}

static json_object *speed_page(Arena *a) {
    const char *folder = speed_temporary(a, "page");
    const char *base = fmt(a, "%s/unwrapping/level/scroll/scan", folder);
    const char *faces = fmt(a, "%s/faces/level-0/unit", base);
    json_object *layout = json_object_new_object(), *pieces = json_object_new_array();
    json_object_object_add(layout, "pieces", pieces);
    Pcg64 rng = pcg64(7);
    for (int i = 0; i < 120; i++) {
        size_t saved = a->used;
        int h = draw(&rng, 1600) + 500, w = draw(&rng, 1600) + 500;
        u8 *pixels = push(a, (i64)h * w), *part = push(a, (i64)h * w);
        for (int y = 0; y < h; y++) {
            double rise = 40 * sin(y / 80.0 + i);
            for (int x = 0; x < w; x++) {
                double grey = 110 + rise + 27 * sin(x / 55.0) + (int)draw(&rng, 35) - 17;
                pixels[(i64)y * w + x] = fmin(255, fmax(1, nearbyint(grey)));
            }
        }
        write_png(a, fmt(a, "%s/piece-%04d.png", faces, i), pixels, w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                part[(i64)y * w + x] = pixels[(i64)y * w + (x + w - 3) % w];
            }
        }
        write_png(a, fmt(a, "%s/piece-%04d-layer0.png", faces, i), part, w, h);
        for (i64 p = 0; p < (i64)h * w; p++) {
            part[p] = pixels[p] > 124 ? 200 : 0;
        }
        write_png(a, fmt(a, "%s/piece-%04d-taught.png", faces, i), part, w, h);
        json_object *piece = json_object_new_object();
        const char *keys[] = {"piece", "area", "um", "stride", "group", "x_mm", "y_mm", "width_mm",
            "height_mm", "flip_rows", "flip_cols"};
        int group = i / 4;
        double values[] = {i, h * w, 0.01, 2, group, 0.5 + pcg64_double(&rng) * 479.5,
            0.5 + pcg64_double(&rng) * 59.5, w * 0.01, h * 0.01, 0, 0};
        for (int k = 0; k < 11; k++) {
            json_object_object_add(piece, keys[k], jnum(values[k]));
        }
        json_object_object_add(piece, "unit", json_object_new_string("unit"));
        json_object_array_add(pieces, piece);
        a->used = saved;
    }
    json_write(fmt(a, "%s/sheet/layout.json", base), layout);
    json_object_put(layout);
    int workers = fmin(32, board_value(env("HERCULEX_COMPOSE_PROCESSES", "4")));
    double began = now();
    fflush(NULL);
    pid_t child = fork();
    need(child >= 0, "cannot start native page benchmark");
    if (!child) {
        setenv("HERCULEX_OUTPUTS", folder, 1);
        setenv("HERCULEX_FACES", "level-0", 1);
        setenv("HERCULEX_INK_MODEL", "reader-v3", 1);
        char processes[16];
        snprintf(processes, sizeof(processes), "%d", workers);
        setenv("HERCULEX_COMPOSE_PROCESSES", processes, 1);
        roots_init();
        char *args[] = {
            "viewer.sheet_tiles", "scroll", "scan", "0.01", "page", "taught,layer0,sheet"};
        exit(hx_sheet_tiles_main(6, args));
    }
    int status;
    while (waitpid(child, &status, 0) < 0) {
        need(errno == EINTR, "cannot wait for page benchmark");
    }
    double elapsed = now() - began;
    need(WIFEXITED(status) && WEXITSTATUS(status) == 0, "native page benchmark failed");
    remove_tree(folder);
    json_object *row = json_object_new_object();
    json_object_object_add(row, "seconds", jnum(elapsed));
    json_object_object_add(row, "pieces", json_object_new_int(120));
    json_object_object_add(row, "processes", json_object_new_int(workers));
    return row;
}

static json_object *speed_context(void) {
    json_object *ctx = json_object_new_object();
    pid_t lister;
    FILE *git = spawn_read((char *[]){"git", "rev-parse", "--short", "HEAD", NULL}, 0, &lister);
    char head[80] = "";
    if (git) {
        if (fgets(head, sizeof(head), git)) {
            head[strcspn(head, "\r\n")] = 0;
        }
        spawn_close(git, lister);
    }
    json_object_object_add(ctx, "commit", json_object_new_string(head));
    int dirty = 0;
    git = spawn_read((char *[]){"git", "status", "--porcelain", "--", ".", NULL}, 0, &lister);
    if (git) {
        dirty = fgetc(git) != EOF;
        spawn_close(git, lister);
    }
    json_object_object_add(ctx, "dirty", json_object_new_boolean(dirty));
    double load = 0;
    getloadavg(&load, 1);
    json_object_object_add(ctx, "load", jnum(round(load * 10) / 10));
    json_object *beside = json_object_new_array();
    char path[PATH_MAX], sqlite[PATH_MAX];
    board_folder(path);
    snprintf(sqlite, sizeof(sqlite), "%s/board.sqlite", path);
    if (exists(sqlite)) {
        Board b = {0};
        int rc = sqlite3_open_v2(sqlite, &b.db, SQLITE_OPEN_READONLY, NULL);
        if (rc == SQLITE_OK) {
            sqlite3_stmt *q = board_query(&b,
                "SELECT name FROM jobs WHERE state IN "
                "('starting','running','stopping') AND name NOT LIKE 'qol-speed%%' ORDER BY added");
            while (sqlite3_step(q) == SQLITE_ROW) {
                json_object_array_add(
                    beside, json_object_new_string((const char *)sqlite3_column_text(q, 0)));
            }
            sqlite3_finalize(q);
        }
        sqlite3_close(b.db);
    }
    json_object_object_add(ctx, "beside", beside);
    return ctx;
}

static double speed_history(const char *path, const char *name, const char *fixture) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return INFINITY;
    }
    size_t cap = 0;
    char *line = NULL;
    double recent[5] = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};
    int seen = 0;
    while (getline(&line, &cap, f) >= 0) {
        json_object *row = json_tokener_parse(line);
        if (row && !strcmp(board_text(row, "name"), name) &&
            !strcmp(board_text(row, "fixture"), fixture)) {
            recent[seen++ % 5] = board_number(row, "seconds");
        }
        json_object_put(row);
    }
    free(line);
    fclose(f);
    double best = INFINITY;
    for (int i = 0; i < 5; i++) {
        best = fmin(best, recent[i]);
    }
    return best;
}

static void speed_record(Arena *a, const char *path, const char *name, const char *fixture,
    int repeats, double seconds, json_object *details, json_object *ctx) {
    size_t saved = a->used;
    json_object *row = json_object_new_object();
    json_object_object_add(row, "name", json_object_new_string(name));
    json_object_object_add(row, "seconds", jnum(round(seconds * 1000) / 1000));
    json_object_object_add(row, "repeats", json_object_new_int(repeats));
    json_object_object_add(row, "at", jnum(board_time()));
    if (*fixture) {
        json_object_object_add(row, "fixture", json_object_new_string(fixture));
    }
    {
        json_object_object_foreach(details, key, value) {
            json_object_object_add(row, key, json_object_get(value));
        }
    }
    {
        json_object_object_foreach(ctx, key, value) {
            json_object_object_add(row, key, json_object_get(value));
        }
    }
    const char *line = fmt(a, "%s\n", board_json(row));
    mkparent(path);
    int fd = open(path, O_APPEND | O_CREAT | O_WRONLY | O_CLOEXEC, 0644);
    need(fd >= 0 && !flock(fd, LOCK_EX), "cannot write benchmark record %s", path);
    write_all(fd, line, strlen(line), path);
    close(fd);
    json_object_put(row);
    a->used = saved;
}

typedef struct {
    const char *name, *fixture;
    int repeats;
} SpeedBench;

static const SpeedBench speed_benches[] = {{"solver", "", 3}, {"level", "native-level-solve", 1},
    {"bricks", "", 3}, {"page", "native-page-synthetic", 1}, {"fetch-curl", "native-loopback", 3},
    {"fetch-threaded", "native-loopback", 3}};

int speed_command(int argc, char **argv) {
    int record = 0, compare = 0, count = 0;
    int runs[argc > 6 ? argc : 6];
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("runner.speed [solver|level|bricks|page|fetch-curl|fetch-threaded ...] "
                 "[--record] [--compare]");
            return 0;
        }
        if (!strcmp(argv[i], "--record")) {
            record = 1;
            continue;
        }
        if (!strcmp(argv[i], "--compare")) {
            compare = 1;
            continue;
        }
        const char *name = argv[i];
        int index = 0;
        while (index < 6 && strcmp(name, speed_benches[index].name) != 0) {
            index++;
        }
        need(index < 6, "unknown benchmark %s", argv[i]);
        runs[count++] = index;
    }
    if (!count) {
        for (; count < 6; count++) {
            runs[count] = count;
        }
    }
    Arena a = arena(16 * GB);
    const char *path = fmt(&a, "%s/records/speed.jsonl", HOME);
    json_object *ctx = speed_context(), *slower = json_object_new_array();
    for (int i = 0; i < count; i++) {
        int index = runs[i];
        SpeedBench bench = speed_benches[index];
        json_object *details = NULL;
        double times[3];
        for (int k = 0; k < bench.repeats; k++) {
            size_t saved = a.used;
            json_object *row = index == 0 ? speed_solver(&a)
                : index == 1              ? speed_level(&a)
                : index == 2              ? speed_bricks(&a)
                : index == 3              ? speed_page(&a)
                                          : speed_fetch(&a, index == 5);
            times[k] = jf64(row, "seconds");
            if (k == bench.repeats / 2) {
                details = row;
                json_object_object_del(row, "seconds");
            } else {
                json_object_put(row);
            }
            a.used = saved;
        }
        if (bench.repeats == 3) {
            double low = fmin(times[0], fmin(times[1], times[2]));
            double high = fmax(times[0], fmax(times[1], times[2]));
            times[1] = times[0] + times[1] + times[2] - low - high;
        }
        double median = times[bench.repeats / 2],
               best = speed_history(path, bench.name, bench.fixture);
        printf("%-14s%8.2f s  %s", bench.name, median, board_json(details));
        if (isfinite(best)) {
            printf("  best of last five %.2f s", best);
            if (best > 0 && median > 1.15 * best) {
                printf("  SLOWER by %.0f%%", 100 * (median / best - 1));
                json_object_array_add(slower, json_object_new_string(bench.name));
            }
        }
        puts("");
        fflush(stdout);
        if (record) {
            speed_record(&a, path, bench.name, bench.fixture, bench.repeats, median, details, ctx);
        }
        json_object_put(details);
    }
    json_object *beside = jget(ctx, "beside");
    if (board_count(beside)) {
        printf("ran beside ");
        for (int i = 0; i < board_count(beside); i++) {
            printf("%s%s", i ? ", " : "", board_item(beside, i));
        }
        puts(": compare with care");
    }
    int failed = compare && board_count(slower);
    if (failed) {
        fputs("slower than recorded: ", stderr);
        for (int i = 0; i < board_count(slower); i++) {
            fprintf(stderr, "%s%s", i ? ", " : "", board_item(slower, i));
        }
        fputc('\n', stderr);
    }
    json_object_put(ctx);
    json_object_put(slower);
    munmap(a.base, a.size);
    return failed;
}
