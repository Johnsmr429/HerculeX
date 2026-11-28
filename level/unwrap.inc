#include <limits.h>

const char hx_usage_unwrap[] =
    "Run the level 1.1 engine from CT to published pages on the native lane.";

typedef struct {
    char *sample;
    char *scan;
} Scroll;

char *hx_chosen_scan(const char *sample) {
    int output[2];
    if (pipe(output))
        return NULL;
    pid_t pid = fork();
    if (pid == 0) {
        close(output[0]);
        dup2(output[1], STDOUT_FILENO);
        close(output[1]);
        execl("./run.sh", "./run.sh", "scans.catalog.scan_choice", "chosen", sample, NULL);
        _exit(127);
    }
    close(output[1]);
    FILE *stream = fdopen(output[0], "r");
    char line[1024] = {0};
    int read = stream && fgets(line, sizeof(line), stream) != NULL;
    if (stream)
        fclose(stream);
    int code = pid < 0 ? 1 : wait_for(pid);
    if (code || !read)
        return NULL;
    char held[512], scan[512];
    if (sscanf(line, "%511s %511s", held, scan) != 2 || strcmp(held, sample) != 0)
        return NULL;
    return strdup(scan);
}

static int hand_curl(Roots *roots, const char *sample, const char *scan) {
    char *path = NULL;
    if (asprintf(&path, "%s/%s/%s/curl.json", roots->route, sample, scan) < 0)
        return 0;
    json_object *document = json_object_from_file(path), *curl, *rule;
    free(path);
    int ok = document && json_object_object_get_ex(document, "curl", &curl) &&
        abs(json_object_get_int(curl)) == 1 && json_object_object_get_ex(document, "rule", &rule) &&
        !strcmp(json_object_get_string(rule), "set by hand");
    if (document)
        json_object_put(document);
    return ok;
}

static int axis_on_hand(const char *sample, const char *scan) {
    char *arguments[] = {
        "./run.sh", "level.level.umbilicus", "check", (char *)sample, (char *)scan, NULL};
    return run(arguments) == 0;
}

static int parse_scroll(Scroll *scroll, const char *word) {
    const char *split = strchr(word, ':');
    scroll->sample = split ? strndup(word, (size_t)(split - word)) : strdup(word);
    scroll->scan = split && split[1] ? strdup(split + 1) : NULL;
    return scroll->sample != NULL;
}

static void free_scrolls(Scroll *scrolls, int count) {
    for (int i = 0; i < count; i++) {
        free(scrolls[i].sample);
        free(scrolls[i].scan);
    }
    free(scrolls);
}

static int unwrap_ready(Roots *roots, Scroll *scroll) {
    if (!scroll->scan)
        scroll->scan = hx_chosen_scan(scroll->sample);
    int ok = 1;
    if (!scroll->scan || !hand_curl(roots, scroll->sample, scroll->scan)) {
        fprintf(stderr, "  ./run.sh level.level.run_levels curl %s %s 1   (or -1)\n",
            scroll->sample, scroll->scan ? scroll->scan : "SCAN");
        ok = 0;
    }
    if (scroll->scan && !axis_on_hand(scroll->sample, scroll->scan)) {
        fprintf(stderr,
            "  no umbilicus on hand: add the scan's line to records/umbilici, then ./run.sh "
            "level.level.umbilicus %s %s\n",
            scroll->sample, scroll->scan);
        ok = 0;
    }
    return ok;
}

static char *job_name(const char *scroll, const char *stage, const char *suffix) {
    char *name = NULL;
    if (asprintf(&name, "%s-%s%s", scroll, stage, suffix) < 0)
        return NULL;
    return name;
}

static int board_job(Scroll *scroll, const char *stage, const char *suffix) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    double ram_gb = (double)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE) / (1 << 30);
    double fit_gb = ram_gb / 2 > 5 ? ram_gb / 2 : 5;
    char cores[32], fit_memory[32], draw_cpus[64], draw_extra[64], compose[64];
    snprintf(cores, sizeof(cores), "%ld", online > 3 ? online - 2 : 1);
    snprintf(fit_memory, sizeof(fit_memory), "%.1f", fit_gb);
    snprintf(draw_cpus, sizeof(draw_cpus), "HERCULEX_LEVEL_DRAW_CPUS=%s", cores);
    snprintf(
        draw_extra, sizeof(draw_extra), "HERCULEX_LEVEL_DRAW_EXTRA=%.0f", (fit_gb - 5) * (1 << 30));
    snprintf(compose, sizeof(compose), "HERCULEX_COMPOSE_PROCESSES=%s", cores);
    char *name = job_name(scroll->sample, stage, suffix);
    char *after = NULL;
    const char *after_stage = !strcmp(stage, "pieces") || !strcmp(stage, "fetch") ? "sheets"
        : !strcmp(stage, "sheet")                                                 ? "pieces"
                                                                                  : NULL;
    if (after_stage)
        after = job_name(scroll->sample, after_stage, suffix);
    char model[512];
    snprintf(model, sizeof(model), "HERCULEX_INK_MODEL=%s", suffix[0] ? suffix + 1 : "reader-v3");
    char *arguments[48] = {0};
    size_t n = 0;
    arguments[n++] = "./run.sh";
    arguments[n++] = "board";
    arguments[n++] = "add";
    arguments[n++] = name;
    if (!strcmp(stage, "sheets")) {
        arguments[n++] = "--gpu";
        arguments[n++] = "--link";
        arguments[n++] = "--share";
        arguments[n++] = "--memory";
        arguments[n++] = "5";
        arguments[n++] = "--setting";
        arguments[n++] = "HERCULEX_LIST_CT_BRICKS=1";
    } else if (!strcmp(stage, "pieces")) {
        arguments[n++] = "--memory";
        arguments[n++] = fit_memory;
        arguments[n++] = "--cores";
        arguments[n++] = cores;
        arguments[n++] = "--restartable";
        arguments[n++] = "--setting";
        arguments[n++] = draw_cpus;
        arguments[n++] = "--setting";
        arguments[n++] = draw_extra;
    } else if (!strcmp(stage, "fetch")) {
        arguments[n++] = "--link";
        arguments[n++] = "--share";
        arguments[n++] = "--memory";
        arguments[n++] = "2";
        arguments[n++] = "--cores";
        arguments[n++] = "2";
        arguments[n++] = "--restartable";
        arguments[n++] = "--setting";
        arguments[n++] = "HERCULEX_PREFETCH_WHILE_FITTING=1";
    } else {
        arguments[n++] = "--gpu";
        arguments[n++] = "--engine";
        arguments[n++] = "--link";
        arguments[n++] = "--memory";
        arguments[n++] = "3.6";
        arguments[n++] = "--cores";
        arguments[n++] = cores;
        arguments[n++] = "--setting";
        arguments[n++] = "HERCULEX_KEEP_STACKS=0";
        arguments[n++] = "--setting";
        arguments[n++] = compose;
        if (suffix[0]) {
            arguments[n++] = "--setting";
            arguments[n++] = model;
        }
    }
    if (after) {
        arguments[n++] = "--after";
        arguments[n++] = after;
    }
    arguments[n++] = "--";
    arguments[n++] = "./run.sh";
    arguments[n++] = "level.pipeline";
    arguments[n++] = "run";
    arguments[n++] = (char *)stage;
    arguments[n++] = scroll->sample;
    arguments[n++] = "--scan";
    arguments[n++] = scroll->scan;
    arguments[n] = NULL;
    int code = name ? run(arguments) : 1;
    free(after);
    free(name);
    return code;
}

static int apple(Scroll *scrolls, int count) {
    for (int i = 0; i < count; i++) {
        char *ready[] = {"./run.sh", "level.level.run_levels", "ready", scrolls[i].sample,
            scrolls[i].scan, NULL};
        if (run(ready))
            return 1;
    }
    const char *model = getenv("HERCULEX_INK_MODEL");
    if (!model || !*model)
        model = "reader-v3";
    char *suffix = NULL;
    if (asprintf(&suffix, "%s%s", !strcmp(model, "reader-v3") ? "" : "-",
            !strcmp(model, "reader-v3") ? "" : model) < 0)
        return 1;
    const char *stages[] = {"sheets", "pieces", "fetch", "sheet"};
    int code = 0;
    for (int i = 0; i < count && !code; i++) {
        for (size_t k = 0; k < sizeof(stages) / sizeof(*stages) && !code; k++)
            code = board_job(&scrolls[i], stages[k], suffix);
    }
    free(suffix);
    if (!code)
        puts("Follow them with ./run.sh board status; each page reaches the start page when its "
             "sheet job ends.");
    return code;
}

enum { CU_DETECT, CU_READY, CU_FIT, CU_JOIN, CU_READ, CU_PUBLISH, CU_DONE };
enum { CU_WAIT, CU_ACTIVE, CU_OK, CU_FAILED };

typedef struct {
    int cpus, reserved, fits, reads, threads, samplers, gpus, draw_cpus;
    unsigned long long draw_extra;
    int cpu[1024];
    char **gpu;
} CudaPlace;

typedef struct {
    char *name;
    off_t size;
    int status;
} CudaRow;

typedef struct {
    Scroll *scroll;
    CudaRow *rows;
    size_t count, finished, failed;
    int stage, active, success;
    char *outcome;
} CudaScroll;

typedef struct {
    pid_t pid;
    int attempt, stage, lane, threads;
    CudaScroll *scroll;
    CudaRow *row;
    char *name, *log;
    time_t started;
    off_t offset;
} CudaTask;

static void cuda_place_limits(CudaPlace *p, int reads, unsigned long long bytes) {
    const unsigned long long fit_bytes = 8000000000ULL;
    p->reads = p->gpus * reads;
    long long wanted = ((long long)p->samplers + 2) * p->reads;
    p->reserved = wanted < p->cpus / 2 ? (int)wanted : p->cpus / 2;
    int fitting = p->cpus - p->reserved;
    p->threads = fitting < 12 ? fitting : 12;
    int by_cpu = fitting / 3;
    unsigned long long by_mem = bytes / fit_bytes;
    p->fits = by_mem < (unsigned)by_cpu ? (int)by_mem : by_cpu;
    if (!p->fits) {
        p->fits = 1;
    }
    unsigned long long share = bytes / p->fits;
    p->draw_extra = share > fit_bytes ? (share - fit_bytes) / 2 : 0;
    p->draw_cpus = fitting / p->fits;
}

#ifdef __linux__
#include <sched.h>

static unsigned long long cuda_limit(const char *path) {
    FILE *f = fopen(path, "r");
    unsigned long long n = 0;
    if (f) {
        if (fscanf(f, "%llu", &n) != 1) {
            n = 0;
        }
        fclose(f);
    }
    return n;
}

static int cuda_cpus(CudaPlace *p) {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) {
        return 0;
    }
    FILE *quota = fopen("/sys/fs/cgroup/cpu.max", "r");
    unsigned long long count = 0, period = 0;
    if (quota) {
        if (fscanf(quota, "%llu %llu", &count, &period) != 2) {
            count = 0;
        }
        fclose(quota);
    } else {
        count = cuda_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
        period = cuda_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        if (count == ULLONG_MAX) {
            count = 0;
        }
    }
    size_t cap = sizeof(p->cpu) / sizeof(*p->cpu);
    if (count && period) {
        unsigned long long limited = count / period + (count % period != 0);
        if (limited < cap) {
            cap = limited;
        }
    }
    for (int i = 0; i < CPU_SETSIZE && (size_t)p->cpus < cap; i++) {
        if (CPU_ISSET(i, &allowed)) {
            p->cpu[p->cpus++] = i;
        }
    }
    return p->cpus;
}

static unsigned long long cuda_memory(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    unsigned long long bytes = 0;
    while (f && fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %llu", &bytes) == 1) {
            break;
        }
    }
    if (f) {
        fclose(f);
    }
    bytes *= 1024;
    const char *limits[] = {
        "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    const char *usage[] = {
        "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"};
    for (int i = 0; i < 2; i++) {
        unsigned long long cap = cuda_limit(limits[i]), used = cuda_limit(usage[i]);
        if (cap && (cap <= used || cap - used < bytes)) {
            bytes = cap > used ? cap - used : 0;
        }
    }
    return bytes;
}
#endif

static int cuda_devices(CudaPlace *p) {
    const char *visible = getenv("CUDA_VISIBLE_DEVICES");
    char *copy = visible ? strdup(visible) : NULL;
    if (visible && (!copy || !*copy || !strcmp(copy, "-1"))) {
        free(copy);
        return 0;
    }
    int listed = 0;
    if (!visible) {
        int output[2];
        if (!pipe(output)) {
            pid_t pid = fork();
            if (!pid) {
                close(output[0]);
                if (dup2(output[1], STDOUT_FILENO) < 0) {
                    _exit(127);
                }
                close(output[1]);
                execlp("nvidia-smi", "nvidia-smi", "-L", (char *)NULL);
                _exit(127);
            }
            close(output[1]);
            FILE *gpu = fdopen(output[0], "r");
            char line[1024];
            while (gpu && fgets(line, sizeof(line), gpu)) {
                listed += strncmp(line, "GPU ", 4) == 0;
            }
            if (gpu) {
                fclose(gpu);
            } else {
                close(output[0]);
            }
            if (pid > 0) {
                wait_for(pid);
            }
        }
    }
    if (copy) {
        listed = 1;
        for (const char *at = copy; *at; at++) {
            listed += *at == ',';
        }
    }
    p->gpu = calloc(listed ? (size_t)listed : 1, sizeof(*p->gpu));
    if (!p->gpu) {
        free(copy);
        return 0;
    }
    if (copy) {
        char *save;
        for (char *s = strtok_r(copy, ",", &save); s; s = strtok_r(NULL, ",", &save)) {
            if (strcmp(s, "-1") == 0) {
                break;
            }
            p->gpu[p->gpus++] = strdup(s);
        }
    } else {
        for (int i = 0; i < (listed ? listed : 1); i++) {
            char *s = NULL;
            if (asprintf(&s, "%d", i) < 0) {
                break;
            }
            p->gpu[p->gpus++] = s;
        }
    }
    free(copy);
    for (int i = 0; i < p->gpus; i++) {
        if (!p->gpu[i]) {
            return 0;
        }
    }
    return p->gpus;
}

static void cuda_pin(const CudaPlace *p, int fitting) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    int first = fitting ? p->reserved : 0;
    int last = fitting || !p->reserved ? p->cpus : p->reserved;
    for (int i = first; i < last; i++) {
        CPU_SET(p->cpu[i], &set);
    }
    if (last > first && sched_setaffinity(0, sizeof(set), &set)) {
        perror("cannot set unwrap CPU affinity");
        _exit(1);
    }
#else
    (void)p;
    (void)fitting;
#endif
}

static int cuda_row_order(const void *a, const void *b) {
    const CudaRow *x = a, *y = b;
    if (x->size != y->size) {
        return x->size > y->size ? -1 : 1;
    }
    return strcmp(x->name, y->name);
}

static size_t cuda_units(CudaScroll *s, const char *nodes) {
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/%s/%s/nodes/rows-*/record.json", nodes, s->scroll->sample,
            s->scroll->scan) < 0) {
        return 0;
    }
    glob_t paths = {0};
    if (glob(pattern, 0, NULL, &paths) == 0) {
        s->rows = calloc(paths.gl_pathc, sizeof(*s->rows));
    }
    free(pattern);
    for (size_t i = 0; s->rows && i < paths.gl_pathc; i++) {
        char *path = paths.gl_pathv[i], *record = strrchr(path, '/');
        *record = 0;
        char *name = strdup(strrchr(path, '/') + 1), *store = path_join(path, "nodes.zst");
        struct stat info;
        if (!store || stat(store, &info)) {
            free(store);
            store = path_join(path, "nodes.bin");
            if (!store || stat(store, &info)) {
                info.st_size = 0;
            }
        }
        off_t size = info.st_size;
        free(store);
        *record = '/';
        if (name) {
            s->rows[s->count++] = (CudaRow){.name = name, .size = size};
        }
    }
    globfree(&paths);
    if (s->count) {
        qsort(s->rows, s->count, sizeof(*s->rows), cuda_row_order);
    }
    return s->count;
}

static void cuda_settle(CudaScroll *s, const char *outcome) {
    s->stage = CU_DONE;
    if (!s->outcome) {
        s->outcome = strdup(outcome);
    }
    printf("settled %s %s\n", s->scroll->sample, s->outcome ? s->outcome : outcome);
    fflush(stdout);
}

static int cuda_fitting(int stage) {
    return stage == CU_READY || stage == CU_FIT || stage == CU_JOIN;
}

static pid_t cuda_launch(CudaTask *t, const CudaPlace *p) {
    int fd = open(t->log, O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) {
        perror(t->log);
        return -1;
    }
    t->offset = lseek(fd, 0, SEEK_END);
    t->started = time(NULL);
    t->attempt++;
    printf("start %s try %d\n", t->name, t->attempt);
    fflush(stdout);
    pid_t pid = fork();
    if (!pid) {
        if (setpgid(0, 0) || dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0) {
            _exit(1);
        }
        close(fd);
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        int stage = t->stage, fitting = cuda_fitting(stage);
        cuda_pin(p, fitting);
        setenv("HERCULEX_LIST_CT_BRICKS", "1", 0);
        setenv("HERCULEX_KEEP_STACKS", "0", 0);
        int compose = p->cpus < 1 ? 1 : p->cpus;
        char compose_text[32];
        snprintf(compose_text, sizeof(compose_text), "%d", compose);
        setenv("HERCULEX_COMPOSE_PROCESSES", compose_text, 0);
        setenv("HERCULEX_FETCH_CONNECTIONS", "32", 0);
        char value[32];
        char *sample = t->scroll->scroll->sample, *scan = t->scroll->scroll->scan;
        if (fitting) {
            int threads = t->threads > 0 ? t->threads : p->threads;
            snprintf(value, sizeof(value), "%d", threads);
            setenv("OMP_NUM_THREADS", value, 1);
            if (stage == CU_FIT) {
                snprintf(value, sizeof(value), "%llu", t->attempt == 1 ? p->draw_extra : 0);
                setenv("HERCULEX_LEVEL_DRAW_EXTRA", value, 1);
                snprintf(value, sizeof(value), "%d", p->draw_cpus);
                setenv("HERCULEX_LEVEL_DRAW_CPUS", value, 1);
            }
            const char *verb = stage == CU_READY ? "ready" : stage == CU_JOIN ? "finish" : "unit";
            char *args[] = {"./run.sh", "level.level.run_levels", (char *)verb, sample,
                stage == CU_FIT ? t->row->name : scan, stage == CU_FIT ? scan : NULL, NULL};
            execv(args[0], args);
        } else if (stage == CU_PUBLISH) {
            const char *site = getenv("HERCULEX_SITE");
            execl("./run.sh", "./run.sh", "runner.review_page", sample, scan, "--url",
                site && *site ? site : "http://127.0.0.1:8765", (char *)NULL);
        } else {
            if (stage == CU_READ) {
                setenv("CUDA_VISIBLE_DEVICES", p->gpu[t->lane % p->gpus], 1);
                setenv("HERCULEX_FOR_PUBLICATION", "1", 1);
                snprintf(value, sizeof(value), "%d", p->samplers);
                setenv("HERCULEX_SAMPLERS", value, 1);
            }
            execl("./run.sh", "./run.sh", "level.pipeline", "run",
                stage == CU_DETECT ? "sheets" : "ink", sample, "--scan", scan, (char *)NULL);
        }
        perror("./run.sh");
        _exit(127);
    }
    close(fd);
    if (pid > 0) {
        setpgid(pid, pid);
    }
    return pid;
}

static void cuda_publication(CudaTask *t) {
    FILE *f = fopen(t->log, "r");
    char *line = NULL;
    size_t n = 0;
    if (f && t->offset >= 0 && !fseeko(f, t->offset, SEEK_SET)) {
        while (getline(&line, &n, f) >= 0) {
            const char *start = strstr(line, "/sheets/sheet-");
            const char *suffix = "/index.html: ", *end = start ? strstr(start, suffix) : NULL;
            if (end) {
                const char *value = end + strlen(suffix);
                char *stop;
                long pieces = strtol(value, &stop, 10);
                if (stop == value || pieces < INT_MIN || pieces > INT_MAX) {
                    continue;
                }
                start += strlen("/sheets/");
                char *summary = NULL;
                if (asprintf(&summary, "published %.*s (%d pieces)", (int)(end - start), start,
                        (int)pieces) >= 0) {
                    free(t->scroll->outcome);
                    t->scroll->outcome = summary;
                }
            }
        }
    }
    if (f) {
        fclose(f);
    }
    free(line);
}

static void cuda_completed(CudaTask *t, int code, const char *nodes) {
    CudaScroll *s = t->scroll;
    s->active = 0;
    if (t->stage == CU_FIT) {
        t->row->status = code ? CU_FAILED : CU_OK;
        s->failed += code != 0;
        if (++s->finished < s->count) {
            return;
        }
        if (s->failed) {
            size_t size = 0;
            char *failed = NULL;
            FILE *out = open_memstream(&failed, &size);
            if (out) {
                fputs("level fits failed:", out);
                const char *separator = " ";
                for (size_t i = 0; i < s->count; i++) {
                    if (s->rows[i].status == CU_FAILED) {
                        fprintf(out, "%s%s", separator, s->rows[i].name);
                        separator = ", ";
                    }
                }
                fclose(out);
            }
            cuda_settle(s, failed ? failed : "level fits failed");
            free(failed);
        } else {
            s->stage = CU_JOIN;
        }
    } else if (code) {
        const char *failure[] = {"detection failed", "no curl set or no umbilicus on hand", NULL,
            "joins failed", "ink read failed", "publishing failed"};
        cuda_settle(s, failure[t->stage]);
    } else if (t->stage == CU_DETECT && !cuda_units(s, nodes)) {
        cuda_settle(s, "no sheet nodes");
    } else if (t->stage == CU_PUBLISH) {
        cuda_publication(t);
        s->success = 1;
        cuda_settle(s, "published");
    } else {
        s->stage++;
    }
}

static void cuda_start(CudaTask *t, int lane, int threads, CudaScroll *s, CudaRow *row,
    const CudaPlace *p, const char *logs, const char *suffix, const char *nodes) {
    const char *names[] = {"sheets", "ready", NULL, "pieces", "ink", "sheet"};
    const char *stage = row ? row->name : names[s->stage];
    *t = (CudaTask){.scroll = s, .row = row, .stage = s->stage, .lane = lane, .threads = threads};
    t->name = job_name(s->scroll->sample, stage, s->stage >= CU_READ ? suffix : "");
    if (!t->name || asprintf(&t->log, "%s/%s.log", logs, t->name) < 0) {
        free(t->name);
        free(t->log);
        cuda_completed(t, 1, nodes);
        *t = (CudaTask){0};
        return;
    }
    if (row) {
        row->status = CU_ACTIVE;
    } else {
        s->active = 1;
    }
    t->pid = cuda_launch(t, p);
    if (t->pid < 0) {
        cuda_completed(t, 1, nodes);
        free(t->name);
        free(t->log);
        *t = (CudaTask){0};
    }
}

static int cuda_schedule(
    Scroll *scrolls, int count, const char *nodes, const char *logs, const CudaPlace *p) {
    CudaScroll *states = calloc((size_t)count, sizeof(*states));
    int read_slots = p->reads < count ? p->reads : count;
    int slots = 1 + p->fits + read_slots + 2;
    CudaTask *tasks = calloc((size_t)slots, sizeof(*tasks));
    const char *model = getenv("HERCULEX_INK_MODEL");
    char *suffix =
        model && *model && strcmp(model, "reader-v3") != 0 ? job_name("", model, "") : NULL;
    if (!states || !tasks || !make_folders(logs)) {
        free(states);
        free(tasks);
        free(suffix);
        return 1;
    }
    for (int i = 0; i < count; i++) {
        states[i].scroll = scrolls + i;
        if (!scrolls[i].scan) {
            cuda_settle(&states[i], "no curl set or no umbilicus on hand");
        }
    }
    printf("unwrap %d scrolls on %d GPUs, %d reads, %d level fits of %d threads\n", count, p->gpus,
        p->reads, p->fits, p->threads);
    struct sigaction action = {.sa_handler = caught}, old_int, old_term;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, &old_term);
    sigaction(SIGINT, &action, &old_int);
    while (!interrupted) {
        for (int k = 0; k < slots; k++) {
            if (tasks[k].pid > 0) {
                continue;
            }
            int reads = k - 1 - p->fits;
            for (int i = 0; i < count; i++) {
                CudaScroll *s = states + i;
                CudaRow *row = NULL;
                if (s->stage == CU_FIT && k && reads < 0) {
                    for (size_t j = 0; j < s->count; j++) {
                        if (s->rows[j].status == CU_WAIT) {
                            row = s->rows + j;
                            break;
                        }
                    }
                }
                if (row ||
                    (!s->active &&
                        ((!k && s->stage == CU_DETECT) ||
                            (k && reads < 0 && (s->stage == CU_READY || s->stage == CU_JOIN)) ||
                            (reads >= 0 && reads < read_slots && s->stage == CU_READ) ||
                            (reads >= read_slots && s->stage == CU_PUBLISH)))) {
                    int fit_threads = p->threads;
                    if (cuda_fitting(s->stage)) {
                        int demand = 0;
                        for (int m = 0; m < slots; m++) {
                            demand += tasks[m].pid > 0 && cuda_fitting(tasks[m].stage);
                        }
                        for (int m = 0; m < count; m++) {
                            if (states[m].stage == CU_FIT) {
                                for (size_t j = 0; j < states[m].count; j++) {
                                    demand += states[m].rows[j].status == CU_WAIT;
                                }
                            } else if (!states[m].active &&
                                (states[m].stage == CU_READY || states[m].stage == CU_JOIN)) {
                                demand++;
                            }
                        }
                        int fitting_cpus = p->cpus - p->reserved;
                        if (fitting_cpus > p->threads && demand > 0) {
                            int share = fitting_cpus / demand;
                            if (share > fit_threads) {
                                fit_threads = share > fitting_cpus ? fitting_cpus : share;
                            }
                        }
                    }
                    cuda_start(tasks + k, reads, fit_threads, s, row, p, logs, suffix ? suffix : "",
                        nodes);
                    break;
                }
            }
        }
        int active = 0, pending = 0;
        for (int k = 0; k < slots; k++) {
            active += tasks[k].pid > 0;
        }
        for (int i = 0; i < count; i++) {
            pending += states[i].stage != CU_DONE;
        }
        if (!active) {
            if (pending) {
                continue;
            }
            break;
        }
        int status;
        pid_t pid = waitpid(-1, &status, 0);
        if (pid < 0 && errno == EINTR) {
            continue;
        }
        if (pid < 0) {
            break;
        }
        for (int k = 0; k < slots; k++) {
            CudaTask *t = tasks + k;
            if (t->pid != pid) {
                continue;
            }
            int code = WIFEXITED(status) ? WEXITSTATUS(status)
                : WIFSIGNALED(status)    ? 128 + WTERMSIG(status)
                                         : 1;
            if (code) {
                printf("end %s %.0f s code %d\n", t->name, difftime(time(NULL), t->started), code);
            } else {
                printf("end %s %.0f s ok\n", t->name, difftime(time(NULL), t->started));
            }
            t->pid = 0;
            if (code && t->attempt < 2 && !interrupted) {
                t->pid = cuda_launch(t, p);
                if (t->pid > 0) {
                    break;
                }
            }
            cuda_completed(t, code, nodes);
            free(t->name);
            free(t->log);
            *t = (CudaTask){0};
            break;
        }
    }
    for (int k = 0; k < slots; k++) {
        if (tasks[k].pid > 0) {
            kill(-tasks[k].pid, SIGTERM);
        }
    }
    for (int k = 0; k < slots; k++) {
        if (tasks[k].pid > 0) {
            wait_for(tasks[k].pid);
        }
        free(tasks[k].name);
        free(tasks[k].log);
    }
    sigaction(SIGTERM, &old_term, NULL);
    sigaction(SIGINT, &old_int, NULL);
    int code = 0;
    for (int i = 0; i < count; i++) {
        CudaScroll *s = states + i;
        printf("%s %s\n", s->scroll->sample, s->outcome ? s->outcome : "unfinished");
        code |= !s->success;
        free(s->outcome);
        for (size_t j = 0; j < s->count; j++) {
            free(s->rows[j].name);
        }
        free(s->rows);
    }
    free(states);
    free(tasks);
    free(suffix);
    return interrupted ? 128 + SIGTERM : code;
}

static int cuda(Roots *roots, Scroll *scrolls, int count, int reads) {
#ifndef __linux__
    (void)roots;
    (void)scrolls;
    (void)count;
    (void)reads;
    return 1;
#else
    CudaPlace p = {.samplers = 6};
    const char *setting = getenv("HERCULEX_SAMPLERS");
    if (setting) {
        char *end;
        long value = strtol(setting, &end, 10);
        if (!*end && value >= 1 && value <= INT_MAX) {
            p.samplers = (int)value;
        }
    }
    int code = 1;
    if (!cuda_cpus(&p) || !cuda_devices(&p) || p.gpus > INT_MAX / reads) {
        fputs("HerculeX: cannot choose CUDA unwrap CPU cores and visible GPUs\n", stderr);
    } else {
        cuda_place_limits(&p, reads, cuda_memory());
        const char *root = getenv("HERCULEX_HOME");
        char *logs = path_join(root ? root : ".", "runtime/logs/unwrap");
        char *nodes = path_join(roots->outputs, "unwrapping/nodes");
        if (logs && nodes) {
            code = cuda_schedule(scrolls, count, nodes, logs, &p);
        }
        free(logs);
        free(nodes);
    }
    for (int i = 0; i < p.gpus; i++) {
        free(p.gpu[i]);
    }
    free(p.gpu);
    return code;
#endif
}

static void unwrap_usage(FILE *out) {
    fputs("Usage: ./run.sh level SCROLL[:SCAN] ... [--reads-per-gpu N]\n", out);
}

int hx_unwrap_main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        unwrap_usage(argc < 2 ? stderr : stdout);
        return argc < 2;
    }
    int reads = 1, count = 0;
    Scroll *scrolls = calloc((size_t)argc, sizeof(*scrolls));
    for (int i = 1; scrolls && i < argc; i++) {
        if (!strcmp(argv[i], "--reads-per-gpu") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            reads = *end || value < 1 || value > 1000 ? 0 : (int)value;
        } else if (!parse_scroll(&scrolls[count++], argv[i])) {
            free_scrolls(scrolls, count);
            return 1;
        }
    }
    if (!scrolls || !count || reads < 1) {
        free_scrolls(scrolls, count);
        unwrap_usage(stderr);
        return 2;
    }
    Roots roots;
    if (!route_open(&roots)) {
        free_scrolls(scrolls, count);
        return 1;
    }
    struct utsname system;
    int is_apple = uname(&system) == 0 && !strcmp(system.sysname, "Darwin");
    int all_ready = 1;
    for (int i = 0; i < count; i++) {
        if (!unwrap_ready(&roots, &scrolls[i])) {
            all_ready = 0;
            if (!is_apple) {
                free(scrolls[i].scan);
                scrolls[i].scan = NULL;
            }
        }
    }
    if (is_apple && !all_ready) {
        fputs("Not ready to unwrap:\n", stderr);
        release_roots(&roots);
        free_scrolls(scrolls, count);
        return 2;
    }
    const char *lane = is_apple ? "apple" : "cuda";
    const char *asked = getenv("HERCULEX_LANE");
    if (asked && *asked && strcmp(asked, lane) != 0) {
        fprintf(stderr, "HERCULEX_LANE=%s, but this machine runs the %s lane\n", asked, lane);
        release_roots(&roots);
        free_scrolls(scrolls, count);
        return 1;
    }
    printf("%s lane: ", lane);
    for (int i = 0; i < count; i++)
        printf("%s%s", i ? ", " : "", scrolls[i].sample);
    putchar('\n');
    fflush(stdout);
    int code = is_apple ? apple(scrolls, count) : cuda(&roots, scrolls, count, reads);
    release_roots(&roots);
    free_scrolls(scrolls, count);
    return code;
}
