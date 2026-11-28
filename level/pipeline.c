#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <json-c/json.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "../storage/route_record.h"

#include "../storage/data_roots.c"

const char hx_usage[] =
    "The native route from a scan to the website, stage by stage or all at once.";

typedef struct {
    char *data;
    char *library;
    char *outputs;
    char *nodes;
    char *route;
} Roots;

typedef struct {
    const char *stage;
    pid_t pid;
} Worker;

enum { STAGE_ALL, STAGE_SHEETS, STAGE_PIECES, STAGE_FETCH, STAGE_LAYOUT, STAGE_INK, STAGE_SHEET };

static const struct {
    const char *name, *module, *upstream;
    int band;
} pipeline_stages[] = {
    [STAGE_ALL] = {.name = "all", .band = 1},
    [STAGE_SHEETS] = {.name = "sheets", .module = "level.detection.detect_nodes", .band = 1},
    [STAGE_PIECES] = {.name = "pieces",
        .module = "level.level.run_levels",
        .upstream = "sheets",
        .band = 1},
    [STAGE_FETCH] = {.name = "fetch", .module = "fetch", .upstream = "pieces"},
    [STAGE_LAYOUT] = {.name = "layout"},
    [STAGE_INK] = {.name = "ink"},
    [STAGE_SHEET] = {.name = "sheet", .upstream = "pieces"},
};

static volatile sig_atomic_t interrupted;

static void release_roots(Roots *roots) {
    free(roots->data);
    free(roots->library);
    free(roots->outputs);
    free(roots->nodes);
    free(roots->route);
    memset(roots, 0, sizeof(*roots));
}

static int route_open(Roots *roots) {
    *roots = (Roots){0};
    roots->outputs = local_root(outputs_root);
    roots->route = roots->outputs ? path_join(roots->outputs, "unwrapping/level") : NULL;
    if (roots->route) {
        return 1;
    }
    release_roots(roots);
    return 0;
}

static int roots_open(Roots *roots) {
    if (!route_open(roots)) {
        return 0;
    }
    roots->data = data_root();
    roots->library = local_root(library_root);
    roots->nodes = path_join(roots->outputs, "unwrapping/nodes");
    if (roots->data && roots->library && roots->nodes) {
        return 1;
    }
    release_roots(roots);
    errno = ENOMEM;
    return 0;
}

static int beneath(const char *path, const char *parent) {
    size_t n = strlen(parent);
    return strncmp(path, parent, n) == 0 && (path[n] == 0 || path[n] == '/');
}

static int make_folders(const char *path) {
    char *copy = strdup(path);
    if (!copy)
        return 0;
    for (char *p = copy + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = 0;
        int ok = mkdir(copy, 0777) == 0 || errno == EEXIST;
        *p = '/';
        if (!ok) {
            free(copy);
            return 0;
        }
    }
    int ok = mkdir(copy, 0777) == 0 || errno == EEXIST;
    free(copy);
    return ok;
}

static int roots_ready(Roots *roots) {
    const char *folders[] = {roots->library, roots->outputs};
    for (size_t i = 0; i < sizeof(folders) / sizeof(*folders); i++) {
        struct stat status;
        if (stat(folders[i], &status) == 0 && S_ISDIR(status.st_mode))
            continue;
        if (beneath(folders[i], roots->data) && make_folders(folders[i]))
            continue;
        fprintf(stderr,
            "HerculeX data folder is unavailable: %s (mounted? see storage/data_roots.c)\n",
            folders[i]);
        return 0;
    }
    return make_folders(roots->route);
}

static int wait_for(pid_t pid) {
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR)
            return 1;
    }
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
}

static pid_t start(char *const arguments[]) {
    pid_t pid = fork();
    if (pid == 0) {
        execv(arguments[0], arguments);
        fprintf(stderr, "HerculeX: cannot run %s: %s\n", arguments[0], strerror(errno));
        _exit(127);
    }
    return pid;
}

static int run(char *const arguments[]) {
    pid_t pid = start(arguments);
    return pid < 0 ? 1 : wait_for(pid);
}

#ifdef HX_UNITY
double runner_event_time(void);
void runner_event(const char *kind, double start, double end, json_object *fields);
#endif

static int run_stage_span(char *const arguments[], const char *stage, const char *scan) {
#ifdef HX_UNITY
    const char *events = getenv("HERCULEX_EVENTS");
    if (events && *events) {
        double start = runner_event_time();
        int code = run(arguments);
        json_object *fields = json_object_new_object();
        if (scan) {
            json_object_object_add(fields, "scan", json_object_new_string(scan));
        }
        runner_event(stage, start, runner_event_time(), fields);
        json_object_put(fields);
        return code;
    }
#endif
    return run(arguments);
}

static int selected(const char *sample, char **samples, int count) {
    if (!count)
        return 1;
    for (int i = 0; i < count; i++) {
        if (strcmp(sample, samples[i]) == 0)
            return 1;
    }
    return 0;
}

static int regular(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static int directory(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISDIR(status.st_mode);
}

#include "../runner/sheet_stage.inc"

static int scan_has_nodes(Roots *roots, const char *sample, const char *scan, const char *pattern) {
    char *match = NULL;
    if (asprintf(&match, "%s/%s/%s/nodes/%s/record.json", roots->nodes, sample, scan ? scan : "*",
            pattern) < 0)
        return 0;
    glob_t paths = {0};
    int found = glob(match, 0, NULL, &paths) == 0 && paths.gl_pathc > 0;
    globfree(&paths);
    free(match);
    return found;
}

static int require_nodes(Roots *roots, char **samples, int count, const char *scan, int band) {
    const char *pattern = band ? "band-*" : "*";
    if (!count)
        return scan_has_nodes(roots, "*", scan, pattern);
    for (int i = 0; i < count; i++) {
        if (!scan_has_nodes(roots, samples[i], scan, pattern)) {
            fprintf(stderr,
                "HerculeX: no finished sheet-node units for %s. Run the sheets stage first.\n",
                samples[i]);
            return 0;
        }
    }
    return 1;
}

static int stage_command(Roots *roots, int stage, char **samples, int count, int band,
    const char *scan, const char *since) {
    const char *name = pipeline_stages[stage].name, *module = pipeline_stages[stage].module;
    if (!module)
        return sheet_run(roots, name, samples, count, scan);
    size_t cap = (size_t)count + 12;
    char **arguments = calloc(cap, sizeof(*arguments));
    if (!arguments)
        return 1;
    size_t n = 0;
    arguments[n++] = "./run.sh";
    arguments[n++] = (char *)module;
    arguments[n++] = "run";
    for (int i = 0; i < count; i++)
        arguments[n++] = samples[i];
    if (band)
        arguments[n++] = "--band";
    if (scan) {
        arguments[n++] = "--scan";
        arguments[n++] = (char *)scan;
    }
    if (since) {
        arguments[n++] = "--since";
        arguments[n++] = (char *)since;
    }
    arguments[n] = NULL;
    int code = run_stage_span(arguments, name, scan);
    free(arguments);
    return code;
}

static char *marker(Roots *roots, const char *stage) {
    char *path = NULL;
    if (asprintf(&path, "%s/%s.finished", roots->route, stage) < 0)
        return NULL;
    return path;
}

static int write_marker(Roots *roots, const char *stage) {
    char *path = marker(roots, stage);
    FILE *file = path ? fopen(path, "w") : NULL;
    int ok = file && fputs("finished\n", file) >= 0;
    if (file && fclose(file))
        ok = 0;
    free(path);
    return ok;
}

static int marker_exists(Roots *roots, const char *stage) {
    char *path = marker(roots, stage);
    int exists = path && regular(path);
    free(path);
    return exists;
}

static void remove_marker(Roots *roots, const char *stage) {
    char *path = marker(roots, stage);
    if (path)
        unlink(path);
    free(path);
}

static int follow(Roots *roots, int stage, char **samples, int count, int band, const char *scan) {
    const char *name = pipeline_stages[stage].name, *upstream = pipeline_stages[stage].upstream;
    int waits = stage == STAGE_SHEET || (stage == STAGE_PIECES && !band);
    time_t began = time(NULL);
    char since[32];
    snprintf(since, sizeof(since), "%lld", (long long)began);
    remove_marker(roots, name);
    for (;;) {
        int upstream_done = !upstream || marker_exists(roots, upstream);
        if ((upstream_done || !waits) &&
            stage_command(roots, stage, samples, count, band, scan, since))
            return 1;
        if (upstream_done)
            break;
        for (int i = 0; i < 15 && !interrupted; i++) {
            if (marker_exists(roots, upstream))
                break;
            sleep(1);
        }
        if (interrupted)
            return 128 + SIGTERM;
    }
    if (!write_marker(roots, name)) {
        fprintf(stderr, "HerculeX: could not write the %s marker: %s\n", name, strerror(errno));
        return 1;
    }
    return 0;
}

static void stop_workers(Worker *workers, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (workers[i].pid > 0) {
            kill(-workers[i].pid, SIGTERM);
        }
    }
    for (size_t i = 0; i < count; i++) {
        if (workers[i].pid > 0)
            wait_for(workers[i].pid);
    }
}

static int run_all(Roots *roots, char **samples, int count, int band, const char *scan) {
    const int stages[] = {STAGE_SHEETS, STAGE_PIECES, STAGE_SHEET, STAGE_FETCH};
    Worker workers[sizeof(stages) / sizeof(*stages)] = {0};
    for (size_t i = 0; i < sizeof(stages) / sizeof(*stages); i++)
        remove_marker(roots, pipeline_stages[stages[i]].name);
    fflush(NULL);
    for (size_t i = 0; i < sizeof(stages) / sizeof(*stages); i++) {
        pid_t pid = fork();
        if (pid < 0) {
            stop_workers(workers, i);
            return 1;
        }
        if (!pid) {
            if (setpgid(0, 0)) {
                _exit(1);
            }
            signal(SIGTERM, SIG_DFL);
            signal(SIGINT, SIG_DFL);
            int code = follow(roots, stages[i], samples, count, band, scan);
            fflush(NULL);
            _exit(code);
        }
        setpgid(pid, pid);
        workers[i] = (Worker){.stage = pipeline_stages[stages[i]].name, .pid = pid};
    }
    size_t left = sizeof(workers) / sizeof(*workers);
    while (left && !interrupted) {
        int status;
        pid_t pid = waitpid(-1, &status, 0);
        if (pid < 0) {
            if (errno == EINTR)
                continue;
            stop_workers(workers, sizeof(workers) / sizeof(*workers));
            return 1;
        }
        for (size_t i = 0; i < sizeof(workers) / sizeof(*workers); i++) {
            if (workers[i].pid != pid)
                continue;
            workers[i].pid = 0;
            left--;
            int code = WIFEXITED(status) ? WEXITSTATUS(status)
                : WIFSIGNALED(status)    ? 128 + WTERMSIG(status)
                                         : 1;
            if (code) {
                fprintf(stderr, "%s failed (%d). Rerun the same command to resume.\n",
                    workers[i].stage, code);
                stop_workers(workers, sizeof(workers) / sizeof(*workers));
                return code;
            }
            break;
        }
    }
    if (interrupted) {
        stop_workers(workers, sizeof(workers) / sizeof(*workers));
        return 128 + SIGTERM;
    }
    return 0;
}

static void stage_totals(
    const char *base, const char *stage, const char *pattern, int *count, double *seconds) {
    char *match = NULL;
    if (asprintf(&match, "%s/%s/%s/record.json", base, stage, pattern) < 0)
        return;
    glob_t paths = {0};
    if (glob(match, 0, NULL, &paths) == 0) {
        for (size_t i = 0; i < paths.gl_pathc; i++) {
            json_object *record = json_object_from_file(paths.gl_pathv[i]);
            if (!record)
                continue;
            (*count)++;
            *seconds += route_seconds(record);
            json_object_put(record);
        }
    }
    globfree(&paths);
    free(match);
}

static void coverage(const char *base, double *laid, double *ink) {
    char *match = NULL;
    if (asprintf(&match, "%s/sheet/review*.json", base) >= 0) {
        glob_t paths = {0};
        if (glob(match, 0, NULL, &paths) == 0 && paths.gl_pathc) {
            json_object *receipt = json_object_from_file(paths.gl_pathv[paths.gl_pathc - 1]);
            json_object *row;
            if (receipt && json_object_object_get_ex(receipt, "coverage", &row)) {
                json_object *value;
                if (json_object_object_get_ex(row, "laid_cm2", &value))
                    *laid = json_object_get_double(value);
                if (json_object_object_get_ex(row, "ink_cm2", &value))
                    *ink = json_object_get_double(value);
            }
            if (receipt)
                json_object_put(receipt);
        }
        globfree(&paths);
    }
    free(match);
    if (*laid > 0)
        return;
    char *layout = NULL;
    if (asprintf(&layout, "%s/sheet/layout.json", base) < 0)
        return;
    json_object *document = json_object_from_file(layout), *pieces;
    free(layout);
    if (!document || !json_object_object_get_ex(document, "pieces", &pieces)) {
        if (document)
            json_object_put(document);
        return;
    }
    size_t n = json_object_array_length(pieces);
    for (size_t i = 0; i < n; i++) {
        json_object *piece = json_object_array_get_idx(pieces, i), *area;
        if (json_object_object_get_ex(piece, "area", &area))
            *laid += json_object_get_double(area) / 100;
    }
    json_object_put(document);
}

static int status(Roots *roots, char **samples, int count, const char *scan) {
    char *match = NULL;
    if (asprintf(&match, "%s/*/*", roots->route) < 0)
        return 1;
    glob_t scans = {0};
    int found = glob(match, 0, NULL, &scans) == 0;
    free(match);
    printf(
        "Route: %s\n", strrchr(roots->route, '/') ? strrchr(roots->route, '/') + 1 : roots->route);
    printf("%-14s%-16s%16s%16s%12s%12s\n", "scroll", "scan", "sheet-nodes", "pieces", "cm2 laid",
        "cm2 read");
    int shown = 0;
    if (found) {
        for (size_t i = 0; i < scans.gl_pathc; i++) {
            char *base = scans.gl_pathv[i];
            if (!directory(base))
                continue;
            char *volume = strrchr(base, '/');
            if (!volume)
                continue;
            *volume++ = 0;
            char *sample = strrchr(base, '/');
            if (!sample)
                continue;
            sample++;
            if (!selected(sample, samples, count) || (scan && strcmp(scan, volume) != 0))
                continue;
            char *scan_base = NULL;
            if (asprintf(&scan_base, "%s/%s/%s", roots->route, sample, volume) < 0)
                continue;
            char *node_base = NULL;
            if (asprintf(&node_base, "%s/%s/%s", roots->nodes, sample, volume) < 0) {
                free(scan_base);
                continue;
            }
            int nodes = 0, pieces = 0;
            double node_seconds = 0, piece_seconds = 0, laid = 0, ink = 0;
            stage_totals(node_base, "nodes", "*", &nodes, &node_seconds);
            stage_totals(scan_base, "pieces", "rows-*", &pieces, &piece_seconds);
            if (!pieces)
                stage_totals(scan_base, "pieces", "band-*", &pieces, &piece_seconds);
            coverage(scan_base, &laid, &ink);
            printf("%-14s%-16s%8d %6.2fh%8d %6.2fh%12.0f%12.0f\n", sample, volume, nodes,
                node_seconds / 3600, pieces, piece_seconds / 3600, laid, ink);
            shown++;
            free(node_base);
            free(scan_base);
        }
    }
    globfree(&scans);
    if (!shown)
        puts("No completed units for this selection.");
    return 0;
}

static void caught(int number) { interrupted = number; }

static int stage_index(const char *name) {
    for (int i = 0; i < (int)(sizeof(pipeline_stages) / sizeof(*pipeline_stages)); i++) {
        if (!strcmp(name, pipeline_stages[i].name))
            return i;
    }
    return -1;
}

static void usage(FILE *out) {
    fputs("Usage: ./run.sh level.pipeline status [SCROLL ...] [--scan ID]\n"
          "       ./run.sh level.pipeline index\n"
          "       ./run.sh level.pipeline run STAGE [SCROLL ...] [--band] [--scan ID]\n"
          "       ./run.sh level.pipeline follow STAGE [SCROLL ...] [--band]\n",
        out);
}

int hx_pipeline_status(int argc, char **argv) {
    int count = 0;
    const char *scan = NULL;
    char *samples[argc + 1];
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--scan") && !scan && i + 1 < argc) {
            scan = argv[++i];
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            puts("Usage: ./run.sh status [SCROLL ...] [--scan ID]");
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Invalid status argument: %s\n", argv[i]);
            return 2;
        } else {
            samples[count++] = argv[i];
        }
    }
    Roots roots;
    if (!roots_open(&roots)) {
        return 1;
    }
    int code = status(&roots, samples, count, scan);
    release_roots(&roots);
    return code;
}

int hx_pipeline_main(int argc, char **argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(argc < 2 ? stderr : stdout);
        return argc < 2;
    }
    if (!strcmp(argv[1], "status")) {
        return hx_pipeline_status(argc - 2, argv + 2);
    }
    Roots roots;
    if (!roots_open(&roots) || !roots_ready(&roots)) {
        release_roots(&roots);
        return 1;
    }
    struct sigaction action = {.sa_handler = caught};
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);
    int code = 0;
    if (!strcmp(argv[1], "index")) {
        char *arguments[] = {"./run.sh", "storage.route_database", "index", NULL};
        code = run(arguments);
    } else if (!strcmp(argv[1], "run") || !strcmp(argv[1], "follow")) {
        int follower = !strcmp(argv[1], "follow");
        int stage = argc < 3 ? -1 : stage_index(argv[2]);
        if (stage < 0 || (follower && stage == STAGE_ALL)) {
            usage(stderr);
            code = 2;
        } else {
            int band = 0;
            const char *scan = NULL;
            char **samples = calloc((size_t)argc, sizeof(*samples));
            int sample_count = 0;
            for (int i = 3; samples && i < argc; i++) {
                if (!strcmp(argv[i], "--band")) {
                    band = 1;
                } else if (!strcmp(argv[i], "--scan") && i + 1 < argc) {
                    scan = argv[++i];
                } else {
                    samples[sample_count++] = argv[i];
                }
            }
            if (!samples) {
                fputs("HerculeX: out of memory\n", stderr);
                code = 1;
            } else if (scan && sample_count != 1) {
                fputs("HerculeX: --scan needs exactly one scroll\n", stderr);
                code = 2;
            } else if (band && !pipeline_stages[stage].band) {
                fputs("HerculeX: --band applies to all, sheets and pieces\n", stderr);
                code = 2;
            } else if ((stage == STAGE_LAYOUT || stage == STAGE_INK) && !sample_count) {
                fprintf(stderr, "HerculeX: run %s needs a scroll\n", pipeline_stages[stage].name);
                code = 2;
            } else if (!follower && stage == STAGE_PIECES &&
                !require_nodes(&roots, samples, sample_count, scan, band)) {
                fputs("HerculeX: the pieces stage has no finished node input\n", stderr);
                code = 1;
            } else if (follower) {
                code = follow(&roots, stage, samples, sample_count, band, scan);
            } else if (stage == STAGE_ALL) {
                char *path = path_join(roots.route, "pipeline.lock");
                int lock = path ? open(path, O_CREAT | O_CLOEXEC | O_RDWR, 0666) : -1;
                if (lock < 0 || flock(lock, LOCK_EX)) {
                    fprintf(stderr, "HerculeX: cannot lock the pipeline: %s\n", strerror(errno));
                    code = 1;
                } else {
                    code = run_all(&roots, samples, sample_count, band, scan);
                }
                if (lock >= 0) {
                    close(lock);
                }
                free(path);
            } else {
                code = stage_command(&roots, stage, samples, sample_count, band, scan, NULL);
            }
            free(samples);
        }
    } else {
        usage(stderr);
        code = 2;
    }
    release_roots(&roots);
    return code;
}

#include "unwrap.inc"

#ifndef HX_UNITY
int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }
    if (!strcmp(argv[1], "runner.sheet_stage")) {
        return hx_sheet_main(argc - 1, argv + 1);
    }
    if (!strcmp(argv[1], "level.unwrap") || !strcmp(argv[1], "runner.unwrap")) {
        return hx_unwrap_main(argc - 1, argv + 1);
    }
    return hx_pipeline_main(argc - 1, argv + 1);
}
#endif
