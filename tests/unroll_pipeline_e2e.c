#include "../base.h"

static int copy_only;

static int audit_link(const char *source, const char *target) {
    if (copy_only) {
        errno = EXDEV;
        return -1;
    }
    return link(source, target);
}

#define link audit_link
#include "../runner/keep_faces.c"
#undef link

static char fixture[PATH_MAX] = "/tmp/hx-unroll.XXXXXX";
static const char *native, *self;
static Arena memory;

static const char *at(const char *path) { return fmt(&memory, "%s/%s", fixture, path); }

static void text(const char *path, const char *value) { write_file(path, value, strlen(value)); }

static int equals(const char *path, const char *value) {
    Bytes bytes = read_file(&memory, path);
    return bytes.size == strlen(value) && !memcmp(bytes.data, value, bytes.size);
}

static void clean(void) { remove_tree(fixture); }

static void executable(const char *path, const char *body) {
    text(path, body);
    need(!chmod(path, 0755), "cannot make fixture executable: %s", path);
}

static const char *argument(json_object *row, int i) {
    return json_object_get_string(json_object_array_get_idx(row, i));
}

static int position(json_object *row, const char *value) {
    for (size_t i = 0; i < json_object_array_length(row); i++) {
        if (!strcmp(argument(row, (int)i), value)) {
            return (int)i;
        }
    }
    return -1;
}

static json_object *jobs(void) {
    Bytes bytes = read_file(&memory, at("commands.jsonl"));
    json_object *out = json_object_new_array();
    char *rest = (char *)bytes.data, *line;
    while ((line = strsep(&rest, "\n"))) {
        if (!*line) {
            continue;
        }
        json_object *row = json_tokener_parse(line);
        need(row != NULL, "invalid fixture command log");
        if (json_object_array_length(row) >= 2 && !strcmp(argument(row, 0), "board") &&
            !strcmp(argument(row, 1), "add")) {
            json_object_array_add(out, row);
        } else {
            json_object_put(row);
        }
    }
    return out;
}

static char *execute(int expected, char **args) {
    pid_t child;
    FILE *stream = spawn_read(args, 0, &child);
    need(stream != NULL, "cannot run fixture command: %s", args[1]);
    size_t capacity = 1 << 20;
    char *output = push(&memory, capacity);
    size_t length = fread(output, 1, capacity - 1, stream);
    need(!ferror(stream) && feof(stream), "fixture output is too large");
    output[length] = 0;
    int code = spawn_close(stream, child);
    need(code == expected, "%s returned %d instead of %d:\n%s", args[1], code, expected, output);
    return output;
}

#define audit(code, ...) execute(code, (char *[]){"./run.sh", __VA_ARGS__, NULL})

static void dispatch_native(int argc, char **argv) {
    char *command[argc + 2];
    command[0] = (char *)native;
    memcpy(command + 1, argv, (size_t)argc * sizeof(*argv));
    command[argc + 1] = NULL;
    execv(native, command);
    die("cannot start native fixture: %s", strerror(errno));
}

static int command_fixture(int argc, char **argv) {
    need(argc > 0, "fixture needs a command");
    json_object *row = json_object_new_array();
    for (int i = 0; i < argc; i++) {
        json_object_array_add(row, json_object_new_string(argv[i]));
    }
    FILE *log = fopen(getenv("HX_AUDIT_LOG"), "a");
    need(log != NULL, "cannot record fixture command");
    fprintf(log, "%s\n", json_object_to_json_string_ext(row, JSON_C_TO_STRING_PLAIN));
    need(!fclose(log), "cannot finish fixture log");
    json_object_put(row);
    if (!strcmp(argv[0], "paths") || !strcmp(argv[0], "runner.keep_faces")) {
        dispatch_native(argc, argv);
    }
    if (!strcmp(argv[0], "unwrap") || !strcmp(argv[0], "unroll")) {
        char *command[argc + 2];
        command[0] = "bash";
        command[1] = !strcmp(argv[0], "unwrap") ? "unroll/unwrap.sh" : "unroll/pipeline.sh";
        memcpy(command + 2, argv + 1, (size_t)(argc - 1) * sizeof(*argv));
        command[argc + 1] = NULL;
        execv("/bin/bash", command);
        die("cannot start fixture pipeline");
    }
    if (!strcmp(argv[0], "scans.catalog.scan_choice")) {
        need(argc == 3, "unexpected scan selection");
        printf("%s V\n", argv[2]);
        return 0;
    }
    const char *model = getenv("HERCULEX_INK_MODEL");
    const char *suffix = model && strcmp(model, "reader-v3") ? fmt(&memory, "-%s", model) : "";
    const char *base = getenv("HERCULEX_OUTPUTS");
    if (!strcmp(argv[0], "ink")) {
        if (getenv("HX_AUDIT_READ_FAIL")) {
            return 4;
        }
        text(fmt(&memory, "%s/unwrapping/level/S/V/faces/level-0%s/ink.png", base, suffix), model);
    } else if (!strcmp(argv[0], "runner.review_page")) {
        text(fmt(&memory, "%s/website/sheets/sheet-S-1keV-level%s/index.html", base, suffix),
            "viewer");
        const char *face =
            fmt(&memory, "%s/unwrapping/level/S/V/faces/level-0%s/ink.png", base, suffix);
        need(!unlink(face) || errno == ENOENT, "cannot consume fixture face");
    } else {
        need(!strcmp(argv[0], "board") || !strcmp(argv[0], "card"), "unexpected command: %s",
            argv[0]);
    }
    return 0;
}

static int trace_fixture(int argc, char **argv) {
    const char *out = NULL, *graph = NULL;
    for (int i = 2; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--out")) {
            out = argv[i + 1];
        } else if (!strcmp(argv[i], "--save-graph")) {
            graph = argv[i + 1];
        }
    }
    need(out && graph, "missing trace fixture output");
    const char *name = strrchr(out, '/'), *failed = getenv("HX_AUDIT_TRACE_FAIL");
    if (failed && !strcmp(name ? name + 1 : out, failed)) {
        return 1;
    }
    text(fmt(&memory, "%s/record.json", out), "{}");
    u8 data[80] = {0};
    write_file(fmt(&memory, "%s/surfels.bin", graph), data, sizeof(data));
    return 0;
}

static const char *python_path(void) {
    char *paths = fmt(&memory, "%s", getenv("PATH")), *part;
    while ((part = strsep(&paths, ":"))) {
        const char *candidate = fmt(&memory, "%s/python3", *part ? part : ".");
        if (!access(candidate, X_OK)) {
            char *resolved = realpath(candidate, NULL);
            need(resolved != NULL, "cannot resolve Python for existing pipeline steps");
            return resolved;
        }
    }
    die("existing pipeline steps need python3");
}

int main(int argc, char **argv) {
    memory = arena(64 << 20);
    native = getenv("HX_AUDIT_NATIVE");
    if (argc >= 2 && !strcmp(argv[1], "--fixture")) {
        return command_fixture(argc - 2, argv + 2);
    }
    if (argc >= 2 && !strcmp(argv[1], "--trace")) {
        return trace_fixture(argc, argv);
    }
    if (argc >= 2 && !strcmp(argv[1], "--fetch")) {
        need(argc >= 5, "missing fetch fixture arguments");
        const char *failed = getenv("HX_AUDIT_FETCH_FAIL");
        if (failed && !strcmp(argv[3], failed)) {
            return 1;
        }
        text(argv[4], "prediction");
        return 0;
    }
    need(argc == 2, "expected the native test runner");
    native = realpath(argv[1], NULL);
    self = realpath(argv[0], NULL);
    char source[PATH_MAX];
    need(native && self && getcwd(source, sizeof(source)) && mkdtemp(fixture),
        "cannot create fixture");
    atexit(clean);
    char *resolved = realpath(fixture, NULL);
    need(resolved && strlen(resolved) < sizeof(fixture), "fixture path too long");
    snprintf(fixture, sizeof(fixture), "%s", resolved);
    free(resolved);
    const char *scripts[] = {"pipeline.sh", "unwrap.sh", "units.sh"};
    for (int i = 0; i < 3; i++) {
        Bytes code = read_file(&memory, fmt(&memory, "%s/unroll/%s", source, scripts[i]));
        const char *target = at(fmt(&memory, "unroll/%s", scripts[i]));
        write_file(target, code.data, code.size);
        need(!chmod(target, 0755), "cannot install fixture pipeline");
    }
    mkdirs(at("level/level/threads_apple"));
    mkdirs(at("level/level/threads_cuda"));
    const char *sources[] = {"level_solver.c", "sheet_flattening.c", "scroll_axis.c"};
    for (int i = 0; i < 3; i++) {
        text(at(fmt(&memory, "level/level/%s", sources[i])), "");
    }
    executable(at("run.sh"), "#!/bin/sh\nexec 2>&1\nexec \"$HX_AUDIT_SELF\" --fixture \"$@\"\n");
    executable(
        at("runtime/bin/unroll-trace"), "#!/bin/sh\nexec \"$HX_AUDIT_SELF\" --trace \"$@\"\n");
    executable(at("runtime/bin/unroll-stitch"), "#!/bin/sh\nexit 9\n");
    const char *python = python_path();
    executable(at("bin/python3"),
        "#!/bin/sh\ncase \"${2:-}\" in\n"
        "  */unroll/fetch.py) shift 2; exec \"$HX_AUDIT_SELF\" --fetch \"$@\";;\n"
        "esac\nexec \"$HX_AUDIT_PYTHON\" \"$@\"\n");
    for (int i = 0; environ[i];) {
        const char *word = environ[i];
        if (!strncmp(word, "HERCULEX_", 9) || !strncmp(word, "UNROLL_", 7) ||
            !strncmp(word, "STITCH_", 7)) {
            char *key = strndup(word, strcspn(word, "="));
            need(key && !unsetenv(key), "cannot isolate fixture environment");
            free(key);
        } else {
            i++;
        }
    }
    setenv("HX_AUDIT_NATIVE", native, 1);
    setenv("HX_AUDIT_SELF", self, 1);
    setenv("HX_AUDIT_PYTHON", python, 1);
    setenv("HX_AUDIT_LOG", at("commands.jsonl"), 1);
    setenv("HERCULEX_DATA", at("data with spaces"), 1);
    setenv("UNROLL_WORK", at("work with spaces"), 1);
    setenv("UNROLL_RUNS", at("runs with spaces"), 1);
    setenv("PATH", fmt(&memory, "%s:%s", at("bin"), getenv("PATH")), 1);
    text(at("work with spaces/scan.json"),
        "{\"scroll\":\"S\",\"scan\":\"V\",\"curl\":1,\"voxel_um\":9.362}");
    text(at("work with spaces/units_plan.json"),
        "[{\"name\":\"good\",\"z0\":0,\"z1\":768,\"x0\":0,\"x1\":64,\"y0\":0,\"y1\":64},"
        "{\"name\":\"missing\",\"z0\":576,\"z1\":1344,\"x0\":0,\"x1\":64,\"y0\":0,\"y1\":64},"
        "{\"name\":\"broken\",\"z0\":1152,\"z1\":1920,\"x0\":0,\"x1\":64,\"y0\":0,\"y1\":64}]");
    need(!chdir(fixture), "cannot enter fixture");
    need(strstr(audit(0, "paths", "library"), at("data with spaces/library")),
        "data root was ignored");
    setenv("HERCULEX_LIBRARY", at("custom library"), 1);
    need(strstr(audit(0, "paths", "library"), at("custom library")), "explicit root was ignored");
    unsetenv("HERCULEX_LIBRARY");
    setenv("HERCULEX_DISC_PCIE", at("custom library"), 1);
    need(strstr(audit(0, "paths", "library"), at("custom library")), "legacy root was ignored");
    unsetenv("HERCULEX_DISC_PCIE");
    need(strstr(audit(0, "unwrap", "--help"), "Usage:"), "missing unwrap help");
    need(strstr(audit(0, "unroll", "--help"), "Usage:"), "missing pipeline help");
    audit(2, "unwrap", "--unknown");
    audit(2, "unwrap", "S:");
    audit(2, "unwrap", "S:V", "--from", "invalid");
    setenv("HX_AUDIT_FETCH_FAIL", "missing", 1);
    setenv("HX_AUDIT_TRACE_FAIL", "broken", 1);
    const char *result = audit(1, "unroll", "S", "V", "trace");
    need(strstr(result, "2 unit(s) failed") && strstr(result, "missing.fetch.log") &&
            strstr(result, "broken.log"),
        "trace failures or their log paths were lost");
    const char *graph = at("work with spaces/graphs/good/surfels.bin");
    i64 stamp = mtime_ns(graph);
    unsetenv("HX_AUDIT_FETCH_FAIL");
    unsetenv("HX_AUDIT_TRACE_FAIL");
    need(strstr(audit(0, "unroll", "S", "V", "trace"), "SKIP good") && mtime_ns(graph) == stamp,
        "trace retry rewrote a finished graph");
    const char *sentinel = at("runs with spaces/pieces/keep");
    text(sentinel, "existing pieces");
    need(!unlink(at("work with spaces/units/missing/record.json")), "cannot remove fixture record");
    need(strstr(audit(1, "unroll", "S", "V", "stitch"), "unfinished") &&
            equals(sentinel, "existing pieces"),
        "incomplete stitching removed earlier pieces");
    setenv("HX_AUDIT_FETCH_FAIL", "missing", 1);
    audit(1, "unroll", "S", "V", "all", "--from", "trace");
    unsetenv("HX_AUDIT_FETCH_FAIL");
    need(equals(sentinel, "existing pieces"), "a failed trace entered stitching");
    text(at("commands.jsonl"), "");
    setenv("UNROLL_PAGE", "a-page", 1);
    setenv("UNROLL_TRACE_ARGS", "--a 1 --b 2", 1);
    audit(0, "unwrap", "--queue", "--rotate180", "S:V");
    unsetenv("UNROLL_PAGE");
    unsetenv("UNROLL_TRACE_ARGS");
    json_object *queued = jobs();
    need(json_object_array_length(queued) == 7, "missing queued stages");
    const char *stages[] = {"plan", "trace", "stitch", "layout", "root", "read", "page"};
    for (int i = 0; i < 7; i++) {
        json_object *job = json_object_array_get_idx(queued, i);
        int start = position(job, "--");
        need(start >= 0 && !strcmp(argument(job, start + 1), "./run.sh") &&
                !strcmp(argument(job, start + 2), "unroll") &&
                !strcmp(argument(job, start + 5), stages[i]),
            "incorrect queued command");
        need(position(job, "UNROLL_TRACE_ARGS=--a 1 --b 2") >= 0 &&
                position(job, fmt(&memory, "UNROLL_WORK=%s", at("work with spaces"))) >= 0 &&
                position(job,
                    fmt(&memory, "HERCULEX_OUTPUTS=%s", at("data with spaces/outputs"))) >= 0 &&
                position(job,
                    fmt(&memory, "HERCULEX_LIBRARY=%s", at("data with spaces/library"))) >= 0 &&
                position(job, fmt(&memory, "UNROLL_RUNS=%s", at("runs with spaces"))) >= 0 &&
                position(job, "UNROLL_INK=reader-v3") >= 0,
            "queued settings were lost");
        int dependency = position(job, "--after");
        need(i ? dependency >= 0 &&
                    !strcmp(argument(job, dependency + 1),
                        argument(json_object_array_get_idx(queued, i - 1), 2))
               : dependency < 0,
            "incorrect queue dependency");
        need((position(job, "--rotate180") >= 0) == (i == 4), "rotation went to the wrong stage");
        if (i == 5) {
            need(position(job, "--gpu") >= 0 && position(job, "--engine") >= 0 &&
                    position(job, "--link") >= 0,
                "reader resource slots were lost");
        }
    }
    json_object_put(queued);
    text(at("commands.jsonl"), "");
    audit(0, "unwrap", "--queue", "--from", "read", "S");
    queued = jobs();
    need(json_object_array_length(queued) == 2 &&
            !strcmp(argument(json_object_array_get_idx(queued, 0),
                        position(json_object_array_get_idx(queued, 0), "--") + 5),
                "read"),
        "queued resume did not start at reading");
    json_object_put(queued);
    const char *keep = at("runs with spaces/root/keep");
    text(keep, "existing read");
    audit(0, "unwrap", "--from", "read", "S:V");
    need(equals(keep, "existing read") &&
            equals(at("runs with spaces/faces/level-0/ink.png"), "reader-v3"),
        "read resume lost its earlier outputs");
    struct stat page;
    need(!lstat(at("data with spaces/outputs/website/sheets/sheet-S-1keV-trace"), &page) &&
            S_ISLNK(page.st_mode) &&
            exists(at("data with spaces/outputs/website/sheets/sheet-S-1keV-trace/index.html")),
        "publication did not link a fresh viewer root");
    audit(0, "unwrap", "--from", "page", "S:V");
    setenv("UNROLL_INK", "other", 1);
    audit(0, "unwrap", "--from", "read", "S:V");
    unsetenv("UNROLL_INK");
    need(equals(at("runs with spaces/faces/level-0-other/ink.png"), "other") &&
            equals(at("runs with spaces/faces/level-0/ink.png"), "reader-v3"),
        "model backups collided");
    text(at("commands.jsonl"), "");
    setenv("HX_AUDIT_READ_FAIL", "1", 1);
    audit(4, "unwrap", "--from", "read", "S:V");
    unsetenv("HX_AUDIT_READ_FAIL");
    need(!strstr((char *)read_file(&memory, at("commands.jsonl")).data, "runner.review_page") &&
            equals(keep, "existing read"),
        "a failed read entered publication");
    const char *original = at("original/ink.png"), *retained = at("retained/ink.png");
    text(original, "first");
    mkdirs(at("original/empty"));
    keep_faces_tree(&memory, at("original"), at("retained"));
    struct stat before, after;
    need(!stat(original, &before) && !stat(retained, &after) && before.st_ino == after.st_ino &&
            before.st_dev == after.st_dev && exists(at("retained/empty")),
        "face backup did not keep hard links");
    text(original, "second");
    copy_only = 1;
    keep_faces_tree(&memory, at("original"), at("retained"));
    need(equals(retained, "second") && !stat(original, &before) && !stat(retained, &after) &&
            before.st_ino != after.st_ino && mtime_ns(original) == mtime_ns(retained),
        "cross-filesystem fallback lost face bytes or metadata");
    need(!chdir(source), "cannot leave fixture");
    puts("unroll pipeline e2e passed");
    return 0;
}
