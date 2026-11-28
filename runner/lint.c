#include <spawn.h>

extern char **environ;

typedef struct {
    char **files;
    int count;
} LintFiles;

static const char *lint_tool(Arena *a, const char *name) {
    const char *path = fmt(a, "runtime/tools/bin/%s", name);
    return access(path, X_OK) ? name : path;
}

static int lint_wait(pid_t pid) {
    int status;
    pid_t result;
    do {
        result = waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    return result != pid || !WIFEXITED(status) || WEXITSTATUS(status);
}

static int lint_exec(char **args) {
    pid_t pid;
    int error = posix_spawnp(&pid, args[0], NULL, NULL, args, environ);
    if (error) {
        fprintf(stderr, "%s: %s\n", args[0], strerror(error));
        return 1;
    }
    return lint_wait(pid);
}

static FILE *lint_output(char **args, pid_t *pid) {
    int endpoints[2];
    need(!pipe(endpoints), "cannot capture %s", args[0]);
    posix_spawn_file_actions_t ops;
    posix_spawn_file_actions_init(&ops);
    posix_spawn_file_actions_addclose(&ops, endpoints[0]);
    posix_spawn_file_actions_adddup2(&ops, endpoints[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&ops, endpoints[1]);
    int rc = posix_spawnp(pid, args[0], &ops, NULL, args, environ);
    posix_spawn_file_actions_destroy(&ops);
    close(endpoints[1]);
    need(!rc, "cannot start %s: %s", args[0], strerror(rc));
    FILE *out = fdopen(endpoints[0], "r");
    need(out != NULL, "cannot read %s", args[0]);
    return out;
}

static int lint_each(const char *tool, char **flags, int count, LintFiles group) {
    int failed = 0;
    for (int first = 0; first < group.count; first += 128) {
        int n = group.count - first;
        if (n > 128) {
            n = 128;
        }
        char *args[1 + count + n + 1];
        args[0] = (char *)tool;
        if (count) {
            memcpy(args + 1, flags, count * sizeof(*args));
        }
        memcpy(args + 1 + count, group.files + first, n * sizeof(*args));
        args[1 + count + n] = NULL;
        failed |= lint_exec(args);
    }
    return failed;
}

static int lint_tidy(Arena *a) {
    pid_t child;
    FILE *pkg = lint_output((char *[]){"pkg-config", "--cflags-only-I", "json-c", "libcurl",
                                "libzstd", "libpng", "libjpeg", "libtiff-4", "libzip", "libcrypto",
                                "sqlite3", "zlib", "libdeflate", "blosc", "freetype2", NULL},
        &child);
    char *line = NULL;
    size_t cap = 0;
    need(getline(&line, &cap, pkg) > 0 && !fclose(pkg) && !lint_wait(child),
        "cannot resolve native lint includes");
    char *flags[128] = {"-std=gnu11", "-I.", "-DHX_UNITY"};
    int n = 3;
    char *word = strtok(line, " \n\t");
    while (word) {
        need(n + 4 < 128 && !strncmp(word, "-I", 2), "invalid native lint include flags");
        flags[n++] = "-isystem";
        flags[n++] = word + 2;
        word = strtok(NULL, " \n\t");
    }
#ifdef __APPLE__
    FILE *xcrun = lint_output((char *[]){"xcrun", "--show-sdk-path", NULL}, &child);
    char sdk[PATH_MAX];
    need(fgets(sdk, sizeof(sdk), xcrun) && !fclose(xcrun) && !lint_wait(child),
        "cannot resolve Apple SDK");
    sdk[strcspn(sdk, "\r\n")] = 0;
    flags[n++] = "-isysroot";
    flags[n++] = sdk;
    flags[n++] = "-fblocks";
    flags[n++] = "-Ilevel/level/threads_apple";
#else
    flags[n++] = "-Ilevel/level/threads_cuda";
#endif
    const char *tidy = lint_tool(a, "clang-tidy");
    char *args[160] = {(char *)tidy, "--quiet", "--header-filter=HerculeX/", "herculex.c", "--"};
    memcpy(args + 5, flags, n * sizeof(*flags));
    int end = 5 + n;
#ifdef __APPLE__
    args[end++] = "-x";
    args[end++] = "objective-c";
    args[end++] = "-fobjc-arc";
    args[end++] = "-DLANE_READER=\"ink/model/reader_apple/reader.m\"";
#else
    args[end++] = "-DLANE_READER=\"ink/model/reader_cuda/reader.c\"";
#endif
    args[end] = NULL;
    int failed = lint_exec(args);
    char *files[] = {"level/pipeline.c", "runner/check.c", "storage/route_database.c",
        "level/detection/detect_nodes.c", "level/level/umbilicus.c", "level/layout/seams.c",
        "level/layout/packing.c", "level/layout/axis_fit.c", "viewer/sheet_tiles.c",
        "viewer/sheet_page.c", "viewer/refresh_viewers.c", "runner/review_check.c"};
    args[5 + n] = NULL;
    for (size_t i = 0; i < sizeof(files) / sizeof(*files); i++) {
        args[3] = files[i];
        failed |= lint_exec(args);
    }
    free(line);
    return failed;
}

int lint_command(int argc, char **argv) {
    int fix = argc == 1 && !strcmp(argv[0], "--fix");
    need(!argc || fix, "usage: ./run.sh check lint [--fix]");
    Arena a = arena(64 << 20);
    pid_t child;
    FILE *git = lint_output((char *[]){"git", "ls-files", "-z", "--cached", "--others",
                                "--exclude-standard", "--", ".", NULL},
        &child);
    int max_files = 65536;
    LintFiles c = {.files = push(&a, max_files * sizeof(char *))};
    LintFiles sh = {.files = push(&a, max_files * sizeof(char *))};
    char *path = NULL;
    size_t cap = 0;
    while (getdelim(&path, &cap, 0, git) >= 0) {
        if (!strncmp(path, "ports/scrollfiesta/", sizeof("ports/scrollfiesta/") - 1) ||
            !strncmp(path, "logs/runs/", sizeof("logs/runs/") - 1)) {
            continue;
        }
        const char *ext = strrchr(path, '.');
        if (!ext || strchr(ext, '/') || !exists(path)) {
            continue;
        }
        LintFiles *group = !strcmp(ext, ".sh") ? &sh : NULL;
        const char *native[] = {
            ".c", ".h", ".m", ".mm", ".metal", ".cu", ".cpp", ".hpp", ".hxx", ".inc"};
        for (size_t i = 0; !group && i < sizeof(native) / sizeof(*native); i++) {
            if (!strcmp(ext, native[i])) {
                group = &c;
            }
        }
        if (group) {
            need(group->count < max_files, "too many files for the lint batch");
            group->files[group->count++] = fmt(&a, "%s", path);
        }
    }
    free(path);
    need(!fclose(git) && !lint_wait(child), "git could not list repository files");
    if (access("runtime/tools/bin/clang-format", X_OK) ||
        access("runtime/tools/bin/clang-tidy", X_OK) ||
        access("runtime/tools/bin/shellcheck", X_OK) || access("runtime/tools/bin/shfmt", X_OK) ||
        access("runtime/tools/lib/clang", F_OK)) {
        need(!lint_exec((char *[]){"./run.sh", "setup", "lint", NULL}),
            "cannot install the pinned lint tools");
    }
    const char *shfmt = lint_tool(&a, "shfmt");
    const char *format = lint_tool(&a, "clang-format");
    const char *shellcheck = lint_tool(&a, "shellcheck");
    const char *biome = "runtime/node/node_modules/.bin/biome";
    if (access(biome, X_OK)) {
        char *install[] = {"npm", "install", "--prefix", "runtime/node", "--save-exact",
            "@biomejs/biome@2.6.4", NULL};
        need(!lint_exec(install), "cannot install the pinned Biome formatter");
    }
    if (fix) {
        lint_each(format, (char *[]){"--style=file", "-i"}, 2, c);
        lint_each(shfmt, (char *[]){"-i", "4", "-ci", "-w"}, 4, sh);
        char *write[] = {(char *)biome, "check", "--write", ".", NULL};
        lint_exec(write);
    }
    int failed = 0;
    failed |= lint_each(format, (char *[]){"--style=file", "--dry-run", "--Werror"}, 3, c);
    failed |= lint_each(shellcheck, NULL, 0, sh);
    failed |= lint_each(shfmt, (char *[]){"-i", "4", "-ci", "-d"}, 4, sh);
    failed |= lint_exec((char *[]){(char *)biome, "ci", "--colors=off", ".", NULL});
    failed |= lint_tidy(&a);
    munmap(a.base, a.size);
    return failed;
}
