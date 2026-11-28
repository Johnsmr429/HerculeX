#define _GNU_SOURCE
#include <errno.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

const char hx_usage_check[] = "Run the native end-to-end and golden checks.";

typedef struct {
    const char *path;
    double start;
    pid_t pid;
} Check;

static double clock_seconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return stamp.tv_sec + stamp.tv_nsec * 1e-9;
}

static int test_kind(const char *path, const char *kind) {
    int sheet = !strcmp(path, "tests/sheet_stage_e2e.sh");
    const char *end = strrchr(path, '.');
    if (!end || (strcmp(end, ".sh") != 0 && strcmp(end, ".c") != 0)) {
        return 0;
    }
    size_t n = (size_t)(end - path);
    if ((n < 4 || memcmp(path + n - 4, "_e2e", 4) != 0) &&
        (n < 7 || memcmp(path + n - 7, "_golden", 7) != 0)) {
        return 0;
    }
    int gpu = sheet || strstr(path, "_gpu_") != NULL || !strcmp(path, "tests/reader_e2e.c");
    return (strcmp(kind, "cpu") != 0 || !gpu);
}

int hx_check_main(int argc, char **argv) {
    const char *kind = "all";
    int workers = 1;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            puts("Usage: ./run.sh check [cpu|all] [--workers N]\n"
                 "       ./run.sh check lint [--fix]");
            return 0;
        }
        if (!strcmp(arg, "cpu") || !strcmp(arg, "all")) {
            kind = arg;
            continue;
        }
        if (!strcmp(arg, "--workers") && i + 1 < argc) {
            char *end;
            long n = strtol(argv[++i], &end, 10);
            if (!*end && n >= 1 && n <= 64) {
                workers = (int)n;
                continue;
            }
        }
        fprintf(stderr, "Invalid test argument: %s\n", arg);
        return 2;
    }
    glob_t files = {0};
    int result = glob("tests/*", 0, NULL, &files);
    if (result && result != GLOB_NOMATCH) {
        fputs("Cannot list native tests\n", stderr);
        globfree(&files);
        return 1;
    }
    int active = 0, passed = 0, skipped = 0, failed = 0;
    Check checks[64] = {0};
    double began = clock_seconds();
    size_t i = 0;
    while (i < files.gl_pathc || active) {
        if (i < files.gl_pathc && active < workers) {
            const char *path = files.gl_pathv[i++];
            if (!test_kind(path, kind)) {
                continue;
            }
            int slot = 0;
            while (checks[slot].pid > 0) {
                slot++;
            }
            double start = clock_seconds();
            pid_t pid = fork();
            if (!pid) {
                if (!strcmp(strrchr(path, '.'), ".c")) {
                    execl("./run.sh", "./run.sh", "test-source", path, NULL);
                } else {
                    execl("/bin/sh", "sh", path, NULL);
                }
                _exit(127);
            }
            if (pid < 0) {
                perror(path);
                failed++;
                continue;
            }
            checks[slot] = (Check){.path = path, .start = start, .pid = pid};
            active++;
            continue;
        }
        int status;
        struct rusage usage;
        pid_t pid = wait4(-1, &status, 0, &usage);
        if (pid < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("wait for native tests");
            failed++;
            break;
        }
        for (int j = 0; j < workers; j++) {
            Check *test = checks + j;
            if (test->pid != pid) {
                continue;
            }
            int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            const char *state = code == 0 ? "passed" : code == 77 ? "skipped" : "failed";
            passed += code == 0;
            skipped += code == 77;
            failed += code != 0 && code != 77;
#ifdef __APPLE__
            double gb = usage.ru_maxrss / 1073741824.0;
#else
            double gb = usage.ru_maxrss / 1048576.0;
#endif
            printf(
                "%s: %s (%.2f s, %.2f GB)\n", test->path, state, clock_seconds() - test->start, gb);
            test->pid = 0;
            active--;
            break;
        }
    }
    globfree(&files);
    printf("%d passed, %d skipped, %d failed in %.2f s\n", passed, skipped, failed,
        clock_seconds() - began);
    return failed != 0;
}
