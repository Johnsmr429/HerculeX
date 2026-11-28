#ifndef HX_UNITY
const char hx_usage[] = "Report peak memory of a native command and its processes.";
#endif

typedef struct {
    int pid;
    i64 born, peak;
    char command[64];
} MemoryTop;

static void memory_trace_name(int pid, char out[64]) {
#ifdef __APPLE__
    if (proc_name(pid, out, 64) <= 0) {
        snprintf(out, 64, "%d", pid);
    }
#else
    char path[80];
    snprintf(path, sizeof(path), "/proc/%d/comm", pid);
    FILE *f = fopen(path, "r");
    if (f) {
        if (fgets(out, 64, f)) {
            out[strcspn(out, "\n")] = 0;
        }
        fclose(f);
    }
    if (!*out) {
        snprintf(out, 64, "%d", pid);
    }
#endif
}

static int memory_trace_order(const void *aa, const void *bb) {
    const MemoryTop *a = aa, *b = bb;
    return a->peak == b->peak ? (a->pid > b->pid) - (a->pid < b->pid) : a->peak > b->peak ? -1 : 1;
}

int memory_trace_command(int argc, char **argv) {
    double per = 0;
    if (argc > 0 && !strcmp(argv[0], "--per")) {
        need(argc > 2, "--per needs a count and a command");
        per = board_value(argv[1]);
        need(per > 0, "--per needs a positive count");
        argc -= 2;
        argv += 2;
    }
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("runner.memory_trace [--per COUNT] MODULE|EXECUTABLE [ARGS]");
        return !argc;
    }
    const char *target = argv[0];
    double start = now();
    pid_t pid = fork();
    need(pid >= 0, "cannot start %s: %s", target, strerror(errno));
    if (!pid) {
        setpgid(0, 0);
        if (strchr(target, '/')) {
            execv(target, argv);
        } else {
            char *command[argc + 2];
            command[0] = "./run.sh";
            memcpy(command + 1, argv, argc * sizeof(*command));
            command[argc + 1] = NULL;
            execv(command[0], command);
        }
        perror(target);
        _exit(127);
    }
    setpgid(pid, pid);
    MemoryTop *tops = NULL;
    int n = 0, cap = 0, status;
    i64 peak = 0;
    struct rusage spent;
    for (;;) {
        pid_t done = wait4(pid, &status, WNOHANG, &spent);
        if (done < 0 && errno == EINTR) {
            continue;
        }
        need(done >= 0, "cannot wait for %s: %s", target, strerror(errno));
        if (done) {
            break;
        }
        int pids[1024];
        int active = board_group(pid, pids, 1024);
        i64 rss = 0;
        for (int p = 0; p < active; p++) {
            int child = pids[p];
            BoardUsage use = board_process(child, NULL);
            if (!use.born) {
                continue;
            }
            rss += use.rss;
            int i = 0;
            while (i < n && (tops[i].pid != child || tops[i].born != use.born)) {
                i++;
            }
            if (i == n) {
                if (n == cap) {
                    cap = cap ? 2 * cap : 16;
                    void *grown = realloc(tops, cap * sizeof(*tops));
                    need(grown != NULL, "cannot track child processes");
                    tops = grown;
                }
                tops[n++] = (MemoryTop){.pid = child, .born = use.born};
            }
            memory_trace_name(child, tops[i].command);
            i64 high = use.rss > use.peak ? use.rss : use.peak;
            if (high > tops[i].peak) {
                tops[i].peak = high;
            }
            if (high > peak) {
                peak = high;
            }
        }
        if (rss > peak) {
            peak = rss;
        }
        board_pause(0.005);
    }
    i64 last = spent.ru_maxrss;
#ifndef __APPLE__
    last *= 1024;
#endif
    if (last > peak) {
        peak = last;
    }
    fprintf(stderr, "peak %.2f GB in %.0f s", peak / 1e9, now() - start);
    if (per) {
        fprintf(stderr, ", %.0f B each", peak / per);
    }
    fputc('\n', stderr);
    if (n > 1) {
        qsort(tops, n, sizeof(*tops), memory_trace_order);
    }
    for (int i = 0; i < n && i < 24; i++) {
        fprintf(stderr, "%7.2f GB", tops[i].peak / 1e9);
        if (per) {
            fprintf(stderr, "%7.0f B", tops[i].peak / per);
        }
        fprintf(stderr, "  %s [%d]\n", tops[i].command, tops[i].pid);
    }
    free(tops);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
