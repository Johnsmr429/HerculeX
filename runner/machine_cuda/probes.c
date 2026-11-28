static i64 board_counter(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    char line[256];
    i64 value = 0;
    size_t n = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, key, n)) {
            value = strtoll(line + n, 0, 10);
            break;
        }
    }
    fclose(f);
    return value;
}

static BoardUsage board_process(int pid, int *group) {
    char path[80], line[4096];
    BoardUsage u = {0};
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE *f = fopen(path, "r");
    if (!f) {
        return u;
    }
    char *end = fgets(line, sizeof(line), f) ? strrchr(line, ')') : NULL;
    fclose(f);
    if (!end || end[2] == 'Z') {
        return u;
    }
    char *save, *zero = "0";
    char *fields[20] = {[2] = zero, [11] = zero, [12] = zero, [19] = zero};
    char *field = strtok_r(end + 2, " ", &save);
    for (int i = 0; i < 20 && field; i++, field = strtok_r(NULL, " ", &save))
        fields[i] = field;
    u.born = strtoll(fields[19], NULL, 10);
    if (group) {
        *group = (int)strtol(fields[2], NULL, 10);
        return u;
    }
    long ticks_per_second = sysconf(_SC_CLK_TCK);
    u.cpu =
        strtod(fields[11], NULL) / ticks_per_second + strtod(fields[12], NULL) / ticks_per_second;
    snprintf(path, sizeof(path), "/proc/%d/status", pid);
    u.rss = board_counter(path, "VmRSS:") * 1024;
    u.peak = board_counter(path, "VmHWM:") * 1024;
    snprintf(path, sizeof(path), "/proc/%d/io", pid);
    u.read = board_counter(path, "read_bytes:");
    u.written = board_counter(path, "write_bytes:");
    return u;
}

static int board_group(int pid, int *pids, int cap) {
    DIR *d = opendir("/proc");
    int count = 0;
    if (!d) {
        return 0;
    }
    for (struct dirent *e; count < cap && (e = readdir(d));) {
        if (isdigit((unsigned char)e->d_name[0])) {
            int group = 0, candidate = (int)strtol(e->d_name, NULL, 10);
            if (board_process(candidate, &group).born && group == pid) {
                pids[count++] = candidate;
            }
        }
    }
    closedir(d);
    return count;
}

static BoardMachine board_probe(void) {
    BoardMachine m = {0};
    m.total = board_counter("/proc/meminfo", "MemTotal:") * 1024 / 1e9;
    m.room = board_counter("/proc/meminfo", "MemAvailable:") * 1024 / 1e9;
    m.swap = (board_counter("/proc/meminfo", "SwapTotal:") -
                 board_counter("/proc/meminfo", "SwapFree:")) *
        1024 / 1e9;
    m.swapped = board_counter("/proc/vmstat", "pswpout ") * sysconf(_SC_PAGE_SIZE) / 1e9;
    pid_t lister;
    FILE *g = spawn_read((char *[]){"nvidia-smi", "--query-gpu=utilization.gpu",
                             "--format=csv,noheader,nounits", NULL},
        1, &lister);
    if (g) {
        char line[128];
        while (fgets(line, sizeof(line), g)) {
            m.gpu = fmax(m.gpu, strtol(line, NULL, 10));
        }
        spawn_close(g, lister);
    }
    m.device = m.gpu;
    getloadavg(&m.load, 1);
    return m;
}

static void board_counters(Board *b, BoardCounters *c) {
    FILE *f = fopen("/proc/stat", "r");
    if (f) {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0;
        if (fscanf(f, "cpu %llu %llu %llu %llu", &user, &nice, &system, &idle) == 4) {
            c->busy = user + nice + system;
            c->idle = idle;
        }
        fclose(f);
    }
    f = fopen("/proc/net/dev", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char *name = line + strspn(line, " \t");
            char *colon = strchr(name, ':');
            if (colon && strncmp(name, "lo", 2) != 0) {
                c->received += strtoll(colon + 1, NULL, 10);
            }
        }
        fclose(f);
    }
}
