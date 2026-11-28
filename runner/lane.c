const char *runner_lane(void) {
#ifdef __APPLE__
    const char *machine = "apple";
#else
    const char *machine = "cuda";
#endif
    const char *configured = getenv("HERCULEX_LANE");
    if (configured && *configured && strcmp(configured, machine) != 0) {
        need(!strcmp(configured, "apple") || !strcmp(configured, "cuda"),
            "HERCULEX_LANE=%s: the lanes are apple and cuda", configured);
        die("HERCULEX_LANE=%s: this machine runs the %s lane", configured, machine);
    }
    return machine;
}

#ifndef __APPLE__
static u64 runner_limit(const char *path, u64 physical) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return physical;
    }
    unsigned long long bytes = 0;
    int n = fscanf(f, "%llu", &bytes);
    fclose(f);
    return n == 1 && bytes < physical ? bytes : physical;
}
#endif

u64 runner_ct_window_bytes(void) {
    long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    need(pages > 0 && size > 0, "cannot determine physical memory");
    u64 memory = (u64)pages * size;
#ifdef __APPLE__
    return memory / 3 < 5 * GB ? memory / 3 : 5 * GB;
#else
    memory = runner_limit("/sys/fs/cgroup/memory.max", memory);
    memory = runner_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes", memory);
    return memory / 3;
#endif
}
