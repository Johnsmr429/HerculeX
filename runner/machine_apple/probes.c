#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <sys/mount.h>
#include <sys/user.h>

static BoardUsage board_process(int pid, int *group) {
    struct rusage_info_v4 u;
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&u)) {
        return (BoardUsage){0};
    }
    mach_timebase_info_data_t scale;
    mach_timebase_info(&scale);
    return (BoardUsage){u.ri_proc_start_abstime, u.ri_phys_footprint,
        u.ri_lifetime_max_phys_footprint, u.ri_diskio_bytesread, u.ri_diskio_byteswritten,
        (u.ri_user_time + u.ri_system_time) * (scale.numer / (double)scale.denom) / 1e9};
}

static int board_group(int pid, int *pids, int cap) {
    return fmax(0, proc_listpgrppids(pid, pids, cap * sizeof(*pids)));
}

typedef struct {
    int pid;
    i64 ticks;
} BoardGpuClient;

static void board_gpu_add(BoardGpuClient *clients, int *count, int pid, i64 ticks) {
    for (int i = 0; i < *count; i++) {
        if (clients[i].pid == pid) {
            clients[i].ticks += ticks;
            return;
        }
    }
    clients[(*count)++] = (BoardGpuClient){pid, ticks};
}

static int board_gpu_clients(BoardGpuClient *clients, int cap) {
    pid_t lister;
    FILE *f = spawn_read(
        (char *[]){"ioreg", "-r", "-c", "AGXDeviceUserClient", "-l", "-w0", NULL}, 0, &lister);
    if (!f) {
        return 0;
    }
    char *line = NULL;
    size_t size = 0;
    int count = 0, pid = 0;
    i64 ticks = 0;
    while (getline(&line, &size, f) >= 0 && count < cap) {
        if (strstr(line, "+-o ")) {
            if (pid) {
                board_gpu_add(clients, &count, pid, ticks);
            }
            pid = 0;
            ticks = 0;
        }
        char *creator = strstr(line, "\"IOUserClientCreator\" = \"pid ");
        if (creator && !strstr(creator, "WindowServer")) {
            pid = (int)strtol(creator + strlen("\"IOUserClientCreator\" = \"pid "), NULL, 10);
        }
        for (char *at = line; (at = strstr(at, "\"accumulatedGPUTime\"="));) {
            at += strlen("\"accumulatedGPUTime\"=");
            ticks += strtoll(at, &at, 10);
        }
    }
    if (pid && count < cap) {
        board_gpu_add(clients, &count, pid, ticks);
    }
    free(line);
    spawn_close(f, lister);
    return count;
}

static BoardMachine board_probe(void) {
    BoardMachine m = {0};
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &n);
    mach_port_deallocate(mach_task_self(), host);
    i64 page = sysconf(_SC_PAGESIZE);
    i64 held = (i64)vm.wire_count + vm.internal_page_count + vm.compressor_page_count;
    u64 memory = 0;
    size_t size = sizeof(memory);
    sysctlbyname("hw.memsize", &memory, &size, 0, 0);
    m.total = (double)memory / 1e9;
    m.room = ((double)memory - held * page) / 1e9;
    m.swapped = vm.swapouts * page / 1e9;
    struct xsw_usage swap = {0};
    size = sizeof(swap);
    sysctlbyname("vm.swapusage", &swap, &size, 0, 0);
    m.swap = swap.xsu_used / 1e9;
    pid_t lister;
    FILE *f =
        spawn_read((char *[]){"ioreg", "-r", "-c", "IOAccelerator", "-d", "1", NULL}, 0, &lister);
    if (f) {
        char *line = NULL;
        size_t cap = 0;
        while (getline(&line, &cap, f) >= 0) {
            char *at = strstr(line, "\"Device Utilization %\"=");
            if (at) {
                m.device = (int)strtol(at + strlen("\"Device Utilization %\"="), NULL, 10);
                break;
            }
        }
        free(line);
        spawn_close(f, lister);
    }
    static BoardGpuClient previous[1024];
    static int count;
    static double measured;
    BoardGpuClient clients[1024];
    if (!measured) {
        count = board_gpu_clients(previous, 1024);
        measured = now();
        board_pause(0.5);
    }
    int found = board_gpu_clients(clients, 1024);
    double stamp = now();
    i64 ticks = 0;
    for (int i = 0; i < found; i++) {
        for (int j = 0; j < count; j++) {
            if (clients[i].pid == previous[j].pid) {
                ticks += clients[i].ticks - previous[j].ticks;
                break;
            }
        }
    }
    m.gpu = fmin(m.device, fmax(0, round(ticks / fmax(stamp - measured, 1e-3) / 1e7)));
    memcpy(previous, clients, found * sizeof(*clients));
    count = found;
    measured = stamp;
    getloadavg(&m.load, 1);
    return m;
}

static void board_disc_name(const char *path, char out[32]) {
    struct statfs v;
    if (statfs(path, &v) || strncmp(v.f_mntfromname, "/dev/disk", 9) != 0) {
        return;
    }
    const char *device = v.f_mntfromname;
    if (strspn(device + 9, "0123456789s") != strlen(device + 9)) {
        return;
    }
    char line[512];
    pid_t lister;
    FILE *f = spawn_read((char *[]){"diskutil", "info", (char *)device, NULL}, 0, &lister);
    if (!f) {
        return;
    }
    while (fgets(line, sizeof(line), f)) {
        char *store = strstr(line, "APFS Physical Store:");
        char *at = store ? store : strstr(line, "Part of Whole:");
        char disk[32];
        if (at && sscanf(strchr(at, ':') + 1, " %31s", disk) == 1 && !strncmp(disk, "disk", 4)) {
            disk[4 + strspn(disk + 4, "0123456789")] = 0;
            snprintf(out, 32, "%s", disk);
            if (store) {
                break;
            }
        }
    }
    spawn_close(f, lister);
}

static void board_counters(Board *b, BoardCounters *c) {
    host_cpu_load_info_data_t cpu = {0};
    mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
    mach_port_t host = mach_host_self();
    host_statistics(host, HOST_CPU_LOAD_INFO, (host_info_t)&cpu, &n);
    mach_port_deallocate(mach_task_self(), host);
    c->busy = (i64)cpu.cpu_ticks[CPU_STATE_USER] + cpu.cpu_ticks[CPU_STATE_SYSTEM] +
        cpu.cpu_ticks[CPU_STATE_NICE];
    c->idle = cpu.cpu_ticks[CPU_STATE_IDLE];
    pid_t lister;
    FILE *f = spawn_read((char *[]){"netstat", "-ibn", NULL}, 0, &lister);
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            char *words[32], *save;
            int count = 0;
            for (char *p = strtok_r(line, " \t\n", &save); p && count < 32;
                p = strtok_r(NULL, " \t\n", &save)) {
                words[count++] = p;
            }
            if (count >= 10 && !strncmp(words[2], "<Link#", 6) && strncmp(words[0], "lo", 2) != 0) {
                c->received += strtoull(words[count - 5], NULL, 10);
            }
        }
        spawn_close(f, lister);
    }
    const char *paths[] = {"/", LIBRARY, OUTPUTS};
    for (int i = 0; i < 3; i++) {
        if (!*b->discs[i] && exists(paths[i])) {
            board_disc_name(paths[i], b->discs[i]);
        }
    }
    f = spawn_read((char *[]){"iostat", "-Id", NULL}, 0, &lister);
    if (!f) {
        return;
    }
    char names[512], line[512];
    if (fgets(names, sizeof(names), f) && fgets(line, sizeof(line), f) &&
        fgets(line, sizeof(line), f)) {
        char *a, *d;
        strtok_r(line, " \t\n", &d);
        for (char *disc = strtok_r(names, " \t\n", &a); disc; disc = strtok_r(NULL, " \t\n", &a)) {
            strtok_r(NULL, " \t\n", &d);
            const char *mb = strtok_r(NULL, " \t\n", &d);
            if (!mb) {
                break;
            }
            double value = strtod(mb, NULL);
            for (int i = 0; i < 3; i++) {
                if (!strcmp(disc, b->discs[i])) {
                    c->mb[i] = value;
                }
            }
            strtok_r(NULL, " \t\n", &d);
        }
    }
    spawn_close(f, lister);
}
