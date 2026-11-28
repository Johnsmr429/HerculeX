#ifndef HX_UNITY
const char hx_usage[] = "Queue and monitor long jobs on one machine.";
#endif

#include <ctype.h>
#include <glob.h>
#include <signal.h>
#include <sqlite3.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/statvfs.h>

typedef struct {
    i64 born, rss, peak, read, written;
    double cpu;
} BoardUsage;

typedef struct {
    double room, swap, swapped, load, total;
    int gpu, device;
} BoardMachine;

typedef struct {
    double at, mb[3];
    i64 busy, idle, received;
} BoardCounters;

typedef struct {
    sqlite3 *db;
    const char *path;
    double tick, retry, margin, cpu;
    BoardCounters previous;
    char discs[3][32];
} Board;

static double board_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void board_pause(double seconds) {
    struct timespec t = {(time_t)seconds, (long)(fmod(seconds, 1) * 1e9)};
    while (nanosleep(&t, &t) && errno == EINTR) {
    }
}

#ifdef __APPLE__
#include "machine_apple/probes.c"
#else
#include "machine_cuda/probes.c"
#endif

static const char board_schema[] =
    "CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY, kind TEXT, command TEXT, "
    "folder TEXT, settings TEXT, owner TEXT, note TEXT, memory_gb REAL, cores REAL, gpu "
    "INTEGER, engine INTEGER, link INTEGER, priority INTEGER, nice INTEGER, background "
    "INTEGER, after TEXT, needs TEXT, clean TEXT, restartable INTEGER, alone INTEGER "
    "DEFAULT 0, last INTEGER DEFAULT 0, free_gib REAL DEFAULT 0, at_most TEXT, share "
    "INTEGER DEFAULT 0, adopted INTEGER DEFAULT 0, state TEXT, held INTEGER DEFAULT 0, "
    "again INTEGER DEFAULT 0, tries INTEGER DEFAULT 0, not_before REAL DEFAULT 0, pid "
    "INTEGER, born INTEGER, wrapper INTEGER, added REAL, started REAL, stopping REAL, "
    "ended REAL, code INTEGER, peak_gb REAL DEFAULT 0, cpu_hours REAL DEFAULT 0, "
    "read_gb REAL DEFAULT 0, written_gb REAL DEFAULT 0, why TEXT, cores_used REAL "
    "DEFAULT 0); CREATE TABLE IF NOT EXISTS samples(time REAL PRIMARY KEY, gpu REAL, "
    "room_gb REAL, swap_gb REAL, swapped_out_gb REAL, load REAL, running INTEGER, "
    "waiting INTEGER, gpu_waiting INTEGER, cpu_pct REAL, net_mb_s REAL, internal_mb_s "
    "REAL, library_mb_s REAL, outputs_mb_s REAL, engine_waiting INTEGER, link_waiting "
    "INTEGER); ";

typedef struct {
    const char *name;
    char *text;
    double real;
    i64 integer;
} BoardCell;

typedef struct {
    int count;
    BoardCell cell[];
} BoardRow;

typedef struct {
    int count;
    const char **item;
} BoardList;

typedef struct {
    BoardRow *row;
    BoardList after, needs;
    const char *name, *state, *pattern;
    double rss, memory, cores, peak, added;
    int slots, pid, priority, share, adopted, scheduled, limit;
    char reason[PATH_MAX + 160];
} BoardJob;

static const char *const board_slots[] = {"gpu", "engine", "link"};

static const char *board_text(json_object *o, const char *key) {
    const char *v = json_object_get_string(json_object_object_get(o, key));
    return v ? v : "";
}

static double board_number(json_object *o, const char *key) {
    return json_object_get_double(json_object_object_get(o, key));
}

static i64 board_integer(json_object *o, const char *key) {
    return json_object_get_int64(json_object_object_get(o, key));
}

static sqlite3_stmt *board_query(Board *b, const char *sql, ...) {
    va_list args;
    va_start(args, sql);
    char *text = sqlite3_vmprintf(sql, args);
    va_end(args);
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(b->db, text, -1, &stmt, NULL);
    sqlite3_free(text);
    need(rc == SQLITE_OK, "board: %s", sqlite3_errmsg(b->db));
    return stmt;
}

static void board_exec(Board *b, const char *sql, ...) {
    va_list args;
    va_start(args, sql);
    char *text = sqlite3_vmprintf(sql, args);
    va_end(args);
    int rc = sqlite3_exec(b->db, text, NULL, NULL, NULL);
    sqlite3_free(text);
    need(rc == SQLITE_OK, "board: %s", sqlite3_errmsg(b->db));
}

static void board_columns(
    Board *b, const char *table, const char **names, const char **types, int n) {
    for (int i = 0; i < n; i++) {
        sqlite3_stmt *q =
            board_query(b, "SELECT name FROM pragma_table_info(%Q) WHERE name=%Q", table, names[i]);
        int absent = sqlite3_step(q) == SQLITE_DONE;
        sqlite3_finalize(q);
        if (absent) {
            board_exec(b, "ALTER TABLE %s ADD COLUMN %s %s", table, names[i], types[i]);
        }
    }
}

static void board_open(Board *b) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/logs", b->path);
    mkdirs(path);
    snprintf(path, sizeof(path), "%s/board.sqlite", b->path);
    need(sqlite3_open(path, &b->db) == SQLITE_OK, "cannot open %s", path);
    sqlite3_busy_timeout(b->db, 30000);
    board_exec(b, "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; BEGIN IMMEDIATE");
    board_exec(b, "%s", board_schema);
    board_columns(b, "jobs", (const char *[]){"last", "free_gib", "at_most", "share", "cores_used"},
        (const char *[]){
            "INTEGER DEFAULT 0", "REAL DEFAULT 0", "TEXT", "INTEGER DEFAULT 0", "REAL DEFAULT 0"},
        5);
    board_columns(b, "samples",
        (const char *[]){"cpu_pct", "net_mb_s", "internal_mb_s", "library_mb_s", "outputs_mb_s",
            "engine_waiting", "link_waiting"},
        (const char *[]){"REAL", "REAL", "REAL", "REAL", "REAL", "INTEGER", "INTEGER"}, 7);
    board_exec(b, "COMMIT");
}

static json_object *board_row(sqlite3_stmt *q) {
    json_object *o = json_object_new_object();
    for (int i = 0; i < sqlite3_column_count(q); i++) {
        json_object *v = NULL;
        switch (sqlite3_column_type(q, i)) {
        case SQLITE_INTEGER: v = json_object_new_int64(sqlite3_column_int64(q, i)); break;
        case SQLITE_FLOAT: v = json_object_new_double(sqlite3_column_double(q, i)); break;
        case SQLITE_TEXT:
            v = json_object_new_string((const char *)sqlite3_column_text(q, i));
            break;
        default: break;
        }
        json_object_object_add(o, sqlite3_column_name(q, i), v);
    }
    return o;
}

static BoardRow *board_record(sqlite3_stmt *q) {
    static char names[96][32];
    static int known;
    int count = sqlite3_column_count(q);
    size_t size = sizeof(BoardRow) + count * sizeof(BoardCell);
    for (int i = 0; i < count; i++) {
        sqlite3_column_text(q, i);
        size += sqlite3_column_bytes(q, i) + 1;
    }
    BoardRow *r = malloc(size);
    need(r != NULL, "cannot allocate a board row");
    char *text = (char *)(r->cell + count);
    r->count = count;
    for (int i = 0; i < count; i++) {
        const char *name = sqlite3_column_name(q, i);
        int k = 0;
        while (k < known && strcmp(names[k], name) != 0) {
            k++;
        }
        if (k == known) {
            need(known < 96 && strlen(name) < 32, "board column %s does not fit", name);
            strcpy(names[known++], name);
        }
        r->cell[i] = (BoardCell){.name = names[k]};
        if (sqlite3_column_type(q, i) == SQLITE_NULL) {
            continue;
        }
        int bytes = sqlite3_column_bytes(q, i);
        r->cell[i].text = memcpy(text, sqlite3_column_text(q, i), bytes);
        text[bytes] = 0;
        text += bytes + 1;
        r->cell[i].real = sqlite3_column_double(q, i);
        r->cell[i].integer = sqlite3_column_int64(q, i);
    }
    return r;
}

static BoardCell *job_cell(BoardRow *r, const char *key) {
    static BoardCell empty;
    if (!r) {
        empty = (BoardCell){.name = key};
        return &empty;
    }
    int i = 0;
    while (i < r->count && strcmp(r->cell[i].name, key) != 0) {
        i++;
    }
    need(i < r->count, "board rows have no column %s", key);
    return &r->cell[i];
}

static const char *job_text(BoardRow *r, const char *key) {
    const char *t = job_cell(r, key)->text;
    return t ? t : "";
}

static double job_number(BoardRow *r, const char *key) { return job_cell(r, key)->real; }

static i64 job_integer(BoardRow *r, const char *key) { return job_cell(r, key)->integer; }

static BoardList job_list(BoardRow *r, const char *key) {
    char *text = job_cell(r, key)->text;
    if (!text) {
        return (BoardList){0};
    }
    BoardList l = {.count = 1};
    for (char *c = text; *c; c++) {
        l.count += *c == 31;
    }
    l.item = malloc(l.count * sizeof(*l.item));
    need(l.item != NULL, "cannot allocate a board list");
    for (int i = 0; i < l.count; i++) {
        l.item[i] = text;
        text += strcspn(text, "\x1f");
        *text++ = 0;
    }
    return l;
}

static BoardRow *board_named(Board *b, const char *name) {
    sqlite3_stmt *q = board_query(b, "SELECT * FROM jobs WHERE name=%Q", name);
    BoardRow *row = sqlite3_step(q) == SQLITE_ROW ? board_record(q) : NULL;
    sqlite3_finalize(q);
    need(row != NULL, "no job named %s", name);
    return row;
}

static int board_active(const char *state) {
    return !strcmp(state, "running") || !strcmp(state, "starting") || !strcmp(state, "stopping");
}

static int board_waiting(const BoardJob *job) { return !strcmp(job->state, "waiting"); }

static double board_footprint(int pid, int adopted) {
    if (pid <= 0) {
        return 0;
    }
    int pids[1024] = {pid};
    int count = adopted ? 1 : board_group(pid, pids, 1024);
    double rss = 0;
    for (int i = 0; i < count; i++) {
        rss += board_process(pids[i], NULL).rss / 1e9;
    }
    return rss;
}

static BoardJob *board_jobs(Board *b, int *n) {
    sqlite3_stmt *q = board_query(b,
        "SELECT *, (SELECT group_concat(value, char(31)) FROM json_each(CASE WHEN "
        "json_valid(after) THEN after ELSE '[]' END)) AS after_items, (SELECT "
        "group_concat(value, char(31)) FROM json_each(CASE WHEN json_valid(needs) THEN needs "
        "ELSE '[]' END)) AS needs_items, CASE WHEN json_valid(at_most) THEN "
        "json_array_length(at_most) = 2 END AS at_most_pair, CASE WHEN json_valid(at_most) "
        "THEN json_extract(at_most, '$[0]') END AS at_most_limit, CASE WHEN "
        "json_valid(at_most) THEN json_extract(at_most, '$[1]') END AS at_most_pattern "
        "FROM jobs ORDER BY priority DESC,added");
    BoardJob *jobs = NULL;
    int rc, cap = 0;
    *n = 0;
    while ((rc = sqlite3_step(q)) == SQLITE_ROW) {
        if (*n == cap) {
            cap = cap ? 2 * cap : 32;
            void *grown = realloc(jobs, cap * sizeof(*jobs));
            need(grown != NULL, "cannot allocate board jobs");
            jobs = grown;
        }
        BoardRow *row = board_record(q);
        BoardJob *j = jobs + (*n)++;
        *j = (BoardJob){.row = row,
            .name = job_text(row, "name"),
            .state = job_text(row, "state"),
            .added = job_number(row, "added"),
            .memory = job_number(row, "memory_gb"),
            .peak = job_number(row, "peak_gb"),
            .cores = job_number(row, "cores"),
            .pid = job_integer(row, "pid"),
            .priority = job_integer(row, "priority"),
            .share = job_integer(row, "share"),
            .adopted = job_integer(row, "adopted")};
        if (board_active(j->state)) {
            j->cores = fmax(j->cores, job_number(row, "cores_used"));
            j->rss = board_footprint(j->pid, j->adopted);
        }
        j->after = job_list(row, "after_items");
        j->needs = job_list(row, "needs_items");
        if (board_waiting(j) && job_integer(row, "at_most_pair")) {
            j->limit = job_integer(row, "at_most_limit");
            j->pattern = job_text(row, "at_most_pattern");
        }
        for (int i = 0; i < 3; i++) {
            j->slots |= (job_integer(row, board_slots[i]) != 0) << i;
        }
    }
    need(rc == SQLITE_DONE, "board: %s", sqlite3_errmsg(b->db));
    sqlite3_finalize(q);
    return jobs;
}

static void board_free(BoardJob *jobs, int n) {
    for (int i = 0; i < n; i++) {
        free(jobs[i].row);
        free(jobs[i].after.item);
        free(jobs[i].needs.item);
    }
    free(jobs);
}

static BoardJob *board_find(BoardJob *jobs, int n, const char *name) {
    for (int i = 0; i < n; i++) {
        if (!strcmp(jobs[i].name, name)) {
            return &jobs[i];
        }
    }
    return NULL;
}

static const char *board_item(json_object *array, int i) {
    return json_object_get_string(json_object_array_get_idx(array, i));
}

static int board_count(json_object *array) {
    return json_object_is_type(array, json_type_array) ? json_object_array_length(array) : 0;
}

static int board_can_run(BoardJob *job, BoardJob *jobs, int n, int depth) {
    if (!job || depth > n) {
        return 0;
    }
    if (board_active(job->state) || !strcmp(job->state, "done")) {
        return 1;
    }
    if (!board_waiting(job) || job_integer(job->row, "held")) {
        return 0;
    }
    for (int i = 0; i < job->after.count; i++) {
        if (!board_can_run(board_find(jobs, n, job->after.item[i]), jobs, n, depth + 1)) {
            return 0;
        }
    }
    return 1;
}

static double board_room(BoardJob *jobs, int n, double room) {
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        if (board_active(j->state)) {
            room -= fmax(0, j->memory - fmax(j->rss, j->peak));
        }
    }
    return room;
}

static double board_disc(const char *path) {
    struct statvfs v;
    return statvfs(path, &v) ? -1 : v.f_bavail * (double)v.f_frsize / GB;
}

static void board_utc(char out[64], double time_value, const char *format) {
    time_t stamp = time_value;
    struct tm utc;
    gmtime_r(&stamp, &utc);
    strftime(out, 64, format, &utc);
}

static double board_swap_limit(BoardMachine m, double fraction, double floor_gb) {
    const char *configured = getenv("HERCULEX_BOARD_SWAP_GB");
    if (configured && *configured) {
        char *end;
        double value = strtod(configured, &end);
        if (end != configured && !*end && isfinite(value) && value >= 0) {
            return value * (floor_gb / 4.2);
        }
    }
    return m.total > 0 ? fmax(floor_gb, m.total * fraction) : floor_gb;
}

static void board_decide(
    Board *b, BoardJob *j, BoardJob *jobs, int n, BoardMachine m, double room, int strangers) {
    BoardRow *r = j->row;
    char *out = j->reason;
    size_t cap = sizeof(j->reason);
    *out = 0;
    if (job_integer(r, "held")) {
        snprintf(out, cap, "held");
        return;
    }
    if (board_time() < job_number(r, "not_before")) {
        char stamp[64];
        board_utc(stamp, job_number(r, "not_before"), "%H:%M");
        snprintf(out, cap, "rests until %s UTC", stamp);
        return;
    }
    for (int i = 0; i < j->after.count; i++) {
        const char *name = j->after.item[i];
        BoardJob *dep = board_find(jobs, n, name);
        if (!dep || strcmp(dep->state, "done") != 0) {
            snprintf(out, cap, "after %s, %s", name, dep ? dep->state : "not on the board");
            return;
        }
    }
    for (int i = 0; i < j->needs.count; i++) {
        const char *name = j->needs.item[i];
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", name[0] == '/' ? "" : job_text(r, "folder"), name);
        if (!exists(path)) {
            snprintf(out, cap, "needs %s", name);
            return;
        }
    }
    if (job_integer(r, "last")) {
        BoardJob *first = NULL;
        int count = 0;
        for (int i = 0; i < n; i++) {
            BoardJob *a = &jobs[i];
            if (a->added < j->added && !job_integer(a->row, "last") &&
                strcmp(a->state, "done") != 0 && board_can_run(a, jobs, n, 0)) {
                count++;
                if (!first || a->added < first->added) {
                    first = a;
                }
            }
        }
        if (count) {
            snprintf(out, cap, "last: %d jobs ahead, first %s", count, first->name);
            return;
        }
    }
    double free_gib = job_number(r, "free_gib");
    double free_space = free_gib > 0 ? board_disc("/") : 0;
    if (free_space < free_gib) {
        snprintf(out, cap, "needs %g GiB free on the Mac, has %.0f", free_gib, free_space);
        return;
    }
    if (j->pattern) {
        int limit = j->limit;
        const char *pattern = j->pattern;
        glob_t paths = {0};
        glob(pattern, 0, NULL, &paths);
        size_t found = paths.gl_pathc;
        globfree(&paths);
        if ((i64)found >= limit) {
            const char *base = strrchr(pattern, '/');
            snprintf(out, cap, "at most %d of %s before it, %zu there", limit - 1,
                base ? base + 1 : pattern, found);
            return;
        }
    }
    double used = 0;
    int substantial = 0, whole[3] = {0}, shared[3] = {0};
    BoardJob *first[3] = {0}, *active = NULL;
    for (int i = 0; i < n; i++) {
        BoardJob *a = &jobs[i];
        if (!board_active(a->state)) {
            continue;
        }
        active = active ? active : a;
        if (job_integer(a->row, "alone")) {
            snprintf(out, cap, "the machine is with %s", a->name);
            return;
        }
        used += a->cores;
        substantial |= a->memory > 0.2;
        for (int s = 0; s < 3; s++) {
            if (a->slots & (1 << s)) {
                first[s] = first[s] ? first[s] : a;
                shared[s] += a->share != 0;
                whole[s] += !a->share;
            }
        }
    }
    if (job_integer(r, "alone") && (active || strangers)) {
        if (active) {
            snprintf(out, cap, "wants the machine to itself: %s", active->name);
        } else {
            snprintf(
                out, cap, "wants the machine to itself: %d processes off the board", strangers);
        }
        return;
    }
    for (int s = 0; s < 3; s++) {
        int limit = (int[]){1, 0, 2}[s];
        int fits = limit &&
            (j->share ? whole[s] <= 1 && shared[s] < limit : !whole[s] && shared[s] <= limit);
        if ((j->slots & (1 << s)) && first[s] && !fits) {
            snprintf(out, cap, "%s is with %s", board_slots[s], first[s]->name);
            return;
        }
    }
    double swap_limit = board_swap_limit(m, 0.25, 4.2);
    if ((j->slots & 1) && m.gpu >= 50 && !first[0]) {
        snprintf(out, cap, "GPU %d%% busy off the board", m.gpu);
    } else if (used + j->cores > b->cpu) {
        snprintf(out, cap, "needs %g cores, %.1f of %g taken", j->cores, used, b->cpu);
    } else if (j->memory > room && substantial) {
        snprintf(out, cap, "needs %.1f GB, room %.1f", j->memory, room);
    } else if (j->memory > 1 && m.swap > swap_limit && (m.total <= 0 || room < j->memory + 4.0)) {
        snprintf(out, cap, "swap %.1f GB, over %.1f", m.swap, swap_limit);
    }
}

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

static void board_path(
    char out[PATH_MAX], Board *b, const char *dir, const char *name, const char *suffix) {
    int n = snprintf(out, PATH_MAX, "%s/%s/", b->path, dir);
    need(n >= 0 && n < PATH_MAX, "board path too long");
    while (*name && n + 1 < PATH_MAX) {
        out[n++] = *name == '/' ? '-' : *name;
        name++;
    }
    need(!*name && (size_t)n + strlen(suffix) < PATH_MAX, "board log name too long");
    strcpy(out + n, suffix);
}

static pid_t board_spawn(const char *action, const char *name, const char *log) {
    char executable[PATH_MAX];
#ifdef __APPLE__
    u32 size = sizeof(executable);
    need(!_NSGetExecutablePath(executable, &size), "cannot locate board executable");
#else
    ssize_t size = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    need(size > 0, "cannot locate board executable");
    executable[size] = 0;
#endif
    pid_t pid = fork();
    if (pid != 0) {
        return pid;
    }
    setsid();
    int input = open("/dev/null", O_RDONLY);
    int output = open(log, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (input < 0 || output < 0) {
        _exit(126);
    }
    dup2(input, STDIN_FILENO);
    dup2(output, STDOUT_FILENO);
    dup2(output, STDERR_FILENO);
    close(input);
    close(output);
    execl(executable, executable, "board", action, name, (char *)NULL);
    _exit(127);
}

static void board_serving(Board *b) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/dispatcher.lock", b->path);
    int fd = lock_file(path, 1, 0);
    if (fd >= 0) {
        close(fd);
        snprintf(path, sizeof(path), "%s/dispatcher.log", b->path);
        need(board_spawn("serve", NULL, path) > 0, "cannot start board dispatcher");
    }
}

static void board_stop(Board *b, BoardJob *j, int again, const char *why) {
    board_exec(b,
        "UPDATE jobs SET state='stopping', stopping=%f, again=%d, why=%Q "
        "WHERE name=%Q AND state='running'",
        board_time(), again, why, j->name);
    if (sqlite3_changes(b->db) && j->pid > 0 &&
        board_process(j->pid, NULL).born == job_integer(j->row, "born")) {
        kill(-j->pid, SIGTERM);
    }
    j->state = "stopping";
}

static void board_follow(Board *b, BoardJob *j, double stamp) {
    if (j->adopted) {
        BoardUsage u = board_process(j->pid, NULL);
        if (u.born && u.born == job_integer(j->row, "born")) {
            board_exec(
                b, "UPDATE jobs SET peak_gb=max(peak_gb,%f) WHERE name=%Q", u.peak / 1e9, j->name);
        } else {
            board_exec(b,
                "UPDATE jobs SET state='done', ended=%f, pid=NULL,"
                " why='adopted process ended' WHERE name=%Q",
                stamp, j->name);
            j->state = "done";
        }
    } else if (!strcmp(j->state, "stopping") && stamp - job_number(j->row, "stopping") > 30 &&
        j->pid > 0) {
        if (board_process(j->pid, NULL).born == job_integer(j->row, "born")) {
            kill(-j->pid, SIGKILL);
        }
    } else if (!strcmp(j->state, "running") && job_integer(j->row, "wrapper") &&
        !board_process(job_integer(j->row, "wrapper"), NULL).born) {
        const char *state = job_integer(j->row, "restartable") && job_integer(j->row, "tries") < 3
            ? "waiting"
            : "lost";
        board_exec(b,
            "UPDATE jobs SET state=%Q, ended=%f, pid=NULL, why='its wrapper vanished'"
            " WHERE name=%Q AND state='running'",
            state, stamp, j->name);
        j->state = "lost";
    } else if (!strcmp(j->state, "starting") && stamp - job_number(j->row, "started") > 30) {
        board_exec(b,
            "UPDATE jobs SET state='lost', ended=%f, why='never started'"
            " WHERE name=%Q AND state='starting'",
            stamp, j->name);
        j->state = "lost";
    }
}

static int board_strangers(BoardJob *jobs, int n, double *rss, int show) {
    pid_t lister;
    FILE *ps = spawn_read((char *[]){"ps", "-Ao", "pid=,command=", NULL}, 0, &lister);
    if (!ps) {
        return 0;
    }
    char *line = NULL;
    size_t cap = 0;
    int count = 0;
    *rss = 0;
    while (getline(&line, &cap, ps) >= 0) {
        char *command;
        int pid = strtol(line, &command, 10);
        while (*command == ' ') {
            command++;
        }
        size_t executable = strcspn(command, " \t\n");
        const char *native = strstr(command, "runtime/bin/herculex-");
        int internal = native && (size_t)(native - command) < executable;
        if (pid <= 0 || pid == getpid() || !internal || strstr(command, " board ") ||
            strstr(command, " runner.job_board ")) {
            continue;
        }
        int known = 0;
        for (int i = 0; i < n && !known; i++) {
            BoardJob *j = &jobs[i];
            if (board_active(j->state)) {
                known = pid == j->pid || pid == job_integer(j->row, "wrapper") ||
                    (!j->adopted && j->pid > 0 && getpgid(pid) == j->pid);
            }
        }
        if (known) {
            continue;
        }
        double footprint = board_process(pid, NULL).rss / 1e9;
        *rss += footprint;
        count += footprint >= 0.5;
        if (show && footprint >= 0.05) {
            command[strcspn(command, "\n")] = 0;
            printf("  off board %-8d%5.1f GB  %.90s\n", pid, footprint, command);
        }
    }
    free(line);
    spawn_close(ps, lister);
    return count;
}

static void board_relieve(Board *b, BoardJob *jobs, int n, BoardMachine m) {
    if (m.swap < board_swap_limit(m, 0.30, 4.9)) {
        return;
    }
    int top = INT_MIN;
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        if (!strcmp(j->state, "stopping")) {
            return;
        }
        if (!strcmp(j->state, "running") && !j->adopted && j->priority > top) {
            top = j->priority;
        }
    }
    BoardJob *victim = NULL, *over = NULL;
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        if (strcmp(j->state, "running") != 0 || j->adopted || !job_integer(j->row, "restartable")) {
            continue;
        }
        if (j->priority < top &&
            (!victim || j->priority < victim->priority ||
                (j->priority == victim->priority && j->rss > victim->rss))) {
            victim = j;
        }
        double extra = j->rss - j->memory;
        if (extra > fmax(1, j->memory / 4) && (!over || extra > over->rss - over->memory)) {
            over = j;
        }
    }
    char why[160];
    if (!victim && over) {
        victim = over;
        double peak = fmax(over->peak, over->rss);
        board_exec(b, "UPDATE jobs SET memory_gb=%.1f WHERE name=%Q", peak, over->name);
        snprintf(why, sizeof(why),
            "swap %.1f GB with its footprint %.1f GB over its %g GB; "
            "reservation raised to its peak",
            m.swap, peak, over->memory);
    } else {
        snprintf(why, sizeof(why), "swap %.1f GB", m.swap);
    }
    if (victim) {
        board_stop(b, victim, 1, why);
    }
}

static int board_feeds_gpu(BoardJob *j, BoardJob *jobs, int n, int depth) {
    if ((j->slots & 1) && !j->share) {
        return 1;
    }
    for (int i = 0; depth && i < n; i++) {
        BoardJob *other = &jobs[i];
        if (!board_waiting(other)) {
            continue;
        }
        for (int d = 0; d < other->after.count; d++) {
            if (!strcmp(other->after.item[d], j->name) &&
                board_feeds_gpu(other, jobs, n, depth - 1)) {
                return 1;
            }
        }
    }
    return 0;
}

static int board_lower(BoardJob *j, BoardJob *other) {
    return !other || j->priority < other->priority ||
        (j->priority == other->priority &&
            job_number(j->row, "started") > job_number(other->row, "started"));
}

static void board_make_way(Board *b, BoardJob *jobs, int n, double room) {
    static double last;
    if (last && now() - last < 60) {
        return;
    }
    BoardJob *first = NULL;
    double used_cores = 0;
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        if (!strcmp(j->state, "stopping")) {
            return;
        }
        if (board_active(j->state)) {
            used_cores += j->cores;
        }
        if (!first && board_waiting(j) &&
            (strstr(j->reason, "cores,") || strstr(j->reason, "GB, room")) &&
            board_feeds_gpu(j, jobs, n, 2)) {
            first = j;
        }
    }
    if (!first) {
        return;
    }
    double short_cores = used_cores + first->cores - b->cpu;
    double short_gb = first->memory - room;
    double candidate_cores = 0, candidate_gb = 0;
    BoardJob *any = NULL, *enough = NULL;
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        if (strcmp(j->state, "running") != 0 || j->adopted || j->priority >= first->priority ||
            !job_integer(j->row, "restartable") ||
            (!j->share && (!(first->slots & 1) || first->share))) {
            continue;
        }
        double held = fmax(j->rss, j->memory);
        candidate_gb += held;
        candidate_cores += j->cores;
        if (board_lower(j, any)) {
            any = j;
        }
        if (j->cores >= short_cores && held >= short_gb && board_lower(j, enough)) {
            enough = j;
        }
    }
    if (any && candidate_cores >= short_cores && candidate_gb >= short_gb) {
        char why[PATH_MAX];
        snprintf(why, sizeof(why), "made way for %s", first->name);
        board_stop(b, enough ? enough : any, 2, why);
        last = now();
    }
}

static void board_dispatch(
    Board *b, BoardJob *jobs, int n, BoardMachine m, double room, int launch, int strangers) {
    BoardJob *first = NULL;
    double ceiling = room;
    for (int i = 0; i < n; i++) {
        if (board_active(jobs[i].state)) {
            ceiling += jobs[i].memory;
        }
    }
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        if (!board_waiting(j)) {
            continue;
        }
        double free_room = room;
        if (first && j->priority < first->priority) {
            double available = ceiling - first->memory;
            for (int k = 0; k < n; k++) {
                BoardJob *other = &jobs[k];
                if (other->priority < first->priority &&
                    (board_active(other->state) ||
                        (k < i && board_waiting(other) && !*other->reason))) {
                    available -= other->memory;
                }
            }
            free_room = fmin(free_room, available);
        }
        board_decide(b, j, jobs, n, m, free_room, strangers);
        if (!*j->reason) {
            room -= j->memory;
            if (launch == 2) {
                j->state = "running";
                j->scheduled = 1;
            } else if (launch == 1) {
                board_exec(b,
                    "UPDATE jobs SET state='starting', started=%f, wrapper=NULL, why=NULL "
                    "WHERE name=%Q AND state='waiting' AND held=0",
                    board_time(), j->name);
                if (sqlite3_changes(b->db)) {
                    if (board_spawn("wrap", j->name, "/dev/null") > 0) {
                        j->state = "starting";
                    } else {
                        board_exec(b,
                            "UPDATE jobs SET state='lost', ended=%f,"
                            " why='cannot launch wrapper' WHERE name=%Q",
                            board_time(), j->name);
                        j->state = "lost";
                    }
                }
            }
        } else {
            if (!first && strstr(j->reason, "GB, room")) {
                first = j;
            }
            if (launch == 1 && strcmp(j->reason, job_text(j->row, "why")) != 0) {
                board_exec(b, "UPDATE jobs SET why=%Q WHERE name=%Q AND state='waiting'", j->reason,
                    j->name);
            }
        }
    }
}

static void board_sample(Board *b, BoardJob *jobs, int n, BoardMachine m, double room) {
    int waiting = 0, running = 0, slots[3] = {0};
    for (int i = 0; i < n; i++) {
        running += board_active(jobs[i].state);
        if (board_waiting(&jobs[i])) {
            waiting++;
            for (int s = 0; s < 3; s++) {
                slots[s] += !!(jobs[i].slots & (1 << s));
            }
        }
    }
    BoardCounters c = {.at = board_time(), .mb = {NAN, NAN, NAN}}, p = b->previous;
    board_counters(b, &c);
    b->previous = c;
    double span = c.at - p.at, busy = c.busy - p.busy, idle = c.idle - p.idle;
    sqlite3_stmt *q = board_query(b,
        "INSERT OR REPLACE INTO samples(time,gpu,room_gb,swap_gb,swapped_out_gb,"
        "load,running,waiting,gpu_waiting,engine_waiting,link_waiting,"
        "cpu_pct,net_mb_s,internal_mb_s,library_mb_s,outputs_mb_s) "
        "VALUES (%f,%d,%f,%f,%f,%f,%d,%d,%d,%d,%d,?,?,?,?,?)",
        c.at, m.gpu, room, m.swap, m.swapped, m.load, running, waiting, slots[0], slots[1],
        slots[2]);
    if (p.at && span > 0) {
        if (busy + idle > 0) {
            sqlite3_bind_double(q, 1, 100 * busy / (busy + idle));
        }
        sqlite3_bind_double(q, 2, (c.received - p.received) / span / 1e6);
        for (int i = 0; i < 3; i++) {
            if (isfinite(c.mb[i]) && isfinite(p.mb[i])) {
                sqlite3_bind_double(q, i + 3, (c.mb[i] - p.mb[i]) / span);
            }
        }
    }
    need(sqlite3_step(q) == SQLITE_DONE, "board: %s", sqlite3_errmsg(b->db));
    sqlite3_finalize(q);
}

static void board_serve(Board *b) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/dispatcher.lock", b->path);
    int fd = lock_file(path, 1, 0);
    if (fd < 0) {
        return;
    }
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    need(ftruncate(fd, 0) == 0, "cannot reset %s", path);
    dprintf(fd, "%d\n", getpid());
    double sampled = 0;
    while (exists(path)) {
        int n;
        BoardJob *jobs = board_jobs(b, &n);
        BoardMachine m = board_probe();
        for (int i = 0; i < n; i++) {
            if (board_active(jobs[i].state)) {
                board_follow(b, &jobs[i], board_time());
            }
        }
        board_relieve(b, jobs, n, m);
        double room = board_room(jobs, n, m.room - b->margin);
        int strangers = 0;
        for (int i = 0; i < n; i++) {
            if (board_waiting(&jobs[i]) && job_integer(jobs[i].row, "alone")) {
                double rss;
                strangers = board_strangers(jobs, n, &rss, 0);
                break;
            }
        }
        board_dispatch(b, jobs, n, m, room, 1, strangers);
        board_make_way(b, jobs, n, room);
        if (board_time() - sampled >= 60) {
            sampled = board_time();
            board_sample(b, jobs, n, m, room);
        }
        board_free(jobs, n);
        while (waitpid(-1, NULL, WNOHANG) > 0) {
        }
        board_pause(b->tick);
    }
    close(fd);
}

typedef struct {
    int pid;
    BoardUsage usage;
} BoardSeen;

static const char *board_json(json_object *o) {
    return o
        ? json_object_to_json_string_ext(o, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE)
        : NULL;
}

static void board_wrap(Board *b, const char *name) {
    board_exec(b,
        "UPDATE jobs SET wrapper=%d WHERE name=%Q AND state IN ('starting','waiting') "
        "AND coalesce(wrapper,0)=0",
        getpid(), name);
    need(sqlite3_changes(b->db), "%s already has a wrapper or is no longer waiting", name);
    BoardRow *job = board_named(b, name);
    json_object *command = json_tokener_parse(job_text(job, "command"));
    json_object *settings = json_tokener_parse(job_text(job, "settings"));
    int length = board_count(command);
    need(length && json_object_is_type(settings, json_type_object), "invalid command for %s", name);
    char log[PATH_MAX], events[PATH_MAX], scratch[PATH_MAX], stamp[64], niceness[32];
    board_path(log, b, "logs", name, ".log");
    board_path(events, b, "events", name, ".jsonl");
    mkparent(events);
    snprintf(scratch, sizeof(scratch), "%s", env("HERCULEX_SCRATCH", ""));
    if (!*scratch) {
        snprintf(scratch, sizeof(scratch), "%s/runtime/scratch", HOME);
    }
    mkdirs(scratch);
    int fd = open(log, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    need(fd >= 0, "cannot open %s", log);
    char **args = calloc(length + 7, sizeof(*args));
    need(args != NULL, "cannot allocate command");
    int n = 0;
    int nice_value = job_integer(job, "nice");
    if (nice_value) {
        snprintf(niceness, sizeof(niceness), "%d", nice_value);
        args[n++] = "nice";
        args[n++] = "-n";
        args[n++] = niceness;
    }
    if (job_integer(job, "background")) {
#ifdef __APPLE__
        args[n++] = "taskpolicy";
        args[n++] = "-b";
#else
        args[n++] = "nice";
        args[n++] = "-n";
        args[n++] = "19";
#endif
    }
    for (int i = 0; i < length; i++) {
        args[n++] = (char *)board_item(command, i);
    }
    board_utc(stamp, board_time(), "%Y-%m-%d %H:%M:%S");
    dprintf(fd, "\n--- %s UTC, try %lld:", stamp, (long long)job_integer(job, "tries") + 1);
    for (int i = 0; i < n; i++) {
        dprintf(fd, " %s", args[i]);
    }
    dprintf(fd, "\n");
    int channel[2];
    need(!socketpair(AF_UNIX, SOCK_STREAM, 0, channel), "cannot open launch channel");
    pid_t pid = fork();
    need(pid >= 0, "cannot fork job %s", name);
    if (!pid) {
        close(channel[0]);
        char go;
        if (setsid() < 0 || write(channel[1], "Y", 1) != 1 || read(channel[1], &go, 1) != 1) {
            _exit(126);
        }
        close(channel[1]);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
        if (chdir(job_text(job, "folder"))) {
            dprintf(
                STDERR_FILENO, "cannot enter %s: %s\n", job_text(job, "folder"), strerror(errno));
            _exit(126);
        }
        unsetenv("HERCULEX_NATIVE_BIN");
        setenv("HERCULEX_EVENTS", events, 1);
        setenv("MallocLargeCache", "0", 1);
        setenv("TMPDIR", scratch, 1);
        json_object_object_foreach(settings, key, value) {
            setenv(key, json_object_get_string(value), 1);
        }
        execvp(args[0], args);
        dprintf(STDERR_FILENO, "cannot execute %s: %s\n", args[0], strerror(errno));
        _exit(127);
    }
    close(channel[1]);
    char ready;
    ssize_t launch;
    do {
        launch = read(channel[0], &ready, 1);
    } while (launch < 0 && errno == EINTR);
    close(fd);
    free(args);
    json_object_put(command);
    json_object_put(settings);
    BoardUsage first = board_process(pid, NULL);
    board_exec(b,
        "UPDATE jobs SET state='running', pid=%d,born=%lld,wrapper=%d,started=%f,"
        "tries=tries+1,why=NULL WHERE name=%Q",
        pid, (long long)first.born, getpid(), board_time(), name);
    if (launch == 1) {
        signal(SIGPIPE, SIG_IGN);
        ssize_t sent;
        do {
            sent = write(channel[0], "Y", 1);
        } while (sent < 0 && errno == EINTR);
    }
    close(channel[0]);
    double peak = job_number(job, "peak_gb"), used_cores = 0, looked = now();
    int status = 0, seen = 0, cap = 0;
    BoardSeen *processes = NULL;
    struct rusage spent = {0};
    for (;;) {
        pid_t finished = wait4(pid, &status, WNOHANG, &spent);
        if (finished < 0 && errno == EINTR) {
            continue;
        }
        need(finished >= 0, "cannot wait for %s: %s", name, strerror(errno));
        if (finished) {
            break;
        }
        int pids[1024];
        int count = board_group(pid, pids, 1024);
        double rss = 0, cpu = 0, stamp_now = now();
        int already_seen = seen;
        for (int p = 0; p < count; p++) {
            BoardUsage u = board_process(pids[p], NULL);
            if (!u.born) {
                continue;
            }
            rss += u.rss / 1e9;
            peak = fmax(peak, u.peak / 1e9);
            int i = 0;
            while (i < seen && processes[i].pid != pids[p]) {
                i++;
            }
            if (i < seen) {
                cpu += fmax(0, u.cpu - processes[i].usage.cpu);
            } else {
                if (seen == cap) {
                    cap = cap ? 2 * cap : 16;
                    void *grown = realloc(processes, cap * sizeof(*processes));
                    need(grown != NULL, "cannot track job processes");
                    processes = grown;
                }
                seen++;
                cpu += already_seen ? u.cpu : 0;
            }
            processes[i] = (BoardSeen){pids[p], u};
        }
        peak = fmax(peak, rss);
        if (already_seen && stamp_now > looked) {
            used_cores = 0.7 * used_cores + 0.3 * fmin(b->cpu, cpu / (stamp_now - looked));
        }
        looked = stamp_now;
        board_exec(
            b, "UPDATE jobs SET peak_gb=%f, cores_used=%f WHERE name=%Q", peak, used_cores, name);
        board_pause(fmin(b->tick, 2));
    }
    kill(-pid, SIGTERM);
    board_exec(b, "BEGIN IMMEDIATE");
    free(job);
    job = board_named(b, name);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
    const char *state = code ? "failed" : "done";
    double stamp_now = board_time(), rest = 0;
    int tries = job_integer(job, "tries");
    if (code == 75 && !strcmp(job_text(job, "state"), "running") && tries < 60) {
        state = "waiting";
        rest = stamp_now + b->retry;
    }
    if (!strcmp(job_text(job, "state"), "stopping")) {
        int again = job_integer(job, "again");
        state = again == 2 || (again && tries < 3) ? "waiting" : "stopped";
        rest = !strcmp(state, "waiting") ? stamp_now + b->retry : 0;
        json_object *clean = json_tokener_parse(job_text(job, "clean"));
        for (int i = 0; i < board_count(clean); i++) {
            char path[PATH_MAX];
            const char *item = board_item(clean, i);
            snprintf(
                path, sizeof(path), "%s/%s", *item == '/' ? "" : job_text(job, "folder"), item);
            remove_tree(path);
        }
        json_object_put(clean);
    }
    double read_gb = 0, written_gb = 0;
    for (int i = 0; i < seen; i++) {
        read_gb += processes[i].usage.read / 1e9;
        written_gb += processes[i].usage.written / 1e9;
    }
    free(processes);
    double cpu_hours = (spent.ru_utime.tv_sec + spent.ru_utime.tv_usec * 1e-6 +
                           spent.ru_stime.tv_sec + spent.ru_stime.tv_usec * 1e-6) /
        3600;
    board_exec(b,
        "UPDATE jobs SET state=%Q,not_before=%f,ended=%f,code=%d,peak_gb=%f,"
        "cpu_hours=cpu_hours+%f,read_gb=read_gb+%f,written_gb=written_gb+%f,pid=NULL WHERE name=%Q",
        state, rest, stamp_now, code, peak, cpu_hours, read_gb, written_gb, name);
    board_exec(b, "COMMIT");
    free(job);
    board_serving(b);
}

static double board_value(const char *s) {
    char *end;
    double n = strtod(s, &end);
    need(end != s && !*end && isfinite(n), "invalid board number %s", s);
    return n;
}

static int board_int(const char *s) {
    char *end;
    errno = 0;
    long value = strtol(s, &end, 10);
    need(end != s && !*end && !errno && value >= INT_MIN && value <= INT_MAX,
        "invalid board integer %s", s);
    return (int)value;
}

static const char *board_kind(json_object *command) {
    for (int i = 0; i < board_count(command); i++) {
        const char *word = board_item(command, i);
        int dots = 0, letters = 0;
        for (; *word; word++) {
            if (*word == '.' && letters) {
                dots++;
                letters = 0;
            } else if ((*word >= 'a' && *word <= 'z') || *word == '_') {
                letters++;
            } else {
                break;
            }
        }
        if (!*word && letters && dots) {
            return board_item(command, i);
        }
    }
    const char *word = board_item(command, 0);
    const char *base = strrchr(word, '/');
    return base ? base + 1 : word;
}

static void board_insert(Board *b, int argc, char **argv, int adopted) {
    need(argc > adopted, "board %s needs %sNAME", adopted ? "adopt" : "add", adopted ? "PID " : "");
    const char *name = argv[adopted], *kind = NULL, *owner = NULL, *note = NULL, *folder = HOME;
    double memory = NAN, cores_wanted = 1, free_gib = 0;
    int priority = 0, nice_value = INT_MIN, background = 0, restartable = 0;
    int alone = 0, last = 0, share = 0, held = 0, slots[3] = {0};
    json_object *after = json_object_new_array(), *needs = json_object_new_array();
    json_object *clean = json_object_new_array(), *settings = json_object_new_object();
    json_object *command = json_object_new_array(), *at_most = NULL;
    enum { FLAG, LIST, SETTING, TEXT, NUMBER, INTEGER, LIMIT };
    const struct {
        const char *name;
        int type;
        void *target;
    } options[] = {{"--gpu", FLAG, slots}, {"--engine", FLAG, slots + 1},
        {"--link", FLAG, slots + 2}, {"--background", FLAG, &background},
        {"--restartable", FLAG, &restartable}, {"--alone", FLAG, &alone}, {"--last", FLAG, &last},
        {"--share", FLAG, &share}, {"--held", FLAG, &held}, {"--after", LIST, after},
        {"--needs", LIST, needs}, {"--clean", LIST, clean}, {"--setting", SETTING, settings},
        {"--kind", TEXT, &kind}, {"--owner", TEXT, &owner}, {"--note", TEXT, &note},
        {"--folder", TEXT, &folder}, {"--memory", NUMBER, &memory},
        {"--cores", NUMBER, &cores_wanted}, {"--free", NUMBER, &free_gib},
        {"--priority", INTEGER, &priority}, {"--nice", INTEGER, &nice_value},
        {"--at-most", LIMIT, NULL}, {NULL, -1, NULL}};
    for (int i = adopted + 1; i < argc; i++) {
        const char *word = argv[i];
        if (!strcmp(word, "--")) {
            for (i++; i < argc; i++) {
                json_object_array_add(command, json_object_new_string(argv[i]));
            }
            break;
        }
        size_t k = 0;
        while (options[k].name && strcmp(word, options[k].name) != 0)
            k++;
        int type = options[k].type;
        void *target = options[k].target;
        if (type == FLAG) {
            *(int *)target = 1;
            continue;
        }
        if (type == LIST || type == SETTING) {
            while (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
                const char *item = argv[++i];
                if (type == SETTING) {
                    const char *eq = strchr(item, '=');
                    need(eq && item != eq, "invalid setting %s: expected KEY=VALUE", item);
                    char *key = strndup(item, eq - item);
                    need(key != NULL, "cannot allocate setting");
                    json_object_object_add(target, key, json_object_new_string(eq + 1));
                    free(key);
                } else {
                    json_object_array_add(target, json_object_new_string(item));
                }
            }
            continue;
        }
        need(++i < argc, "%s needs a value", word);
        const char *value = argv[i];
        switch (type) {
        case TEXT: *(const char **)target = value; break;
        case NUMBER: *(double *)target = board_value(value); break;
        case INTEGER: *(int *)target = board_int(value); break;
        case LIMIT:
            need(++i < argc, "--at-most needs N PATTERN");
            json_object_put(at_most);
            at_most = json_object_new_array();
            json_object_array_add(at_most, json_object_new_int(board_int(value)));
            json_object_array_add(at_most, json_object_new_string(argv[i]));
            break;
        default: die("unknown board option %s", word);
        }
    }
    int pid = adopted ? board_int(argv[0]) : 0;
    BoardUsage usage = {0};
    if (adopted) {
        need(pid > 0, "invalid process %d", pid);
        usage = board_process(pid, NULL);
        need(usage.born, "no process %d", pid);
        char number[32], line[4096];
        snprintf(number, sizeof(number), "%d", pid);
        pid_t lister;
        FILE *ps = spawn_read((char *[]){"ps", "-o", "command=", "-p", number, NULL}, 0, &lister);
        if (ps) {
            if (fgets(line, sizeof(line), ps)) {
                char *save;
                for (char *s = strtok_r(line, " \t\n", &save); s;
                    s = strtok_r(NULL, " \t\n", &save)) {
                    json_object_array_add(command, json_object_new_string(s));
                }
            }
            spawn_close(ps, lister);
        }
        if (!board_count(command)) {
            json_object_array_add(command, json_object_new_string("process"));
        }
    }
    need(board_count(command), "add needs a command after --");
    kind = kind ? kind : board_kind(command);
    if (adopted) {
        memory = 0;
    } else if (isnan(memory)) {
        sqlite3_stmt *q = board_query(b,
            "SELECT max(peak_gb) FROM (SELECT peak_gb FROM jobs "
            "WHERE kind=%Q AND state='done' AND peak_gb>0 ORDER BY ended DESC LIMIT 3)",
            kind);
        memory = sqlite3_step(q) == SQLITE_ROW && sqlite3_column_type(q, 0) != SQLITE_NULL
            ? round(sqlite3_column_double(q, 0) * 110) / 100
            : 2;
        sqlite3_finalize(q);
    }
    if (nice_value == INT_MIN) {
        nice_value = last || adopted ? 0 : 5;
    }
    char resolved[PATH_MAX];
    if (!realpath(folder, resolved)) {
        if (*folder == '/') {
            snprintf(resolved, sizeof(resolved), "%s", folder);
        } else {
            char cwd[PATH_MAX];
            need(getcwd(cwd, sizeof(cwd)) != NULL, "cannot find current directory");
            snprintf(resolved, sizeof(resolved), "%s/%s", cwd, folder);
        }
    }
    double stamp = board_time();
    sqlite3_stmt *q = board_query(b,
        "INSERT INTO jobs(name,kind,command,folder,settings,owner,note,memory_gb,cores,gpu,engine,"
        "link,priority,nice,background,after,needs,clean,restartable,alone,last,free_gib,at_most,"
        "share,state,held,added,adopted,pid,born,started,peak_gb) VALUES "
        "(%Q,%Q,%Q,%Q,%Q,%Q,%Q,%f,%f,%d,%d,%d,%d,%d,%d,%Q,%Q,%Q,%d,%d,%d,%f,%Q,%d,%Q,%d,"
        "%f,%d,NULLIF(%d,0),NULLIF(%lld,0),NULLIF(%f,0),%f)",
        name, kind, board_json(command), resolved, board_json(settings), owner, note, memory,
        cores_wanted, slots[0], slots[1], slots[2], priority, nice_value, background,
        board_json(after), board_json(needs), board_json(clean), restartable, alone, last, free_gib,
        board_json(at_most), share, adopted ? "running" : "waiting", held, stamp, adopted, pid,
        (long long)usage.born, adopted ? stamp : 0.0, usage.peak / 1e9);
    int rc = sqlite3_step(q);
    need(rc != SQLITE_CONSTRAINT, "a job named %s is already on the board", name);
    need(rc == SQLITE_DONE, "board: %s", sqlite3_errmsg(b->db));
    sqlite3_finalize(q);
    if (adopted) {
        printf("%s: process %d adopted, %.1f GB now\n", name, pid, usage.rss / 1e9);
    } else {
        printf("%s: %s, %.1f GB, waiting\n", name, kind, memory);
    }
    json_object_put(after);
    json_object_put(needs);
    json_object_put(clean);
    json_object_put(settings);
    json_object_put(command);
    json_object_put(at_most);
    board_serving(b);
}

static void board_duration(char text[64], double seconds) {
    int minutes = fmax(0, seconds) / 60;
    snprintf(text, 64, "%dh%02dm", minutes / 60, minutes % 60);
}

static void board_slot_text(const BoardJob *j, char text[64]) {
    *text = 0;
    for (int s = 0; s < 3; s++) {
        if (j->slots & (1 << s)) {
            size_t length = strlen(text);
            snprintf(text + length, 64 - length, "%s%s%s", length ? "+" : "", board_slots[s],
                j->share && s != 1 ? "~" : "");
        }
    }
    if (!*text) {
        strcpy(text, "-");
    }
}

static int board_finished(const char *state) {
    return !strcmp(state, "done") || !strcmp(state, "failed") || !strcmp(state, "stopped") ||
        !strcmp(state, "lost") || !strcmp(state, "cancelled");
}

static int board_display_order(const void *aa, const void *bb) {
    const BoardJob *a = aa, *b = bb;
    int ar = board_active(a->state) ? 0 : board_waiting(a) ? 1 : 2;
    int br = board_active(b->state) ? 0 : board_waiting(b) ? 1 : 2;
    if (ar != br) {
        return ar - br;
    }
    if (ar == 1 && a->priority != b->priority) {
        return a->priority > b->priority ? -1 : 1;
    }
    double at = ar == 2 ? job_number(a->row, "ended") : a->added;
    double bt = br == 2 ? job_number(b->row, "ended") : b->added;
    return (at > bt) - (at < bt);
}

static void board_status(Board *b, int all) {
    int n;
    BoardJob *jobs = board_jobs(b, &n);
    BoardMachine m = board_probe();
    double room = board_room(jobs, n, m.room - b->margin), rss = 0, stamp = board_time();
    int strangers = board_strangers(jobs, n, &rss, 0);
    char text[64];
    board_utc(text, stamp, "%H:%M");
    printf("%s UTC  room %.1f GB  off board %.1f GB  swap %.1f GB  "
           "GPU %d%%, display %d%%  load %.1f  free",
        text, room, rss, m.swap, m.device, m.device - m.gpu, m.load);
    const char *paths[] = {"/", LIBRARY, OUTPUTS};
    const char *roles[] = {"internal", "library", "outputs"};
    for (int i = 0; i < 3; i++) {
        double free_space = board_disc(paths[i]);
        if (free_space >= 0) {
            printf("%s %s %.0f GiB", i ? "," : "", roles[i], free_space);
        }
    }
    puts("");
    board_dispatch(b, jobs, n, m, room, 0, strangers);
    if (n > 1) {
        qsort(jobs, n, sizeof(*jobs), board_display_order);
    }
    int waiting = 0, ended = 0;
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        BoardRow *r = j->row;
        char slots[64];
        board_slot_text(j, slots);
        double started = job_number(r, "started"), ended_at = job_number(r, "ended");
        if (board_active(j->state)) {
            board_duration(text, stamp - started);
            printf("  %-10s%-34s%-12s%5.1f GB, peak %.1f  %s%s", j->state, j->name, slots, j->rss,
                j->peak, text, j->adopted ? "  adopted" : "");
            double declared = job_number(r, "cores");
            if (j->cores > declared + 0.5) {
                printf("  using %.1f cores, declared %g", j->cores, declared);
            }
            puts("");
        } else if (board_waiting(j)) {
            if (all || waiting < 12) {
                printf("  %-10s%-34s%-12s%5.1f GB  %s\n", j->state, j->name, slots, j->memory,
                    *j->reason ? j->reason : "starting");
            }
            waiting++;
        } else if (board_finished(j->state)) {
            if (!ended++ && !all && waiting > 12) {
                printf("  ... and %d more waiting: status --all\n", waiting - 12);
            }
            if (!all && i < n - 6) {
                continue;
            }
            board_duration(text, started ? (ended_at ? ended_at : stamp) - started : 0);
            const char *why = job_text(r, "why");
            printf("  %-10s%-34s%-12speak %.1f GB  %s  cpu %.2f h  read %.1f wrote %.1f GB%s%s\n",
                j->state, j->name, slots, j->peak, text, job_number(r, "cpu_hours"),
                job_number(r, "read_gb"), job_number(r, "written_gb"), *why ? "  " : "", why);
        }
    }
    if (!ended && !all && waiting > 12) {
        printf("  ... and %d more waiting: status --all\n", waiting - 12);
    }
    board_strangers(jobs, n, &rss, 1);
    board_free(jobs, n);
}

static int board_log(Board *b, const char *name, const char *lines) {
    char path[PATH_MAX];
    board_path(path, b, "logs", name, ".log");
    if (!exists(path)) {
        printf("%s has no log yet\n", name);
        return 0;
    }
    char count[32];
    int n = lines ? board_int(lines) : 30;
    if (n <= 0) {
        snprintf(count, sizeof(count), "+%lld", 1LL - n);
    } else {
        snprintf(count, sizeof(count), "%d", n);
    }
    execlp("tail", "tail", "-n", count, path, (char *)NULL);
    die("cannot read %s: %s", path, strerror(errno));
    return 1;
}

static void board_change(Board *b, const char *name, int argc, char **argv) {
    need(argc > 0 && !(argc % 2),
        "change needs --priority, --cores, --share, "
        "--restartable or --owner");
    board_exec(b, "BEGIN IMMEDIATE");
    for (int i = 0; i < argc; i += 2) {
        const char *key = argv[i], *value = argv[i + 1];
        if (!strcmp(key, "--owner")) {
            board_exec(b, "UPDATE jobs SET owner=%Q WHERE name=%Q", value, name);
        } else if (!strcmp(key, "--cores")) {
            board_exec(b, "UPDATE jobs SET cores=%f WHERE name=%Q", board_value(value), name);
        } else if (!strcmp(key, "--priority") || !strcmp(key, "--share") ||
            !strcmp(key, "--restartable")) {
            int v = board_int(value);
            need(!strcmp(key, "--priority") || v == 0 || v == 1, "%s needs 0 or 1", key);
            board_exec(b, "UPDATE jobs SET %s=%d WHERE name=%Q", key + 2, v, name);
        } else {
            die("unknown board option %s", key);
        }
    }
    board_exec(b, "COMMIT");
}

static int board_action(Board *b, const char *action, int argc, char **argv) {
    need(argc > 0, "board %s needs NAME", action);
    const char *name = argv[0];
    if (!strcmp(action, "log")) {
        return board_log(b, name, argc > 1 ? argv[1] : NULL);
    }
    BoardRow *job = board_named(b, name);
    const char *state = job_text(job, "state");
    int adopted = job_integer(job, "adopted"), result = 0;
    if (!strcmp(action, "hold")) {
        board_exec(b, "UPDATE jobs SET held=1 WHERE name=%Q", name);
    } else if (!strcmp(action, "change")) {
        board_change(b, name, argc - 1, argv + 1);
    } else if (!strcmp(action, "resize") && (board_active(state) || !strcmp(state, "waiting"))) {
        need(argc == 2, "board resize needs NAME GB");
        board_exec(b, "UPDATE jobs SET memory_gb=%f WHERE name=%Q", board_value(argv[1]), name);
    } else if (!strcmp(action, "release")) {
        board_exec(b, "UPDATE jobs SET held=0,not_before=0 WHERE name=%Q", name);
        board_serving(b);
    } else if (!strcmp(action, "cancel") && !strcmp(state, "running") && !adopted) {
        BoardJob j = {.name = name, .row = job, .pid = job_integer(job, "pid")};
        board_stop(b, &j, 0, "cancelled");
    } else if (!strcmp(action, "cancel") && !strcmp(state, "waiting")) {
        board_exec(b,
            "UPDATE jobs SET state='cancelled',ended=%f WHERE name=%Q AND state='waiting'",
            board_time(), name);
    } else if (!strcmp(action, "retry") && board_finished(state)) {
        board_exec(b,
            "UPDATE jobs SET state='waiting',tries=0,not_before=0,code=NULL,why=NULL "
            "WHERE name=%Q",
            name);
        board_serving(b);
    } else if (!strcmp(action, "remove") && board_finished(state)) {
        board_exec(b, "DELETE FROM jobs WHERE name=%Q", name);
    } else if (!strcmp(action, "wait")) {
        while (!board_finished(state)) {
            free(job);
            board_pause(5);
            job = board_named(b, name);
            state = job_text(job, "state");
        }
        BoardCell *code = job_cell(job, "code");
        printf("%s %s %s peak %.1f GB\n", name, state, code->text ? code->text : "None",
            job_number(job, "peak_gb"));
        result = strcmp(state, "done") ? (code->text ? (int)code->integer : 1) : 0;
    } else {
        die("cannot %s %s: it is %s%s", action, name, state, adopted ? ", adopted" : "");
    }
    free(job);
    return result;
}

static void board_owners_report(Board *b, double since, double stamp) {
    sqlite3_stmt *q = board_query(b,
        "WITH held AS (SELECT owner, "
        "max(0,coalesce(CASE WHEN state IN ('starting','running','stopping') "
        "THEN %f ELSE ended END,0)-max(started,%f))/3600 AS hours, gpu, "
        "max(cores,coalesce(cores_used,0)) AS cores FROM jobs WHERE owner<>'' AND started>0), "
        "owners AS (SELECT owner,sum(hours*gpu) AS gpu,sum(hours*cores) AS cores FROM held "
        "GROUP BY owner HAVING sum(hours*gpu)>0 OR sum(hours*cores)>0) "
        "SELECT owner,gpu,cores,max(length(owner)) "
        "OVER (),max(1e-9,sum(gpu) OVER ()),max(1e-9,sum(cores) OVER ()) FROM owners "
        "ORDER BY owner",
        stamp, since);
    while (sqlite3_step(q) == SQLITE_ROW) {
        double gpu = sqlite3_column_double(q, 1), cpu = sqlite3_column_double(q, 2);
        printf("  %-*sGPU %6.2f h (%3.0f%%)  cores %7.2f h (%3.0f%%)\n",
            sqlite3_column_int(q, 3) + 2, sqlite3_column_text(q, 0), gpu,
            gpu / sqlite3_column_double(q, 4) * 100, cpu, cpu / sqlite3_column_double(q, 5) * 100);
    }
    sqlite3_finalize(q);
}

static double board_slot_held(Board *b, const char *slot, double since, double stamp) {
    sqlite3_stmt *q = board_query(b,
        "SELECT max(started,%f),coalesce(ended,%f) FROM jobs "
        "WHERE %s=1 AND started>0 AND (ended IS NULL OR ended>%f) ORDER BY 1",
        since, stamp, slot, since);
    double reached = since, held = 0;
    while (sqlite3_step(q) == SQLITE_ROW) {
        double begin = sqlite3_column_double(q, 0), end = sqlite3_column_double(q, 1);
        held += fmax(0, end - fmax(reached, begin));
        reached = fmax(reached, end);
    }
    sqlite3_finalize(q);
    return 100 * held / (stamp - since);
}

static void board_usage_report(Board *b, double hours) {
    need(hours > 0, "board usage needs positive hours");
    double stamp = board_time(), since = stamp - hours * 3600;
    sqlite3_stmt *q = board_query(b,
        "SELECT count(*),avg(gpu>=50)*100,"
        "avg(gpu<50 AND gpu_waiting>0)*100,max(swap_gb),min(room_gb),"
        "(SELECT swapped_out_gb FROM samples ORDER BY time DESC LIMIT 1)-"
        "(SELECT swapped_out_gb FROM samples WHERE time>=%f ORDER BY time LIMIT 1),"
        "(SELECT count(*) FROM jobs WHERE ended>=%f AND state='done'),"
        "(SELECT sum(cpu_hours) FROM jobs WHERE ended>=%f AND state='done') "
        "FROM samples WHERE time>=%f",
        since, since, since, since);
    int samples = sqlite3_step(q) == SQLITE_ROW ? sqlite3_column_int(q, 0) : 0;
    if (!samples) {
        puts("no samples yet");
    } else {
        printf("%d minutes sampled: GPU busy %.0f%%, idle with GPU work waiting %.0f%%, "
               "swap peak %.1f GB, swapped out %.1f GB, least room %.1f GB, %d jobs done, "
               "%.1f CPU-hours\n",
            samples, sqlite3_column_double(q, 1), sqlite3_column_double(q, 2),
            sqlite3_column_double(q, 3), sqlite3_column_double(q, 5), sqlite3_column_double(q, 4),
            sqlite3_column_int(q, 6), sqlite3_column_double(q, 7));
        for (int s = 0; s < 3; s++) {
            printf("%s%s held %.0f%%", s ? ", " : "", board_slots[s],
                board_slot_held(b, board_slots[s], since, stamp));
        }
        puts("");
    }
    sqlite3_finalize(q);
    board_owners_report(b, since, stamp);
    if (!samples) {
        return;
    }
    const char *criteria[] = {"peak_gb>1.25*max(memory_gb,0.1) ORDER BY memory_gb-peak_gb",
        "state IN ('done','failed','stopped','lost','cancelled') AND peak_gb>0 "
        "AND memory_gb>max(2*peak_gb,peak_gb+2) ORDER BY peak_gb-memory_gb"};
    const char *headings[] = {"peaked over their reservation:",
        "reserved far more than they used (room others waited for):"};
    for (int i = 0; i < 2; i++) {
        q = board_query(b,
            "SELECT name,memory_gb,peak_gb FROM jobs WHERE started>=%f AND %s "
            "LIMIT 12",
            since, criteria[i]);
        int rows = 0;
        while (sqlite3_step(q) == SQLITE_ROW) {
            if (!rows++) {
                puts(headings[i]);
            }
            printf("  %-34sreserved %5.1f GB, peaked %5.1f GB\n", sqlite3_column_text(q, 0),
                sqlite3_column_double(q, 1), sqlite3_column_double(q, 2));
        }
        sqlite3_finalize(q);
    }
}

static double board_minutes(Board *b, BoardJob *j) {
    const char *dash = strrchr(j->name, '-');
    const char *kind = job_text(j->row, "kind");
    char *prefix = strndup(j->name, dash ? (size_t)(dash - j->name) : 0);
    need(prefix != NULL, "cannot allocate job kind prefix");
    for (int i = 0; i < 2; i++) {
        char *where = i ? sqlite3_mprintf("kind=%Q", kind)
            : dash      ? sqlite3_mprintf("substr(name,1,%d)=%Q AND substr(name,%d,1)='-' "
                                          "AND instr(substr(name,%d),'-')=0",
                              (int)(dash - j->name), prefix, (int)(dash - j->name) + 1,
                              (int)(dash - j->name) + 2)
                        : sqlite3_mprintf("instr(name,'-')=0");
        sqlite3_stmt *q = board_query(b,
            "WITH data AS (SELECT (ended-started)/60 AS t,"
            "memory_gb AS m FROM jobs WHERE state='done' AND ended>started AND %s), "
            "ranks AS (SELECT t,m,row_number() OVER (ORDER BY t) AS ti,"
            "row_number() OVER (ORDER BY m) AS mi,count(*) OVER () AS n FROM data) "
            "SELECT avg(CASE WHEN ti IN ((n+1)/2,(n+2)/2) THEN t END),"
            "avg(CASE WHEN mi IN ((n+1)/2,(n+2)/2) THEN m END) FROM ranks",
            where);
        sqlite3_free(where);
        if (sqlite3_step(q) == SQLITE_ROW && sqlite3_column_type(q, 0) != SQLITE_NULL) {
            double minutes = sqlite3_column_double(q, 0), memory = sqlite3_column_double(q, 1);
            sqlite3_finalize(q);
            free(prefix);
            return memory ? minutes * j->memory / memory : minutes;
        }
        sqlite3_finalize(q);
    }
    free(prefix);
    return 10;
}

typedef struct {
    double duration, left, start, end;
    int gate;
} BoardForecast;

static void board_forecast(Board *b, const char *name) {
    double stamp = board_time(), clock = 0;
    int n;
    BoardJob *jobs = board_jobs(b, &n);
    BoardMachine m = {0};
    double ceiling = board_room(jobs, n, board_probe().room - b->margin);
    BoardForecast *times = calloc(n ? n : 1, sizeof(*times));
    need(times != NULL, "cannot allocate board forecast");
    for (int i = 0; i < n; i++) {
        BoardJob *j = &jobs[i];
        BoardForecast *t = &times[i];
        t->start = t->end = -1;
        if (board_active(j->state) || board_waiting(j)) {
            t->duration = board_minutes(b, j);
            t->gate = !strcmp(job_text(j->row, "kind"), "runner.wait_for_queue");
        }
        if (board_active(j->state)) {
            ceiling += j->memory;
            t->left = fmax(t->duration - (stamp - job_number(j->row, "started")) / 60, 1);
        } else if (board_waiting(j)) {
            *job_cell(j->row, "not_before") = (BoardCell){.name = "not_before"};
            *job_cell(j->row, "free_gib") = (BoardCell){.name = "free_gib"};
            j->needs.count = 0;
            j->pattern = NULL;
        }
    }
    for (;;) {
        double room = ceiling, step = INFINITY;
        for (int i = 0; i < n; i++) {
            if (board_active(jobs[i].state)) {
                room -= jobs[i].memory;
            }
        }
        board_dispatch(b, jobs, n, m, room, 2, 0);
        int count = 0;
        for (int i = 0; i < n; i++) {
            BoardJob *j = &jobs[i];
            BoardForecast *t = &times[i];
            if (j->scheduled) {
                j->scheduled = 0;
                t->left = t->duration;
                t->start = clock;
            }
            if (board_active(j->state)) {
                count++;
                if (!t->gate) {
                    step = fmin(step, t->left);
                }
            }
        }
        if (!count) {
            break;
        }
        clock += isfinite(step) ? step : 0;
        for (int i = 0; i < n; i++) {
            BoardJob *j = &jobs[i];
            BoardForecast *t = &times[i];
            int finished = board_active(j->state) && !isfinite(step);
            if (board_active(j->state) && isfinite(step) && !t->gate) {
                t->left -= step;
                finished = t->left <= 1e-9;
            }
            if (finished) {
                j->state = "done";
                t->end = clock;
            }
        }
    }
    char when[64], ends[64];
    if (name) {
        BoardJob *j = board_find(jobs, n, name);
        BoardForecast *t = j ? &times[j - jobs] : NULL;
        if (t && t->end >= 0) {
            board_utc(ends, stamp + t->end * 60, "%H:%M UTC");
            printf("%s ", name);
            if (t->start >= 0) {
                board_utc(when, stamp + t->start * 60, "%H:%M UTC");
                printf("starts about %s, ", when);
            }
            printf("ends about %s\n", ends);
        } else {
            printf("%s will not run as things stand: it is held, ended, or waits behind "
                   "one that is\n",
                name);
        }
    } else {
        board_utc(when, stamp + clock * 60, "%H:%M UTC");
        printf("the queue should end about %s, %.1f h from now, by past times of each sort "
               "of job; disc limits are not counted",
            when, clock / 60);
        int stuck = 0;
        double busy[3] = {0};
        for (int i = 0; i < n; i++) {
            stuck += board_waiting(&jobs[i]);
            for (int s = 0; s < 3; s++) {
                if (times[i].end >= 0 && (jobs[i].slots & (1 << s))) {
                    busy[s] += times[i].end - fmax(times[i].start, 0);
                }
            }
        }
        if (stuck) {
            printf("; %d held or behind one held are left out", stuck);
        }
        puts("");
        if (clock) {
            printf("gpu busy %.0f%%, engine busy %.0f%%, link busy %.0f%%\n", 100 * busy[0] / clock,
                100 * busy[1] / clock, 100 * busy[2] / clock);
        }
        sqlite3_stmt *q = board_query(b,
            "SELECT DISTINCT CASE WHEN owner IS NULL OR owner='' "
            "THEN 'no owner' ELSE owner END FROM jobs ORDER BY 1");
        while (sqlite3_step(q) == SQLITE_ROW) {
            const char *owner = (const char *)sqlite3_column_text(q, 0);
            double last = -1;
            for (int i = 0; i < n; i++) {
                const char *job_owner = job_text(jobs[i].row, "owner");
                if (!strcmp(owner, *job_owner ? job_owner : "no owner")) {
                    last = fmax(last, times[i].end);
                }
            }
            if (last >= 0) {
                board_utc(when, stamp + last * 60, "%H:%M UTC");
                printf("  %-24s%s\n", owner, when);
            }
        }
        sqlite3_finalize(q);
    }
    free(times);
    board_free(jobs, n);
}

static void board_folder(char path[PATH_MAX]) {
    char cwd[PATH_MAX];
    const char *configured = env("HERCULEX_BOARD", "");
    if (*configured == '/') {
        snprintf(path, PATH_MAX, "%s", configured);
    } else {
        need(getcwd(cwd, sizeof(cwd)) != NULL, "cannot locate board folder");
        if (*configured) {
            snprintf(path, PATH_MAX, "%s/%s", cwd, configured);
        } else {
            snprintf(path, PATH_MAX, "%s/runtime/board", *HOME == '/' ? HOME : cwd);
        }
    }
}

int board_command(int argc, char **argv) {
    if (!argc || !strcmp(argv[0], "--help") || !strcmp(argv[0], "-h")) {
        puts("board add|adopt|status|room|serve|wait|hold|release|cancel|retry|remove|"
             "resize|change|log|usage|forecast ...");
        return !argc;
    }
    for (int i = 1; i < argc && strcmp(argv[i], "--") != 0; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            if (!strcmp(argv[0], "add")) {
                puts("board add NAME [--memory GB] [--cores N] [--owner NAME] [--priority N] "
                     "[--gpu] [--engine] [--link] [--share] [--after JOB ...] [--needs PATH ...] "
                     "[--held] [--restartable] [--setting KEY=VALUE ...] -- COMMAND ...");
            } else {
                printf("board %s [ARGUMENTS]\n", argv[0]);
            }
            return 0;
        }
    }
    Arena option_names = arena(1 << 20);
    char *expanded[2 * argc];
    int raw = 0, count = 0;
    for (int i = 0; i < argc; i++) {
        char *word = argv[i];
        raw |= !strcmp(word, "--");
        char *equal = !raw && !strncmp(word, "--", 2) ? strchr(word + 2, '=') : NULL;
        if (equal) {
            expanded[count++] = fmt(&option_names, "%.*s", (int)(equal - word), word);
            expanded[count++] = equal + 1;
        } else {
            expanded[count++] = word;
        }
    }
    argv = expanded;
    argc = count;
    char path[PATH_MAX];
    board_folder(path);
    Board b = {.path = path,
        .tick = board_value(env("HERCULEX_BOARD_TICK", "2")),
        .retry = board_value(env("HERCULEX_BOARD_RETRY", "240")),
        .margin = board_value(env("HERCULEX_BOARD_MARGIN_GB", "1.0")),
        .cpu = sysconf(_SC_NPROCESSORS_ONLN)};
    const char *cpu = getenv("HERCULEX_BOARD_CORES");
    if (cpu && *cpu) {
        b.cpu = board_value(cpu);
    }
    need(b.tick > 0 && b.retry >= 0 && b.cpu > 0, "invalid board timing or core count");
    board_open(&b);
    const char *action = *argv++;
    argc--;
    int result = 0;
    if (!strcmp(action, "serve")) {
        board_serve(&b);
    } else if (!strcmp(action, "wrap")) {
        need(argc == 1, "board wrap needs NAME");
        board_wrap(&b, argv[0]);
    } else {
        board_serving(&b);
        if (!strcmp(action, "add") || !strcmp(action, "adopt")) {
            board_insert(&b, argc, argv, !strcmp(action, "adopt"));
        } else if (!strcmp(action, "status")) {
            need(!argc || (argc == 1 && !strcmp(argv[0], "--all")), "board status [--all]");
            board_status(&b, argc != 0);
        } else if (!strcmp(action, "usage")) {
            need(argc <= 1, "board usage [HOURS]");
            board_usage_report(&b, argc ? board_value(argv[0]) : 24);
        } else if (!strcmp(action, "forecast")) {
            need(argc <= 1, "board forecast [NAME]");
            board_forecast(&b, argc ? argv[0] : NULL);
        } else if (!strcmp(action, "room")) {
            need(argc == 1, "board room needs GB");
            int n;
            BoardJob *jobs = board_jobs(&b, &n);
            BoardMachine m = board_probe();
            double room = board_room(jobs, n, m.room - b.margin), gb = board_value(argv[0]);
            double swap_limit = board_swap_limit(m, 0.25, 4.2);
            int fits = gb <= room &&
                (gb <= 1 || m.swap <= swap_limit || (m.total > 0 && room >= gb + 4.0));
            printf("%s: %g GB against room %.1f, swap %.1f\n", fits ? "fits" : "wait", gb, room,
                m.swap);
            result = !fits;
            board_free(jobs, n);
        } else {
            result = board_action(&b, action, argc, argv);
        }
    }
    sqlite3_close(b.db);
    munmap(option_names.base, option_names.size);
    return result;
}
