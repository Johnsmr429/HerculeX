#ifndef HX_UNITY
#define HX_UNITY
#endif
#include "../base.h"
#include "../storage/storage.h"
#include "../runner/job_board.c"
#include "../runner/events.c"

static char folder[] = "/tmp/herculex-job-board.XXXXXXXX";
static char lock_path[sizeof(folder) + 24];
static const char *binary;

static void clean(void) { unlink(lock_path); }

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "board e2e: %s (database in %s)\n", message, folder);
        exit(1);
    }
}

static void run(char **argv) {
    pid_t child = fork();
    require(child >= 0, "could not fork");
    if (!child) {
        argv[0] = (char *)binary;
        execv(binary, argv);
        _exit(127);
    }
    int status;
    while (waitpid(child, &status, 0) < 0) {
        require(errno == EINTR, "waitpid failed");
    }
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, argv[2]);
}

#define RUN(...) run((char *[]){NULL, "board", __VA_ARGS__, NULL})

static sqlite3 *database(void) {
    char path[sizeof(folder) + 20];
    snprintf(path, sizeof(path), "%s/board.sqlite", folder);
    sqlite3 *db;
    require(sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, NULL) == SQLITE_OK,
        "could not open board database");
    sqlite3_busy_timeout(db, 1000);
    return db;
}

static int query(sqlite3 *db, const char *sql) {
    sqlite3_stmt *q;
    require(sqlite3_prepare_v2(db, sql, -1, &q, NULL) == SQLITE_OK, sqlite3_errmsg(db));
    require(sqlite3_step(q) == SQLITE_ROW, sqlite3_errmsg(db));
    int result = sqlite3_column_int(q, 0);
    sqlite3_finalize(q);
    return result;
}

static void until(sqlite3 *db, const char *sql, const char *message) {
    for (int i = 0; i < 300; i++) {
        if (query(db, sql)) {
            return;
        }
        nanosleep(&(struct timespec){.tv_nsec = 50000000}, NULL);
    }
    require(0, message);
}

static void board_decisions_golden(void) {
    Board b = {.cpu = 10};
    BoardMachine m = {.room = 5, .swap = 1};
    BoardJob large = {.name = "large", .state = "waiting", .memory = 8, .priority = 5, .cores = 1};
    BoardJob same = {.name = "same", .state = "waiting", .memory = 2, .priority = 5, .cores = 1};
    BoardJob small = {.name = "small", .state = "waiting", .memory = 2, .priority = 1, .cores = 1};
    BoardJob busy = {.name = "busy", .state = "running", .memory = 1, .cores = 1};
    BoardJob jobs[] = {large, same, small, busy};
    board_dispatch(&b, jobs, 4, m, 3, 0, 0);
    require(strstr(jobs[0].reason, "needs 8.0 GB") != NULL, "large job awaits memory");
    require(!*jobs[1].reason, "same priority uses safe memory");
    require(strstr(jobs[2].reason, "needs 2.0 GB") != NULL, "lower priority respects reservation");
    busy.memory = 5;
    busy.peak = 1;
    require(board_room(&busy, 1, 6) == 2, "running job retains reservation");
    busy.peak = 5.2;
    require(board_room(&busy, 1, 6) == 6, "measured peak releases reservation");

    BoardJob read = {.name = "read", .state = "running", .slots = 7, .memory = 1, .cores = 1};
    BoardJob nodes = {
        .name = "nodes", .state = "waiting", .slots = 5, .memory = 1, .cores = 1, .share = 1};
    board_decide(&b, &nodes, &read, 1, m, 5, 0);
    require(!*nodes.reason, "shared GPU and link beside whole job");
    nodes.slots |= 2;
    board_decide(&b, &nodes, &read, 1, m, 5, 0);
    require(!strcmp(nodes.reason, "engine is with read"), "engine remains exclusive");
    nodes.slots = 1;
    nodes.share = 0;
    board_decide(&b, &nodes, &read, 1, m, 5, 0);
    require(!strcmp(nodes.reason, "gpu is with read"), "whole GPU remains exclusive");
    nodes.slots = 0;
    nodes.cores = 10;
    board_decide(&b, &nodes, &read, 1, m, 5, 0);
    require(strstr(nodes.reason, "needs 10 cores") != NULL, "running cores count");
    nodes.cores = 1;
    nodes.memory = 2;
    board_decide(&b, &nodes, NULL, 0, (BoardMachine){.swap = 5}, 5, 0);
    require(strstr(nodes.reason, "swap 5.0 GB") != NULL, "swap blocks large launches");
    nodes.memory = 0.5;
    board_decide(&b, &nodes, NULL, 0, (BoardMachine){.swap = 5}, 5, 0);
    require(!*nodes.reason, "small jobs can run under swap");

    BoardJob forecasts[] = {{.name = "one", .state = "waiting", .slots = 4, .cores = 1},
        {.name = "two", .state = "waiting", .slots = 4, .cores = 1}};
    board_dispatch(&b, forecasts, 2, m, 5, 2, 0);
    require(forecasts[0].scheduled && !forecasts[1].scheduled, "forecast honors exclusive link");
    require(!strcmp(forecasts[1].reason, "link is with one"), "forecast has the first slot holder");
}

static double covered_seconds(json_object *spans) {
    double total = 0;
    for (size_t i = 0; i < json_object_array_length(spans); i++) {
        json_object *one = json_object_array_get_idx(spans, i);
        total += json_object_get_double(json_object_array_get_idx(one, 1)) -
            json_object_get_double(json_object_array_get_idx(one, 0));
    }
    return total;
}

static void timeline_golden(void) {
    Arena a = arena(1 << 20);
    char *timeline_db = fmt(&a, "%s/timeline", folder);
    setenv("HERCULEX_BOARD", timeline_db, 1);
    Board b = {.path = timeline_db};
    board_open(&b);
    const char *names[] = {"s-sheets", "s-pieces", "s-sheet", "s-fetch"};
    const int times[][3] = {{0, 0, 10}, {0, 10, 30}, {0, 30, 40}, {40, 45, 50}};
    const char *deps[] = {"[]", "[\"s-sheets\"]", "[\"s-pieces\"]", "[]"};
    const int slots[] = {5, 0, 7, 4};
    double start = 1000000;
    for (int i = 0; i < 4; i++) {
        board_exec(&b,
            "INSERT INTO jobs(name,state,added,started,ended,gpu,engine,link,after) "
            "VALUES (%Q,'done',%f,%f,%f,%d,%d,%d,%Q)",
            names[i], start + 60 * times[i][0], start + 60 * times[i][1], start + 60 * times[i][2],
            slots[i] & 1, !!(slots[i] & 2), !!(slots[i] & 4), deps[i]);
    }
    for (int i = 0; i < 50; i++) {
        board_exec(&b, "INSERT INTO samples(time,gpu,net_mb_s,room_gb) VALUES (%f,%d,%d,5)",
            start + 60 * i + 1, i < 10 ? 90 : 0, i < 10 ? 30 : 0);
    }
    sqlite3_close(b.db);
    char *events = fmt(&a, "%s/events/s-sheet.jsonl", timeline_db);
    mkparent(events);
    setenv("HERCULEX_EVENTS", events, 1);
    runner_event("wait-fetch", start + 1800, start + 1980, NULL);
    runner_event("read", start + 1980, start + 2400, NULL);
    unsetenv("HERCULEX_EVENTS");
    int fd = open(events, O_WRONLY | O_APPEND);
    require(fd >= 0, "cannot append a partial event");
    write_all(fd, "{cut short", 10, events);
    close(fd);
    char *json = fmt(&a, "%s/timeline.json", folder);
    run((char *[]){NULL, "runner.timeline", "s-*", "--width", "50", "--json", json, NULL});
    json_object *data = json_read(json);
    require(data != NULL, "timeline JSON missing");
    json_object *slot = jget(jget(data, "slots"), "link");
    require(covered_seconds(jget(slot, "held")) == 1500, "timeline records link held");
    require(covered_seconds(jget(slot, "idle_ready")) == 300, "timeline link ready wait");
    require(covered_seconds(jget(slot, "idle_behind")) == 1200, "timeline link dependency wait");
    json_object *reported = jget(data, "jobs");
    for (int i = 0; i < 4; i++) {
        json_object *job = json_object_array_get_idx(reported, i);
        require(!strcmp(jstr(job, "name"), names[i]), "timeline job order");
    }
    json_object *phases = jget(json_object_array_get_idx(reported, 2), "phases");
    require(covered_seconds(jget(phases, "wait-fetch")) == 180, "timeline fetch event");
    require(covered_seconds(jget(phases, "read")) == 420, "timeline read event");
    json_object_put(data);
    munmap(a.base, a.size);
}

int main(int argc, char **argv) {
    require(argc == 2, "usage: job_board_e2e PATH-TO-HERCULEX");
    binary = argv[1];
    board_decisions_golden();
    require(mkdtemp(folder) != NULL, "could not create temporary board");
    snprintf(lock_path, sizeof(lock_path), "%s/dispatcher.lock", folder);
    atexit(clean);
    setenv("HERCULEX_BOARD", folder, 1);
    setenv("HERCULEX_SCRATCH", folder, 1);
    setenv("HERCULEX_BOARD_TICK", "0.05", 1);
    setenv("HERCULEX_BOARD_RETRY", "0.05", 1);
    setenv("HERCULEX_BOARD_CORES", "2", 1);
    char marker[sizeof(folder) + 20];
    snprintf(marker, sizeof(marker), "MARKER=%s/retry", folder);
    RUN("add", "first", "--held", "--memory", "0.1", "--setting", marker, "FLAG=works", "--",
        "/bin/sh", "-c",
        "test \"$FLAG\" = works || exit 3; test -e \"$MARKER\" && exit 0; "
        "touch \"$MARKER\"; exit 75");
    RUN("add", "dependent", "--after", "first", "--memory", "0.1", "--link", "--", "/bin/sh", "-c",
        "exit 0");
    sqlite3 *db = database();
    require(query(db, "SELECT count(*)=2 FROM jobs WHERE state='waiting'"), "held dependency");
    RUN("forecast", "dependent");
    RUN("release", "first");
    until(db, "SELECT count(*)=2 FROM jobs WHERE state='done' AND code=0", "dependency result");
    require(query(db, "SELECT tries=2 FROM jobs WHERE name='first'"), "try later count");
    require(query(db,
                "SELECT (SELECT ended FROM jobs WHERE name='first')<=started "
                "FROM jobs WHERE name='dependent'"),
        "dependency timestamps");
    RUN("add", "cancel", "--memory", "0.1", "--", "/bin/sleep", "20");
    until(db, "SELECT state='running' FROM jobs WHERE name='cancel'", "launch for cancellation");
    RUN("cancel", "cancel");
    until(db, "SELECT state='stopped' AND code=-15 FROM jobs WHERE name='cancel'", "cancellation");
    RUN("usage");
    RUN("status", "--all");
    require(query(db, "SELECT count(*)>0 FROM samples"), "board sample saved");
    sqlite3_close(db);
    timeline_golden();
    puts("job board e2e passed");
    return 0;
}
