#ifndef HX_UNITY
const char hx_usage[] = "Divide waiting survey jobs into a fetch and a read.";
#endif

int fetch_ahead_command(int argc, char **argv) {
    if (argc == 1 && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("runner.fetch_ahead [apply]");
        return 0;
    }
    need(!argc || (argc == 1 && !strcmp(argv[0], "apply")), "runner.fetch_ahead [apply]");
    char path[PATH_MAX], home[PATH_MAX];
    board_folder(path);
    need(realpath(HOME, home) != NULL, "cannot resolve HerculeX home %s", HOME);
    Arena a = arena(1 << 20);
    const char *bricks = getenv("HERCULEX_BRICKS");
    const char *pattern = bricks && *bricks ? fmt(&a, "%s/*/ct_bricks/survey*", bricks)
                                            : fmt(&a, "%s/cache/bricks/*/survey*", LIBRARY);
    json_object *at_most = json_object_new_array();
    json_object_array_add(at_most, json_object_new_int(3));
    json_object_array_add(at_most, json_object_new_string(pattern));
    Board b = {.path = path};
    board_open(&b);
    board_exec(&b, "BEGIN IMMEDIATE");
    sqlite3_stmt *q = board_query(&b,
        "SELECT * FROM jobs WHERE state='waiting' AND link "
        "ORDER BY added");
    json_object *jobs = json_object_new_array();
    while (sqlite3_step(q) == SQLITE_ROW) {
        json_object_array_add(jobs, board_row(q));
    }
    sqlite3_finalize(q);
    int chosen = 0, split = 0, gated = 0;
    const char *modes[] = {"look", "replicate", "reread", "row-sample", "second"};
    for (int i = 0; i < board_count(jobs); i++) {
        json_object *job = json_object_array_get_idx(jobs, i);
        json_object *command = json_tokener_parse(board_text(job, "command"));
        int words = board_count(command), survey = 0;
        for (int w = 0; w < words; w++) {
            survey |= !strcmp(board_item(command, w), "calibrate.survey.scroll_survey");
        }
        const char *mode = words >= 2 ? board_item(command, words - 2) : "";
        int match = 0;
        for (int m = 0; m < 5; m++) {
            match |= !strcmp(mode, modes[m]);
        }
        if (survey && match) {
            const char *name = board_text(job, "name");
            int priority = board_integer(job, "priority");
            json_object *after = json_tokener_parse(board_text(job, "after"));
            json_object *needs = json_tokener_parse(board_text(job, "needs"));
            printf("%-30s priority %3d  after %s  needs %d\n", name, priority, board_json(after),
                board_count(needs));
            chosen++;
            if (argc) {
                const char *fetch = fmt(&a, "fetch-%s", name);
                json_object_array_add(after, json_object_new_string(fetch));
                board_exec(&b,
                    "UPDATE jobs SET link=0,after=%Q WHERE name=%Q "
                    "AND state='waiting' AND link=1",
                    board_json(after), name);
                if (sqlite3_changes(b.db)) {
                    json_object *fetch_command = json_object_new_array();
                    const char *words[] = {"./run.sh", "calibrate.survey.survey_fetch", mode,
                        board_item(command, board_count(command) - 1)};
                    for (int k = 0; k < 4; k++) {
                        json_object_array_add(fetch_command, json_object_new_string(words[k]));
                    }
                    board_exec(&b,
                        "INSERT INTO jobs(name,kind,command,folder,settings,owner,note,"
                        "memory_gb,cores,gpu,engine,link,priority,nice,background,after,needs,"
                        "clean,"
                        "restartable,free_gib,at_most,state,added) VALUES "
                        "(%Q,'calibrate.survey.survey_fetch',%Q,%Q,'{}','quality of life',%Q,"
                        "0.7,1,0,0,1,%d,5,0,%Q,%Q,'[]',1,40,%Q,'waiting',%f)",
                        fetch, board_json(fetch_command), home, fmt(&a, "bricks of %s", name),
                        priority, board_text(job, "after"), board_text(job, "needs"),
                        board_json(at_most), board_number(job, "added") - 0.001);
                    json_object_put(fetch_command);
                    split++;
                }
            }
            json_object_put(after);
            json_object_put(needs);
        }
        json_object_put(command);
    }
    if (argc) {
        board_exec(&b,
            "UPDATE jobs SET free_gib=40,at_most=%Q WHERE name LIKE 'fetch-%%' "
            "AND state='waiting' AND at_most IS NULL",
            board_json(at_most));
        gated = sqlite3_changes(b.db);
    }
    board_exec(&b, argc ? "COMMIT" : "ROLLBACK");
    char stamp[64];
    time_t seconds = time(NULL);
    struct tm local;
    localtime_r(&seconds, &local);
    strftime(stamp, sizeof(stamp), "%H:%M:%S", &local);
    printf("%d jobs to split, %d split, %d fetches gated on the board %s\n", chosen, split, gated,
        stamp);
    json_object_put(jobs);
    json_object_put(at_most);
    sqlite3_close(b.db);
    munmap(a.base, a.size);
    return 0;
}
