#include <spawn.h>
#include "reflatten.c"
#include "flatten.c"

extern char **environ;

static const char *spot_tables[] = {"spots", "surfaces", "reads", "verdicts"};

static char *spot_sql(Arena *a, const char *text) {
    char *quoted = push(a, 2 * strlen(text) + 3), *p = quoted;
    *p++ = '\'';
    for (; *text; text++) {
        *p++ = *text;
        if (*text == '\'') {
            *p++ = '\'';
        }
    }
    *p++ = '\'';
    *p = 0;
    return quoted;
}

static void spot_query(Arena *a, const char *sql, int output) {
    const char *engine = getenv("HERCULEX_DUCKDB");
    if (!engine || !*engine) {
        const char *cached = fmt(a, "%s/runtime/deps/duckdb", HOME);
        engine = access(cached, X_OK) ? "duckdb" : cached;
    }
    char *args[] = {(char *)engine, "-batch", "-bail", "-init", "/dev/null", "-noheader", "-list",
        "-c", (char *)sql, NULL};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    if (output < 0) {
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    } else if (output != STDOUT_FILENO) {
        posix_spawn_file_actions_adddup2(&actions, output, STDOUT_FILENO);
    }
    pid_t pid;
    int error = posix_spawnp(&pid, engine, &actions, NULL, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    need(!error, "cannot open DuckDB for the spot tables: %s; run ./run.sh setup spots",
        strerror(error));
    int status;
    pid_t result;
    do {
        result = waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    need(result == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "DuckDB could not read or write the spot tables");
}

static char *spot_table_path(Arena *a, const char *table) {
    return fmt(a, "%s/studies/shared/spot-candidates/db/%s.parquet", LIBRARY, table);
}

json_object *spot_rows(Arena *a, const char *table) {
    const char *file = spot_table_path(a, table);
    json_object *rows = json_object_new_array();
    if (!exists(file)) {
        return rows;
    }
    FILE *result = tmpfile();
    need(result != NULL, "cannot open a result for %s", table);
    spot_query(a, fmt(a, "SELECT to_json(row) FROM read_parquet(%s) AS row", spot_sql(a, file)),
        fileno(result));
    need(!fseek(result, 0, SEEK_SET), "cannot read the spot table %s", table);
    size_t cap = 0;
    char *line = NULL;
    while (getline(&line, &cap, result) >= 0) {
        json_object *row = json_tokener_parse(line);
        need(row != NULL, "cannot decode a saved row in %s", table);
        json_object_array_add(rows, row);
    }
    free(line);
    fclose(result);
    return rows;
}

void spot_write(Arena *a, const char *table, json_object *rows) {
    size_t count = json_object_array_length(rows);
    if (!count) {
        return;
    }
    const char *file = spot_table_path(a, table);
    const char *part = fmt(a, "%s.part", file);
    int lock = lock_file(fmt(a, "%s/studies/shared/spot-candidates/db/.lock", LIBRARY), 1, 1);
    need(lock >= 0, "cannot lock the spot tables");
    char *incoming = fmt(a, "%s.XXXXXXXX", file);
    int fd = mkstemp(incoming);
    need(fd >= 0, "cannot stage %s rows", table);
    const char *stamp = utc_stamp(a, "%Y-%m-%dT%H:%M:%SZ");
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        int timed = json_object_object_get_ex(row, "written_at", NULL);
        if (!timed) {
            json_object_object_add(row, "written_at", json_object_new_string(stamp));
        }
        size_t n;
        const char *line = json_object_to_json_string_length(row, JSON_C_TO_STRING_PLAIN, &n);
        write_all(fd, line, n, incoming);
        write_all(fd, "\n", 1, incoming);
        if (!timed) {
            json_object_object_del(row, "written_at");
        }
    }
    close(fd);
    const char *key = "old.spot_id IS NOT DISTINCT FROM new.spot_id";
    if (strcmp(table, "spots") != 0) {
        key = fmt(a, "%s AND old.surface IS NOT DISTINCT FROM new.surface", key);
    }
    if (!strcmp(table, "reads")) {
        key = fmt(a,
            "%s AND old.model IS NOT DISTINCT FROM new.model "
            "AND old.side IS NOT DISTINCT FROM new.side AND old.depth IS NOT DISTINCT FROM "
            "new.depth",
            key);
    }
    if (!strcmp(table, "verdicts")) {
        key = fmt(a, "%s AND old.judged_at IS NOT DISTINCT FROM new.judged_at", key);
    }
    const char *reader = fmt(a,
        "read_ndjson_auto(%s, sample_size=-1, "
        "dateformat='%%Y-%%m-%%d+spot', timestampformat='%%Y-%%m-%%d%%H%%M%%S+spot')",
        spot_sql(a, incoming));
    const char *old = exists(file)
        ? fmt(a,
              "SELECT old.* FROM read_parquet(%s) AS old WHERE NOT EXISTS "
              "(SELECT 1 FROM fresh AS new WHERE %s) UNION ALL BY NAME ",
              spot_sql(a, file), key)
        : "";
    need(!unlink(part) || errno == ENOENT, "cannot reset %s", part);
    spot_query(a,
        fmt(a,
            "COPY (WITH fresh AS (SELECT * FROM %s) %sSELECT * FROM fresh) TO %s "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            reader, old, spot_sql(a, part)),
        -1);
    unlink(incoming);
    need(!rename(part, file), "cannot update %s", file);
    close(lock);
}

void spot_forget(Arena *a, json_object *ids) {
    size_t n = json_object_array_length(ids);
    if (!n) {
        return;
    }
    const char *list = "";
    for (size_t i = 0; i < n; i++) {
        const char *id = json_object_get_string(json_object_array_get_idx(ids, i));
        need(id != NULL, "invalid spot ID");
        list = fmt(a, "%s%s%s", list, i ? ", " : "", spot_sql(a, id));
    }
    int lock = lock_file(fmt(a, "%s/studies/shared/spot-candidates/db/.lock", LIBRARY), 1, 1);
    need(lock >= 0, "cannot lock the spot tables");
    for (int i = 0; i < 3; i++) {
        const char *file = spot_table_path(a, spot_tables[i]);
        if (!exists(file)) {
            continue;
        }
        const char *part = fmt(a, "%s.part", file);
        need(!unlink(part) || errno == ENOENT, "cannot reset %s", part);
        spot_query(a,
            fmt(a,
                "COPY (SELECT * FROM read_parquet(%s) WHERE spot_id NOT IN (%s)) "
                "TO %s (FORMAT PARQUET, COMPRESSION ZSTD)",
                spot_sql(a, file), list, spot_sql(a, part)),
            -1);
        need(!rename(part, file), "cannot update %s", file);
    }
    close(lock);
}

static void spot_verdict(Arena *a, const char *db, int argc, char **argv) {
    need(argc >= 2, "usage: spots verdict SPOT SURFACE --by NAME --readable GRADE");
    const char *spot = argv[0], *surface = argv[1];
    const char *by = NULL, *readable = NULL, *stack = "ensemble", *side = "inward";
    const char *reading = "", *note = "";
    json_object *depth = NULL;
    i64 letters = 0;
    for (int i = 2; i < argc; i += 2) {
        const char *key = argv[i], *value = i + 1 < argc ? argv[i + 1] : NULL;
        need(value != NULL, "%s requires a value", key);
        if (!strcmp(key, "--by")) {
            by = value;
        } else if (!strcmp(key, "--readable")) {
            readable = value;
        } else if (!strcmp(key, "--stack")) {
            stack = value;
        } else if (!strcmp(key, "--side")) {
            side = value;
        } else if (!strcmp(key, "--reading")) {
            reading = value;
        } else if (!strcmp(key, "--note")) {
            note = value;
        } else if (!strcmp(key, "--letters") || !strcmp(key, "--depth")) {
            char *end;
            errno = 0;
            i64 n = strtoll(value, &end, 10);
            need(!errno && end != value && !*end, "%s needs an integer", key);
            if (!strcmp(key, "--letters")) {
                letters = n;
            } else {
                json_object_put(depth);
                depth = json_object_new_int64(n);
            }
        } else {
            die("unknown verdict option %s", key);
        }
    }
    need(by && readable &&
            (!strcmp(readable, "no") || !strcmp(readable, "traces") ||
                !strcmp(readable, "letters") || !strcmp(readable, "passage")),
        "verdict needs --by NAME and --readable no|traces|letters|passage");
    const char *stamp = utc_stamp(a, "%Y-%m-%dT%H:%M:%SZ");
    json_object *row = json_object_new_object();
    const char *keys[] = {
        "spot_id", "surface", "by", "readable", "stack", "side", "reading", "note", "judged_at"};
    const char *values[] = {spot, surface, by, readable, stack, side, reading, note, stamp};
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++) {
        json_object_object_add(row, keys[i], json_object_new_string(values[i]));
    }
    json_object_object_add(row, "letters", json_object_new_int64(letters));
    json_object_object_add(row, "depth", depth);
    const char *file = fmt(a, "%s/verdicts.parquet", db), *part = fmt(a, "%s.part", file);
    int lock = lock_file(fmt(a, "%s/.lock", db), 1, 1);
    need(lock >= 0, "cannot lock the spot tables");
    const char *old = exists(file)
        ? fmt(a,
              "SELECT * FROM read_parquet(%s) WHERE NOT "
              "(spot_id = %s AND surface = %s AND judged_at = %s) UNION ALL BY NAME ",
              spot_sql(a, file), spot_sql(a, spot), spot_sql(a, surface), spot_sql(a, stamp))
        : "";
    const char *select = fmt(a,
        "%sSELECT %s AS spot_id, %s AS surface, %s AS \"by\", "
        "%s AS readable, %s AS stack, %s AS side, %s AS reading, %s AS note, "
        "%s AS judged_at, %s::BIGINT AS letters, %s::BIGINT AS depth, %s AS written_at",
        old, spot_sql(a, spot), spot_sql(a, surface), spot_sql(a, by), spot_sql(a, readable),
        spot_sql(a, stack), spot_sql(a, side), spot_sql(a, reading), spot_sql(a, note),
        spot_sql(a, stamp), json_object_get_string(jget(row, "letters")),
        depth ? json_object_get_string(depth) : "NULL", spot_sql(a, stamp));
    need(!unlink(part) || errno == ENOENT, "cannot reset %s", part);
    spot_query(a,
        fmt(a, "COPY (%s) TO %s (FORMAT PARQUET, COMPRESSION ZSTD)", select, spot_sql(a, part)),
        -1);
    need(!rename(part, file), "cannot update %s", file);
    close(lock);
    puts(json_object_to_json_string_ext(row, JSON_C_TO_STRING_PLAIN));
    json_object_put(row);
}

#include "catalogue.c"
#include "sweep.c"
#include "surfaces.c"
#include "images.c"
#include "pilot.c"

int spots_command(int argc, char **argv) {
    if (argc && !strcmp(argv[0], "catalogue")) {
        return hx_spot_catalogue(argc - 1, argv + 1);
    }
    if (argc && !strcmp(argv[0], "pilot")) {
        return spot_pilot_command(argc - 1, argv + 1);
    }
    if (argc && !strcmp(argv[0], "database")) {
        argv++;
        argc--;
    }
    if (argc && (!strcmp(argv[0], "--help") || !strcmp(argv[0], "-h"))) {
        puts("Usage: ./run.sh spots [catalogue|pilot|spots|surfaces|reads|verdicts]\n"
             "       ./run.sh spots verdict SPOT SURFACE --by NAME --readable GRADE");
        return 0;
    }
    Arena a = arena(32 << 20);
    const char *db = fmt(&a, "%s/studies/shared/spot-candidates/db", LIBRARY);
    if (argc && !strcmp(argv[0], "verdict")) {
        spot_verdict(&a, db, argc - 1, argv + 1);
    } else if (argc) {
        int found = 0;
        for (size_t i = 0; i < sizeof(spot_tables) / sizeof(*spot_tables); i++) {
            if (!strcmp(argv[0], spot_tables[i])) {
                found = 1;
                const char *file = fmt(&a, "%s/%s.parquet", db, spot_tables[i]);
                if (exists(file)) {
                    spot_query(&a,
                        fmt(&a, "SELECT to_json(row) FROM read_parquet(%s) AS row",
                            spot_sql(&a, file)),
                        STDOUT_FILENO);
                }
                break;
            }
        }
        need(found && argc == 1, "usage: ./run.sh spots --help");
    } else {
        const char *fields = "";
        int existing = 0;
        for (size_t i = 0; i < sizeof(spot_tables) / sizeof(*spot_tables); i++) {
            const char *name = spot_tables[i], *file = fmt(&a, "%s/%s.parquet", db, name);
            const char *count = "0";
            if (exists(file)) {
                count = fmt(&a, "(SELECT count(*) FROM read_parquet(%s))", spot_sql(&a, file));
                existing++;
            }
            fields = fmt(&a, "%s%s%s := %s", fields, i ? ", " : "", name, count);
        }
        if (existing) {
            spot_query(&a, fmt(&a, "SELECT to_json(struct_pack(%s))", fields), STDOUT_FILENO);
        } else {
            puts("{\"spots\":0,\"surfaces\":0,\"reads\":0,\"verdicts\":0}");
        }
    }
    munmap(a.base, a.size);
    return 0;
}
