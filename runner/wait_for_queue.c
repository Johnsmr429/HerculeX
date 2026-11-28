#include <errno.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char hx_usage[] = "Ends when none of the named jobs can still run, however they ended.";

static sqlite3_stmt *statement(sqlite3 *db, const char *query) {
    sqlite3_stmt *s = NULL;
    if (sqlite3_prepare_v2(db, query, -1, &s, NULL) == SQLITE_OK)
        return s;
    return NULL;
}

static int requests(sqlite3 *db, int argc, char **argv) {
    if (sqlite3_exec(db, "CREATE TEMP TABLE requested(name TEXT)", NULL, NULL, NULL) != SQLITE_OK)
        return -1;
    sqlite3_stmt *add = statement(db, "INSERT INTO requested VALUES (?)");
    if (!add)
        return -1;
    int rc = SQLITE_DONE;
    for (int i = 1; i < argc && rc == SQLITE_DONE; i++) {
        rc = sqlite3_bind_text(add, 1, argv[i], -1, SQLITE_STATIC);
        if (rc == SQLITE_OK)
            rc = sqlite3_step(add);
        sqlite3_reset(add);
    }
    sqlite3_finalize(add);
    return rc == SQLITE_DONE ? 0 : -1;
}

static void quoted(const unsigned char *s) {
    int q = strchr((const char *)s, '\'') && !strchr((const char *)s, '"') ? '"' : '\'';
    fputc(q, stderr);
    for (; *s; s++) {
        if (*s == q || *s == '\\')
            fputc('\\', stderr);
        if (*s == '\n' || *s == '\r' || *s == '\t')
            fprintf(stderr, "\\%c", *s == '\n' ? 'n' : (*s == '\r' ? 'r' : 't'));
        else if (*s < 32 || *s == 127)
            fprintf(stderr, "\\x%02x", *s);
        else
            fputc(*s, stderr);
    }
    fputc(q, stderr);
}

static int unknown(sqlite3 *db, int empty, int missing_table) {
    const char *query = missing_table
        ? "SELECT DISTINCT name FROM requested ORDER BY name"
        : "SELECT name FROM requested EXCEPT SELECT name FROM jobs ORDER BY name";
    sqlite3_stmt *s = statement(db, query);
    if (!s)
        return -1;
    int count = 0, rc;
    while ((rc = sqlite3_step(s)) == SQLITE_ROW) {
        fputs(count ? ", " : "not on the board: [", stderr);
        quoted(sqlite3_column_text(s, 0));
        count++;
    }
    sqlite3_finalize(s);
    if (count)
        fputs("]\n", stderr);
    else if (empty)
        fputs("not on the board: []\n", stderr);
    return rc == SQLITE_DONE ? !!(empty || count) : -1;
}

static int monitor(sqlite3 *db) {
    if (sqlite3_exec(db, "CREATE TEMP TABLE ready(name TEXT PRIMARY KEY) WITHOUT ROWID", NULL, NULL,
            NULL) != SQLITE_OK)
        return -1;
    sqlite3_stmt *grow = statement(db,
        "INSERT OR IGNORE INTO ready SELECT name FROM jobs j WHERE state='waiting' AND NOT held "
        "AND NOT EXISTS (SELECT 1 FROM json_each(j.after) d WHERE NOT EXISTS "
        "(SELECT 1 FROM ready WHERE name=d.value))");
    sqlite3_stmt *left = statement(db,
        "SELECT r.name FROM requested r JOIN ready USING(name) JOIN jobs j USING(name) "
        "WHERE j.state!='done' ORDER BY r.rowid");
    int previous = -1, failed = !grow || !left;
    while (!failed) {
        int rc = sqlite3_exec(db,
            "BEGIN;DELETE FROM ready;INSERT INTO ready SELECT name FROM jobs "
            "WHERE state IN ('done','starting','running','stopping')",
            NULL, NULL, NULL);
        if (rc != SQLITE_OK) {
            failed = 1;
            break;
        }
        do {
            rc = sqlite3_step(grow);
            sqlite3_reset(grow);
        } while (rc == SQLITE_DONE && sqlite3_changes(db) > 0);
        if (rc != SQLITE_DONE) {
            failed = 1;
            break;
        }
        int count = 0;
        char *first = NULL;
        while ((rc = sqlite3_step(left)) == SQLITE_ROW) {
            if (!count)
                first = strdup((const char *)sqlite3_column_text(left, 0));
            count++;
        }
        sqlite3_reset(left);
        failed = rc != SQLITE_DONE || sqlite3_exec(db, "COMMIT", NULL, NULL, NULL) != SQLITE_OK ||
            (count && !first);
        if (!failed && count && count != previous) {
            printf("%d jobs can still run, first %s\n", count, first);
            fflush(stdout);
        }
        free(first);
        if (failed || !count)
            break;
        previous = count;
        sleep(60);
    }
    sqlite3_finalize(left);
    sqlite3_finalize(grow);
    if (!failed)
        puts("queue ended");
    return failed ? -1 : 0;
}

int main(int argc, char **argv) {
    const char *root = getenv("HERCULEX_BOARD");
    if (!root)
        root = "runtime/board";
    char *path = sqlite3_mprintf("%s/board.sqlite", root);
    if (!path)
        return 1;
    int absent = access(path, F_OK) != 0 && errno == ENOENT;
    sqlite3 *db = NULL;
    int flags = absent ? SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE : SQLITE_OPEN_READONLY;
    int rc = sqlite3_open_v2(absent ? ":memory:" : path, &db, flags, NULL);
    sqlite3_free(path);
    int result = rc == SQLITE_OK && !requests(db, argc, argv) ? 0 : -1;
    if (!result) {
        sqlite3_busy_timeout(db, 30000);
        sqlite3_stmt *s = statement(db, "SELECT 1 FROM sqlite_master WHERE name='jobs'");
        int found = s ? sqlite3_step(s) : SQLITE_ERROR;
        sqlite3_finalize(s);
        result = found == SQLITE_ROW || found == SQLITE_DONE
            ? unknown(db, argc == 1, found == SQLITE_DONE)
            : -1;
        if (!result)
            result = monitor(db);
    }
    if (result < 0)
        fprintf(stderr, "read job board: %s\n", db ? sqlite3_errmsg(db) : "out of memory");
    sqlite3_close(db);
    return result ? 1 : 0;
}
