#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

const char hx_usage[] = "The wrap studies side by side at fixed wrong-wrap rates.";

static const double rates[] = {0.05, 0.10, 0.15, 0.20};

typedef struct {
    const char *name;
    double area[2][4][2];
} Study;

static int insert_rows(sqlite3_stmt *insert, int id, const char *name) {
    char *path = sqlite3_mprintf("records/surface/wrap-study-%s.json", name);
    if (!path)
        return 0;
    FILE *stream = fopen(path, "rb");
    if (!stream)
        perror(path);
    sqlite3_free(path);
    if (!stream)
        return 0;
    struct stat st;
    if (fstat(fileno(stream), &st) || st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fclose(stream);
        return 0;
    }
    size_t n = (size_t)st.st_size;
    char *data = malloc(n + 1);
    int read = data && fread(data, 1, n, stream) == n;
    fclose(stream);
    if (!read) {
        free(data);
        return 0;
    }
    data[n] = 0;
    int ok = sqlite3_bind_int(insert, 1, id) == SQLITE_OK &&
        sqlite3_bind_text64(insert, 2, data, n, SQLITE_STATIC, SQLITE_UTF8) == SQLITE_OK &&
        sqlite3_step(insert) == SQLITE_DONE;
    sqlite3_reset(insert);
    sqlite3_clear_bindings(insert);
    free(data);
    return ok;
}

static void header(char **names, int count, const char *label, const char *title) {
    printf("| %s |", label);
    for (int i = 0; i < count; i++)
        printf(" %s %s |", names[i], title);
    printf("\n| --- |");
    for (int i = 0; i < count; i++)
        printf(" --- |");
    putchar('\n');
}

static int print_report(sqlite3 *db, Study *studies, int distinct, char **names, int count) {
    const char *sql =
        "WITH shared AS (SELECT control,MAX(json_extract(data,'$.usable.wrong')>0.005) AS usable "
        "FROM results GROUP BY control HAVING COUNT(*)=?1) "
        "SELECT control,id,shared.usable,json_extract(data,'$.usable.claimed_cm2'),"
        "json_extract(data,'$.usable.correct'),json_extract(data,'$.usable.wrong_rate'),"
        "json_extract(data,'$.usable.correct_cm2'),json_extract(data,'$.window.claimed_cm2'),"
        "json_extract(data,'$.window.wrong_rate'),json_extract(data,'$.window.correct_cm2') "
        "FROM shared JOIN results USING(control) ORDER BY control,id";
    sqlite3_stmt *s = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &s, NULL) != SQLITE_OK)
        return 0;
    sqlite3_bind_int(s, 1, distinct);
    header(names, count, "control", "claimed cm2 / correct / wrong rate");
    int measured = 0, rc;
    while ((rc = sqlite3_step(s)) == SQLITE_ROW) {
        int id = sqlite3_column_int(s, 1);
        int measurable = sqlite3_column_int(s, 2);
        if (!id) {
            printf("| %s |", (const char *)sqlite3_column_text(s, 0));
            measured += measurable;
        }
        double claim[] = {sqlite3_column_double(s, 3), sqlite3_column_double(s, 7)};
        double wrong[] = {sqlite3_column_double(s, 5), sqlite3_column_double(s, 8)};
        double correct[] = {sqlite3_column_double(s, 6), sqlite3_column_double(s, 9)};
        printf(" %.0f / %.3f / %.3f |", claim[0], sqlite3_column_double(s, 4), wrong[0]);
        if (id == distinct - 1)
            putchar('\n');
        if (!measurable)
            continue;
        for (int j = 0; j < 4; j++) {
            for (int f = 0; f < 2; f++) {
                if (wrong[f] <= rates[j]) {
                    studies[id].area[0][j][f] += correct[f];
                    studies[id].area[1][j][f] += claim[f] * (1 - wrong[f]);
                }
            }
        }
    }
    sqlite3_finalize(s);
    if (rc != SQLITE_DONE)
        return 0;
    const char *title[] = {"confirmed by meshes", "claimed, less the measured wrong share"};
    for (int t = 0; t < 2; t++) {
        printf("\nContiguous correct-wrap cm2 %s, summed over the %d controls that can show "
               "a wrong wrap, a control counting only within the limit, pieces of at least 1 cm2 "
               "and of at least 4 cm2:\n\n",
            title[t], measured);
        header(names, count, "wrong-wrap rate at most", "1 cm2 / 4 cm2");
        for (int j = 0; j < 4; j++) {
            printf("| %.2f |", rates[j]);
            for (int i = 0; i < distinct; i++)
                printf(" %.0f / %.0f |", studies[i].area[t][j][0], studies[i].area[t][j][1]);
            putchar('\n');
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fputs("supply at least one wrap study\n", stderr);
        return 1;
    }
    sqlite3 *db = NULL;
    sqlite3_stmt *insert = NULL;
    Study *studies = calloc((size_t)argc - 1, sizeof(*studies));
    int ok = studies && sqlite3_open(":memory:", &db) == SQLITE_OK &&
        sqlite3_exec(db,
            "CREATE TABLE results(id INTEGER,control TEXT NOT NULL,data TEXT NOT NULL,"
            "PRIMARY KEY(id,control)) WITHOUT ROWID",
            NULL, NULL, NULL) == SQLITE_OK &&
        sqlite3_prepare_v2(db,
            "INSERT OR REPLACE INTO results "
            "SELECT ?,json_extract(value,'$.control'),value FROM json_each(?)",
            -1, &insert, NULL) == SQLITE_OK;
    int distinct = 0;
    for (int i = 1; ok && i < argc; i++) {
        int j = 0;
        while (j < distinct && strcmp(studies[j].name, argv[i]) != 0)
            j++;
        if (j == distinct) {
            studies[distinct++].name = argv[i];
            ok = insert_rows(insert, j, argv[i]);
        }
    }
    sqlite3_finalize(insert);
    if (ok)
        ok = print_report(db, studies, distinct, argv + 1, argc - 1);
    if (!ok && db && sqlite3_errcode(db) != SQLITE_OK)
        fprintf(stderr, "wrap study: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    free(studies);
    return ok ? 0 : 1;
}
