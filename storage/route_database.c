#define _GNU_SOURCE
#include <glob.h>
#include <json-c/json.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "route_record.h"

#include "data_roots.c"

const char hx_usage_route_database[] = "Rebuild the index of finished route units and pieces.";

static const char native_schema[] =
    "CREATE TABLE IF NOT EXISTS units(stage TEXT,sample TEXT,volume TEXT,unit TEXT,method TEXT,"
    "seconds REAL,record TEXT,PRIMARY KEY(stage,sample,volume,unit));"
    "CREATE TABLE IF NOT EXISTS pieces(sample TEXT,volume TEXT,unit TEXT,piece INTEGER,"
    "area_mm2 REAL,nodes INTEGER,winding INTEGER,frame INTEGER,written INTEGER,"
    "PRIMARY KEY(sample,volume,unit,piece));";

typedef struct {
    sqlite3 *database;
    sqlite3_stmt *unit;
    sqlite3_stmt *piece;
    int nodes;
    int pieces;
} NativeIndex;

static int directory(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISDIR(status.st_mode);
}

static int bind_text(sqlite3_stmt *statement, int field, const char *value) {
    int code = value ? sqlite3_bind_text(statement, field, value, -1, SQLITE_TRANSIENT)
                     : sqlite3_bind_null(statement, field);
    return code == SQLITE_OK;
}

static int insert(sqlite3_stmt *statement) {
    int code = sqlite3_step(statement);
    sqlite3_reset(statement);
    sqlite3_clear_bindings(statement);
    return code == SQLITE_DONE;
}

static int native_pieces(NativeIndex *index, const char *sample, const char *volume,
    const char *unit, json_object *record) {
    static const char *fields[] = {"piece", "area_mm2", "nodes", "winding", "frame", "written"};
    json_object *rows = json_object_object_get(record, "pieces");
    if (!json_object_is_type(rows, json_type_array))
        return 1;
    sqlite3_stmt *statement = index->piece;
    for (size_t i = 0; i < json_object_array_length(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        if (!bind_text(statement, 1, sample) || !bind_text(statement, 2, volume) ||
            !bind_text(statement, 3, unit))
            return 0;
        for (int j = 0; j < (int)(sizeof(fields) / sizeof(*fields)); j++) {
            json_object *value = json_object_object_get(row, fields[j]);
            int code = j == 1 ? sqlite3_bind_double(statement, j + 4, json_object_get_double(value))
                              : sqlite3_bind_int64(statement, j + 4, json_object_get_int64(value));
            if (code != SQLITE_OK)
                return 0;
        }
        if (!insert(statement))
            return 0;
    }
    return 1;
}

static int native_record(NativeIndex *index, const char *stage, const char *sample,
    const char *volume, const char *unit, const char *path) {
    json_object *record = json_object_from_file(path), *method;
    if (!record)
        return 0;
    const char *method_name = json_object_object_get_ex(record, "method", &method)
        ? json_object_get_string(method)
        : NULL;
    const char *encoded = json_object_to_json_string_ext(record, JSON_C_TO_STRING_PLAIN);
    sqlite3_stmt *statement = index->unit;
    int ok = bind_text(statement, 1, stage) && bind_text(statement, 2, sample) &&
        bind_text(statement, 3, volume) && bind_text(statement, 4, unit) &&
        bind_text(statement, 5, method_name) &&
        sqlite3_bind_double(statement, 6, route_seconds(record)) == SQLITE_OK &&
        bind_text(statement, 7, encoded) && insert(statement);
    if (ok && !strcmp(stage, "pieces") && strcmp(unit, "joints") != 0 && strcmp(unit, "frame") != 0)
        ok = native_pieces(index, sample, volume, unit, record);
    if (ok) {
        if (!strcmp(stage, "sheet-nodes"))
            index->nodes++;
        else
            index->pieces++;
    }
    json_object_put(record);
    return ok;
}

static int native_stage(NativeIndex *index, const char *base, const char *folder, const char *stage,
    const char *sample, const char *volume) {
    char *pattern = NULL;
    if (asprintf(&pattern, "%s/%s/*/record.json", base, folder) < 0)
        return 0;
    glob_t records = {0};
    int found = glob(pattern, 0, NULL, &records) == 0;
    free(pattern);
    int ok = 1;
    if (found) {
        for (size_t i = 0; i < records.gl_pathc && ok; i++) {
            char *copy = strdup(records.gl_pathv[i]);
            char *end = copy ? strrchr(copy, '/') : NULL;
            if (end)
                *end = 0;
            const char *unit = end ? strrchr(copy, '/') : NULL;
            unit = unit ? unit + 1 : NULL;
            if (unit && !strstr(unit, ".building"))
                ok = native_record(index, stage, sample, volume, unit, records.gl_pathv[i]);
            free(copy);
        }
    }
    globfree(&records);
    return ok;
}

static int native_scan(NativeIndex *index, const char *scan) {
    char *copy = strdup(scan);
    char *volume = copy ? strrchr(copy, '/') : NULL;
    if (!volume) {
        free(copy);
        return 0;
    }
    *volume++ = 0;
    const char *sample = strrchr(copy, '/');
    sample = sample ? sample + 1 : NULL;
    int ok = sample && native_stage(index, scan, "nodes", "sheet-nodes", sample, volume) &&
        native_stage(index, scan, "pieces", "pieces", sample, volume);
    free(copy);
    return ok;
}

static int rebuild(const char *route, const char *database) {
    NativeIndex index = {0};
    int code = sqlite3_open_v2(database, &index.database,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, NULL);
    if (code != SQLITE_OK) {
        fprintf(stderr, "HerculeX: %s\n", sqlite3_errmsg(index.database));
        sqlite3_close(index.database);
        return 1;
    }
    sqlite3_busy_timeout(index.database, 5000);
    char *error = NULL;
    code = sqlite3_exec(index.database, "PRAGMA journal_mode=WAL", NULL, NULL, &error);
    if (code == SQLITE_OK)
        code = sqlite3_exec(index.database, native_schema, NULL, NULL, &error);
    if (code == SQLITE_OK)
        code = sqlite3_exec(index.database, "BEGIN IMMEDIATE;DELETE FROM units;DELETE FROM pieces",
            NULL, NULL, &error);
    if (code == SQLITE_OK)
        code = sqlite3_prepare_v2(
            index.database, "INSERT INTO units VALUES (?,?,?,?,?,?,?)", -1, &index.unit, NULL);
    if (code == SQLITE_OK)
        code = sqlite3_prepare_v2(index.database, "INSERT INTO pieces VALUES (?,?,?,?,?,?,?,?,?)",
            -1, &index.piece, NULL);
    char *pattern = NULL;
    glob_t scans = {0};
    int ok = code == SQLITE_OK && asprintf(&pattern, "%s/*/*", route) >= 0;
    int found = ok && glob(pattern, 0, NULL, &scans) == 0;
    free(pattern);
    if (found) {
        for (size_t i = 0; i < scans.gl_pathc && ok; i++) {
            if (directory(scans.gl_pathv[i]))
                ok = native_scan(&index, scans.gl_pathv[i]);
        }
    }
    globfree(&scans);
    if (ok)
        code = sqlite3_exec(index.database, "COMMIT", NULL, NULL, &error);
    else
        sqlite3_exec(index.database, "ROLLBACK", NULL, NULL, NULL);
    if (code != SQLITE_OK || !ok)
        fprintf(stderr, "HerculeX: %s\n", error ? error : sqlite3_errmsg(index.database));
    sqlite3_free(error);
    sqlite3_finalize(index.unit);
    sqlite3_finalize(index.piece);
    sqlite3_close(index.database);
    if (code != SQLITE_OK || !ok)
        return 1;
    printf("{'sheet-nodes': %d, 'pieces': %d}\n", index.nodes, index.pieces);
    return 0;
}

int hx_route_index(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "index") != 0) {
        puts("Usage: ./run.sh storage.route_database index");
        return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
    }
    char *outputs = local_root(outputs_root);
    char *unwrap = outputs ? path_join(outputs, "unwrapping") : NULL;
    free(outputs);
    char *route = unwrap ? path_join(unwrap, "level") : NULL;
    char *database = unwrap ? path_join(unwrap, "level_route.sqlite") : NULL;
    int code = route && database ? rebuild(route, database) : 1;
    free(database);
    free(route);
    free(unwrap);
    return code;
}

#ifndef HX_UNITY
int main(int argc, char **argv) { return hx_route_index(argc, argv); }
#endif
