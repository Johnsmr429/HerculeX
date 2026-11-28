#define _GNU_SOURCE
#include <curl/curl.h>
#include <errno.h>
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../storage/data_roots.c"

#ifndef HX_UNITY
const char hx_usage[] = "Keep and show the native scroll axis used by every geometry stage.";
#endif

typedef struct {
    char *source;
    char *maker;
    char *where;
    int remote;
    int rank;
} Entry;

typedef struct {
    char *library;
    char *records;
    char *lines;
    char *route;
} Roots;

typedef struct {
    double z;
    double x;
    double y;
} Point;

static char *absolute_path(const char *path) {
    if (path[0] == '/')
        return strdup(path);
    char *cwd = getcwd(NULL, 0);
    char *result = cwd ? path_join(cwd, path) : NULL;
    free(cwd);
    return result;
}

static void roots_close(Roots *roots) {
    free(roots->library);
    free(roots->records);
    free(roots->lines);
    free(roots->route);
    memset(roots, 0, sizeof(*roots));
}

static int roots_open(Roots *roots) {
    memset(roots, 0, sizeof(*roots));
    char *library = local_root(library_root);
    char *outputs = local_root(outputs_root);
    roots->library = library ? absolute_path(library) : NULL;
    char *absolute_outputs = outputs ? absolute_path(outputs) : NULL;
    free(library);
    free(outputs);
    char *cwd = getcwd(NULL, 0);
    roots->records = cwd ? path_join(cwd, "records/umbilici") : NULL;
    roots->lines = roots->records ? path_join(roots->records, "vesuvius-center-lines") : NULL;
    char *unwrap = absolute_outputs ? path_join(absolute_outputs, "unwrapping") : NULL;
    roots->route = unwrap ? path_join(unwrap, "level") : NULL;
    free(unwrap);
    free(absolute_outputs);
    free(cwd);
    if (roots->library && roots->records && roots->lines && roots->route)
        return 1;
    roots_close(roots);
    return 0;
}

static int regular(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static int folders(char *path) {
    for (char *p = path + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = 0;
        int ok = mkdir(path, 0777) == 0 || errno == EEXIST;
        *p = '/';
        if (!ok)
            return 0;
    }
    return 1;
}

static char *axis_path(Roots *roots, const char *sample, const char *scan, const char *name) {
    char *path = NULL;
    if (asprintf(&path, "%s/%s/%s/sheet/%s", roots->route, sample, scan, name) < 0)
        return NULL;
    return path;
}

static int person_rank(const char *source, const char *kind) {
    size_t n = strlen(source);
    int mapped = n > 7 && !strcmp(source + n - 7, "-mapped");
    char *base = mapped ? strndup(source, n - 7) : strdup(source);
    const char *carried[] = {"bucket", "drobkov", "tauil", "dopico"};
    const char *models[] = {"finder-v2", "finder-polished", "finder"};
    int rank = 300;
    if (!strcmp(kind, "person"))
        rank = !strcmp(base, "bucket") ? 0 : 10;
    if (mapped) {
        for (size_t i = 0; i < sizeof(carried) / sizeof(*carried); i++) {
            if (!strcmp(base, carried[i]))
                rank = !i ? 100 : 110;
        }
    }
    for (size_t i = 0; i < sizeof(models) / sizeof(*models); i++) {
        if (!strcmp(base, models[i]))
            rank = 200 + (int)i;
    }
    free(base);
    return rank;
}

static void entry_free(Entry *entry) {
    free(entry->source);
    free(entry->maker);
    free(entry->where);
}

static int entry_order(const void *left, const void *right) {
    const Entry *a = left, *b = right;
    if (a->rank != b->rank)
        return a->rank - b->rank;
    int source = strcmp(a->source, b->source);
    return source ? source : strcmp(a->where, b->where);
}

static int add_entry(Entry **entries, size_t *used, size_t *capacity, const char *source,
    const char *maker, const char *where, int remote, int rank) {
    for (size_t i = 0; i < *used; i++) {
        if (!strcmp((*entries)[i].where, where))
            return 1;
    }
    if (*used == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 8;
        Entry *grown = realloc(*entries, next * sizeof(*grown));
        if (!grown)
            return 0;
        *entries = grown;
        *capacity = next;
    }
    Entry row = {.source = strdup(source),
        .maker = strdup(maker),
        .where = strdup(where),
        .remote = remote,
        .rank = rank};
    if (!row.source || !row.maker || !row.where) {
        entry_free(&row);
        return 0;
    }
    (*entries)[(*used)++] = row;
    return 1;
}

static int chosen_entry(Roots *roots, const char *sample, const char *scan, Entry **entries,
    size_t *used, size_t *capacity) {
    char *path = path_join(roots->records, "chosen.json");
    json_object *document = path ? json_object_from_file(path) : NULL;
    free(path);
    char *key = NULL;
    int made = asprintf(&key, "%s-%s", sample, scan) >= 0;
    json_object *entry, *value;
    int ok = 1;
    if (made && document && json_object_object_get_ex(document, key, &entry)) {
        const char *source = json_object_object_get_ex(entry, "source", &value)
            ? json_object_get_string(value)
            : "chosen";
        const char *maker = json_object_object_get_ex(entry, "maker", &value)
            ? json_object_get_string(value)
            : "person";
        const char *file =
            json_object_object_get_ex(entry, "file", &value) ? json_object_get_string(value) : NULL;
        char *line = file ? path_join(roots->lines, file) : NULL;
        if (line && regular(line))
            ok = add_entry(entries, used, capacity, source, maker, line, 0, -1);
        free(line);
    }
    free(key);
    if (document)
        json_object_put(document);
    return ok;
}

static int index_entries(Roots *roots, const char *sample, const char *scan, Entry **entries,
    size_t *used, size_t *capacity) {
    char *path = path_join(roots->lines, "index.json");
    json_object *document = path ? json_object_from_file(path) : NULL;
    free(path);
    if (!document)
        return 1;
    const char *lists[] = {"files", "other_scans"};
    int ok = 1;
    for (size_t k = 0; k < sizeof(lists) / sizeof(*lists) && ok; k++) {
        json_object *rows;
        if (!json_object_object_get_ex(document, lists[k], &rows))
            continue;
        size_t count = json_object_array_length(rows);
        for (size_t i = 0; i < count && ok; i++) {
            json_object *row = json_object_array_get_idx(rows, i), *value;
            const char *held_sample = json_object_object_get_ex(row, "scroll", &value)
                ? json_object_get_string(value)
                : "";
            const char *held_scan =
                json_object_object_get_ex(row, "scan", &value) ? json_object_get_string(value) : "";
            if (strcmp(held_sample, sample) != 0 || strcmp(held_scan, scan) != 0 ||
                json_object_object_get_ex(row, "superseded_by", &value))
                continue;
            const char *source = json_object_object_get_ex(row, "source", &value)
                ? json_object_get_string(value)
                : "unknown";
            const char *kind = json_object_object_get_ex(row, "kind", &value)
                ? json_object_get_string(value)
                : "model";
            const char *file = json_object_object_get_ex(row, "file", &value)
                ? json_object_get_string(value)
                : NULL;
            char *line = file ? path_join(roots->lines, file) : NULL;
            if (line && regular(line))
                ok = add_entry(
                    entries, used, capacity, source, kind, line, 0, person_rank(source, kind));
            free(line);
        }
    }
    json_object_put(document);
    return ok;
}

static char *catalog_url(Roots *roots, const char *sample, const char *scan) {
    char *path = path_join(roots->library, "catalog/metadata.json");
    json_object *document = path ? json_object_from_file(path) : NULL;
    free(path);
    json_object *samples, *scroll, *volumes, *volume, *data;
    int found = document && json_object_object_get_ex(document, "samples", &samples) &&
        json_object_object_get_ex(samples, sample, &scroll) &&
        json_object_object_get_ex(scroll, "volumes", &volumes) &&
        json_object_object_get_ex(volumes, scan, &volume) &&
        json_object_object_get_ex(volume, "data", &data);
    char *url = NULL;
    if (found) {
        size_t count = json_object_array_length(data);
        for (size_t i = 0; i < count && !url; i++) {
            json_object *row = json_object_array_get_idx(data, i), *value;
            const char *type =
                json_object_object_get_ex(row, "type", &value) ? json_object_get_string(value) : "";
            if (strcmp(type, "umbilicus") != 0)
                continue;
            json_object *origins, *origin, *access, *root_value;
            if (!json_object_object_get_ex(row, "origins", &origins) ||
                !json_object_array_length(origins))
                continue;
            origin = json_object_array_get_idx(origins, 0);
            const char *remote_path = json_object_object_get_ex(origin, "path", &value)
                ? json_object_get_string(value)
                : NULL;
            if (!remote_path || !json_object_object_get_ex(origin, "access_roots", &access) ||
                !json_object_array_length(access))
                continue;
            root_value = json_object_array_get_idx(access, 0);
            const char *root = json_object_object_get_ex(root_value, "url", &value)
                ? json_object_get_string(value)
                : NULL;
            char *https = NULL;
            if (root && !strncmp(root, "s3://", 5))
                asprintf(&https, "https://%s.s3.us-east-1.amazonaws.com", root + 5);
            else if (root)
                https = strdup(root);
            if (https) {
                size_t n = strlen(https);
                while (n && https[n - 1] == '/')
                    https[--n] = 0;
                while (*remote_path == '/')
                    remote_path++;
                asprintf(&url, "%s/%s", https, remote_path);
            }
            free(https);
        }
    }
    if (document)
        json_object_put(document);
    return url;
}

static Entry *on_hand(
    Roots *roots, const char *sample, const char *scan, size_t *count, int with_catalog) {
    Entry *entries = NULL;
    size_t used = 0, capacity = 0;
    int ok = chosen_entry(roots, sample, scan, &entries, &used, &capacity) &&
        index_entries(roots, sample, scan, &entries, &used, &capacity);
    if (ok && used > 1)
        qsort(entries, used, sizeof(*entries), entry_order);
    if (ok && with_catalog) {
        char *url = catalog_url(roots, sample, scan);
        if (url) {
            ok = add_entry(&entries, &used, &capacity, "catalog", "organisers", url, 1, 1000);
            free(url);
        }
    }
    if (!ok) {
        for (size_t i = 0; entries && i < used; i++)
            entry_free(&entries[i]);
        free(entries);
        return NULL;
    }
    *count = entries ? used : 0;
    return entries;
}

static json_object *read_remote(const char *url) {
    FILE *payload = tmpfile();
    CURL *curl = payload ? curl_easy_init() : NULL;
    int ok = curl != NULL;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, payload);
        ok = curl_easy_perform(curl) == CURLE_OK;
        curl_easy_cleanup(curl);
    }
    ok = ok && fseek(payload, 0, SEEK_SET) == 0;
    int fd = ok ? dup(fileno(payload)) : -1;
    json_object *document = fd >= 0 ? json_object_from_fd(fd) : NULL;
    if (fd >= 0)
        close(fd);
    if (payload)
        fclose(payload);
    return document;
}

static int point_order(const void *left, const void *right) {
    const Point *a = left, *b = right;
    return a->z < b->z ? -1 : a->z > b->z;
}

static json_object *axis_of(json_object *document) {
    json_object *rows;
    if (!json_object_object_get_ex(document, "control_points", &rows) ||
        !json_object_is_type(rows, json_type_array))
        return NULL;
    size_t count = json_object_array_length(rows);
    Point *points = calloc(count ? count : 1, sizeof(*points));
    if (!points)
        return NULL;
    for (size_t i = 0; i < count; i++) {
        json_object *row = json_object_array_get_idx(rows, i), *value;
        points[i].z =
            json_object_object_get_ex(row, "z", &value) ? json_object_get_double(value) : 0;
        points[i].x =
            json_object_object_get_ex(row, "x", &value) ? json_object_get_double(value) : 0;
        points[i].y =
            json_object_object_get_ex(row, "y", &value) ? json_object_get_double(value) : 0;
    }
    qsort(points, count, sizeof(*points), point_order);
    json_object *axis = json_object_new_array();
    for (size_t i = 0; axis && i < count; i++) {
        json_object *point = json_object_new_array();
        json_object_array_add(point, json_object_new_double(points[i].z));
        json_object_array_add(point, json_object_new_double(points[i].x));
        json_object_array_add(point, json_object_new_double(points[i].y));
        json_object_array_add(axis, point);
    }
    free(points);
    return axis;
}

static int atomic_json(const char *path, json_object *document, int pretty) {
    char *part = NULL;
    int made = asprintf(&part, "%s.%ld", path, (long)getpid()) >= 0 && folders(part);
    int ok = made &&
        json_object_to_file_ext(
            part, document, pretty ? JSON_C_TO_STRING_PRETTY : JSON_C_TO_STRING_PLAIN) == 0 &&
        rename(part, path) == 0;
    if (part && !ok)
        unlink(part);
    free(part);
    return ok;
}

static int keep(Roots *roots, const char *sample, const char *scan, const char *source) {
    char *axis_pathname = axis_path(roots, sample, scan, "axis.json");
    if (!source && axis_pathname && regular(axis_pathname)) {
        free(axis_pathname);
        return 1;
    }
    size_t count = 0;
    Entry *entries = on_hand(roots, sample, scan, &count, 1);
    Entry *selected = NULL;
    for (size_t i = 0; entries && i < count && !selected; i++) {
        if (!source || !strcmp(source, entries[i].source))
            selected = &entries[i];
    }
    if (!selected) {
        fprintf(stderr, "%s %s: no umbilicus%s%s on hand\n", sample, scan, source ? " from " : "",
            source ? source : "");
        for (size_t i = 0; entries && i < count; i++)
            entry_free(&entries[i]);
        free(entries);
        free(axis_pathname);
        return 0;
    }
    json_object *document =
        selected->remote ? read_remote(selected->where) : json_object_from_file(selected->where);
    json_object *axis = document ? axis_of(document) : NULL;
    json_object *told = json_object_new_object();
    json_object_object_add(told, "source", json_object_new_string(selected->source));
    json_object_object_add(told, "maker", json_object_new_string(selected->maker));
    const char *from = selected->where;
    if (!selected->remote && !strncmp(from, roots->records, strlen(roots->records))) {
        from += strlen(roots->records);
        if (*from == '/')
            from++;
    }
    json_object_object_add(told, "from", json_object_new_string(from));
    char *source_path = axis_path(roots, sample, scan, "axis-source.json");
    int ok = axis && axis_pathname && source_path && atomic_json(source_path, told, 1) &&
        atomic_json(axis_pathname, axis, 0);
    if (!ok)
        fprintf(stderr, "HerculeX: could not keep the umbilicus for %s %s\n", sample, scan);
    free(source_path);
    free(axis_pathname);
    if (axis)
        json_object_put(axis);
    json_object_put(told);
    if (document)
        json_object_put(document);
    for (size_t i = 0; entries && i < count; i++)
        entry_free(&entries[i]);
    free(entries);
    return ok;
}

int hx_umbilicus_keep(const char *sample, const char *scan) {
    Roots roots;
    if (!roots_open(&roots)) {
        return 1;
    }
    int status = !keep(&roots, sample, scan, NULL);
    roots_close(&roots);
    return status;
}

static int show(Roots *roots, const char *sample, const char *scan) {
    char *axis = axis_path(roots, sample, scan, "axis.json");
    char *source = axis_path(roots, sample, scan, "axis-source.json");
    json_object *result = json_object_new_object();
    json_object *kept = source ? json_object_from_file(source) : NULL;
    if (kept)
        json_object_object_add(result, "kept", kept);
    else if (axis && regular(axis))
        json_object_object_add(result, "kept", json_object_new_string("kept"));
    else
        json_object_object_add(result, "kept", json_object_new_null());
    size_t count = 0;
    Entry *entries = on_hand(roots, sample, scan, &count, 1);
    json_object *list = json_object_new_array();
    for (size_t i = 0; entries && i < count; i++) {
        json_object *row = json_object_new_array();
        json_object_array_add(row, json_object_new_string(entries[i].source));
        json_object_array_add(row, json_object_new_string(entries[i].maker));
        json_object_array_add(row, json_object_new_string(entries[i].where));
        json_object_array_add(list, row);
        entry_free(&entries[i]);
    }
    free(entries);
    json_object_object_add(result, "on_hand", list);
    puts(json_object_to_json_string_ext(result, JSON_C_TO_STRING_PRETTY));
    json_object_put(result);
    free(source);
    free(axis);
    return 0;
}

static void usage(FILE *out) {
    fputs("Usage: ./run.sh level.level.umbilicus SCROLL SCAN [SOURCE]\n"
          "       ./run.sh level.level.umbilicus check SCROLL SCAN\n",
        out);
}

int hx_umbilicus_main(int argc, char **argv) {
    int help = argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"));
    if (help || argc < 3) {
        usage(help ? stdout : stderr);
        return help ? 0 : 2;
    }
    Roots roots;
    if (!roots_open(&roots))
        return 1;
    int code;
    if (!strcmp(argv[1], "check")) {
        if (argc != 4) {
            usage(stderr);
            code = 2;
        } else {
            char *axis = axis_path(&roots, argv[2], argv[3], "axis.json");
            size_t count = 0;
            Entry *entries = on_hand(&roots, argv[2], argv[3], &count, 1);
            code = (axis && regular(axis)) || count ? 0 : 1;
            for (size_t i = 0; entries && i < count; i++)
                entry_free(&entries[i]);
            free(entries);
            free(axis);
        }
    } else if (argc > 4) {
        usage(stderr);
        code = 2;
    } else {
        code = argc == 4 && !keep(&roots, argv[1], argv[2], argv[3])
            ? 1
            : show(&roots, argv[1], argv[2]);
    }
    roots_close(&roots);
    return code;
}

#if !defined(HX_UNITY)
int main(int argc, char **argv) { return hx_umbilicus_main(argc, argv); }
#endif
