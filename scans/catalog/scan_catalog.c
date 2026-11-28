#define _GNU_SOURCE
#include <curl/curl.h>
#include <errno.h>
#include <fcntl.h>
#include <json-c/json.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include "../../storage/data_roots.c"

const char hx_usage[] =
    "Index of every scan, volume and segment product in the open-data bucket's metadata.";
static const char bucket[] = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";

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

static int fetch_file(const char *origin, const char *path) {
    char *url = path_join(origin, "metadata.min.json"), *temporary = NULL;
    int ok = url && asprintf(&temporary, "%s.XXXXXX", path) >= 0 && folders(temporary);
    FILE *payload = ok ? tmpfile() : NULL;
    CURL *curl = payload ? curl_easy_init() : NULL;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, payload);
        ok = curl_easy_perform(curl) == CURLE_OK;
        curl_easy_cleanup(curl);
    } else {
        ok = 0;
    }
    ok = ok && fseek(payload, 0, SEEK_SET) == 0;
    int input = ok ? dup(fileno(payload)) : -1;
    gzFile contents = input >= 0 ? gzdopen(input, "rb") : NULL;
    if (input >= 0 && !contents)
        close(input);
    int fd = contents ? mkstemp(temporary) : -1;
    FILE *out = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (fd >= 0 && !out)
        close(fd);
    unsigned char block[65536];
    ok = out != NULL;
    while (ok) {
        int n = gzread(contents, block, sizeof(block));
        if (!n)
            break;
        ok = n > 0 && fwrite(block, 1, (size_t)n, out) == (size_t)n;
    }
    if (out && fclose(out))
        ok = 0;
    if (contents && gzclose(contents) != Z_OK)
        ok = 0;
    if (payload)
        fclose(payload);
    if (ok)
        ok = rename(temporary, path) == 0;
    if (!ok && fd >= 0)
        unlink(temporary);
    free(temporary);
    free(url);
    return ok;
}

typedef json_object Node;
typedef struct {
    struct json_object_iterator iter, end;
} Cursor;
static Node *mapping(void) { return json_object_new_object(); }
static Node *sequence(void) { return json_object_new_array(); }
static Node *str_node(const char *s, size_t n) { return json_object_new_string_len(s, (int)n); }
static const char *string_value(Node *s) { return json_object_get_string(s); }
static int is_map(Node *o) { return json_object_is_type(o, json_type_object); }
static size_t fields(Node *o) { return (size_t)json_object_object_length(o); }
static size_t elements(Node *o) {
    return json_object_is_type(o, json_type_array) ? json_object_array_length(o) : 0;
}
static Node *element(Node *o, size_t i) {
    return i < elements(o) ? json_object_array_get_idx(o, i) : NULL;
}
static int lookup(Node *o, const char *key, Node **value) {
    return json_object_object_get_ex(o, key, value);
}
static Cursor entries(Node *o) {
    return (Cursor){.iter = json_object_iter_begin(o), .end = json_object_iter_end(o)};
}
static const char *next_entry(Cursor *c, Node **value) {
    if (json_object_iter_equal(&c->iter, &c->end))
        return NULL;
    const char *key = json_object_iter_peek_name(&c->iter);
    *value = json_object_iter_peek_value(&c->iter);
    json_object_iter_next(&c->iter);
    return key;
}
static void drop(Node *o) { json_object_put(o); }
static int add(Node *o, Node *v) { return json_object_array_add(o, v) == 0; }
static int assign(Node *o, const char *key, Node *v) {
    return json_object_object_add(o, key, v) == 0;
}
static Node *retain(Node *v) { return json_object_get(v); }

static Node *value_of(Node *o, const char *key) {
    Node *value = NULL;
    lookup(o, key, &value);
    return value;
}

static Node *required(Node *o, const char *key, const char **error) {
    Node *value = NULL;
    if (!lookup(o, key, &value) && !*error)
        *error = key;
    return value;
}

static Node *root_url(Node *origin, const char **error) {
    Node *roots = required(origin, "access_roots", error);
    const char *url = string_value(required(element(roots, 0), "url", error));
    if (!url) {
        if (!*error)
            *error = "url";
        return NULL;
    }
    if (!strncmp(url, "s3://", 5)) {
        char *rendered = NULL;
        if (asprintf(&rendered, "https://%s.s3.us-east-1.amazonaws.com", url + 5) < 0)
            return NULL;
        Node *root = str_node(rendered, strlen(rendered));
        free(rendered);
        return root;
    }
    size_t n = strlen(url);
    while (n && url[n - 1] == '/')
        n--;
    return str_node(url, n);
}

static int fill_products(Node *data, Node *paths, Node *roots, const char **error) {
    for (size_t i = 0; i < elements(data); i++) {
        Node *p = element(data, i);
        const char *kind = string_value(required(p, "type", error));
        Node *origin = element(required(p, "origins", error), 0);
        Node *path = required(origin, "path", error);
        if (*error || !kind)
            return 0;
        if (roots) {
            if (!assign(paths, kind, retain(path)) || !assign(roots, kind, root_url(origin, error)))
                return 0;
        } else {
            Node *list = value_of(paths, kind);
            if (!list) {
                list = sequence();
                if (!list || !assign(paths, kind, list))
                    return 0;
            }
            if (!add(list, retain(path)))
                return 0;
        }
    }
    return !*error;
}

static Node *product_row(
    const char *name, const char *id, Node *owner, Node *item, int segment, const char **error) {
    Node *r = mapping(), *paths = mapping(), *data = NULL, *roots = NULL;
    int ok = r && paths && assign(r, "sample", str_node(name, strlen(name)));
    if (ok && segment) {
        Node *suffix = NULL;
        int present = lookup(item, "suffix", &suffix);
        ok = assign(r, "segment", str_node(id, strlen(id))) &&
            assign(r, "suffix", present ? retain(suffix) : str_node("", 0));
        data = value_of(item, "data");
    } else if (ok) {
        Node *features = required(required(owner, "sample", error), "properties", error);
        Node *props = required(item, "properties", error);
        roots = mapping();
        ok = roots && assign(r, "type", retain(value_of(features, "type"))) &&
            assign(r, "volume", str_node(id, strlen(id))) &&
            assign(r, "voxel_um", retain(value_of(props, "pixel_size_um"))) &&
            assign(r, "energy_kev", retain(value_of(props, "energy_keV")));
        data = required(item, "data", error);
    }
    ok = ok && !*error && fill_products(data, paths, roots, error);
    if (ok)
        ok = assign(r, "products", paths);
    else
        drop(paths);
    if (roots) {
        if (ok)
            ok = assign(r, "roots", roots);
        else
            drop(roots);
    }
    if (ok)
        return r;
    drop(r);
    return NULL;
}

static int by_name(const void *left, const void *right) {
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

static Node *catalog_rows(Node *samples, int segment, const char **error) {
    if (!is_map(samples)) {
        *error = "samples";
        return NULL;
    }
    size_t count = fields(samples), n = 0;
    const char **names = calloc(count ? count : 1, sizeof(*names));
    Node *rows = sequence(), *owner;
    int ok = names && rows;
    Cursor sample_iter = entries(samples);
    while (ok) {
        const char *name = next_entry(&sample_iter, &owner);
        if (!name)
            break;
        names[n++] = name;
    }
    if (ok)
        qsort(names, n, sizeof(*names), by_name);
    for (size_t i = 0; i < n && ok; i++) {
        owner = value_of(samples, names[i]);
        const char *group = segment ? "segments" : "volumes";
        Node *items = required(owner, group, error);
        if (!is_map(items)) {
            *error = group;
            ok = 0;
            break;
        }
        Cursor iter = entries(items);
        Node *item;
        while (ok) {
            const char *id = next_entry(&iter, &item);
            if (!id)
                break;
            Node *row = product_row(names[i], id, owner, item, segment, error);
            ok = row && add(rows, row);
        }
    }
    free(names);
    if (ok && !*error)
        return rows;
    drop(rows);
    return NULL;
}

static int report(json_object *samples) {
    const char *error = NULL;
    json_object *segments = catalog_rows(samples, 1, &error);
    json_object *volumes = segments ? catalog_rows(samples, 0, &error) : NULL;
    int ok = segments && volumes;
    for (size_t i = 0; ok && i < json_object_array_length(segments); i++) {
        json_object *r = json_object_array_get_idx(segments, i);
        json_object *products = json_object_object_get(r, "products");
        json_object *paths = json_object_object_get(products, "ink-labels");
        size_t count = paths ? json_object_array_length(paths) : 0;
        for (size_t j = 0; ok && j < count; j++) {
            const char *path = json_object_get_string(json_object_array_get_idx(paths, j));
            const char *part = path ? strstr(path, "/ink-labels/") : NULL;
            if (!part) {
                ok = 0;
                break;
            }
            part += strlen("/ink-labels/");
            const char *end = strstr(part, "/ink-labels/");
            const char *keys[] = {"sample", "segment", "suffix"};
            for (int k = 0; k < 3; k++) {
                const char *value = json_object_get_string(json_object_object_get(r, keys[k]));
                printf("%s ", value ? value : "None");
            }
            printf("%.*s\n", (int)(end ? (size_t)(end - part) : strlen(part)), part);
        }
    }
    if (ok)
        fprintf(stderr, "%zu volumes %zu segments\n", json_object_array_length(volumes),
            json_object_array_length(segments));
    else
        fprintf(stderr, "incomplete catalogue near %s\n", error ? error : "ink-labels");
    json_object_put(volumes);
    json_object_put(segments);
    return ok;
}

static json_object *cached_catalog(char **path) {
    char *root = local_root(library_root);
    char *location = root ? path_join(root, "catalog/metadata.json") : NULL;
    free(root);
    int ok = location && (access(location, F_OK) == 0 || fetch_file(bucket, location));
    json_object *doc = ok ? json_object_from_file(location) : NULL;
    if (path) {
        *path = location;
    } else {
        free(location);
    }
    return doc;
}

int main(void) {
    char *path;
    json_object *doc = cached_catalog(&path);
    int ok = doc && report(json_object_object_get(doc, "samples"));
    if (!doc) {
        fprintf(stderr, "cannot read catalogue %s\n", path ? path : "");
    }
    json_object_put(doc);
    free(path);
    return ok ? 0 : 1;
}
