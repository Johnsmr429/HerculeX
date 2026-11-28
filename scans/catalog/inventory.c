#include <pthread.h>

static const char inventory_server[] = "https://dl.ash2txt.org";

#include "../chunks/bucket_listing.h"

static int inventory_write(const char *path, json_object *doc, int lines) {
    char *temp = NULL;
    if (asprintf(&temp, "%s.XXXXXX", path) < 0) {
        return 0;
    }
    int fd = mkstemp(temp);
    FILE *out = fd >= 0 ? fdopen(fd, "w") : NULL;
    int ok = out != NULL;
    if (ok && lines) {
        for (size_t i = 0; ok && i < json_object_array_length(doc); i++) {
            const char *line = json_object_to_json_string_ext(json_object_array_get_idx(doc, i),
                JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE);
            ok = fputs(line, out) >= 0 && fputc('\n', out) >= 0;
        }
    } else if (ok) {
        const char *text = json_object_to_json_string_ext(
            doc, JSON_C_TO_STRING_PRETTY | JSON_C_TO_STRING_NOSLASHESCAPE);
        ok = fputs(text, out) >= 0 && fputc('\n', out) >= 0;
    }
    if (out) {
        ok = fclose(out) == 0 && ok;
    } else if (fd >= 0) {
        close(fd);
    }
    ok = ok && !rename(temp, path);
    if (!ok && fd >= 0) {
        unlink(temp);
    }
    free(temp);
    return ok;
}

typedef struct {
    const char *kind;
    uint64_t bytes, count, open;
    size_t order;
} InventoryTotal;

static int inventory_count_order(const void *left, const void *right) {
    const InventoryTotal *a = left, *b = right;
    if (a->count != b->count) {
        return a->count > b->count ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static int inventory_byte_order(const void *left, const void *right) {
    const InventoryTotal *a = left, *b = right;
    if (a->bytes != b->bytes) {
        return a->bytes > b->bytes ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static int inventory_totals(json_object *rows, int server) {
    json_object *map = json_object_new_object();
    size_t count = 0, capacity = 0;
    InventoryTotal *totals = NULL;
    int ok = map != NULL;
    for (size_t i = 0; ok && i < json_object_array_length(rows); i++) {
        json_object *row = json_object_array_get_idx(rows, i), *index = NULL;
        const char *key = text_field(row, server ? "path" : "kind");
        char *group = NULL;
        if (server) {
            const char *begin = key + 1, *end = begin;
            for (int n = 0; n < 3; n++) {
                end += strcspn(end, "/");
                if (!*end) {
                    break;
                }
                if (n < 2) {
                    end++;
                }
            }
            key = group = strndup(begin, end - begin);
        }
        if (key && !json_object_object_get_ex(map, key, &index)) {
            if (count == capacity) {
                capacity = capacity ? capacity * 2 : 32;
                InventoryTotal *next = realloc(totals, capacity * sizeof(*next));
                if (next) {
                    totals = next;
                } else {
                    ok = 0;
                }
            }
            if (ok) {
                index = json_object_new_uint64(count);
                ok = index && !json_object_object_add(map, key, index);
                totals[count] = (InventoryTotal){.kind = strdup(key), .order = count};
                ok = ok && totals[count++].kind;
            }
        }
        free(group);
        if (!ok || !index) {
            ok = 0;
            break;
        }
        InventoryTotal *total = totals + json_object_get_uint64(index);
        uint64_t size = json_object_get_uint64(value_of(row, "bytes"));
        total->bytes += size;
        total->count++;
        total->open += !json_object_get_boolean(value_of(row, "complete"));
    }
    if (ok) {
        qsort(
            totals, count, sizeof(*totals), server ? inventory_byte_order : inventory_count_order);
        for (size_t i = 0; i < count && (!server || i < 40); i++) {
            const InventoryTotal *t = totals + i;
            if (server) {
                printf("%12.2f GB  %s\n", t->bytes / 1e9, t->kind);
            } else {
                printf("%-28s %5llu %10.2f GB  unsized %llu\n", t->kind,
                    (unsigned long long)t->count, t->bytes / 1e9, (unsigned long long)t->open);
            }
        }
    }
    for (size_t i = 0; i < count; i++) {
        free((char *)totals[i].kind);
    }
    free(totals);
    json_object_put(map);
    return ok;
}

static const char *inventory_root(int argc, char **argv, const char *fallback) {
    if (argc == 1) {
        return fallback;
    }
    return argc == 3 && !strcmp(argv[1], "--root") && *argv[2] ? argv[2] : NULL;
}

static char *inventory_base(const char *origin) {
    size_t n = strlen(origin);
    while (n && origin[n - 1] == '/') {
        n--;
    }
    return strndup(origin, n);
}

static int inventory_help(int argc, char **argv) {
    return argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"));
}

typedef struct {
    json_object *rows;
    const char *origin;
} ProductInventory;

static void inventory_product(void *data, size_t i) {
    ProductInventory *inventory = data;
    json_object *row = json_object_array_get_idx(inventory->rows, i);
    const char *root = text_field(row, "root"), *path = text_field(row, "path");
    int owned = !strcmp(root, bucket) || !strcmp(root, inventory->origin);
    if (!owned) {
        json_object_object_add(row, "complete", json_object_new_boolean(0));
        return;
    }
    InventoryListing listing = {0};
    char error[128] = "";
    CURL *curl = curl_easy_init();
    int ok = inventory_listing(curl, inventory->origin, path, 0, 3, &listing, error);
    if (curl) {
        curl_easy_cleanup(curl);
    }
    if (ok) {
        json_object_object_add(row, "keys", json_object_new_uint64(listing.count));
        json_object_object_add(row, "bytes", json_object_new_uint64(listing.bytes));
        json_object_object_add(row, "complete", json_object_new_boolean(listing.complete));
    } else {
        json_object_object_add(row, "error",
            json_object_new_string_len(error, strlen(error) > 120 ? 120 : strlen(error)));
    }
}

static int inventory_string_order(const void *left, const void *right) {
    const char *const *a = left, *const *b = right;
    return strcmp(*a, *b);
}

static json_object *inventory_products(json_object *samples) {
    size_t size = fields(samples), selected = 0;
    const char **names = calloc(size ? size : 1, sizeof(*names));
    json_object *rows = names ? sequence() : NULL;
    Cursor cursor = entries(samples);
    Node *sample = NULL;
    for (const char *name = rows ? next_entry(&cursor, &sample) : NULL; name;
        name = next_entry(&cursor, &sample)) {
        names[selected++] = name;
    }
    if (selected) {
        qsort(names, selected, sizeof(*names), inventory_string_order);
    }
    const char *types[] = {"volumes", "segments"};
    int ok = rows != NULL;
    const char *error = NULL;
    for (size_t i = 0; ok && i < selected; i++) {
        sample = value_of(samples, names[i]);
        for (size_t kind = 0; ok && kind < 2; kind++) {
            Node *items = value_of(sample, types[kind]), *item;
            if (!is_map(items)) {
                continue;
            }
            Cursor owners = entries(items);
            for (const char *owner = next_entry(&owners, &item); ok && owner;
                owner = next_entry(&owners, &item)) {
                Node *data = value_of(item, "data");
                for (size_t j = 0; ok && j < elements(data); j++) {
                    Node *product = element(data, j),
                         *origin = element(value_of(product, "origins"), 0);
                    Node *root = root_url(origin, &error);
                    Node *path = value_of(origin, "path"), *type = value_of(product, "type");
                    Node *row = root && path && type ? mapping() : NULL;
                    ok = row && assign(row, "sample", str_node(names[i], strlen(names[i]))) &&
                        assign(row, "owner", str_node(owner, strlen(owner))) &&
                        assign(row, "kind", retain(type)) && assign(row, "root", retain(root)) &&
                        assign(row, "path", retain(path)) && add(rows, retain(row));
                    drop(row);
                    drop(root);
                }
            }
        }
    }
    free(names);
    if (!ok) {
        json_object_put(rows);
        rows = NULL;
    }
    return rows;
}

int hx_product_inventory_main(int argc, char **argv) {
    const char *origin = inventory_root(argc, argv, bucket);
    if (!origin) {
        puts("Usage: ./run.sh scans.catalog.product_inventory [--root URL]");
        return inventory_help(argc, argv) ? 0 : 2;
    }
    char *catalog = NULL;
    json_object *meta = cached_catalog(&catalog), *samples = value_of(meta, "samples");
    json_object *rows = is_map(samples) ? inventory_products(samples) : NULL;
    char *path = NULL, *slash = catalog ? strrchr(catalog, '/') : NULL;
    char *base = inventory_base(origin);
    int ok = rows && slash && base &&
        asprintf(&path, "%.*s/products.jsonl", (int)(slash - catalog), catalog) >= 0 &&
        curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    if (ok) {
        ProductInventory inventory = {rows, base};
        inventory_parallel(inventory_product, &inventory, json_object_array_length(rows), 16);
        ok = inventory_write(path, rows, 1) && inventory_totals(rows, 0);
        curl_global_cleanup();
    }
    if (!ok) {
        fprintf(stderr, "HerculeX: could not write %s\n", path ? path : "products.jsonl");
    }
    free(base);
    free(path);
    free(catalog);
    json_object_put(rows);
    json_object_put(meta);
    return !ok;
}

typedef struct {
    json_object *files, *folders;
    int failed, store;
} InventoryPage;

typedef struct {
    json_object *frontier;
    InventoryPage *pages;
    const char *origin;
} ServerInventory;

static char *inventory_quote_path(CURL *curl, const char *path) {
    char *quoted = curl_easy_escape(curl, path, 0);
    if (quoted) {
        char *out = quoted;
        for (const char *p = quoted; *p;) {
            if (!strncmp(p, "%2F", 3)) {
                *out++ = '/';
                p += 3;
            } else {
                *out++ = *p++;
            }
        }
        *out = 0;
    }
    return quoted;
}

static int inventory_store(const char *folder) {
    static const char *skip[] = {"volumes", "volumes_masked", "volumes_small", "volumes_zarr",
        "volumes_zarr_standardized", "volume_grids", "normal_grids", "thaumato_outputs", "layers",
        "surface_volume", "representations"};
    size_t n = strlen(folder);
    while (n && folder[n - 1] == '/') {
        n--;
    }
    size_t first = n;
    while (first && folder[first - 1] != '/') {
        first--;
    }
    for (size_t i = 0; i < sizeof(skip) / sizeof(*skip); i++) {
        if (strlen(skip[i]) == n - first && !memcmp(folder + first, skip[i], n - first)) {
            return 1;
        }
    }
    return n - first >= 5 && !memcmp(folder + n - 5, ".zarr", 5);
}

static uint64_t inventory_size(const char *at, size_t n) {
    const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"}, *spaces = " \t\r\n\v\f";
    char text[80], *end;
    if (n >= sizeof(text)) {
        return 0;
    }
    memcpy(text, at, n);
    text[n] = 0;
    double value = strtod(text, &end);
    size_t skipped = strspn(end, spaces);
    if (end == text || !skipped) {
        return 0;
    }
    const char *unit = end + skipped;
    size_t length = strcspn(unit, spaces);
    const char *tail = unit + length;
    if (tail[strspn(tail, spaces)]) {
        return 0;
    }
    for (size_t i = 0; i < sizeof(units) / sizeof(*units); i++) {
        if (length == strlen(units[i]) && !memcmp(unit, units[i], length)) {
            double size = ldexp(value, 10 * (int)i);
            return size >= 0. && size < 0x1p64 ? (uint64_t)size : 0;
        }
    }
    return 0;
}

static void inventory_server_page(void *data, size_t i) {
    ServerInventory *inventory = data;
    InventoryPage *result = inventory->pages + i;
    const char *path = json_object_get_string(json_object_array_get_idx(inventory->frontier, i));
    CURL *curl = curl_easy_init();
    char *quoted = curl ? inventory_quote_path(curl, path) : NULL, *url = NULL;
    if (quoted && asprintf(&url, "%s%s", inventory->origin, quoted) < 0) {
        url = NULL;
    }
    char error[128];
    char *page = url ? inventory_page(curl, url, error) : NULL;
    result->files = json_object_new_array();
    result->folders = json_object_new_array();
    result->failed = !page || !result->files || !result->folders;
    const char link[] = "<td class=\"link\"><a href=\"", length[] = "</a></td><td class=\"size\">";
    const char *start = !result->failed ? strstr(page, link) : NULL;
    for (const char *at = start; !result->failed && !result->store && at; at = strstr(at, link)) {
        const char *href = at + sizeof(link) - 1, *end = strchr(href, '"');
        at = end ? end + 1 : href;
        const char *size = end ? strstr(end, length) : NULL;
        if (!size || memchr(href, '\n', size - href) || href[0] == '/' || href[0] == '?' ||
            !strncmp(href, "..", 2) || !strncmp(href, "http", 4)) {
            continue;
        }
        size += sizeof(length) - 1;
        const char *stop = strchr(size, '<');
        if (!stop || strncmp(stop, "</td>", 5) != 0 || end == href || end - href > INT_MAX) {
            continue;
        }
        char *decoded = curl_easy_unescape(curl, href, (int)(end - href), NULL), *joined = NULL;
        int made = decoded && asprintf(&joined, "%s%s", quoted, decoded) >= 0;
        size_t n = decoded ? strlen(decoded) : 0;
        int folder = n && decoded[n - 1] == '/';
        char *full = made ? curl_easy_unescape(curl, joined, 0, NULL) : NULL;
        const char *name = decoded ? strrchr(decoded, '/') : NULL;
        name = name ? name + 1 : decoded;
        if (!full) {
            result->failed = 1;
        } else if (folder) {
            result->failed = !add(result->folders, json_object_new_string(full));
        } else if (!strcmp(name, ".zgroup") || !strcmp(name, ".zarray") ||
            !strcmp(name, "zarr.json")) {
            result->store = 1;
        } else {
            json_object *record = json_object_new_object();
            result->failed = !record || !assign(record, "path", json_object_new_string(full)) ||
                !assign(
                    record, "bytes", json_object_new_uint64(inventory_size(size, stop - size))) ||
                !add(result->files, retain(record));
            drop(record);
        }
        curl_free(full);
        curl_free(decoded);
        free(joined);
    }
    curl_free(quoted);
    if (curl) {
        curl_easy_cleanup(curl);
    }
    free(page);
    free(url);
}

int hx_server_inventory_main(int argc, char **argv) {
    const char *origin = inventory_root(argc, argv, inventory_server);
    if (!origin) {
        puts("Usage: ./run.sh scans.catalog.server_inventory [--root URL]");
        return inventory_help(argc, argv) ? 0 : 2;
    }
    const char *start[] = {"/full-scrolls/", "/fragments/", "/datasets/", "/community-uploads/",
        "/other/", "/ml-models/"};
    char *library = local_root(library_root), *path = NULL, *sidecar = NULL;
    char *base = inventory_base(origin);
    json_object *frontier = json_object_new_array(), *files = json_object_new_array();
    json_object *stores = json_object_new_array(), *failed = json_object_new_array();
    int curl = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    int ok = curl && library && base && frontier && files && stores && failed &&
        asprintf(&path, "%s/catalog/server.jsonl", library) >= 0 &&
        asprintf(&sidecar, "%s/catalog/server.stores.json", library) >= 0 && folders(path);
    for (size_t i = 0; ok && i < sizeof(start) / sizeof(*start); i++) {
        ok = add(frontier, json_object_new_string(start[i]));
    }
    uint64_t bytes = 0;
    for (int depth = 0; ok && depth < 9 && json_object_array_length(frontier); depth++) {
        size_t count = json_object_array_length(frontier);
        InventoryPage *pages = calloc(count, sizeof(*pages));
        json_object *next = json_object_new_array();
        ok = pages && next;
        if (ok) {
            ServerInventory inventory = {frontier, pages, base};
            inventory_parallel(inventory_server_page, &inventory, count, 8);
        }
        for (size_t i = 0; pages && i < count; i++) {
            InventoryPage *page = pages + i;
            json_object *entry = json_object_array_get_idx(frontier, i);
            if (ok && page->failed) {
                ok = add(failed, retain(entry));
            } else if (ok && page->store) {
                ok = add(stores, retain(entry));
            } else if (ok) {
                for (size_t j = 0; ok && j < json_object_array_length(page->files); j++) {
                    json_object *row = json_object_array_get_idx(page->files, j);
                    ok = add(files, retain(row));
                    bytes += json_object_get_uint64(value_of(row, "bytes"));
                }
                for (size_t j = 0; ok && j < json_object_array_length(page->folders); j++) {
                    json_object *folder = json_object_array_get_idx(page->folders, j);
                    ok = add(inventory_store(json_object_get_string(folder)) ? stores : next,
                        retain(folder));
                }
            }
            drop(page->files);
            drop(page->folders);
        }
        free(pages);
        drop(frontier);
        frontier = next;
        json_object *state = json_object_new_object();
        ok = ok && state && assign(state, "stores", retain(stores)) &&
            assign(state, "failed", retain(failed)) &&
            assign(state, "unwalked", retain(frontier)) && inventory_write(path, files, 1) &&
            inventory_write(sidecar, state, 0);
        drop(state);
        if (ok) {
            printf("depth %d pages %zu files %zu stores %zu\n", depth, count,
                json_object_array_length(files), json_object_array_length(stores));
            fflush(stdout);
        }
    }
    if (ok && inventory_totals(files, 1)) {
        printf("%zu files %.2f TB %zu stores %zu failed %zu unwalked\n",
            json_object_array_length(files), bytes / 1e12, json_object_array_length(stores),
            json_object_array_length(failed), json_object_array_length(frontier));
    } else {
        ok = 0;
        fprintf(stderr, "HerculeX: could not write %s\n", path ? path : "server.jsonl");
    }
    if (curl) {
        curl_global_cleanup();
    }
    drop(files);
    drop(stores);
    drop(failed);
    drop(frontier);
    free(base);
    free(sidecar);
    free(path);
    free(library);
    return !ok;
}
