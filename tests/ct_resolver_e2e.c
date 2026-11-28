#include "../base.h"
#include <sqlite3.h>

static char temporary[] = "/tmp/hx-ct-resolver.XXXXXX";

static void clean(void) { remove_tree(temporary); }

static void resolve(const char *binary, const char *sample, const char *expected) {
    pid_t pid;
    char *args[] = {(char *)binary, "scans.chunks.resolve", (char *)sample, "scan", NULL};
    FILE *out = spawn_read(args, 0, &pid);
    need(out != NULL, "cannot launch CT resolver");
    char line[4096] = "", kind[16], url[4096];
    int level;
    int got = fgets(line, sizeof(line), out) != NULL;
    int code = spawn_close(out, pid);
    if (expected) {
        need(code == 0 && got && sscanf(line, "%15s %d %4095s", kind, &level, url) == 3 &&
                !strcmp(kind, "ct") && level == 0 && !strcmp(url, expected),
            "%s resolved to '%s' (exit %d), expected %s", sample, line, code, expected);
    } else {
        need(code == 1 && !got, "%s has no CT product in the catalogue", sample);
    }
}

static void catalogue(Arena *a, const char *file, const char *path) {
    const char *text = fmt(a,
        "{\"samples\":{\"PHerc0846B\":{\"volumes\":{\"scan\":{\"data\":["
        "{\"type\":\"surface-prediction-zarr\",\"origins\":[{\"path\":\"surface.zarr\"}]},"
        "{\"type\":\"ome-zarr\",\"origins\":[{\"path\":\"/%s/\",\"access_roots\":["
        "{\"url\":\"s3://vesuvius-challenge-open-data\"}]}]}]}}},"
        "\"PHerc0306B\":{\"volumes\":{\"scan\":{\"data\":[{\"type\":\"ome-zarr\",\"origins\":[{"
        "\"path\":\"PHerc0306B/volumes/original.zarr/\",\"access_roots\":["
        "{\"url\":\"https://mirror.invalid/\"}]}]}]}}}}}",
        path);
    write_file(file, text, strlen(text));
}

int main(int argc, char **argv) {
    need(argc == 2 && mkdtemp(temporary), "expected a native binary and a fixture folder");
    atexit(clean);
    Arena a = arena(1 << 20);
    const char *library = fmt(&a, "%s/library", temporary);
    const char *chunks = fmt(&a, "%s/chunks", temporary);
    const char *file = fmt(&a, "%s/catalog/metadata.json", library);
    setenv("HERCULEX_LIBRARY", library, 1);
    setenv("HERCULEX_CHUNKS", chunks, 1);
    setenv("HERCULEX_CT_BUCKET", "http://127.0.0.1:1/", 1);
    catalogue(&a, file, "PHerc0846B/volumes/scan-masked.zarr");
    const char *good = "http://127.0.0.1:1/PHerc0846B/volumes/scan-masked.zarr";
    resolve(argv[1], "PHerc0846B", good);
    resolve(argv[1], "PHerc0306B", "https://mirror.invalid/PHerc0306B/volumes/original.zarr");
    sqlite3 *db = NULL;
    need(sqlite3_open(fmt(&a, "%s/arrays.sqlite", chunks), &db) == SQLITE_OK &&
            sqlite3_exec(db,
                "UPDATE resolved SET url='scan-masked-fast.vcd3d.zarr',level=7,source='listing' "
                "WHERE sample='PHerc0846B'",
                NULL, NULL, NULL) == SQLITE_OK &&
            sqlite3_close(db) == SQLITE_OK,
        "cannot seed an obsolete cached resolution");
    resolve(argv[1], "PHerc0846B", good);
    resolve(argv[1], "PHerc0846B", good);
    catalogue(&a, file, "PHerc0846B/volumes/revised.zarr");
    resolve(argv[1], "PHerc0846B", "http://127.0.0.1:1/PHerc0846B/volumes/revised.zarr");
    resolve(argv[1], "PHercUnknown", NULL);
    munmap(a.base, a.size);
    puts("CT resolver e2e passed");
    return 0;
}
