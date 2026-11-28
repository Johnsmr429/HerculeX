#include "../base.h"
#include <arpa/inet.h>
#include <blosc.h>
#include <signal.h>
#include <sys/socket.h>

static char temporary[] = "/tmp/hx-spot-surfaces.XXXXXX";
static pid_t site;

static void clean(void) {
    if (site > 0) {
        kill(site, SIGTERM);
        waitpid(site, NULL, 0);
    }
    remove_tree(temporary);
}

static void execute(char *const args[]) {
    pid_t pid = fork();
    need(pid >= 0, "could not launch spot fixture");
    if (!pid) {
        execv(args[0], args);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid && WIFEXITED(code) && !WEXITSTATUS(code),
        "%s did not complete", args[1]);
}

static int serve(const char *binary, Arena *a) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t bytes = sizeof(address);
    need(fd >= 0 && !bind(fd, (struct sockaddr *)&address, bytes) &&
            !getsockname(fd, (struct sockaddr *)&address, &bytes),
        "could not reserve fixture port");
    close(fd);
    int port = ntohs(address.sin_port);
    char *word = fmt(a, "%d", port);
    site = fork();
    need(site >= 0, "could not launch fixture server");
    if (!site) {
        execl(binary, binary, "serve", word, (char *)NULL);
        _exit(127);
    }
    for (int i = 0; i < 100; i++) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        need(fd >= 0, "could not connect to fixture server");
        int ready = !connect(fd, (struct sockaddr *)&address, bytes);
        close(fd);
        if (ready) {
            return port;
        }
        usleep(20000);
    }
    die("fixture server did not start");
    return 0;
}

static void write_text(const char *path, const char *text) { write_file(path, text, strlen(text)); }

int main(int argc, char **argv) {
    need(argc == 2 && mkdtemp(temporary) != NULL, "expected a native binary and a fixture folder");
    atexit(clean);
    Arena a = arena(64 << 20);
    char *library = fmt(&a, "%s/library", temporary), *outputs = fmt(&a, "%s/outputs", temporary);
    char *work = fmt(&a, "%s/studies/shared/spot-candidates/spots/fixture8", library);
    setenv("HERCULEX_LIBRARY", library, 1);
    setenv("HERCULEX_OUTPUTS", outputs, 1);
    unsetenv("HERCULEX_SPOTS_SINCE");
    int port = serve(argv[1], &a);
    const char *source = fmt(&a, "%s/unwrapping/level/PHercTest/scan", outputs);
    const char *piece = fmt(&a, "%s/pieces/rows-001/piece-0001", source);
    write_text(fmt(&a, "%s/sheet/axis.json", source), "[[0,1,1],[79,1,1]]");
    write_text(fmt(&a, "%s/meta.json", piece),
        "{\"width_mm\":2.8,\"area_mm2\":7.84,\"winding\":1,\"stride\":2,\"voxel_um\":100}");
    f32 plane[15 * 15];
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 15; y++) {
            for (int x = 0; x < 15; x++) {
                plane[y * 15 + x] = c == 0 ? 18 + 2 * x : c == 1 ? 18 + 2 * y : 40;
            }
        }
        tiff_write(fmt(&a, "%s/%c.tif", piece, "xyz"[c]), plane, 15, 15);
    }
    write_text(fmt(&a, "%s/marks/sheets.json", library),
        "[{\"key\":\"parts/fixture\",\"scroll\":\"PHercTest\",\"scan\":\"scan\","
        "\"route\":\"level\",\"page\":\"fixture\","
        "\"pieces\":\"unwrapping/level/PHercTest/scan/pieces\"}]");
    write_text(fmt(&a, "%s/marks/parts/fixture/layouts/saved.json", library),
        "{\"pieces\":[{\"unit\":\"rows-001\",\"piece\":1,\"group\":1,\"level\":0,\"shape\":[15,15],"
        "\"stride\":2,\"um\":0.1,\"y_mm\":0,\"x_mm\":0,\"width_mm\":2.8,\"height_mm\":2.8,"
        "\"area\":7.84,\"flip_rows\":false,\"flip_cols\":false}]}");
    const char *log_path = fmt(&a, "%s/marks/parts/fixture/marks.jsonl", library);
    const char *log =
        "{\"id\":\"fixture8\",\"kind\":\"spot\",\"layout\":\"saved\","
        "\"time\":\"2026-11-17T01:00:00Z\",\"points\":[[0.2,0.2],[2.7,0.2],[2.7,2.7],[0.2,2.7]]}\n";
    write_text(log_path, log);
    const char *strip = fmt(&a, "%s/other-root/website/sheets/strip", temporary);
    write_text(fmt(&a, "%s/index.html", strip), "strip");
    mkdirs(fmt(&a, "%s/website/sheets", outputs));
    need(!symlink(strip, fmt(&a, "%s/website/sheets/strip", outputs)),
        "could not link another unroll's page");
    write_text(fmt(&a, "%s/marks/level_route/PHercTest/scan/pages/strip/marks.jsonl", library),
        "{\"id\":\"strip1\",\"kind\":\"ink\",\"layout\":\"saved\",\"time\":\"2026-11-27T01:00:"
        "00Z\","
        "\"points\":[[0.2,0.2]]}\n{\"time\":\"2026-11-27T01:00:01Z\",\"remove\":\"strip1\"}\n");
    write_text(fmt(&a, "%s/catalog/metadata.json", library),
        fmt(&a,
            "{\"samples\":{\"PHercTest\":{\"volumes\":{\"scan\":{\"data\":[{\"type\":\"ome-zarr\","
            "\"origins\":[{\"path\":\"ct\",\"access_roots\":[{\"url\":\"http://127.0.0.1:%d/"
            "sheets\"}]}]}]}}}}}",
            port));
    write_text(fmt(&a, "%s/catalog/products.jsonl", library),
        fmt(&a,
            "{\"sample\":\"PHercTest\",\"owner\":\"scan\",\"kind\":\"surface-prediction-zarr\","
            "\"root\":\"http://127.0.0.1:%d/sheets\",\"path\":\"prediction-L0-m7/\"}\n",
            port));
    const char *ct = fmt(&a, "%s/website/sheets/ct/0", outputs);
    const char *prediction = fmt(&a, "%s/website/sheets/prediction-L0-m7/0", outputs);
    const char *shape = "{\"shape\":[80,64,64],\"chunks\":[80,64,64]}";
    write_text(fmt(&a, "%s/.zarray", ct), shape);
    write_text(fmt(&a, "%s/.zarray", prediction), shape);
    size_t n = 80 * 64 * 64;
    u8 *raw = push(&a, n), *mask = push(&a, n), *encoded = push(&a, n + BLOSC_MAX_OVERHEAD);
    for (int z = 0; z < 80; z++) {
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                size_t i = (z * 64 + y) * 64 + x;
                raw[i] = (3 * z + y + x) % 250 + 1;
                mask[i] = abs(z - 43) <= 1;
            }
        }
    }
    int size = blosc_compress_ctx(
        3, BLOSC_SHUFFLE, 1, n, mask, encoded, n + BLOSC_MAX_OVERHEAD, "zstd", 0, 1);
    need(size > 0, "could not compress the m7 fixture");
    write_file(fmt(&a, "%s/0/0/0", ct), raw, n);
    write_file(fmt(&a, "%s/0/0/0", prediction), encoded, size);
    const char *cache = fmt(&a, "%s/ct", work);
    write_text(fmt(&a, "%s/meta.json", cache), "{\"shape\":[80,64,64],\"brick\":[80,64,64]}");
    write_file(fmt(&a, "%s/bricks.bin", cache), pushz(&a, n), n);
    i64 old = 0;
    npy_write(fmt(&a, "%s/offsets.npy", cache),
        (Array){.descr = "<i8", .ndim = 3, .shape = {1, 1, 1}, .data = &old, .size = 8});
    execute((char *[]){argv[1], "spots.catalogue", NULL});
    execute((char *[]){argv[1], "spots.pilot", "fixture8", "build", "grow", "render", NULL});
    const char *records = fmt(&a, "%s/pieces.json", work);
    json_object *doc = json_read(records);
    json_object *row = json_object_array_get_idx(jget(doc, "pieces"), 0);
    json_object *snap = jget(row, "snap"), *grow = jget(row, "grow");
    need(ji64(snap, "nodes") == 225 && jf64(snap, "median_move_vox") == 3 &&
            ji64(grow, "nodes_before") == 225 && ji64(grow, "nodes") == 992,
        "spot snap or growth differs from reference");
    json_object_put(doc);
    const char *surfaces[] = {"original", "m7", "grown"};
    const char *digest[] = {"b3dee8fd25d01534972bec5e515404c19ab35cc7d890ee04ee5f4b683405c6b4",
        "d156c8f474cd07923cc029b529d058a3a7f0b6a33089f02af7289dacfda91feb",
        "9fd29434bcad648b5ccd5d09c099268a499fd1f0b1e05680e31599a9fd734d6f"};
    for (int i = 0; i < 3; i++) {
        const char *path = fmt(&a, "%s/stacks/%s/rows-001-piece-0001.npy", work, surfaces[i]);
        Array stack = npy_read(&a, path);
        char hash[65];
        sha256_hex(stack.data, stack.size, hash);
        need(stack.shape[0] == 67 && !strcmp(hash, digest[i]),
            "%s stack differs from reference: %s", surfaces[i], hash);
    }
    const char *pack = fmt(&a, "%s/bricks.bin", cache);
    Bytes kept = read_file(&a, pack);
    need(kept.size == 2 * n && !memcmp(kept.data + n, raw, n),
        "legacy CT did not refresh as a complete brick");
    Array whole = npy_read(&a, fmt(&a, "%s/complete.npy", cache));
    need(whole.size == 1 && *(u8 *)whole.data, "CT completeness was not kept");
    i64 stamp = mtime_ns(pack);
    execute((char *[]){argv[1], "spots.pilot", "fixture8", "render", NULL});
    need(stamp == mtime_ns(pack), "an unchanged rerender fetched CT again");
    Bytes marks = read_file(&a, log_path);
    need(!strcmp((char *)marks.data, log), "spot catalogue changed a mark log");
    munmap(a.base, a.size);
    puts("spot surfaces e2e passed");
    return 0;
}
