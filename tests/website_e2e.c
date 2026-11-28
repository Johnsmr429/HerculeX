#include "../base.h"
#include <arpa/inet.h>
#include <curl/curl.h>
#include <signal.h>
#include <sys/socket.h>

#include "../storage/tile_index.h"
#include "../viewer/tile_decode.c"
#include "../viewer/sheet_page.h"

static pid_t viewer;
static char temp[] = "/tmp/hx-website.XXXXXX";

static void clean(void) {
    if (viewer > 0) {
        kill(viewer, SIGTERM);
        waitpid(viewer, NULL, 0);
    }
    remove_tree(temp);
}

typedef struct {
    u8 *data;
    size_t size, limit;
} Answer;

static size_t consume(char *input, size_t size, size_t count, void *opaque) {
    Answer *answer = opaque;
    size_t bytes = size * count;
    if (bytes > answer->limit - answer->size) {
        return 0;
    }
    memcpy(answer->data + answer->size, input, bytes);
    answer->size += bytes;
    answer->data[answer->size] = 0;
    return bytes;
}

static long request(CURL *curl, Answer *answer, const char *url, const char *body) {
    struct curl_slist *headers = NULL;
    answer->size = 0;
    curl_easy_setopt(curl, CURLOPT_URL, url);
    if (body) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    } else {
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    }
    long status = 0;
    if (curl_easy_perform(curl) == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, NULL);
    curl_slist_free_all(headers);
    return status;
}

static int port_number(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t bytes = sizeof(address);
    need(fd >= 0 && !bind(fd, (struct sockaddr *)&address, bytes) &&
            !getsockname(fd, (struct sockaddr *)&address, &bytes),
        "could not reserve test port");
    close(fd);
    return ntohs(address.sin_port);
}

static void compose(const char *binary, const char *page, const char *surface) {
    pid_t pid = fork();
    need(pid >= 0, "could not fork compositor");
    if (!pid) {
        execl(binary, binary, "viewer.sheet_tiles", "PHerc0000", "fixture", "0.01", page, "sheet",
            surface ? "--review-surface" : (char *)NULL, surface, (char *)NULL);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid && WIFEXITED(code) && !WEXITSTATUS(code),
        "native tile composition failed");
}

static void registered_card_fixture(Arena *a, const char *binary, const char *output) {
    const char *page = "sheet-PHerc0000-fixture-level";
    const char *surface = "level, umbilicus (organisers), curl +1";
    const char *fit = "level 1.1";
    compose(binary, page, "level");
    pid_t pid = fork();
    need(pid >= 0, "could not fork card publisher");
    if (!pid) {
        execl(binary, binary, "card", page, "PHerc0000", "--surface", surface, "--fit", fit,
            (char *)NULL);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid && WIFEXITED(code) && !WEXITSTATUS(code),
        "native card publication failed");
    Bytes html = read_file(a, fmt(a, "%s/website/sheets/index.html", output));
    const char *href = fmt(a, "%s/index.html", page);
    const char *listed = strstr((const char *)html.data, href);
    need(listed && !strstr(listed + strlen(href), href) && strstr(listed, surface) &&
            strstr(listed, fit) && strstr((const char *)html.data, "\"um\":10.0") &&
            strstr((const char *)html.data, "String(um)") &&
            strstr((const char *)html.data, "s.um === um") &&
            strstr((const char *)html.data, "FPL") && strstr((const char *)html.data, "GP"),
        "a registered catalogued viewer must preserve its surface and exact voxel size");
}

static Bytes refresh_viewers(Arena *a, const char *binary, const char *log) {
    pid_t pid = fork();
    need(pid >= 0, "could not fork viewer refresh");
    if (!pid) {
        int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || dup2(fd, STDOUT_FILENO) < 0) {
            _exit(127);
        }
        close(fd);
        execl(binary, binary, "viewer.refresh_viewers", (char *)NULL);
        _exit(127);
    }
    int code;
    need(waitpid(pid, &code, 0) == pid && WIFEXITED(code) && !WEXITSTATUS(code),
        "native viewer refresh failed");
    return read_file(a, log);
}

static void refresh_fixture(Arena *a, const char *binary) {
    const char *output = fmt(a, "%s/refresh-output", temp);
    const char *sheets = fmt(a, "%s/website/sheets", output);
    const char *log = fmt(a, "%s/refresh.log", temp);
    const char *names[] = {
        "level-PHerc0211-20250821151803", "part-sheet-PHerc0211-113keV-level-2026-11-13"};
    const char *scans[] = {
        "level/PHerc0211/20250821151803", "part/sheet-PHerc0211-113keV-level-2026-11-13"};
    const char *layers = "[{\"path\":\"taught\",\"stamp\":123,\"mm\":0.009,\"levels\":8}]";
    const char *panel = "const M={\"title\":\"Guide\"}, images = [];";
    const char *page[2], *meta[2], *label[2], *tile[2];
    const char *guide = fmt(a, "%s/guide/index.html", sheets);
    setenv("HERCULEX_OUTPUTS", output, 1);
    write_file(guide, panel, strlen(panel));
    for (int i = 0; i < 2; i++) {
        meta[i] = fmt(a,
            "const M={\"scan\":\"%s\",\"title\":\"6 November\",\"layers\":%s,\"tile\":512,"
            "\"area\":{\"x\":0,\"y\":0,\"w\":10,\"h\":20},\"shelves\":[],\"places\":[],"
            "\"boxes\":[],\"back\":\"../index.html\"}",
            scans[i], layers);
        page[i] = fmt(a, "%s/%s/index.html", sheets, names[i]);
        label[i] = fmt(a, "%s/%s/marks.jsonl", sheets, names[i]);
        tile[i] = fmt(a, "%s/%s/tile.jpg", sheets, names[i]);
        const char *before =
            fmt(a, "%s,TOP=32; floor=kept.floor??128,scan=kept.scan??0.25;", meta[i]);
        write_file(page[i], before, strlen(before));
        write_file(label[i], "saved labels\n", 13);
        write_file(tile[i], "original pixels", 15);
    }
    Bytes changed = refresh_viewers(a, binary, log);
    for (int i = 0; i < 2; i++) {
        const char *html = (const char *)read_file(a, page[i]).data;
        need(strstr((const char *)changed.data, page[i]) != NULL &&
                strstr(html, strchr(meta[i], '{')) && strstr(html, "id=\"labelopen\"") &&
                strstr(html, "class=\"menu right\"") && strstr(html, "kept.floor ?? 128") &&
                strstr(html, "kept.scan ?? 0.25"),
            "viewer refresh lost archive metadata or controls for %s", names[i]);
        need(!strcmp((const char *)read_file(a, label[i]).data, "saved labels\n") &&
                !strcmp((const char *)read_file(a, tile[i]).data, "original pixels"),
            "viewer refresh changed archived marks or pixels");
    }
    need(!strcmp((const char *)read_file(a, guide).data, panel), "viewer refresh changed a panel");
    need(!refresh_viewers(a, binary, log).size, "viewer refreshed an already current page");
}

static void panel_fixture(Arena *a, const char *binary) {
    const char *output = fmt(a, "%s/panel-output", temp);
    const char *folder = fmt(a, "%s/website/sheets/panel", output);
    setenv("HERCULEX_OUTPUTS", output, 1);
    const char *title = "<title>Old &amp; title</title><h1>x</h1>";
    const char *index = fmt(a, "%s/index.html", folder);
    write_file(index, title, strlen(title));
    const char *pictures[] = {"b-place/2-x.png", "a-place/10-model-depth-mean.png",
        "a-place/1-site.png", "a-place/0-scan.png"};
    for (size_t i = 0; i < sizeof(pictures) / sizeof(*pictures); i++) {
        write_file(fmt(a, "%s/%s", folder, pictures[i]), "", 0);
    }
    pid_t pid = fork();
    need(pid >= 0, "could not fork panel publisher");
    if (!pid) {
        execl(binary, binary, "viewer.panel_pages", "panel", (char *)NULL);
        _exit(127);
    }
    int status;
    need(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "native panel publication failed");
    const char *page = (const char *)read_file(a, index).data;
    const char *metadata = strstr(page, "const M = ");
    need(metadata && strstr(page, "<title>Old &amp; title</title>") && !strstr(page, "<h1") &&
            !strstr(page, "<p"),
        "panel page lost its title or template");
    json_object *meta = json_tokener_parse(metadata + strlen("const M = "));
    json_object *places = jget(meta, "places");
    json_object *first = json_object_array_get_idx(places, 0);
    json_object *second = json_object_array_get_idx(places, 1);
    json_object *ordered = jget(first, "pictures");
    need(!strcmp(jstr(meta, "title"), "Old & title") && !strcmp(jstr(first, "name"), "a-place") &&
            !strcmp(jstr(second, "name"), "b-place") && json_object_array_length(ordered) == 3,
        "panel page lost its metadata or place order");
    const char *names[] = {"scan", "site", "model depth mean"};
    for (int i = 0; i < 3; i++) {
        json_object *picture = json_object_array_get_idx(ordered, i);
        need(!strcmp(jstr(picture, "title"), names[i]) &&
                !strcmp(jstr(picture, "src"), pictures[3 - i]),
            "panel page lost its picture order");
    }
    json_object_put(meta);
}

static void proposals_golden(Arena *a, const char *binary) {
    const char *output = fmt(a, "%s/proposals-output", temp);
    const char *base = fmt(a, "%s/unwrapping/level/fixture/volume", output);
    const char *folder = fmt(a, "%s/faces/level-0/unit", base);
    setenv("HERCULEX_OUTPUTS", output, 1);
    const int side = 200;
    u8 *face = push(a, side * side), *ink = pushz(a, side * side);
    memset(face, 90, side * side);
    for (int y = 120; y < 150; y++) {
        for (int x = 120; x < 150; x++) {
            ink[y * side + x] = 250;
        }
    }
    Bytes raw = png_encode(a, face, side, side, 6);
    Bytes map = png_encode(a, ink, side, side, 6);
    const double near[] = {.15, .02, .15, .04, .02, .15};
    const double far[] = {.03, .14, .12, 0, .14, .03};
    json_object *layout = json_object_new_object(), *pieces = json_object_new_array();
    for (int i = 0; i < 6; i++) {
        const char *piece = fmt(a,
            "{\"unit\":\"unit\",\"piece\":%d,\"group\":%d,\"area\":900,\"x_mm\":%d,"
            "\"y_mm\":0,\"width_mm\":30,\"height_mm\":30,\"um\":0.15,\"stride\":1}",
            i, i, 100 * i);
        json_object_array_add(pieces, json_tokener_parse(piece));
        const char *stem = fmt(a, "%s/piece-%04d", folder, i);
        write_file(fmt(a, "%s.png", stem), raw.data, raw.size);
        write_file(fmt(a, "%s-taught.png", stem), map.data, map.size);
        const char *written = i == 4 ? "far" : "near";
        const char *reading =
            fmt(a, "{\"written\":\"%s\",\"near\":%.2f,\"far\":%.2f,\"scrambled\":%.2f}", written,
                near[i], far[i], i == 5 ? .13 : 0);
        write_file(fmt(a, "%s-taught-side.json", stem), reading, strlen(reading));
    }
    json_object_object_add(layout, "pieces", pieces);
    json_write(fmt(a, "%s/sheet/layout.json", base), layout);
    json_object_put(layout);
    pid_t pid = fork();
    need(pid >= 0, "could not fork window proposals");
    if (!pid) {
        execl(binary, binary, "viewer.window_proposals", "fixture", "volume", (char *)NULL);
        _exit(127);
    }
    int status;
    need(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
        "native window proposals failed");
    json_object *report = json_read(fmt(a, "%s/sheet/proposals.json", base));
    json_object *boxes = jget(report, "boxes");
    need(ji64(report, "windows") == 18 && json_object_array_length(boxes) == 2,
        "proposals accepted a weak, ambiguous, or scrambled piece");
    for (int i = 0; i < 2; i++) {
        json_object *box = json_object_array_get_idx(boxes, i);
        double x = jf64(box, "x");
        need((x < 20 || (x >= 399 && x < 420)) && jf64(box, "share") > .05,
            "proposals lost the strongest region of a written face");
    }
    json_object_put(report);
}

static void suggested_golden(void) {
    double square[4];
    hx_sheet_suggested_square(square, 100, 50, .01, 16, 200, 100, 1940, 1740, 0, 0);
    need(square[0] == 101.68 && square[1] == 50.68 && square[2] == 14.4 && square[3] == 14.4,
        "suggested square did not follow the layout's margin and scale");
    hx_sheet_suggested_square(square, 100, 50, .01, 16, 200, 100, 1940, 1740, 1, 1);
    need(square[0] == 102.68 && square[1] == 51.68,
        "suggested square did not follow the layout's orientation");
}

static void pyramid_golden(Arena *a) {
    size_t saved = a->used;
    int side = 512, pitch = 1027;
    size_t pixels = (size_t)side * side;
    u8 *source = push(a, 1024 * pitch), *result = push(a, 16 * pixels);
    for (int y = 0; y < 1024; y++) {
        for (int x = 0; x < 1024; x++) {
            source[(1023 - y) * pitch + x] = 19 * x + 23 * y + ((x ^ y) >> 3);
        }
    }
    for (int mask = 0; mask < 16; mask++) {
        const u8 *children[4] = {0};
        int64_t stride[] = {-pitch, -pitch, -pitch, -pitch};
        for (int slot = 0; slot < 4; slot++) {
            if (mask & (1 << slot)) {
                children[slot] = source + (1023 - slot / 2 * side) * pitch + slot % 2 * side;
            }
        }
        hx_half_tile(children, stride, result + mask * pixels);
    }
    char hash[65];
    sha256_hex(result, pixels * 16, hash);
    need(!strcmp(hash, "13058e002f8a77d35079c73f5109c9bd35268ca879b228ae87227de299176821"),
        "Lanczos tile levels differ from the Pillow golden");
    a->used = saved;
}

int main(int argc, char **argv) {
    need(argc == 2, "usage: website_e2e HERCULEX_BINARY");
    need(mkdtemp(temp) != NULL, "cannot make website fixture");
    atexit(clean);
    Arena a = arena(64 << 20);
    char *library = fmt(&a, "%s/library", temp), *output = fmt(&a, "%s/outputs", temp);
    setenv("HERCULEX_LIBRARY", library, 1);
    setenv("HERCULEX_OUTPUTS", output, 1);
    unsetenv("HERCULEX_FACES");
    unsetenv("HERCULEX_INK_MODEL");
    char *base = fmt(&a, "%s/unwrapping/level/PHerc0000/fixture", output);
    const char *layout =
        "{\"pieces\":[{\"unit\":\"rows-0000\",\"piece\":0,\"group\":0,"
        "\"x_mm\":1,\"y_mm\":1,\"width_mm\":19,\"height_mm\":7,\"area\":420,"
        "\"stride\":1,\"um\":0.01,\"shape\":[700,1900]}],\"record\":{\"moved\":0,\"joints\":0}}";
    write_file(fmt(&a, "%s/sheet/layout.json", base), layout, strlen(layout));
    int w = 1904, h = 704;
    u8 *pixels = push(&a, w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            pixels[y * w + x] = 32 + (x / 16 + 3 * (y / 16)) % 192;
        }
    }
    Bytes png = png_encode(&a, pixels, w, h, 3);
    write_file(fmt(&a, "%s/faces/level-0/rows-0000/piece-0000.png", base), png.data, png.size);
    const char *page = "fixture";
    compose(argv[1], page, NULL);
    registered_card_fixture(&a, argv[1], output);
    int directory = open(fmt(&a, "%s/website/sheets/%s", output, page), O_RDONLY | O_DIRECTORY);
    TileIndex index;
    need(directory >= 0 && tile_index_open(&index, directory) == 1, "could not open fixture pack");
    close(directory);
    TileSpan span;
    json_object *sheet = json_object_object_get(index.layers, "sheet");
    need(tile_index_level(&index, sheet, 0, &span) && span.w > 2 && span.h > 1,
        "expected a non-square tile grid");
    uint64_t offset;
    uint32_t size;
    need(tile_index_blob(&index, span.first + span.w + 2, &offset, &size) && size,
        "fixture tile missing");
    Bytes packed = {.data = push(&a, size), .size = size};
    need(transfer_at(index.pack, packed.data, size, offset, 0), "cannot read fixture tile");
    int cols, rows;
    u8 *expected = png_decode(&a, packed, &cols, &rows);
    need(cols == 512 && rows == 512, "wrong fixture tiles");
    char *port = fmt(&a, "%d", port_number()), *url = fmt(&a, "http://127.0.0.1:%s", port);
    viewer = fork();
    need(viewer >= 0, "could not fork viewer");
    if (!viewer) {
        execl(argv[1], argv[1], "serve", port, (char *)NULL);
        _exit(127);
    }
    CURL *curl = curl_easy_init();
    need(curl != NULL, "cannot initialize viewer request");
    Answer answer = {.limit = 1 << 20};
    answer.data = push(&a, answer.limit + 1);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, consume);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &answer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 2000L);
    char *tilemap = fmt(&a, "%s/tilemap?dir=/sheets/%s", url, page);
    long code = 0;
    for (int i = 0; !code && i < 100; i++) {
        code = request(curl, &answer, tilemap, NULL);
        if (!code) {
            usleep(20000);
        }
    }
    need(code == 200 && answer.size > 4, "viewer tile map failed");
    size_t length =
        answer.data[0] | answer.data[1] << 8 | answer.data[2] << 16 | (size_t)answer.data[3] << 24;
    need(length + 4 < answer.size, "tile map header size does not fit");
    char *header = fmt(&a, "%.*s", (int)length, answer.data + 4);
    json_object *meta = json_tokener_parse(header);
    need(meta && (size_t)ji64(meta, "cells") == index.cells, "tile map differs from pack");
    const char *asked = fmt(&a,
        "{\"dir\":\"/sheets/%s\",\"stamp\":\"%s\",\"tiles\":"
        "[[\"sheet\",0,%d,%d],[\"sheet\",0,999,999]]}",
        page, jstr(meta, "stamp"), span.x + 2, span.y + 1);
    need(request(curl, &answer, fmt(&a, "%s/tiles", url), asked) == 200 &&
            answer.size == 2 * 512 * 512 && !memcmp(answer.data, expected, 512 * 512),
        "viewer decoded the wrong tile");
    for (size_t i = 512 * 512; i < answer.size; i++) {
        need(!answer.data[i], "empty tile has pixels");
    }
    need(request(curl, &answer, fmt(&a, "%s/tiles", url),
             "{\"dir\":\"/sheets/fixture\",\"stamp\":\"0\",\"tiles\":[[\"sheet\",0,0,0]]}") == 409,
        "stale tile stamp was accepted");
    char *range =
        fmt(&a, "%llu-%llu", (unsigned long long)offset, (unsigned long long)(offset + size - 1));
    curl_easy_setopt(curl, CURLOPT_RANGE, range);
    need(request(curl, &answer, fmt(&a, "%s/sheets/%s/%s?v=1", url, page, index.name), NULL) ==
                206 &&
            answer.size == packed.size && !memcmp(answer.data, packed.data, packed.size),
        "the tile pack range differs");
    curl_easy_setopt(curl, CURLOPT_RANGE, NULL);
    char *status_url = fmt(&a, "%s/status", url);
    need(request(curl, &answer, status_url, NULL) == 200, "could not read viewer status");
    json_object *status = json_tokener_parse((char *)answer.data);
    need(ji64(status, "read_scrolls") == 0 && jf64(status, "laid_cm2") == 4.2,
        "viewer status before reading differs");
    json_object_put(status);
    write_file(
        fmt(&a, "%s/faces/level-0/rows-0000/piece-0000-taught.png", base), png.data, png.size);
    need(request(curl, &answer, status_url, NULL) == 200, "could not refresh viewer status");
    status = json_tokener_parse((char *)answer.data);
    need(ji64(status, "read_scrolls") == 1 && jf64(status, "ink_cm2") == 4.2,
        "viewer status after reading differs");
    json_object_put(status);
    const char *circle = "circles-PHerc0211/review/depths/1b68c2ba-new/3-depth-+0.png";
    const char *archived = "/submission/PHerc0211/work/depths/1b68c2ba-new";
    write_file(fmt(&a, "%s/website/sheets/%s", output, circle), "depth image", 11);
    need(request(curl, &answer, fmt(&a, "%s%s/3-depth-+0.png", url, archived), NULL) == 200 &&
            answer.size == 11 && !memcmp(answer.data, "depth image", 11),
        "archived depth image did not load");
    need(request(curl, &answer, fmt(&a, "%s%s/spot.json", url, archived), NULL) == 404 &&
            request(curl, &answer, fmt(&a, "%s/blind_pairs/key.json", url), NULL) == 404,
        "viewer exposed a missing archive page");
    const char *marks = fmt(&a, "%s/marks", url);
    const char *key = "level/PHerc0000/fixture";
    need(request(curl, &answer, marks,
             fmt(&a,
                 "{\"scan\":\"%s\",\"kind\":\"spot\","
                 "\"layer\":\"taught\",\"points\":[[1,2],[3,4]],\"width_mm\":0.4}",
                 key)) == 200,
        "viewer could not save fixture mark");
    json_object *mark = json_tokener_parse((char *)answer.data);
    Bytes placed = read_file(&a, fmt(&a, "%s/sheet/placed-taught.json", base));
    char hash[65];
    sha256_hex(placed.data, placed.size, hash);
    hash[12] = 0;
    const char *log = fmt(&a, "%s/marks/level_route/PHerc0000/fixture", library);
    Bytes kept = read_file(&a, fmt(&a, "%s/layouts/%s.json", log, hash));
    need(!strcmp(hash, jstr(mark, "layout")) && placed.size == kept.size &&
            !memcmp(placed.data, kept.data, kept.size),
        "wrong mark placement snapshot");
    need(request(curl, &answer, marks,
             fmt(&a, "{\"scan\":\"%s\",\"remove\":\"%s\"}", key, jstr(mark, "id"))) == 200,
        "viewer could not remove fixture mark");
    need(request(curl, &answer, fmt(&a, "%s?scan=%s", marks, key), NULL) == 200 &&
            !strcmp((char *)answer.data, "[]"),
        "removed mark remains visible");
    Bytes log_bytes = read_file(&a, fmt(&a, "%s/marks.jsonl", log));
    const char *next = strchr((char *)log_bytes.data, '\n');
    need(next && strstr(next, "\"remove\"") && strchr(next + 1, '\n'),
        "mark removal was not appended");
    const char *other = fmt(&a, "%s/other-root", temp);
    const char *other_layout = "{\"pieces\":[{\"unit\":\"rows-0000\",\"piece\":0,\"group\":0,"
                               "\"x_mm\":4,\"y_mm\":2,\"width_mm\":19,\"height_mm\":7,\"area\":420,"
                               "\"stride\":1,\"um\":0.01,\"shape\":[700,1900]}]}";
    write_file(fmt(&a, "%s/unwrapping/level/PHerc0000/fixture/sheet/layout.json", other),
        other_layout, strlen(other_layout));
    write_file(fmt(&a, "%s/website/sheets/strip/index.html", other), "strip", 5);
    need(!symlink(
             fmt(&a, "%s/website/sheets/strip", other), fmt(&a, "%s/website/sheets/strip", output)),
        "could not link the other unroll's page");
    need(request(curl, &answer, marks,
             fmt(&a,
                 "{\"scan\":\"%s\",\"page\":\"strip\",\"kind\":\"ink\",\"points\":[[5,6]],"
                 "\"width_mm\":0.4}",
                 key)) == 200,
        "could not save a mark on another unroll's page");
    json_object *strip_mark = json_tokener_parse((char *)answer.data);
    sha256_hex((const u8 *)other_layout, strlen(other_layout), hash);
    hash[12] = 0;
    const char *strip_log = fmt(&a, "%s/pages/strip", log);
    need(!strcmp(hash, jstr(strip_mark, "layout")) &&
            exists(fmt(&a, "%s/marks.jsonl", strip_log)) &&
            exists(fmt(&a, "%s/layouts/%s.json", strip_log, hash)),
        "another unroll's mark was not kept with its own layout");
    json_object_put(strip_mark);
    need(request(curl, &answer, fmt(&a, "%s?scan=%s&page=strip", marks, key), NULL) == 200,
        "could not load another unroll's marks");
    json_object *shown = json_tokener_parse((char *)answer.data);
    need(json_object_array_length(shown) == 1 &&
            !strcmp(jstr(json_object_array_get_idx(shown, 0), "kind"), "ink"),
        "another unroll's page lost its mark");
    json_object_put(shown);
    need(request(curl, &answer, fmt(&a, "%s?scan=%s&page=%s", marks, key, page), NULL) == 200 &&
            !strcmp((char *)answer.data, "[]") &&
            request(curl, &answer, fmt(&a, "%s?scan=%s", marks, key), NULL) == 200 &&
            !strcmp((char *)answer.data, "[]"),
        "a mark moved to another unroll of the same scan");
    const char *bundle = fmt(&a, "%s/bundle", temp);
    write_file(fmt(&a, "%s/meta/layout.json", bundle), layout, strlen(layout));
    write_file(fmt(&a, "%s/sheet-PHerc0000/index.html", bundle), "bundle", 6);
    need(!symlink(
             fmt(&a, "%s/sheet-PHerc0000", bundle), fmt(&a, "%s/website/sheets/bundle", output)),
        "could not link a downloaded viewer page");
    need(request(curl, &answer, marks,
             fmt(&a,
                 "{\"scan\":\"%s\",\"page\":\"bundle\",\"kind\":\"spot\",\"points\":[[1,1]],"
                 "\"width_mm\":0.4}",
                 key)) == 200 &&
            exists(fmt(&a, "%s/pages/bundle/marks.jsonl", log)) &&
            request(curl, &answer, fmt(&a, "%s?scan=%s&page=bundle", marks, key), NULL) == 200 &&
            strstr((char *)answer.data, "\"spot\"") &&
            request(curl, &answer, fmt(&a, "%s?scan=%s&page=strip", marks, key), NULL) == 200 &&
            !strstr((char *)answer.data, "\"spot\""),
        "a downloaded viewer page did not keep its own marks");
    need(request(curl, &answer, marks,
             fmt(&a,
                 "{\"scan\":\"%s\",\"page\":\"%s\",\"kind\":\"blank\",\"points\":[[7,8]],"
                 "\"width_mm\":0.4}",
                 key, page)) == 200,
        "could not save a mark on the main unroll");
    const char *names[] = {"placed-taught.json", "placed.json", "layout.json"};
    Bytes before[3];
    for (int i = 0; i < 3; i++) {
        const char *file = fmt(&a, "%s/sheet/%s", base, names[i]);
        before[i] = exists(file) ? read_file(&a, file) : (Bytes){0};
        if (before[i].data) {
            write_file(file, other_layout, strlen(other_layout));
        }
    }
    need(request(curl, &answer, fmt(&a, "%s?scan=%s&page=%s", marks, key, page), NULL) == 200 &&
            !strcmp((char *)answer.data, "[]"),
        "a mark of the replaced unroll stayed on the new one");
    for (int i = 0; i < 3; i++) {
        if (before[i].data) {
            write_file(fmt(&a, "%s/sheet/%s", base, names[i]), before[i].data, before[i].size);
        }
    }
    need(request(curl, &answer, fmt(&a, "%s?scan=%s&page=%s", marks, key, page), NULL) == 200,
        "could not reload the main unroll's marks");
    shown = json_tokener_parse((char *)answer.data);
    need(json_object_array_length(shown) == 1 &&
            !strcmp(jstr(json_object_array_get_idx(shown, 0), "kind"), "blank"),
        "the main unroll's mark did not come back with its layout");
    json_object_put(shown);
    const char *part = "part/fixture";
    write_file(fmt(&a, "%s/website/sheets/fixture/placed.json", output), layout, strlen(layout));
    need(request(curl, &answer, marks,
             "{\"scan\":\"part/fixture\",\"kind\":\"blank\",\"points\":[[1,2]],"
             "\"width_mm\":0.4}") == 200,
        "could not save a part-page mark");
    need(request(curl, &answer, marks,
             "{\"scan\":\"part/fixture\",\"kind\":\"blank\",\"points\":[[1,2]],"
             "\"width_mm\":0.0}") == 400,
        "viewer accepted an invalid stroke");
    need(request(curl, &answer, fmt(&a, "%s?scan=%s", marks, part), NULL) == 200,
        "could not load part-page marks");
    shown = json_tokener_parse((char *)answer.data);
    need(json_object_array_length(shown) == 1 &&
            !strcmp(jstr(json_object_array_get_idx(shown, 0), "kind"), "blank"),
        "part-page marks were not kept separately");
    json_object_put(shown);
    kill(viewer, SIGTERM);
    need(waitpid(viewer, NULL, 0) == viewer, "could not stop the fixture viewer");
    viewer = fork();
    need(viewer >= 0, "could not restart the fixture viewer");
    if (!viewer) {
        execl(argv[1], argv[1], "serve", port, (char *)NULL);
        _exit(127);
    }
    const char *part_marks = fmt(&a, "%s?scan=%s", marks, part);
    long restarted = 0;
    for (int i = 0; !restarted && i < 100; i++) {
        restarted = request(curl, &answer, part_marks, NULL);
        if (!restarted) {
            usleep(20000);
        }
    }
    shown = json_tokener_parse((char *)answer.data);
    need(restarted == 200 && shown && json_object_array_length(shown) == 1 &&
            !strcmp(jstr(json_object_array_get_idx(shown, 0), "kind"), "blank"),
        "fixture mark did not survive a viewer restart");
    json_object_put(shown);
    char *away = fmt(&a, "%s/outputs-unmounted", temp);
    need(!rename(output, away), "could not simulate an unavailable output root");
    long offline = request(curl, &answer, status_url, NULL);
    need(!rename(away, output) && offline == 503,
        "viewer did not report an unavailable output root");
    curl_easy_cleanup(curl);
    json_object_put(mark);
    json_object_put(meta);
    tile_index_close(&index);
    refresh_fixture(&a, argv[1]);
    panel_fixture(&a, argv[1]);
    pyramid_golden(&a);
    suggested_golden();
    proposals_golden(&a, argv[1]);
    munmap(a.base, a.size);
    puts("website e2e passed");
    return 0;
}
