#include "../base.h"
#include <curl/curl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <zlib.h>

#include "../storage/tile_store.c"

#define TILE 512
#define COLUMNS 4
#define ROWS 2
#define PIXEL_MM 0.01
#define SHEET_VALUE 120

static char temp[] = "/tmp/hx-viewer.XXXXXX";
static const char *binary;
static pid_t server, browser;
static int port;

static void clean(void) {
    if (server > 0) {
        kill(server, SIGTERM);
        waitpid(server, NULL, 0);
    }
    if (browser > 0) {
        kill(browser, SIGKILL);
        waitpid(browser, NULL, 0);
    }
    pid_t pid = fork();
    if (!pid) {
        execlp("rm", "rm", "-rf", temp, (char *)NULL);
        _exit(127);
    }
    if (pid > 0)
        waitpid(pid, NULL, 0);
}

static void pause_ms(int ms) {
    struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000};
    while (nanosleep(&t, &t) && errno == EINTR) {
    }
}

static int level_value(int c, int r) { return 30 + 50 * c + 40 * r; }

static int ink_value(int c, int r) {
    (void)r;
    return c < 2 ? 200 : 100;
}

static int flat_layer(int c, int r) { return c == 3 && r == 1 ? 0 : level_value(c, r); }

static int sheet_value(int c, int r) {
    (void)c;
    (void)r;
    return SHEET_VALUE;
}

static void chunk(Arena *a, Bytes *out, const char *kind, const u8 *data, size_t n) {
    u8 *at = out->data + out->size;
    at[0] = n >> 24, at[1] = n >> 16, at[2] = n >> 8, at[3] = n;
    memcpy(at + 4, kind, 4);
    if (n)
        memcpy(at + 8, data, n);
    uLong crc = crc32(crc32(0, NULL, 0), at + 4, (uInt)n + 4);
    at[8 + n] = crc >> 24, at[9 + n] = crc >> 16, at[10 + n] = crc >> 8, at[11 + n] = crc;
    out->size += n + 12;
    (void)a;
}

static Bytes filtered_png(Arena *a, const u8 *pixels, int w, int h, const int *filters, int count) {
    size_t line = (size_t)w + 1;
    u8 *rows = push(a, line * h);
    for (int y = 0; y < h; y++) {
        const u8 *row = pixels + (size_t)y * w, *above = y ? row - w : NULL;
        int kind = filters[y % count];
        u8 *out = rows + y * line;
        out[0] = (u8)kind;
        for (int x = 0; x < w; x++) {
            int left = x ? row[x - 1] : 0, up = above ? above[x] : 0;
            int corner = x && above ? above[x - 1] : 0, guess = 0;
            if (kind == 1)
                guess = left;
            else if (kind == 2)
                guess = up;
            else if (kind == 3)
                guess = (left + up) / 2;
            else if (kind == 4) {
                int pa = abs(up - corner), pb = abs(left - corner),
                    pc = abs(left + up - 2 * corner);
                guess = pa <= pb && pa <= pc ? left : pb <= pc ? up : corner;
            }
            out[1 + x] = (u8)(row[x] - guess);
        }
    }
    uLongf packed = compressBound(line * h);
    u8 *deflated = push(a, packed);
    need(compress(deflated, &packed, rows, line * h) == Z_OK, "cannot deflate a PNG");
    Bytes png = {.data = push(a, packed + 64)};
    memcpy(png.data, "\x89PNG\r\n\x1a\n", 8);
    png.size = 8;
    u8 header[13] = {w >> 24, w >> 16, w >> 8, w, h >> 24, h >> 16, h >> 8, h, 8, 0, 0, 0, 0};
    chunk(a, &png, "IHDR", header, 13);
    chunk(a, &png, "IDAT", deflated, packed);
    chunk(a, &png, "IEND", NULL, 0);
    return png;
}

static Bytes tile_png(Arena *a, const u8 *pixels, int w, int h, const int *filters, int count) {
    return filters ? filtered_png(a, pixels, w, h, filters, count) : png_encode(a, pixels, w, h, 6);
}

static json_object *pair(int x, int y) {
    json_object *p = json_object_new_array();
    json_object_array_add(p, json_object_new_int(x));
    json_object_array_add(p, json_object_new_int(y));
    return p;
}

static TileLayer *layer_of(TilePack *pack, const char *name, json_object *grid) {
    json_object *meta = json_object_new_object();
    json_object_object_add(meta, "grid", grid);
    json_object_object_add(meta, "raw", json_object_new_boolean(1));
    json_object_object_add(meta, "low", json_object_new_double(0.0));
    json_object_object_add(meta, "high", json_object_new_double(255.0));
    TileLayer *layer = tile_layer_add(pack, name, meta);
    json_object_put(meta);
    need(layer != NULL, "cannot add layer %s", name);
    return layer;
}

static void add(TilePack *pack, TileLayer *layer, int level, int c, int r, Bytes png) {
    need(tile_blob(pack, layer, tile_cell(layer, level, c, r), png.data, png.size),
        "cannot store a tile");
}

static void write_layer(Arena *a, TilePack *pack, const char *name, int (*value)(int, int)) {
    json_object *grid = json_object_new_array();
    json_object_array_add(grid, pair(COLUMNS, ROWS));
    json_object_array_add(grid, pair(2, 1));
    json_object_array_add(grid, pair(1, 1));
    TileLayer *layer = layer_of(pack, name, grid);
    int width = COLUMNS * TILE, height = ROWS * TILE;
    u8 *whole = push(a, (size_t)width * height);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            whole[(size_t)y * width + x] = (u8)value(x / TILE, y / TILE);
    const int every[] = {0, 1, 2, 3, 4};
    u8 *cut = push(a, (size_t)TILE * TILE);
    for (int level = 0; level < 3; level++) {
        int step = 1 << level, w = (width + step - 1) / step, h = (height + step - 1) / step;
        for (int c = 0; c * TILE < w; c++) {
            for (int r = 0; r * TILE < h; r++) {
                int any = 0;
                for (int y = 0; y < TILE; y++) {
                    for (int x = 0; x < TILE; x++) {
                        int gx = (c * TILE + x) * step, gy = (r * TILE + y) * step;
                        u8 v = gx < width && gy < height ? whole[(size_t)gy * width + gx] : 0;
                        cut[y * TILE + x] = v;
                        any |= v;
                    }
                }
                if (any) {
                    size_t saved = a->used;
                    add(pack, layer, level, c, r,
                        tile_png(a, cut, TILE, TILE, level ? NULL : every, level ? 0 : 5));
                    a->used = saved;
                }
            }
        }
    }
}

typedef struct {
    const char *path, *name;
    int (*value)(int, int);
} Layer;

static const char *sheet_folder(Arena *a) {
    return fmt(a, "%s/outputs/website/sheets/sheet-test", temp);
}

static long request(Arena *a, const char *method, const char *path, const char *body, Bytes *out);

static void serve(Arena *a, const Layer *layers, int count, const char *tint) {
    const char *folder = sheet_folder(a);
    mkdirs(folder);
    TilePack pack;
    need(tile_pack_open(&pack, folder), "cannot open the test pack");
    json_object *shown = json_object_new_array();
    for (int i = 0; i < count; i++) {
        write_layer(a, &pack, layers[i].path, layers[i].value);
        json_object *row = json_object_new_object();
        json_object_object_add(row, "path", json_object_new_string(layers[i].path));
        json_object_object_add(row, "mm", json_object_new_double(PIXEL_MM));
        json_object_object_add(row, "levels", json_object_new_int(3));
        json_object_object_add(row, "low", json_object_new_double(0.0));
        json_object_object_add(row, "high", json_object_new_double(255.0));
        json_object_object_add(row, "raw", json_object_new_boolean(1));
        json_object_object_add(row, "name", json_object_new_string(layers[i].name));
        json_object_array_add(shown, row);
    }
    need(tile_pack_publish(&pack), "cannot publish the test pack");
    tile_pack_close(&pack);
    json_object *meta = json_object_new_object(), *area = json_object_new_object();
    json_object_object_add(area, "x", json_object_new_int(0));
    json_object_object_add(area, "y", json_object_new_int(0));
    json_object_object_add(area, "w", json_object_new_double(COLUMNS * TILE * PIXEL_MM));
    json_object_object_add(area, "h", json_object_new_double(ROWS * TILE * PIXEL_MM));
    json_object_object_add(meta, "title", json_object_new_string("test sheet"));
    json_object_object_add(meta, "layers", shown);
    json_object_object_add(meta, "area", area);
    json_object_object_add(meta, "tile", json_object_new_int(TILE));
    json_object_object_add(meta, "back", json_object_new_string("../index.html"));
    json_object_object_add(meta, "scan", json_object_new_string("level/scroll/volume"));
    const char *empty[] = {"boxes", "places", "shelves", "suggested"};
    for (int i = 0; i < 4; i++)
        json_object_object_add(meta, empty[i], json_object_new_array());
    json_object_object_add(meta, "tint", json_object_new_string(tint));
    Bytes template = read_file(a, "viewer/assets/sheet_viewer.html");
    const char *html = fmt(a, "%.*s", (int)template.size, (const char *)template.data);
    const char *keys[] = {"__TITLE__", "__META__", "__FLOOR__", "__SCAN__"};
    const char *values[] = {"test sheet",
        json_object_to_json_string_ext(
            meta, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE),
        "160", "0.3"};
    for (int i = 0; i < 4; i++) {
        const char *at = strstr(html, keys[i]);
        need(at != NULL, "the viewer template has no %s", keys[i]);
        html = fmt(a, "%.*s%s%s", (int)(at - html), html, values[i], at + strlen(keys[i]));
    }
    write_file(fmt(a, "%s/index.html", folder), html, strlen(html));
    json_object_put(meta);
    const char *layout = "{\"pieces\": []}";
    write_file(fmt(a, "%s/outputs/unwrapping/level/scroll/volume/sheet/layout.json", temp), layout,
        strlen(layout));
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t bytes = sizeof(address);
    need(fd >= 0 && !bind(fd, (struct sockaddr *)&address, bytes) &&
            !getsockname(fd, (struct sockaddr *)&address, &bytes),
        "cannot reserve a port");
    close(fd);
    port = ntohs(address.sin_port);
    server = fork();
    need(server >= 0, "cannot fork the site server");
    if (!server) {
        char number[16];
        snprintf(number, sizeof(number), "%d", port);
        execl(binary, binary, "serve", number, (char *)NULL);
        _exit(127);
    }
    for (int i = 0; i < 200; i++) {
        Bytes page;
        if (request(a, "GET", "/sheets/sheet-test/index.html", NULL, &page) == 200)
            return;
        pause_ms(25);
    }
    need(0, "the site server did not start");
}

static void stop(void) {
    kill(server, SIGTERM);
    waitpid(server, NULL, 0);
    server = 0;
    pid_t pid = fork();
    if (!pid) {
        execlp("rm", "rm", "-rf", temp, (char *)NULL);
        _exit(127);
    }
    waitpid(pid, NULL, 0);
    need(!mkdir(temp, 0700), "cannot make the fixture again");
}

typedef struct {
    Arena *a;
    Bytes *out;
    size_t capacity;
} Sink;

static size_t consume(char *data, size_t size, size_t count, void *opaque) {
    Sink *sink = opaque;
    size_t n = size * count;
    if (sink->out->size + n > sink->capacity) {
        size_t capacity = (sink->out->size + n) * 2;
        u8 *grown = push(sink->a, capacity);
        memcpy(grown, sink->out->data, sink->out->size);
        sink->out->data = grown;
        sink->capacity = capacity;
    }
    memcpy(sink->out->data + sink->out->size, data, n);
    sink->out->size += n;
    return n;
}

static long request(Arena *a, const char *method, const char *path, const char *body, Bytes *out) {
    *out = (Bytes){0};
    Sink sink = {a, out, 0};
    CURL *curl = curl_easy_init();
    need(curl != NULL, "cannot start a request");
    struct curl_slist *headers = NULL;
    curl_easy_setopt(curl, CURLOPT_URL, fmt(a, "http://127.0.0.1:%d%s", port, path));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, consume);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sink);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 10000L);
    if (body) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    }
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method);
    long code = 0;
    if (curl_easy_perform(curl) == CURLE_OK)
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return code;
}

static json_object *tile_map(Arena *a, Bytes *body) {
    need(request(a, "GET", "/tilemap?dir=/sheets/sheet-test/", NULL, body) == 200,
        "the tile map is missing");
    u32 size;
    memcpy(&size, body->data, 4);
    json_tokener *tok = json_tokener_new();
    json_object *head = json_tokener_parse_ex(tok, (const char *)body->data + 4, (int)size);
    json_tokener_free(tok);
    need(head != NULL, "the tile map head does not parse");
    return head;
}

static const char *tiles_body(Arena *a, const char *dir, const char *stamp, const char *layer,
    const int (*asked)[3], int count, int repeat) {
    char *list = fmt(a, "%s", "");
    for (int k = 0; k < repeat; k++)
        for (int i = 0; i < count; i++)
            list = fmt(a, "%s%s[\"%s\",%d,%d,%d]", list, *list ? "," : "", layer, asked[i][0],
                asked[i][1], asked[i][2]);
    return fmt(a, "{\"dir\":\"%s\",\"stamp\":\"%s\",\"tiles\":[%s]}", dir, stamp, list);
}

static void tile_map_and_batches(Arena *a) {
    Layer layers[] = {{"layer0", "+0 \xc2\xb5m", flat_layer}};
    serve(a, layers, 1, "");
    Bytes body;
    json_object *head = tile_map(a, &body);
    json_object *grid = json_object_object_get(
        json_object_object_get(json_object_object_get(head, "layers"), "layer0"), "grid");
    need(!strcmp(
             json_object_to_json_string_ext(grid, JSON_C_TO_STRING_PLAIN), "[[4,2],[2,1],[1,1]]"),
        "the tile map names the layer's grid");
    u32 size;
    memcpy(&size, body.data, 4);
    const u8 *bits = body.data + 4 + size;
    int directory = open(sheet_folder(a), O_RDONLY | O_DIRECTORY);
    TileIndex index;
    need(directory >= 0 && tile_index_open(&index, directory) == 1, "cannot open the test pack");
    close(directory);
    need((size_t)json_object_get_int64(json_object_object_get(head, "cells")) == index.cells,
        "the tile map counts every cell");
    for (size_t k = 0; k < index.cells; k++) {
        uint64_t offset;
        uint32_t length;
        need(tile_index_blob(&index, k, &offset, &length), "damaged test index");
        need(((bits[k / 8] >> (k % 8)) & 1) == (length > 0), "the tile map marks cell %zu wrongly",
            k);
    }
    need(!((bits[(COLUMNS + 3) / 8] >> ((COLUMNS + 3) % 8)) & 1), "the empty cell is unmarked");
    const char *stamp = json_object_get_string(json_object_object_get(head, "stamp"));
    int asked[COLUMNS * ROWS + 1][3];
    int n = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLUMNS; c++)
            asked[n][0] = 0, asked[n][1] = c, asked[n][2] = r, n++;
    asked[n][0] = 1, asked[n][1] = 1, asked[n][2] = 0, n++;
    Bytes tiles;
    need(request(a, "POST", "/tiles",
             tiles_body(a, "/sheets/sheet-test/", stamp, "layer0", asked, n, 1), &tiles) == 200 &&
            tiles.size == (size_t)n * TILE * TILE,
        "a batch comes back decoded");
    json_object *entry = json_object_object_get(index.layers, "layer0");
    for (int i = 0; i < n; i++) {
        TileSpan span;
        need(tile_index_level(&index, entry, asked[i][0], &span), "bad test level");
        uint64_t offset;
        uint32_t length;
        need(tile_index_blob(
                 &index, span.first + asked[i][2] * span.w + asked[i][1], &offset, &length),
            "bad test cell");
        const u8 *got = tiles.data + (size_t)i * TILE * TILE;
        if (!length) {
            for (int p = 0; p < TILE * TILE; p++)
                need(!got[p], "an empty cell comes back black");
            continue;
        }
        Bytes stored = {.data = push(a, length), .size = length};
        need(transfer_at(index.pack, stored.data, length, offset, 0), "cannot read a test tile");
        int w, h;
        u8 *expected = png_decode(a, stored, &w, &h);
        need(w == TILE && h == TILE && !memcmp(got, expected, (size_t)TILE * TILE),
            "tile %d comes back as stored", i);
    }
    for (int p = 0; p < TILE * TILE; p++)
        need(tiles.data[p] == level_value(0, 0), "the first tile holds its level value");
    tile_index_close(&index);
    Bytes answer;
    need(request(a, "POST", "/tiles",
             tiles_body(a, "/sheets/sheet-test/", "0", "layer0", asked, 1, 1), &answer) == 409,
        "a stale stamp is refused");
    need(request(a, "POST", "/tiles",
             tiles_body(a, "/sheets/sheet-test/", stamp, "layer0", asked, n, 10), &answer) == 400,
        "an oversized batch is refused");
    need(request(a, "POST", "/tiles",
             tiles_body(a, "/sheets/missing/", stamp, "layer0", asked, 1, 1), &answer) == 404,
        "a missing sheet's batch is not found");
    need(request(a, "GET", "/tilemap?dir=/sheets/missing/", NULL, &answer) == 404,
        "a missing sheet's tile map is not found");
    json_object_put(head);
}

static void every_png_filter(Arena *a) {
    Pcg64 rng = pcg64(7);
    u8 *smooth = push(a, (size_t)TILE * TILE), *noise = push(a, (size_t)TILE * TILE);
    for (int y = 0; y < TILE; y++) {
        int sum = 0;
        for (int x = 0; x < TILE; x++) {
            sum += (int)draw(&rng, 7) - 3;
            smooth[y * TILE + x] = (u8)sum;
        }
    }
    for (int p = 0; p < TILE * TILE; p++)
        noise[p] = (u8)pcg64_next32(&rng);
    const u8 *samples[] = {noise, smooth};
    const int filters[6][5] = {{0}, {1}, {2}, {3}, {4}, {4, 3, 2, 1, 0}};
    const int counts[7] = {1, 1, 1, 1, 1, 5, 0};
    const char *folder = sheet_folder(a);
    TilePack pack;
    need(tile_pack_open(&pack, folder), "cannot reopen the test pack");
    json_object *grid = json_object_new_array();
    json_object_array_add(grid, pair(15, 1));
    TileLayer *layer = layer_of(&pack, "filters", grid);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 7; j++)
            add(&pack, layer, 0, i * 7 + j, 0,
                tile_png(a, samples[i], TILE, TILE, counts[j] ? filters[j] : NULL, counts[j]));
    u8 *corner = push(a, 200 * 100);
    for (int y = 0; y < 100; y++)
        memcpy(corner + y * 200, smooth + y * TILE, 200);
    add(&pack, layer, 0, 14, 0, png_encode(a, corner, 200, 100, 6));
    need(tile_pack_publish(&pack), "cannot publish the filter tiles");
    tile_pack_close(&pack);
    Bytes body;
    json_object *head = tile_map(a, &body);
    int asked[15][3];
    for (int c = 0; c < 15; c++)
        asked[c][0] = 0, asked[c][1] = c, asked[c][2] = 0;
    Bytes decoded;
    need(request(a, "POST", "/tiles",
             tiles_body(a, "/sheets/sheet-test/",
                 json_object_get_string(json_object_object_get(head, "stamp")), "filters", asked,
                 15, 1),
             &decoded) == 200 &&
            decoded.size == (size_t)15 * TILE * TILE,
        "the filter tiles come back");
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 7; j++)
            need(!memcmp(decoded.data + (size_t)(i * 7 + j) * TILE * TILE, samples[i],
                     (size_t)TILE * TILE),
                "the native decoder reads PNG filter set %d of sample %d", j, i);
    const u8 *last = decoded.data + (size_t)14 * TILE * TILE;
    for (int y = 0; y < TILE; y++)
        for (int x = 0; x < TILE; x++)
            need(last[y * TILE + x] == (y < 100 && x < 200 ? smooth[y * TILE + x] : 0),
                "a small tile is padded with black");
    json_object_put(head);
}

static const char *chrome_path(void) {
    const char *path = getenv("CHROME");
    return path && *path ? path : "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
}

typedef struct {
    int write, read, id;
    char *buffer;
    size_t used, capacity;
    char session[128];
} Chrome;

static json_object *call(Chrome *c, const char *method, json_object *params) {
    json_object *message = json_object_new_object();
    json_object_object_add(message, "id", json_object_new_int(++c->id));
    json_object_object_add(message, "method", json_object_new_string(method));
    json_object_object_add(message, "params", params ? params : json_object_new_object());
    if (*c->session)
        json_object_object_add(message, "sessionId", json_object_new_string(c->session));
    const char *text = json_object_to_json_string_ext(message, JSON_C_TO_STRING_PLAIN);
    size_t n = strlen(text) + 1;
    for (size_t sent = 0; sent < n;) {
        ssize_t k = write(c->write, text + sent, n - sent);
        need(k > 0 || errno == EINTR, "Chrome closed its pipe");
        sent += k > 0 ? (size_t)k : 0;
    }
    json_object_put(message);
    for (;;) {
        char *end;
        while (!(end = memchr(c->buffer, 0, c->used))) {
            if (c->used == c->capacity) {
                c->capacity *= 2;
                c->buffer = realloc(c->buffer, c->capacity);
                need(c->buffer != NULL, "out of memory for Chrome");
            }
            ssize_t k = read(c->read, c->buffer + c->used, c->capacity - c->used);
            need(k > 0 || errno == EINTR, "Chrome stopped answering");
            c->used += k > 0 ? (size_t)k : 0;
        }
        json_object *answer = json_tokener_parse(c->buffer);
        size_t line = (size_t)(end - c->buffer) + 1;
        memmove(c->buffer, c->buffer + line, c->used - line);
        c->used -= line;
        json_object *id;
        if (answer && json_object_object_get_ex(answer, "id", &id) &&
            json_object_get_int(id) == c->id) {
            json_object *error = json_object_object_get(answer, "error");
            need(!error, "Chrome refused %s: %s", method, json_object_to_json_string(error));
            json_object *result = json_object_get(json_object_object_get(answer, "result"));
            json_object_put(answer);
            return result;
        }
        if (answer)
            json_object_put(answer);
    }
}

static json_object *params(const char *json) {
    json_object *p = json_tokener_parse(json);
    need(p != NULL, "bad DevTools parameters %s", json);
    return p;
}

static Chrome chrome_open(int width, int height) {
    int commands[2], replies[2];
    need(!pipe(commands) && !pipe(replies), "cannot make the DevTools pipes");
    char profile[] = "/tmp/hx-chrome.XXXXXX";
    need(mkdtemp(profile) != NULL, "cannot make a Chrome profile");
    browser = fork();
    need(browser >= 0, "cannot fork Chrome");
    if (!browser) {
        int in = dup(commands[0]), out = dup(replies[1]);
        dup2(in, 3);
        dup2(out, 4);
        int null = open("/dev/null", O_WRONLY);
        dup2(null, STDOUT_FILENO);
        dup2(null, STDERR_FILENO);
        char size[64], data[96];
        snprintf(size, sizeof(size), "--window-size=%d,%d", width, height);
        snprintf(data, sizeof(data), "--user-data-dir=%s", profile);
        execl(chrome_path(), chrome_path(), "--headless=new", "--remote-debugging-pipe", data,
            "--no-first-run", size, "--force-device-scale-factor=1", "about:blank", (char *)NULL);
        _exit(127);
    }
    close(commands[0]);
    close(replies[1]);
    Chrome c = {.write = commands[1], .read = replies[0], .capacity = 1 << 20};
    c.buffer = malloc(c.capacity);
    json_object *target = call(&c, "Target.createTarget", params("{\"url\":\"about:blank\"}"));
    char *body = NULL;
    need(asprintf(&body, "{\"targetId\":\"%s\",\"flatten\":true}",
             json_object_get_string(json_object_object_get(target, "targetId"))) > 0,
        "out of memory");
    json_object *session = call(&c, "Target.attachToTarget", params(body));
    free(body);
    snprintf(c.session, sizeof(c.session), "%s",
        json_object_get_string(json_object_object_get(session, "sessionId")));
    json_object_put(session);
    json_object_put(target);
    char *metrics = NULL;
    need(asprintf(&metrics, "{\"width\":%d,\"height\":%d,\"deviceScaleFactor\":1,\"mobile\":false}",
             width, height) > 0,
        "out of memory");
    json_object_put(call(&c, "Emulation.setDeviceMetricsOverride", params(metrics)));
    free(metrics);
    return c;
}

static void chrome_close(Chrome *c) {
    kill(browser, SIGKILL);
    waitpid(browser, NULL, 0);
    browser = 0;
    close(c->write);
    close(c->read);
    free(c->buffer);
}

static json_object *evaluate(Chrome *c, const char *expression) {
    json_object *p = json_object_new_object();
    json_object_object_add(p, "expression", json_object_new_string(expression));
    json_object_object_add(p, "returnByValue", json_object_new_boolean(1));
    json_object *result = call(c, "Runtime.evaluate", p);
    json_object *value =
        json_object_get(json_object_object_get(json_object_object_get(result, "result"), "value"));
    json_object_put(result);
    return value;
}

static json_object *view(Chrome *c) {
    json_object *raw = evaluate(c, "window.viewer ? JSON.stringify(viewer.view()) : null");
    json_object *state = raw ? json_tokener_parse(json_object_get_string(raw)) : NULL;
    json_object_put(raw);
    return state;
}

static int settled(json_object *state) {
    return state && !json_object_get_boolean(json_object_object_get(state, "busy")) &&
        !json_object_get_boolean(json_object_object_get(state, "queued"));
}

static json_object *opened(Chrome *c) {
    char *url = NULL;
    need(asprintf(&url, "{\"url\":\"http://127.0.0.1:%d/sheets/sheet-test/index.html\"}", port) > 0,
        "out of memory");
    json_object_put(call(c, "Page.navigate", params(url)));
    free(url);
    json_object *state = NULL;
    for (int i = 0; i < 200; i++) {
        pause_ms(50);
        json_object_put(state);
        state = view(c);
        if (settled(state) && json_object_get_boolean(json_object_object_get(state, "resident")))
            break;
    }
    need(state && json_object_get_double(json_object_object_get(state, "zoom")) > 0,
        "the viewer did not open");
    json_object_put(evaluate(c, "viewer.voxel()"));
    pause_ms(500);
    for (int i = 0; i < 100; i++) {
        json_object_put(state);
        state = view(c);
        if (settled(state))
            break;
        pause_ms(50);
    }
    return state;
}

typedef struct {
    u8 *rgb;
    int w, h;
} Screen;

static Screen screen(Arena *a, Chrome *c) {
    json_object *shot = call(c, "Page.captureScreenshot", params("{\"format\":\"png\"}"));
    const char *data = json_object_get_string(json_object_object_get(shot, "data"));
    size_t n = strlen(data);
    u8 *png = push(a, n);
    int size = EVP_DecodeBlock(png, (const u8 *)data, (int)n);
    need(size > 0, "the screenshot is not base64");
    size -= (n && data[n - 1] == '=') + (n > 1 && data[n - 2] == '=');
    json_object_put(shot);
    png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop info = png_create_info_struct(p);
    PngIo io = {.in = png, .left = (size_t)size};
    need(!setjmp(png_jmpbuf(p)), "the screenshot does not decode");
    png_set_read_fn(p, &io, png_get);
    png_read_info(p, info);
    png_set_strip_16(p);
    png_set_palette_to_rgb(p);
    png_set_gray_to_rgb(p);
    png_set_strip_alpha(p);
    png_read_update_info(p, info);
    Screen s = {.w = png_get_image_width(p, info), .h = png_get_image_height(p, info)};
    s.rgb = push(a, (size_t)s.w * s.h * 3);
    for (int row = 0; row < s.h; row++)
        png_read_row(p, s.rgb + (size_t)row * s.w * 3, 0);
    png_destroy_read_struct(&p, &info, 0);
    return s;
}

static const u8 *pixel(Screen s, int y, int x) { return s.rgb + ((size_t)y * s.w + x) * 3; }

static void press(Chrome *c, char key) {
    const char *kinds[] = {"keyDown", "keyUp"};
    for (int i = 0; i < 2; i++) {
        char *body = NULL;
        need(asprintf(&body, "{\"type\":\"%s\",\"key\":\"%c\",\"code\":\"Key%c\"}", kinds[i], key,
                 key - 32) > 0,
            "out of memory");
        json_object_put(call(c, "Input.dispatchKeyEvent", params(body)));
        free(body);
    }
}

static void mouse(Chrome *c, const char *kind, int x, int y, int buttons) {
    char *body = NULL;
    need(asprintf(&body,
             "{\"type\":\"%s\",\"x\":%d,\"y\":%d,\"button\":\"left\",\"buttons\":%d,\"clickCount\":"
             "1}",
             kind, x, y, buttons) > 0,
        "out of memory");
    json_object_put(call(c, "Input.dispatchMouseEvent", params(body)));
    free(body);
}

static int lines_in(Arena *a, const char *path) {
    if (access(path, R_OK))
        return 0;
    Bytes text = read_file(a, path);
    int n = 0;
    for (size_t i = 0; i < text.size; i++)
        n += text.data[i] == '\n';
    return n;
}

static json_object *line_of(Arena *a, const char *path, int k) {
    Bytes text = read_file(a, path);
    const char *at = (const char *)text.data;
    for (int i = 0; i < k; i++)
        at = strchr(at, '\n') + 1;
    const char *end = strchr(at, '\n');
    return json_tokener_parse(fmt(a, "%.*s", (int)(end - at), at));
}

static void a_stroke_is_saved_drawn_and_erased(Arena *a) {
    Layer layers[] = {{"layer0", "+0 \xc2\xb5m", flat_layer}};
    serve(a, layers, 1, "");
    const char *log = fmt(a, "%s/library/marks/level_route/scroll/volume/marks.jsonl", temp);
    Chrome c = chrome_open(1024, 600);
    json_object_put(opened(&c));
    for (int i = 0; i < 50; i++) {
        json_object *shown = evaluate(&c, "!document.getElementById('marking').hidden");
        int ready = json_object_get_boolean(shown);
        json_object_put(shown);
        if (ready)
            break;
        pause_ms(50);
    }
    press(&c, 'g');
    mouse(&c, "mousePressed", 200, 300, 1);
    for (int x = 210; x < 420; x += 10)
        mouse(&c, "mouseMoved", x, 300, 1);
    mouse(&c, "mouseReleased", 420, 300, 0);
    for (int i = 0; i < 50 && lines_in(a, log) != 1; i++)
        pause_ms(50);
    need(lines_in(a, log) == 1, "the stroke was not saved");
    json_object *saved = line_of(a, log, 0);
    need(!strcmp(json_object_get_string(json_object_object_get(saved, "kind")), "ink") &&
            json_object_array_length(json_object_object_get(saved, "points")) >= 10 &&
            json_object_get_double(json_object_object_get(saved, "width_mm")) == 0.4,
        "the stroke is saved as ink with its points and width");
    pause_ms(200);
    Screen shot = screen(a, &c);
    const u8 *red = pixel(shot, 300, 310), *beside = pixel(shot, 400, 310);
    need(red[0] > red[1] + 60 && abs(beside[0] - beside[1]) <= 1, "the stroke is drawn in red");
    press(&c, 'g');
    press(&c, 'x');
    mouse(&c, "mousePressed", 310, 300, 1);
    mouse(&c, "mouseReleased", 310, 300, 0);
    for (int i = 0; i < 50 && lines_in(a, log) != 2; i++)
        pause_ms(50);
    need(lines_in(a, log) == 2, "the erase was not saved");
    json_object *removed = line_of(a, log, 1);
    need(!strcmp(json_object_get_string(json_object_object_get(removed, "remove")),
             json_object_get_string(json_object_object_get(saved, "id"))),
        "the erase withdraws the stroke");
    pause_ms(200);
    const u8 *erased = pixel(screen(a, &c), 300, 310);
    need(abs(erased[0] - erased[1]) <= 1, "the erased stroke is gone");
    json_object_put(saved);
    json_object_put(removed);
    chrome_close(&c);
    stop();
}

static void every_tile_at_its_stored_value(Arena *a) {
    Layer layers[] = {{"layer0", "+0 \xc2\xb5m", flat_layer}};
    serve(a, layers, 1, "");
    Chrome c = chrome_open(1024, 600);
    json_object *state = opened(&c);
    Screen shot = screen(a, &c);
    double zoom = json_object_get_double(json_object_object_get(state, "zoom"));
    double ox = json_object_get_double(json_object_object_get(state, "ox"));
    double oy = json_object_get_double(json_object_object_get(state, "oy"));
    int checked = 0;
    for (int col = 0; col < COLUMNS; col++) {
        for (int row = 0; row < ROWS; row++) {
            int x = (int)lround(((col + 0.5) * TILE * PIXEL_MM - ox) * zoom);
            int y = (int)lround(((row + 0.5) * TILE * PIXEL_MM - oy) * zoom);
            if (y >= 60 && y < shot.h - 30 && x >= 0 && x < shot.w) {
                int got = pixel(shot, y, x)[0], expected = flat_layer(col, row);
                need(abs(got - expected) <= 1, "tile %d,%d is drawn at %d, not %d", col, row, got,
                    expected);
                checked++;
            }
        }
    }
    need(checked >= 2, "too few tiles were on screen");
    json_object *label = evaluate(&c, "document.getElementById('label').textContent");
    const char *text = json_object_get_string(label), *tail = "test sheet \xc2\xb7 +0 \xc2\xb5m";
    need(strlen(text) >= strlen(tail) && !strcmp(text + strlen(text) - strlen(tail), tail),
        "the label names the sheet and layer: %s", text);
    json_object_put(label);
    json_object_put(state);
    chrome_close(&c);
    stop();
}

static Screen slide(Arena *a, Chrome *c, const char *name, int value) {
    char *script = NULL;
    need(asprintf(&script,
             "(f => { f.value = %d; f.dispatchEvent(new Event('input')); "
             "})(document.getElementById('%s'))",
             value, name) > 0,
        "out of memory");
    json_object_put(evaluate(c, script));
    free(script);
    pause_ms(300);
    return screen(a, c);
}

static void sliders_redraw_the_purple_ink(Arena *a) {
    Layer layers[] = {
        {"sheet", "brightest across the sheet", sheet_value}, {"taught", "ink", ink_value}};
    serve(a, layers, 2, "purple");
    Chrome c = chrome_open(1024, 600);
    json_object *state = opened(&c);
    double zoom = json_object_get_double(json_object_object_get(state, "zoom"));
    double ox = json_object_get_double(json_object_object_get(state, "ox"));
    double oy = json_object_get_double(json_object_object_get(state, "oy"));
    long y = lround((ROWS * TILE * PIXEL_MM / 2 - oy) * zoom);
    y = y < 80 ? 80 : y > 540 ? 540 : y;
    int columns[2] = {-1, -1};
    for (int col = 0; col < COLUMNS; col++) {
        int x = (int)lround(((col + 0.5) * TILE * PIXEL_MM - ox) * zoom);
        int above = ink_value(col, 0) > 160;
        if (x >= 20 && x < 1004 && columns[above] < 0)
            columns[above] = x;
    }
    need(columns[0] >= 0 && columns[1] >= 0, "ink above and below the floor is on screen");
    json_object_put(evaluate(
        &c, "window.failures = []; addEventListener('error', (e) => failures.push(e.message))"));
    Screen before = screen(a, &c);
    u8 ink[3], under[3];
    memcpy(ink, pixel(before, y, columns[1]), 3);
    memcpy(under, pixel(before, y, columns[0]), 3);
    need(ink[2] > ink[1] + 60, "ink above the floor is purple");
    json_object_put(evaluate(&c,
        "(f => { f.value = '#00ff00'; f.dispatchEvent(new Event('input')); })"
        "(document.getElementById('colour'))"));
    pause_ms(300);
    const u8 *green = pixel(screen(a, &c), y, columns[1]);
    need(green[1] > green[0] + 60 && green[1] > green[2] + 60, "a picked colour paints the ink");
    json_object *picked = evaluate(&c, "JSON.parse(localStorage.inkLook).colour");
    need(!strcmp(json_object_get_string(picked), "#00ff00"), "the picked colour is remembered");
    json_object_put(picked);
    json_object_put(evaluate(&c,
        "document.getElementById('colour').dispatchEvent("
        "new MouseEvent('contextmenu', {cancelable: true}))"));
    pause_ms(300);
    const u8 *own = pixel(screen(a, &c), y, columns[1]);
    need(own[2] > own[1] + 60, "a right-click brings back the page's own tint");
    const u8 *faded = pixel(slide(a, &c, "floor", 220), y, columns[1]);
    need(abs(faded[2] - faded[1]) <= 3 && faded[1] < ink[1], "a higher floor fades the ink");
    const u8 *brighter = pixel(slide(a, &c, "scan", 100), y, columns[0]);
    need(brighter[1] > under[1] + 50, "the scan slider brightens the scan");
    json_object *failures = evaluate(&c, "failures");
    need(json_object_array_length(failures) == 0, "the page raised errors: %s",
        json_object_to_json_string(failures));
    json_object_put(failures);
    json_object *look = evaluate(&c, "JSON.parse(localStorage.inkLook)");
    need(json_object_get_int(json_object_object_get(look, "floor")) == 220 &&
            json_object_get_int(json_object_object_get(look, "scan")) == 1,
        "the look is remembered");
    json_object_put(look);
    json_object_put(state);
    chrome_close(&c);
    stop();
}

int main(int argc, char **argv) {
    need(argc == 2, "usage: sheet_viewer_e2e HERCULEX_BINARY");
    binary = argv[1];
    need(mkdtemp(temp) != NULL, "cannot make the viewer fixture");
    atexit(clean);
    Arena a = arena(1ull << 30);
    setenv("HERCULEX_LIBRARY", fmt(&a, "%s/library", temp), 1);
    setenv("HERCULEX_OUTPUTS", fmt(&a, "%s/outputs", temp), 1);
    tile_map_and_batches(&a);
    every_png_filter(&a);
    stop();
    if (access(chrome_path(), X_OK)) {
        printf(
            "sheet viewer e2e: the browser checks need Chrome at %s (set CHROME)\n", chrome_path());
        puts("sheet viewer e2e passed");
        return 0;
    }
    a_stroke_is_saved_drawn_and_erased(&a);
    every_tile_at_its_stored_value(&a);
    sliders_redraw_the_purple_ink(&a);
    puts("sheet viewer e2e passed");
    return 0;
}
