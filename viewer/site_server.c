#include <arpa/inet.h>
#include <ctype.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <openssl/rand.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>

#include "../storage/tile_index.h"

enum { SITE_POINT_LIMIT = 20000, SITE_TILE_LIMIT = 64, SITE_BODY_LIMIT = 4 << 20 };
static pthread_mutex_t site_marks_lock = PTHREAD_MUTEX_INITIALIZER;
static int site_start_status(Arena *a, json_object **status);
static int site_raw_tile(const u8 *data, size_t bytes, u8 *out, int size);
static int site_jpeg_tile(const u8 *data, size_t bytes, u8 *out, int size);

static void site_head(
    FILE *out, int code, const char *type, uint64_t length, int cache, const char *extra) {
    static const char *const reasons[] = {[200] = "OK",
        [206] = "Partial Content",
        [302] = "Found",
        [304] = "Not Modified",
        [400] = "Bad Request",
        [403] = "Forbidden",
        [404] = "Not Found",
        [409] = "Conflict",
        [416] = "Range Not Satisfiable",
        [500] = "Internal Server Error",
        [503] = "Service Unavailable"};
    const char *reason = (size_t)code < sizeof(reasons) / sizeof(*reasons) ? reasons[code] : NULL;
    fprintf(out,
        "HTTP/1.1 %d %s\r\nConnection: close\r\nCache-Control: %s\r\n"
        "Content-Length: %" PRIu64 "\r\n",
        code, reason ? reason : "", cache ? "public, max-age=31536000, immutable" : "no-cache",
        length);
    if (type) {
        fprintf(out, "Content-Type: %s\r\n", type);
    }
    fprintf(out, "%s\r\n", extra ? extra : "");
}

static void site_json(FILE *out, int code, json_object *value) {
    size_t length;
    const char *data = json_object_to_json_string_length(
        value, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE, &length);
    site_head(out, code, "application/json", length, 0, NULL);
    fwrite(data, 1, length, out);
}

static void site_error(FILE *out, int code, const char *message) {
    json_object *value = json_object_new_object();
    json_object_object_add(value, "error", json_object_new_string(message));
    site_json(out, code, value);
    json_object_put(value);
}

static size_t site_items(json_object *value) {
    return json_object_is_type(value, json_type_array) ? json_object_array_length(value) : 0;
}

static int site_name(const char *value) {
    return value && *value && strcmp(value, ".") != 0 && strcmp(value, "..") != 0 &&
        strspn(value, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-") ==
        strlen(value);
}

static char *site_path(Arena *a, const char *url) {
    if (url[0] != '/') {
        return NULL;
    }
    size_t length = strcspn(url, "?#");
    char *copy = push(a, length + 1);
    size_t size = 0;
    for (size_t i = 1; i < length; i++) {
        unsigned char c = url[i];
        if (c == '%' && i + 2 < length && isxdigit((unsigned char)url[i + 1]) &&
            isxdigit((unsigned char)url[i + 2])) {
            char hex[] = {url[i + 1], url[i + 2], 0};
            c = strtol(hex, NULL, 16);
            i += 2;
        }
        if (c < 32 || c == 127 || c == '\\') {
            return NULL;
        }
        copy[size++] = c;
    }
    copy[size] = 0;
    char *rest = copy, *mount = strsep(&rest, "/");
    if (!rest) {
        rest = copy + size;
    }
    for (char *part = rest; part && *part;) {
        size_t n = strcspn(part, "/");
        if ((n == 1 && part[0] == '.') || (n == 2 && !strncmp(part, "..", 2))) {
            return NULL;
        }
        part = part[n] ? part + n + 1 : NULL;
    }
    if (!strcmp(mount, "sheets") || !strcmp(mount, "atlas")) {
        return fmt(a, "%s/website/%s/%s", OUTPUTS, mount, rest);
    }
    if (!strcmp(mount, "website")) {
        return fmt(a, "%s/viewer/assets/%s", HOME, rest);
    }
    if (!strcmp(mount, "submission") && !strncmp(rest, "PHerc0211/work/", 15)) {
        rest += 15;
        const char *end = strrchr(rest, '.');
        int paper =
            !strncmp(rest, "paper/", 6) && end && !strcmp(end, ".jpg") && !strchr(rest + 6, '/');
        int depths = !strncmp(rest, "depths/", 7) && end && !strcmp(end, ".png");
        const char *slash = depths ? strchr(rest + 7, '/') : NULL;
        if (paper || (slash && !strchr(slash + 1, '/'))) {
            return fmt(a, "%s/website/sheets/circles-PHerc0211/review/%s", OUTPUTS, rest);
        }
    }
    return NULL;
}

static const char *site_type(const char *path) {
    const char *suffix = strrchr(path, '.');
    const char *types[][2] = {{".html", "text/html; charset=utf-8"}, {".js", "text/javascript"},
        {".css", "text/css"}, {".json", "application/json"}, {".jpg", "image/jpeg"},
        {".jpeg", "image/jpeg"}, {".png", "image/png"}, {".svg", "image/svg+xml"},
        {".webp", "image/webp"}, {".txt", "text/plain"}};
    for (size_t i = 0; suffix && i < sizeof(types) / sizeof(*types); i++) {
        if (!strcasecmp(suffix, types[i][0])) {
            return types[i][1];
        }
    }
    return "application/octet-stream";
}

static int site_range(const char *range, uint64_t *first, uint64_t *last) {
    if (strncmp(range, "bytes=", 6) != 0 || !isdigit((unsigned char)range[6])) {
        return 0;
    }
    char *stop;
    *first = strtoull(range + 6, &stop, 10);
    if (*stop != '-') {
        return 0;
    }
    const char *end = stop + 1;
    *last = strtoull(end, &stop, 10);
    return stop != end && !*stop;
}

static void site_file(
    Arena *a, FILE *out, const char *url, const char *range, int head, const char *since) {
    int cache = strstr(url, "?v=") != NULL;
    size_t length = strcspn(url, "?");
    if (length == 1 && url[0] == '/') {
        site_head(out, 302, NULL, 0, cache, "Location: /sheets/\r\n");
        return;
    }
    char *path = site_path(a, url);
    struct stat info;
    if (path && !stat(path, &info) && S_ISDIR(info.st_mode) && !range) {
        if (url[length - 1] != '/') {
            char *location = fmt(a, "Location: %.*s/%s\r\n", (int)length, url, url + length);
            site_head(out, 302, NULL, 0, cache, location);
            return;
        }
        path = fmt(a, "%s/index.html", path);
    }
    int fd = path ? open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK) : -1;
    if (fd < 0 || fstat(fd, &info) || !S_ISREG(info.st_mode)) {
        if (fd >= 0) {
            close(fd);
        }
        site_head(out, 404, NULL, 0, cache, NULL);
        return;
    }
    uint64_t first = 0, bytes = info.st_size, last;
    char modified[40];
    struct tm utc;
    gmtime_r(&info.st_mtime, &utc);
    strftime(modified, sizeof(modified), "%a, %d %b %Y %T GMT", &utc);
    char *extra = fmt(a, "Last-Modified: %s\r\n", modified);
    int code = 200;
    if (range) {
        if (!site_range(range, &first, &last)) {
            code = 404;
        } else if (first > last || last >= bytes) {
            code = 416;
        } else {
            code = 206;
            extra = fmt(a, "%sContent-Range: bytes %" PRIu64 "-%" PRIu64 "/%" PRIu64 "\r\n", extra,
                first, last, bytes);
            bytes = last - first + 1;
        }
    } else if (since && !strcmp(since, modified)) {
        code = 304;
    }
    if (code >= 300) {
        bytes = 0;
    }
    site_head(
        out, code, code == 206 ? "application/octet-stream" : site_type(path), bytes, cache, extra);
    u8 block[65536];
    while (!head && bytes) {
        size_t n = bytes > sizeof(block) ? sizeof(block) : bytes;
        if (!transfer_at(fd, block, n, first, 0) || fwrite(block, 1, n, out) != n) {
            break;
        }
        bytes -= n;
        first += n;
    }
    close(fd);
}

static int site_open_tiles(Arena *a, const char *url, TileIndex *index) {
    *index = (TileIndex){.pack = -1};
    char *path = url ? site_path(a, url) : NULL;
    int fd = path ? open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC) : -1;
    if (fd < 0) {
        return 0;
    }
    int opened = tile_index_open(index, fd);
    if (opened < 0) {
        struct stat info;
        if (!fstatat(fd, index_name, &info, 0) &&
            (uint64_t)mtime_ns(fmt(a, "%s/%s", path, index_name)) != index->stamp) {
            tile_index_close(index);
            opened = tile_index_open(index, fd);
        }
    }
    close(fd);
    return opened == 1;
}

static void site_tilemap(Arena *a, FILE *out, const char *url) {
    TileIndex index;
    if (!site_open_tiles(a, url, &index)) {
        site_head(out, 404, NULL, 0, 0, NULL);
        tile_index_close(&index);
        return;
    }
    size_t bytes = (index.cells + 7) / 8, length;
    u8 *present = calloc(bytes ? bytes : 1, 1);
    json_object *meta = json_object_new_object();
    if (present && meta) {
        char stamp[24];
        snprintf(stamp, sizeof(stamp), "%" PRIu64, index.stamp);
        json_object_object_add(meta, "stamp", json_object_new_string(stamp));
        json_object_object_add(meta, "cells", json_object_new_int64(index.cells));
        json_object_object_add(meta, "layers", json_object_get(index.layers));
        const char *json = json_object_to_json_string_length(meta, JSON_C_TO_STRING_PLAIN, &length);
        const u8 *sizes = index.data + index.start + index.cells * 8;
        for (size_t i = 0; i < index.cells; i++) {
            uint32_t n;
            memcpy(&n, sizes + i * 4, 4);
            present[i / 8] |= (n != 0) << (i % 8);
        }
        u8 header[] = {length, length >> 8, length >> 16, length >> 24};
        site_head(out, 200, "application/octet-stream", 4 + length + bytes, 0, NULL);
        fwrite(header, 1, 4, out);
        fwrite(json, 1, length, out);
        fwrite(present, 1, bytes, out);
    } else {
        site_error(out, 500, "could not read tile map");
    }
    free(present);
    json_object_put(meta);
    tile_index_close(&index);
}

static const char *site_string(json_object *obj, const char *key) {
    json_object *value = json_object_object_get(obj, key);
    if (!json_object_is_type(value, json_type_string)) {
        return NULL;
    }
    const char *s = json_object_get_string(value);
    return strlen(s) == (size_t)json_object_get_string_len(value) ? s : NULL;
}

static int site_integer(json_object *value, int64_t *out) {
    switch (json_object_get_type(value)) {
    case json_type_int:
    case json_type_boolean: *out = json_object_get_int64(value); return 1;
    case json_type_double: {
        double parsed = json_object_get_double(value);
        if (!(parsed >= -0x1p63 && parsed < 0x1p63)) {
            return 0;
        }
        *out = (int64_t)parsed;
        return 1;
    }
    case json_type_string: {
        const char *start = json_object_get_string(value);
        char *end;
        errno = 0;
        long long parsed = strtoll(start, &end, 10);
        if (errno || end == start) {
            return 0;
        }
        while (isspace((unsigned char)*end)) {
            end++;
        }
        *out = parsed;
        return (size_t)(end - start) == (size_t)json_object_get_string_len(value);
    }
    default: return 0;
    }
}

static void site_tiles(Arena *a, FILE *out, json_object *body) {
    json_object *value = json_object_object_get(body, "size");
    int64_t size = 512;
    json_object *tiles = json_object_object_get(body, "tiles");
    size_t count = site_items(tiles);
    const char *url = site_string(body, "dir"), *stamp = site_string(body, "stamp");
    if (!url || !stamp || !count || count > SITE_TILE_LIMIT ||
        (value && !site_integer(value, &size)) || size < 16 || size > 2048) {
        site_error(out, 400, "bad tile request");
        return;
    }
    TileIndex index;
    int code = 200;
    const char *error = NULL;
    uint64_t offsets[SITE_TILE_LIMIT] = {0};
    uint32_t lengths[SITE_TILE_LIMIT] = {0}, largest = 0;
    char current[24];
    if (!site_open_tiles(a, url, &index)) {
        code = 404;
        error = "no tile store";
    } else {
        snprintf(current, sizeof(current), "%" PRIu64, index.stamp);
        if (strcmp(current, stamp) != 0) {
            code = 409;
            error = "the tile store was republished";
        }
    }
    for (size_t i = 0; code == 200 && i < count; i++) {
        json_object *tile = json_object_array_get_idx(tiles, i);
        if (!json_object_is_type(tile, json_type_array) || json_object_array_length(tile) != 4) {
            code = 400;
            error = "bad tile coordinates";
            break;
        }
        json_object *layer_name = json_object_array_get_idx(tile, 0);
        const char *name = json_object_is_type(layer_name, json_type_string)
            ? json_object_get_string(layer_name)
            : NULL;
        json_object *layer = name && strlen(name) == (size_t)json_object_get_string_len(layer_name)
            ? json_object_object_get(index.layers, name)
            : NULL;
        if (!layer) {
            code = 400;
            error = "no such layer";
            break;
        }
        int64_t level, x, y;
        if (!site_integer(json_object_array_get_idx(tile, 1), &level) ||
            !site_integer(json_object_array_get_idx(tile, 2), &x) ||
            !site_integer(json_object_array_get_idx(tile, 3), &y)) {
            code = 400;
            error = "bad tile coordinates";
            break;
        }
        TileSpan span;
        if (level >= 0 && tile_index_level(&index, layer, level, &span) && x >= span.x &&
            x < (int64_t)span.x + span.w && y >= span.y && y < (int64_t)span.y + span.h) {
            size_t cell = span.first + (size_t)(y - span.y) * span.w + (x - span.x);
            if (!tile_index_blob(&index, cell, offsets + i, lengths + i)) {
                code = 500;
                error = "could not read tile";
            }
            if (lengths[i] > largest) {
                largest = lengths[i];
            }
        }
    }
    size_t cells = (size_t)size * size;
    u8 *pixels = code == 200 ? calloc(count, cells) : NULL;
    u8 *blob = pixels && largest ? malloc(largest) : NULL;
    if (code == 200 && (!pixels || (largest && !blob))) {
        code = 500;
        error = "could not allocate tiles";
    }
    for (size_t i = 0; code == 200 && i < count; i++) {
        if (lengths[i] &&
            (!transfer_at(index.pack, blob, lengths[i], offsets[i], 0) ||
                !site_raw_tile(blob, lengths[i], pixels + i * cells, size))) {
            code = 500;
            error = "could not decode tile";
        }
    }
    if (code == 200) {
        site_head(out, 200, "application/octet-stream", count * cells, 0, NULL);
        fwrite(pixels, cells, count, out);
    } else {
        site_error(out, code, error);
    }
    free(blob);
    free(pixels);
    tile_index_close(&index);
}

typedef struct {
    struct jpeg_error_mgr manager;
    jmp_buf jump;
} SiteJpegError;

static void site_jpeg_error(j_common_ptr context) {
    SiteJpegError *error = (SiteJpegError *)context->err;
    longjmp(error->jump, 1);
}

static int site_jpeg_read(
    struct jpeg_decompress_struct *codec, const u8 *data, size_t bytes, u8 *out, int size) {
    SiteJpegError error;
    codec->err = jpeg_std_error(&error.manager);
    error.manager.error_exit = site_jpeg_error;
    if (setjmp(error.jump)) {
        return 0;
    }
    jpeg_create_decompress(codec);
    jpeg_mem_src(codec, data, bytes);
    if (jpeg_read_header(codec, TRUE) != JPEG_HEADER_OK || codec->image_width > 8192 ||
        codec->image_height > 8192) {
        return 0;
    }
    codec->out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(codec);
    JSAMPARRAY buffer =
        (*codec->mem->alloc_sarray)((j_common_ptr)codec, JPOOL_IMAGE, codec->output_width, 1);
    size_t width = codec->output_width < (unsigned)size ? codec->output_width : (unsigned)size;
    while (codec->output_scanline < codec->output_height) {
        unsigned y = codec->output_scanline;
        if (jpeg_read_scanlines(codec, buffer, 1) != 1) {
            return 0;
        }
        if (y < (unsigned)size) {
            memcpy(out + (size_t)y * size, buffer[0], width);
        }
    }
    return jpeg_finish_decompress(codec);
}

static int site_jpeg_tile(const u8 *data, size_t bytes, u8 *out, int size) {
    struct jpeg_decompress_struct codec = {0};
    int ok = site_jpeg_read(&codec, data, bytes, out, size);
    codec.err = NULL;
    jpeg_destroy_decompress(&codec);
    return ok;
}

static int site_raw_tile(const u8 *data, size_t bytes, u8 *out, int size) {
    if (bytes >= 8 && !png_sig_cmp(data, 0, 8)) {
        png_image image = {.version = PNG_IMAGE_VERSION};
        if (!png_image_begin_read_from_memory(&image, data, bytes)) {
            return 0;
        }
        image.format = PNG_FORMAT_GRAY;
        int w = image.width, h = image.height;
        u8 *pixels = w == size && h == size ? out : NULL;
        if (!pixels && w > 0 && h > 0 && w <= 8192 && h <= 8192) {
            pixels = malloc((size_t)w * h);
        }
        png_color black = {0};
        int ok = pixels && png_image_finish_read(&image, &black, pixels, 0, NULL);
        if (ok && pixels != out) {
            int width = w < size ? w : size, height = h < size ? h : size;
            for (int y = 0; y < height; y++) {
                memcpy(out + (size_t)y * size, pixels + (size_t)y * w, width);
            }
        }
        if (pixels != out) {
            free(pixels);
        }
        png_image_free(&image);
        return ok;
    }
    return site_jpeg_tile(data, bytes, out, size);
}

static char *site_page_sheet(
    Arena *a, const char *page, const char *scroll, const char *volume, int *own) {
    char real[PATH_MAX], home[PATH_MAX];
    if (!site_name(page) || !realpath(fmt(a, "%s/website/sheets/%s", OUTPUTS, page), real) ||
        !realpath(fmt(a, "%s/website/sheets", OUTPUTS), home)) {
        return NULL;
    }
    char *parent = fmt(a, "%s", real), *slash = strrchr(parent, '/');
    if (!slash || slash == parent) {
        return NULL;
    }
    *slash = 0;
    *own = strcmp(parent, home) != 0;
    if (!*own) {
        return fmt(a, "%s/unwrapping/level/%s/%s/sheet", OUTPUTS, scroll, volume);
    }
    char *root = fmt(a, "%s", parent);
    for (int i = 0; i < 2; i++) {
        slash = strrchr(root, '/');
        if (slash && slash != root) {
            *slash = 0;
        }
    }
    char *found[] = {
        fmt(a, "%s/unwrapping/level/%s/%s/sheet", root, scroll, volume), fmt(a, "%s/meta", parent)};
    for (int i = 0; i < 2; i++) {
        if (exists(fmt(a, "%s/layout.json", found[i]))) {
            return found[i];
        }
    }
    return NULL;
}

static int site_mark_paths(
    Arena *a, const char *scan, const char *page, char **marks, char **layout) {
    if (!scan) {
        return 0;
    }
    char *parts = fmt(a, "%s", scan), *kind = strsep(&parts, "/");
    char *scroll = strsep(&parts, "/"), *volume = strsep(&parts, "/");
    const char *needed;
    if (!strcmp(kind, "level") && site_name(scroll) && site_name(volume) && !parts) {
        *marks = fmt(a, "%s/marks/level_route/%s/%s", LIBRARY, scroll, volume);
        *layout = fmt(a, "%s/unwrapping/level/%s/%s/sheet", OUTPUTS, scroll, volume);
        needed = "layout.json";
        if (page) {
            int own = 0;
            *layout = site_page_sheet(a, page, scroll, volume, &own);
            if (!*layout) {
                return 0;
            }
            if (own) {
                *marks = fmt(a, "%s/pages/%s", *marks, page);
            }
        }
    } else if (!strcmp(kind, "part") && site_name(scroll) && !volume) {
        *marks = fmt(a, "%s/marks/parts/%s", LIBRARY, scroll);
        *layout = fmt(a, "%s/website/sheets/%s", OUTPUTS, scroll);
        needed = "placed.json";
    } else {
        return 0;
    }
    struct stat info;
    return !stat(fmt(a, "%s/%s", *layout, needed), &info) && S_ISREG(info.st_mode);
}

static json_object *site_standing(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) {
        return errno == ENOENT ? json_object_new_array() : NULL;
    }
    if (flock(fileno(file), LOCK_SH)) {
        fclose(file);
        return NULL;
    }
    json_object *active = json_object_new_object();
    char *line = NULL;
    size_t capacity = 0;
    int good = active != NULL;
    while (good && getline(&line, &capacity, file) >= 0) {
        json_object *entry = json_tokener_parse(line);
        const char *remove = site_string(entry, "remove"), *id = site_string(entry, "id");
        if (remove) {
            json_object_object_del(active, remove);
        } else if (id && !json_object_object_add(active, id, entry)) {
            entry = NULL;
        } else {
            good = 0;
        }
        json_object_put(entry);
    }
    json_object *result = good && !ferror(file) ? json_object_new_array() : NULL;
    if (result) {
        json_object_object_foreach(active, id, value) {
            (void)id;
            json_object_array_add(result, json_object_get(value));
        }
    }
    json_object_put(active);
    free(line);
    fclose(file);
    return result;
}

static int site_mark_layout(
    Arena *a, const char *marks, const char *layout, const char *layer, char digest_text[65]) {
    const char *names[] = {"placed-taught.json", "placed.json", "layout.json"};
    for (size_t i = strstr(layer, "taught") ? 0 : 1; i < 3; i++) {
        char *path = fmt(a, "%s/%s", layout, names[i]);
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0 && errno == ENOENT) {
            continue;
        }
        struct stat info;
        int ok = fd >= 0 && !fstat(fd, &info) && S_ISREG(info.st_mode) && info.st_size >= 0;
        size_t size = ok ? (size_t)info.st_size : 0;
        void *data = size ? mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
        if (fd >= 0) {
            close(fd);
        }
        if (!ok || data == MAP_FAILED) {
            return 0;
        }
        sha256_hex(data, size, digest_text);
        digest_text[12] = 0;
        if (marks) {
            char *folder = fmt(a, "%s/layouts", marks);
            char *kept = fmt(a, "%s/%s.json", folder, digest_text);
            ok = exists(kept) || (hx_site_folders(folder) && hx_write_sheet_file(kept, data, size));
        }
        if (data) {
            munmap(data, size);
        }
        return ok;
    }
    return 0;
}

static int site_number(json_object *value, double *number) {
    if (json_object_is_type(value, json_type_string)) {
        const char *s = json_object_get_string(value);
        char *end;
        double parsed = strtod(s, &end);
        if (end == s) {
            return 0;
        }
        while (isspace((unsigned char)*end)) {
            end++;
        }
        *number = parsed;
        return !*end && isfinite(parsed);
    }
    enum json_type type = json_object_get_type(value);
    *number = json_object_get_double(value);
    return (type == json_type_double || type == json_type_int || type == json_type_boolean) &&
        isfinite(*number);
}

static int site_stroke(json_object *body, json_object *entry) {
    double width;
    const char *kind = site_string(body, "kind"), *layer = site_string(body, "layer");
    json_object *points = json_object_object_get(body, "points");
    size_t count =
        json_object_is_type(points, json_type_array) ? json_object_array_length(points) : 0;
    if (!kind ||
        (strcmp(kind, "ink") != 0 && strcmp(kind, "blank") != 0 && strcmp(kind, "spot") != 0) ||
        !site_number(json_object_object_get(body, "width_mm"), &width) || width < .01 ||
        width > 50 || !count || count > SITE_POINT_LIMIT) {
        return 0;
    }
    json_object *rows = json_object_new_array_ext(count);
    if (!rows) {
        return 0;
    }
    json_object_object_add(entry, "kind", json_object_new_string(kind));
    json_object_object_add(entry, "width_mm", jnum(width));
    if (!layer) {
        layer = "";
    }
    size_t length = 0;
    for (int i = 0; i < 40 && layer[length]; i++) {
        length++;
        while (((unsigned char)layer[length] & 0xc0) == 0x80) {
            length++;
        }
    }
    json_object_object_add(entry, "layer", json_object_new_string_len(layer, length));
    json_object_object_add(entry, "points", rows);
    for (size_t i = 0; i < count; i++) {
        json_object *pair = json_object_array_get_idx(points, i);
        double x, y;
        if (!json_object_is_type(pair, json_type_array) || json_object_array_length(pair) != 2 ||
            !site_number(json_object_array_get_idx(pair, 0), &x) ||
            !site_number(json_object_array_get_idx(pair, 1), &y)) {
            return 0;
        }
        json_object *row = json_object_new_array_ext(2);
        json_object_array_add(row, jnum(x));
        json_object_array_add(row, jnum(y));
        json_object_array_add(rows, row);
    }
    return 1;
}

static json_object *site_current_marks(Arena *a, json_object *active, const char *layout) {
    char plain[65], taught[65];
    int have_plain = site_mark_layout(a, NULL, layout, "", plain);
    int have_taught = site_mark_layout(a, NULL, layout, "taught", taught);
    json_object *shown = json_object_new_array();
    for (size_t i = 0; shown && i < site_items(active); i++) {
        json_object *entry = json_object_array_get_idx(active, i);
        const char *kept = site_string(entry, "layout");
        if (!kept || (have_plain && !strcmp(kept, plain)) ||
            (have_taught && !strcmp(kept, taught))) {
            json_object_array_add(shown, json_object_get(entry));
        }
    }
    return shown;
}

static void site_marks(Arena *a, FILE *out, const char *scan, const char *page, json_object *body) {
    char *marks, *layout;
    if (!site_mark_paths(a, scan, page, &marks, &layout)) {
        site_error(out, body ? 400 : 404, "no such scan");
        return;
    }
    char *path = fmt(a, "%s/marks.jsonl", marks);
    pthread_mutex_lock(&site_marks_lock);
    if (!body) {
        json_object *active = site_standing(path);
        pthread_mutex_unlock(&site_marks_lock);
        json_object *shown = active ? site_current_marks(a, active, layout) : NULL;
        if (shown) {
            site_json(out, 200, shown);
        } else {
            site_error(out, 500, "could not read marks");
        }
        json_object_put(shown);
        json_object_put(active);
        return;
    }
    json_object *entry = json_object_new_object(), *remove;
    json_object_object_add(
        entry, "time", json_object_new_string(utc_stamp(a, "%Y-%m-%dT%H:%M:%SZ")));
    int code = 200;
    if (json_object_object_get_ex(body, "remove", &remove)) {
        const char *id = json_object_get_string(remove);
        json_object_object_add(entry, "remove", json_object_new_string(id ? id : "None"));
    } else if (!site_stroke(body, entry)) {
        code = 400;
    } else {
        char digest_text[65], id[33];
        u8 random[16];
        if (RAND_bytes(random, sizeof(random)) == 1 &&
            site_mark_layout(a, marks, layout, site_string(entry, "layer"), digest_text)) {
            random[6] = (random[6] & 15) | 64;
            random[8] = (random[8] & 63) | 128;
            for (size_t i = 0; i < sizeof(random); i++) {
                snprintf(id + i * 2, 3, "%02x", random[i]);
            }
            json_object_object_add(entry, "id", json_object_new_string(id));
            json_object_object_add(entry, "layout", json_object_new_string(digest_text));
        } else {
            code = 500;
        }
    }
    if (code == 200) {
        const char *json = json_object_to_json_string_ext(
            entry, JSON_C_TO_STRING_SPACED | JSON_C_TO_STRING_NOSLASHESCAPE);
        char *line = fmt(a, "%s\n", json);
        int fd = hx_site_folders(marks)
            ? open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0666)
            : -1;
        size_t length = strlen(line);
        off_t start = fd >= 0 && !flock(fd, LOCK_EX) ? lseek(fd, 0, SEEK_END) : -1;
        size_t written = 0;
        while (start >= 0 && written < length) {
            ssize_t n = write(fd, line + written, length - written);
            if (n > 0) {
                written += n;
            } else if (n >= 0 || errno != EINTR) {
                if (written) {
                    ftruncate(fd, start);
                }
                break;
            }
        }
        if (written != length) {
            code = 500;
        }
        if (fd >= 0 && close(fd)) {
            code = 500;
        }
    }
    pthread_mutex_unlock(&site_marks_lock);
    if (code == 200) {
        site_json(out, 200, entry);
    } else {
        site_error(out, code, code == 400 ? "not a stroke" : "could not write marks");
    }
    json_object_put(entry);
}

static void site_http(int fd, Arena *a) {
    struct timeval timeout = {.tv_sec = 15};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    int outfd = dup(fd);
    FILE *in = fdopen(fd, "r"), *out = outfd >= 0 ? fdopen(outfd, "w") : NULL;
    if (in && out) {
        char line[8192], method[8], url[4096], version[9], *range = NULL, *since = NULL;
        size_t length = 0, headers = 0;
        int json = 0, complete = 0, good = 1;
        if (fgets(line, sizeof(line), in) && strchr(line, '\n') &&
            sscanf(line, "%7s %4095s %8s", method, url, version) == 3) {
            while (good && fgets(line, sizeof(line), in)) {
                headers += strlen(line);
                if (!strchr(line, '\n') || headers > 65536) {
                    break;
                }
                line[strcspn(line, "\r\n")] = 0;
                if (!*line) {
                    complete = 1;
                    break;
                }
                char *value = line, *name = strsep(&value, ":");
                if (!value) {
                    break;
                }
                value += strspn(value, " \t");
                size_t at = strlen(value);
                while (at && (value[at - 1] == ' ' || value[at - 1] == '\t')) {
                    value[--at] = 0;
                }
                if (!strcasecmp(name, "Content-Type")) {
                    json = !strcmp(value, "application/json");
                } else if (!strcasecmp(name, "Content-Length")) {
                    char *end;
                    uint64_t requested = strtoull(value, &end, 10);
                    good = end != value && !*end && requested <= SITE_BODY_LIMIT;
                    length = requested;
                } else if (!strcasecmp(name, "Range")) {
                    range = fmt(a, "%s", value);
                } else if (!strcasecmp(name, "If-Modified-Since")) {
                    since = fmt(a, "%s", value);
                }
            }
        }
        if (!complete || !good) {
            site_error(out, 400, "bad HTTP request");
        } else if (!strcmp(method, "GET")) {
            if (!range && !strncmp(url, "/tilemap?dir=", 13)) {
                site_tilemap(a, out, url + 13);
            } else if (!range && !strncmp(url, "/marks?scan=", 12)) {
                char *query = fmt(a, "%s", url + 12), *page = strstr(query, "&page=");
                if (page) {
                    *page = 0;
                    page += 6;
                }
                site_marks(a, out, query, page, NULL);
            } else if (!range && !strcmp(url, "/status")) {
                json_object *status = NULL;
                if (site_start_status(a, &status)) {
                    site_json(out, 200, status);
                } else {
                    site_error(out, 503, "could not read the website status");
                }
                json_object_put(status);
            } else {
                site_file(a, out, url, range, 0, since);
            }
        } else if (!strcmp(method, "HEAD")) {
            site_file(a, out, url, range, 1, since);
        } else if (!strcmp(method, "POST") && json &&
            (!strcmp(url, "/marks") || !strcmp(url, "/tiles"))) {
            char *data = push(a, length + 1);
            json_tokener *parser = json_tokener_new();
            json_object *body = NULL;
            if (parser && fread(data, 1, length, in) == length) {
                json_tokener_set_flags(parser, JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);
                body = json_tokener_parse_ex(parser, data, length);
            }
            if (!json_object_is_type(body, json_type_object) ||
                json_tokener_get_error(parser) != json_tokener_success) {
                site_error(out, 400, "bad JSON request");
            } else if (!strcmp(url, "/marks")) {
                site_marks(a, out, site_string(body, "scan"), site_string(body, "page"), body);
            } else {
                site_tiles(a, out, body);
            }
            json_object_put(body);
            if (parser) {
                json_tokener_free(parser);
            }
        } else {
            site_error(out, 403, "marks only");
        }
    }
    if (out) {
        fclose(out);
    } else if (outfd >= 0) {
        close(outfd);
    }
    if (in) {
        fclose(in);
    } else {
        close(fd);
    }
}

static void *site_worker(void *opaque) {
    int socket_fd = *(int *)opaque;
    Arena a = arena(64 << 20);
    for (;;) {
        int fd = accept(socket_fd, NULL, NULL);
        if (fd >= 0) {
            site_http(fd, &a);
            a.used = 0;
        } else if (errno != EINTR && errno != ECONNABORTED) {
            perror("herculex: accept");
            break;
        }
    }
    munmap(a.base, a.size);
    return NULL;
}

int serve_command(int argc, char **argv) {
    char *end = NULL;
    long port = argc ? strtol(argv[0], &end, 10) : 8765;
    need(argc <= 1 && (!end || (end != argv[0] && !*end)) && port >= 0 && port <= 65535,
        "usage: ./run.sh serve [PORT]");
    int fd = socket(AF_INET, SOCK_STREAM, 0), reuse = 1;
    need(fd >= 0, "cannot open viewer socket: %s", strerror(errno));
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    signal(SIGPIPE, SIG_IGN);
    struct sockaddr_in address = {
        .sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    need(!bind(fd, (struct sockaddr *)&address, sizeof(address)) && !listen(fd, 64),
        "cannot serve viewer on port %ld: %s", port, strerror(errno));
    pthread_t workers[7];
    int started = 0;
    while (started < 7 && !pthread_create(workers + started, NULL, site_worker, &fd)) {
        started++;
    }
    site_worker(&fd);
    close(fd);
    for (int i = 0; i < started; i++) {
        pthread_join(workers[i], NULL);
    }
    return 1;
}
