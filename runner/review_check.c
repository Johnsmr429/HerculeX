#define _GNU_SOURCE
#include <ctype.h>
#include <curl/curl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "../storage/tile_index.h"

typedef struct {
    char *data;
    size_t length, capacity, limit;
} Response;

static size_t receive(char *data, size_t width, size_t count, void *context) {
    Response *reply = context;
    if (width && count > (reply->limit - reply->length) / width) {
        return 0;
    }
    size_t length = width * count;
    size_t needed = reply->length + length + 1;
    if (needed > reply->capacity) {
        size_t size = reply->capacity * 2;
        char *next = realloc(reply->data, size > needed ? size : needed);
        if (!next) {
            return 0;
        }
        reply->data = next;
        reply->capacity = size > needed ? size : needed;
    }
    memcpy(reply->data + reply->length, data, length);
    reply->length += length;
    reply->data[reply->length] = 0;
    return length;
}

static long request(CURL *curl, const char *url, const char *range, long timeout, Response *reply) {
    if (!url) {
        return 0;
    }
    reply->length = 0;
    if (reply->data) {
        reply->data[0] = 0;
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_RANGE, range);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, reply);
    long status = 0;
    if (!curl_easy_perform(curl)) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    }
    return status;
}

static int get(CURL *curl, const char *url, long timeout, Response *reply) {
    long status = request(curl, url, NULL, timeout, reply);
    return status >= 200 && status < 300;
}

static int valid_marks(const Response *reply) {
    json_tokener *parser = json_tokener_new();
    if (!parser) {
        return 0;
    }
    json_tokener_set_flags(parser, JSON_TOKENER_STRICT);
    json_object *marks = json_tokener_parse_ex(parser, reply->data, (int)reply->length);
    size_t end = json_tokener_get_parse_end(parser);
    while (end < reply->length && isspace((unsigned char)reply->data[end])) {
        end++;
    }
    int ok = json_object_is_type(marks, json_type_array) && end == reply->length;
    json_object_put(marks);
    json_tokener_free(parser);
    return ok;
}

static int first_tile(const TileIndex *index, const char *name, uint64_t *offset, uint32_t *size) {
    json_object *layer = json_object_object_get(index->layers, name);
    TileSpan span;
    if (!tile_index_level(index, layer, 0, &span)) {
        return 0;
    }
    for (size_t i = span.first, end = span.first + (size_t)span.w * span.h; i < end; i++) {
        if (!tile_index_blob(index, i, offset, size)) {
            break;
        }
        if (*size) {
            return 1;
        }
    }
    return 0;
}

static int check_tiles(CURL *curl, const char *page, const TileIndex *index, Response *reply) {
    const char *layers[] = {"layer0", "sheet", "taught"};
    char *pack = curl_easy_escape(curl, index->name, 0), *url = NULL;
    if (!pack || asprintf(&url, "%s/%s", page, pack) < 0) {
        curl_free(pack);
        return 0;
    }
    curl_free(pack);
    int ok = 1;
    for (size_t i = 0; ok && i < sizeof(layers) / sizeof(*layers); i++) {
        uint64_t offset;
        uint32_t size;
        char range[48];
        if (!first_tile(index, layers[i], &offset, &size)) {
            fprintf(stderr, "%s: no full-resolution %s tiles\n", page, layers[i]);
            ok = 0;
            break;
        }
        snprintf(range, sizeof(range), "%" PRIu64 "-%" PRIu64, offset, offset + size - 1);
        reply->limit = size;
        ok = request(curl, url, range, 15L, reply) == 206 && reply->length == size &&
            same_tile(index->pack, offset, (const uint8_t *)reply->data, size) == 1;
        if (!ok) {
            fprintf(stderr, "%s: could not read a %s tile over HTTP\n", page, layers[i]);
        }
    }
    free(url);
    return ok;
}

static int verify(
    CURL *curl, const char *base, const char *name, const char *scan, const TileIndex *index) {
    Response reply = {.limit = INT_MAX - 1};
    char *start = NULL, *page = NULL, *href = NULL, *viewer = NULL, *marks = NULL;
    char *part = curl_easy_escape(curl, name, 0);
    int status = -1;
    if (!part || asprintf(&page, "%s/sheets/%s", base, part) < 0 ||
        asprintf(&href, "%s/index.html", name) < 0 ||
        asprintf(&start, "%s/sheets/index.html", base) < 0 ||
        asprintf(&viewer, "%s/index.html", page) < 0 ||
        asprintf(&marks, "%s/marks?scan=%s", base, scan) < 0) {
        goto done;
    }
    if (!get(curl, start, 3L, &reply)) {
        status = 0;
    } else if (!reply.data || !strstr(reply.data, href)) {
        fprintf(stderr, "%s: viewer missing from the live homepage\n", name);
    } else if (!get(curl, viewer, 15L, &reply) || !reply.data || !strstr(reply.data, scan)) {
        fprintf(stderr, "%s: viewer has no marks key for %s\n", name, scan);
    } else if (!get(curl, marks, 15L, &reply) || !reply.length || !valid_marks(&reply)) {
        fprintf(stderr, "%s: invalid marks response\n", scan);
    } else if (check_tiles(curl, page, index, &reply)) {
        status = 1;
    }
done:
    free(reply.data);
    free(marks);
    free(viewer);
    free(start);
    free(page);
    free(href);
    curl_free(part);
    return status;
}

int hx_check_review(int argc, char **argv) {
    if (argc != 4) {
        puts("Usage: ./run.sh viewer.sheet_tiles --verify FOLDER SCAN URL");
        return argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) ? 0 : 2;
    }
    const char *slash = strrchr(argv[1], '/'), *name = slash ? slash + 1 : argv[1];
    int directory = open(argv[1], O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    TileIndex index = {.pack = -1};
    int opened = directory >= 0 ? tile_index_open(&index, directory) : 0;
    if (directory >= 0) {
        close(directory);
    }
    int result = -1;
    CURL *curl = opened == 1 ? curl_easy_init() : NULL;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, receive);
        result = verify(curl, argv[3], name, argv[2], &index);
        curl_easy_cleanup(curl);
    } else {
        fprintf(stderr, "%s: cannot open the review pack or initialize HTTP\n", argv[1]);
    }
    if (result >= 0) {
        printf("%s %" PRIu64 "\n", result ? "checked" : "offline", index.pack_bytes);
    }
    tile_index_close(&index);
    return result < 0;
}
