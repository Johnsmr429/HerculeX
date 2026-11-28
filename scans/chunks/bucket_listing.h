#ifndef HX_BUCKET_LISTING_H
#define HX_BUCKET_LISTING_H

#include <curl/curl.h>
#include <json-c/json.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static size_t inventory_status(char *data, size_t size, size_t count, void *context) {
    size_t n = size * count;
    char *reason = context;
    if (n >= 12 && !memcmp(data, "HTTP/", 5)) {
        const char *end = data + n, *space = memchr(data, ' ', n);
        char *code = space ? memchr(space + 1, ' ', end - space - 1) : NULL;
        reason[0] = 0;
        if (code) {
            end = memchr(code, '\r', end - code);
            size_t length = end ? (size_t)(end - code - 1) : 0;
            snprintf(reason, 96, "%.*s", (int)(length < 95 ? length : 95), code + 1);
        }
    }
    return n;
}

static char *inventory_page(CURL *curl, const char *url, char error[128]) {
    char *page = NULL;
    size_t length = 0;
    char reason[96] = "";
    CURLcode result = CURLE_FAILED_INIT;
    long status = 0;
    if (curl && url) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, inventory_status);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, reason);
        for (int retry = 0; retry < 5; retry++) {
            FILE *out = open_memstream(&page, &length);
            if (!out) {
                break;
            }
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
            result = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
            if (fclose(out)) {
                result = CURLE_WRITE_ERROR;
            }
            if (result == CURLE_OK && status >= 200 && status < 300) {
                return page;
            }
            free(page);
            page = NULL;
            if (retry == 4 || (status >= 400 && status < 500 && status != 429)) {
                break;
            }
            sleep(1 << retry);
        }
    }
    if (result == CURLE_OK) {
        snprintf(error, 128, "HTTP Error %ld: %s", status, reason);
    } else {
        snprintf(error, 128, "%s", curl_easy_strerror(result));
    }
    return NULL;
}

typedef struct {
    size_t next, count;
    void (*run)(void *, size_t);
    void *data;
    pthread_mutex_t lock;
} InventoryWork;

static void *inventory_worker(void *context) {
    InventoryWork *work = context;
    for (;;) {
        pthread_mutex_lock(&work->lock);
        size_t i = work->next++;
        pthread_mutex_unlock(&work->lock);
        if (i >= work->count) {
            return NULL;
        }
        work->run(work->data, i);
    }
}

static void inventory_parallel(
    void (*run)(void *, size_t), void *data, size_t count, size_t workers) {
    pthread_t threads[15];
    InventoryWork work = {0, count, run, data, PTHREAD_MUTEX_INITIALIZER};
    size_t made = 0;
    while (made + 1 < workers && made + 1 < count &&
        !pthread_create(threads + made, NULL, inventory_worker, &work)) {
        made++;
    }
    inventory_worker(&work);
    for (size_t i = 0; i < made; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&work.lock);
}

static char *inventory_token(const char *start, const char *close) {
    const char *end = strstr(start, close);
    return end ? strndup(start, end - start) : NULL;
}

typedef struct {
    json_object *keys, *folders;
    uint64_t count, bytes;
    int complete;
} InventoryListing;

static char *inventory_xml_text(const char *text, const char *end) {
    char *value = strndup(text, end - text);
    if (!value) {
        return NULL;
    }
    char *out = value;
    for (const char *at = text; at < end;) {
        uint32_t code = 0;
        size_t n = 0;
        if (*at == '&') {
            const char *entities[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
            const char escaped[] = "&<>\"'";
            for (size_t i = 0; i < sizeof(entities) / sizeof(*entities); i++) {
                size_t width = strlen(entities[i]);
                if ((size_t)(end - at) >= width && !memcmp(at, entities[i], width)) {
                    n = width;
                    code = (uint8_t)escaped[i];
                    break;
                }
            }
            if (!n && end - at > 3 && at[1] == '#') {
                char *tail = NULL;
                int hex = at[2] == 'x' || at[2] == 'X';
                const char *digits = at + 2 + hex;
                unsigned long cp = strtoul(digits, &tail, hex ? 16 : 10);
                if (tail > digits && tail < end && *tail == ';' && cp > 0 && cp <= 0x10ffff &&
                    !(cp >= 0xd800 && cp <= 0xdfff)) {
                    code = cp;
                    n = tail + 1 - at;
                }
            }
        }
        if (!n) {
            *out++ = *at++;
            continue;
        }
        if (code < 0x80) {
            *out++ = (char)code;
        } else {
            int count = code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
            char *next = out + count;
            for (char *tail = next - 1; tail > out; tail--, code >>= 6) {
                *tail = (char)(0x80 | (code & 63));
            }
            *out = (char)((0xff << (8 - count)) | code);
            out = next;
        }
        at += n;
    }
    *out = 0;
    return value;
}

static int inventory_listing(CURL *curl, const char *base, const char *prefix, int directory,
    int limit, InventoryListing *listing, char error[128]) {
    char *token = NULL, *escaped = curl ? curl_easy_escape(curl, prefix, 0) : NULL;
    int ok = escaped != NULL;
    for (int i_page = 0; ok && i_page < limit; i_page++) {
        char *next = token ? curl_easy_escape(curl, token, 0) : NULL, *url = NULL;
        free(token);
        token = NULL;
        int built = asprintf(&url, "%s/?list-type=2&prefix=%s&max-keys=1000%s%s%s", base, escaped,
                        directory ? "&delimiter=%2F" : "", next ? "&continuation-token=" : "",
                        next ? next : "") >= 0;
        curl_free(next);
        char *page = built ? inventory_page(curl, url, error) : NULL;
        free(url);
        ok = page && strstr(page, "<ListBucketResult ") && strstr(page, "</ListBucketResult>");
        const char opening[] = "<Contents>";
        for (const char *at = ok ? strstr(page, opening) : NULL; ok && at;) {
            const char *end = strstr(at, "</Contents>"), *size = strstr(at, "<Size>");
            const char *key = strstr(at, "<Key>");
            if (!end) {
                ok = 0;
                break;
            }
            if (key && size && key < size && size < end) {
                char *tail;
                uint64_t bytes = strtoull(size + 6, &tail, 10);
                const char *key_end = strstr(key + 5, "</Key>");
                if (tail > size + 6 && !strncmp(tail, "</Size>", 7) && key_end && key_end < size) {
                    listing->count++;
                    listing->bytes += bytes;
                    if (listing->keys) {
                        char *name = inventory_xml_text(key + 5, key_end);
                        json_object *row = name ? json_object_new_object() : NULL;
                        ok = row &&
                            !json_object_object_add(row, "key", json_object_new_string(name)) &&
                            !json_object_object_add(row, "bytes", json_object_new_uint64(bytes)) &&
                            !json_object_array_add(listing->keys, row);
                        free(name);
                        if (!ok) {
                            json_object_put(row);
                        }
                    }
                }
            }
            at = strstr(end + 11, opening);
        }
        const char boundary[] = "<CommonPrefixes><Prefix>";
        const char *at = ok && listing->folders ? strstr(page, boundary) : NULL;
        while (ok && at) {
            at += sizeof(boundary) - 1;
            const char *end = strstr(at, "</Prefix>");
            char *name = end ? inventory_xml_text(at, end) : NULL;
            ok = name && !json_object_array_add(listing->folders, json_object_new_string(name));
            free(name);
            at = end ? strstr(end, boundary) : NULL;
        }
        const char *next_marker = ok ? strstr(page, "<NextContinuationToken>") : NULL;
        if (next_marker) {
            token = inventory_token(next_marker + 23, "</NextContinuationToken>");
            ok = token != NULL;
        } else if (ok) {
            listing->complete = strstr(page, "<IsTruncated>false</IsTruncated>") != NULL;
        }
        free(page);
        if (!token) {
            break;
        }
    }
    if (!ok && !*error) {
        snprintf(error, 128, "invalid S3 listing for %.90s", prefix);
    }
    free(token);
    curl_free(escaped);
    return ok;
}

#endif
