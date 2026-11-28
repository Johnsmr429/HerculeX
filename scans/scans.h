#include <curl/curl.h>

#define CONNECTIONS 32
#define RETRIES 6

char *zarr_level(Arena *a, const char *sample, const char *volume, int level, char **source) {
    json_object *catalog = json_read(fmt(a, "%s/catalog/metadata.json", LIBRARY));
    need(catalog != 0, "no scan catalog at %s/catalog/metadata.json", LIBRARY);
    json_object *data =
        jget(jget(jget(jget(jget(catalog, "samples"), sample), "volumes"), volume), "data");
    const char *root = 0, *path = 0;
    for (size_t i = 0, n = json_object_array_length(data); i < n; i++) {
        json_object *d = json_object_array_get_idx(data, i);
        if (strcmp(jstr(d, "type"), "ome-zarr") != 0) {
            continue;
        }
        json_object *origin = json_object_array_get_idx(jget(d, "origins"), 0);
        path = jstr(origin, "path");
        root = jstr(json_object_array_get_idx(jget(origin, "access_roots"), 0), "url");
    }
    need(root && path, "%s %s has no ome-zarr product in the catalog", sample, volume);
    char *base = !strncmp(root, "s3://", 5)
        ? fmt(a, "https://%s.s3.us-east-1.amazonaws.com", root + 5)
        : fmt(a, "%s", root);
    char *end = base + strlen(base);
    while (end > base && end[-1] == '/')
        *--end = 0;
    while (*path == '/') {
        path++;
    }
    const char *last = path + strlen(path);
    while (last > path && last[-1] == '/')
        last--;
    *source = fmt(a, "%s/%d", path, level);
    char *url = fmt(a, "%s/%.*s/%d", base, (int)(last - path), path, level);
    json_object_put(catalog);
    return url;
}

typedef struct {
    u8 *data;
    size_t size, cap;
} CurlBody;

size_t body_write(char *data, size_t size, size_t count, void *context) {
    CurlBody *b = context;
    if (size && count > (SIZE_MAX - b->size - 1) / size) {
        return 0;
    }
    size_t n = size * count, required = b->size + n + 1;
    if (required > b->cap) {
        size_t capacity = required <= SIZE_MAX / 2 ? required * 2 : required;
        u8 *buffer = realloc(b->data, capacity);
        if (!buffer) {
            return 0;
        }
        b->data = buffer;
        b->cap = capacity;
    }
    memcpy(b->data + b->size, data, n);
    b->size += n;
    b->data[b->size] = 0;
    return n;
}

json_object *http_json(const char *url) {
    CurlBody b = {0};
    long status = 0;
    for (int attempt = 0; attempt < RETRIES && status != 200; attempt++) {
        b.size = 0;
        CURL *c = curl_easy_init();
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, body_write);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, &b);
        curl_easy_setopt(c, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
        if (curl_easy_perform(c) == CURLE_OK) {
            curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
        }
        curl_easy_cleanup(c);
        if (status != 200 && attempt + 1 < RETRIES) {
            sleep(1 << attempt);
        }
    }
    need(status == 200, "cannot fetch %s", url);
    json_object *o = json_tokener_parse(b.data ? (char *)b.data : "");
    need(o != 0, "%s is not JSON", url);
    free(b.data);
    return o;
}

typedef struct {
    i64 index;
    char path[64];
    i64 first, last;
} Ask;

typedef struct {
    Ask *ask;
    CurlBody body;
    int attempt;
    CURL *curl;
    char range[64];
} Pending;

typedef void (*Deliver)(void *context, Ask *ask, const u8 *body, size_t size);

void start_ask(CURLM *multi, Pending *p, const char *base, Ask *ask) {
    p->ask = ask;
    p->body.size = 0;
    p->curl = curl_easy_init();
    char url[1024];
    snprintf(url, sizeof(url), "%s/%s", base, p->ask->path);
    curl_easy_setopt(p->curl, CURLOPT_URL, url);
    if (p->ask->first >= 0) {
        snprintf(p->range, sizeof(p->range), "%lld-%lld", (long long)p->ask->first,
            (long long)p->ask->last);
        curl_easy_setopt(p->curl, CURLOPT_RANGE, p->range);
    }
    curl_easy_setopt(p->curl, CURLOPT_WRITEFUNCTION, body_write);
    curl_easy_setopt(p->curl, CURLOPT_WRITEDATA, &p->body);
    curl_easy_setopt(p->curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(p->curl, CURLOPT_PRIVATE, p);
    curl_multi_add_handle(multi, p->curl);
}

i64 fetch_ranges(const char *base, Ask *asks, i64 count, Deliver deliver, void *context) {
    CURLM *multi = curl_multi_init();
    curl_multi_setopt(multi, CURLMOPT_MAX_HOST_CONNECTIONS, (long)CONNECTIONS);
    Pending *slots = calloc(CONNECTIONS, sizeof(Pending));
    i64 next = 0, done = 0, failed = 0;
    for (int s = 0; s < CONNECTIONS && next < count; s++)
        start_ask(multi, slots + s, base, asks + next++);
    while (done < count) {
        int running;
        curl_multi_perform(multi, &running);
        curl_multi_poll(multi, 0, 0, 1000, 0);
        CURLMsg *m;
        int left;
        while ((m = curl_multi_info_read(multi, &left))) {
            if (m->msg != CURLMSG_DONE) {
                continue;
            }
            Pending *p;
            long status = 0;
            curl_easy_getinfo(m->easy_handle, CURLINFO_PRIVATE, &p);
            curl_easy_getinfo(m->easy_handle, CURLINFO_RESPONSE_CODE, &status);
            CURLcode result = m->data.result;
            curl_multi_remove_handle(multi, p->curl);
            curl_easy_cleanup(p->curl);
            int answered = result == CURLE_OK && (status == 200 || status == 206);
            int absent = result == CURLE_OK && (status == 403 || status == 404);
            if (!answered && !absent && ++p->attempt < RETRIES) {
                sleep(1 << p->attempt < 30 ? 1 << p->attempt : 30);
                start_ask(multi, p, base, p->ask);
                continue;
            }
            if (answered || absent) {
                deliver(context, p->ask, answered ? p->body.data : 0, answered ? p->body.size : 0);
            } else {
                failed++;
            }
            done++;
            p->attempt = 0;
            if (next < count)
                start_ask(multi, p, base, asks + next++);
        }
    }
    for (int s = 0; s < CONNECTIONS; s++) {
        free(slots[s].body.data);
    }
    free(slots);
    curl_multi_cleanup(multi);
    return failed;
}
