#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <sys/stat.h>

#include "../chunks/bucket_listing.h"

const char hx_usage[] = "Fetch a fragment's surface volume and its infrared truth.";
static const char default_server[] = "https://dl.ash2txt.org/fragments";

typedef struct {
    const char *base, *out;
    json_object *keys;
    atomic_int failed;
} Fragment;

static int fragment_dirs(const char *folder) {
    char *name = strdup(folder);
    if (!name) {
        return 0;
    }
    int ok = 1;
    for (char *p = name + 1; ok; p++) {
        char next = *p;
        if (!next || next == '/') {
            *p = 0;
            ok = mkdir(name, 0777) == 0 || errno == EEXIST;
            *p = next;
        }
        if (!next) {
            break;
        }
    }
    free(name);
    return ok;
}

static int fragment_pending(Fragment *job, int surface) {
    char *url = NULL, error[128];
    int ok = asprintf(&url, "%s/%s", job->base, surface ? "surface_volume/" : "") >= 0;
    CURL *curl = ok ? curl_easy_init() : NULL;
    char *html = curl ? inventory_page(curl, url, error) : NULL;
    if (curl && !html) {
        fprintf(stderr, "%s: %s\n", url, error);
    }
    ok = html != NULL;
    for (const char *start = ok ? strstr(html, "href=\"") : NULL; ok && start;) {
        start += 6;
        const char *end = strchr(start, '"');
        if (!end) {
            break;
        }
        size_t len = (size_t)(end - start);
        int take = surface ? len > 4 && !memcmp(end - 4, ".tif", 4)
                           : (len == 13 && !memcmp(start, "inklabels.png", len)) ||
                (len == 6 && !memcmp(start, "ir.png", len)) ||
                (len == 8 && !memcmp(start, "mask.png", len));
        if (take && start[0] != '?' && !memchr(start, '/', len)) {
            char *key = NULL, *dest = NULL;
            ok = len <= INT_MAX &&
                asprintf(&key, "%s%.*s", surface ? "surface_volume/" : "", (int)len, start) >= 0 &&
                asprintf(&dest, "%s/%s", job->out, key) >= 0;
            if (ok && access(dest, F_OK) != 0) {
                json_object *item = json_object_new_string(key);
                ok = item && json_object_array_add(job->keys, item) == 0;
                if (!ok) {
                    json_object_put(item);
                }
            }
            free(key);
            free(dest);
        }
        start = strstr(end + 1, "href=\"");
    }
    free(html);
    curl_easy_cleanup(curl);
    free(url);
    return ok;
}

static void fragment_transfer(void *ctx, size_t i) {
    Fragment *job = ctx;
    const char *key = json_object_get_string(json_object_array_get_idx(job->keys, i));
    char *url = NULL, *dest = NULL, *part = NULL;
    int ok = asprintf(&url, "%s/%s", job->base, key) >= 0 &&
        asprintf(&dest, "%s/%s", job->out, key) >= 0 &&
        asprintf(&part, "%s.part.XXXXXX", dest) >= 0;
    int fd = ok ? mkstemp(part) : -1;
    FILE *file = fd >= 0 ? fdopen(fd, "wb") : NULL;
    CURL *curl = file ? curl_easy_init() : NULL;
    long status = 0;
    off_t bytes = 0;
    CURLcode result = CURLE_FAILED_INIT;
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);
        for (int attempt = 0; attempt < 5; attempt++) {
            result = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
            if ((result == CURLE_OK && status == 200) ||
                (status >= 400 && status < 500 && status != 429) || attempt == 4) {
                break;
            }
            if (fflush(file) || ftruncate(fd, 0) || fseek(file, 0, SEEK_SET)) {
                result = CURLE_WRITE_ERROR;
                break;
            }
            sleep(1u << attempt);
        }
        bytes = ftello(file);
    }
    ok = curl && result == CURLE_OK && status == 200 && bytes >= 0 && fchmod(fd, 0644) == 0;
    if (file) {
        ok = fclose(file) == 0 && ok;
    } else if (fd >= 0) {
        close(fd);
    }
    ok = ok && rename(part, dest) == 0;
    if (ok) {
        printf("%s %lld\n", key, (long long)bytes);
        fflush(stdout);
    } else {
        if (fd >= 0) {
            unlink(part);
        }
        if (status == 404) {
            fprintf(stderr, "missing %s\n", key);
        } else if (result == CURLE_OK) {
            fprintf(stderr, "%s HTTP %ld\n", key, status);
        } else {
            fprintf(stderr, "%s %s\n", key, curl_easy_strerror(result));
        }
        atomic_fetch_add(&job->failed, 1);
    }
    curl_easy_cleanup(curl);
    free(part);
    free(dest);
    free(url);
}

static int fragment_fetch(const char *server, const char *frag, const char *package,
    const char *working, const char *out) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        return 0;
    }
    char *base = NULL, *surface = NULL;
    Fragment job = {.out = out, .keys = json_object_new_array()};
    int ok = job.keys &&
        asprintf(&base, "%s/%s/%s.volpkg/working/%s", server, frag, package, working) >= 0 &&
        asprintf(&surface, "%s/surface_volume", job.out) >= 0 && fragment_dirs(surface);
    job.base = base;
    if (ok && fragment_pending(&job, 0) && fragment_pending(&job, 1)) {
        inventory_parallel(fragment_transfer, &job, json_object_array_length(job.keys), 8);
        ok = !atomic_load(&job.failed);
    } else {
        ok = 0;
    }
    free(base);
    free(surface);
    json_object_put(job.keys);
    curl_global_cleanup();
    return ok;
}

int main(int argc, char **argv) {
    const char usage[] =
        "Usage: ./run.sh scans.transfer.fetch_fragment FRAGMENT PACKAGE WORKING OUT";
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts(usage);
        return 0;
    }
    if (argc != 5 || !*argv[4]) {
        puts(usage);
        return 2;
    }
    const char *server = getenv("HERCULEX_FRAGMENT_SERVER");
    server = server && *server ? server : default_server;
    if (fragment_fetch(server, argv[1], argv[2], argv[3], argv[4])) {
        puts("[]");
        return 0;
    }
    return 1;
}
