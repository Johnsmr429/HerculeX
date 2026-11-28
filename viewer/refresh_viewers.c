#define _GNU_SOURCE
#include <ctype.h>
#include <fcntl.h>
#include <glob.h>
#include <json-c/json.h>
#include <limits.h>
#include <sys/file.h>
#include <sys/stat.h>

#include "../storage/data_roots.c"
#include "page_template.h"

#ifndef HX_UNITY
const char hx_usage[] = "Refresh published sheet viewers while keeping their tiles and labels.";
#endif

static const char *after(const char *text, const char *name) {
    const char *start = strstr(text, name);
    if (start) {
        start += strlen(name);
        while (isspace((unsigned char)*start)) {
            start++;
        }
    }
    return start;
}

static char *setting(const char *text, const char *name) {
    const char *start = after(text, name);
    if (!start || strncmp(start, "??", 2) != 0) {
        return NULL;
    }
    start += 2;
    while (isspace((unsigned char)*start)) {
        start++;
    }
    size_t length = strspn(start, "0123456789.");
    return length ? strndup(start, length) : NULL;
}

static int replace_page(const char *path, const char *html, size_t length, mode_t mode) {
    char *pending = NULL;
    if (asprintf(&pending, "%s.XXXXXXXX", path) < 0) {
        return 0;
    }
    int fd = mkstemp(pending);
    FILE *output = fd >= 0 ? fdopen(fd, "wb") : NULL;
    int ok = output && !fchmod(fd, mode & 0777) && fwrite(html, 1, length, output) == length;
    if (output) {
        ok = fclose(output) == 0 && ok;
    } else if (fd >= 0) {
        close(fd);
    }
    if (ok) {
        ok = !rename(pending, path);
    }
    if (!ok && fd >= 0) {
        unlink(pending);
    }
    free(pending);
    return ok;
}

static json_object *viewer_meta(
    const char *old, const char **from, size_t *length, const char **tail_at) {
    const char *assign = old;
    do {
        assign = after(assign, "const M");
    } while (assign && *assign != '=');
    if (!assign) {
        return NULL;
    }
    const char *body = assign + 1;
    while (isspace((unsigned char)*body)) {
        body++;
    }
    json_tokener *parser = json_tokener_new();
    json_object *meta = parser ? json_tokener_parse_ex(parser, body, (int)strlen(body)) : NULL;
    size_t end = parser ? json_tokener_get_parse_end(parser) : 0;
    const char *tail = body + end;
    if (*tail == ',') {
        do {
            tail++;
        } while (isspace((unsigned char)*tail));
    }
    int valid = parser && json_tokener_get_error(parser) == json_tokener_success &&
        json_object_is_type(meta, json_type_object) && body[end] == ',' && !strncmp(tail, "TOP", 3);
    if (parser) {
        json_tokener_free(parser);
    }
    if (!valid) {
        json_object_put(meta);
        return NULL;
    }
    *from = body;
    *length = end;
    *tail_at = tail;
    return meta;
}

json_object *hx_viewer_metadata(const char *path) {
    char *old = read_page_text(path);
    const char *body, *tail;
    size_t end;
    json_object *meta = old ? viewer_meta(old, &body, &end, &tail) : NULL;
    free(old);
    return meta;
}

static int update_viewer(
    const char *path, const char *page, const char *wanted, json_object *extras) {
    struct stat info;
    if (lstat(path, &info) || !S_ISREG(info.st_mode)) {
        return 0;
    }
    char *old = read_page_text(path);
    if (!old) {
        return -1;
    }
    const char *body, *tail;
    size_t end;
    json_object *meta = viewer_meta(old, &body, &end, &tail);
    if (!meta) {
        free(old);
        return 0;
    }
    if (wanted) {
        json_object_object_add(meta, "title", json_object_new_string(wanted));
    }
    if (extras) {
        json_object_object_foreach(extras, key, value) {
            json_object_object_add(meta, key, json_object_get(value));
        }
    }
    char *literal = wanted || extras ? strdup(json_object_to_json_string_ext(meta,
                                           JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE))
                                     : strndup(body, end);
    char *floor = setting(tail, "kept.floor");
    char *scan = setting(tail, "kept.scan");
    json_object *title = json_object_object_get(meta, "title");
    char *html = NULL;
    size_t length = 0;
    int status = -1;
    if (literal && floor && scan && json_object_is_type(title, json_type_string)) {
        const char *values[] = {json_object_get_string(title), literal, floor, scan};
        FILE *out = open_memstream(&html, &length);
        if (out) {
            int ok = render_viewer(out, page, values);
            if (!fclose(out) && ok) {
                status = !strcmp(old, html)                          ? 0
                    : replace_page(path, html, length, info.st_mode) ? 1
                                                                     : -1;
            }
        }
    }
    free(html);
    free(scan);
    free(floor);
    free(literal);
    json_object_put(meta);
    free(old);
    return status;
}

int hx_refresh_title(const char *path, const char *page, const char *title) {
    return update_viewer(path, page, title, NULL);
}

int hx_refresh_fields(const char *path, const char *page, json_object *fields) {
    return update_viewer(path, page, NULL, fields);
}

#ifdef HX_UNITY
int hx_refresh_viewers_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    if (argc != 1) {
        puts("Usage: ./run.sh viewer.refresh_viewers");
        return argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) ? 0 : 2;
    }
    char *outputs = local_root(outputs_root);
    char *root = outputs ? realpath(outputs, NULL) : NULL;
    char *sheets = outputs ? path_join(root ? root : outputs, "website/sheets") : NULL;
    char *pattern = sheets ? path_join(sheets, "*/index.html") : NULL;
    free(root);
    free(outputs);
    if (!pattern) {
        free(sheets);
        return 1;
    }
    glob_t viewers = {0};
    int found = glob(pattern, 0, NULL, &viewers);
    free(pattern);
    char *lock_path = found == 0 ? path_join(sheets, ".publish.lock") : NULL;
    free(sheets);
    int lock = lock_path ? open(lock_path, O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0666) : -1;
    free(lock_path);
    char *page = lock >= 0 && !flock(lock, LOCK_EX) ? read_page_text(viewer_template) : NULL;
    int failed = found != GLOB_NOMATCH && !page;
    for (size_t i = 0; page && i < viewers.gl_pathc; i++) {
        const char *path = viewers.gl_pathv[i];
        int changed = update_viewer(path, page, NULL, NULL);
        if (changed > 0) {
            puts(path);
        } else if (changed < 0) {
            fprintf(stderr, "HerculeX: cannot refresh %s\n", path);
            failed = 1;
        }
    }
    if (lock >= 0) {
        close(lock);
    }
    free(page);
    globfree(&viewers);
    return failed;
}
