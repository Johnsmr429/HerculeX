#include "studio_cuda.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const char *p;
    const char *end;
} Parser;

static void skip_ws(Parser *s) {
    while (s->p < s->end && isspace((unsigned char)*s->p))
        s->p++;
}

static int parse_str(Parser *s, char *out, size_t cap) {
    skip_ws(s);
    if (s->p >= s->end || *s->p != '"')
        return -1;
    s->p++;
    size_t n = 0;
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\' && s->p < s->end)
            c = *s->p++;
        if (out && n + 1 < cap)
            out[n++] = c;
    }
    if (s->p >= s->end || *s->p != '"')
        return -1;
    s->p++;
    if (out && cap > 0)
        out[n] = '\0';
    return 0;
}

static int parse_int(Parser *s, int64_t *val) {
    skip_ws(s);
    if (s->p >= s->end)
        return -1;
    int neg = 0;
    if (*s->p == '-') {
        neg = 1;
        s->p++;
    }
    if (s->p >= s->end || !isdigit((unsigned char)*s->p))
        return -1;
    int64_t v = 0;
    while (s->p < s->end && isdigit((unsigned char)*s->p))
        v = v * 10 + (*s->p++ - '0');
    *val = neg ? -v : v;
    return 0;
}

static int skip_value(Parser *s);

static int skip_object(Parser *s) {
    skip_ws(s);
    if (s->p >= s->end || *s->p != '{')
        return -1;
    s->p++;
    skip_ws(s);
    if (s->p < s->end && *s->p == '}') {
        s->p++;
        return 0;
    }
    for (;;) {
        if (parse_str(s, NULL, 0))
            return -1;
        skip_ws(s);
        if (s->p >= s->end || *s->p != ':')
            return -1;
        s->p++;
        if (skip_value(s))
            return -1;
        skip_ws(s);
        if (s->p < s->end && *s->p == ',') {
            s->p++;
            continue;
        }
        if (s->p < s->end && *s->p == '}') {
            s->p++;
            return 0;
        }
        return -1;
    }
}

static int skip_array(Parser *s) {
    skip_ws(s);
    if (s->p >= s->end || *s->p != '[')
        return -1;
    s->p++;
    skip_ws(s);
    if (s->p < s->end && *s->p == ']') {
        s->p++;
        return 0;
    }
    for (;;) {
        if (skip_value(s))
            return -1;
        skip_ws(s);
        if (s->p < s->end && *s->p == ',') {
            s->p++;
            continue;
        }
        if (s->p < s->end && *s->p == ']') {
            s->p++;
            return 0;
        }
        return -1;
    }
}

static int skip_value(Parser *s) {
    skip_ws(s);
    if (s->p >= s->end)
        return -1;
    if (*s->p == '"')
        return parse_str(s, NULL, 0);
    if (*s->p == '{')
        return skip_object(s);
    if (*s->p == '[')
        return skip_array(s);
    while (s->p < s->end && *s->p != ',' && *s->p != '}' && *s->p != ']' &&
        !isspace((unsigned char)*s->p))
        s->p++;
    return 0;
}

static int parse_int_array(Parser *s, int64_t *vals, int max_n, int *out_n) {
    skip_ws(s);
    if (s->p >= s->end || *s->p != '[')
        return -1;
    s->p++;
    int n = 0;
    skip_ws(s);
    if (s->p < s->end && *s->p == ']') {
        s->p++;
        *out_n = 0;
        return 0;
    }
    for (;;) {
        int64_t v = 0;
        if (parse_int(s, &v))
            return -1;
        if (n >= max_n)
            return -1;
        vals[n++] = v;
        skip_ws(s);
        if (s->p < s->end && *s->p == ',') {
            s->p++;
            continue;
        }
        if (s->p < s->end && *s->p == ']') {
            s->p++;
            *out_n = n;
            return 0;
        }
        return -1;
    }
}

int studio_weights_open(const char *path, StudioWeights *out) {
    memset(out, 0, sizeof(*out));
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        close(fd);
        return -1;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED)
        return -1;
    uint64_t header_len = 0;
    memcpy(&header_len, map, 8);
    if (8 + header_len > (uint64_t)st.st_size) {
        munmap(map, (size_t)st.st_size);
        return -1;
    }
    const uint8_t *base = (const uint8_t *)map + 8 + header_len;
    size_t data_bytes = (size_t)st.st_size - (size_t)(8 + header_len);
    Parser ps = {(const char *)map + 8, (const char *)map + 8 + header_len};
    skip_ws(&ps);
    if (ps.p >= ps.end || *ps.p != '{') {
        munmap(map, (size_t)st.st_size);
        return -1;
    }
    ps.p++;
    int cap = 256;
    StudioTensor *list = (StudioTensor *)calloc((size_t)cap, sizeof(StudioTensor));
    int count = 0;
    skip_ws(&ps);
    while (ps.p < ps.end && *ps.p != '}') {
        char key[128];
        if (parse_str(&ps, key, sizeof(key)))
            goto fail;
        skip_ws(&ps);
        if (ps.p >= ps.end || *ps.p != ':')
            goto fail;
        ps.p++;
        if (!strcmp(key, "__metadata__")) {
            if (skip_value(&ps))
                goto fail;
        } else {
            skip_ws(&ps);
            if (ps.p >= ps.end || *ps.p != '{')
                goto fail;
            ps.p++;
            StudioTensor t;
            memset(&t, 0, sizeof(t));
            snprintf(t.name, sizeof(t.name), "%s", key);
            char dtype[16] = "";
            int64_t offsets[2] = {0, 0};
            int noff = 0;
            skip_ws(&ps);
            while (ps.p < ps.end && *ps.p != '}') {
                char field[32];
                if (parse_str(&ps, field, sizeof(field)))
                    goto fail;
                skip_ws(&ps);
                if (ps.p >= ps.end || *ps.p != ':')
                    goto fail;
                ps.p++;
                if (!strcmp(field, "dtype")) {
                    if (parse_str(&ps, dtype, sizeof(dtype)))
                        goto fail;
                } else if (!strcmp(field, "shape")) {
                    if (parse_int_array(&ps, t.shape, 6, &t.ndim))
                        goto fail;
                } else if (!strcmp(field, "data_offsets")) {
                    if (parse_int_array(&ps, offsets, 2, &noff) || noff != 2)
                        goto fail;
                } else {
                    if (skip_value(&ps))
                        goto fail;
                }
                skip_ws(&ps);
                if (ps.p < ps.end && *ps.p == ',') {
                    ps.p++;
                    skip_ws(&ps);
                }
            }
            if (ps.p >= ps.end || *ps.p != '}')
                goto fail;
            ps.p++;
            if (strcmp(dtype, "F32") != 0 || offsets[0] < 0 || offsets[1] < offsets[0] ||
                (size_t)offsets[1] > data_bytes)
                goto fail;
            int64_t elems = 1;
            for (int d = 0; d < t.ndim; d++)
                elems *= t.shape[d];
            if (elems * 4 != offsets[1] - offsets[0])
                goto fail;
            t.elements = elems;
            t.data = (const float *)(base + offsets[0]);
            if (count == cap) {
                cap *= 2;
                StudioTensor *grown =
                    (StudioTensor *)realloc(list, (size_t)cap * sizeof(StudioTensor));
                if (!grown)
                    goto fail;
                list = grown;
            }
            list[count++] = t;
        }
        skip_ws(&ps);
        if (ps.p < ps.end && *ps.p == ',') {
            ps.p++;
            skip_ws(&ps);
        }
    }
    out->map = map;
    out->map_bytes = (size_t)st.st_size;
    out->tensors = list;
    out->count = count;
    return 0;
fail:
    free(list);
    munmap(map, (size_t)st.st_size);
    return -1;
}

void studio_weights_close(StudioWeights *w) {
    if (w->map)
        munmap(w->map, w->map_bytes);
    free(w->tensors);
    memset(w, 0, sizeof(*w));
}

const StudioTensor *studio_weights_find(const StudioWeights *w, const char *name) {
    for (int i = 0; i < w->count; i++) {
        if (!strcmp(w->tensors[i].name, name))
            return w->tensors + i;
    }
    fprintf(stderr, "studio_cuda: missing tensor %s\n", name);
    exit(1);
}
