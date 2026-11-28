#ifndef FETCH_H
#define FETCH_H

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    uint8_t *data;
    void *map;
    size_t map_bytes;
    int64_t shape[3];
} Volume;

typedef struct {
    int64_t lo[3], hi[3];
} Box;

static int box_parse(const char *text, Box *box) {
    long long v[6];
    if (sscanf(text, "%lld:%lld,%lld:%lld,%lld:%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) !=
        6)
        return -1;
    for (int a = 0; a < 3; a++) {
        box->lo[a] = v[2 * a];
        box->hi[a] = v[2 * a + 1];
        if (box->hi[a] <= box->lo[a])
            return -1;
    }
    return 0;
}

static int npy_open(const char *path, Volume *v) {
    memset(v, 0, sizeof(*v));
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    struct stat st;
    if (fstat(fd, &st)) {
        close(fd);
        return -1;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED)
        return -1;
    const uint8_t *bytes = (const uint8_t *)map;
    if (st.st_size < 16 || memcmp(bytes, "\x93NUMPY", 6)) {
        munmap(map, (size_t)st.st_size);
        return -1;
    }
    size_t header = bytes[6] == 1
        ? (size_t)(bytes[8] | (bytes[9] << 8))
        : (size_t)(bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | ((size_t)bytes[11] << 24));
    size_t start = bytes[6] == 1 ? 10 : 12;
    char *text = (char *)calloc(header + 1, 1);
    if (!text) {
        munmap(map, (size_t)st.st_size);
        return -1;
    }
    memcpy(text, bytes + start, header);
    const char *shape = strstr(text, "'shape'");
    int ok = strstr(text, "|u1") && strstr(text, "'fortran_order': False") && shape &&
        sscanf(strchr(shape, '(') + 1, "%lld, %lld, %lld", (long long *)&v->shape[0],
            (long long *)&v->shape[1], (long long *)&v->shape[2]) == 3;
    free(text);
    size_t need = start + header + (size_t)(v->shape[0] * v->shape[1] * v->shape[2]);
    if (!ok || (size_t)st.st_size < need) {
        munmap(map, (size_t)st.st_size);
        return -1;
    }
    v->map = map;
    v->map_bytes = (size_t)st.st_size;
    v->data = (uint8_t *)map + start + header;
    return 0;
}

static void npy_close(Volume *v) {
    if (v->map)
        munmap(v->map, v->map_bytes);
    memset(v, 0, sizeof(*v));
}

static int npy_header(FILE *file, const char *descr, const int64_t *shape, int dims) {
    char text[256], sizes[160] = "";
    int used = 0;
    for (int i = 0; i < dims; i++)
        used += snprintf(
            sizes + used, sizeof(sizes) - (size_t)used, i ? ", %lld" : "%lld", (long long)shape[i]);
    if (dims == 1)
        snprintf(sizes + used, sizeof(sizes) - (size_t)used, ",");
    int n = snprintf(text, sizeof(text), "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }",
        descr, sizes);
    int total = 10 + n + 1;
    int pad = (64 - total % 64) % 64;
    unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    int length = n + pad + 1;
    magic[8] = (unsigned char)(length & 0xff);
    magic[9] = (unsigned char)(length >> 8);
    if (fwrite(magic, 1, 10, file) != 10 || fwrite(text, 1, (size_t)n, file) != (size_t)n)
        return -1;
    for (int i = 0; i < pad; i++)
        fputc(' ', file);
    fputc('\n', file);
    return 0;
}

#endif
