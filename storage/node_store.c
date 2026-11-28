#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>

#ifdef HX_NODE_STORE_LOCAL
#define HX_NODE_API static
#else
#define HX_NODE_API
#endif

enum { HX_NODE_BYTES = 20, RECORD_BLOCK = 2048, PACK_LEVEL = 6 };

static int node_read_file(const char *path, void **data, size_t *size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    struct stat info;
    int status = fstat(fd, &info) || info.st_size < 0 || (uintmax_t)info.st_size > SIZE_MAX;
    *size = status ? 0 : (size_t)info.st_size;
    *data = !status && *size ? malloc(*size) : NULL;
    status = status || (*size && !*data);
    size_t done = 0;
    while (!status && done < *size) {
        ssize_t n = read(fd, (char *)*data + done, *size - done);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0) {
            status = 1;
            if (!n)
                errno = EIO;
        } else {
            done += (size_t)n;
        }
    }
    int saved = errno;
    close(fd);
    if (status) {
        free(*data);
        *data = NULL;
    }
    errno = saved;
    return status ? -1 : 0;
}

static int node_write_file(const char *path, const void *data, size_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0)
        return -1;
    size_t done = 0;
    while (done < size) {
        ssize_t n = write(fd, (const char *)data + done, size - done);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0) {
            if (!n)
                errno = EIO;
            break;
        }
        done += (size_t)n;
    }
    int status = close(fd) || done != size;
    return status ? -1 : 0;
}

static void shuffle(const char *raw, char *columns, size_t count) {
    for (size_t begin = 0; begin < count; begin += RECORD_BLOCK) {
        size_t end = begin + RECORD_BLOCK < count ? begin + RECORD_BLOCK : count;
        for (int field = 0; field < HX_NODE_BYTES; field++)
            for (size_t i = begin; i < end; i++)
                columns[(size_t)field * count + i] = raw[i * HX_NODE_BYTES + field];
    }
}

static void interleave(const char *columns, char *raw, size_t count, size_t first, size_t take) {
    for (size_t begin = 0; begin < take; begin += RECORD_BLOCK) {
        size_t end = begin + RECORD_BLOCK < take ? begin + RECORD_BLOCK : take;
        for (int field = 0; field < HX_NODE_BYTES; field++)
            for (size_t i = begin; i < end; i++)
                raw[i * HX_NODE_BYTES + field] = columns[(size_t)field * count + first + i];
    }
}

static void steps(const char *input, char *output, size_t bytes, int encode) {
    uint32_t previous = 0;
    for (size_t i = 0; i < bytes; i += 4) {
        uint32_t value;
        memcpy(&value, input + i, 4);
        uint32_t result = encode ? value - previous : previous + value;
        memcpy(output + i, &result, 4);
        previous = encode ? value : result;
    }
}

static int transform(const char *folder, int nodes) {
    const char *plain_name = nodes ? "nodes.bin" : "cells.bin";
    const char *packed_name = nodes ? "nodes.zst" : "cells.zst";
    char *plain = NULL, *packed = NULL, *part = NULL;
    if (asprintf(&plain, "%s/%s", folder, plain_name) < 0 ||
        asprintf(&packed, "%s/%s", folder, packed_name) < 0 ||
        asprintf(&part, "%s/%s.part", folder, packed_name) < 0)
        goto fail;
    void *raw = NULL;
    size_t size = 0;
    if (node_read_file(plain, &raw, &size)) {
        if (errno == ENOENT && !access(packed, R_OK)) {
            free(part);
            free(packed);
            free(plain);
            return 0;
        }
        goto fail;
    }
    size_t width = nodes ? HX_NODE_BYTES : 4;
    if (size % width) {
        errno = EINVAL;
        free(raw);
        goto fail;
    }
    void *ordered = malloc(size ? size : 1);
    size_t bound = ZSTD_compressBound(size);
    void *frame = malloc(bound ? bound : 1);
    if (!ordered || !frame) {
        free(frame);
        free(ordered);
        free(raw);
        goto fail;
    }
    if (nodes)
        shuffle(raw, ordered, size / width);
    else
        steps(raw, ordered, size, 1);
    size_t frame_size = ZSTD_compress(frame, bound, ordered, size, PACK_LEVEL);
    int status = ZSTD_isError(frame_size) || node_write_file(part, frame, frame_size) ||
        rename(part, packed);
    free(frame);
    free(ordered);
    free(raw);
    if (status)
        goto fail;
    free(part);
    free(packed);
    free(plain);
    return 0;
fail:
    if (part)
        unlink(part);
    free(part);
    free(packed);
    free(plain);
    return -1;
}

HX_NODE_API int hx_node_pack(const char *folder) {
    int dir = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir < 0) {
        return -1;
    }
    int status;
    do {
        status = flock(dir, LOCK_EX);
    } while (status && errno == EINTR);
    if (!status && (transform(folder, 1) || transform(folder, 0))) {
        status = -1;
    }
    const char *raw[] = {"nodes.bin", "cells.bin"};
    for (size_t i = 0; !status && i < sizeof(raw) / sizeof(*raw); i++) {
        if (unlinkat(dir, raw[i], 0) && errno != ENOENT) {
            status = -1;
        }
    }
    int saved = errno;
    close(dir);
    errno = saved;
    return status;
}

static int node_window(
    const char *path, size_t width, size_t first, size_t last, void **records, size_t *count) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    struct stat info;
    int ok = !fstat(fd, &info) && info.st_size >= 0 && (uintmax_t)info.st_size <= SIZE_MAX &&
        (size_t)info.st_size % width == 0;
    size_t rows = ok ? (size_t)info.st_size / width : 0;
    if (last == SIZE_MAX) {
        last = rows;
    }
    ok = ok && first <= last && last <= rows;
    size_t size = ok ? (last - first) * width : 0;
    char *data = ok ? malloc(size ? size : 1) : NULL;
    off_t offset = ok ? (off_t)first * (off_t)width : 0;
    for (size_t done = 0; data && done < size;) {
        ssize_t n = pread(fd, data + done, size - done, offset);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            free(data);
            data = NULL;
            break;
        }
        done += (size_t)n;
        offset += n;
    }
    close(fd);
    if (!data) {
        errno = ok ? EIO : EINVAL;
        return -1;
    }
    *records = data;
    *count = last - first;
    return 0;
}

static void *node_map(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return MAP_FAILED;
    }
    struct stat info;
    void *data = MAP_FAILED;
    if (!fstat(fd, &info) && info.st_size > 0 && (uintmax_t)info.st_size <= SIZE_MAX) {
        *size = (size_t)info.st_size;
        data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    close(fd);
    return data;
}

static int node_stream(
    const void *data, size_t size, size_t rows, size_t first, size_t take, char *raw) {
    ZSTD_DCtx *decoder = ZSTD_createDCtx();
    if (!decoder) {
        return -1;
    }
    char buffer[ZSTD_BLOCKSIZE_MAX];
    ZSTD_inBuffer input = {data, size, 0};
    size_t remaining = 1, offset = 0, decoded = rows * HX_NODE_BYTES;
    while (remaining) {
        size_t before = input.pos;
        ZSTD_outBuffer output = {buffer, sizeof(buffer), 0};
        remaining = ZSTD_decompressStream(decoder, &output, &input);
        if (ZSTD_isError(remaining) || output.pos > decoded - offset ||
            (remaining && !output.pos && before == input.pos)) {
            remaining = 1;
            break;
        }
        for (size_t at = 0; at < output.pos;) {
            size_t column = offset / rows, row = offset % rows;
            size_t length = rows - row < output.pos - at ? rows - row : output.pos - at;
            size_t lo = row > first ? row : first;
            size_t hi = row + length < first + take ? row + length : first + take;
            for (size_t i = lo; i < hi; i++) {
                raw[(i - first) * HX_NODE_BYTES + column] = buffer[at + i - row];
            }
            at += length;
            offset += length;
        }
    }
    ZSTD_freeDCtx(decoder);
    return remaining || input.pos != size || offset != decoded ? -1 : 0;
}

HX_NODE_API int hx_node_decode(const void *data, size_t size, int nodes, size_t first, size_t last,
    void **records, size_t *count) {
    size_t width = nodes ? HX_NODE_BYTES : 4;
    unsigned long long length = ZSTD_getFrameContentSize(data, size);
    if (length == ZSTD_CONTENTSIZE_ERROR || length == ZSTD_CONTENTSIZE_UNKNOWN ||
        length > SIZE_MAX || length % width) {
        return -1;
    }
    size_t decoded = (size_t)length, rows = decoded / width;
    if (last == SIZE_MAX) {
        last = rows;
    }
    if (first > last || last > rows) {
        return -1;
    }
    size_t take = last - first, bytes = take * width;
    void *raw = malloc(bytes ? bytes : 1);
    if (!raw) {
        return -1;
    }
    int failed = 0;
    if (nodes && decoded > ZSTD_BLOCKSIZE_MAX && take <= rows / 2) {
        failed = node_stream(data, size, rows, first, take, raw);
    } else {
        void *ordered = malloc(decoded ? decoded : 1);
        size_t got = ordered ? ZSTD_decompress(ordered, decoded, data, size) : 0;
        failed = !ordered || ZSTD_isError(got) || got != decoded;
        if (!failed && nodes)
            interleave(ordered, raw, rows, first, take);
        else if (!failed) {
            steps(ordered, ordered, decoded, 0);
            memcpy(raw, (const char *)ordered + first * width, bytes);
        }
        free(ordered);
    }
    if (failed) {
        free(raw);
        return -1;
    }
    *records = raw;
    *count = take;
    return 0;
}

static int unpack(
    const char *folder, int nodes, size_t first, size_t last, void **records, size_t *count) {
    char *plain = NULL, *packed = NULL;
    if (asprintf(&plain, "%s/%s", folder, nodes ? "nodes.bin" : "cells.bin") < 0 ||
        asprintf(&packed, "%s/%s", folder, nodes ? "nodes.zst" : "cells.zst") < 0)
        goto fail;
    size_t width = nodes ? HX_NODE_BYTES : 4;
    if (!node_window(plain, width, first, last, records, count)) {
        free(packed);
        free(plain);
        return 0;
    }
    size_t size = 0;
    void *data = errno == ENOENT ? node_map(packed, &size) : MAP_FAILED;
    if (data == MAP_FAILED) {
        goto fail;
    }
    int failed = hx_node_decode(data, size, nodes, first, last, records, count);
    munmap(data, size);
    if (failed) {
        errno = EINVAL;
    }
    free(packed);
    free(plain);
    return failed;
fail:
    free(packed);
    free(plain);
    return -1;
}

HX_NODE_API int hx_node_count(const char *folder, size_t *count) {
    char *path = NULL;
    if (asprintf(&path, "%s/nodes.bin", folder) < 0) {
        return -1;
    }
    struct stat info;
    int found = !stat(path, &info);
    free(path);
    unsigned long long length;
    if (found) {
        length = info.st_size;
    } else {
        if (errno != ENOENT || asprintf(&path, "%s/nodes.zst", folder) < 0) {
            return -1;
        }
        size_t size = 0;
        void *frame = node_map(path, &size);
        free(path);
        if (frame == MAP_FAILED) {
            return -1;
        }
        length = ZSTD_getFrameContentSize(frame, size);
        munmap(frame, size);
    }
    if (length > SIZE_MAX || length == ZSTD_CONTENTSIZE_ERROR ||
        length == ZSTD_CONTENTSIZE_UNKNOWN || length % HX_NODE_BYTES) {
        errno = EINVAL;
        return -1;
    }
    *count = (size_t)length / HX_NODE_BYTES;
    return 0;
}

HX_NODE_API int hx_node_read(const char *folder, void **nodes, size_t *count) {
    return unpack(folder, 1, 0, SIZE_MAX, nodes, count);
}

HX_NODE_API int hx_node_read_range(
    const char *folder, size_t first, size_t last, void **nodes, size_t *count) {
    return unpack(folder, 1, first, last, nodes, count);
}

HX_NODE_API int hx_cell_read(const char *folder, int32_t **cells, size_t *count) {
    return unpack(folder, 0, 0, SIZE_MAX, (void **)cells, count);
}

HX_NODE_API void hx_node_free(void *records) { free(records); }

#undef HX_NODE_API
