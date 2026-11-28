#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int fd;
    const int64_t *chunk;
    const int64_t *box;
    const int64_t *first;
    const int64_t *grid;
    const int64_t *offsets;
    uint8_t *out;
    uint8_t fill;
    int64_t next;
    int64_t total;
    int failed;
} job;

static int64_t floor_div(int64_t a, int64_t b) { return a / b - (a % b < 0); }
static int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }
static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

static int read_span(int fd, uint8_t *into, int64_t length, int64_t offset) {
    while (length > 0) {
        ssize_t got = pread(fd, into, (size_t)length, (off_t)offset);
        if (got < 0 && errno == EINTR)
            continue;
        if (got <= 0)
            return -1;
        into += got;
        length -= got;
        offset += got;
    }
    return 0;
}

static int copy_chunk(job *j, int64_t index, uint8_t *scratch) {
    const int64_t *c = j->chunk, *b = j->box;
    int64_t gz = index / (j->grid[1] * j->grid[2]), gy = (index / j->grid[2]) % j->grid[1],
            gx = index % j->grid[2];
    int64_t oz = (j->first[0] + gz) * c[0], oy = (j->first[1] + gy) * c[1],
            ox = (j->first[2] + gx) * c[2];
    int64_t z0 = max64(b[0], oz), z1 = min64(b[1], oz + c[0]);
    int64_t y0 = max64(b[2], oy), y1 = min64(b[3], oy + c[1]);
    int64_t x0 = max64(b[4], ox), x1 = min64(b[5], ox + c[2]);
    if (z0 >= z1 || y0 >= y1 || x0 >= x1)
        return 0;
    int64_t sy = b[5] - b[4], sz = sy * (b[3] - b[2]);
    int64_t cy = c[2], cz = c[1] * c[2];
    int64_t row = x1 - x0;
    int64_t offset = j->offsets[index];
    if (offset < 0) {
        for (int64_t z = z0; z < z1; ++z)
            for (int64_t y = y0; y < y1; ++y)
                memset(
                    j->out + (z - b[0]) * sz + (y - b[2]) * sy + (x0 - b[4]), j->fill, (size_t)row);
        return 0;
    }
    int64_t start = (z0 - oz) * cz + (y0 - oy) * cy + (x0 - ox);
    int64_t plane_span = (y1 - y0 - 1) * cy + row;
    int64_t planes = z1 - z0, span = (planes - 1) * cz + plane_span;
    if (offset > INT64_MAX - start || offset + start > INT64_MAX - span)
        return -1;
    uint8_t *dest = j->out + (z0 - b[0]) * sz + (y0 - b[2]) * sy + x0 - b[4];
    if (cy == sy && row == sy && (planes == 1 || (cz == sz && plane_span == sz)))
        return read_span(j->fd, dest, span, offset + start);
    int64_t batch = planes;
    if (planes > 1 && cz - plane_span > 32768 && plane_span < cz / 4)
        batch = 1;
    for (int64_t z = 0; z < planes; z += batch) {
        if (read_span(j->fd, scratch, (batch - 1) * cz + plane_span, offset + start + z * cz))
            return -1;
        for (int64_t k = 0; k < batch; k++)
            for (int64_t y = 0; y < y1 - y0; y++)
                memcpy(dest + (z + k) * sz + y * sy, scratch + k * cz + y * cy, (size_t)row);
    }
    return 0;
}

static void *work(void *argument) {
    job *j = argument;
    uint8_t *scratch = malloc((size_t)(j->chunk[0] * j->chunk[1] * j->chunk[2]));
    if (!scratch) {
        __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
        return NULL;
    }
    for (;;) {
        int64_t index = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (index >= j->total)
            break;
        if (copy_chunk(j, index, scratch))
            __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
    }
    free(scratch);
    return NULL;
}

int lg_read_box(int fd, const int64_t *chunk, const int64_t *box, const int64_t *offsets,
    uint8_t fill, uint8_t *out, int32_t threads) {
    int64_t first[3], grid[3];
    for (int a = 0; a < 3; ++a) {
        first[a] = floor_div(box[2 * a], chunk[a]);
        grid[a] = floor_div(box[2 * a + 1] - 1, chunk[a]) - first[a] + 1;
    }
    job j = {fd, chunk, box, first, grid, offsets, out, fill, 0, grid[0] * grid[1] * grid[2], 0};
    if (threads > j.total)
        threads = (int32_t)j.total;
    if (threads <= 1) {
        work(&j);
        return j.failed ? -1 : 0;
    }
    pthread_t pool[64];
    if (threads > 64)
        threads = 64;
    int started = 0;
    while (started + 1 < threads && pthread_create(&pool[started], NULL, work, &j) == 0)
        ++started;
    work(&j);
    for (int i = 0; i < started; ++i)
        pthread_join(pool[i], NULL);
    return j.failed ? -1 : 0;
}

static int is_finite(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

void lg_chunks_near_points(const float *points, int64_t count, double margin, const int64_t *chunk,
    const int64_t *grid, uint8_t *marks) {
    for (int64_t i = 0; i < count; ++i) {
        const float *p = points + 3 * i;
        int64_t lo[3], hi[3];
        int inside = 1;
        for (int a = 0; a < 3; ++a) {
            double low = (double)p[a] - margin, high = (double)p[a] + margin,
                   end = (double)(grid[a] * chunk[a]);
            inside = is_finite(p[a]) && high >= 0 && low < end;
            if (!inside)
                break;
            lo[a] = low < 0 ? 0 : (int64_t)low / chunk[a];
            hi[a] = min64(grid[a] - 1, (int64_t)(high < end ? high : end) / chunk[a]);
        }
        if (!inside)
            continue;
        for (int64_t z = lo[0]; z <= hi[0]; ++z)
            for (int64_t y = lo[1]; y <= hi[1]; ++y)
                for (int64_t x = lo[2]; x <= hi[2]; ++x)
                    marks[(z * grid[1] + y) * grid[2] + x] = 1;
    }
}
