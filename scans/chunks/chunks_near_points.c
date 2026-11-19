#include <stdint.h>
#include <string.h>

static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

static int is_finite(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

void lg_chunks_near_points(const float *points, int64_t count, double margin, const int64_t *chunk, const int64_t *grid,
                           uint8_t *marks) {
    for (int64_t i = 0; i < count; ++i) {
        const float *p = points + 3 * i;
        int64_t lo[3], hi[3];
        int inside = 1;
        for (int a = 0; a < 3; ++a) {
            double low = (double)p[a] - margin, high = (double)p[a] + margin, end = (double)(grid[a] * chunk[a]);
            inside = is_finite(p[a]) && high >= 0 && low < end;
            if (!inside) break;
            lo[a] = low < 0 ? 0 : (int64_t)low / chunk[a];
            hi[a] = min64(grid[a] - 1, (int64_t)(high < end ? high : end) / chunk[a]);
        }
        if (!inside) continue;
        for (int64_t z = lo[0]; z <= hi[0]; ++z)
            for (int64_t y = lo[1]; y <= hi[1]; ++y)
                for (int64_t x = lo[2]; x <= hi[2]; ++x) marks[(z * grid[1] + y) * grid[2] + x] = 1;
    }
}
