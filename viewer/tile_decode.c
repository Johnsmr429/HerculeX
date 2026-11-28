#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint8_t paeth(int a, int b, int c) {
    int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return (uint8_t)((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c));
}

int hx_png_unfilter(const uint8_t *in, uint8_t *out, int width, int height) {
    if (!in || !out || width <= 0 || height <= 0)
        return -1;
    const uint8_t *row = in;
    for (int y = 0; y < height; y++, row += width + 1) {
        const uint8_t *src = row + 1;
        uint8_t *dst = out + (size_t)y * width;
        const uint8_t *up = y ? dst - width : NULL;
        switch (row[0]) {
        case 0: memcpy(dst, src, width); break;
        case 1: {
            uint8_t left = 0;
            for (int x = 0; x < width; x++)
                dst[x] = left = (uint8_t)(src[x] + left);
            break;
        }
        case 2:
            if (up)
                for (int x = 0; x < width; x++)
                    dst[x] = (uint8_t)(src[x] + up[x]);
            else
                memcpy(dst, src, width);
            break;
        case 3: {
            int left = 0;
            for (int x = 0; x < width; x++) {
                int above = up ? up[x] : 0;
                dst[x] = (uint8_t)(src[x] + ((left + above) >> 1));
                left = dst[x];
            }
            break;
        }
        case 4: {
            if (!up) {
                uint8_t left = 0;
                for (int x = 0; x < width; x++)
                    dst[x] = left = (uint8_t)(src[x] + left);
                break;
            }
            int left = 0, corner = 0;
            for (int x = 0; x < width; x++) {
                int above = up[x];
                dst[x] = (uint8_t)(src[x] + paeth(left, above, corner));
                left = dst[x];
                corner = above;
            }
            break;
        }
        default: return -1;
        }
    }
    return 0;
}

void hx_part_mosaics(uint8_t *covered, uint8_t *pixels, const int32_t *owner, int h, int w,
    int width, int32_t *expires) {
    int before = (width - 1) / 2, after = width / 2, next = 0;
    for (int y = 0; y < h; y++) {
        while (next < h && next <= y + before) {
            const int32_t *row = owner + (size_t)next * w;
            const int32_t *above = next ? row - w : row;
            int end = 0;
            for (int x = 0; x < w; x++) {
                int a = row[x], b = above[x], c = x ? row[x - 1] : a;
                if (a <= 0 || ((b <= 0 || a == b) && (c <= 0 || a == c)))
                    continue;
                int start = x - before > end ? x - before : end;
                end = x + after + 1 < w ? x + after + 1 : w;
                for (int col = start; col < end; col++)
                    expires[col] = next + after + 1;
            }
            next++;
        }
        for (int x = 0; x < w; x++) {
            if (expires[x] > y) {
                covered[(size_t)y * w + x] = 0;
                if (pixels)
                    pixels[(size_t)y * w + x] = 0;
            }
        }
    }
}

void hx_paint_rows(uint8_t *pixels, uint8_t *covered, int32_t *owner, int stride,
    const uint8_t *image, const uint8_t *seen, const int64_t *layout, int mosaic) {
    int h = layout[0], w = layout[1];
    for (int y = 0; y < h; y++) {
        const uint8_t *src = image + y * layout[2], *mask = seen + y * layout[4];
        size_t out = (size_t)y * stride;
        for (int x = 0; x < w; x++) {
            size_t i = out + x;
            if (!covered[i] && mask[x * layout[5]]) {
                pixels[i] = src[x * layout[3]];
                covered[i] = 1;
                if (owner)
                    owner[i] = mosaic;
            }
        }
    }
}

#define SIDE 512
#define FILTER_BITS 22
#define FILTER_ROUND (1 << (FILTER_BITS - 1))

static int filter_weights[4][12];
static pthread_once_t filter_once = PTHREAD_ONCE_INIT;

static void half_weights(void) {
    double pi = acos(-1.0);
    for (int edge = 0; edge < 4; edge++) {
        int start = edge == 3 ? 1 : 0;
        int count = edge == 3 ? 12 : 7 + 2 * edge;
        double taps[12], total = 0;
        for (int k = 0; k < count; k++) {
            double t = (start + k - 2 * edge - 0.5) * (pi * 0.5);
            taps[k] = (sin(t) / t) * (sin(t / 3) / (t / 3));
            total += taps[k];
        }
        for (int k = 0; k < count; k++) {
            filter_weights[edge][k] = (int)lround(taps[k] / total * (1 << FILTER_BITS));
        }
    }
}

static uint8_t filtered_pixel(int value) {
    if (value <= 0) {
        return 0;
    }
    return value >> FILTER_BITS < 255 ? value >> FILTER_BITS : 255;
}

static void filter_row(const uint8_t *input, uint8_t *out) {
    int *full = filter_weights[3];
    for (int x = 3; x < SIDE - 3; x++) {
        const uint8_t *p = input + 2 * x - 5;
        int value = FILTER_ROUND;
        for (int k = 0; k < 6; k++) {
            value += full[k] * (p[k] + p[11 - k]);
        }
        out[x] = filtered_pixel(value);
    }
    for (int x = 0; x < 3; x++) {
        int left = FILTER_ROUND, right = FILTER_ROUND;
        for (int k = 0; k < 7 + 2 * x; k++) {
            left += filter_weights[x][k] * input[k];
            right += filter_weights[x][k] * input[2 * SIDE - 1 - k];
        }
        out[x] = filtered_pixel(left);
        out[SIDE - 1 - x] = filtered_pixel(right);
    }
}

void hx_half_tile(const uint8_t **child, const int64_t *stride, uint8_t *out) {
    pthread_once(&filter_once, half_weights);
    uint8_t rows[12][SIDE], input[2 * SIDE];
    int next = 0;
    for (int y = 0; y < SIDE; y++) {
        int end = 2 * y + 7 < 2 * SIDE ? 2 * y + 7 : 2 * SIDE;
        while (next < end) {
            int half = 2 * (next / SIDE);
            if (!child[half] && !child[half + 1]) {
                memset(rows[next % 12], 0, SIDE);
                next++;
                continue;
            }
            for (int i = 0; i < 2; i++) {
                int id = half + i;
                if (child[id]) {
                    memcpy(input + i * SIDE, child[id] + (next % SIDE) * stride[id], SIDE);
                } else {
                    memset(input + i * SIDE, 0, SIDE);
                }
            }
            filter_row(input, rows[next % 12]);
            next++;
        }
        int edge = y < SIDE - 1 - y ? y : SIDE - 1 - y;
        if (edge > 3) {
            edge = 3;
        }
        int count = edge == 3 ? 12 : 7 + 2 * edge;
        int first = y >= 3 ? 2 * y - 5 : 0;
        const uint8_t *line[12];
        for (int k = 0; k < count; k++) {
            int index = y >= SIDE - 3 ? 2 * SIDE - 1 - k : first + k;
            line[k] = rows[index % 12];
        }
        int *weights = filter_weights[edge];
        uint8_t *dest = out + y * SIDE;
        if (edge == 3) {
            for (int x = 0; x < SIDE; x++) {
                int sum = FILTER_ROUND;
                for (int k = 0; k < 6; k++) {
                    sum += weights[k] * (line[k][x] + line[11 - k][x]);
                }
                dest[x] = filtered_pixel(sum);
            }
        } else {
            for (int x = 0; x < SIDE; x++) {
                int sum = FILTER_ROUND;
                for (int k = 0; k < count; k++) {
                    sum += weights[k] * line[k][x];
                }
                dest[x] = filtered_pixel(sum);
            }
        }
    }
}
