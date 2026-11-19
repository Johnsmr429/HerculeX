#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint8_t paeth(int a, int b, int c) {
    int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return (uint8_t)((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c));
}

int hx_png_unfilter(const uint8_t *in, uint8_t *out, int width, int height) {
    const uint8_t *row = in;
    for (int y = 0; y < height; y++, row += width + 1) {
        const uint8_t *src = row + 1;
        uint8_t *dst = out + (size_t)y * width;
        const uint8_t *up = y ? dst - width : NULL;
        switch (row[0]) {
        case 0: memcpy(dst, src, width); break;
        case 1: {
            uint8_t left = 0;
            for (int x = 0; x < width; x++) dst[x] = left = (uint8_t)(src[x] + left);
            break;
        }
        case 2:
            if (up)
                for (int x = 0; x < width; x++) dst[x] = (uint8_t)(src[x] + up[x]);
            else memcpy(dst, src, width);
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
                for (int x = 0; x < width; x++) dst[x] = left = (uint8_t)(src[x] + left);
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
