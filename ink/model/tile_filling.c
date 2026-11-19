#include <stdint.h>

static inline int input_at(int position, int shift, int length, int flip) {
    int from = position - shift;
    if (from < 0) from += length;
    return flip ? length - 1 - from : from;
}

static inline int output_at(int position, int shift, int length, int flip) {
    int to = (flip ? length - 1 - position : position) + shift;
    if (to >= length) to -= length;
    return to;
}

int64_t hx_fill_pending(int32_t *source, const uint8_t *keep, int32_t *rows, int32_t *cols, int height, int width,
                        int dy, int dx, int flip_y, int flip_x, int64_t count) {
    int64_t left = 0;
    for (int64_t k = 0; k < count; k++) {
        int row = rows[k], col = cols[k];
        int from = input_at(row, dy, height, flip_y) * width + input_at(col, dx, width, flip_x);
        if (keep[from]) {
            source[row * width + col] = from;
        } else {
            rows[left] = row;
            cols[left++] = col;
        }
    }
    return left;
}

// When the sheet is sparse, visit its covered columns and apply the inverse mapping.
int64_t hx_fill_valid(int32_t *source, const int32_t *rows, const int32_t *cols, int height, int width, int dy, int dx,
                      int flip_y, int flip_x, int64_t count) {
    int64_t added = 0;
    for (int64_t k = 0; k < count; k++) {
        int row = rows[k], col = cols[k];
        int dest = output_at(row, dy, height, flip_y) * width + output_at(col, dx, width, flip_x);
        if (source[dest] < 0) {
            source[dest] = row * width + col;
            added++;
        }
    }
    return added;
}

void hx_gather_filled(const uint8_t *raw, const int32_t *source, int64_t layers, int64_t columns, uint8_t *out) {
    for (int64_t z = 0; z < layers; ++z) {
        const uint8_t *layer = raw + z * columns;
        uint8_t *target = out + z * columns;
        for (int64_t i = 0; i < columns; ++i) target[i] = layer[source[i]];
    }
}
