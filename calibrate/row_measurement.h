#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

static double row_coverage(const uint8_t *good, int height, int width, int rows, int columns) {
    int64_t seen = 0;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            seen += good[(int64_t)y * rows + (int64_t)x * columns];
    return seen / ((double)height * width);
}

static void row_projection(const double *values, const uint8_t *good, int height, int width,
    int rows, int columns, double pitch, double tilt, const double *hann, double *profile,
    double *hits) {
    memset(profile, 0, (size_t)height * sizeof(*profile));
    memset(hits, 0, (size_t)height * sizeof(*hits));
    for (int x = 0; x < width; x++) {
        double delta = nearbyint(tilt * ((x - width / 2.0) * pitch) / pitch);
        int shift = fabs(delta) < height ? (int)delta : (int)fmod(delta, height);
        if (shift < 0)
            shift += height;
        int64_t base = (int64_t)x * columns;
        for (int y = 0; y < height; y++) {
            int64_t at = base + (int64_t)shift * rows;
            profile[y] += values[at];
            hits[y] += good[at];
            if (++shift == height)
                shift = 0;
        }
    }
    for (int y = 0; y < height; y++)
        profile[y] = profile[y] / sqrt(fmax(hits[y], 1)) * hann[y];
}
