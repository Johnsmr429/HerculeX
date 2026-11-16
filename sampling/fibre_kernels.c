#include <math.h>
#include <stdint.h>
#include <stdlib.h>

// The numeric core of the fibre test in the order scipy and numpy compute it, so every value comes out the same to the bit.

#define CHUNK 1024

static long reflected(long i, long n) {
    long m = i % (2 * n);
    if (m < 0) m += 2 * n;
    return m < n ? m : 2 * n - 1 - m;
}

// scipy's correlate1d, symmetric kernel, reflected edges: the centre, then the pairs of taps from the outermost in, in double.
static void across_rows(const float *in, float *out, long outer, long length, long inner, const double *weights, int radius, int plain) {
    double sum[CHUNK];
    for (long o = 0; o < outer; o++) {
        const float *plane = in + o * length * inner;
        float *result = out + o * length * inner;
        for (long start = 0; start < inner; start += CHUNK) {
            long n = inner - start < CHUNK ? inner - start : CHUNK;
            for (long l = 0; l < length; l++) {
                const float *centre = plane + l * inner + start;
                for (long i = 0; i < n; i++) sum[i] = (double)centre[i] * weights[radius];
                for (int j = radius, tap = 0; j > 0; j--, tap++) {
                    const float *a = plane + reflected(l - j, length) * inner + start, *b = plane + reflected(l + j, length) * inner + start;
                    double w = weights[tap];
                    if (tap < plain) for (long i = 0; i < n; i++) sum[i] = sum[i] + ((double)a[i] + (double)b[i]) * w;
                    else for (long i = 0; i < n; i++) sum[i] = __builtin_fma((double)a[i] + (double)b[i], w, sum[i]);
                }
                float *row = result + l * inner + start;
                for (long i = 0; i < n; i++) row[i] = (float)sum[i];
            }
        }
    }
}

static void along_lines(const float *in, float *out, long lines, long length, const double *weights, int radius, int plain) {
    double *line = malloc((size_t)(length + 2 * radius) * sizeof(double)), *sum = malloc((size_t)length * sizeof(double));
    for (long n = 0; n < lines; n++) {
        const float *source = in + n * length;
        for (long i = 0; i < radius; i++) line[i] = source[reflected(i - radius, length)];
        for (long i = 0; i < length; i++) line[radius + i] = source[i];
        for (long i = 0; i < radius; i++) line[radius + length + i] = source[reflected(length + i, length)];
        const double *centre = line + radius;
        for (long i = 0; i < length; i++) sum[i] = centre[i] * weights[radius];
        for (int j = radius, tap = 0; j > 0; j--, tap++) {
            double w = weights[tap];
            if (tap < plain) for (long i = 0; i < length; i++) sum[i] = sum[i] + (centre[i - j] + centre[i + j]) * w;
            else for (long i = 0; i < length; i++) sum[i] = __builtin_fma(centre[i - j] + centre[i + j], w, sum[i]);
        }
        float *result = out + n * length;
        for (long i = 0; i < length; i++) result[i] = (float)sum[i];
    }
    free(line);
    free(sum);
}

// One axis of a C-ordered array; out may not be in.
void hx_gaussian_axis(const float *in, float *out, const long *dims, int rank, int axis, const double *weights, int radius, int plain) {
    long outer = 1, inner = 1;
    for (int k = 0; k < axis; k++) outer *= dims[k];
    for (int k = axis + 1; k < rank; k++) inner *= dims[k];
    if (inner == 1) along_lines(in, out, outer, dims[axis], weights, radius, plain);
    else across_rows(in, out, outer, dims[axis], inner, weights, radius, plain);
}

// Per column of smoothed layers, the sub-layer depth of the steepest fall; layer_stride may be negative.
void hx_face_depth(const float *smoothed, long layer_stride, long layers, long pixels, long start, double *depth) {
    long falls = layers - 1;
    float *least = malloc((size_t)pixels * sizeof(float));
    int32_t *steepest = calloc((size_t)pixels, sizeof(int32_t));
    for (long i = 0; i < pixels; i++) least[i] = smoothed[layer_stride + i] - smoothed[i];
    for (long z = 1; z < falls; z++) {
        const float *a = smoothed + z * layer_stride, *b = a + layer_stride;
        for (long i = 0; i < pixels; i++) {
            float fall = b[i] - a[i];
            if (fall < least[i]) { least[i] = fall; steepest[i] = (int32_t)z; }
        }
    }
    for (long i = 0; i < pixels; i++) {
        long s = steepest[i];
        s = s < 1 ? 1 : s > falls - 2 ? falls - 2 : s;
        const float *at_layer = smoothed + s * layer_stride + i;
        float before = at_layer[0] - at_layer[-layer_stride], at = at_layer[layer_stride] - at_layer[0], after = at_layer[2 * layer_stride] - at_layer[layer_stride];
        float curve = (before - 2 * at) + after;
        float offset = fabsf(curve) > 1e-3f ? (0.5f * (before - after)) / curve : 0.0f;
        offset = offset < -1 ? -1 : offset > 1 ? 1 : offset;
        depth[i] = ((double)(start + s) + 0.5) + (double)offset;
    }
    free(least);
    free(steepest);
}

// A layer at a depth per column, linear between layers, of a block with any strides in bytes.
void hx_layer_at(const uint8_t *block, long layer_stride, long row_stride, long column_stride, long layers, long height, long width, const double *depth, double offset, float *out) {
    long last = layers - 1;
    for (long y = 0; y < height; y++) {
        const uint8_t *row = block + y * row_stride;
        for (long x = 0; x < width; x++) {
            double d = depth[y * width + x] + offset, floored = floor(d);
            long whole = (long)floored, low = whole < 0 ? 0 : whole > last ? last : whole, high = whole + 1 < 0 ? 0 : whole + 1 > last ? last : whole + 1;
            float part = (float)(d - floored);
            float lower = row[low * layer_stride + x * column_stride], upper = row[high * layer_stride + x * column_stride];
            float a = lower * (1.0f - part), b = upper * part;
            out[y * width + x] = a + b;
        }
    }
}
