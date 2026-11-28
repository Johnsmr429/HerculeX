#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define CHUNK 1024

static long reflected(long i, long n) {
    long m = i % (2 * n);
    if (m < 0)
        m += 2 * n;
    return m < n ? m : 2 * n - 1 - m;
}

static void across_rows(const float *in, float *out, long outer, long length, long inner,
    const double *weights, int radius, int plain) {
    double sum[CHUNK];
    for (long o = 0; o < outer; o++) {
        const float *plane = in + o * length * inner;
        float *result = out + o * length * inner;
        for (long start = 0; start < inner; start += CHUNK) {
            long n = inner - start < CHUNK ? inner - start : CHUNK;
            for (long l = 0; l < length; l++) {
                const float *centre = plane + l * inner + start;
                for (long i = 0; i < n; i++)
                    sum[i] = (double)centre[i] * weights[radius];
                for (int j = radius, tap = 0; j > 0; j--, tap++) {
                    const float *a = plane + reflected(l - j, length) * inner + start,
                                *b = plane + reflected(l + j, length) * inner + start;
                    double w = weights[tap];
                    if (tap < plain)
                        for (long i = 0; i < n; i++)
                            sum[i] = sum[i] + ((double)a[i] + (double)b[i]) * w;
                    else
                        for (long i = 0; i < n; i++)
                            sum[i] = __builtin_fma((double)a[i] + (double)b[i], w, sum[i]);
                }
                float *row = result + l * inner + start;
                for (long i = 0; i < n; i++)
                    row[i] = (float)sum[i];
            }
        }
    }
}

static int along_lines(const float *in, float *out, long lines, long length, const double *weights,
    int radius, int plain) {
    double *line = malloc((size_t)(length + 2 * radius) * sizeof(double)),
           *sum = malloc((size_t)length * sizeof(double));
    if (!line || !sum) {
        free(line);
        free(sum);
        return -1;
    }
    for (long n = 0; n < lines; n++) {
        const float *source = in + n * length;
        for (long i = 0; i < radius; i++)
            line[i] = source[reflected(i - radius, length)];
        for (long i = 0; i < length; i++)
            line[radius + i] = source[i];
        for (long i = 0; i < radius; i++)
            line[radius + length + i] = source[reflected(length + i, length)];
        const double *centre = line + radius;
        for (long i = 0; i < length; i++)
            sum[i] = centre[i] * weights[radius];
        for (int j = radius, tap = 0; j > 0; j--, tap++) {
            double w = weights[tap];
            if (tap < plain)
                for (long i = 0; i < length; i++)
                    sum[i] = sum[i] + (centre[i - j] + centre[i + j]) * w;
            else
                for (long i = 0; i < length; i++)
                    sum[i] = __builtin_fma(centre[i - j] + centre[i + j], w, sum[i]);
        }
        float *result = out + n * length;
        for (long i = 0; i < length; i++)
            result[i] = (float)sum[i];
    }
    free(line);
    free(sum);
    return 0;
}

static int gaussian_axis(const float *in, float *out, const long *dims, int rank, int axis,
    const double *weights, int radius, int plain) {
    long outer = 1, inner = 1;
    for (int k = 0; k < axis; k++)
        outer *= dims[k];
    for (int k = axis + 1; k < rank; k++)
        inner *= dims[k];
    if (inner == 1)
        return along_lines(in, out, outer, dims[axis], weights, radius, plain);
    across_rows(in, out, outer, dims[axis], inner, weights, radius, plain);
    return 0;
}
