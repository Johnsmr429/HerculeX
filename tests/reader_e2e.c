#include "base.h"
#include "storage/storage.h"
#include "scans/scans.h"
#include "sampling/sampling.h"
#include "ink/ink.h"

int main(void) {
    roots_init();
    Arena a = arena(8 * GB);
    const char *weights = fmt(&a, "%s/weights/ink/reader-v3/tile-reader.pth", LIBRARY);
    if (!exists(weights)) {
        puts("The Reader v3 weights are unavailable.");
        return 77;
    }
    int shapes[2][2] = {{700, 900}, {300, 260}};
    Reader *r = reader_open(&a, weights, 700 * 900);
    u64 state = 0x9e3779b97f4a7c15;
    for (int k = 0; k < 2; k++) {
        int h = shapes[k][0], w = shapes[k][1];
        i64 n = (i64)h * w;
        u8 *stack = push(&a, DEPTH * n);
        for (int l = 0; l < DEPTH; l++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    double v = 110 + 35 * sin(x / 9.0 + l * 0.3) * cos(y / 13.0) + (state >> 59) -
                        16 + (abs(y - h / 2) < 6 && l > 5 && l < 12 ? 60 : 0);
                    stack[l * n + (i64)y * w + x] = (u8)fmin(fmax(v, 0), 255);
                }
            }
        }
        f32 *first = push(&a, n * sizeof(f32)), *second = push(&a, n * sizeof(f32));
        fix_stem(r, stack, h, w);
        ink_read(r, stack, h, w, first);
        f64 started = now();
        fix_stem(r, stack, h, w);
        ink_read(r, stack, h, w, second);
        f64 sum = 0, squares = 0;
        for (i64 i = 0; i < n; i++) {
            need(isfinite(first[i]) && first[i] >= 0 && first[i] <= 1, "%dx%d: value %g at %lld", h,
                w, first[i], (long long)i);
            sum += first[i];
            squares += (f64)first[i] * first[i];
        }
        f64 mean = sum / n, spread = sqrt(fmax(squares / n - mean * mean, 0));
        printf("%dx%d: %.3f s, mean %.4f, spread %.4f\n", h, w, now() - started, mean, spread);
        need(!memcmp(first, second, n * sizeof(f32)), "%dx%d: two reads differ", h, w);
        need(spread > 1e-4, "%dx%d: the read is flat", h, w);
    }
    puts("reader e2e passed");
    return 0;
}
