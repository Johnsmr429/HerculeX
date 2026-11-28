#define _POSIX_C_SOURCE 200809L
#include "studio_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int read_exact(FILE *f, void *buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t r = fread((uint8_t *)buf + done, 1, n - done, f);
        if (!r)
            return -1;
        done += r;
    }
    return 0;
}

static int write_exact(FILE *f, const void *buf, size_t n) {
    size_t done = 0;
    while (done < n) {
        size_t w = fwrite((const uint8_t *)buf + done, 1, n - done, f);
        if (!w)
            return -1;
        done += w;
    }
    return 0;
}

static int npy_load_u8_3d(const char *path, uint8_t **data, int *d, int *h, int *w) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) {
        fclose(f);
        return -1;
    }
    size_t hlen = pre[6] == 1 ? (size_t)(pre[8] | (pre[9] << 8)) : 0;
    if (pre[6] != 1) {
        unsigned char more[2];
        if (fread(more, 1, 2, f) != 2) {
            fclose(f);
            return -1;
        }
        hlen = (size_t)(pre[8] | (pre[9] << 8) | (more[0] << 16) | (more[1] << 24));
    }
    char *hdr = (char *)calloc(hlen + 1, 1);
    if (fread(hdr, 1, hlen, f) != hlen) {
        free(hdr);
        fclose(f);
        return -1;
    }
    const char *sh = strstr(hdr, "'shape'");
    long long s0 = 0, s1 = 0, s2 = 0;
    int ok = strstr(hdr, "|u1") && sh &&
        sscanf(strchr(sh, '('), "(%lld, %lld, %lld)", &s0, &s1, &s2) == 3;
    free(hdr);
    if (!ok) {
        fclose(f);
        return -1;
    }
    size_t n = (size_t)s0 * (size_t)s1 * (size_t)s2;
    uint8_t *buf = (uint8_t *)malloc(n);
    if (!buf || fread(buf, 1, n, f) != n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *data = buf;
    *d = (int)s0;
    *h = (int)s1;
    *w = (int)s2;
    return 0;
}

static int npy_load_f32_5d(const char *path, float **data, int *b) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    unsigned char pre[10];
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) {
        fclose(f);
        return -1;
    }
    size_t hlen = pre[6] == 1 ? (size_t)(pre[8] | (pre[9] << 8)) : 0;
    if (pre[6] != 1) {
        unsigned char more[2];
        if (fread(more, 1, 2, f) != 2) {
            fclose(f);
            return -1;
        }
        hlen = (size_t)(pre[8] | (pre[9] << 8) | (more[0] << 16) | (more[1] << 24));
    }
    char *hdr = (char *)calloc(hlen + 1, 1);
    if (fread(hdr, 1, hlen, f) != hlen) {
        free(hdr);
        fclose(f);
        return -1;
    }
    const char *sh = strstr(hdr, "'shape'");
    long long s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    int ok = strstr(hdr, "<f4") && sh &&
        sscanf(strchr(sh, '('), "(%lld, %lld, %lld, %lld, %lld)", &s0, &s1, &s2, &s3, &s4) == 5 &&
        s1 == 1 && s2 == 16 && s3 == 128 && s4 == 128;
    free(hdr);
    if (!ok) {
        fclose(f);
        return -1;
    }
    size_t n = (size_t)s0 * 16 * 128 * 128;
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf || fread(buf, sizeof(float), n, f) != n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *data = buf;
    *b = (int)s0;
    return 0;
}

static int npy_save_f32(const char *path, const float *data, const int *shape, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    char sizes[128] = "", hdr[256];
    int used = 0;
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        used += snprintf(sizes + used, sizeof(sizes) - (size_t)used, i ? ", %d" : "%d", shape[i]);
        total *= (size_t)shape[i];
    }
    if (ndim == 1)
        snprintf(sizes + used, sizeof(sizes) - (size_t)used, ",");
    int n = snprintf(
        hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", sizes);
    int pad = (64 - (10 + n + 1) % 64) % 64;
    int len = n + pad + 1;
    unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char)(len & 0xff),
        (unsigned char)(len >> 8)};
    fwrite(magic, 1, 10, f);
    fwrite(hdr, 1, (size_t)n, f);
    for (int i = 0; i < pad; i++)
        fputc(' ', f);
    fputc('\n', f);
    size_t wrote = fwrite(data, sizeof(float), total, f);
    fclose(f);
    return wrote == total ? 0 : -1;
}

static void parse_flags(int argc, char **argv, int start, StudioConfig *cfg) {
    for (int i = start; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--batch"))
            cfg->batch_size = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--mode"))
            cfg->use_f444 = strcmp(argv[i + 1], "f222") != 0;
        else if (!strcmp(argv[i], "--profile"))
            cfg->profile = atoi(argv[i + 1]);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: studio_cuda worker WEIGHTS [--batch B] [--mode f444|f222]\n"
            "       studio_cuda tiles WEIGHTS IN.npy OUT.npy [--batch B] [--mode f444|f222] "
            "[--profile 1]\n"
            "       studio_cuda run WEIGHTS IN.npy OUT.npy [--batch B] [--mode f444|f222] "
            "[--profile 1]\n"
            "       studio_cuda bench WEIGHTS H W [--batch B] [--mode f444|f222] [--profile 1]\n");
        return 2;
    }
    const char *cmd = argv[1];
    const char *wpath = argv[2];
    StudioConfig cfg = {.batch_size = 8, .use_f444 = 1, .profile = 0};
    StudioWeights w;
    if (studio_weights_open(wpath, &w) != 0) {
        fprintf(stderr, "studio_cuda: cannot open %s\n", wpath);
        return 1;
    }
    if (!strcmp(cmd, "worker")) {
        parse_flags(argc, argv, 3, &cfg);
        StudioEngine *eng = studio_engine_create(&w, cfg);
        studio_weights_close(&w);
        if (write_exact(stdout, "HXOK", 4) != 0 || fflush(stdout) != 0)
            return 1;
        uint32_t hdr[3];
        while (read_exact(stdin, hdr, sizeof(hdr)) == 0) {
            size_t n_in = (size_t)hdr[0] * hdr[1] * hdr[2];
            size_t n_out = (size_t)hdr[1] * hdr[2];
            uint8_t *vol = (uint8_t *)malloc(n_in);
            float *out = (float *)malloc(n_out * sizeof(float));
            if (read_exact(stdin, vol, n_in) != 0) {
                free(vol);
                free(out);
                break;
            }
            studio_engine_predict(eng, vol, (int)hdr[0], (int)hdr[1], (int)hdr[2], out);
            int bad = write_exact(stdout, out, n_out * sizeof(float)) != 0 || fflush(stdout) != 0;
            free(vol);
            free(out);
            if (bad)
                break;
        }
        studio_engine_destroy(eng);
        return 0;
    }
    if (!strcmp(cmd, "tiles") && argc >= 5) {
        parse_flags(argc, argv, 5, &cfg);
        float *in = NULL;
        int b = 0;
        if (npy_load_f32_5d(argv[3], &in, &b) != 0) {
            fprintf(stderr, "studio_cuda: cannot read 5D float32 npy %s\n", argv[3]);
            return 1;
        }
        if (cfg.batch_size < b)
            cfg.batch_size = b;
        StudioEngine *eng = studio_engine_create(&w, cfg);
        studio_weights_close(&w);
        float *out = (float *)calloc((size_t)b * 128 * 128, sizeof(float));
        studio_engine_forward_tiles(eng, in, b, out);
        double t0 = now_s();
        studio_engine_forward_tiles(eng, in, b, out);
        double dt = now_s() - t0;
        printf("studio_cuda tiles: B=%d mode=%s %.3f ms total (%.3f ms/tile)\n", b,
            cfg.use_f444 ? "f444" : "f222", dt * 1e3, dt * 1e3 / b);
        if (cfg.profile)
            studio_engine_print_profile(eng);
        int sh[4] = {b, 1, 128, 128};
        npy_save_f32(argv[4], out, sh, 4);
        free(in);
        free(out);
        studio_engine_destroy(eng);
        return 0;
    }
    if (!strcmp(cmd, "run") && argc >= 5) {
        parse_flags(argc, argv, 5, &cfg);
        uint8_t *vol = NULL;
        int d = 0, h = 0, wd = 0;
        if (npy_load_u8_3d(argv[3], &vol, &d, &h, &wd) != 0) {
            fprintf(stderr, "studio_cuda: cannot read 3D uint8 npy %s\n", argv[3]);
            return 1;
        }
        StudioEngine *eng = studio_engine_create(&w, cfg);
        studio_weights_close(&w);
        float *out = (float *)calloc((size_t)h * wd, sizeof(float));
        studio_engine_predict(eng, vol, d, h, wd, out);
        double t0 = now_s();
        studio_engine_predict(eng, vol, d, h, wd, out);
        double dt = now_s() - t0;
        printf("studio_cuda run: %dx%dx%d batch=%d mode=%s %.3f ms (%.2f Mpix/s)\n", d, h, wd,
            cfg.batch_size, cfg.use_f444 ? "f444" : "f222", dt * 1e3, (double)h * wd / dt / 1e6);
        if (cfg.profile)
            studio_engine_print_profile(eng);
        int sh[2] = {h, wd};
        npy_save_f32(argv[4], out, sh, 2);
        free(vol);
        free(out);
        studio_engine_destroy(eng);
        return 0;
    }
    if (!strcmp(cmd, "bench") && argc >= 5) {
        int h = atoi(argv[3]), wd = atoi(argv[4]);
        parse_flags(argc, argv, 5, &cfg);
        double t_load = now_s();
        StudioEngine *eng = studio_engine_create(&w, cfg);
        studio_weights_close(&w);
        printf("studio_cuda load+transform: %.3f s\n", now_s() - t_load);
        uint8_t *vol = (uint8_t *)malloc((size_t)16 * h * wd);
        for (size_t i = 0; i < (size_t)16 * h * wd; i++)
            vol[i] = (uint8_t)((i * 2654435761u >> 24) % 180 + 20);
        float *out = (float *)calloc((size_t)h * wd, sizeof(float));
        studio_engine_predict(eng, vol, 16, h, wd, out);
        double t0 = now_s();
        int runs = 3;
        for (int r = 0; r < runs; r++)
            studio_engine_predict(eng, vol, 16, h, wd, out);
        double dt = (now_s() - t0) / runs;
        printf("studio_cuda bench: 16x%dx%d batch=%d mode=%s %.3f ms (%.2f Mpix/s)\n", h, wd,
            cfg.batch_size, cfg.use_f444 ? "f444" : "f222", dt * 1e3, (double)h * wd / dt / 1e6);
        if (cfg.profile)
            studio_engine_print_profile(eng);
        free(vol);
        free(out);
        studio_engine_destroy(eng);
        return 0;
    }
    studio_weights_close(&w);
    return 2;
}
