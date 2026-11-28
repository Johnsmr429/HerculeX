#ifndef TIFF_IO_INCLUDED
#define TIFF_IO_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include "arena.h"

int TiffIO_load(Arena_T arena, const char *path,
                uint8_t **out_vol, int *out_D, int *out_H, int *out_W);

int TiffIO_save(const char *path,
                const uint8_t *vol, int D, int H, int W);

int TiffIO_save_float2d(const char *path, const float *img, int W, int H);

int TiffIO_save_deflate(const char *path, const uint8_t *vol, int D, int H, int W);
int TiffIO_save_float2d_deflate(const char *path, const float *img, int W, int H);
int TiffIO_save_float2d_predicted(const char *path, const float *img, int W, int H);

int TiffIO_storage_probe(const char *input, const char *fresh_directory);

int TiffIO_load_float2d(Arena_T arena, const char *path,
                        float **out_img, int *out_W, int *out_H);

typedef struct TiffFloatWindow TiffFloatWindow;
TiffFloatWindow *TiffIO_float_window_open(const char *path, size_t cache_bytes, int *W, int *H);
int TiffIO_float_window_read(TiffFloatWindow *reader, int y, int x, int h, int w, float *dst);
void TiffIO_float_window_close(TiffFloatWindow *reader);

int TiffIO_load_int32_2d(Arena_T arena, const char *path,
                         int32_t **out_img, int *out_W, int *out_H);

typedef struct TiffRowReader {
    FILE     *f;
    int       W, H;
    int       px_bytes;
    uint32_t  rows_per_strip;
    uint64_t *strip_off;
    uint32_t  n_strips;
} TiffRowReader;

int  TiffIO_rows_open(Arena_T arena, const char *path, TiffRowReader *rd);

int  TiffIO_rows_read(TiffRowReader *rd, int y, int x0, int w, uint8_t *dst);

int  TiffIO_rows_read_f32(TiffRowReader *rd, int y, int x0, int w, float *dst);

void TiffIO_rows_close(TiffRowReader *rd);

int TiffIO_selftest(const char *dir);

#endif
