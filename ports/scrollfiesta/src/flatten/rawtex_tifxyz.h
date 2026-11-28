#ifndef RAWTEX_TIFXYZ_H
#define RAWTEX_TIFXYZ_H

#include <stddef.h>
#include <stdint.h>

#include "../common/raw_sample.h"

typedef struct RawtexTifxyzPiece {
    size_t col0, col1;
    size_t row0, row1;
    size_t px_team, px_mirror;
    int    reflected;
} RawtexTifxyzPiece;

size_t RawtexTifxyz_reflect(float *pos, int8_t *sign, size_t W, size_t H,
                            RawtexTifxyzPiece *pieces, size_t npieces);

typedef struct RawtexTifxyzMaskStats {
    size_t rendered_px;
    size_t clean_px;
    size_t polluted_px;
    size_t reversed_px;
    size_t unsupported_px;
    size_t mask_px;
} RawtexTifxyzMaskStats;

int RawtexTifxyz_canvas_mask(const float *pos, const int8_t *sign, size_t W, size_t H,
                             CubeTable *ct, int depth, double step, int flip,
                             uint8_t *mask, RawtexTifxyzMaskStats *st);

int RawtexTifxyz_write(const char *dir, const char *uuid, float *pos, size_t W, size_t H,
                       const char *extra_json);

int RawtexTifxyz_write_packed(const char *dir, const char *uuid, float *pos, size_t W, size_t H,
                              const char *extra_json);

int RawtexTifxyz_canvas_to_sheet(const uint8_t *canvas, size_t cW, size_t cH,
                                 const RawtexTifxyzPiece *pieces, size_t npieces,
                                 uint8_t *sheet, size_t sW, size_t sH);

int RawtexTifxyz_canvas_to_sheet_zarr(const char *canvas_array_dir, const RawtexTifxyzPiece *pieces,
                                      size_t npieces, const uint8_t *sheet_mask, size_t sW, size_t sH,
                                      const char *out_dir, const char *attrs_json, size_t *chunks_written,
                                      size_t *canvas_chunks_absent);

int RawtexTifxyz_canvas_to_sheet_zarr_depth(const char *canvas_array_dir, const RawtexTifxyzPiece *pieces,
                                            size_t npieces, const uint8_t *sheet_mask,
                                            const uint8_t *sheet_reverse, size_t sW, size_t sH,
                                            const char *out_dir, const char *attrs_json, size_t *chunks_written,
                                            size_t *canvas_chunks_absent);
int RawtexTifxyz_canvas_to_sheet_zarr_storage(const char *canvas_array_dir, const RawtexTifxyzPiece *pieces,
                                              size_t npieces, const uint8_t *sheet_mask,
                                              const uint8_t *sheet_reverse, size_t sW, size_t sH,
                                              const char *out_dir, const char *attrs_json, size_t shard_rows,
                                              size_t *chunks_written, size_t *canvas_chunks_absent);

int RawtexTifxyz_stretch(uint8_t *img, const uint8_t *mask, size_t n, double pct_lo, double pct_hi,
                         int *lo, int *hi);

void RawtexTifxyz_histogram(const uint8_t *img, const uint8_t *mask, size_t n, size_t hist[256]);
int RawtexTifxyz_window(const size_t hist[256], double pct_lo, double pct_hi, int *lo, int *hi);
void RawtexTifxyz_apply_window(uint8_t *img, const uint8_t *mask, size_t n, int lo, int hi);

int RawtexTifxyz_compose(uint8_t *sheet, size_t W, size_t H, const uint8_t *block, size_t bW, size_t bH,
                         size_t col0);

int RawtexTifxyz_load_gray(const char *path, uint8_t **img, size_t *W, size_t *H);

int RawtexTifxyz_save_gray(const char *path, const uint8_t *img, size_t W, size_t H);

int RawtexTifxyz_selftest(void);

#endif
