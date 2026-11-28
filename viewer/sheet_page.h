#pragma once

#include <json-c/json.h>

int hx_site_folders(const char *path);
int hx_mosaic_slot(json_object *groups, json_object *piece);
int hx_write_sheet_file(const char *path, void *data, size_t size);
int hx_sheet_card(const char *folder, int level, const double extent[4], const char *path);
int hx_sheet_review(json_object *rows, json_object *layers, const char *folder, const char *sample,
    const char *volume, const char *surface);

int hx_sheet_page(json_object *rows, json_object *layers, const char *folder, const char *title,
    const char *scan, int card_level);

json_object *hx_viewer_metadata(const char *path);
int hx_site_save_jpeg(const char *path, uint8_t *pixels, int w, int h, int quality);
int hx_site_resize_rgb(const uint8_t *input, int w, int h, uint8_t *output, int width, int height);

int hx_sheet_orient(json_object *rows, const char *folder, const char *side);
uint8_t *hx_sheet_png(const char *path, int *w, int *h);
json_object *hx_sheet_metadata(json_object *rows, json_object *visible, json_object *layers,
    const char *title, const char *scan, const double *fixed, json_object *boxes,
    json_object *suggested);
int hx_sheet_render(json_object *meta, const char *folder);
int hx_refresh_fields(const char *path, const char *page, json_object *fields);
int hx_site_save_jpeg_channels(
    const char *path, uint8_t *pixels, int w, int h, int quality, int channels);
int hx_site_resize_cubic(const uint8_t *input, int w, int h, double x_end, double y_end,
    uint8_t *output, int cols, int rows);
double decimal_round(double value, int digits);

int hx_site_resize_grey(
    const uint8_t *in, int w, int h, uint8_t *out, int cols, int rows, int linear);

static inline void hx_sheet_suggested_square(double square[4], double x_mm, double y_mm, double mm,
    double stride, double x, double y, int width, int height, int flip_cols, int flip_rows) {
    const int field = 1440;
    double margin = 2 * mm * stride;
    if (flip_cols) {
        x = width - x - field;
    }
    if (flip_rows) {
        y = height - y - field;
    }
    square[0] = decimal_round(x_mm - margin + x * mm, 2);
    square[1] = decimal_round(y_mm - margin + y * mm, 2);
    square[2] = square[3] = decimal_round(field * mm, 2);
}
