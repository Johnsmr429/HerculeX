#ifndef INK_LETTERS_H
#define INK_LETTERS_H

#include <stddef.h>
#include <stdint.h>

typedef struct InkLettersOpts {
    double smooth_px;
    double t_hi, t_lo;
    int    close_px;
    double min_area, max_area;
    double row_link_h, row_tol_h;
    int    row_min;
    double warp_px;
    double w_row, w_conf, w_smooth, w_anchor;
} InkLettersOpts;

void InkLettersOpts_default(InkLettersOpts *o);

typedef struct InkLetter {
    double cx, cy;
    double angle;
    double ha, hb;
    double bx0, by0, bx1, by1;
    double area, mean_p, max_p;
    int    row;
} InkLetter;

typedef struct InkRow {
    double a, b, c;
    double x0, x1;
    double height;
    double rms;
    size_t n;
} InkRow;

int InkLetters_detect(const uint8_t *ink, const uint8_t *mask, size_t W, size_t H, const InkLettersOpts *o,
                      InkLetter **letters, size_t *n);

int InkLetters_rows(InkLetter *letters, size_t n, const InkLettersOpts *o, InkRow **rows, size_t *nrows,
                    double *h_med);

typedef struct InkWarp {
    size_t nx, ny;
    double step;
    double *dx, *dy;
    double *level;
    size_t nrows;
    double rms_before, rms_after;
    double scale_min, scale_max;
    double rot_max_deg;
} InkWarp;

int  InkLetters_warp(const InkLetter *letters, size_t n, const InkRow *rows, size_t nrows, size_t W, size_t H,
                     const InkLettersOpts *o, InkWarp *w);
void InkWarp_free(InkWarp *w);
void InkWarp_eval(const InkWarp *w, double x, double y, double *dx, double *dy);

void InkDraw_line(uint8_t *rgb, size_t W, size_t H, double x0, double y0, double x1, double y1, int thick,
                  uint8_t r, uint8_t g, uint8_t b);
void InkDraw_obb(uint8_t *rgb, size_t W, size_t H, const InkLetter *L, int thick, uint8_t r, uint8_t g, uint8_t b);
void InkDraw_text(uint8_t *rgb, size_t W, size_t H, int x, int y, int scale, const char *s,
                  uint8_t r, uint8_t g, uint8_t b);

int InkLetters_selftest(void);

#endif
