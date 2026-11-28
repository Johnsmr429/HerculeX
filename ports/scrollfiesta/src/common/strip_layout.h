#ifndef STRIP_LAYOUT_INCLUDED
#define STRIP_LAYOUT_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include "arena.h"

#ifndef INT32_MAX
#include <limits.h>
#endif

typedef struct StripTurn { int32_t rot, flip; } StripTurn;

static inline int StripTurn_identity(StripTurn t) { return t.rot == 0 && !t.flip; }
static inline int StripTurn_swaps(StripTurn t) { return t.rot == 90 || t.rot == 270; }
static inline void StripTurn_size(StripTurn t, int32_t h, int32_t w, int32_t *th, int32_t *tw)
{
    if (StripTurn_swaps(t)) { *th = w; *tw = h; } else { *th = h; *tw = w; }
}
static inline void StripTurn_to_page(StripTurn t, int32_t u, int32_t v, int32_t h, int32_t w, int32_t *x, int32_t *y)
{
    if (t.flip) u = w - 1 - u;
    switch (t.rot) {
    case 90: *x = h - 1 - v; *y = u; break;
    case 180: *x = w - 1 - u; *y = h - 1 - v; break;
    case 270: *x = v; *y = w - 1 - u; break;
    default: *x = u; *y = v; break;
    }
}
static inline void StripTurn_to_source(StripTurn t, int32_t x, int32_t y, int32_t h, int32_t w, int32_t *u, int32_t *v)
{
    switch (t.rot) {
    case 90: *u = y; *v = h - 1 - x; break;
    case 180: *u = w - 1 - x; *v = h - 1 - y; break;
    case 270: *u = w - 1 - y; *v = x; break;
    default: *u = x; *v = y; break;
    }
    if (t.flip) *u = w - 1 - *u;
}
static inline void StripTurn_rect_to_source(StripTurn t, int32_t h, int32_t w, int32_t x0, int32_t y0, int32_t x1,
                                            int32_t y1, int32_t *u0, int32_t *v0, int32_t *u1, int32_t *v1)
{
    const int32_t xs[4] = {x0, x1 - 1, x0, x1 - 1}, ys[4] = {y0, y0, y1 - 1, y1 - 1};
    *u0 = *v0 = INT32_MAX; *u1 = *v1 = INT32_MIN;
    for (int k = 0; k < 4; k++) {
        int32_t u = 0, v = 0;
        StripTurn_to_source(t, xs[k], ys[k], h, w, &u, &v);
        if (u < *u0) *u0 = u;
        if (v < *v0) *v0 = v;
        if (u + 1 > *u1) *u1 = u + 1;
        if (v + 1 > *v1) *v1 = v + 1;
    }
}

typedef struct StripParams {
    double du, dv;
    double merge_gap_vox;
    double kibble_area_vox2;
    double aspect;
    int64_t max_line_pixels;
    double line_gap_vox;
    double piece_gap_vox;
    double margin_vox;
    double tray_break_lines;
    double wrap_whole_fraction;
    double soft_break_fraction;
    double min_room_fraction;
} StripParams;

typedef struct StripColumns {
    int32_t  W, H;
    int32_t *top;
    int32_t *bottom;
    int32_t *count;
} StripColumns;

typedef struct StripPiece {
    int32_t x0, x1;
    int32_t y0, y1;
    int64_t covered;
    double  area_vox2;
    int32_t tray;
    StripTurn turn;
} StripPiece;

typedef struct StripFragment {
    int32_t piece;
    int32_t line;
    int32_t sx0, sx1;
    int32_t sy0, sy1;
    int32_t dx;
    StripTurn turn;
} StripFragment;

static inline void StripFragment_page_size(const StripFragment *g, int32_t *w, int32_t *h)
{
    StripTurn_size(g->turn, g->sy1 - g->sy0, g->sx1 - g->sx0, h, w);
}

typedef struct StripLine {
    int32_t dy;
    int32_t h;
    int32_t tray;
} StripLine;

typedef struct StripPage {
    StripPiece    *pieces;      size_t n_pieces;
    StripFragment *frags;       size_t n_frags;
    StripLine     *lines;       size_t n_lines;
    int32_t W, H;
    int32_t flow_width;
    int32_t margin;
    size_t  n_tray;
    size_t  n_split;
    double  main_area_vox2, tray_area_vox2;
} StripPage;

void StripLayout_params_default(StripParams *p, double du, double dv);

int StripLayout_columns(Arena_T arena, const uint8_t *cov, int32_t W, int32_t H,
                        StripColumns *out);

int StripLayout_plan(Arena_T arena, const StripColumns *cols, const StripParams *p,
                     StripPage *out);

int StripLayout_segments(Arena_T arena, const StripColumns *cols, const StripParams *p,
                         StripPiece **out, size_t *n);

int StripLayout_columns_turned(Arena_T arena, const uint8_t *cov, int32_t w, int32_t h,
                               StripTurn t, StripColumns *out);

int StripLayout_plan_pieces(Arena_T arena, const StripColumns *cols, const StripPiece *pieces, size_t n,
                            const StripTurn *turns, const StripColumns *const *prof, const StripParams *p,
                            StripPage *out);

int StripLayout_render(const StripPage *pg, const uint8_t *tex, const uint8_t *cov,
                       int32_t W, int32_t H, uint8_t *page, int64_t *out_copied);

int StripLayout_downsample(const uint8_t *img, int32_t W, int32_t H, int32_t f,
                           uint8_t *out, int32_t *out_w, int32_t *out_h);

int StripLayout_write_ledger(FILE *f, const StripPage *pg, const StripParams *p);

int StripLayout_selftest(void);

#endif
