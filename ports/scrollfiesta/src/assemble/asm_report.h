#ifndef ASM_REPORT_INCLUDED
#define ASM_REPORT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"
#include "asm_types.h"
#include "asm_conflict.h"

typedef struct AsmCanvas {
    uint8_t *rgb;
    int      w, h;
} AsmCanvas;

AsmCanvas AsmCanvas_new(Arena_T arena, int w, int h, uint8_t r, uint8_t g, uint8_t b);
void      AsmCanvas_tri(AsmCanvas *c, double x0, double y0, double x1, double y1,
                        double x2, double y2, const uint8_t rgb[3]);
void      AsmCanvas_line(AsmCanvas *c, double x0, double y0, double x1, double y1,
                         const uint8_t rgb[3]);
void      AsmCanvas_rect(AsmCanvas *c, double x0, double y0, double x1, double y1,
                         const uint8_t rgb[3]);
int       AsmCanvas_write(const AsmCanvas *c, const char *path);

void AsmReport_colour(uint32_t key, uint8_t rgb[3]);

int AsmReport_charts_atlas_png(Arena_T arena, const char *path,
                               const AsmChart *charts, size_t n,
                               int cell_px, size_t max_cells);

int AsmReport_layout_png(Arena_T arena, const char *path,
                         const AsmChart *charts, size_t n,
                         const AsmConflictCell *cells, size_t n_cells, double cell_size,
                         int max_px, size_t max_components);

int AsmReport_poses_csv(const AsmRun *run, const char *path, const char *stage);

#endif
