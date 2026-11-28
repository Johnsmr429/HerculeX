#ifndef STRIP_PREVIEW_INCLUDED
#define STRIP_PREVIEW_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct StripPreviewOpts {
    int ds_u, ds_v;
    int rows;
    int max_w;
    int gap;
    const char *mode;
    const char *overlay;
    int verbose;
} StripPreviewOpts;

void StripPreviewOpts_default(StripPreviewOpts *o);

int StripPreview_overview(Arena_T arena, const char *tif, const char *png,
                          const StripPreviewOpts *o);

int StripPreview_crop(Arena_T arena, const char *tif, const char *png,
                      long x0, long x1, int y0, int y1,
                      const StripPreviewOpts *o);

int StripPreview_windows(Arena_T arena, const char *tif, const char *png,
                         const long *cols, size_t ncols, int win,
                         const StripPreviewOpts *o);

int StripPreview_selftest(void);

#endif
