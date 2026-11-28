#ifndef TIF_STRIP_INCLUDED
#define TIF_STRIP_INCLUDED

#include <stdint.h>

int TifStrip_write_gray(const char *path, const uint8_t *img, int W, int H,
                        int target_strip_w, int gap);

int TifStrip_selftest(void);

#endif
