#ifndef PNG_WRITE_INCLUDED
#define PNG_WRITE_INCLUDED

#include <stdint.h>

int PngWrite_rgb(const char *path, int w, int h, const uint8_t *rgb);

#endif
