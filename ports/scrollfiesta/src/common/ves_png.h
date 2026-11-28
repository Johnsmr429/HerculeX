#ifndef VES_PNG_INCLUDED
#define VES_PNG_INCLUDED

#include <stdint.h>

int VesPng_write_gray(const char *path, const uint8_t *data, int w, int h);

int VesPng_write_rgb(const char *path, const uint8_t *data, int w, int h);

#endif
