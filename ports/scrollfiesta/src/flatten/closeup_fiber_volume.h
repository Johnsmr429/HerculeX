#ifndef CLOSEUP_FIBER_VOLUME_INCLUDED
#define CLOSEUP_FIBER_VOLUME_INCLUDED

#include "closeup_fiber.h"

int CloseupFiberVolume_measure(const float *xyz, const float *normal, size_t nv,
                              const int32_t *faces, size_t nf, const char *field,
                              SlimRefineDirection *out, CloseupFiberStats *stats);
int CloseupFiberVolume_selftest(void);

#endif
