#ifndef CLOSEUP_FIBER_INCLUDED
#define CLOSEUP_FIBER_INCLUDED

#include "slim_refine.h"
#include "../common/zarr_u8.h"

typedef struct CloseupFiberStats {
    size_t patches, measured_patches, faces, samples;
    size_t chunk_misses, chunk_missing;
    double area, measured_area, seconds;
} CloseupFiberStats;

int CloseupFiber_measure(const float *xyz, const float *normal,
                          const double *frame_uv, size_t nv,
                          const int32_t *faces, size_t nf, const char *raw,
                          SlimRefineDirection *out, CloseupFiberStats *stats);
int CloseupFiber_measure_volume(ZarrU8 *volume, const float *xyz, const float *normal,
                                 const double *frame_uv, size_t nv,
                                 const int32_t *faces, size_t nf,
                                 SlimRefineDirection *out, CloseupFiberStats *stats);
int CloseupFiber_selftest(void);

#endif
