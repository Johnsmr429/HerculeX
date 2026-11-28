#ifndef CC_COLOR_INCLUDED
#define CC_COLOR_INCLUDED

#include <stddef.h>
#include <stdint.h>

typedef struct {
    double sat;
    size_t min_faces;
} CCColorOpts;

typedef struct {
    size_t ncomp;
    size_t nv, nf;
    size_t largest_faces;
    size_t cover50, cover90, cover99;
    size_t dust_comps, dust_faces;
} CCColorStats;

void CCColor_default_opts(CCColorOpts *o);

size_t CCColor_compute(size_t nv, const int32_t *faces, size_t nf,
                       const CCColorOpts *opts, float *out_rgb, CCColorStats *stats);

int CCColor_selftest(void);

#endif
