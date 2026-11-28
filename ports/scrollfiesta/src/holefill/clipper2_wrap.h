#ifndef CLIPPER2_WRAP_INCLUDED
#define CLIPPER2_WRAP_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int Clipper2_union(const double *pts_xy, size_t n,
                   double **out_pts, int **out_counts,
                   int *out_n_polys);

#ifdef __cplusplus
}
#endif

#endif
