#ifndef PCA_INCLUDED
#define PCA_INCLUDED

#include <stddef.h>

int PCA_normal(const float *points, size_t n,
               float out_normal[3], float out_centroid[3]);

int PCA_principal_axis(const float *points, size_t n,
                       float out_axis[3], float out_centroid[3]);

void PCA_project(const float *points, size_t n, const float dir[3],
                 float *out_proj, float *out_min, float *out_max);

void PCA_orthonormal_basis(const float normal[3],
                           float out_u[3], float out_v[3]);

#endif
