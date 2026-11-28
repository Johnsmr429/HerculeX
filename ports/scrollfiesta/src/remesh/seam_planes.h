#ifndef SEAM_PLANES_H
#define SEAM_PLANES_H

#include <stddef.h>
#include <stdint.h>

typedef struct { int axis; double coord; } SeamPlane;

size_t SeamPlanes_detect(const float *verts, size_t nv,
                         const uint8_t *used, double cube_size,
                         double band, SeamPlane *planes, size_t max_planes);

int SeamPlanes_edge_in(const float *verts, int32_t va, int32_t vb,
                       const SeamPlane *planes, size_t np, double band);

int SeamPlanes_face_straddle(const float *verts, int32_t a, int32_t b, int32_t c,
                             const SeamPlane *planes, size_t np, double band);

double SeamPlanes_vert_dist(const float *verts, int32_t v,
                            const SeamPlane *planes, size_t np);

#endif
