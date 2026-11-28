#ifndef SEAM_REFINE_INCLUDED
#define SEAM_REFINE_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "seam_planes.h"

typedef struct {
    float target_len;
    float band;
    float min_parent_alt;
    int   max_rounds;
    int   flip_max_rounds;
} SeamRefineParams;

typedef struct {
    size_t planes;
    size_t rounds;
    size_t bnd_splits;
    size_t int_splits;
    size_t flips;
    size_t verts_added;
    size_t faces_added;
} SeamRefineStats;

void SeamRefine_default_params(SeamRefineParams *p);

int SeamRefine_process(Arena_T arena,
                       const float *verts, size_t nv,
                       const int32_t *faces, size_t nf,
                       const SeamPlane *planes, size_t np,
                       const SeamRefineParams *params,
                       float **out_verts, size_t *out_nv,
                       int32_t **out_faces, size_t *out_nf,
                       int32_t **out_new_vert_src, size_t *out_n_new,
                       SeamRefineStats *st);

int SeamRefine_process_with_parents(
                       Arena_T arena,
                       const float *verts, size_t nv,
                       const int32_t *faces, size_t nf,
                       const SeamPlane *planes, size_t np,
                       const SeamRefineParams *params,
                       float **out_verts, size_t *out_nv,
                       int32_t **out_faces, size_t *out_nf,
                       int32_t **out_new_vert_parent0,
                       int32_t **out_new_vert_parent1,
                       size_t *out_n_new,
                       SeamRefineStats *st);

int SeamRefine_process_masked_with_roots(
                       Arena_T arena,
                       const float *verts, size_t nv,
                       const int32_t *faces, size_t nf,
                       const SeamPlane *planes, size_t np,
                       const SeamRefineParams *params,
                       const uint8_t *freeze_source_faces,
                       size_t freeze_source_nf,
                       float **out_verts, size_t *out_nv,
                       int32_t **out_faces, size_t *out_nf,
                       int32_t **out_new_vert_parent0,
                       int32_t **out_new_vert_parent1,
                       int32_t **out_face_source,
                       size_t *out_n_new,
                       SeamRefineStats *st);

#endif
