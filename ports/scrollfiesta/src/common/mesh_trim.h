#ifndef MESH_TRIM_INCLUDED
#define MESH_TRIM_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

int Mesh_trim_to_owned_box(Arena_T arena,
                           const float *verts_in, size_t nv_in,
                           const int32_t *faces_in, size_t nf_in,
                           const uint8_t *pin_mask_in,
                           float owned_lo, float owned_hi,
                           float **out_verts, size_t *out_nv,
                           int32_t **out_faces, size_t *out_nf,
                           uint8_t **out_pin_mask);

int Mesh_trim_cut_to_owned_box(Arena_T arena,
                               const float *verts_in, size_t nv_in,
                               const int32_t *faces_in, size_t nf_in,
                               float owned_lo, float owned_hi,
                               float snap_eps,
                               float **out_verts, size_t *out_nv,
                               int32_t **out_faces, size_t *out_nf,
                               size_t *out_faces_cut);

#endif
