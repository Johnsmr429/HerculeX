#ifndef VERT_WELD_INCLUDED
#define VERT_WELD_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "arena.h"

void Weld_verts(Arena_T arena,
                const float *verts, size_t nv,
                const float *in_normals,
                int32_t *faces, size_t nf, size_t *out_nf,
                float eps,
                bool guard_orient,
                float **out_verts, size_t *out_nv,
                float **out_normals);

void Weld_verts_grouped(Arena_T arena,
                        const float *verts, size_t nv,
                        const float *in_normals,
                        int32_t *faces, size_t nf, size_t *out_nf,
                        float eps,
                        bool guard_orient,
                        const int32_t *merge_group,
                        float **out_verts, size_t *out_nv,
                        float **out_normals);

typedef bool (*WeldPairFilter)(size_t i, size_t j, void *context);

void Weld_verts_filtered(Arena_T arena,
                         const float *verts, size_t nv,
                         const float *in_normals,
                         int32_t *faces, size_t nf, size_t *out_nf,
                         float eps,
                         bool guard_orient,
                         const int32_t *merge_group,
                         WeldPairFilter pair_filter,
                         void *pair_filter_context,
                         float **out_verts, size_t *out_nv,
                         float **out_normals);

int Weld_selftest(void);

#endif
