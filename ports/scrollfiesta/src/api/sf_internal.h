#ifndef SF_INTERNAL_H_INCLUDED
#define SF_INTERNAL_H_INCLUDED

#include "scrollfiesta.h"

#include "../common/arena.h"
#include "../common/except.h"
#include "../common/mesh_types.h"
#include "../common/run_ctx.h"
#include "../common/ves_platform.h"

extern ves_mutex_t g_sf_op_mutex;

typedef struct SfOpGuard {
    Arena_T  arena;
    SfRunCtx ctx;
    int      locked;
} SfOpGuard;

sf_status sf_op_begin(SfOpGuard *g, const sf_common_opts *opts);
void      sf_op_end(SfOpGuard *g);

#define SF_EXCEPT_TAIL(g_, rc_)                                              \
    EXCEPT(Arena_Failed) (rc_) = SF_ERROR_OOM;                               \
    EXCEPT(IO_Failed)    (rc_) = SF_ERROR_IO;                                \
    EXCEPT(Timeout)      (rc_) = SF_ERROR_TIMEOUT;                           \
    EXCEPT(Sf_Cancelled) (rc_) = (g_).ctx.timed_out ? SF_ERROR_TIMEOUT       \
                                                    : SF_CANCELLED;

float   *sf_xyz_to_zyx(Arena_T a, const float *xyz, size_t n);
int32_t *sf_faces_swap(Arena_T a, const int32_t *faces, size_t nf);

int sf_cm_from_mesh(Arena_T a, const sf_mesh *in, ComponentMesh *cm);

sf_status sf_mesh_out(sf_mesh *out,
                      const float *verts_zyx, size_t nv,
                      const int32_t *faces_internal, size_t nf,
                      const float *normals_zyx,
                      const uint8_t *pins,
                      const int32_t *vmap);
sf_status sf_mesh_out_cm(sf_mesh *out, const ComponentMesh *cm, const int32_t *vmap);
sf_status sf_mesh_out_raw(sf_mesh *out,
                          const float *verts_xyz, size_t nv,
                          const int32_t *faces, size_t nf,
                          const float *normals_xyz,
                          const uint8_t *pins,
                          const int32_t *vmap);

int32_t *sf_vmap_by_position(Arena_T a, const float *in_verts, size_t in_nv,
                             const float *out_verts, size_t out_nv);
int32_t *sf_vmap_identity(Arena_T a, size_t nv);

long sf_count_components(Arena_T a, size_t nv, const int32_t *faces, size_t nf);
int  sf_concat_cms(Arena_T a, const ComponentMesh *ms, size_t n, ComponentMesh *out);

#endif
