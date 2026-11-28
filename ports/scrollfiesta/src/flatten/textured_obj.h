#ifndef TEXTURED_OBJ_INCLUDED
#define TEXTURED_OBJ_INCLUDED

#include "rawtex_bake.h"

int TexturedObj_write(const char *prefix, const float *verts, const float *uv,
                       const float *normals, size_t nv,
                       const int32_t *faces, size_t nf, const RawtexPlan *plan,
                       double du, double dv, int do_3d, int do_flat);

int TexturedObj_write_field(const char *prefix, const float *verts, RawtexUv uv,
                       const float *normals, size_t nv,
                       const int32_t *faces, size_t nf, const RawtexPlan *plan,
                       double du, double dv, int do_3d, int do_flat);
int TexturedObj_selftest(void);
int TexturedObj_pyramid(const char *input,const char *output_prefix);

#endif
