#ifndef MLS_PROJECT_INCLUDED
#define MLS_PROJECT_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "arena.h"

void MLS_set_point_weights(const float *weights);

int MLS_selftest(void);

void MLS_test_reference_kernel(int on);

void MLS_project_verts(Arena_T arena,
                       const float *verts, size_t nv,
                       float radius_vox,
                       const float cell_origin[3],
                       float *out_verts,
                       float *out_normals);

#ifdef VESUVIUS_MLS_CUBECL
const char *MLS_cubecl_last_error(void);
int MLS_cubecl_last_call_ok(void);
int MLS_cubecl_preflight(void);
#else
static inline const char *MLS_cubecl_last_error(void) { return "no error"; }
static inline int MLS_cubecl_last_call_ok(void) { return 1; }
static inline int MLS_cubecl_preflight(void) { return 1; }
#endif

#endif
