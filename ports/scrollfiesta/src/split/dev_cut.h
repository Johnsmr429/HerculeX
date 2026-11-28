#ifndef DEV_CUT_INCLUDED
#define DEV_CUT_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"

int DevCut_process(Arena_T arena, const ComponentMesh *mesh,
                   double eps, double gap_depth, size_t min_comp_verts,
                   ComponentMesh **out_meshes, size_t *out_count);

double DevCut_nondev_fraction(Arena_T arena, const ComponentMesh *mesh,
                              double eps, double *out_mean_lambda);

int DevCut_selftest(void);

#endif
