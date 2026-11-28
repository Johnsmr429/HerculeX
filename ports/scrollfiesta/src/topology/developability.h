#ifndef DEVELOPABILITY_INCLUDED
#define DEVELOPABILITY_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct {
    int    max_iters;
    double tol_grad;
    double tol_e;
    double max_disp_vox;
    int    pin_boundary;
    double c1;
    double t0;
    int    max_ls;
    int    verbose;
} DevelopOpts;

typedef struct {
    double e_initial;
    double e_final;
    int    iters;
    double max_disp;
    int    converged;
} DevelopStats;

int Develop_optimize(Arena_T arena, float *verts, size_t nv,
                     const int32_t *faces, size_t nf,
                     const uint8_t *pin_mask, float *vert_normals,
                     const DevelopOpts *opts, DevelopStats *out);

int Develop_vertex_energy(Arena_T arena, const float *verts, size_t nv,
                          const int32_t *faces, size_t nf, double *lam_out);

int Develop_selftest(void);

#endif
