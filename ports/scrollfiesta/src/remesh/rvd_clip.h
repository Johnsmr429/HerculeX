#ifndef RVD_CLIP_INCLUDED
#define RVD_CLIP_INCLUDED

#include "../common/arena.h"
#include "sizing_field.h"
#include <stddef.h>
#include <stdint.h>

typedef struct Rvd_T *Rvd_T;

typedef struct {
    uint64_t tk_near;
    uint64_t tk_cand;
    uint64_t tk_clip;
    uint64_t n_passes;
    uint64_t n_faces;
    uint64_t n_fast;
    uint64_t n_cand;
    uint64_t n_sat;
    uint64_t n_cells;
    uint64_t n_planes;
    uint64_t n_preds;
    uint64_t n_wide;
} RvdStats;

Rvd_T Rvd_new(Arena_T arena, const double *verts, size_t nv,
              const int32_t *faces, size_t nf);

void Rvd_set_stats(Rvd_T rvd, RvdStats *stats);

int Rvd_accumulate(Rvd_T rvd, const double *sites, size_t nsites,
                   double *out_area, double *out_centroid,
                   double *out_NA, double *out_bNA,
                   int32_t **out_dual_tris, size_t *out_ndual,
                   const CvtField *dens);

double Rvd_mesh_area(Rvd_T rvd);

int Rvd_selftest(void);

#endif
