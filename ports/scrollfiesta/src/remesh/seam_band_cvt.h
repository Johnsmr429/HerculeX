#ifndef SEAM_BAND_CVT_INCLUDED
#define SEAM_BAND_CVT_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "seam_planes.h"

typedef struct {
    double band;
    double target_h;
    int    iters;
    size_t min_faces;
    double tile;
    int    offset_half;
    double fine_frac;
} SeamBandCvtParams;

typedef struct {
    size_t patches;
    size_t accepted;
    size_t rejected;
    size_t rej_rc;
    size_t rej_pin;
    size_t rej_bnd;
    size_t rej_chi;
    size_t rej_manifold;
    size_t rej_vtx;
    size_t rej_comp;
    size_t skipped_small;
    size_t skipped_clean;
    size_t faces_band_in;
    size_t faces_band_out;
    size_t verts_added;
} SeamBandCvtStats;

void SeamBandCvt_default_params(SeamBandCvtParams *p);

int SeamBandCvt_process(Arena_T arena,
                        const float *verts, size_t nv,
                        const int32_t *faces, size_t nf,
                        const SeamPlane *planes, size_t np,
                        const SeamBandCvtParams *params,
                        float **out_verts, size_t *out_nv,
                        int32_t **out_faces, size_t *out_nf,
                        int32_t **out_new_vert_src, size_t *out_n_new,
                        SeamBandCvtStats *st);

#endif
