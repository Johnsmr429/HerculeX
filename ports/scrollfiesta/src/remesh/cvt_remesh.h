#ifndef CVT_REMESH_INCLUDED
#define CVT_REMESH_INCLUDED

#include "../common/arena.h"
#include "rvd_clip.h"
#include "sizing_field.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t tk_setup;
    uint64_t tk_bext;
    uint64_t tk_seed;
    uint64_t tk_rvd;
    uint64_t tk_bnd;
    uint64_t tk_proj;
    uint64_t tk_dual;
    uint64_t n_calls;
    uint64_t n_sites;
    uint64_t n_iters;
    uint64_t n_bnd_segs;
    uint64_t n_seed_tries;
    uint64_t n_bnd_calls;
    uint64_t n_seg_scans;
    uint64_t n_proj_calls;
    uint64_t n_proj_cand;
    uint64_t n_proj_cap;
    RvdStats rvd;
    uint64_t tk_probe;
    uint64_t n_probe;
} CvtPhaseStats;

void CVT_phase_stats_add(CvtPhaseStats *dst, const CvtPhaseStats *src);

typedef int (*CvtProbeFn)(void *ctx, int iter, const float *verts, size_t nv,
                          const int32_t *faces, size_t nf);
#define CVT_RC_PROBE_ABORT (-10)

typedef struct {
    int      n_iters;
    float    lambda_na;
    float    lambda_cvt0;
    float    cvt_decay;
    float    lambda_cvt_min;
    uint32_t seed;
    int      verbose;
    CvtPhaseStats *stats;
    CvtProbeFn probe;
    void    *probe_ctx;
    uint64_t probe_mask;
} CvtOpts;

void CVT_default_opts(CvtOpts *o);

int CVT_remesh(Arena_T arena,
               const float *in_verts, size_t in_nv,
               const int32_t *in_faces, size_t in_nf,
               size_t target_nsites,
               const CvtOpts *opts,
               const CvtField *field,
               float **out_verts, size_t *out_nv,
               int32_t **out_faces, size_t *out_nf);

typedef struct CvtPrep_T *CvtPrep_T;
CvtPrep_T CVT_prepare(Arena_T arena,
                      const float *in_verts, size_t in_nv,
                      const int32_t *in_faces, size_t in_nf,
                      const CvtField *field, CvtPhaseStats *stats, int *out_rc);
int CVT_remesh_prepared(Arena_T arena, CvtPrep_T prep,
                        size_t target_nsites,
                        const CvtOpts *opts,
                        float **out_verts, size_t *out_nv,
                        int32_t **out_faces, size_t *out_nf);

int CVT_projection_selftest(void);

int CVT_probe_selftest(void);

int CVT_boundary_bvh_selftest(void);

int CVT_remesh_pinned(Arena_T arena,
                      const float *in_verts, size_t in_nv,
                      const int32_t *in_faces, size_t in_nf,
                      double target_h,
                      const CvtOpts *opts,
                      float **out_verts, size_t *out_nv,
                      int32_t **out_faces, size_t *out_nf,
                      int32_t **out_site_src, size_t *out_n_pin);

#endif
