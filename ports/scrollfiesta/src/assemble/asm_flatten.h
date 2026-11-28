#ifndef ASM_FLATTEN_INCLUDED
#define ASM_FLATTEN_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "../common/arena.h"

enum {
    ASM_FLAT_OK = 0,
    ASM_FLAT_EMPTY,
    ASM_FLAT_BAD_INDEX,
    ASM_FLAT_NONMANIFOLD,
    ASM_FLAT_NO_BOUNDARY,
    ASM_FLAT_DEGENERATE,
    ASM_FLAT_SOLVE,
    ASM_FLAT_FOLDED
};

typedef struct AsmFlattenStats {
    size_t nv, nf;
    size_t n_boundary_loops;
    size_t n_holes_capped;
    size_t n_flipped;
    double area3d, area_uv;
    double sigma_lo, sigma_hi;
    double stress_frac;
    int    arap_iters;
    int    metric_refine_iters;
    int    ok;
    int    fail_reason;
    int    seed_mode;
    int    fail_stage;
    size_t fail_contacts;
} AsmFlattenStats;

int AsmFlatten_chart(Arena_T arena,
                     const float *xyz, size_t nv,
                     const int32_t *faces, size_t nf,
                     double band, int max_iters,
                     float *out_uv, AsmFlattenStats *stats);

size_t AsmFlatten_drop_degenerate(const float *xyz, size_t nv, const int32_t *faces, size_t nf,
                                  int32_t *out_faces, int32_t *out_vmap, size_t *out_nv);

int AsmFlatten_face_degenerate(const float *xyz, const int32_t *face);

int AsmFlatten_chart_certified(Arena_T arena, const float *xyz, size_t nv, const int32_t *faces, size_t nf,
                               double band, int max_iters, float *out_uv, AsmFlattenStats *st);

size_t AsmFlatten_crumple_faces(Arena_T arena, const float *xyz, size_t nv, const int32_t *faces, size_t nf,
                                double cone_rad, int ring, int cut_holes, const uint8_t *seed, uint8_t *excise);

int AsmFlatten_rescue_crumpled(Arena_T arena, const float *xyz, size_t nv, const int32_t *faces, size_t nf,
                               double band, int max_iters, double min_piece_area,
                               int32_t *piece, float *uv, size_t *n_pieces, AsmFlattenStats *st);

void AsmFlatten_face_singular(const float *xyz, const int32_t *faces, size_t nf,
                              const float *uv, double *sig_lo, double *sig_hi,
                              int8_t *det_sign);

int AsmFlatten_selftest(void);

#endif
