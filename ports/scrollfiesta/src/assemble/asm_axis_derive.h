#ifndef ASM_AXIS_DERIVE_INCLUDED
#define ASM_AXIS_DERIVE_INCLUDED

#include <stddef.h>

#include "../common/arena.h"
#include "asm_axis.h"
#include "asm_types.h"

typedef struct AsmAxisDeriveOpts {
    double        slab_vox;
    int           iters;
    size_t        per_chart;
    const AsmAxis *prior;
} AsmAxisDeriveOpts;

typedef struct AsmAxisDeriveRow {
    double s, z, y, x;
    size_t n;
    size_t kept;
    double cond;
    double aperture_deg;
    double resid_p50;
    double unc_major, unc_minor;
    double void_ratio;
    int    sectors;
    int    one_sided;
    int    status;
} AsmAxisDeriveRow;

typedef struct AsmAxisDeriveReport {
    AsmAxisDeriveRow *rows; size_t n_rows, n_lock;
    int    seeded;
    size_t seed_slabs;
    int    iters;
    double last_move;
    double tilt_deg;
    double fit_rms;
    double r_curv_min;
    double prior_rms;
    size_t one_sided;
    double cond_min, aperture_min_deg, resid_p50_max;
    size_t samples;
    double normal_dir[3], normal_cond, normal_frac, normal_tilt_deg, normal_disagree_deg;
    int    accepted;
    double sec;
} AsmAxisDeriveReport;

void AsmAxisDerive_default_opts(AsmAxisDeriveOpts *o);

int AsmAxis_normal_direction(const float *nrm, const float *w, size_t n, double dir[3], double *cond, double *frac);

AsmAxis *AsmAxis_derive(Arena_T arena, const float *p, const float *nrm, const float *w, size_t n,
                        const AsmAxisDeriveOpts *o, AsmAxisDeriveReport *rep);

AsmAxis *AsmAxis_derive_run(Arena_T arena, const AsmRun *run, const AsmAxisDeriveOpts *o, AsmAxisDeriveReport *rep);

int AsmAxisDerive_write_csv(const AsmAxisDeriveReport *rep, const char *path);

int AsmAxisDerive_write_png(Arena_T arena, const AsmAxis *derived, const AsmAxis *prior, const AsmAxisDeriveReport *rep, const char *path);

int AsmAxisDerive_selftest(void);

#endif
