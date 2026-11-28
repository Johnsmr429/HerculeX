#ifndef ASM_RELATE_INCLUDED
#define ASM_RELATE_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "asm_types.h"

#define ASM_RELATE_HIST 16

typedef struct AsmRelateOpts {
    double cube_size;
    double seam_max_vox;
    double seam_band_vox;
    double seam_min_len;
    double normal_dot_min;
    double tangency_max_vox;
    double iso_tol;
    double seam_noise_vox;
    double layer_probe_vox;
    int    layer_samples;
    const char *diag_csv;
} AsmRelateOpts;

typedef struct AsmRelateStats {
    size_t cube_pairs;
    size_t candidates;
    size_t chart_pairs;
    size_t rej_len, rej_tangency, rej_iso, rej_unimodal, rej_fit;
    size_t wrap_on_iso;
    size_t wrap_anchors;
    size_t weak_kept, short_kept;
    size_t bridge_demoted;
    size_t bridge_corroborated;
    size_t bridge_leaves;
    size_t contact_ledgered;
    size_t short_rej_corr, short_rej_rms;
    size_t accepted;
    size_t corr_total;
    size_t parity_flipped;
    double rms_p50, rms_p95;
    double iso_disc_sum;
    size_t iso_disc_n;
    size_t layer_hits;
    size_t layer_hist[ASM_RELATE_HIST];
    size_t gap_hist[ASM_RELATE_HIST];
    size_t gap_pairs;
    size_t layer_pairs;
    size_t charts_with_layer;
    double layer_d_global;
    double seam_sec, layer_sec;
} AsmRelateStats;

void AsmRelate_default_opts(AsmRelateOpts *o);

int AsmRelate_run(AsmRun *run, const AsmRelateOpts *opts, int threads,
                  AsmRelateStats *stats);

int AsmRelate_rigid_fit(const double *qa, const double *qb, const double *w, size_t n,
                        double *theta, double *tx, double *ty, double *rms);

int AsmRelate_selftest(void);

#endif
