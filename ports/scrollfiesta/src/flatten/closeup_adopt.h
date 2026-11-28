#ifndef CLOSEUP_ADOPT_H
#define CLOSEUP_ADOPT_H

#include <stddef.h>
#include <stdint.h>

#include "closeup_surface.h"

typedef struct CloseupPool {
    size_t         nv, nf, nch;
    const float   *verts;
    const float   *nrm;
    const float   *inward;
    const double  *uv;
    const int32_t *faces;
    const size_t  *vofs;
    const size_t  *fofs;
    void         (*chart_fill)(const void *ctx, size_t chart);
    const void    *chart_ctx;
    uint8_t       *chart_ready;
} CloseupPool;

typedef struct CloseupAdoptOpts {
    double on_vox, foot_fill_px, min_extent_px, rms_px, conflict_vox, agree_frac, normal_deg, new_frac;
    int    min_corr;
    double metric_ratio, metric_frac;
    double continuity_px, continuity_slope;
    double layer_vox, layer_frac, layer_margin_px;
} CloseupAdoptOpts;

void CloseupAdoptOpts_default(CloseupAdoptOpts *o);

typedef struct CloseupPlacement {
    size_t chart;
    int    reflect;
    double c, s, tx, ty;
    size_t corr, landed, agree, fresh;
    double rms, extent;
} CloseupPlacement;

typedef struct CloseupAdoptStats {
    size_t candidates, touched, placed;
    size_t refused_extent, refused_rms, refused_conflict, refused_nothing_new, deferred;
    size_t refused_metric, refused_continuity;
    size_t refused_layer;
    size_t deferred_layer;
} CloseupAdoptStats;

int CloseupAdopt_round(const CloseupGrid *g, const CloseupPool *pool, const uint8_t *done,
                       const CloseupAdoptOpts *o, CloseupPlacement **out, size_t *nout, CloseupAdoptStats *st);

typedef struct CloseupAdoptAudit {
    CloseupPlacement p;
    size_t screen_hits, data_landed, data_agree, fill_landed, fill_agree;
    size_t ignored_fill_landed, ignored_fill_agree;
    double data_max_error, fill_max_error;
    double ignored_fill_max_error;
    int reason;
    double metric_bad_share;
    size_t cont_checked, cont_bad;
    size_t layer_checked, layer_bad;
} CloseupAdoptAudit;

typedef int (*CloseupAdoptObserver)(const CloseupAdoptAudit *record, void *context);

int CloseupAdopt_round_audited(const CloseupGrid *g, const CloseupPool *pool, const uint8_t *done,
                              const CloseupAdoptOpts *o, CloseupPlacement **out, size_t *nout,
                              CloseupAdoptStats *st, CloseupAdoptObserver observer, void *context);

void CloseupPlacement_apply(const CloseupPlacement *p, double u, double v, double *fu, double *fv);

int CloseupAdopt_selftest(void);

#endif
