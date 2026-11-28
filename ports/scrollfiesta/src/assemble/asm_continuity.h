#ifndef ASM_CONTINUITY_INCLUDED
#define ASM_CONTINUITY_INCLUDED
#include "asm_types.h"

enum {
    ASM_CONT_SOURCE = 1u,
    ASM_CONT_VERIFIED = 2u,
    ASM_CONT_CUT = 4u,
    ASM_CONT_INCUMBENT = 8u,
    ASM_CONT_REQUIRED = 16u,
    ASM_CONT_RECONSTRUCTED = 32u,
    ASM_CONT_INTRACHART_CUT = 64u
};

typedef struct AsmContinuityHalfspace {
    int ids[6], count;
    double values[6], rhs;
} AsmContinuityHalfspace;
int AsmContinuity_project_step(Arena_T arena, int n, int nt,
    const int *rows, const int *cols, const double *values, const double *rhs,
    double *step, const AsmContinuityHalfspace *constraints, size_t count);

typedef struct AsmContinuityStats {
    size_t source_runs, verified_joins, unresolved_hypotheses;
    size_t unsupported_islands, unsupported_charts;
    double unsupported_area;
    size_t unexplained_breaks, metadata_only_joins, wrong_placements;
    int audit_complete, confetti_free;
} AsmContinuityStats;

typedef struct AsmContinuityTarget {
    int32_t chart;
    double u, v, x, y, weight;
    int32_t parent;
} AsmContinuityTarget;

int AsmContinuity_chart_gaps(Arena_T arena, const AsmChart *chart,
                             const int32_t *vertices, const double *displacements,
                             size_t count, float *gaps, uint8_t *valid);

int AsmContinuity_prepare(AsmRun *run);
int AsmContinuity_reconstruct(AsmRun *run);
int AsmContinuity_preserves_incumbent(const AsmRun *run);
int AsmContinuity_preserves_incumbent_trial(const AsmRun *run, const AsmChart *saved, size_t n);
int AsmContinuity_target(const AsmRun *run, const AsmRelation *r,
                         const AsmCorr *c, int reverse, double target[2]);
int AsmContinuity_measure(const AsmRun *run, const AsmRelation *r,
                         double *rms, size_t *support);

double AsmContinuity_pair_error(const AsmRun *run, const AsmRelation *r, const AsmCorr *c);
size_t AsmContinuity_confirm(AsmRun *run);
int AsmContinuity_close(AsmRun *run, const AsmChart *saved, size_t n, const uint8_t *selected,
                        const AsmContinuityTarget *targets, size_t n_targets);
int AsmContinuity_close_boundaries(AsmRun *run, const AsmChart *saved, size_t n, const uint8_t *selected);
typedef struct AsmContinuityUvPair {
    int32_t chart_a, vertex_a, chart_b, vertex_b;
} AsmContinuityUvPair;

typedef struct AsmContinuityAuditCache {
    size_t baseline_bad, baseline_queries;
    int baseline_ready;
    uint8_t *baseline_violations;
    Arena_T scratch;
    uint8_t *blocked_charts;
    size_t blocked_charts_count;
    uint8_t **blocked_vertices;
    size_t blocked_vertices_count;
    AsmContinuityUvPair *pairs;
    size_t pairs_capacity, pairs_count;
    const AsmChart *guard_charts;
    size_t guard_count;
} AsmContinuityAuditCache;
typedef int (*AsmContinuityUvAudit)(AsmRun *run, const AsmChart *saved, size_t n, AsmContinuityAuditCache *cache);
size_t AsmContinuity_register_components(AsmRun *run, AsmContinuityUvAudit audit);
size_t AsmContinuity_register_failed(AsmRun *run, const AsmChart *entry, AsmContinuityUvAudit audit);
size_t AsmContinuity_relax_uv(AsmRun *run, AsmContinuityUvAudit audit, double stress_band, const char *diag_dir);
size_t AsmContinuity_relax_placed_uv(AsmRun *run, AsmContinuityUvAudit audit, double stress_band, const char *diag_dir);
int AsmContinuity_relax_boundary_trial(AsmRun *run, const AsmChart *saved, size_t n,
                                       const uint8_t *selected, AsmContinuityUvAudit audit, double stress_band);
int AsmContinuity_relax_boundary_step(AsmRun *run, const AsmChart *saved, size_t n,
                                      const uint8_t *selected, AsmContinuityUvAudit audit, double stress_band);
size_t AsmContinuity_components(AsmRun *run);
void AsmContinuity_audit(const AsmRun *run, int32_t primary,
                         const int32_t *emitted_chart, const float *xyz, size_t nv,
                         const int32_t *faces, size_t nf,
                         const char *path, AsmContinuityStats *st);
int AsmContinuity_selftest(void);
#endif
