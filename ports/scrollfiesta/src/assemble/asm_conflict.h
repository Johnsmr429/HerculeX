#ifndef ASM_CONFLICT_INCLUDED
#define ASM_CONFLICT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "asm_types.h"
#include "asm_pose.h"

typedef struct AsmConflictOpts {
    double cell;
    double contra_lo;
    double contra_hi;
    int    rounds;
    int    max_path;
    double drop_frac;
    double extras_ratio;
    const char *diag_pairs_csv;
} AsmConflictOpts;

typedef struct AsmConflictCell {
    int32_t component;
    int32_t cx, cy;
    float   w;
    float   dn;
} AsmConflictCell;

typedef struct AsmConflictReport {
    size_t cells;
    size_t cells_multi;
    size_t cells_contra;
    double covered_area;
    double contra_area;
    double score;
    size_t pairs;
    AsmConflictCell *contra_cells;
    size_t n_contra_cells;
} AsmConflictReport;

typedef struct AsmConflictStats {
    int    rounds;
    size_t rels_dropped;
    size_t rels_reverted;
    size_t charts_extras;
    size_t splits_refused;
    double split_area_refused;
    double score_first, score_last;
    double contra_first, contra_last;
    double covered_first, covered_last;
    size_t pairs_first, pairs_last;
    double sec;
} AsmConflictStats;

void AsmConflict_default_opts(AsmConflictOpts *o);

typedef struct AsmConflictGrid AsmConflictGrid;
AsmConflictGrid *AsmConflictGrid_new(Arena_T arena, const AsmConflictOpts *o, const AsmRun *run);
void AsmConflictGrid_insert(AsmConflictGrid *g, const AsmChart *c);
double AsmConflictGrid_probe(AsmConflictGrid *g, const AsmChart *const *charts, size_t n,
                             int32_t *partner, double *partner_mass);
double AsmConflictGrid_probe_excluding(AsmConflictGrid *g, const AsmChart *const *charts, size_t n,
                                       const uint8_t *excluded, int32_t *partner, double *partner_mass);
size_t AsmConflictGrid_cells(const AsmConflictGrid *g);
size_t AsmConflictGrid_stacked(const AsmConflictGrid *g, size_t *cells);
void AsmConflictGrid_dispose(AsmConflictGrid *g);

int AsmConflict_audit(AsmRun *run, const AsmConflictOpts *o,
                      double *out_chart_mass, AsmConflictReport *rep);

typedef struct AsmJoinStats {
    size_t pairs_tested, pairs_joined, pairs_refused_disagree, pairs_refused_short, pairs_refused_veto;
    size_t cells_agree, cells_disagree;
    size_t lineages_before, lineages_after;
    size_t charts_shifted;
    double shift_max;
    double rot_max;
    double sec;
} AsmJoinStats;
int AsmConflict_confirm_joins(AsmRun *run, const AsmConflictOpts *o, const char *diag_csv, AsmJoinStats *st);

typedef struct AsmRefineStats {
    size_t cand_join, cand_by_kind[5];
    size_t readmitted, readmit_by_kind[5], readmit_switched, unreadmitted;
    size_t refused_gap, refused_rot, refused_parity, refused_nocorr, internal_fixed;
    size_t charts_placed, charts_fixed, charts_free, charts_pinned;
    int    iters; size_t switched; int rounds; int kept; int reverted;
    size_t moved; double moved_area, shift_p50, shift_p90, shift_max, rot_max;
    double contra_before, contra_after, covered_before, covered_after;
    size_t lineages_before, lineages_after;
    double sec;
} AsmRefineStats;
int AsmConflict_refine_seams(AsmRun *run, const AsmConflictOpts *o, const AsmPoseOpts *po, const char *diag_csv, AsmRefineStats *st);

int AsmConflict_clean(AsmRun *run, const AsmConflictOpts *o,
                      const AsmPoseOpts *pose_opts,
                      AsmConflictReport *final_report, AsmConflictStats *st);

int AsmConflict_audit_selftest(void);
int AsmConflict_selftest(void);

#endif
