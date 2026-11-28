#ifndef ASM_VERDICT_INCLUDED
#define ASM_VERDICT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "asm_axis.h"
#include "asm_types.h"
#include "asm_continuity.h"

typedef struct AsmVerdictOpts {
    double      du, dv;
    double      umb_y, umb_x;
    double      pitch;
    double      core_radius;
    const char *verdict_exe;
    int         include_blobs_in_source;
    const AsmAxis *axis;
} AsmVerdictOpts;

typedef struct AsmVerdictStats {
    AsmContinuityStats continuity;
    size_t lattice_verts, lattice_faces;
    size_t lattice_rows, lattice_cols;
    size_t cells_contested;
    size_t stacked_verts;
    size_t lineages;
    size_t faces_by_relation;
    size_t faces_by_geometry;
    size_t quads_refused;
    size_t quads_refused_lineage;
    size_t long_join_edges, long_join_pairs;
    size_t long_join_edges_tail;
    size_t cells_bridged_uv;
    size_t gap_within_n, gap_between_n;
    double gap_within_vox, gap_between_vox;
    double gap_within_p50, gap_within_p90, gap_between_p50, gap_between_p90;
    int    gap_measured;
    size_t solid_verts, solid_faces, solid_rows, solid_cols;
    size_t solid_contested;
    size_t cells_bridged;
    size_t source_verts, source_faces;
    int    axis_frame;
    int    frame_measured;
    int    verdict_rc;
    double sec;
} AsmVerdictStats;

void AsmVerdict_default_opts(AsmVerdictOpts *o);

int AsmVerdict_run(AsmRun *run, const AsmVerdictOpts *o, const char *out_dir, AsmVerdictStats *st);

const char *AsmVerdict_gate_word(const char *line, int *pass, int *measured);

int AsmVerdict_selftest(void);

#endif
