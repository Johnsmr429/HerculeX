#ifndef ASM_EMIT_INCLUDED
#define ASM_EMIT_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "asm_types.h"

typedef struct AsmEmitOpts {
    const char *raw_source;
    const char *bake_exe;
    double      normal_range;
    double      raster_du, raster_dv;
    int         level_v_to_z;
} AsmEmitOpts;

typedef struct AsmEmitStats {
    size_t sheet_charts, sheet_verts, sheet_faces;
    size_t extras_charts, extras_verts, extras_faces;
    size_t metric_checked_charts, metric_regressions;
    double sheet_area, extras_area;
    double u_span, v_span;
    double gauge_theta;
    int    bake_rc;
    double sec;
} AsmEmitStats;

void AsmEmit_default_opts(AsmEmitOpts *o);

int AsmEmit_run(AsmRun *run, const AsmEmitOpts *o, const char *out_dir, AsmEmitStats *st);
int AsmEmit_selftest(void);

#endif
