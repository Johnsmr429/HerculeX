#ifndef ASM_DISCOVER_INCLUDED
#define ASM_DISCOVER_INCLUDED

#include <stddef.h>
#include "asm_types.h"

typedef struct AsmDiscoverOpts {
    int enabled;
    int min_sides;
    int soft_min_sides;
    int rounds;
    int selected_obligations;
} AsmDiscoverOpts;

typedef struct AsmDiscoverStats {
    size_t examined;
    size_t with_partner;
    size_t admitted;
    double admitted_area;
    size_t refused_disagree, refused_sides, refused_rim, refused_wrap;
    size_t admitted_overlapping;
    int    rounds;
    size_t control_poses, control_beyond;
    double control_p50, control_p90, control_max;
    double sec;
} AsmDiscoverStats;

void AsmDiscover_default_opts(AsmDiscoverOpts *o);

int AsmDiscover_run(AsmRun *run, const AsmDiscoverOpts *o, const char *out_dir, AsmDiscoverStats *st);

int AsmDiscover_selftest(void);

#endif
