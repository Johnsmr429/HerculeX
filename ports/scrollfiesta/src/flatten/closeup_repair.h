#ifndef CLOSEUP_REPAIR_INCLUDED
#define CLOSEUP_REPAIR_INCLUDED
#include <stddef.h>
#include <stdint.h>

enum {
    CLOSEUP_REPAIR_KEEP = 0,
    CLOSEUP_REPAIR_DUPLICATE = 1,
    CLOSEUP_REPAIR_WRONG_FILL = 2,
    CLOSEUP_REPAIR_FLAP = 3,
    CLOSEUP_REPAIR_ORPHAN = 4
};

typedef struct CloseupRepairStats {
    size_t verts, observed, fill, undefined;
    size_t duplicate, wrong_fill, flap, orphan;
    size_t pairs_fill_observed, pairs_fill_fill;
    size_t pairs_observed_observed;
    size_t flap_components, flap_specks;
    size_t orphan_components;
    int inward_flipped;
    double observed_facing_axis;
    double seconds;
} CloseupRepairStats;

int CloseupRepair_mark(const float *verts, const double *uv, size_t nv, const int32_t *faces, size_t nf,
                       const uint8_t *state, const float *nrm, const float *inward, uint8_t *reason,
                       CloseupRepairStats *st);

int CloseupRepair_selftest(void);

#endif
