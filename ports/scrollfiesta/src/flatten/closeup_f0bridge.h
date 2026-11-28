#ifndef CLOSEUP_F0BRIDGE_H
#define CLOSEUP_F0BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "closeup_surface.h"

typedef struct CloseupF0 *CloseupF0;

CloseupF0 CloseupF0_open(const char *const *dirs, size_t ndirs, size_t max_cubes);
void      CloseupF0_close(CloseupF0 *f);
long      CloseupF0_preload(CloseupF0 f, const double lo[3], const double hi[3]);
int       CloseupF0_at(const CloseupF0 f, double z, double y, double x);

typedef struct CloseupF0BridgeOpts {
    double window_vox;
    double step_vox;
    double shift_vox;
    double max_run_vox;
    double depth_px;
    double dup_vox;
    double dup_margin_px;
    double ambig_vox;
    double layer_vox;
    int    rings;
    double box[4];
} CloseupF0BridgeOpts;

void CloseupF0BridgeOpts_default(CloseupF0BridgeOpts *o);

typedef struct CloseupF0BridgeStats {
    size_t candidates;
    size_t snapped;
    size_t no_cube;
    size_t no_run;
    size_t shifted;
    size_t thick;
    size_t broken;
    size_t deep;
    size_t dup;
    size_t ambiguous;
    size_t layered;
    size_t faces;
    size_t observations;
} CloseupF0BridgeStats;

int CloseupF0Bridge_pass(const CloseupGrid *g, CloseupF0 f0, const CloseupF0BridgeOpts *o, int32_t f0_chart,
                         CloseupObs *obs, CloseupF0BridgeStats *st);

int CloseupF0Bridge_selftest(void);

#endif
