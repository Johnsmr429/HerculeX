#ifndef RELAX_GRID_INCLUDED
#define RELAX_GRID_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"
#include "../flatten/ribbon_relax.h"
#include "piece_set.h"

typedef struct RelaxGridOpts {
    RibbonRelaxOpts base;
    size_t band_px;
    size_t halo_px;
    int    rounds;
    int    threads;
    double fib_grad_sigma, fib_tensor_sigma, fib_axis_tol;
    int    verbose;
} RelaxGridOpts;

void RelaxGridOpts_default(RelaxGridOpts *o);

typedef struct RelaxGridStats {
    size_t n_bands, bands_run, bands_reverted, bands_no_fiber;
    size_t n_moved;
    double energy_before, energy_after;
    double stretch_before, stretch_after;
    double mean_disp, max_disp;
    double seconds;
} RelaxGridStats;

int RelaxGrid_run(Arena_T arena, PieceSet *ps, const char *rawtex_tif,
                  double u0, double v0, double du, double dv,
                  const RelaxGridOpts *opts, RelaxGridStats *out);

int RelaxGrid_selftest(void);

#endif
