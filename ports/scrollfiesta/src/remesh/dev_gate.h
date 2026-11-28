#ifndef DEV_GATE_INCLUDED
#define DEV_GATE_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"

enum { DEVGATE_NONDEV = 1, DEVGATE_ORACLE = 2 };

typedef struct {
    double k_thresh;
    double frac_thresh;
    size_t min_bad;
    int    oracle_grid;
} DevGateParams;

typedef struct {
    int    bad;
    int    reason;
    size_t n_interior;
    size_t n_interior_bad;
    double frac_interior_bad;
    double max_abs_k;
    int    oracle_sheets;
} DevGateVerdict;

void DevGate_default_params(DevGateParams *p);

int DevGate_classify(Arena_T arena, const ComponentMesh *cm,
                     const DevGateParams *p, DevGateVerdict *v);

#endif
