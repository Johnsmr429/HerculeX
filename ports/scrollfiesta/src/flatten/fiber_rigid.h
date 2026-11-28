#ifndef FIBER_RIGID_INCLUDED
#define FIBER_RIGID_INCLUDED

#include "slim_refine.h"

typedef struct FiberRigidStats {
    size_t faces, supported_faces;
    double area, supported_area, evidence_mass;
    double radians, center[2];
    double rms_before_degrees, rms_after_degrees;
    double signed_mean_before_degrees, signed_mean_after_degrees;
} FiberRigidStats;

int FiberRigid_fit(Arena_T arena, const float *xyz, size_t nv,
                   const int32_t *faces, size_t nf, const double *uv,
                   const SlimRefineDirection *directions, FiberRigidStats *stats);

int FiberRigid_apply(const double *uv, size_t nv, const FiberRigidStats *stats,
                     double *out);

int FiberRigid_selftest(void);
#endif
