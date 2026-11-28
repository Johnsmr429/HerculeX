#ifndef VESUVIUS_AXIS_WARP_H
#define VESUVIUS_AXIS_WARP_H

#include <stddef.h>
#include <stdint.h>
#include "shaft_warp.h"

typedef struct {
    double source_voxel_um_zyx[3], source_origin_um_zyx[3];
    double metric_voxel_um;
    double maximum_radius_um, minimum_jacobian, ambiguity_um;
    double initial_normal_zyx[3];
    int has_initial_normal;
} AxisWarpPhysical;

typedef struct {
    double *z;
    double *y;
    double *x;
    size_t n;
    double reference_y;
    double reference_x;
    ShaftWarp *physical;
    AxisWarpPhysical physical_config;
} AxisWarp;

void AxisWarp_init(AxisWarp *warp);
void AxisWarp_dispose(AxisWarp *warp);

int AxisWarp_load_csv(AxisWarp *warp, const char *path);

int AxisWarp_create_physical(AxisWarp *warp, const double *points_um_zyx,
                            size_t count, const AxisWarpPhysical *config,
                            double reference_y, double reference_x);
int AxisWarp_to_metric(const AxisWarp *warp, const double source_zyx[3],
                       double metric_suv[3], uint32_t *flags);
int AxisWarp_to_world(const AxisWarp *warp, const double metric_suv[3],
                      double source_zyx[3], uint32_t *flags);
uint64_t AxisWarp_fingerprint(const AxisWarp *warp);

int AxisWarp_valid(const AxisWarp *warp);
void AxisWarp_eval(const AxisWarp *warp, double z, double *y, double *x);

void AxisWarp_straighten_point(const AxisWarp *warp,
                               const float in_zyx[3], float out_zyx[3]);
void AxisWarp_straighten_vertices(const AxisWarp *warp,
                                  const float *in_zyx, float *out_zyx,
                                  size_t nvertices);

int AxisWarp_selftest(void);

#endif
