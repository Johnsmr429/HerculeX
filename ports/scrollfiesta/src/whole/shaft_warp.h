#ifndef VESUVIUS_SHAFT_WARP_H
#define VESUVIUS_SHAFT_WARP_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(SHAFT_WARP_BUILD_DLL)
#define SHAFT_WARP_API __declspec(dllexport)
#else
#define SHAFT_WARP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShaftWarp ShaftWarp;

enum {
    SHAFT_WARP_ENDPOINT = 1u,
    SHAFT_WARP_AMBIGUOUS = 2u,
    SHAFT_WARP_OUTSIDE_SUPPORT = 4u,
    SHAFT_WARP_FOLDED = 8u,
    SHAFT_WARP_OUTSIDE_RADIUS = 16u,
    SHAFT_WARP_BELOW_JACOBIAN = 32u,
    SHAFT_WARP_DIFFERENT_BRANCH = 64u
};

typedef struct {
    double parameter_um, s_um;
    double foot_um_zyx[3], tangent_zyx[3];
    double normal1_zyx[3], normal2_zyx[3], curvature_per_um_zyx[3];
    double distance_um, normal_residual_um, local_jacobian;
    uint32_t flags;
} ShaftWarpProjection;

SHAFT_WARP_API uint32_t ShaftWarp_abi(void);
SHAFT_WARP_API size_t ShaftWarp_projection_size(void);

SHAFT_WARP_API int ShaftWarp_create(ShaftWarp **out, const double *points_um_zyx,
                                   size_t count, const double initial_normal_zyx[3]);
SHAFT_WARP_API void ShaftWarp_free(ShaftWarp *warp);
SHAFT_WARP_API double ShaftWarp_length_um(const ShaftWarp *warp);
SHAFT_WARP_API double ShaftWarp_parameter_length_um(const ShaftWarp *warp);
SHAFT_WARP_API int ShaftWarp_eval_parameter(const ShaftWarp *warp, double parameter_um,
                                           ShaftWarpProjection *out);
SHAFT_WARP_API int ShaftWarp_eval_arc(const ShaftWarp *warp, double s_um,
                                     ShaftWarpProjection *out);
SHAFT_WARP_API int ShaftWarp_bounds_arc(const ShaftWarp *warp, double begin_um,
    double end_um, double radius_um, double lower_um_zyx[3], double upper_um_zyx[3]);
SHAFT_WARP_API int ShaftWarp_project(const ShaftWarp *warp, const double point_um_zyx[3],
                                    double ambiguity_um, ShaftWarpProjection *out);

SHAFT_WARP_API int ShaftWarp_to_metric(const ShaftWarp *warp, const double point_um_zyx[3],
    double maximum_radius_um, double minimum_jacobian, double ambiguity_um,
    double out_suv_um[3], uint32_t *flags);
SHAFT_WARP_API int ShaftWarp_from_metric(const ShaftWarp *warp, const double suv_um[3],
    double maximum_radius_um, double minimum_jacobian, double ambiguity_um,
    double out_point_um_zyx[3], uint32_t *flags);

#ifdef __cplusplus
}
#endif
#endif
