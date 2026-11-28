#ifndef BALL_PIVOT_INCLUDED
#define BALL_PIVOT_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"

int BallPivot_reconstruct(Arena_T arena,
                          const float *verts,
                          const float *normals,
                          size_t nv,
                          float rho,
                          int32_t **out_faces,
                          size_t *out_nf);

typedef struct {
    double umb_y, umb_x;
    double origin_z, origin_y, origin_x;
    double pitch;
    double tol;
    const double *axis_z;
    const double *axis_y;
    const double *axis_x;
    size_t axis_n;
} BpaReconGate;

void BpaReconGate_set_axis_samples(const double *z, const double *y,
                                   const double *x, size_t n);
void BpaReconGate_clear_axis_samples(void);

void BpaReconGate_attach_axis(BpaReconGate *gate);
void BpaReconGate_axis_at(const BpaReconGate *gate, double world_z,
                          double *out_y, double *out_x);

int BallPivot_reconstruct_gated(Arena_T arena,
                                const float *verts,
                                const float *normals,
                                size_t nv,
                                float rho,
                                const BpaReconGate *gate,
                                int32_t **out_faces,
                                size_t *out_nf);

int BpaReconGate_from_env(BpaReconGate *out, const float origin_zyx[3]);

typedef struct {
    int32_t va, vb, v_opp;
} BpaInitEdge;

typedef struct {
    double umb_y, umb_x;
    double pitch;
    double tol;
    double hard;
    double span;
    const double *axis_z;
    const double *axis_y;
    const double *axis_x;
    size_t axis_n;
} BpaBridgeGate;

void BpaBridgeGate_axis_at(const BpaBridgeGate *gate, double z,
                           double *out_y, double *out_x);

int BallPivot_bridge(Arena_T arena,
                     const float *verts,
                     const float *normals,
                     size_t nv,
                     float rho,
                     float rho_max,
                     const BpaInitEdge *init_edges,
                     size_t n_init,
                     const BpaBridgeGate *gate,
                     int32_t **out_faces,
                     size_t *out_nf);

int BallPivot_winding_selftest(void);

int BallPivot_query_selftest(void);

#endif
