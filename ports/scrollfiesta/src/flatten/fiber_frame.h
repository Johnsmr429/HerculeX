#ifndef FIBER_FRAME_INCLUDED
#define FIBER_FRAME_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include "../common/arena.h"
#include "fiber_map.h"

typedef struct FiberVolume *FiberVolume_T;

int  FiberVolume_open(const char *root, FiberVolume_T *out);
void FiberVolume_close(FiberVolume_T *fv);
double FiberVolume_sample(FiberVolume_T fv, double x, double y, double z, double d[3]);

typedef struct {
    int W, H;
    int x0, y0, x1, y1;
    int gh, gw, cell, gy0, gx0;
    float   *theta;
    float   *weight;
    float   *coherence;
    uint8_t *real;
    uint8_t *closed;
    double  *pu, *pv;
    double  *U, *V;
    size_t samples, samples_with_fibres;
} FiberFrame;

typedef struct {
    double p50, p90, p99;
} FiberQuant;

typedef struct {
    size_t nodes_real, nodes_gap, edges;
    double rms_residual_u, rms_residual_v;
    double L_h_p50, L_v_p50;
    double evidence_share;
    double H_old_signed_mean, H_new_signed_mean;
    FiberQuant H_old_abs, H_new_abs;
    double climb_old_range, climb_new_range;
    double climb_old_bow, climb_new_bow;
    double climb_old_end, climb_new_end;
    double climb_old_width, climb_new_width;
    double col_angle_old_p10, col_angle_old_p90, col_angle_new_p10, col_angle_new_p90;
    double rotation_deg;
    FiberQuant nonrigid_px;  double nonrigid_max;
    FiberQuant tile_nonrigid_px, tile_rotation_deg;
    size_t tiles, flipped;
    double area_p1, area_p10, area_p50, area_p90, area_p99;
    double A[4], b[2];
} FiberFrameStats;

typedef struct FiberFrameMapStats {
    size_t real_nodes, evidence_nodes, supported_nodes;
    size_t missing_nodes, invalid_nodes, folded_nodes, cell_sides;
    FiberQuant H_old_abs, H_new_abs, V_old_abs, V_new_abs;
    double determinant_min, determinant_max;
} FiberFrameMapStats;

typedef struct FiberFrameBranchExample {
    int col_a, row_a, col_b, row_b, quarter;
    double selected_deg, residual_deg, normal_deg, coherence, maximum_world_step;
} FiberFrameBranchExample;

typedef struct FiberFrameBranchStats {
    size_t evidence_nodes, neighbor_pairs, unsupported_paths, invalid_frames;
    size_t antipodal_normals, compared_pairs, nearest_quarter[4], switches;
    FiberQuant selected_deg, residual_deg, switch_normal_deg, switch_coherence;
    FiberQuant switch_maximum_world_step;
    size_t examples;
    FiberFrameBranchExample example[8];
} FiberFrameBranchStats;

int FiberFrame_field(Arena_T arena, const float *X, const uint8_t *valid, int H, int W,
                     int x0, int y0, int x1, int y1, FiberVolume_T fv, FiberFrame *ff);

int FiberFrame_solve(Arena_T arena, FiberFrame *ff, FiberFrameStats *st);

int FiberFrame_audit(Arena_T arena, const FiberFrame *ff, FiberFrameStats *st, const char *profile_csv);

int FiberFrame_audit_map(Arena_T arena, const FiberFrame *ff,
                         const FiberMapResult *map, FiberFrameMapStats *stats);

int FiberFrame_audit_branches(Arena_T arena, const FiberFrame *ff,
                              const float *X, const uint8_t *valid, int H, int W,
                              FiberFrameBranchStats *stats);

int FiberFrame_write_directions(Arena_T arena, const FiberFrame *ff, const char *path);

int FiberFrame_inverse(Arena_T arena, const FiberFrame *ff, const FiberFrameStats *st, int threads,
                       int *Hn, int *Wn, double *u_min, double *v_min, float **old_rc);

int FiberFrame_resample(Arena_T arena, const float *X, const uint8_t *valid, int H, int W, int x0, int y0, int x1, int y1,
                        const float *old_rc, int Hn, int Wn, float **Xn, uint8_t **vn, size_t *n_valid);

int FiberFrame_selftest(void);

#endif
