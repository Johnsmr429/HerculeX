#ifndef VESUVIUS_AXIS_TRACK_H
#define VESUVIUS_AXIS_TRACK_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    AXIS_ROW_ABSTAIN = 0,
    AXIS_ROW_LOCK    = 1,
    AXIS_ROW_INTERP  = 2,
    AXIS_ROW_EXTRAP  = 3
} AxisRowStatus;

typedef struct {
    double z0, z1;
    double *fy, *fx, *fny, *fnx, *fa;
    size_t  nf, nf_cap, nf_seen;
    double *vy, *vx;
    size_t  nv, nv_cap, nv_seen;
    uint32_t rng;
} AxisSlab;

typedef struct {
    double z;
    double y, x;
    double y1, x1;
    double y2, x2;
    int    status;
    int    primary_ok, secondary_ok;
    size_t n_in;
    double cond;
    int    sectors;
    double void_ratio;
    double resid_p50;
    double d1;
    size_t ridges;
    double pitch;
    double r_wall;
    double agree;
    double jump;
} AxisRow;

int  AxisSlab_init(AxisSlab *s, double z0, double z1,
                   size_t face_cap, size_t vert_cap, uint32_t seed);
void AxisSlab_dispose(AxisSlab *s);
void AxisSlab_add_face(AxisSlab *s, const float a_zyx[3],
                       const float b_zyx[3], const float c_zyx[3]);
void AxisSlab_add_vertex(AxisSlab *s, const float p_zyx[3]);

int AxisTrack_primary(const AxisSlab *s, double prior_y, double prior_x,
                      AxisRow *row);

int AxisTrack_secondary(const AxisSlab *s, double start_y, double start_x,
                        AxisRow *row);

int AxisTrack_seed(const AxisSlab *s, double *seed_y, double *seed_x,
                   double *cl_y, double *cl_x, size_t *cl_members,
                   double *cl_d1, size_t *cl_ridges, size_t max_clusters,
                   size_t *n_clusters);

int AxisTrack_seed_in(const AxisSlab *s, double win_y, double win_x, double win_r,
                      double *seed_y, double *seed_x,
                      double *cl_y, double *cl_x, size_t *cl_members,
                      double *cl_d1, size_t *cl_ridges, size_t max_clusters,
                      size_t *n_clusters);

size_t AxisTrack_finish(AxisRow *rows, size_t n);

int AxisTrack_write_csv(const char *path, const AxisRow *rows, size_t n,
                        const char *header_lines);

int AxisTrack_sense(const AxisRow *rows, size_t n, double *agree);

extern int AxisTrack_verbose;

int AxisTrack_polar_debug(const AxisSlab *s, double cy, double cx,
                          int r_max, const char *png_path);

int AxisTrack_selftest(void);

#endif
