#ifndef OVERLAP_QUILT_INCLUDED
#define OVERLAP_QUILT_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct QuiltOpts {
    float  axis_point[3];
    float  axis_dir[3];
    const char *raw_dir;
    long   chunk;
    double radius_gate;
    double cell;
    double tex_du, tex_dv;
    double normal_range;
    int    normal_samples;
    double spiral_a, spiral_b, pitch;
    int    relocate;
    int    cross_component_only;
    int    verbose;
} QuiltOpts;

void QuiltOpts_default(QuiltOpts *o);

typedef struct QuiltResult {
    float   *uv;
    uint8_t *face_keep;
    uint8_t *face_dec;
    int32_t *face_region;
    int32_t *face_layer;
    double  *face_energy;

    size_t n_overlap_faces;
    size_t n_regions;
    size_t n_layers_total;
    size_t max_region_faces;
    size_t n_kept, n_relocated, n_dropped;
    size_t multi_cells_before;
    size_t multi_cells_after;
    double energy_mean;
    int    have_texture;
} QuiltResult;

int Quilt_run(Arena_T arena,
              const float *verts, size_t nv,
              const int32_t *faces, size_t nf,
              const float *uv,
              const QuiltOpts *opts, QuiltResult *out);

struct CubeTable;

typedef struct QuiltAdjEdge { int32_t fa, fb; float uvlen; } QuiltAdjEdge;

QuiltAdjEdge *Quilt_build_adjacency(const int32_t *faces, size_t nf,
                                    const float *uv, size_t *out_n);

double Quilt_point_radius(const float *p, const float ap[3],
                          const double e1[3], const double e2[3]);

void Quilt_raster_faces(const int32_t *flist, size_t nfl,
                        const float *verts, const float *uv, const float *nrm,
                        const int32_t *faces,
                        struct CubeTable *ct, double range, int nsteps,
                        double u0, double v0, double du, double dv,
                        int gw, int gh, float *img, uint8_t *cov);

double Quilt_boundary_energy(const float *vL, const uint8_t *covL,
                             const float *vA, const uint8_t *covA,
                             int gw, int gh);

int Quilt_selftest(void);

#endif
