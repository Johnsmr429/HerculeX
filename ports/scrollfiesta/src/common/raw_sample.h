#ifndef RAW_SAMPLE_INCLUDED
#define RAW_SAMPLE_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

typedef struct CubeTable {
    Arena_T   arena;
    char      dir[2048];
    long      chunk;
    long      cz0, cy0, cx0;
    long      nz, ny, nx;
    uint8_t **slot;
    int       n_loaded, n_missing;
    size_t    n_outside;
    size_t    n_untouched;
    int       is_zarr;
    int       support_only;
    long      shape[3];
} CubeTable;

int cubetable_init(CubeTable *ct, Arena_T arena, const char *dir,
                   long chunk, const float *verts, size_t nv, double pad);

void   cubetable_resident(int on);
size_t cubetable_resident_bytes(void);
size_t cubetable_resident_chunks(void);
void   cubetable_resident_reset(void);

int cube_fetch(CubeTable *ct, long iz, long iy, long ix);

int cube_supported(CubeTable *ct, long iz, long iy, long ix);

int cubetable_prewarm_support(CubeTable *ct);

int cubetable_prewarm_all(CubeTable *ct);

int cubetable_prewarm_points(CubeTable *ct, const float *z, const float *y, const float *x, size_t n, double pad);

size_t cubetable_expected_chunks(const CubeTable *ct);
int cubetable_is_complete(const CubeTable *ct);

size_t cubetable_missing_reached(const CubeTable *ct, const float *verts,
                                 const int32_t *faces, size_t nf, double pad,
                                 uint8_t *reached);

size_t cubetable_faces_reaching(const CubeTable *ct, const float *verts,
                                const int32_t *faces, size_t nf, double pad,
                                const uint8_t *slots, uint8_t *face_mask);

double sample_trilinear(CubeTable *ct, double z, double y, double x);

typedef struct RawTangentTensor {
    double energy;
    double rms_gradient;
    double coherence;
    double axis_alignment;
    double normal_asymmetry;
    double ridge_center;
    double quality;
} RawTangentTensor;

int sample_tangent_tensor(CubeTable *ct, const double p[3],
                          const double tu[3], const double tv[3],
                          double radius, RawTangentTensor *out);

double sample_vertex(CubeTable *ct, const float *p, const float *n,
                     double range, int nsteps);

#endif
