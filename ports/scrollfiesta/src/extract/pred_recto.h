#ifndef PRED_RECTO_INCLUDED
#define PRED_RECTO_INCLUDED

#include "../common/arena.h"
#include "../common/mesh_types.h"
#include "../remesh/ball_pivot.h"

typedef struct PredRectoField {
    const uint8_t *vol;
    int d, h, w, halo;
    BpaReconGate axis;
} PredRectoField;

typedef struct PredRectoStats {
    size_t tested, supported, outside, ambiguous, missing, broad, reverted;
    double sum_abs, sum_signed, max_abs;
} PredRectoStats;

int PredRecto_target(const PredRectoField *field, const float pos[3],
                     const float normal[3], float inward[3], float *distance,
                     PredRectoStats *stats);
void PredRecto_cloud(const PredRectoField *field, float *verts,
                     const float *normals, size_t nv, PredRectoStats *stats);
void PredRecto_meshes(Arena_T arena, const PredRectoField *field,
                      ComponentMesh *meshes, size_t count, const char *stage);
void PredRecto_report(const char *stage, const PredRectoStats *stats);
int PredRecto_selftest(void);

#endif
