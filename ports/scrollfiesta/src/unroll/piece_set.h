#ifndef PIECE_SET_INCLUDED
#define PIECE_SET_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

typedef struct {
    float   *verts;
    float   *uv;
    float   *phi;
    float   *normals;
    int32_t *faces;
    int32_t *face_cube;
    int32_t *gid;
    size_t   nv, nf;
    char   (*ids)[48];
    size_t  *cube_voff;
    long   (*cube_org)[3];
    size_t   n_cubes;
    double   u_min, u_max, v_min, v_max;
} PieceSet;

int PieceSet_build(Arena_T arena, const char *placed_dir, PieceSet *out);

int PieceSet_build_z(Arena_T arena, const char *placed_dir,
                     long z_lo, long z_hi, PieceSet *out);

size_t PieceSet_apply_facekeep(PieceSet *ps, const uint8_t *keep);

void PieceSet_refresh_normals(PieceSet *ps);

int PieceSet_selftest(void);

#endif
