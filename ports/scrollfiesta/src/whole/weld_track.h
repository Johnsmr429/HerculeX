#ifndef WELD_TRACK_INCLUDED
#define WELD_TRACK_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "../common/arena.h"

#define WELD_TRACK_MAGIC   "WTRK"
#define WELD_TRACK_VERSION 1u
#define WTRK_ID_LEN        48

typedef struct {
    uint32_t cube_a, cube_b;
    uint32_t vert_a, vert_b;
    int32_t  gid_a, gid_b;
    float    dr;
    float    dphi;
    float    dist3d;
    uint8_t  seam_axis;
    uint8_t  conf;
    uint8_t  _pad[2];
} WtrkCorr;

typedef struct WeldTrack {
    uint32_t version;
    double   spiral_a, spiral_b, pitch;
    float    axis_point[3];
    size_t   n_cubes;
    char   (*ids)[WTRK_ID_LEN];
    size_t   n_corr;
    WtrkCorr *corr;
    void    *self;
} WeldTrack;

int WeldTrack_write(const WeldTrack *w, const char *path);
int WeldTrack_read(Arena_T arena, const char *path, WeldTrack *out);

int32_t WeldTrack_cube_index(const WeldTrack *w, const char *id);

int WeldTrack_selftest(void);

#endif
