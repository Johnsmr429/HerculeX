#ifndef SCAFFOLD_INCLUDED
#define SCAFFOLD_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "../common/arena.h"
#include "piece_set.h"

#ifndef SCAFFOLD_2PI
#define SCAFFOLD_2PI 6.283185307179586476925286766559
#endif

typedef struct {
    double spiral_a, spiral_b;
    double pitch;
    int    sense;
    float  axis_point[3];
    float  axis_dir[3];
    char   axis_table[1024];
} ScaffoldCalib;

void Scaffold_calib_default(ScaffoldCalib *c);

int Scaffold_read_calib(const char *placed_dir, ScaffoldCalib *c);

static inline int Scaffold_wrap_index(const ScaffoldCalib *c, double phi)
{
    return (int)floor((double)c->sense * phi / SCAFFOLD_2PI);
}

static inline double Scaffold_unwound(const ScaffoldCalib *c, double phi)
{
    return (double)c->sense * phi;
}

typedef struct {
    size_t nv_used;
    size_t nf_used;
    int    k_min, k_max;
    int    k_extent;
    size_t face_spread_ge_pi;
    double face_spread_frac;
    double p50_face_spread;
    double p99_face_spread;
    size_t n_groups;
    size_t group_spread_ge_pi;
} ScaffoldKStats;

int Scaffold_kstats(Arena_T arena, const PieceSet *ps, const ScaffoldCalib *c,
                    ScaffoldKStats *out, const char *hist_path);

int Scaffold_selftest(void);

#endif
