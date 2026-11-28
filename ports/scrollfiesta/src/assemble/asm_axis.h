#ifndef ASM_AXIS_INCLUDED
#define ASM_AXIS_INCLUDED

#include <stddef.h>

#include "../common/arena.h"

typedef struct AsmAxis {
    size_t  n;
    size_t  n_rows;
    double  fit_rms;
    double  r_curv_min;
    double *p;
    double *s;
    double *t;
    int     is_line;
} AsmAxis;

AsmAxis *AsmAxis_load(Arena_T arena, const char *path);

AsmAxis *AsmAxis_from_rows(Arena_T arena, const double *rows, const double *row_w, size_t n);

AsmAxis *AsmAxis_line(Arena_T arena, const double point[3], const double dir[3], double z0, double z1);

void AsmAxis_project(const AsmAxis *a, const double p[3], double *s, double *r, double dir[3], double foot[3]);

typedef struct AsmAxisProjector AsmAxisProjector;
AsmAxisProjector *AsmAxis_projector(Arena_T arena, const AsmAxis *a);
void AsmAxis_project_foot(const AsmAxisProjector *index, const double p[3], double foot[3]);

void AsmAxis_frame(const AsmAxis *a, const double p[3], double out[3]);

void AsmAxis_mean_dir(const AsmAxis *a, double z0, double z1, double dir[3]);

int AsmAxis_selftest(void);

#endif
