#ifndef VES_CONVEX_QP_INCLUDED
#define VES_CONVEX_QP_INCLUDED
#include <stddef.h>
int ConvexQp_solve(int n,int hn,const int *hr,const int *hc,const double *hv,
    const double *target,int m,int an,const int *ar,const int *ac,const double *av,
    const double *upper,double *result);
int ConvexQp_translation_seed(int n,int hn,const int *hr,const int *hc,const double *hv,
    const double *target,int m,int an,const int *ar,const int *ac,const double *av,
    const double *upper,double *result);
void ConvexQp_time_limit(double seconds);
void ConvexQp_deadline(double deadline);
int ConvexQp_selftest(void);
#endif
