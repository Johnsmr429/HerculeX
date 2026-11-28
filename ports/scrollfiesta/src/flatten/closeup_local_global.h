#ifndef CLOSEUP_LOCAL_GLOBAL_INCLUDED
#define CLOSEUP_LOCAL_GLOBAL_INCLUDED

#include "closeup_angular_metric_angular.h"

int CloseupLocalGlobal_run(Arena_T arena,const float *xyz,size_t nv,
    const int32_t *faces,size_t nf,const double *seed,const PMAOptions *options,
    double *output,PMAStats *stats);
int CloseupLocalGlobal_selftest(void);

#endif
