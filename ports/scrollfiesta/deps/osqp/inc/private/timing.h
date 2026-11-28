#ifndef TIMING_H_
#define TIMING_H_

#include "osqp_configure.h"
#include "types.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifdef OSQP_ENABLE_PROFILING

OSQPTimer* OSQPTimer_new();

void OSQPTimer_free(OSQPTimer* t);

void osqp_tic(OSQPTimer* t);

OSQPFloat osqp_toc(OSQPTimer* t);

#endif

#ifdef __cplusplus
}
#endif

#endif
