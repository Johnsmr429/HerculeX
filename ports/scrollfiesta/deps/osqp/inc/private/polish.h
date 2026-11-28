#ifndef POLISH_H
#define POLISH_H


#include "osqp.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSQPInt polish(OSQPSolver* solver);

#ifdef __cplusplus
}
#endif

#endif
