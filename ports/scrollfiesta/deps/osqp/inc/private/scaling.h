#ifndef SCALING_H
#define SCALING_H


#include "osqp.h"
#include "types.h"
#include "lin_alg.h"

#ifdef __cplusplus
extern "C" {
#endif

# if OSQP_EMBEDDED_MODE != 1

OSQPInt scale_data(OSQPSolver* solver);
# endif


OSQPInt unscale_data(OSQPSolver* solver);


OSQPInt unscale_solution(OSQPVectorf*       usolx,
                         OSQPVectorf*       usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace*     work);

#ifdef __cplusplus
}
#endif

#endif
