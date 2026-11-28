#ifndef KKT_H
#define KKT_H


#include "osqp.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OSQP_EMBEDDED_MODE

#include "csc_utils.h"

 OSQPCscMatrix* form_KKT(OSQPCscMatrix* P,
                         OSQPCscMatrix* A,
                         OSQPInt        format,
                         OSQPFloat      param1,
                         OSQPFloat*     param2,
                         OSQPFloat      param2_sc,
                         OSQPInt*       PtoKKT,
                         OSQPInt*       AtoKKT,
                         OSQPInt*       param2toKKT);
# endif


# if OSQP_EMBEDDED_MODE != 1

 void update_KKT_P(OSQPCscMatrix* KKT,
                   OSQPCscMatrix* P,
                   const OSQPInt* Px_new_idx,
                   OSQPInt        P_new_n,
                   OSQPInt*       PtoKKT,
                   OSQPFloat      param1,
                   OSQPInt        format);


 void update_KKT_A(OSQPCscMatrix* KKT,
                   OSQPCscMatrix* A,
                   const OSQPInt* Ax_new_idx,
                   OSQPInt        A_new_n,
                   OSQPInt*       AtoKKT);


void update_KKT_param2(OSQPCscMatrix* KKT,
                       OSQPFloat*     param2,
                       OSQPFloat      param2_sc,
                       OSQPInt*       param2toKKT,
                       OSQPInt        m);

# endif

#ifdef __cplusplus
}
#endif

#endif
