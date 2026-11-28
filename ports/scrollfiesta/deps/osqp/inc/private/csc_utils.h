#ifndef CSC_UTILS_H
#define CSC_UTILS_H

#include "osqp_configure.h"
#include "osqp_api_types.h"
#include "glob_opts.h"

#ifdef __cplusplus
extern "C" {
#endif


OSQPInt csc_is_eq(OSQPCscMatrix* A, OSQPCscMatrix* B, OSQPFloat tol);


#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* csc_spalloc(OSQPInt m,
                           OSQPInt n,
                           OSQPInt nzmax,
                           OSQPInt values,
                           OSQPInt triplet);


void csc_spfree(OSQPCscMatrix* A);

OSQPCscMatrix* csc_submatrix_byrows(const OSQPCscMatrix* A,
                                    OSQPInt*             rows);


OSQPCscMatrix* csc_done(OSQPCscMatrix* C,
                        void*          w,
                        void*          x,
                        OSQPInt        ok);
#endif


#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* csc_copy(const OSQPCscMatrix* A);


OSQPFloat* csc_to_dns(OSQPCscMatrix* M);

#endif


#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* triplet_to_csc(const OSQPCscMatrix* T,
                                    OSQPInt*       TtoC);


OSQPCscMatrix* triplet_to_csr(const OSQPCscMatrix* T,
                                    OSQPInt*       TtoC);



OSQPCscMatrix* triu_to_csc(OSQPCscMatrix* M);


OSQPCscMatrix* vstack(OSQPCscMatrix* A,
                      OSQPCscMatrix* B);

#endif



void csc_extract_diag(const OSQPCscMatrix* A,
                            OSQPFloat*     d);

#ifndef OSQP_EMBEDDED_MODE

OSQPInt* csc_pinv(const OSQPInt* p,
                        OSQPInt  n);

OSQPCscMatrix* csc_symperm(const OSQPCscMatrix* A,
                           const OSQPInt*       pinv,
                                 OSQPInt*       AtoC,
                                 OSQPInt        values);

#endif

#ifdef __cplusplus
}
#endif

#endif
