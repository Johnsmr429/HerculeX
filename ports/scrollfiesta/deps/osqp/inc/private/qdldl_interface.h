#ifndef QDLDL_INTERFACE_H
#define QDLDL_INTERFACE_H


#include "osqp.h"
#include "types.h"
#include "qdldl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qdldl qdldl_solver;

struct qdldl {
    enum osqp_linsys_solver_type type;

       const char* (*name)(struct qdldl* s);

    OSQPInt (*solve)(struct qdldl*       self,
                            OSQPVectorf* b,
                            OSQPInt      admm_iter);

    void (*update_settings)(struct qdldl*        self,
                            const  OSQPSettings* settings);

    void (*warm_start)(struct qdldl*       self,
                       const  OSQPVectorf* x);

#ifndef OSQP_EMBEDDED_MODE
    OSQPInt (*adjoint_derivative)(qdldl_solver**     s,
                                  const OSQPMatrix*  P,
                                  const OSQPMatrix*  G,
                                  const OSQPMatrix*  A_eq,
                                  const OSQPMatrix*  GDiagLambda,
                                  const OSQPVectorf* slacks,
                                        OSQPVectorf* rhs);

    void (*free)(struct qdldl* self);
#endif

   #if OSQP_EMBEDDED_MODE != 1
    OSQPInt (*update_matrices)(struct qdldl*      self,
                               const  OSQPMatrix* P,
                               const  OSQPInt*    Px_new_idx,
                                      OSQPInt     P_new_n,
                               const  OSQPMatrix* A,
                               const  OSQPInt*    Ax_new_idx,
                                      OSQPInt     A_new_n);

    OSQPInt (*update_rho_vec)(struct qdldl*       self,
                              const  OSQPVectorf* rho_vec,
                                     OSQPFloat    rho_sc);
#endif

    OSQPInt nthreads;


       OSQPCscMatrix* L;
    OSQPFloat*     Dinv;
    OSQPInt*       P;
    OSQPFloat*     bp;
    OSQPFloat*     sol;
    OSQPFloat*     rho_inv_vec;
    OSQPFloat      sigma;
    OSQPFloat      rho_inv;
#ifndef OSQP_EMBEDDED_MODE
    OSQPInt        polishing;
#endif
    OSQPInt        n;
    OSQPInt        m;


#if OSQP_EMBEDDED_MODE != 1
       OSQPCscMatrix* KKT;
    OSQPInt*       PtoKKT;
    OSQPInt*       AtoKKT;
    OSQPInt*       rhotoKKT;
       QDLDL_float* D;
    QDLDL_int*   etree;
    QDLDL_int*   Lnz;
    QDLDL_int*   iwork;
    QDLDL_bool*  bwork;
    QDLDL_float* fwork;

    OSQPCscMatrix* adj;
#endif

   };



OSQPInt init_linsys_solver_qdldl(qdldl_solver**      sp,
                                 const OSQPMatrix*   P,
                                 const OSQPMatrix*   A,
                                 const OSQPVectorf*  rho_vec,
                                 const OSQPSettings* settings,
                                 OSQPInt             polishing);

const char* name_qdldl(qdldl_solver* s);

OSQPInt solve_linsys_qdldl(qdldl_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter);


void update_settings_linsys_solver_qdldl(qdldl_solver*       s,
                                         const OSQPSettings* settings);

void warm_start_linsys_solver_qdldl(qdldl_solver*      s,
                                    const OSQPVectorf* x);


#if OSQP_EMBEDDED_MODE != 1
OSQPInt update_linsys_solver_matrices_qdldl(qdldl_solver*     s,
                                            const OSQPMatrix* P,
                                            const OSQPInt*    Px_new_idx,
                                            OSQPInt           P_new_n,
                                            const OSQPMatrix* A,
                                            const OSQPInt*    Ax_new_idx,
                                            OSQPInt           A_new_n);




OSQPInt update_linsys_solver_rho_vec_qdldl(qdldl_solver*      s,
                                           const OSQPVectorf* rho_vec,
                                           OSQPFloat          rho_sc);

#endif

#ifndef OSQP_EMBEDDED_MODE
void free_linsys_solver_qdldl(qdldl_solver* s);

OSQPInt adjoint_derivative_qdldl(qdldl_solver**     s,
                                 const OSQPMatrix*  P,
                                 const OSQPMatrix*  G,
                                 const OSQPMatrix*  A_eq,
                                 const OSQPMatrix*  GDiagLambda,
                                 const OSQPVectorf* slacks,
                                       OSQPVectorf* rhs);

#endif

#ifdef __cplusplus
}
#endif

#endif
