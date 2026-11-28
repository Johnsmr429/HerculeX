#ifndef TYPES_H
#define TYPES_H

#include "osqp.h"

#include "algebra_matrix.h"
#include "algebra_vector.h"
#include "glob_opts.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct linsys_solver LinSysSolver;

typedef struct OSQPTimer_ OSQPTimer;

typedef struct {
  OSQPFloat    c;
  OSQPVectorf* D;
  OSQPVectorf* E;
  OSQPFloat    cinv;
  OSQPVectorf* Dinv;
  OSQPVectorf* Einv;
} OSQPScaling;




# ifndef OSQP_EMBEDDED_MODE


typedef struct {
  OSQPMatrix*  Ared;
  OSQPInt      n_active;
  OSQPVectori* active_flags;
  OSQPVectorf* x;
  OSQPVectorf* z;
  OSQPVectorf* y;
  OSQPFloat    obj_val;
  OSQPFloat    dual_obj_val;
  OSQPFloat    duality_gap;
  OSQPFloat    prim_res;
  OSQPFloat    dual_res;
} OSQPPolish;
# endif



typedef struct {
  OSQPInt      n;
  OSQPInt      m;
  OSQPMatrix*  P;
  OSQPMatrix*  A;
  OSQPVectorf* q;
  OSQPVectorf* l;
  OSQPVectorf* u;
} OSQPData;

typedef struct {
    OSQPInt n_ineq_l;
    OSQPInt n_ineq_u;
    OSQPInt n_eq;
    OSQPVectorf *y_l;
    OSQPVectorf *y_u;
    OSQPVectorf *ryl;
    OSQPVectorf *ryu;
    OSQPVectorf *rhs;
                      } OSQPDerivativeData;


struct OSQPWorkspace_ {
   OSQPData* data;

   LinSysSolver* linsys_solver;

# ifndef OSQP_EMBEDDED_MODE
   OSQPPolish* pol;
# endif

   OSQPVectorf* rho_vec;
  OSQPVectorf* rho_inv_vec;


# if OSQP_EMBEDDED_MODE != 1
  OSQPVectori* constr_type;
# endif

   OSQPVectorf* x;
  OSQPVectorf* y;
  OSQPVectorf* z;
  OSQPVectorf* xz_tilde;
  OSQPVectorf* xtilde_view;
  OSQPVectorf* ztilde_view;

  OSQPVectorf* x_prev;
  OSQPVectorf* z_prev;

   OSQPVectorf* Ax;
  OSQPVectorf* Px;
  OSQPVectorf* Aty;


   OSQPFloat    xtPx;
  OSQPFloat    qtx;
  OSQPFloat    SC;
  OSQPFloat    scaled_dual_gap;


   OSQPVectorf* delta_y;
  OSQPVectorf* Atdelta_y;


   OSQPVectorf* delta_x;
  OSQPVectorf* Pdelta_x;
  OSQPVectorf* Adelta_x;


 #if OSQP_EMBEDDED_MODE != 1
  OSQPVectorf* D_temp;
  OSQPVectorf* D_temp_A;
  OSQPVectorf* E_temp;
#endif

   OSQPScaling* scaling;

    OSQPFloat scaled_prim_res;
  OSQPFloat scaled_dual_res;

   OSQPFloat rho_inv;

# ifdef OSQP_ENABLE_PROFILING
  OSQPTimer* timer;

   OSQPInt first_run;

   OSQPInt clear_update_time;

   OSQPInt rho_update_from_solve;
# endif

# ifdef OSQP_ENABLE_PRINTING
  OSQPInt summary_printed;
# endif

# ifdef OSQP_ENABLE_DERIVATIVES
  OSQPDerivativeData *derivative_data;
# endif

   OSQPInt rho_updated;

   OSQPFloat last_rel_kkt;
};



struct linsys_solver {
  enum osqp_linsys_solver_type type;

  const char* (*name)(LinSysSolver* self);

  OSQPInt (*solve)(LinSysSolver* self,
                   OSQPVectorf*  b,
                   OSQPInt       admm_iter);

  void (*update_settings)(LinSysSolver*       self,
                          const OSQPSettings* settings);

  void (*warm_start)(LinSysSolver*      self,
                     const OSQPVectorf* x);

# ifndef OSQP_EMBEDDED_MODE
  OSQPInt (*adjoint_derivative)(LinSysSolver* self);

  void (*free)(LinSysSolver* self);
# endif

# if OSQP_EMBEDDED_MODE != 1
  OSQPInt (*update_matrices)(LinSysSolver*     self,
                             const OSQPMatrix* P,
                             const OSQPInt*    Px_new_idx,
                             OSQPInt           P_new_n,
                             const OSQPMatrix* A,
                             const OSQPInt*    Ax_new_idx,
                             OSQPInt           A_new_n);

  OSQPInt (*update_rho_vec)(LinSysSolver*      self,
                            const OSQPVectorf* rho_vec,
                            OSQPFloat          rho_sc);
# endif

  OSQPInt nthreads;
};

#ifdef __cplusplus
}
#endif

#endif
