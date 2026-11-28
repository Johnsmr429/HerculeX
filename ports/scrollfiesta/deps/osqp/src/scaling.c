#include "scaling.h"

#if OSQP_EMBEDDED_MODE != 1



OSQPFloat limit_scaling_scalar(OSQPFloat v) {
    v = v < OSQP_MIN_SCALING ? 1.0 : v;
    v = v > OSQP_MAX_SCALING ? OSQP_MAX_SCALING : v;
    return v;
}

void limit_scaling_vector(OSQPVectorf* v) {
  OSQPVectorf_set_scalar_if_lt(v,v,OSQP_MIN_SCALING,1.0);
  OSQPVectorf_set_scalar_if_gt(v,v,OSQP_MAX_SCALING,OSQP_MAX_SCALING);
}

void compute_inf_norm_cols_KKT(const OSQPMatrix*  P,
                               const OSQPMatrix*  A,
                                     OSQPVectorf* D,
                                     OSQPVectorf* D_temp_A,
                                     OSQPVectorf* E) {
     OSQPMatrix_col_norm_inf(P,D);
  OSQPMatrix_col_norm_inf(A, D_temp_A);
  OSQPVectorf_ew_max_vec(D, D_temp_A, D);

     OSQPMatrix_row_norm_inf(A,E);
}

OSQPInt scale_data(OSQPSolver* solver) {

  OSQPInt   i;
  OSQPInt   n;
  OSQPFloat c_temp;
  OSQPFloat inf_norm_q;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  n = work->data->n;

   work->scaling->c = 1.0;
  OSQPVectorf_set_scalar(work->scaling->D,    1.);
  OSQPVectorf_set_scalar(work->scaling->Dinv, 1.);
  OSQPVectorf_set_scalar(work->scaling->E,    1.);
  OSQPVectorf_set_scalar(work->scaling->Einv, 1.);


  for (i = 0; i < settings->scaling; i++) {

       compute_inf_norm_cols_KKT(work->data->P, work->data->A,
                              work->D_temp,
                              work->D_temp_A,
                              work->E_temp);

       limit_scaling_vector(work->D_temp);
    limit_scaling_vector(work->E_temp);

       OSQPVectorf_ew_sqrt(work->D_temp);
    OSQPVectorf_ew_sqrt(work->E_temp);

       OSQPVectorf_ew_reciprocal(work->D_temp, work->D_temp);
    OSQPVectorf_ew_reciprocal(work->E_temp, work->E_temp);

          OSQPMatrix_lmult_diag(work->data->P,work->D_temp);
    OSQPMatrix_rmult_diag(work->data->P,work->D_temp);

       OSQPMatrix_lmult_diag(work->data->A,work->E_temp);
    OSQPMatrix_rmult_diag(work->data->A,work->D_temp);

       OSQPVectorf_ew_prod(work->data->q, work->data->q, work->D_temp);

       OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, work->D_temp);
    OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);


       OSQPMatrix_col_norm_inf(work->data->P, work->D_temp);
    c_temp = OSQPVectorf_norm_1(work->D_temp);
    c_temp = c_temp / n;

       inf_norm_q = OSQPVectorf_norm_inf(work->data->q);

          inf_norm_q = limit_scaling_scalar(inf_norm_q);

       c_temp = c_max(c_temp, inf_norm_q);

       c_temp = limit_scaling_scalar(c_temp);

       c_temp = 1. / c_temp;

       OSQPMatrix_mult_scalar(work->data->P,c_temp);

       OSQPVectorf_mult_scalar(work->data->q, c_temp);

       work->scaling->c *= c_temp;
  }


   work->scaling->cinv = 1. / work->scaling->c;
  OSQPVectorf_ew_reciprocal(work->scaling->Dinv, work->scaling->D);
  OSQPVectorf_ew_reciprocal(work->scaling->Einv, work->scaling->E);


   OSQPVectorf_ew_prod(work->data->l, work->data->l, work->scaling->E);
  OSQPVectorf_ew_prod(work->data->u, work->data->u, work->scaling->E);

  return 0;
}

#endif


OSQPInt unscale_data(OSQPSolver* solver) {

  OSQPWorkspace* work     = solver->work;

   OSQPMatrix_mult_scalar(work->data->P, work->scaling->cinv);
  OSQPMatrix_lmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPMatrix_rmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPVectorf_mult_scalar(work->data->q,work->scaling->cinv);
  OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->Dinv);

   OSQPMatrix_lmult_diag(work->data->A,work->scaling->Einv);
  OSQPMatrix_rmult_diag(work->data->A,work->scaling->Dinv);

  OSQPVectorf_ew_prod(work->data->l,
                      work->data->l,
                      work->scaling->Einv);
  OSQPVectorf_ew_prod(work->data->u,
                      work->data->u,
                      work->scaling->Einv);

  return 0;
}

OSQPInt unscale_solution(OSQPVectorf*       usolx,
                         OSQPVectorf*       usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace*     work) {

   OSQPVectorf_ew_prod(usolx,solx,work->scaling->D);

   OSQPVectorf_ew_prod(usoly,soly,work->scaling->E);

  OSQPVectorf_mult_scalar(usoly,work->scaling->cinv);
  return 0;
}
