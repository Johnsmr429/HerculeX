#include "polish.h"
#include "lin_alg.h"
#include "osqp_api_constants.h"
#include "printing.h"
#include "util.h"
#include "auxil.h"
#include "error.h"
#include "timing.h"

static OSQPInt form_Ared(OSQPWorkspace* work){

  OSQPInt j, n_active;
  OSQPInt m = work->data->m;

  OSQPInt* active_flags = OSQP_NULL;
  OSQPFloat* z = OSQP_NULL;
  OSQPFloat* y = OSQP_NULL;
  OSQPFloat* u = OSQP_NULL;
  OSQPFloat* l = OSQP_NULL;

   active_flags = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
  z = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  y = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  l = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  u = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));

   if (!active_flags || !z || !y || !l || !u) {
    c_free(active_flags);
    c_free(z);
    c_free(y);
    c_free(l);
    c_free(u);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

   OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(z, work->z);
  OSQPVectorf_to_raw(y, work->y);
  OSQPVectorf_to_raw(l, work->data->l);
  OSQPVectorf_to_raw(u, work->data->u);

   n_active = 0;


  for (j = 0; j < work->data->m; j++) {

    if ((z[j] - l[j] < -y[j]) || (l[j] == u[j]) ) {
      active_flags[j] = -1;
      n_active++;
    }
    else if (u[j] - z[j] < y[j]) {
      active_flags[j] = +1;
      n_active++;
    }
    else{
      active_flags[j] = 0;
    }
  }

   OSQPVectori_from_raw(work->pol->active_flags, active_flags);

   work->pol->n_active = n_active;

   work->pol->Ared = OSQPMatrix_submatrix_byrows(work->data->A, work->pol->active_flags);

   c_free(active_flags);
  c_free(z);
  c_free(y);
  c_free(l);
  c_free(u);

  if (!work->pol->Ared)
    return osqp_error(OSQP_MEM_ALLOC_ERROR);;

  return OSQP_NO_ERROR;
}

static OSQPInt form_rhs_red(OSQPWorkspace* work, OSQPVectorf* rhs) {

  OSQPInt j, counter;
  OSQPInt n = work->data->n;
  OSQPInt m = work->data->m;
  OSQPInt n_plus_mred = OSQPVectorf_length(rhs);

  OSQPInt *active_flags = OSQP_NULL;
  OSQPFloat* rhsv = OSQP_NULL;
  OSQPFloat* q = OSQP_NULL;
  OSQPFloat* l = OSQP_NULL;
  OSQPFloat* u = OSQP_NULL;

   active_flags = (OSQPInt *)   c_malloc(m           * sizeof(OSQPInt));
  rhsv         = (OSQPFloat *) c_malloc(n_plus_mred * sizeof(OSQPFloat));
  q            = (OSQPFloat *) c_malloc(n           * sizeof(OSQPFloat));
  l            = (OSQPFloat *) c_malloc(m           * sizeof(OSQPFloat));
  u            = (OSQPFloat *) c_malloc(m           * sizeof(OSQPFloat));

  if (!active_flags || !rhsv || !q || !l || !u) {
    c_free(active_flags);
    c_free(rhsv);
    c_free(q);
    c_free(l);
    c_free(u);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);;
  }

   OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(rhsv, rhs);
  OSQPVectorf_to_raw(q, work->data->q);
  OSQPVectorf_to_raw(l, work->data->l);
  OSQPVectorf_to_raw(u, work->data->u);

  for(j = 0; j < work->data->n; j++){
    rhsv[j] = -q[j];
  }

  counter = 0;

  for (j = 0; j < work->data->m; j++) {
    if(active_flags[j] == -1){
       rhsv[work->data->n + counter] = l[j];
       counter++;
    }
    else if(active_flags[j] == 1){
       rhsv[work->data->n + counter] = u[j];
       counter++;
    }
  }

   OSQPVectorf_from_raw(rhs, rhsv);

   c_free(active_flags);
  c_free(rhsv);
  c_free(q);
  c_free(l);
  c_free(u);

  return OSQP_NO_ERROR;
}

static OSQPInt iterative_refinement(OSQPSolver*   solver,
                                    LinSysSolver* p,
                                    OSQPVectorf*  z,
                                    OSQPVectorf*  b) {
  OSQPInt i, mred;
  OSQPVectorf *rhs, *rhs1, *rhs2;
  OSQPVectorf *z1, *z2;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (settings->polish_refine_iter > 0) {
    mred = OSQPMatrix_get_m(work->pol->Ared);

       rhs = OSQPVectorf_malloc(work->data->n + mred);

       rhs1 = OSQPVectorf_view(rhs,0,work->data->n);
    rhs2 = OSQPVectorf_view(rhs,work->data->n,mred);
    z1   = OSQPVectorf_view(z,0,work->data->n);
    z2   = OSQPVectorf_view(z,work->data->n,mred);

    if (!rhs || !rhs1 || !rhs2 || !z1 || !z2) {
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    }

    for (i = 0; i < settings->polish_refine_iter; i++) {

           OSQPVectorf_copy(rhs,b);

                OSQPMatrix_Axpy(work->data->P, z1, rhs1, -1.0, 1.0);

           OSQPMatrix_Atxpy(work->pol->Ared, z2, rhs1, -1.0, 1.0);

                OSQPMatrix_Axpy(work->pol->Ared, z1, rhs2, -1.0, 1.0);

           p->solve(p, rhs, 1);

           OSQPVectorf_plus(z,z,rhs);
    }

    OSQPVectorf_free(rhs);
    OSQPVectorf_view_free(rhs1);
    OSQPVectorf_view_free(rhs2);
    OSQPVectorf_view_free(z1);
    OSQPVectorf_view_free(z2);
  }
  return 0;
}

static OSQPInt get_ypol_from_yred(OSQPWorkspace* work, OSQPVectorf* yred_vf) {

  OSQPInt j, counter;
  OSQPInt m = work->data->m;
  OSQPInt mred = OSQPVectorf_length(yred_vf);

  OSQPInt *active_flags = OSQP_NULL;
  OSQPFloat* y = OSQP_NULL;
  OSQPFloat* yred = OSQP_NULL;

   active_flags = (OSQPInt *)   c_malloc(m    * sizeof(OSQPInt));
  y            = (OSQPFloat *) c_malloc(m    * sizeof(OSQPFloat));
  yred         = (OSQPFloat *) c_malloc(mred * sizeof(OSQPFloat));

  if (!active_flags || !y || !yred) {
       c_free(active_flags);
    c_free(y);
    c_free(yred);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);;
  }

   OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(y, work->y);
  OSQPVectorf_to_raw(yred, yred_vf);

   if (work->pol->n_active == 0) {
    OSQPVectorf_set_scalar(work->pol->y, 0.);

       c_free(active_flags);
    c_free(y);
    c_free(yred);

    return OSQP_NO_ERROR;
  }

  counter = 0;

  for (j = 0; j < work->data->m; j++) {

    if (active_flags[j] == 0) {
      y[j] = 0;
    }
    else {
      y[j] = yred[counter];
      counter++;
    }
  }

   OSQPVectorf_from_raw(work->pol->y, y);

   c_free(active_flags);
  c_free(y);
  c_free(yred);

  return OSQP_NO_ERROR;
}

OSQPInt polish(OSQPSolver* solver) {

  OSQPInt polish_successful = 0;
  OSQPInt exitflag = 0;

  LinSysSolver* plsh = OSQP_NULL;
  OSQPVectorf*  rhs_red = OSQP_NULL;
  OSQPVectorf*  pol_sol = OSQP_NULL;
  OSQPVectorf*  pol_sol_xview = OSQP_NULL;
  OSQPVectorf*  pol_sol_yview = OSQP_NULL;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->timer);
#endif

   exitflag = form_Ared(work);

  if (exitflag) {
       info->status_polish = OSQP_POLISH_FAILED;
    return exitflag;
  } else if (work->pol->n_active == 0) {
       c_print("Polishing not needed - no active set detected at optimal point\n");
    info->status_polish = OSQP_POLISH_NO_ACTIVE_SET_FOUND;

       OSQPMatrix_free(work->pol->Ared);

    return OSQP_NO_ERROR;
  }

   exitflag = osqp_algebra_init_linsys_solver(&plsh, work->data->P, work->pol->Ared,
                                             OSQP_NULL, settings, OSQP_NULL, OSQP_NULL, 1);

  if (exitflag) {
       info->status_polish = OSQP_POLISH_LINSYS_ERROR;

       OSQPMatrix_free(work->pol->Ared);

    return exitflag;
  }

   rhs_red = OSQPVectorf_malloc(work->data->n + work->pol->n_active);

  if (!rhs_red) {
       info->status_polish = OSQP_POLISH_FAILED;

       OSQPMatrix_free(work->pol->Ared);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  exitflag = form_rhs_red(work, rhs_red);

  if (exitflag) {
       info->status_polish = OSQP_POLISH_FAILED;

       OSQPMatrix_free(work->pol->Ared);

    return exitflag;
  }

  pol_sol = OSQPVectorf_copy_new(rhs_red);

  if (!pol_sol) {
       info->status_polish = OSQP_POLISH_FAILED;

       OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_free(rhs_red);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

  pol_sol_xview = OSQPVectorf_view(pol_sol,0,work->data->n);
  pol_sol_yview = OSQPVectorf_view(pol_sol,work->data->n, work->pol->n_active);

  if (!pol_sol_xview || !pol_sol_yview) {

       info->status_polish = OSQP_POLISH_FAILED;

       OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);

    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }

   plsh->warm_start(plsh, work->x);

   plsh->solve(plsh, pol_sol, 1);

   exitflag = iterative_refinement(solver, plsh, pol_sol, rhs_red);

  if (exitflag) {
       info->status_polish = OSQP_POLISH_FAILED;

       OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);

    return exitflag;
  }

   OSQPVectorf_copy(work->pol->x, pol_sol_xview);
  OSQPMatrix_Axpy(work->data->A, work->pol->x, work->pol->z, 1.0, 0.0);
  get_ypol_from_yred(work, pol_sol_yview);

    OSQPVectorf_plus(work->pol->y, work->pol->y, work->pol->z);
  OSQPVectorf_ew_bound_vec(work->pol->z, work->pol->y, work->data->l, work->data->u);
  OSQPVectorf_minus(work->pol->y, work->pol->y, work->pol->z);

   update_info(solver, 0, 1);

   polish_successful = (work->pol->prim_res < info->prim_res &&
                       work->pol->dual_res < info->dual_res) ||
                                                                                                                                                            (work->pol->prim_res < info->prim_res &&
                       info->dual_res < 1e-10) ||
                                                                                                                                                                                                                               (work->pol->dual_res < info->dual_res &&
                       info->prim_res < 1e-10);

  if (polish_successful) {
       info->obj_val       = work->pol->obj_val;
    info->dual_obj_val  = work->pol->dual_obj_val;
    info->duality_gap   = work->pol->duality_gap;
    info->prim_res      = work->pol->prim_res;
    info->dual_res      = work->pol->dual_res;
    info->status_polish = OSQP_POLISH_SUCCESS;

          OSQPVectorf_copy(work->x, work->pol->x);
    OSQPVectorf_copy(work->z, work->pol->z);
    OSQPVectorf_copy(work->y, work->pol->y);

   #ifdef OSQP_ENABLE_PRINTING

    if (settings->verbose) print_polish(solver);
#endif
  } else {
    info->status_polish = OSQP_POLISH_FAILED;

        }

   plsh->free(plsh);

   OSQPMatrix_free(work->pol->Ared);
  OSQPVectorf_free(rhs_red);
  OSQPVectorf_free(pol_sol);
  OSQPVectorf_view_free(pol_sol_xview);
  OSQPVectorf_view_free(pol_sol_yview);

  return OSQP_NO_ERROR;
}
