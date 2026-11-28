#include "osqp.h"
#include "auxil.h"
#include "lin_alg.h"
#include "scaling.h"
#include "util.h"
#include "printing.h"
#include "timing.h"

#if OSQP_EMBEDDED_MODE != 1

OSQPFloat compute_rho_estimate(const OSQPSolver* solver) {

  OSQPFloat prim_res, dual_res;
  OSQPFloat prim_res_norm, dual_res_norm;
  OSQPFloat temp_res_norm;
  OSQPFloat rho_estimate;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   prim_res = work->scaled_prim_res;
  dual_res = work->scaled_dual_res;

   prim_res_norm = OSQPVectorf_norm_inf(work->z);
  temp_res_norm = OSQPVectorf_norm_inf(work->Ax);
  prim_res_norm = c_max(prim_res_norm, temp_res_norm);
  prim_res     /= (prim_res_norm + OSQP_DIVISION_TOL);

   dual_res_norm = OSQPVectorf_norm_inf(work->data->q);
  temp_res_norm = OSQPVectorf_norm_inf(work->Aty);
  dual_res_norm = c_max(dual_res_norm, temp_res_norm);
  temp_res_norm = OSQPVectorf_norm_inf(work->Px);
  dual_res_norm = c_max(dual_res_norm, temp_res_norm);
  dual_res     /= (dual_res_norm + OSQP_DIVISION_TOL);

   rho_estimate = settings->rho * c_sqrt(prim_res / dual_res);
  rho_estimate = c_min(c_max(rho_estimate, OSQP_RHO_MIN), OSQP_RHO_MAX);

  return rho_estimate;
}

OSQPInt adapt_rho(OSQPSolver* solver) {

  OSQPInt   exitflag;
  OSQPFloat rho_new;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;

  exitflag = 0;

   rho_new = compute_rho_estimate(solver);

   info->rho_estimate = rho_new;

   if ((rho_new > settings->rho * settings->adaptive_rho_tolerance) ||
      (rho_new < settings->rho / settings->adaptive_rho_tolerance)) {
    exitflag                 = osqp_update_rho(solver, rho_new);
    info->rho_updates += 1;
    solver->work->rho_updated = 1;
  }

  return exitflag;
}

OSQPInt set_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_types_changed = 0;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  settings->rho = c_min(c_max(settings->rho, OSQP_RHO_MIN), OSQP_RHO_MAX);

  constr_types_changed = OSQPVectorf_ew_bounds_type(work->constr_type,
                                                    work->data->l,
                                                    work->data->u,
                                                    OSQP_RHO_TOL,
                                                    OSQP_INFTY * OSQP_MIN_SCALING);


      OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                     work->constr_type,
                                     OSQP_RHO_MIN,
                                     settings->rho,
                                     OSQP_RHO_EQ_OVER_RHO_INEQ*settings->rho);

  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);

  return constr_types_changed;
}

OSQPInt update_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_type_changed;
  OSQPInt exitflag = 0;
  OSQPWorkspace* work = solver->work;

   constr_type_changed = set_rho_vec(solver);

   if (constr_type_changed == 1) {
    exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);
  }

  return exitflag;
}

#endif


void swap_vectors(OSQPVectorf** a,
                  OSQPVectorf** b) {
  OSQPVectorf* temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}

static void compute_rhs(OSQPSolver* solver) {

  OSQPWorkspace* work     = solver->work;
  OSQPSettings*  settings = solver->settings;

   OSQPVectorf_add_scaled(work->xtilde_view,
                         settings->sigma,work->x_prev,
                         -1., work->data->q);

   if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->ztilde_view, work->rho_inv_vec, work->y);
    OSQPVectorf_add_scaled(work->ztilde_view,
                           -1.0, work->ztilde_view,
                           1.0, work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled(work->ztilde_view,
                           1.0, work->z_prev,
                           -work->rho_inv, work->y);
  }
}

void update_xz_tilde(OSQPSolver* solver,
                     OSQPInt     admm_iter) {

  OSQPWorkspace* work = solver->work;

   compute_rhs(solver);

   work->linsys_solver->solve(work->linsys_solver, work->xz_tilde, admm_iter);
}

void update_x(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   OSQPVectorf_add_scaled(work->x,
                         settings->alpha,work->xtilde_view,
                         (1.0 - settings->alpha),work->x_prev);

   OSQPVectorf_minus(work->delta_x,work->x,work->x_prev);
}

void update_z(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->z, work->rho_inv_vec,work->y);
    OSQPVectorf_add_scaled3(work->z,
                            1., work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled3(work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev,
                            work->rho_inv, work->y);
  }

   OSQPVectorf_ew_bound_vec(work->z, work->z, work->data->l, work->data->u);
}

void update_y(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_add_scaled3(work->delta_y,
                          settings->alpha, work->ztilde_view,
                          (1.0 - settings->alpha), work->z_prev,
                          -1.0, work->z);

  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->delta_y, work->delta_y, work->rho_vec);
  }
  else {
    OSQPVectorf_mult_scalar(work->delta_y, settings->rho);
  }

  OSQPVectorf_plus(work->y, work->y, work->delta_y);

}

void compute_obj_val_dual_gap(const OSQPSolver*  solver,
                              const OSQPVectorf* x,
                              const OSQPVectorf* y,
                                    OSQPFloat*   prim_obj_val,
                                    OSQPFloat*   dual_obj_val,
                                    OSQPFloat*   duality_gap) {
  OSQPFloat quad_term = 0.0;
  OSQPFloat lin_term  = 0.0;
  OSQPFloat sup_term  = 0.0;
  OSQPWorkspace* work = solver->work;

   quad_term = OSQPVectorf_dot_prod(work->Px, x);
  lin_term  = OSQPVectorf_dot_prod(work->data->q, x);

   OSQPVectorf_copy(work->z_prev, y);
  OSQPVectorf_project_polar_reccone(work->z_prev,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

    OSQPVectorf_round_to_zero(work->z_prev, OSQP_ZERO_DEADZONE);

  sup_term  = OSQPVectorf_dot_prod_signed(work->data->u, work->z_prev, +1);
  sup_term += OSQPVectorf_dot_prod_signed(work->data->l, work->z_prev, -1);

   *prim_obj_val = 0.5 * quad_term + lin_term;

   *dual_obj_val = -0.5 * quad_term - sup_term;

   work->scaled_dual_gap = quad_term + lin_term + sup_term;

  if (solver->settings->scaling) {
    *prim_obj_val *= work->scaling->cinv;
    *dual_obj_val *= work->scaling->cinv;

          *duality_gap = work->scaling->cinv * work->scaled_dual_gap;
  } else {
    *duality_gap = work->scaled_dual_gap;
  }

   work->xtPx = quad_term;
  work->qtx  = lin_term;
  work->SC   = sup_term;
}

static OSQPFloat compute_duality_gap_tol(const OSQPSolver* solver,
                                               OSQPFloat   eps_abs,
                                               OSQPFloat   eps_rel) {
  OSQPFloat max_rel_eps = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   max_rel_eps = c_absval(work->xtPx);
  max_rel_eps = c_max(max_rel_eps, c_absval(work->qtx));
  max_rel_eps = c_max(max_rel_eps, c_absval(work->SC));

   if (settings->scaling && !settings->scaled_termination) {
    max_rel_eps = work->scaling->cinv * max_rel_eps;
  }

   return eps_abs + eps_rel * max_rel_eps;
}

static OSQPFloat compute_prim_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* z) {


  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat prim_res;

  OSQPMatrix_Axpy(work->data->A,x,work->Ax, 1.0, 0.0);
  OSQPVectorf_minus(work->z_prev, work->Ax, z);

  work->scaled_prim_res = OSQPVectorf_norm_inf(work->z_prev);

   if (settings->scaling && !settings->scaled_termination) {
    prim_res =  OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z_prev);
  }
  else{
    prim_res  = work->scaled_prim_res;
  }
  return prim_res;
}

static OSQPFloat compute_prim_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   if (settings->scaling && !settings->scaled_termination) {
       max_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z);

       temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->Ax);

       max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

  else {
       max_rel_eps = OSQPVectorf_norm_inf(work->z);

       temp_rel_eps = OSQPVectorf_norm_inf(work->Ax);

       max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

   return eps_abs + eps_rel * max_rel_eps;
}

static OSQPFloat compute_dual_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* y) {


  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat      dual_res;

   OSQPVectorf_copy(work->x_prev, work->data->q);

   OSQPMatrix_Axpy(work->data->P, x, work->Px, 1.0, 0.0);

   OSQPVectorf_plus(work->x_prev, work->x_prev, work->Px);

   if (work->data->m) {
    OSQPMatrix_Atxpy(work->data->A, y, work->Aty, 1.0, 0.0);
    OSQPVectorf_plus(work->x_prev, work->x_prev, work->Aty);
  }

  work->scaled_dual_res = OSQPVectorf_norm_inf(work->x_prev);

   if (settings->scaling && !settings->scaled_termination) {
    dual_res =  work->scaling->cinv * OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                                                  work->x_prev);
  }
  else {
    dual_res = work->scaled_dual_res;
  }

  return dual_res;
}

static OSQPFloat compute_dual_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   if (settings->scaling && !settings->scaled_termination) {
       max_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->data->q);

       temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->Aty);

    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

       temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->Px);

    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

       max_rel_eps *= work->scaling->cinv;
  } else {
       max_rel_eps = OSQPVectorf_norm_inf(work->data->q);

       temp_rel_eps = OSQPVectorf_norm_inf(work->Aty);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);

       temp_rel_eps = OSQPVectorf_norm_inf(work->Px);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
  }

   return eps_abs + eps_rel * max_rel_eps;
}

OSQPInt is_primal_infeasible(OSQPSolver* solver,
                             OSQPFloat   eps_prim_inf) {


  OSQPFloat norm_delta_y;
  OSQPFloat ineq_lhs = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   OSQPVectorf_project_polar_reccone(work->delta_y,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

   if (settings->scaling && !settings->scaled_termination) {
       OSQPVectorf_ew_prod(work->Adelta_x,
                        work->scaling->E,
                        work->delta_y);
    norm_delta_y = OSQPVectorf_norm_inf(work->Adelta_x);
  }
  else
    norm_delta_y = OSQPVectorf_norm_inf(work->delta_y);

  if (norm_delta_y > OSQP_DIVISION_TOL) {

    ineq_lhs  = OSQPVectorf_dot_prod_signed(work->data->u, work->delta_y, +1);
    ineq_lhs += OSQPVectorf_dot_prod_signed(work->data->l, work->delta_y, -1);

       if (ineq_lhs < 0.0) {
      OSQPMatrix_Atxpy(work->data->A, work->delta_y, work->Atdelta_y, 1.0, 0.0);

           if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Atdelta_y,
                            work->Atdelta_y,
                            work->scaling->Dinv);
      }

      return OSQPVectorf_norm_inf(work->Atdelta_y) < eps_prim_inf * norm_delta_y;
    }
  }

   return 0;
}

OSQPInt is_dual_infeasible(OSQPSolver* solver,
                           OSQPFloat   eps_dual_inf) {


  OSQPFloat norm_delta_x;
  OSQPFloat cost_scaling;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   if (settings->scaling && !settings->scaled_termination) {

    norm_delta_x =
    OSQPVectorf_scaled_norm_inf(work->scaling->D,
                                work->delta_x);
    cost_scaling = work->scaling->c;
  }
  else {
    norm_delta_x = OSQPVectorf_norm_inf(work->delta_x);
    cost_scaling = 1.0;
  }

   if (norm_delta_x > OSQP_DIVISION_TOL) {


       if (OSQPVectorf_dot_prod(work->data->q, work->delta_x) < 0.0) {
           OSQPMatrix_Axpy(work->data->P, work->delta_x, work->Pdelta_x, 1.0, 0.0);

           if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Pdelta_x,
                            work->Pdelta_x,
                            work->scaling->Dinv);
      }

           if (OSQPVectorf_norm_inf(work->Pdelta_x) <
          cost_scaling * eps_dual_inf * norm_delta_x) {
               OSQPMatrix_Axpy(work->data->A, work->delta_x, work->Adelta_x,1.0,0.0);

               if (settings->scaling && !settings->scaled_termination) {
          OSQPVectorf_ew_prod(work->Adelta_x, work->Adelta_x, work->scaling->Einv);
        }


                      return OSQPVectorf_in_reccone(work->Adelta_x,
                                      work->data->l,
                                      work->data->u,
                                      OSQP_INFTY * OSQP_MIN_SCALING,
                                      eps_dual_inf * norm_delta_x);
      }
    }
  }

   return 0;
}

OSQPInt has_solution(const OSQPInfo* info) {
  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX));
}

void store_solution(OSQPSolver *solver, OSQPSolution* solution) {

#ifndef OSQP_EMBEDDED_MODE
  OSQPFloat norm_vec;
#endif

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   if (!solution)
    return;


  if (has_solution(info)) {
         if (settings->scaling){
                   unscale_solution(work->x_prev,work->z_prev,
                           work->x,work->y,
                           work);
          OSQPVectorf_to_raw(solution->x, work->x_prev);
          OSQPVectorf_to_raw(solution->y, work->z_prev);
      }
      else{
          OSQPVectorf_to_raw(solution->x, work->x);
          OSQPVectorf_to_raw(solution->y, work->y);
      }
           OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
  }

  else {

       OSQPVectorf_set_scalar(work->x, OSQP_NAN);
    OSQPVectorf_set_scalar(work->y, OSQP_NAN);
    OSQPVectorf_to_raw(solution->x, work->x);
    OSQPVectorf_to_raw(solution->y, work->y);

       osqp_cold_start(solver);


#ifndef OSQP_EMBEDDED_MODE

          if ((info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
        ((info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_y);
      OSQPVectorf_mult_scalar(work->delta_y, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);

           OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
    }

    if ((info->status_val == OSQP_DUAL_INFEASIBLE) ||
        ((info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_x);
      OSQPVectorf_mult_scalar(work->delta_x, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);

           OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
    }

#endif
  }
}

void update_info(OSQPSolver* solver,
                 OSQPInt     iter,
                 OSQPInt     polishing) {

  OSQPVectorf* x;
  OSQPVectorf* z;
  OSQPVectorf* y;

   OSQPFloat* prim_obj_val;
  OSQPFloat* dual_obj_val;
  OSQPFloat* dual_gap;
  OSQPFloat* prim_res;
  OSQPFloat* dual_res;

  OSQPInfo*      info     = solver->info;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  OSQPFloat* run_time;
#endif

#ifndef OSQP_EMBEDDED_MODE

  if (polishing) {
    x            = work->pol->x;
    y            = work->pol->y;
    z            = work->pol->z;
    prim_obj_val = &work->pol->obj_val;
    dual_obj_val = &work->pol->dual_obj_val;
    dual_gap     = &work->pol->duality_gap;
    prim_res     = &work->pol->prim_res;
    dual_res     = &work->pol->dual_res;
# ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->polish_time;
# endif
  }
  else {
#endif
    x            = work->x;
    y            = work->y;
    z            = work->z;
    prim_obj_val = &info->obj_val;
    dual_obj_val = &info->dual_obj_val;
    dual_gap     = &info->duality_gap;
    prim_res     = &info->prim_res;
    dual_res     = &info->dual_res;
    info->iter   = iter;
#ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->solve_time;
#endif
#ifndef OSQP_EMBEDDED_MODE
}

#endif

   if (work->data->m == 0) {
       *prim_res = 0.;
  } else {
    *prim_res = compute_prim_res(solver, x, z);
  }

   *dual_res = compute_dual_res(solver, x, y);

   compute_obj_val_dual_gap(solver, x, y, prim_obj_val, dual_obj_val, dual_gap);

   if (!polishing) {
    info->primdual_int += c_absval(*dual_gap);
  }

 #ifdef OSQP_ENABLE_PROFILING
  *run_time = osqp_toc(work->timer);
#endif

   info->rel_kkt_error = c_max( c_max(*dual_res, *prim_res), *dual_gap);

#ifdef OSQP_ENABLE_PRINTING
  work->summary_printed = 0;
#endif
}


void reset_info(OSQPInfo *info) {
#ifdef OSQP_ENABLE_PROFILING

   info->solve_time = 0.0;
# ifndef OSQP_EMBEDDED_MODE
  info->polish_time = 0.0;
# endif

 #endif

  update_status(info, OSQP_UNSOLVED);

#if OSQP_EMBEDDED_MODE != 1
  info->rho_updates = 0;
#endif
}

const char *OSQP_STATUS_MESSAGE[] = {
  "",
  "solved",
  "solved inaccurate",
  "primal infeasible",
  "primal infeasible inaccurate",
  "dual infeasible",
  "dual infeasible inaccurate",
  "maximum iterations reached",
  "run time limit reached",
  "problem non convex",
  "interrupted",
  "unsolved"
};

void update_status(OSQPInfo* info,
                   OSQPInt   status_val) {
   info->status_val = status_val;

   c_strcpy(info->status, OSQP_STATUS_MESSAGE[status_val]);
}

OSQPInt check_termination(OSQPSolver* solver,
                          OSQPInt     approximate) {

  OSQPFloat eps_prim, eps_dual, eps_duality_gap, eps_prim_inf, eps_dual_inf;
  OSQPInt   exitflag;
  OSQPInt   prim_res_check, dual_res_check, duality_gap_check, prim_inf_check, dual_inf_check;
  OSQPFloat eps_abs, eps_rel;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

   exitflag       = 0;
  prim_res_check = 0; dual_res_check = 0;
  prim_inf_check = 0; dual_inf_check = 0;
  duality_gap_check = 0;

   eps_abs      = settings->eps_abs;
  eps_rel      = settings->eps_rel;
  eps_prim_inf = settings->eps_prim_inf;
  eps_dual_inf = settings->eps_dual_inf;

   if ((info->prim_res > OSQP_INFTY) ||
      (info->dual_res > OSQP_INFTY)){
          update_status(info, OSQP_NON_CVX);
    info->obj_val = OSQP_NAN;
    return 1;
  }

   if (approximate) {
    eps_abs      *= 10;
    eps_rel      *= 10;
    eps_prim_inf *= 10;
    eps_dual_inf *= 10;
  }

   if (work->data->m == 0) {
    prim_res_check = 1;
  }
  else {
       eps_prim = compute_prim_tol(solver, eps_abs, eps_rel);

       if (info->prim_res < eps_prim) {
      prim_res_check = 1;
    } else {
           prim_inf_check = is_primal_infeasible(solver, eps_prim_inf);
    }
  }

   eps_dual = compute_dual_tol(solver, eps_abs, eps_rel);

   if (info->dual_res < eps_dual) {
    dual_res_check = 1;
  } else {
       dual_inf_check = is_dual_infeasible(solver, eps_dual_inf);
  }

  if (settings->check_dualgap ) {
       eps_duality_gap = compute_duality_gap_tol(solver, eps_abs, eps_rel);

       if (settings->scaling && !settings->scaled_termination) {
           if (c_absval(info->duality_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    } else {
           if (c_absval(work->scaled_dual_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    }
  } else {
       duality_gap_check = 1;
  }

   if (prim_res_check && dual_res_check && duality_gap_check) {
       if (approximate) {
      update_status(info, OSQP_SOLVED_INACCURATE);
    } else {
      update_status(info, OSQP_SOLVED);
    }
    exitflag = 1;
  }
  else if (prim_inf_check) {
       if (approximate) {
      update_status(info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_PRIMAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
           OSQPVectorf_ew_prod(work->delta_y,
                          work->delta_y,
                          work->scaling->E);
    }
    info->obj_val = OSQP_INFTY;
    exitflag            = 1;
  }
  else if (dual_inf_check) {
       if (approximate) {
      update_status(info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_DUAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
           OSQPVectorf_ew_prod(work->delta_x,
                          work->delta_x,
                          work->scaling->D);
    }
    info->obj_val = -OSQP_INFTY;
    exitflag            = 1;
  }

  return exitflag;
}


#ifndef OSQP_EMBEDDED_MODE

OSQPInt validate_data(const OSQPCscMatrix* P,
                      const OSQPFloat*     q,
                      const OSQPCscMatrix* A,
                      const OSQPFloat*     l,
                      const OSQPFloat*     u,
                            OSQPInt        m,
                            OSQPInt        n) {
  OSQPInt j, ptr;

  if (!P) {
    c_eprint("Missing quadratic cost matrix P");
    return 1;
  }

  if (!A) {
    c_eprint("Missing constraint matrix A");
    return 1;
  }

  if (!q) {
    c_eprint("Missing linear cost vector q");
    return 1;
  }

   if ((n <= 0) || (m < 0)) {
    c_eprint("n must be positive and m nonnegative; n = %i, m = %i",
             (int)n, (int)m);
    return 1;
  }

   if (P->m != n) {
    c_eprint("P does not have dimension n x n with n = %i", (int)n);
    return 1;
  }

  if (P->m != P->n) {
    c_eprint("P is not square");
    return 1;
  }

  for (j = 0; j < n; j++) {
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) {
      if (P->i[ptr] > j) {
        c_eprint("P is not upper triangular");
        return 1;
      }
    }
  }

   if ((A->m != m) || (A->n != n)) {
    c_eprint("A does not have dimension %i x %i", (int)m, (int)n);
    return 1;
  }

   for (j = 0; j < m; j++) {
    if (l[j] > u[j]) {
      c_eprint("Lower bound at index %d is greater than upper bound: %.4e > %.4e",
               (int)j, l[j], u[j]);
      return 1;
    }
  }

  return 0;
}

#endif


OSQPInt validate_linsys_solver(OSQPInt linsys_solver) {
   if ( (linsys_solver == OSQP_INDIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_INDIRECT_SOLVER) ) {
    return 0;
  }

   if ( (linsys_solver == OSQP_DIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_DIRECT_SOLVER) ) {
    return 0;
  }

   return 1;
}


OSQPInt validate_settings(const OSQPSettings* settings,
                          OSQPInt             from_setup) {

  if (!settings) {
    c_eprint("Missing settings!");
    return 1;
  }

  if (from_setup &&
      validate_linsys_solver(settings->linsys_solver)) {
    c_eprint("linsys_solver not recognized");
    return 1;
  }

  if (from_setup &&
      settings->allocate_solution != 0 &&
      settings->allocate_solution != 1) {
    c_eprint("allocate_solution must be either 0 or 1");
    return 1;
  }

  if (settings->verbose != 0 &&
      settings->verbose != 1) {
    c_eprint("verbose must be either 0 or 1");
    return 1;
  }

  if (settings->profiler_level != 0 &&
      settings->profiler_level != 1 &&
      settings->profiler_level != 2) {
    c_eprint("profiler_level must be either 0, 1 or 2");
    return 1;
  }

  if (settings->warm_starting != 0 &&
      settings->warm_starting != 1) {
    c_eprint("warm_start must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->scaling < 0) {
    c_eprint("scaling must be nonnegative");
    return 1;
  }

  if (settings->polishing != 0 &&
      settings->polishing != 1) {
    c_eprint("polishing must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->rho <= 0.0) {
    c_eprint("rho must be positive");
    return 1;
  }

  if (from_setup &&
      (settings->rho_is_vec != 0) &&
      (settings->rho_is_vec != 1)) {
    c_eprint("rho_is_vec must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->sigma <= 0.0) {
    c_eprint("sigma must be positive");
    return 1;
  }

  if (settings->alpha <= 0.0 ||
      settings->alpha >= 2.0) {
    c_eprint("alpha must be strictly between 0 and 2");
    return 1;
  }

  if (settings->cg_max_iter <= 0) {
    c_eprint("cg_max_iter must be positive");
    return 1;
  }

  if (settings->cg_tol_reduction <= 0) {
    c_eprint("cg_tol_reduction must be positive");
    return 1;
  }

  if (settings->cg_tol_fraction <= 0.0 ||
      settings->cg_tol_fraction >= 1.0) {
    c_eprint("cg_tol_fraction must be strictly between 0 and 1");
    return 1;
  }

  if (from_setup &&
      settings->adaptive_rho < 0 ||
      settings->adaptive_rho >= _OSQP_ADAPTIVE_RHO_UPDATE_LAST_VALUE) {
    c_eprint("adaptive_rho not a valid choice");
    return 1;
  }

#if !defined(OSQP_ENABLE_PROFILING)
  if (from_setup && settings->adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_TIME) {
    c_eprint("adaptive_rho time-based adaptation requires profiling to be enabled");
    return 1;
  }
#endif

  if (from_setup && settings->adaptive_rho_interval < 0) {
    c_eprint("adaptive_rho_interval must be nonnegative");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_fraction <= 0) {
    c_eprint("adaptive_rho_fraction must be positive");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_tolerance < 1.0) {
    c_eprint("adaptive_rho_tolerance must be >= 1");
    return 1;
  }

  if (settings->max_iter <= 0) {
    c_eprint("max_iter must be positive");
    return 1;
  }

  if (settings->eps_abs < 0.0) {
    c_eprint("eps_abs must be nonnegative");
    return 1;
  }

  if (settings->eps_rel < 0.0) {
    c_eprint("eps_rel must be nonnegative");
    return 1;
  }

  if (settings->eps_rel == 0.0 &&
      settings->eps_abs == 0.0) {
    c_eprint("at least one of eps_abs and eps_rel must be positive");
    return 1;
  }

  if (settings->eps_prim_inf <= 0.0) {
    c_eprint("eps_prim_inf must be positive");
    return 1;
  }

  if (settings->eps_dual_inf <= 0.0) {
    c_eprint("eps_dual_inf must be positive");
    return 1;
  }

  if (settings->scaled_termination != 0 &&
      settings->scaled_termination != 1) {
    c_eprint("scaled_termination must be either 0 or 1");
    return 1;
  }

  if (settings->check_termination < 0) {
    c_eprint("check_termination must be nonnegative");
    return 1;
  }

  if (settings->check_dualgap != 0 &&
      settings->check_dualgap != 1) {
    c_eprint("check_dualgap must be either 0 or 1");
    return 1;
  }

  if (settings->time_limit <= 0.0) {
    c_eprint("time_limit must be positive\n");
    return 1;
  }

  if (settings->delta <= 0.0) {
    c_eprint("delta must be positive");
    return 1;
  }

  if (settings->polish_refine_iter < 0) {
    c_eprint("polish_refine_iter must be nonnegative");
    return 1;
  }

  return 0;
}
