#include "glob_opts.h"
#include "osqp.h"
#include "auxil.h"
#include "osqp_api_constants.h"
#include "osqp_api_functions.h"
#include "osqp_api_types.h"
#include "util.h"
#include "scaling.h"
#include "error.h"
#include "version.h"
#include "lin_alg.h"
#include "printing.h"
#include "timing.h"
#include "profilers.h"


#ifdef OSQP_CODEGEN
  #include "codegen.h"
#endif

#ifndef OSQP_EMBEDDED_MODE
# include "polish.h"
#endif

#ifdef OSQP_ENABLE_DERIVATIVES
# include "derivative.h"
#endif

#ifdef OSQP_ENABLE_INTERRUPT
# include "interrupt.h"
#endif

#ifndef OSQP_EMBEDDED_MODE
OSQPCscMatrix* OSQPCscMatrix_new(OSQPInt    m,
                                 OSQPInt    n,
                                 OSQPInt    nzmax,
                                 OSQPFloat* x,
                                 OSQPInt*   i,
                                 OSQPInt*   p) {
  OSQPCscMatrix* mat = c_calloc(1, sizeof(OSQPCscMatrix));

  if (!mat) return OSQP_NULL;

  OSQPCscMatrix_set_data(mat, m, n, nzmax, x, i, p);

  return mat;
}

OSQPCscMatrix* OSQPCscMatrix_identity(OSQPInt m) {
  return OSQPCscMatrix_diag_scalar(m, m, 1.0);
}

OSQPCscMatrix* OSQPCscMatrix_diag_scalar(OSQPInt   m,
                                         OSQPInt   n,
                                         OSQPFloat scalar) {
  OSQPInt i;
  OSQPInt min_elem = c_min(m, n);
  OSQPCscMatrix* mat = c_calloc(1, sizeof(OSQPCscMatrix));

  if (!mat) return OSQP_NULL;

  mat->m  = m;
  mat->n  = n;
  mat->nz = -1;
  mat->p  = (OSQPInt*) c_calloc(n + 1, sizeof(OSQPInt));

  if( m < n) {
       for(i=0; i < n; i++) {
      if(i < m) {
        mat->p[i] = i;
      } else {
        mat->p[i] = m;
      }
    }
  } else {
       for(i=0; i < n; i++) {
      mat->p[i] = i;
    }
  }

  mat->nzmax = min_elem;
  mat->i     = (OSQPInt*) c_malloc(min_elem * sizeof(OSQPInt));
  mat->x     = (OSQPFloat*) c_malloc(min_elem * sizeof(OSQPFloat));

  for(i=0; i < min_elem; i++) {
    mat->i[i] = i;
    mat->x[i] = scalar;
  }

   mat->p[n] = min_elem;

   mat->owned = 1;

  return mat;
}

OSQPCscMatrix* OSQPCscMatrix_diag_vec(OSQPInt    m,
                                      OSQPInt    n,
                                      OSQPFloat* vec) {
  OSQPInt i;
  OSQPInt min_elem = c_min(m,n);

   OSQPCscMatrix* mat = OSQPCscMatrix_diag_scalar(m, n, 1.0);

  if (!mat) return OSQP_NULL;

  for(i=0; i < min_elem; i++) {
    mat->x[i] = vec[i];
  }

  return mat;
}

OSQPCscMatrix* OSQPCscMatrix_zeros(OSQPInt    m,
                                   OSQPInt    n) {
  OSQPCscMatrix* mat = c_calloc(1, sizeof(OSQPCscMatrix));

  if (!mat) return OSQP_NULL;

  mat->m     = m;
  mat->n     = n;
  mat->nz    = -1;
  mat->nzmax = 0;

   mat->x = OSQP_NULL;
  mat->i = OSQP_NULL;

   mat->p = (OSQPInt*) c_calloc(n + 1, sizeof(OSQPInt));

   mat->owned = 1;

  return mat;
}

void OSQPCscMatrix_free(OSQPCscMatrix* mat) {
  if (mat) {
       if(mat->owned) {
      if(mat->p) c_free(mat->p);
      if(mat->i) c_free(mat->i);
      if(mat->x) c_free(mat->x);
    }

    c_free(mat);
  }
}
#endif

void OSQPCscMatrix_set_data(OSQPCscMatrix* M,
                            OSQPInt        m,
                            OSQPInt        n,
                            OSQPInt        nzmax,
                            OSQPFloat*     x,
                            OSQPInt*       i,
                            OSQPInt*       p) {
  M->m     = m;
  M->n     = n;
  M->nz    = -1;
  M->nzmax = nzmax;
  M->x     = x;
  M->i     = i;
  M->p     = p;

   M->owned = 0;
}

#ifndef OSQP_EMBEDDED_MODE
OSQPSettings* OSQPSettings_new() {
  OSQPSettings* settings = (OSQPSettings*) c_calloc(1, sizeof(OSQPSettings));

  if (!settings)
    return OSQP_NULL;

  osqp_set_default_settings(settings);

  return settings;
}

void OSQPSettings_free(OSQPSettings* settings) {
  if (settings)
    c_free(settings);
}

OSQPCodegenDefines* OSQPCodegenDefines_new() {
  OSQPCodegenDefines* defs = (OSQPCodegenDefines*) c_calloc(1, sizeof(OSQPCodegenDefines));

  if (!defs)
    return OSQP_NULL;

  osqp_set_default_codegen_defines(defs);

  return defs;
}

void OSQPCodegenDefines_free(OSQPCodegenDefines* defs) {
  if (defs)
    c_free(defs);
}
#endif


OSQPInt osqp_capabilities(void) {
  OSQPInt capabilities = 0;

  capabilities |= osqp_algebra_linsys_supported();

#if OSQP_EMBEDDED_MODE != 1
  capabilities |= OSQP_CAPABILITY_UPDATE_MATRICES;
#endif

#ifdef OSQP_CODEGEN
  capabilities |= OSQP_CAPABILITY_CODEGEN;
#endif

#ifdef OSQP_ENABLE_DERIVATIVES
    capabilities |= OSQP_CAPABILITY_DERIVATIVES;
#endif

  return capabilities;
}

const char* osqp_version(void) {
  return OSQP_VERSION;
}


const char* osqp_error_message(OSQPInt error_flag) {
  if( error_flag >= OSQP_LAST_ERROR_PLACE ) {
    return OSQP_ERROR_MESSAGE[OSQP_LAST_ERROR_PLACE-1];
  }


  return OSQP_ERROR_MESSAGE[error_flag-1];
}


void osqp_get_dimensions(OSQPSolver* solver,
                         OSQPInt*    m,
                         OSQPInt*    n) {

   if (!solver || !solver->work || !solver->work->data) {
    *m = -1;
    *n = -1;
  }
  else {
    *m = solver->work->data->m;
    *n = solver->work->data->n;
  }
}


void osqp_set_default_codegen_defines(OSQPCodegenDefines* defines) {

   if (!defines)
    return;

  defines->embedded_mode      = 1;
  defines->float_type         = 0;
  defines->printing_enable    = 0;
  defines->profiling_enable   = 0;
  defines->interrupt_enable   = 0;
  defines->derivatives_enable = 0;
}


void osqp_set_default_settings(OSQPSettings* settings) {

   if (!settings)
    return;

  settings->device = 0;
  settings->linsys_solver  = osqp_algebra_default_linsys();

  settings->allocate_solution = 1;
  settings->profiler_level    = 0;

  settings->verbose           = OSQP_VERBOSE;
  settings->warm_starting     = OSQP_WARM_STARTING;
  settings->scaling           = OSQP_SCALING;
  settings->polishing         = OSQP_POLISHING;

  settings->rho           = (OSQPFloat)OSQP_RHO;
  settings->rho_is_vec    = OSQP_RHO_IS_VEC;
  settings->sigma         = (OSQPFloat)OSQP_SIGMA;
  settings->alpha         = (OSQPFloat)OSQP_ALPHA;

  settings->cg_max_iter      = OSQP_CG_MAX_ITER;
  settings->cg_tol_reduction = OSQP_CG_TOL_REDUCTION;
  settings->cg_tol_fraction  = OSQP_CG_TOL_FRACTION;
  settings->cg_precond       = OSQP_DIAGONAL_PRECONDITIONER;

  settings->adaptive_rho           = OSQP_ADAPTIVE_RHO_UPDATE_DEFAULT;
  settings->adaptive_rho_interval  = OSQP_ADAPTIVE_RHO_INTERVAL;
  settings->adaptive_rho_fraction  = (OSQPFloat)OSQP_ADAPTIVE_RHO_FRACTION;
  settings->adaptive_rho_tolerance = (OSQPFloat)OSQP_ADAPTIVE_RHO_TOLERANCE;

  settings->max_iter           = OSQP_MAX_ITER;
  settings->eps_abs            = (OSQPFloat)OSQP_EPS_ABS;
  settings->eps_rel            = (OSQPFloat)OSQP_EPS_REL;
  settings->eps_prim_inf       = (OSQPFloat)OSQP_EPS_PRIM_INF;
  settings->eps_dual_inf       = (OSQPFloat)OSQP_EPS_DUAL_INF;
  settings->scaled_termination = OSQP_SCALED_TERMINATION;
  settings->check_termination  = OSQP_CHECK_TERMINATION;
  settings->check_dualgap      = OSQP_CHECK_DUALGAP;
  settings->time_limit         = OSQP_TIME_LIMIT;

  settings->delta              = OSQP_DELTA;
  settings->polish_refine_iter = OSQP_POLISH_REFINE_ITER;
}

#ifndef OSQP_EMBEDDED_MODE


OSQPInt osqp_setup(OSQPSolver**         solverp,
                   const OSQPCscMatrix* P,
                   const OSQPFloat*     q,
                   const OSQPCscMatrix* A,
                   const OSQPFloat*     l,
                   const OSQPFloat*     u,
                   OSQPInt              m,
                   OSQPInt              n,
                   const OSQPSettings*  settings) {

  OSQPInt exitflag;

  OSQPSolver*    solver;
  OSQPWorkspace* work;

   if (validate_data(P,q,A,l,u,m,n)) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

   if (validate_settings(settings, 1)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);

  osqp_profiler_init(settings->profiler_level);
  osqp_profiler_sec_push(OSQP_PROFILER_SEC_SETUP);

   solver = c_calloc(1, sizeof(OSQPSolver));
  if (!(solver)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  *solverp = solver;

   work   = c_calloc(1, sizeof(OSQPWorkspace));
  if (!(work)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  solver->work = work;

   solver->info = c_calloc(1, sizeof(OSQPInfo));
  if (!(solver->info)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

 # ifdef OSQP_ENABLE_PROFILING
  work->timer = OSQPTimer_new();
  if (!(work->timer)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  osqp_tic(work->timer);
# endif

   exitflag = osqp_algebra_init_libs(settings->device);
  if (exitflag) return osqp_error(OSQP_ALGEBRA_LOAD_ERROR);

   work->data = c_calloc(1, sizeof(OSQPData));
  if (!(work->data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->m = m;
  work->data->n = n;

   work->data->P = OSQPMatrix_new_from_csc(P,1);
  work->data->q = OSQPVectorf_new(q,n);
  if (!(work->data->P) || !(work->data->q)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

   work->data->A = OSQPMatrix_new_from_csc(A,0);
  if (!(work->data->A)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->l = OSQPVectorf_new(l,m);
  work->data->u = OSQPVectorf_new(u,m);
  if (!(work->data->l) || !(work->data->u))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  if (settings->rho_is_vec) {
       work->rho_vec     = OSQPVectorf_malloc(m);
    work->rho_inv_vec = OSQPVectorf_malloc(m);
    if (!(work->rho_vec) || !(work->rho_inv_vec))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);

       work->constr_type = OSQPVectori_calloc(m);
    if (!(work->constr_type)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }
  else {
    work->rho_vec     = OSQP_NULL;
    work->rho_inv_vec = OSQP_NULL;
  }

   work->x           = OSQPVectorf_calloc(n);
  work->z           = OSQPVectorf_calloc(m);
  work->xz_tilde    = OSQPVectorf_calloc(n + m);
  work->xtilde_view = OSQPVectorf_view(work->xz_tilde,0,n);
  work->ztilde_view = OSQPVectorf_view(work->xz_tilde,n,m);
  work->x_prev      = OSQPVectorf_calloc(n);
  work->z_prev      = OSQPVectorf_calloc(m);
  work->y           = OSQPVectorf_calloc(m);
  if (!(work->x) || !(work->z) || !(work->xz_tilde))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->xtilde_view) || !(work->ztilde_view))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->x_prev) || !(work->z_prev) || !(work->y))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

   work->Ax  = OSQPVectorf_calloc(m);
  work->Px  = OSQPVectorf_calloc(n);
  work->Aty = OSQPVectorf_calloc(n);

   work->delta_y   = OSQPVectorf_calloc(m);
  work->Atdelta_y = OSQPVectorf_calloc(n);

   work->delta_x  = OSQPVectorf_calloc(n);
  work->Pdelta_x = OSQPVectorf_calloc(n);
  work->Adelta_x = OSQPVectorf_calloc(m);

  if (!(work->Ax) || !(work->Px) || !(work->Aty))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->delta_y) || !(work->Atdelta_y))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->delta_x) || !(work->Pdelta_x) || !(work->Adelta_x))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

   solver->settings = copy_settings(settings);
  if (!(solver->settings)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

   if (settings->scaling) {
       work->scaling = c_malloc(sizeof(OSQPScaling));
    if (!(work->scaling)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    work->scaling->D    = OSQPVectorf_calloc(n);
    work->scaling->Dinv = OSQPVectorf_calloc(n);
    work->scaling->E    = OSQPVectorf_calloc(m);
    work->scaling->Einv = OSQPVectorf_calloc(m);
    if (!(work->scaling->D) || !(work->scaling->Dinv) ||
        !(work->scaling->E) || !(work->scaling->Einv))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);


       work->D_temp   = OSQPVectorf_calloc(n);
    work->D_temp_A = OSQPVectorf_calloc(n);
    work->E_temp   = OSQPVectorf_calloc(m);
    if (!(work->D_temp) || !(work->D_temp_A) || !(work->E_temp))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);

       osqp_profiler_sec_push(OSQP_PROFILER_SEC_SCALE);
    scale_data(solver);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_SCALE);
  } else {
    work->scaling  = OSQP_NULL;
    work->D_temp   = OSQP_NULL;
    work->D_temp_A = OSQP_NULL;
    work->E_temp   = OSQP_NULL;
  }

  if (settings->rho_is_vec) {
          set_rho_vec(solver);
  }
  else {
    solver->settings->rho = c_min(c_max(settings->rho, OSQP_RHO_MIN), OSQP_RHO_MAX);
    work->rho_inv = 1. / settings->rho;
  }

   exitflag = osqp_algebra_init_linsys_solver(&(work->linsys_solver), work->data->P, work->data->A,
                                             work->rho_vec, solver->settings,
                                             &work->scaled_prim_res, &work->scaled_dual_res, 0);

  if (exitflag == OSQP_NONCVX_ERROR) {
    update_status(solver->info, OSQP_NON_CVX);
    return osqp_error(exitflag);
  }
  else if (exitflag) {
    return osqp_error(exitflag);
  }

   osqp_cold_start(solver);

   work->pol = c_malloc(sizeof(OSQPPolish));
  if (!(work->pol)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->pol->active_flags = OSQPVectori_malloc(m);
  work->pol->x            = OSQPVectorf_malloc(n);
  work->pol->z            = OSQPVectorf_malloc(m);
  work->pol->y            = OSQPVectorf_malloc(m);
  if (!(work->pol->x)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->pol->active_flags) ||
      !(work->pol->z) || !(work->pol->y))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

   if (settings->allocate_solution) {
    solver->solution = c_calloc(1, sizeof(OSQPSolution));

    if (!(solver->solution)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

    solver->solution->x             = c_calloc(1, n * sizeof(OSQPFloat));
    solver->solution->y             = c_calloc(1, m * sizeof(OSQPFloat));
    solver->solution->prim_inf_cert = c_calloc(1, m * sizeof(OSQPFloat));
    solver->solution->dual_inf_cert = c_calloc(1, n * sizeof(OSQPFloat));

    if ( !(solver->solution->x) || !(solver->solution->dual_inf_cert) )
      return osqp_error(OSQP_MEM_ALLOC_ERROR);

    if ( m && (!(solver->solution->y) || !(solver->solution->prim_inf_cert)) )
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }
  else {
    solver->solution = OSQP_NULL;
  }

   solver->info->status_polish = OSQP_POLISH_NOT_PERFORMED;
  update_status(solver->info, OSQP_UNSOLVED);
# ifdef OSQP_ENABLE_PROFILING
  solver->info->solve_time  = 0.0;
  solver->info->update_time = 0.0;
  solver->info->polish_time = 0.0;
  solver->info->run_time    = 0.0;
  solver->info->setup_time  = osqp_toc(work->timer);

  work->first_run         = 1;
  work->clear_update_time = 0;
  work->rho_update_from_solve = 0;
# endif
  solver->info->rho_updates   = 0;
  solver->info->rho_estimate  = solver->settings->rho;
  solver->info->obj_val       = OSQP_INFTY;
  solver->info->prim_res      = OSQP_INFTY;
  solver->info->dual_res      = OSQP_INFTY;
  solver->info->rel_kkt_error = OSQP_INFTY;

  work->last_rel_kkt = OSQP_INFTY;

   work->rho_updated = 0;

  switch(solver->settings->adaptive_rho)
  {
  case OSQP_ADAPTIVE_RHO_UPDATE_DISABLED:
       break;

  case OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS:
       if(solver->settings->adaptive_rho_interval == 0) {
      if (solver->settings->check_termination) {
                      solver->settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_MULTIPLE_TERMINATION *
                                                  solver->settings->check_termination;
      } else {
               solver->settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_FIXED;
      }
    }
    break;

  case OSQP_ADAPTIVE_RHO_UPDATE_TIME:
       break;

  case OSQP_ADAPTIVE_RHO_UPDATE_KKT_ERROR:
       if(solver->settings->adaptive_rho_interval == 0) {
           solver->settings->adaptive_rho_interval = 1;
    }
    break;
  }

# ifdef OSQP_ENABLE_DERIVATIVES
  work->derivative_data = c_calloc(1, sizeof(OSQPDerivativeData));
  if (!(work->derivative_data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->derivative_data->y_u = OSQPVectorf_malloc(m);
  work->derivative_data->y_l = OSQPVectorf_malloc(m);
  work->derivative_data->ryl = OSQPVectorf_malloc(m);
  work->derivative_data->ryu = OSQPVectorf_malloc(m);
  work->derivative_data->rhs = OSQPVectorf_malloc(2 * (n + 2*m));
  if (!(work->derivative_data->y_u) || !(work->derivative_data->y_l) ||
    !(work->derivative_data->ryl) || !(work->derivative_data->ryu))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
# endif

  osqp_profiler_sec_pop(OSQP_PROFILER_SEC_SETUP);

 # ifdef OSQP_ENABLE_PRINTING
  if (solver->settings->verbose) print_setup_header(solver);
  work->summary_printed = 0;
# endif

   return 0;
}

#endif


OSQPInt osqp_solve(OSQPSolver *solver) {

  OSQPInt exitflag;
  OSQPInt iter, max_iter;

  OSQPInt can_print = 0;
  OSQPInt can_adapt_rho = 0;
  OSQPInt can_check_termination = 0;

  OSQPWorkspace* work;
  OSQPSettings*  settings;

#ifdef OSQP_ENABLE_PROFILING
  OSQPFloat temp_run_time;
#endif

   if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  work = solver->work;
  settings = solver->settings;

#ifdef OSQP_ENABLE_PROFILING
  if (work->clear_update_time == 1)
    solver->info->update_time = 0.0;
  work->rho_update_from_solve = 1;
#endif

   exitflag              = 0;
  can_check_termination = 0;
#ifdef OSQP_ENABLE_PRINTING
  can_print = settings->verbose;
#endif

#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->timer);
#endif

osqp_profiler_sec_push(OSQP_PROFILER_SEC_OPT_SOLVE);


#ifdef OSQP_ENABLE_PRINTING
  if (settings->verbose) {
       print_header();
  }
#endif

#ifdef OSQP_ENABLE_INTERRUPT

   osqp_start_interrupt_listener();
#endif

    if (!settings->warm_starting) osqp_cold_start(solver);


  max_iter = settings->max_iter;
  for (iter = 1; iter <= max_iter; iter++) {
    osqp_profiler_sec_push(OSQP_PROFILER_SEC_ADMM_ITER);

       swap_vectors(&(work->x), &(work->x_prev));
    swap_vectors(&(work->z), &(work->z_prev));

          osqp_profiler_sec_push(OSQP_PROFILER_SEC_ADMM_KKT_SOLVE);
    update_xz_tilde(solver, iter);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_ADMM_KKT_SOLVE);

    osqp_profiler_sec_push(OSQP_PROFILER_SEC_ADMM_UPDATE);

       update_x(solver);

       osqp_profiler_sec_push(OSQP_PROFILER_SEC_ADMM_PROJ);
    update_z(solver);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_ADMM_PROJ);

       update_y(solver);

       osqp_profiler_sec_pop(OSQP_PROFILER_SEC_ADMM_UPDATE);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_ADMM_ITER);

#ifdef OSQP_ENABLE_INTERRUPT

       if (osqp_is_interrupted()) {
      update_status(solver->info, OSQP_SIGINT);
      c_print("Solver interrupted\n");
      exitflag = 1;
      goto exit;
    }
#endif

#ifdef OSQP_ENABLE_PROFILING

          if (work->first_run) {
      temp_run_time = solver->info->setup_time + osqp_toc(work->timer);
    }
    else {
      temp_run_time = solver->info->update_time + osqp_toc(work->timer);
    }

    if (settings->time_limit &&
        (temp_run_time >= settings->time_limit)) {
      update_status(solver->info, OSQP_TIME_LIMIT_REACHED);
# ifdef OSQP_ENABLE_PRINTING

      if (settings->verbose) c_print("run time limit reached\n");
      can_print = 0;
# endif
      break;
    }
#endif


       can_check_termination = settings->check_termination &&
                            (iter % settings->check_termination == 0);

   #ifdef OSQP_ENABLE_PRINTING
    can_print = settings->verbose &&
                ((iter % OSQP_PRINT_INTERVAL == 0) || (iter == 1));
#else
    can_print = 0;
#endif

#if OSQP_EMBEDDED_MODE != 1
    switch(settings->adaptive_rho)
    {
    case OSQP_ADAPTIVE_RHO_UPDATE_DISABLED:
           can_adapt_rho = 0;
      break;

    case OSQP_ADAPTIVE_RHO_UPDATE_TIME:
#ifdef OSQP_ENABLE_PROFILING
                     if(settings->adaptive_rho_interval == 0) {
               if(osqp_toc(work->timer) >
            settings->adaptive_rho_fraction * solver->info->setup_time)
        {
                   if (settings->check_termination)
          {
                                  settings->adaptive_rho_interval = (OSQPInt)c_roundmultiple(iter,
                                                                       settings->check_termination);
          }
          else
          {
                                                        settings->adaptive_rho_interval = (OSQPInt)c_roundmultiple(iter, OSQP_CHECK_TERMINATION);
          }

                   settings->adaptive_rho_interval = c_min(1, c_max(settings->adaptive_rho_interval,
                                                           settings->check_termination));
        }
        else
        {
                   can_adapt_rho = 0;
          break;
        }
      }
     #else
           can_adapt_rho = 0;
      break;
#endif

    case OSQP_ADAPTIVE_RHO_UPDATE_KKT_ERROR:
         case OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS:
           if(settings->adaptive_rho_interval && (iter % settings->adaptive_rho_interval == 0)) {
        can_adapt_rho = 1;
      } else {
        can_adapt_rho = 0;
      }
      break;
    }
#else
    can_adapt_rho = 0;
#endif

    if(can_check_termination || can_print || can_adapt_rho || iter == 1) {
                                         update_info(solver, iter, 0);
    }

       if (can_check_termination) {
      if (check_termination(solver, 0)) {
               break;
      }
    }

    work->rho_updated = 0;
#if OSQP_EMBEDDED_MODE != 1
          if(can_adapt_rho && (settings->adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_KKT_ERROR)) {
      if(solver->info->rel_kkt_error <= ( settings->adaptive_rho_fraction * work->last_rel_kkt) ) {
        can_adapt_rho = 1;
      }
      else {
        can_adapt_rho = 0;
      }
    }

       if(can_adapt_rho) {
      osqp_profiler_event_mark(OSQP_PROFILER_EVENT_RHO_UPDATE);

      if (adapt_rho(solver)) {
        c_eprint("Failed rho update");
        exitflag = 1;
        goto exit;
      }
    }
#endif

       if(work->rho_updated) {
      work->last_rel_kkt = solver->info->rel_kkt_error;
    }

#ifdef OSQP_ENABLE_PRINTING
       if (can_print || (settings->verbose && work->rho_updated)) {
      print_summary(solver);
    }
#endif
  }


    if (!can_check_termination) {
   #ifdef OSQP_ENABLE_PRINTING

    if (!can_print) {
                update_info(solver, iter - 1, 0);
    }
#else

       update_info(solver, iter - 1, 0);
#endif

#ifdef OSQP_ENABLE_PRINTING

       if (settings->verbose && !work->summary_printed) print_summary(solver);
#endif

       check_termination(solver, 0);

  }

    if (has_solution(solver->info)){
    compute_obj_val_dual_gap(solver, work->x, work->y,
                             &(solver->info->obj_val),
                             &(solver->info->dual_obj_val),
                             &(solver->info->duality_gap));
  }


#ifdef OSQP_ENABLE_PRINTING
   if (settings->verbose && !work->summary_printed) {
    print_summary(solver);
  }
#endif

   if (solver->info->status_val == OSQP_UNSOLVED) {
    if (!check_termination(solver, 1)) {
      update_status(solver->info, OSQP_MAX_ITER_REACHED);
    }
  }

#ifdef OSQP_ENABLE_PROFILING
  if (solver->info->status_val == OSQP_TIME_LIMIT_REACHED) {
    if (!check_termination(solver, 1)) {
      update_status(solver->info, OSQP_TIME_LIMIT_REACHED);
    }
  }
#endif


#if OSQP_EMBEDDED_MODE != 1
   solver->info->rho_estimate = compute_rho_estimate(solver);
#endif

 #ifdef OSQP_ENABLE_PROFILING
  solver->info->solve_time = osqp_toc(work->timer);
#endif


#ifndef OSQP_EMBEDDED_MODE
   if (settings->polishing && (solver->info->status_val == OSQP_SOLVED)) {
    osqp_profiler_sec_push(OSQP_PROFILER_SEC_POLISH);
    exitflag = polish(solver);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_POLISH);

    if (exitflag > 0) {
      c_eprint("Failed polishing");
      goto exit;
    }
  }
#endif

#ifdef OSQP_ENABLE_PROFILING
   if (work->first_run) {
       solver->info->run_time = solver->info->setup_time +
                             solver->info->solve_time +
                             solver->info->polish_time;
  } else {
       solver->info->run_time = solver->info->update_time +
                             solver->info->solve_time +
                             solver->info->polish_time;
  }

   if (work->first_run) work->first_run = 0;

   work->clear_update_time = 1;

   work->rho_update_from_solve = 0;
#endif

#ifdef OSQP_ENABLE_PRINTING
   if (settings->verbose) print_footer(solver->info, settings->polishing);
#endif

   store_solution(solver, solver->solution);

#if defined(OSQP_ENABLE_PROFILING) || defined(OSQP_ENABLE_INTERRUPT) || OSQP_EMBEDDED_MODE != 1
exit:
#endif

#ifdef OSQP_ENABLE_INTERRUPT
   osqp_end_interrupt_listener();
#endif

  osqp_profiler_sec_pop(OSQP_PROFILER_SEC_OPT_SOLVE);

  return exitflag;
}


OSQPInt osqp_get_solution(OSQPSolver* solver, OSQPSolution* solution) {
  if (!solver || !solver->work || !solver->settings || !solver->info) {
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  if (!solution)
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  store_solution(solver, solution);

  return OSQP_NO_ERROR;
}


#ifndef OSQP_EMBEDDED_MODE

OSQPInt osqp_cleanup(OSQPSolver* solver) {

  OSQPInt exitflag = 0;
  OSQPWorkspace* work;

  if(!solver) return 0;

  work = solver->work;

  if (work) {
       osqp_algebra_free_libs();

       if (work->data) {
      OSQPMatrix_free(work->data->P);
      OSQPMatrix_free(work->data->A);
      OSQPVectorf_free(work->data->q);
      OSQPVectorf_free(work->data->l);
      OSQPVectorf_free(work->data->u);
      c_free(work->data);
    }

       if (work->scaling){
      OSQPVectorf_free(work->scaling->D);
      OSQPVectorf_free(work->scaling->Dinv);
      OSQPVectorf_free(work->scaling->E);
      OSQPVectorf_free(work->scaling->Einv);
    }
    c_free(work->scaling);

       OSQPVectorf_free(work->D_temp);
    OSQPVectorf_free(work->D_temp_A);
    OSQPVectorf_free(work->E_temp);

       if (work->linsys_solver) {
      if (work->linsys_solver->free) {
        work->linsys_solver->free(work->linsys_solver);
      }
    }

#ifndef OSQP_EMBEDDED_MODE
       if (work->pol) {
      OSQPVectori_free(work->pol->active_flags);
      OSQPVectorf_free(work->pol->x);
      OSQPVectorf_free(work->pol->z);
      OSQPVectorf_free(work->pol->y);
      c_free(work->pol);
    }
#endif

       OSQPVectorf_free(work->rho_vec);
    OSQPVectorf_free(work->rho_inv_vec);
#if OSQP_EMBEDDED_MODE != 1
    OSQPVectori_free(work->constr_type);
#endif
    OSQPVectorf_free(work->x);
    OSQPVectorf_free(work->z);
    OSQPVectorf_free(work->xz_tilde);
    OSQPVectorf_view_free(work->xtilde_view);
    OSQPVectorf_view_free(work->ztilde_view);
    OSQPVectorf_free(work->x_prev);
    OSQPVectorf_free(work->z_prev);
    OSQPVectorf_free(work->y);
    OSQPVectorf_free(work->Ax);
    OSQPVectorf_free(work->Px);
    OSQPVectorf_free(work->Aty);
    OSQPVectorf_free(work->delta_y);
    OSQPVectorf_free(work->Atdelta_y);
    OSQPVectorf_free(work->delta_x);
    OSQPVectorf_free(work->Pdelta_x);
    OSQPVectorf_free(work->Adelta_x);

       if (solver->settings) c_free(solver->settings);

       if (solver->solution) {
      c_free(solver->solution->x);
      c_free(solver->solution->y);
      c_free(solver->solution->prim_inf_cert);
      c_free(solver->solution->dual_inf_cert);
      c_free(solver->solution);
    }

       if (solver->info) c_free(solver->info);

# ifdef OSQP_ENABLE_PROFILING
       if (work->timer) OSQPTimer_free(work->timer);
# endif

# ifdef OSQP_ENABLE_DERIVATIVES
      if (work->derivative_data){
          if (work->derivative_data->y_l) OSQPVectorf_free(work->derivative_data->y_l);
          if (work->derivative_data->y_u) OSQPVectorf_free(work->derivative_data->y_u);
          if (work->derivative_data->ryl) OSQPVectorf_free(work->derivative_data->ryl);
          if (work->derivative_data->ryu) OSQPVectorf_free(work->derivative_data->ryu);
          if (work->derivative_data->rhs) OSQPVectorf_free(work->derivative_data->rhs);
          c_free(work->derivative_data);
      }
#endif

       c_free(work);
  }

   c_free(solver);

  return exitflag;
}

#endif




OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                             const OSQPFloat* q_new,
                             const OSQPFloat* l_new,
                             const OSQPFloat* u_new) {

  OSQPInt exitflag = 0;

  OSQPVectorf*   l_tmp;
  OSQPVectorf*   u_tmp;
  OSQPWorkspace* work;

   if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
   osqp_tic(work->timer);
#endif

   if (l_new || u_new) {
       l_tmp = work->z_prev;
    u_tmp = work->delta_y;

       if (l_new) OSQPVectorf_from_raw(l_tmp, l_new);
    if (u_new) OSQPVectorf_from_raw(u_tmp, u_new);

    if (solver->settings->scaling) {
      if (l_new) OSQPVectorf_ew_prod(l_tmp, l_tmp, work->scaling->E);
      if (u_new) OSQPVectorf_ew_prod(u_tmp, u_tmp, work->scaling->E);
    }

           if (l_new && u_new) exitflag = !OSQPVectorf_all_leq(l_tmp, u_tmp);
      else if (l_new)     exitflag = !OSQPVectorf_all_leq(l_tmp, work->data->u);
      else                exitflag = !OSQPVectorf_all_leq(work->data->l, u_tmp);
      if (exitflag) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

           if (l_new) swap_vectors(&work->z_prev,  &work->data->l);
      if (u_new) swap_vectors(&work->delta_y, &work->data->u);

#if OSQP_EMBEDDED_MODE != 1
           if (solver->settings->rho_is_vec) exitflag = update_rho_vec(solver);
#endif
  }

   if (q_new) {
    OSQPVectorf_from_raw(work->data->q, q_new);
    if (solver->settings->scaling) {
      OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->D);
      OSQPVectorf_mult_scalar(work->data->q, work->scaling->c);
    }
  }

   reset_info(solver->info);

#ifdef OSQP_ENABLE_PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif

  return exitflag;
}


OSQPInt osqp_warm_start(OSQPSolver*      solver,
                        const OSQPFloat* x,
                        const OSQPFloat* y) {

  OSQPWorkspace* work;

   if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

   if (!solver->settings->warm_starting) solver->settings->warm_starting = 1;

   if (x) OSQPVectorf_from_raw(work->x, x);
  if (y) OSQPVectorf_from_raw(work->y, y);

   if (solver->settings->scaling) {
    if (x) OSQPVectorf_ew_prod(work->x, work->x, work->scaling->Dinv);
    if (y) {
      OSQPVectorf_ew_prod(work->y, work->y, work->scaling->Einv);
      OSQPVectorf_mult_scalar(work->y, work->scaling->c);
    }
  }

   if (x) OSQPMatrix_Axpy(work->data->A, work->x, work->z, 1.0, 0.0);

   work->linsys_solver->warm_start(work->linsys_solver, work->x);

  return 0;
}


void osqp_cold_start(OSQPSolver *solver) {
  OSQPWorkspace *work = solver->work;
  OSQPVectorf_set_scalar(work->x, 0.);
  OSQPVectorf_set_scalar(work->z, 0.);
  OSQPVectorf_set_scalar(work->y, 0.);

   work->linsys_solver->warm_start(work->linsys_solver, work->x);
}


#if OSQP_EMBEDDED_MODE != 1

OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                             const OSQPFloat* Px_new,
                             const OSQPInt*   Px_new_idx,
                             OSQPInt          P_new_n,
                             const OSQPFloat* Ax_new,
                             const OSQPInt*   Ax_new_idx,
                             OSQPInt          A_new_n) {

  OSQPInt exitflag;
  OSQPInt nnzP, nnzA;
  OSQPWorkspace *work;

   if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  osqp_tic(work->timer);
#endif

  nnzP = OSQPMatrix_get_nz(work->data->P);
  nnzA = OSQPMatrix_get_nz(work->data->A);


   if (P_new_n > nnzP || P_new_n < 0) {
    c_eprint("new number of elements (%i) out of bounds for P (%i max)",
             (int)P_new_n, (int)nnzP);
    return 1;
  }
   if(Px_new_idx == OSQP_NULL && P_new_n != 0 && P_new_n != nnzP){
        c_eprint("index vector is required for partial updates of P");
        return 1;
  }

   if(P_new_n == 0) {
    P_new_n = nnzP;
  }

   if (A_new_n > nnzA || A_new_n < 0) {
    c_eprint("new number of elements (%i) out of bounds for A (%i max)",
             (int)A_new_n,
             (int)nnzA);
    return 2;
  }
   if(Ax_new_idx == OSQP_NULL && A_new_n != 0 && A_new_n != nnzA){
    c_eprint("index vector is required for partial updates of A");
    return 2;
  }

     if(A_new_n == 0) {
    A_new_n = nnzA;
  }

  if (solver->settings->scaling) unscale_data(solver);

  if (Px_new){
    OSQPMatrix_update_values(work->data->P, Px_new, Px_new_idx, P_new_n);
  }
  if (Ax_new){
    OSQPMatrix_update_values(work->data->A, Ax_new, Ax_new_idx, A_new_n);
  }

  if (solver->settings->scaling) scale_data(solver);

    if(solver->settings->scaling){
    exitflag = work->linsys_solver->update_matrices(
                  work->linsys_solver,
                  work->data->P, OSQP_NULL, nnzP,
                  work->data->A, OSQP_NULL, nnzA);
  }
  else{
    exitflag = work->linsys_solver->update_matrices(
                  work->linsys_solver,
                  work->data->P, Px_new_idx, P_new_n,
                  work->data->A, Ax_new_idx, A_new_n);
  }


   reset_info(solver->info);

  if (exitflag != 0){c_eprint("new KKT matrix is not quasidefinite");}

#ifdef OSQP_ENABLE_PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif

  return exitflag;
}


OSQPInt osqp_update_rho(OSQPSolver* solver,
                        OSQPFloat     rho_new) {

    OSQPInt exitflag;
    OSQPWorkspace *work;

       if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
    work = solver->work;

   if (rho_new <= 0) {
    c_eprint("rho must be positive");
    return 1;
  }

#ifdef OSQP_ENABLE_PROFILING
  if (work->rho_update_from_solve == 0) {
    if (work->clear_update_time == 1) {
      work->clear_update_time = 0;
      solver->info->update_time = 0.0;
    }
    osqp_tic(work->timer);
  }
#endif

   solver->settings->rho = c_min(c_max(rho_new, OSQP_RHO_MIN), OSQP_RHO_MAX);

  if (solver->settings->rho_is_vec) {
       OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                       work->constr_type,
                                       OSQP_RHO_MIN,
                                       solver->settings->rho,
                                       OSQP_RHO_EQ_OVER_RHO_INEQ*solver->settings->rho);

    OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);
  }
  else {
    work->rho_inv = 1. / solver->settings->rho;
  }

   exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);

#ifdef OSQP_ENABLE_PROFILING
  if (work->rho_update_from_solve == 0)
    solver->info->update_time += osqp_toc(work->timer);
#endif

  return exitflag;
}

#endif




OSQPInt osqp_update_settings(OSQPSolver*         solver,
                             const OSQPSettings* new_settings) {

  OSQPSettings* settings = solver->settings;

   if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

   if (validate_settings(new_settings, 0)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);


   settings->profiler_level = new_settings->profiler_level;
  osqp_profiler_update_level(settings->profiler_level);

  settings->verbose       = new_settings->verbose;
  settings->warm_starting = new_settings->warm_starting;
   settings->polishing     = new_settings->polishing;

     settings->alpha = new_settings->alpha;

  settings->cg_max_iter      = new_settings->cg_max_iter;
  settings->cg_tol_reduction = new_settings->cg_tol_reduction;
  settings->cg_tol_fraction  = new_settings->cg_tol_fraction;
  settings->cg_precond       = new_settings->cg_precond;


  settings->max_iter           = new_settings->max_iter;
  settings->eps_abs            = new_settings->eps_abs;
  settings->eps_rel            = new_settings->eps_rel;
  settings->eps_prim_inf       = new_settings->eps_prim_inf;
  settings->eps_dual_inf       = new_settings->eps_dual_inf;
  settings->scaled_termination = new_settings->scaled_termination;
  settings->check_termination  = new_settings->check_termination;
  settings->check_dualgap      = new_settings->check_dualgap;
  settings->time_limit         = new_settings->time_limit;

  settings->delta              = new_settings->delta;
  settings->polish_refine_iter = new_settings->polish_refine_iter;

   solver->work->linsys_solver->update_settings(solver->work->linsys_solver, settings);

  return 0;
}



OSQPInt osqp_codegen(OSQPSolver*         solver,
                     const char*         output_dir,
                     const char*         file_prefix,
                     OSQPCodegenDefines* defines){

  OSQPInt exitflag = 0;

#ifdef OSQP_CODEGEN
  if (!solver || !solver->work || !solver->settings || !solver->info) {
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  }
   else if (solver->info->status_val == OSQP_NON_CVX) {
    return osqp_error(OSQP_NONCVX_ERROR);
  }
   else if (!solver->work->data || !solver->work->linsys_solver) {
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  }
  else if (!defines || (defines->embedded_mode != 1    && defines->embedded_mode != 2)
                    || (defines->float_type != 0       && defines->float_type != 1)
                    || (defines->printing_enable != 0  && defines->printing_enable != 1)
                    || (defines->profiling_enable != 0 && defines->profiling_enable != 1)
                    || (defines->interrupt_enable != 0 && defines->interrupt_enable != 1)
                    || (defines->derivatives_enable != 0 && defines->derivatives_enable != 1)) {
    return osqp_error(OSQP_CODEGEN_DEFINES_ERROR);
  }

  exitflag = codegen_inc(output_dir, file_prefix);
  if (!exitflag) exitflag = codegen_src(output_dir, file_prefix, solver, defines->embedded_mode);
  if (!exitflag) exitflag = codegen_example(output_dir, file_prefix);
  if (!exitflag) exitflag = codegen_defines(output_dir, defines);
#else
  OSQP_UnusedVar(solver);
  OSQP_UnusedVar(output_dir);
  OSQP_UnusedVar(file_prefix);
  OSQP_UnusedVar(defines);
  exitflag = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return exitflag;
}

OSQPInt osqp_adjoint_derivative_compute(OSQPSolver* solver,
                                        OSQPFloat*  dx,
                                        OSQPFloat*  dy) {
  OSQPInt status = 0;

#ifdef OSQP_ENABLE_DERIVATIVES
  status = adjoint_derivative_compute(solver, dx, dy, dy);
#else
  OSQP_UnusedVar(solver);
  OSQP_UnusedVar(dx);
  OSQP_UnusedVar(dy);
  status = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return status;
}

OSQPInt osqp_adjoint_derivative_get_mat(OSQPSolver*    solver,
                                        OSQPCscMatrix* dP,
                                        OSQPCscMatrix* dA) {
  OSQPInt status = 0;

#ifdef OSQP_ENABLE_DERIVATIVES
  status = adjoint_derivative_get_mat(solver, dP, dA);
#else
  OSQP_UnusedVar(solver);
  OSQP_UnusedVar(dP);
  OSQP_UnusedVar(dA);
  status = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return status;
}

OSQPInt osqp_adjoint_derivative_get_vec(OSQPSolver* solver,
                                        OSQPFloat*  dq,
                                        OSQPFloat*  dl,
                                        OSQPFloat*  du) {
  OSQPInt status = 0;

#ifdef OSQP_ENABLE_DERIVATIVES
  status = adjoint_derivative_get_vec(solver, dq, dl, du);
#else
  OSQP_UnusedVar(solver);
  OSQP_UnusedVar(dq);
  OSQP_UnusedVar(dl);
  OSQP_UnusedVar(du);
  status = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return status;
}
