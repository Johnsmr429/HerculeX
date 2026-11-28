#ifndef OSQP_API_TYPES_H
#define OSQP_API_TYPES_H


# include "osqp_api_constants.h"


# ifdef OSQP_USE_LONG
typedef long long OSQPInt;
# else
typedef int OSQPInt;
# endif


# ifndef OSQP_USE_FLOAT
typedef double OSQPFloat;
# else
typedef float OSQPFloat;
# endif

#ifdef OSQP_PACK_SETTINGS
#define OSQP_ATTR_PACK __attribute__((packed))
#else
#define OSQP_ATTR_PACK
#endif

typedef struct {
  OSQPInt    m;
  OSQPInt    n;
  OSQPInt   *p;
  OSQPInt   *i;
  OSQPFloat *x;
  OSQPInt    nzmax;
  OSQPInt    nz;
  OSQPInt    owned;
} OSQPCscMatrix;

typedef struct OSQP_ATTR_PACK {
    OSQPInt device;
  enum osqp_linsys_solver_type linsys_solver;

   OSQPInt allocate_solution;
  OSQPInt verbose;
  OSQPInt profiler_level;
  OSQPInt warm_starting;
  OSQPInt scaling;
  OSQPInt polishing;

   OSQPFloat rho;
  OSQPInt   rho_is_vec;
  OSQPFloat sigma;
  OSQPFloat alpha;

   OSQPInt           cg_max_iter;
  OSQPInt           cg_tol_reduction;
  OSQPFloat         cg_tol_fraction;
  osqp_precond_type cg_precond;

    OSQPInt adaptive_rho;

   OSQPInt adaptive_rho_interval;

   OSQPFloat adaptive_rho_fraction;

   OSQPFloat adaptive_rho_tolerance;

   OSQPInt   max_iter;
  OSQPFloat eps_abs;
  OSQPFloat eps_rel;
  OSQPFloat eps_prim_inf;
  OSQPFloat eps_dual_inf;
  OSQPInt   scaled_termination;
  OSQPInt   check_termination;
  OSQPInt   check_dualgap;
  OSQPFloat time_limit;

   OSQPFloat delta;
  OSQPInt   polish_refine_iter;
} OSQPSettings;


typedef struct {
   char    status[32];
  OSQPInt status_val;
  OSQPInt status_polish;

   OSQPFloat obj_val;
  OSQPFloat dual_obj_val;
  OSQPFloat prim_res;
  OSQPFloat dual_res;
  OSQPFloat duality_gap;

   OSQPInt   iter;
  OSQPInt   rho_updates;
  OSQPFloat rho_estimate;

   OSQPFloat setup_time;
  OSQPFloat solve_time;
  OSQPFloat update_time;
  OSQPFloat polish_time;
  OSQPFloat run_time;

   OSQPFloat primdual_int;
  OSQPFloat rel_kkt_error;
} OSQPInfo;


typedef struct {
  OSQPFloat* x;
  OSQPFloat* y;
  OSQPFloat* prim_inf_cert;
  OSQPFloat* dual_inf_cert;
} OSQPSolution;


typedef struct OSQPWorkspace_ OSQPWorkspace;


typedef struct {
   OSQPSettings*  settings;
  OSQPSolution*  solution;
  OSQPInfo*      info;
  OSQPWorkspace* work;
} OSQPSolver;



typedef struct {
  OSQPInt embedded_mode;
  OSQPInt float_type;
  OSQPInt printing_enable;
  OSQPInt profiling_enable;
  OSQPInt interrupt_enable;
  OSQPInt derivatives_enable;
} OSQPCodegenDefines;

#endif
