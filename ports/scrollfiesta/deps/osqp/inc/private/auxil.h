#ifndef AUXIL_H
#define AUXIL_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

# if OSQP_EMBEDDED_MODE != 1

OSQPFloat compute_rho_estimate(const OSQPSolver* solver);

OSQPInt adapt_rho(OSQPSolver* solver);

OSQPInt set_rho_vec(OSQPSolver* solver);

OSQPInt update_rho_vec(OSQPSolver *solver);

# endif

void swap_vectors(OSQPVectorf** a,
                  OSQPVectorf** b);


void update_xz_tilde(OSQPSolver* solver,
                     OSQPInt     admm_iter);


void update_x(OSQPSolver* solver);


void update_z(OSQPSolver* solver);


void update_y(OSQPSolver* solver);


void compute_obj_val_dual_gap(const OSQPSolver*  solver,
                              const OSQPVectorf* x,
                              const OSQPVectorf* y,
                                    OSQPFloat*   prim_obj_val,
                                    OSQPFloat*   dual_obj_val,
                                    OSQPFloat*   duality_gap);

OSQPInt has_solution(const OSQPInfo* info);

void store_solution(OSQPSolver* solver, OSQPSolution* solution);


void update_info(OSQPSolver* solver,
                 OSQPInt     iter,
                 OSQPInt     polishing);


void reset_info(OSQPInfo* info);


void update_status(OSQPInfo* info,
                   OSQPInt   status_val);


OSQPInt check_termination(OSQPSolver* solver,
                          OSQPInt     approximate);


# ifndef OSQP_EMBEDDED_MODE

OSQPInt validate_data(const OSQPCscMatrix* P,
                      const OSQPFloat*     q,
                      const OSQPCscMatrix* A,
                      const OSQPFloat*     l,
                      const OSQPFloat*     u,
                            OSQPInt        m,
                            OSQPInt        n);

# endif


OSQPInt validate_settings(const OSQPSettings* settings,
                          OSQPInt             from_setup);

#ifdef __cplusplus
}
#endif

#endif
