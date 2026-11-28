#ifndef  OSQP_API_FUNCTIONS_H
#define  OSQP_API_FUNCTIONS_H

# include "osqp_api_types.h"
# include "osqp_export_define.h"

# ifdef __cplusplus
extern "C" {
# endif




#ifndef OSQP_EMBEDDED_MODE
OSQPCscMatrix* OSQPCscMatrix_new(OSQPInt    m,
                                 OSQPInt    n,
                                 OSQPInt    nzmax,
                                 OSQPFloat* x,
                                 OSQPInt*   i,
                                 OSQPInt*   p);

void OSQPCscMatrix_free(OSQPCscMatrix* mat);

OSQPCscMatrix* OSQPCscMatrix_zeros(OSQPInt m,
                                   OSQPInt n);

OSQPCscMatrix* OSQPCscMatrix_identity(OSQPInt m);

OSQPCscMatrix* OSQPCscMatrix_diag_scalar(OSQPInt   m,
                                         OSQPInt   n,
                                         OSQPFloat scalar);

OSQPCscMatrix* OSQPCscMatrix_diag_vec(OSQPInt    m,
                                      OSQPInt    n,
                                      OSQPFloat* vals);

#endif

OSQP_API void OSQPCscMatrix_set_data(OSQPCscMatrix* M,
                                     OSQPInt        m,
                                     OSQPInt        n,
                                     OSQPInt        nzmax,
                                     OSQPFloat*     x,
                                     OSQPInt*       i,
                                     OSQPInt*       p);



#ifndef OSQP_EMBEDDED_MODE
OSQPSettings* OSQPSettings_new();

void OSQPSettings_free(OSQPSettings* settings);
#endif



#ifndef OSQP_EMBEDDED_MODE
OSQPCodegenDefines* OSQPCodegenDefines_new();

void OSQPCodegenDefines_free(OSQPCodegenDefines* defs);
#endif





OSQP_API OSQPInt osqp_capabilities(void);

OSQP_API const char* osqp_version(void);


OSQP_API const char* osqp_error_message(OSQPInt error_flag);


OSQP_API void osqp_get_dimensions(OSQPSolver* solver,
                                  OSQPInt*    m,
                                  OSQPInt*    n);

# ifndef OSQP_EMBEDDED_MODE

OSQP_API OSQPInt osqp_setup(OSQPSolver**         solverp,
                            const OSQPCscMatrix* P,
                            const OSQPFloat*     q,
                            const OSQPCscMatrix* A,
                            const OSQPFloat*     l,
                            const OSQPFloat*     u,
                            OSQPInt              m,
                            OSQPInt              n,
                            const OSQPSettings*  settings);

# endif

OSQP_API OSQPInt osqp_solve(OSQPSolver* solver);

OSQPInt osqp_get_solution(OSQPSolver* solver, OSQPSolution* solution);

# ifndef OSQP_EMBEDDED_MODE

OSQP_API OSQPInt osqp_cleanup(OSQPSolver* solver);

# endif






OSQP_API OSQPInt osqp_warm_start(OSQPSolver*      solver,
                                 const OSQPFloat* x,
                                 const OSQPFloat* y);

OSQP_API void osqp_cold_start(OSQPSolver* solver);

OSQP_API OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                                      const OSQPFloat* q_new,
                                      const OSQPFloat* l_new,
                                      const OSQPFloat* u_new);

# if OSQP_EMBEDDED_MODE != 1

OSQP_API OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                                      const OSQPFloat* Px_new,
                                      const OSQPInt*   Px_new_idx,
                                      OSQPInt          P_new_n,
                                      const OSQPFloat* Ax_new,
                                      const OSQPInt*   Ax_new_idx,
                                      OSQPInt          A_new_n);


# endif




OSQP_API void osqp_set_default_settings(OSQPSettings* settings);

OSQP_API OSQPInt osqp_update_settings(OSQPSolver*         solver,
                                      const OSQPSettings* new_settings);



# if OSQP_EMBEDDED_MODE != 1

OSQP_API OSQPInt osqp_update_rho(OSQPSolver* solver,
                                 OSQPFloat   rho_new);

# endif




OSQP_API OSQPInt osqp_adjoint_derivative_compute(OSQPSolver*    solver,
                                                 OSQPFloat*     dx,
                                                 OSQPFloat*     dy);

OSQP_API OSQPInt osqp_adjoint_derivative_get_mat(OSQPSolver*    solver,
                                                 OSQPCscMatrix* dP,
                                                 OSQPCscMatrix* dA);

OSQP_API OSQPInt osqp_adjoint_derivative_get_vec(OSQPSolver*    solver,
                                                 OSQPFloat* dq,
                                                 OSQPFloat* dl,
                                                 OSQPFloat* du);




OSQP_API void osqp_set_default_codegen_defines(OSQPCodegenDefines* defines);

OSQP_API OSQPInt osqp_codegen(OSQPSolver*         solver,
                              const char*         output_dir,
                              const char*         prefix,
                              OSQPCodegenDefines* defines);


# ifdef __cplusplus
}
# endif

#endif
