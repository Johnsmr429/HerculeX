#ifndef LIN_ALG_H
#define LIN_ALG_H

# include "algebra_vector.h"
# include "algebra_matrix.h"
# include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSQPInt osqp_algebra_linsys_supported(void);

enum osqp_linsys_solver_type osqp_algebra_default_linsys(void);

OSQPInt osqp_algebra_init_libs(OSQPInt device);

void osqp_algebra_free_libs(void);

OSQPInt osqp_algebra_name(char* name, OSQPInt nameLen);

OSQPInt osqp_algebra_device_name(char* name, OSQPInt nameLen);



OSQPInt osqp_algebra_init_linsys_solver(LinSysSolver**      s,
                                        const OSQPMatrix*   P,
                                        const OSQPMatrix*   A,
                                        const OSQPVectorf*  rho_vec,
                                        const OSQPSettings* settings,
                                        OSQPFloat*          scaled_prim_res,
                                        OSQPFloat*          scaled_dual_res,
                                        OSQPInt             polishing);


#ifdef OSQP_ALGEBRA_BUILTIN
#ifndef OSQP_EMBEDDED_MODE
OSQPInt adjoint_derivative_linsys_solver(LinSysSolver**      s,
                                         const OSQPSettings* settings,
                                         const OSQPMatrix*   P,
                                         const OSQPMatrix*   G,
                                         const OSQPMatrix*   A_eq,
                                         OSQPMatrix*         GDiagLambda,
                                         OSQPVectorf*        slacks,
                                         OSQPVectorf*        rhs);

#endif
#endif

#ifdef __cplusplus
}
#endif

#endif
