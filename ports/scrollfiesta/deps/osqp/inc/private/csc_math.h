#ifndef CSC_MATH_H
# define CSC_MATH_H


# include "osqp_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif



void csc_update_values(OSQPCscMatrix*   M,
                       const OSQPFloat* Mx_new,
                       const OSQPInt*   Mx_new_idx,
                       OSQPInt          P_new_n);


void csc_scale(OSQPCscMatrix* A, OSQPFloat sc);

void csc_lmult_diag(OSQPCscMatrix* A, const OSQPFloat* L);

void csc_rmult_diag(OSQPCscMatrix* A, const OSQPFloat* R);

void csc_AtDA_extract_diag(const OSQPCscMatrix* A,
                           const OSQPFloat*     D,
                                 OSQPFloat*     d);

void csc_Axpy_sym_triu(const OSQPCscMatrix* A,
                       const OSQPFloat*     x,
                             OSQPFloat*     y,
                             OSQPFloat      alpha,
                             OSQPFloat      beta);

void csc_Axpy(const OSQPCscMatrix* A,
              const OSQPFloat*     x,
                    OSQPFloat*     y,
                    OSQPFloat      alpha,
                    OSQPFloat      beta);

void csc_Atxpy(const OSQPCscMatrix* A,
               const OSQPFloat*     x,
                     OSQPFloat*     y,
                     OSQPFloat      alpha,
                     OSQPFloat      beta);


void csc_col_norm_inf(const OSQPCscMatrix* M, OSQPFloat* E);

void csc_row_norm_inf(const OSQPCscMatrix* M, OSQPFloat* E);

void csc_row_norm_inf_sym_triu(const OSQPCscMatrix* M, OSQPFloat* E);

#ifdef __cplusplus
}
#endif

#endif
