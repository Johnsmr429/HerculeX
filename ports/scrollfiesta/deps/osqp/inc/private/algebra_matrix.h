#ifndef ALGEBRA_MATRIX_H
#define ALGEBRA_MATRIX_H

#include "osqp_api_types.h"

#include "glob_opts.h"
#include "algebra_vector.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct OSQPMatrix_ OSQPMatrix;


#ifndef OSQP_EMBEDDED_MODE


OSQPInt OSQPMatrix_is_eq(const OSQPMatrix* A,
                         const OSQPMatrix* B,
                               OSQPFloat   tol);



OSQPMatrix* OSQPMatrix_new_from_csc(const OSQPCscMatrix* A,
                                          OSQPInt        is_triu);

OSQPCscMatrix* OSQPMatrix_get_csc(const OSQPMatrix* M);

OSQPMatrix* OSQPMatrix_copy_new(const OSQPMatrix* A);

OSQPMatrix* OSQPMatrix_triu_to_symm(const OSQPMatrix* A);

OSQPMatrix* OSQPMatrix_vstack(const OSQPMatrix* A, const OSQPMatrix* B);

#endif




void OSQPMatrix_update_values(OSQPMatrix*      M,
                              const OSQPFloat* Mx_new,
                              const OSQPInt*   Mx_new_idx,
                              OSQPInt          M_new_n);

OSQPInt    OSQPMatrix_get_m(const OSQPMatrix* M);

OSQPInt    OSQPMatrix_get_n(const OSQPMatrix* M);

OSQPFloat* OSQPMatrix_get_x(const OSQPMatrix* M);

OSQPInt*   OSQPMatrix_get_i(const OSQPMatrix* M);

OSQPInt*   OSQPMatrix_get_p(const OSQPMatrix* M);

OSQPInt    OSQPMatrix_get_nz(const OSQPMatrix* M);


void OSQPMatrix_mult_scalar(OSQPMatrix* A,
                            OSQPFloat     sc);

void OSQPMatrix_lmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* L);

void OSQPMatrix_rmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* R);

void OSQPMatrix_AtDA_extract_diag(const OSQPMatrix*  A,
                                  const OSQPVectorf* D,
                                        OSQPVectorf* d);

void OSQPMatrix_extract_diag(const OSQPMatrix*  A,
                                   OSQPVectorf* d);

void OSQPMatrix_Axpy(const OSQPMatrix*  A,
                     const OSQPVectorf* x,
                     OSQPVectorf*       y,
                     OSQPFloat          alpha,
                     OSQPFloat          beta);

void OSQPMatrix_Atxpy(const OSQPMatrix*  A,
                      const OSQPVectorf* x,
                      OSQPVectorf*       y,
                      OSQPFloat          alpha,
                      OSQPFloat          beta);


#if OSQP_EMBEDDED_MODE != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E);

void OSQPMatrix_row_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E);

#endif

#ifndef OSQP_EMBEDDED_MODE

void OSQPMatrix_free(OSQPMatrix* M);

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix*  A,
                                        const OSQPVectori* rows);

#endif

#ifdef __cplusplus
}
#endif

#endif
