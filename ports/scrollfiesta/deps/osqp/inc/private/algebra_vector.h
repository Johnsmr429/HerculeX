#ifndef ALGEBRA_VECTOR_H
#define ALGEBRA_VECTOR_H

#include "osqp_api_types.h"

#include "glob_opts.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct OSQPVectori_ OSQPVectori;

typedef struct OSQPVectorf_ OSQPVectorf;



# ifndef OSQP_EMBEDDED_MODE

OSQPInt OSQPVectorf_is_eq(const OSQPVectorf* A,
                          const OSQPVectorf* B,
                          OSQPFloat          tol);

OSQPVectorf* OSQPVectorf_malloc(OSQPInt length);
OSQPVectorf* OSQPVectorf_calloc(OSQPInt length);
OSQPVectori* OSQPVectori_malloc(OSQPInt length);
OSQPVectori* OSQPVectori_calloc(OSQPInt length);

OSQPVectorf* OSQPVectorf_new(const OSQPFloat* a,
                             OSQPInt          length);

OSQPVectori* OSQPVectori_new(const OSQPInt* a,
                             OSQPInt        length);

OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf* a);

void OSQPVectorf_free(OSQPVectorf* a);

void OSQPVectori_free(OSQPVectori* a);

void OSQPVectorf_subvector_assign(OSQPVectorf* A,
                                  OSQPFloat*   b,
                                  OSQPInt      start,
                                  OSQPInt      length,
                                  OSQPFloat    multiplier);

void OSQPVectorf_subvector_assign_scalar(OSQPVectorf* A,
                                  OSQPFloat    sc,
                                  OSQPInt      start,
                                  OSQPInt      length);

OSQPVectorf* OSQPVectorf_subvector_byrows(const OSQPVectorf* A,
                                          const OSQPVectori* rows);

OSQPVectorf* OSQPVectorf_concat(const OSQPVectorf* A,
                                const OSQPVectorf* B);

OSQPVectorf* OSQPVectorf_view(const OSQPVectorf* a,
                              OSQPInt            head,
                              OSQPInt            length);

void OSQPVectorf_view_update(OSQPVectorf* a, const OSQPVectorf* b, OSQPInt head, OSQPInt length);

void OSQPVectorf_view_free(OSQPVectorf* a);

# endif


OSQPInt OSQPVectorf_length(const OSQPVectorf* a);

OSQPInt OSQPVectori_length(const OSQPVectori* a);

OSQPFloat* OSQPVectorf_data(const OSQPVectorf* a);

void OSQPVectorf_copy(OSQPVectorf*       b,
                      const OSQPVectorf* a);

void OSQPVectorf_from_raw(OSQPVectorf*     b,
                          const OSQPFloat* a);

void OSQPVectori_from_raw(OSQPVectori*   b,
                          const OSQPInt* a);

void OSQPVectorf_to_raw(OSQPFloat*         bv,
                        const OSQPVectorf* a);

void OSQPVectori_to_raw(OSQPInt*           bv,
                        const OSQPVectori* a);

void OSQPVectorf_set_scalar(OSQPVectorf* a,
                            OSQPFloat    sc);

void OSQPVectorf_set_scalar_conditional(OSQPVectorf*       a,
                                        const OSQPVectori* test,
                                        OSQPFloat          val_if_neg,
                                        OSQPFloat          val_if_zero,
                                        OSQPFloat          val_if_pos);

void OSQPVectorf_round_to_zero(OSQPVectorf* a,
                               OSQPFloat    tol);

void OSQPVectorf_mult_scalar(OSQPVectorf* a,
                             OSQPFloat    sc);

void OSQPVectorf_plus(OSQPVectorf*       x,
                      const OSQPVectorf* a,
                      const OSQPVectorf* b);

void OSQPVectorf_minus(OSQPVectorf*      x,
                      const OSQPVectorf* a,
                      const OSQPVectorf* b);

void OSQPVectorf_add_scaled(OSQPVectorf*       x,
                            OSQPFloat          sca,
                            const OSQPVectorf* a,
                            OSQPFloat          scb,
                            const OSQPVectorf* b);

void OSQPVectorf_add_scaled3(OSQPVectorf*       x,
                             OSQPFloat          sca,
                             const OSQPVectorf* a,
                             OSQPFloat          scb,
                             const OSQPVectorf* b,
                             OSQPFloat          scc,
                             const OSQPVectorf* c);

OSQPFloat OSQPVectorf_norm_inf(const OSQPVectorf* v);

OSQPFloat OSQPVectorf_scaled_norm_inf(const OSQPVectorf* S,
                                      const OSQPVectorf* v);

OSQPFloat OSQPVectorf_norm_inf_diff(const OSQPVectorf* a,
                                    const OSQPVectorf* b);

OSQPFloat OSQPVectorf_norm_2(const OSQPVectorf* v);

OSQPFloat OSQPVectorf_norm_1(const OSQPVectorf* a);

OSQPFloat OSQPVectorf_dot_prod(const OSQPVectorf* a,
                               const OSQPVectorf* b);

OSQPFloat OSQPVectorf_dot_prod_signed(const OSQPVectorf* a,
                                      const OSQPVectorf* b,
                                      OSQPInt            sign);

void OSQPVectorf_ew_prod(OSQPVectorf*       c,
                         const OSQPVectorf* a,
                         const OSQPVectorf* b);

OSQPInt OSQPVectorf_all_leq(const OSQPVectorf* l,
                            const OSQPVectorf* u);

void OSQPVectorf_ew_bound_vec(OSQPVectorf*       x,
                              const OSQPVectorf* z,
                              const OSQPVectorf* l,
                              const OSQPVectorf* u);


void OSQPVectorf_project_polar_reccone(OSQPVectorf*       y,
                                       const OSQPVectorf* l,
                                       const OSQPVectorf* u,
                                       OSQPFloat          infval);

OSQPInt OSQPVectorf_in_reccone(const OSQPVectorf* y,
                               const OSQPVectorf* l,
                               const OSQPVectorf* u,
                               OSQPFloat          infval,
                               OSQPFloat          tol);

# if OSQP_EMBEDDED_MODE != 1

void OSQPVectorf_ew_reciprocal(OSQPVectorf*       b,
                               const OSQPVectorf* a);

void OSQPVectorf_ew_sqrt(OSQPVectorf* a);

void OSQPVectorf_ew_max_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b);

void OSQPVectorf_ew_min_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b);

OSQPInt OSQPVectorf_ew_bounds_type(OSQPVectori*       iseq,
                                   const OSQPVectorf* l,
                                   const OSQPVectorf* u,
                                   OSQPFloat          tol,
                                   OSQPFloat          infval);


void OSQPVectorf_set_scalar_if_lt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval);

void OSQPVectorf_set_scalar_if_gt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval);

# endif

#ifdef __cplusplus
}
#endif

#endif
