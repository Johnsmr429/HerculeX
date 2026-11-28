#ifndef ATTENE_PREDICATES_WRAP_INCLUDED
#define ATTENE_PREDICATES_WRAP_INCLUDED

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AttenePredCtx AttenePredCtx;
typedef int32_t AttenePt;

AttenePredCtx *AttenePred_new(void);
void           AttenePred_free(AttenePredCtx *ctx);
void           AttenePred_reset(AttenePredCtx *ctx);

int32_t AttenePred_count(AttenePredCtx *ctx);
void    AttenePred_truncate(AttenePredCtx *ctx, int32_t n);

AttenePt AttenePred_explicit(AttenePredCtx *ctx, double x, double y, double z);

AttenePt AttenePred_edge_bisector(AttenePredCtx *ctx,
                                  AttenePt a, AttenePt b,
                                  AttenePt si, AttenePt sj);

AttenePt AttenePred_facet_bisector2(AttenePredCtx *ctx,
                                    AttenePt r, AttenePt s, AttenePt t,
                                    AttenePt si, AttenePt sj, AttenePt sk);

int AttenePred_side_bisector(AttenePredCtx *ctx, AttenePt v, AttenePt si, AttenePt sj);

void AttenePred_side_bisector_batch(AttenePredCtx *ctx, const AttenePt *v, int n,
                                    AttenePt si, AttenePt sj, int *out);

int AttenePred_xyz(AttenePredCtx *ctx, AttenePt v, double *x, double *y, double *z);

int AttenePred_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
