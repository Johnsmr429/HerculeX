#ifndef MULTICUT_WRAP_INCLUDED
#define MULTICUT_WRAP_INCLUDED

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int LiftedMulticut_kernighan_lin(
    int32_t        num_nodes,
    int32_t        num_adj_edges,
    const int32_t *adj_from,
    const int32_t *adj_to,
    int32_t        num_lifted_edges,
    const int32_t *lifted_from,
    const int32_t *lifted_to,
    const double  *lifted_weights,
    int32_t        min_clusters,
    int32_t       *out_labels,
    int32_t       *out_num_clusters);

int LiftedMulticut_selftest(void);

int LiftedMulticut_kl_selftest(void);

int LiftedMulticut_gaec_selftest(void);

#ifdef __cplusplus
}
#endif

#endif
