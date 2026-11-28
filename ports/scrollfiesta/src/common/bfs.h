#ifndef BFS_INCLUDED
#define BFS_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "csr.h"

size_t BFS_kring(const CSR_T graph, int32_t seed, int32_t k,
                 int32_t *out_vertices, size_t max_out,
                 uint32_t *dist);

void BFS_multi_source(const CSR_T graph, const int32_t *seeds,
                      size_t n_seeds, uint32_t *dist);

int BFS_augmenting_path(const int32_t *offset, const int32_t *target,
                        const int32_t *cap, const int32_t *flow,
                        int32_t n_nodes, int32_t source, int32_t sink,
                        int32_t *parent_edge,
                        int32_t *queue);

#endif
