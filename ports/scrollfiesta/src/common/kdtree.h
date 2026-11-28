#ifndef KDTREE_INCLUDED
#define KDTREE_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

typedef struct KDTree_T *KDTree_T;

KDTree_T KDTree_new(Arena_T arena, const float *points, size_t n);

size_t KDTree_nearest(const KDTree_T tree, const float query[3],
                      float *out_dist_sq);

size_t KDTree_nearest_bounded(const KDTree_T tree, const float query[3],
                              float bound_sq, float *out_dist_sq);

size_t KDTree_ball_query(const KDTree_T tree, const float center[3],
                         float radius_sq,
                         int32_t *out_indices, size_t max_results);

size_t KDTree_nearest_excluding(const KDTree_T tree, const float query[3],
                                const int32_t *labels, int32_t exclude,
                                float bound_dist_sq, float *out_dist_sq);

int KDTree_selftest(void);

#endif
