#ifndef UNION_FIND_INCLUDED
#define UNION_FIND_INCLUDED

#include <stdint.h>
#include "arena.h"

typedef struct {
    int32_t *parent;
    int32_t *rank;
    int32_t  count;
    int32_t  n;
} UnionFind;

UnionFind UF_new(Arena_T arena, int32_t n);

int32_t uf_find(UnionFind *uf, int32_t x);

void uf_union(UnionFind *uf, int32_t a, int32_t b);

#endif
