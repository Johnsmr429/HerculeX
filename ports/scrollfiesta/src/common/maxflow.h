#ifndef MAXFLOW_INCLUDED
#define MAXFLOW_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

typedef struct Maxflow_T *Maxflow_T;

Maxflow_T Maxflow_new(Arena_T arena, int32_t n, size_t edge_hint);

void Maxflow_add_edge(Maxflow_T mf, int32_t u, int32_t v,
                      int64_t cap, int64_t rcap);

void Maxflow_add_terminal(Maxflow_T mf, int32_t v, int64_t cap_s,
                          int64_t cap_t);

int64_t Maxflow_solve(Maxflow_T mf);

int Maxflow_in_source_side(const Maxflow_T mf, int32_t v);

int Maxflow_selftest(void);

#endif
