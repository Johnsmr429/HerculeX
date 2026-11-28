#ifndef CSR_INCLUDED
#define CSR_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include "arena.h"

typedef struct CSR_T *CSR_T;

CSR_T CSR_from_faces(Arena_T arena, const int32_t *faces, size_t nf,
                     size_t nv);

CSR_T CSR_from_coo(Arena_T arena, const int32_t *rows, const int32_t *cols,
                   const float *vals, size_t nnz, size_t nrows);

int32_t  CSR_nrows(const CSR_T csr);
int32_t  CSR_nnz(const CSR_T csr);
const int32_t *CSR_offset(const CSR_T csr);
const int32_t *CSR_target(const CSR_T csr);
const float   *CSR_weight(const CSR_T csr);

CSR_T CSR_uniform_laplacian(Arena_T arena, const int32_t *faces, size_t nf,
                            size_t nv);

void CSR_matvec3(const CSR_T csr, const float *x, float *y);

void CSR_matvec(const CSR_T csr, const float *x, float *y);

void CSR_validate(const CSR_T csr);

#endif
