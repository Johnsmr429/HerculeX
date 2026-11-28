#ifndef ALGEBRA_IMPL_H
#define ALGEBRA_IMPL_H

#include "csc_math.h"

#ifdef __cplusplus
extern "C" {
#endif


struct OSQPVectori_ {
  OSQPInt* values;
  OSQPInt  length;
};

struct OSQPVectorf_ {
  OSQPFloat* values;
  OSQPInt    length;
};



typedef enum OSQPMatrix_symmetry_type {NONE,TRIU} OSQPMatrix_symmetry_type;

struct OSQPMatrix_ {
  OSQPCscMatrix*           csc;
  OSQPMatrix_symmetry_type symmetry;
};

#ifdef __cplusplus
}
#endif

#endif
