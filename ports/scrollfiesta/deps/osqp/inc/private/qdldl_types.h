
#ifndef QDLDL_TYPES_H
# define QDLDL_TYPES_H

#include "osqp_api_types.h"

# ifdef __cplusplus
extern "C" {
# endif

#include <limits.h>

typedef OSQPInt    QDLDL_int;
typedef OSQPFloat  QDLDL_float;

typedef int   QDLDL_bool;

#ifdef OSQP_USE_LONG
#define QDLDL_INT_MAX LLONG_MAX
#else
#define QDLDL_INT_MAX INT_MAX
#endif

# ifdef __cplusplus
}
# endif

#endif
