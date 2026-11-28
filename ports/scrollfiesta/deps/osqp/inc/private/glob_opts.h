#ifndef GLOB_OPTS_H
#define GLOB_OPTS_H


#include "osqp_configure.h"

#ifdef __cplusplus
extern "C" {
#endif


# ifndef OSQP_EMBEDDED_MODE
#  ifdef OSQP_CUSTOM_MEMORY
#    include OSQP_CUSTOM_MEMORY
#  else
#   include <stdlib.h>
#   define c_malloc  malloc
#   define c_calloc  calloc
#   define c_free    free
#   define c_realloc realloc
#  endif
# endif



# ifndef c_absval
#  define c_absval(x) (((x) < 0) ? -(x) : (x))
# endif

# ifndef c_max
#  define c_max(a, b) (((a) > (b)) ? (a) : (b))
# endif

# ifndef c_min
#  define c_min(a, b) (((a) < (b)) ? (a) : (b))
# endif

# ifndef c_roundmultiple
#  define c_roundmultiple(x, N) ((x) + .5 * (N)-c_fmod((x) + .5 * (N), (N)))
# endif



# if OSQP_EMBEDDED_MODE != 1

#  include <math.h>
#  ifndef OSQP_USE_FLOAT
#   define c_sqrt sqrt
#   define c_fmod fmod
#  else
#   define c_sqrt sqrtf
#   define c_fmod fmodf
#  endif

# endif

#ifdef __cplusplus
}
#endif

#endif
