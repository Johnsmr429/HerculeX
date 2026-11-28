#ifndef  OSQP_EXPORT_DEFINE_H
#define  OSQP_EXPORT_DEFINE_H

#if defined(_WIN32)
#  if defined(BUILDING_OSQP)
#    define OSQP_API_EXPORT __declspec(dllexport)
#  else
#    define OSQP_API_EXPORT __declspec(dllimport)
#  endif
#else
#  if defined(BUILDING_OSQP)
#    define OSQP_API_EXPORT __attribute__((visibility("default")))
#  else
#    define OSQP_API_EXPORT
#  endif
#endif

#if defined(OSQP_SHARED_LIB)
#  define OSQP_API OSQP_API_EXPORT
#else
#  define OSQP_API
#endif

#endif
