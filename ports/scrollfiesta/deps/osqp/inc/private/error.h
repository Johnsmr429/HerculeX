#ifndef ERROR_H
#define ERROR_H


#include "osqp.h"

#ifdef __cplusplus
extern "C" {
#endif

#if __STDC_VERSION__ >= 199901L
#  define osqp_error(error_code) _osqp_error(error_code, __func__);
#else
#  define osqp_error(error_code) _osqp_error(error_code, __FUNCTION__);
#endif



OSQPInt _osqp_error(enum osqp_error_type  error_code,
                    const char*           function_name);

OSQPInt _osqp_error_line(enum osqp_error_type error_code,
                         const char*          function_name,
                         const char*          filename,
                         OSQPInt              line_number);

#ifdef __cplusplus
}
#endif

#endif
