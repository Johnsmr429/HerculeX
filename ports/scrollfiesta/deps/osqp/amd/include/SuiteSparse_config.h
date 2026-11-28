
/*
 * Copyright (c) 2012, Timothy A. Davis.  No licensing restrictions apply
 * to this file or to the SuiteSparse_config directory.
 * Author: Timothy A. Davis.
 */

#ifndef SUITESPARSE_CONFIG_H
#define SUITESPARSE_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "osqp.h"
#include <limits.h>
#include <stdlib.h>


#ifndef SuiteSparse_long

#define SuiteSparse_long long long
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "ld"

#define SuiteSparse_long_id "%" SuiteSparse_long_idd
#endif



struct SuiteSparse_config_struct
{
    void *(*malloc_func) (size_t) ;
    void *(*realloc_func) (void *, size_t) ;
    void (*free_func) (void *) ;
#if defined PYTHON
    void (*printf_func) (const char *, ...) ;
#elif defined R_LANG
    void (*printf_func) (const char *, ...) ;
#else
    int (*printf_func) (const char *, ...) ;
#endif
    OSQPFloat (*hypot_func) (OSQPFloat, OSQPFloat) ;
    int (*divcomplex_func) (OSQPFloat, OSQPFloat, OSQPFloat, OSQPFloat, OSQPFloat *, OSQPFloat *);
} ;

extern struct SuiteSparse_config_struct SuiteSparse_config ;

void *SuiteSparse_malloc
(
    size_t nitems,
    size_t size_of_item
) ;

void *SuiteSparse_calloc
(
    size_t nitems,
    size_t size_of_item
) ;

void *SuiteSparse_realloc

(
    size_t nitems_new,
    size_t nitems_old,
    size_t size_of_item,
    void *p,
    int *ok
) ;

void *SuiteSparse_free
(
    void *p
) ;

void SuiteSparse_tic
(
    OSQPFloat tic [2]
) ;

OSQPFloat SuiteSparse_toc
(
    OSQPFloat tic [2]
) ;

OSQPFloat SuiteSparse_time
(
    void
) ;

OSQPFloat SuiteSparse_hypot (OSQPFloat x, OSQPFloat y) ;

int SuiteSparse_divcomplex
(
    OSQPFloat ar, OSQPFloat ai,
    OSQPFloat br, OSQPFloat bi,
    OSQPFloat *cr, OSQPFloat *ci
) ;

#define NTIMER

#define NPRINT

#ifndef NTIMER
#ifdef _POSIX_C_SOURCE
#if    _POSIX_C_SOURCE >= 199309L
#define SUITESPARSE_TIMER_ENABLED
#endif
#endif
#endif

#define SUITESPARSE_PRINTF(params) \
{ \
    if (SuiteSparse_config.printf_func != NULL) \
    { \
        (void) (SuiteSparse_config.printf_func) params ; \
    } \
}



int SuiteSparse_version
(
       int version [3]
) ;

#define SUITESPARSE_HAS_VERSION_FUNCTION

#define SUITESPARSE_DATE "May 4, 2016"
#define SUITESPARSE_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define SUITESPARSE_MAIN_VERSION 4
#define SUITESPARSE_SUB_VERSION 5
#define SUITESPARSE_SUBSUB_VERSION 3
#define SUITESPARSE_VERSION \
    SUITESPARSE_VER_CODE(SUITESPARSE_MAIN_VERSION,SUITESPARSE_SUB_VERSION)

#ifdef __cplusplus
}
#endif
#endif
