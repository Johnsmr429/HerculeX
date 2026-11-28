

/* Copyright (c) 2013, Timothy A. Davis.  No licensing restrictions
 * apply to this file or to the SuiteSparse_config directory.
 * Author: Timothy A. Davis.
 */

#include "glob_opts.h"

#include <math.h>
#include <stdlib.h>



#ifndef NPRINT
#include <stdio.h>
#endif

#ifdef MATLAB
#include "mex.h"
#include "matrix.h"
#endif

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "SuiteSparse_config.h"



#if defined(SPARSE_TAUCS_THREAD_SAFE)
extern void *taucs_malloc_stub(size_t size);
extern void *taucs_realloc_stub(void *ptr, size_t size);
extern void  taucs_free_stub(void *ptr);
#define VES_SS_MALLOC  taucs_malloc_stub
#define VES_SS_REALLOC taucs_realloc_stub
#define VES_SS_FREE    taucs_free_stub
#else
#define VES_SS_MALLOC  c_malloc
#define VES_SS_REALLOC c_realloc
#define VES_SS_FREE    c_free
#endif
struct SuiteSparse_config_struct SuiteSparse_config =
{
       VES_SS_MALLOC, VES_SS_REALLOC, VES_SS_FREE,
       NULL,
    SuiteSparse_hypot,
    SuiteSparse_divcomplex

} ;



void *SuiteSparse_malloc
(
    size_t nitems,
    size_t size_of_item
)
{
    void *p ;
    size_t size ;
    if (nitems < 1) nitems = 1 ;
    if (size_of_item < 1) size_of_item = 1 ;
    size = nitems * size_of_item  ;

    if (size != ((OSQPFloat) nitems) * size_of_item)
    {
               p = NULL ;
    }
    else
    {
        p = (void *) (SuiteSparse_config.malloc_func) (size) ;
           }
    return (p) ;
}





void *SuiteSparse_realloc

(
    size_t nitems_new,
    size_t nitems_old,
    size_t size_of_item,
    void *p,
    int *ok
)
{
    size_t size ;
    if (nitems_old < 1) nitems_old = 1 ;
    if (nitems_new < 1) nitems_new = 1 ;
    if (size_of_item < 1) size_of_item = 1 ;
    size = nitems_new * size_of_item  ;

    if (size != ((OSQPFloat) nitems_new) * size_of_item)
    {
               (*ok) = 0 ;
    }
    else if (p == NULL)
    {
               p = SuiteSparse_malloc (nitems_new, size_of_item) ;
        (*ok) = (p != NULL) ;
    }
    else if (nitems_old == nitems_new)
    {
               (*ok) = 1 ;
    }
    else
    {
               void *pnew ;
               pnew = (void *) c_realloc (p, size) ;
        if (pnew == NULL)
        {
            if (nitems_new < nitems_old)
            {
                               (*ok) = 1 ;
            }
            else
            {
                               (*ok) = 0 ;
            }
        }
        else
        {
                       p = pnew ;
            (*ok) = 1 ;
        }
    }
    return (p) ;
}


void *SuiteSparse_free
(
    void *p
)
{
    if (p)
    {
        (SuiteSparse_config.free_func) (p) ;
           }
    return (NULL) ;
}




#ifdef SUITESPARSE_TIMER_ENABLED

#include <time.h>

void SuiteSparse_tic
(
    OSQPFloat tic [2]
)
{
       struct timespec t ;
    clock_gettime (CLOCK_MONOTONIC, &t) ;
    tic [0] = (OSQPFloat) (t.tv_sec) ;
    tic [1] = (OSQPFloat) (t.tv_nsec) ;
}

#else

void SuiteSparse_tic
(
    OSQPFloat tic [2]
)
{
       tic [0] = 0 ;
    tic [1] = 0 ;
}

#endif




OSQPFloat SuiteSparse_toc
(
    OSQPFloat tic [2]
)
{
    OSQPFloat toc [2] ;
    SuiteSparse_tic (toc) ;
    return ((toc [0] - tic [0]) + 1e-9 * (toc [1] - tic [1])) ;
}




OSQPFloat SuiteSparse_time
(
    void
)
{
    OSQPFloat toc [2] ;
    SuiteSparse_tic (toc) ;
    return (toc [0] + 1e-9 * toc [1]) ;
}



int SuiteSparse_version
(
    int version [3]
)
{
    if (version != NULL)
    {
        version [0] = SUITESPARSE_MAIN_VERSION ;
        version [1] = SUITESPARSE_SUB_VERSION ;
        version [2] = SUITESPARSE_SUBSUB_VERSION ;
    }
    return (SUITESPARSE_VERSION) ;
}



OSQPFloat SuiteSparse_hypot (OSQPFloat x, OSQPFloat y)
{
    OSQPFloat s, r ;
    x = fabs (x) ;
    y = fabs (y) ;
    if (x >= y)
    {
        if (x + y == x)
        {
            s = x ;
        }
        else
        {
            r = y / x ;
            s = x * sqrt (1.0 + r*r) ;
        }
    }
    else
    {
        if (y + x == y)
        {
            s = y ;
        }
        else
        {
            r = x / y ;
            s = y * sqrt (1.0 + r*r) ;
        }
    }
    return (s) ;
}



int SuiteSparse_divcomplex
(
    OSQPFloat ar, OSQPFloat ai,
    OSQPFloat br, OSQPFloat bi,
    OSQPFloat *cr, OSQPFloat *ci
)
{
    OSQPFloat tr, ti, r, den ;
    if (fabs (br) >= fabs (bi))
    {
        r = bi / br ;
        den = br + r * bi ;
        tr = (ar + ai * r) / den ;
        ti = (ai - ar * r) / den ;
    }
    else
    {
        r = br / bi ;
        den = r * br + bi ;
        tr = (ar * r + ai) / den ;
        ti = (ai * r - ar) / den ;
    }
    *cr = tr ;
    *ci = ti ;
    return (den == 0.) ;
}
