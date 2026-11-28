
/* AMD Version 2.4, Copyright (c) 1996-2013 by Timothy A. Davis,             */


#ifndef AMD_H
#define AMD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "SuiteSparse_config.h"

#include <stddef.h>


int amd_order

(
    int n,
    const int Ap [ ],
    const int Ai [ ],
    int P [ ],
    OSQPFloat Control [ ],
    OSQPFloat Info [ ]
) ;

SuiteSparse_long amd_l_order
(
    SuiteSparse_long n,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    SuiteSparse_long P [ ],
    OSQPFloat Control [ ],
    OSQPFloat Info [ ]
) ;




void amd_2
(
    int n,
    int Pe [ ],
    int Iw [ ],
    int Len [ ],
    int iwlen,
    int pfree,
    int Nv [ ],
    int Next [ ],
    int Last [ ],
    int Head [ ],
    int Elen [ ],
    int Degree [ ],
    int W [ ],
    OSQPFloat Control [ ],
    OSQPFloat Info [ ]
) ;

void amd_l2
(
    SuiteSparse_long n,
    SuiteSparse_long Pe [ ],
    SuiteSparse_long Iw [ ],
    SuiteSparse_long Len [ ],
    SuiteSparse_long iwlen,
    SuiteSparse_long pfree,
    SuiteSparse_long Nv [ ],
    SuiteSparse_long Next [ ],
    SuiteSparse_long Last [ ],
    SuiteSparse_long Head [ ],
    SuiteSparse_long Elen [ ],
    SuiteSparse_long Degree [ ],
    SuiteSparse_long W [ ],
    OSQPFloat Control [ ],
    OSQPFloat Info [ ]
) ;



int amd_valid
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ]
) ;

SuiteSparse_long amd_l_valid
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ]
) ;




void amd_defaults   (OSQPFloat Control [ ]) ;
void amd_l_defaults (OSQPFloat Control [ ]) ;

void amd_control    (OSQPFloat Control [ ]) ;
void amd_l_control  (OSQPFloat Control [ ]) ;

void amd_info       (OSQPFloat Info [ ]) ;
void amd_l_info     (OSQPFloat Info [ ]) ;

#define AMD_CONTROL 5
#define AMD_INFO 20

#define AMD_DENSE 0
#define AMD_AGGRESSIVE 1

#define AMD_DEFAULT_DENSE 10.0
#define AMD_DEFAULT_AGGRESSIVE 1

#define AMD_STATUS 0
#define AMD_N 1
#define AMD_NZ 2
#define AMD_SYMMETRY 3
#define AMD_NZDIAG 4
#define AMD_NZ_A_PLUS_AT 5
#define AMD_NDENSE 6
#define AMD_MEMORY 7
#define AMD_NCMPA 8
#define AMD_LNZ 9
#define AMD_NDIV 10
#define AMD_NMULTSUBS_LDL 11
#define AMD_NMULTSUBS_LU 12
#define AMD_DMAX 13


#define AMD_OK 0
#define AMD_OUT_OF_MEMORY -1
#define AMD_INVALID -2
#define AMD_OK_BUT_JUMBLED 1






#define AMD_DATE "May 4, 2016"
#define AMD_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define AMD_MAIN_VERSION 2
#define AMD_SUB_VERSION 4
#define AMD_SUBSUB_VERSION 6
#define AMD_VERSION AMD_VERSION_CODE(AMD_MAIN_VERSION,AMD_SUB_VERSION)

#ifdef __cplusplus
}
#endif

#endif
