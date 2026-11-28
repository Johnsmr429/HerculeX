
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"


GLOBAL Int AMD_order
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int P [ ],
    OSQPFloat Control [ ],
    OSQPFloat Info [ ]
)
{
    Int *Len, *S, nz, i, *Pinv, info, status, *Rp, *Ri, *Cp, *Ci, ok ;
    size_t nzaat, slen ;
    OSQPFloat mem = 0 ;

#ifndef NDEBUG
    AMD_debug_init ("amd") ;
#endif

       info = Info != (OSQPFloat *) NULL ;
    if (info)
    {
	for (i = 0 ; i < AMD_INFO ; i++)
	{
	    Info [i] = EMPTY ;
	}
	Info [AMD_N] = n ;
	Info [AMD_STATUS] = AMD_OK ;
    }

       if (Ai == (Int *) NULL || Ap == (Int *) NULL || P == (Int *) NULL || n < 0)
    {
	if (info) Info [AMD_STATUS] = AMD_INVALID ;
	return (AMD_INVALID) ;
    }

    if (n == 0)
    {
	return (AMD_OK) ;
    }

    nz = Ap [n] ;
    if (info)
    {
	Info [AMD_NZ] = nz ;
    }
    if (nz < 0)
    {
	if (info) Info [AMD_STATUS] = AMD_INVALID ;
	return (AMD_INVALID) ;
    }

       if (((size_t) n) >= SIZE_T_MAX / sizeof (Int)
     || ((size_t) nz) >= SIZE_T_MAX / sizeof (Int))
    {
	if (info) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	return (AMD_OUT_OF_MEMORY) ;
    }

       status = AMD_valid (n, n, Ap, Ai) ;

    if (status == AMD_INVALID)
    {
	if (info) Info [AMD_STATUS] = AMD_INVALID ;
	return (AMD_INVALID) ;
    }

       Len  = SuiteSparse_malloc (n, sizeof (Int)) ;
    Pinv = SuiteSparse_malloc (n, sizeof (Int)) ;
    mem += n ;
    mem += n ;
    if (!Len || !Pinv)
    {
	SuiteSparse_free (Len) ;
	SuiteSparse_free (Pinv) ;
	if (info) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	return (AMD_OUT_OF_MEMORY) ;
    }

    if (status == AMD_OK_BUT_JUMBLED)
    {
	AMD_DEBUG1 (("Matrix is jumbled\n")) ;
	Rp = SuiteSparse_malloc (n+1, sizeof (Int)) ;
	Ri = SuiteSparse_malloc (nz,  sizeof (Int)) ;
	mem += (n+1) ;
	mem += MAX (nz,1) ;
	if (!Rp || !Ri)
	{
		    SuiteSparse_free (Rp) ;
	    SuiteSparse_free (Ri) ;
	    SuiteSparse_free (Len) ;
	    SuiteSparse_free (Pinv) ;
	    if (info) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	    return (AMD_OUT_OF_MEMORY) ;
	}
	AMD_preprocess (n, Ap, Ai, Rp, Ri, Len, Pinv) ;
	Cp = Rp ;
	Ci = Ri ;
    }
    else
    {
	Rp = NULL ;
	Ri = NULL ;
	Cp = (Int *) Ap ;
	Ci = (Int *) Ai ;
    }


    nzaat = AMD_aat (n, Cp, Ci, Len, P, Info) ;
    AMD_DEBUG1 (("nzaat: %g\n", (OSQPFloat) nzaat)) ;
    ASSERT ((MAX (nz-n, 0) <= nzaat) && (nzaat <= 2 * (size_t) nz)) ;


    S = NULL ;
    slen = nzaat ;
    ok = ((slen + nzaat/5) >= slen) ;
    slen += nzaat/5 ;
    for (i = 0 ; ok && i < 7 ; i++)
    {
	ok = ((slen + n) > slen) ;
	slen += n ;
    }
    mem += slen ;
    ok = ok && (slen < SIZE_T_MAX / sizeof (Int)) ;
    ok = ok && (slen < Int_MAX) ;
    if (ok)
    {
	S = SuiteSparse_malloc (slen, sizeof (Int)) ;
    }
    AMD_DEBUG1 (("slen %g\n", (OSQPFloat) slen)) ;
    if (!S)
    {
	SuiteSparse_free (Rp) ;
	SuiteSparse_free (Ri) ;
	SuiteSparse_free (Len) ;
	SuiteSparse_free (Pinv) ;
	if (info) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	return (AMD_OUT_OF_MEMORY) ;
    }
    if (info)
    {
	Info [AMD_MEMORY] = mem * sizeof (Int) ;
    }


    AMD_1 (n, Cp, Ci, P, Pinv, Len, slen, S, Control, Info) ;


    SuiteSparse_free (Rp) ;
    SuiteSparse_free (Ri) ;
    SuiteSparse_free (Len) ;
    SuiteSparse_free (Pinv) ;
    SuiteSparse_free (S) ;
    if (info) Info [AMD_STATUS] = status ;
    return (status) ;
}
