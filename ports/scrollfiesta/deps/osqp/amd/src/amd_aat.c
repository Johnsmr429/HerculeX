
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"

GLOBAL size_t AMD_aat
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int Len [ ],
    Int Tp [ ],
    OSQPFloat Info [ ]
)
{
    Int p1, p2, p, i, j, pj, pj2, k, nzdiag, nzboth, nz ;
    OSQPFloat sym ;
    size_t nzaat ;

#ifndef NDEBUG
    AMD_debug_init ("AMD AAT") ;
    for (k = 0 ; k < n ; k++) Tp [k] = EMPTY ;
    ASSERT (AMD_valid (n, n, Ap, Ai) == AMD_OK) ;
#endif

    if (Info != (OSQPFloat *) NULL)
    {
	for (i = 0 ; i < AMD_INFO ; i++)
	{
	    Info [i] = EMPTY ;
	}
	Info [AMD_STATUS] = AMD_OK ;
    }

    for (k = 0 ; k < n ; k++)
    {
	Len [k] = 0 ;
    }

    nzdiag = 0 ;
    nzboth = 0 ;
    nz = Ap [n] ;

    for (k = 0 ; k < n ; k++)
    {
	p1 = Ap [k] ;
	p2 = Ap [k+1] ;
	AMD_DEBUG2 (("\nAAT Column: "ID" p1: "ID" p2: "ID"\n", k, p1, p2)) ;

	for (p = p1 ; p < p2 ; )
	{
		    j = Ai [p] ;
	    if (j < k)
	    {
			Len [j]++ ;
		Len [k]++ ;
		AMD_DEBUG3 (("    upper ("ID","ID") ("ID","ID")\n", j,k, k,j));
		p++ ;
	    }
	    else if (j == k)
	    {
			p++ ;
		nzdiag++ ;
		break ;
	    }
	    else
	    {
			break ;
	    }
		    ASSERT (Tp [j] != EMPTY) ;
	    ASSERT (Ap [j] <= Tp [j] && Tp [j] <= Ap [j+1]) ;
	    pj2 = Ap [j+1] ;
	    for (pj = Tp [j] ; pj < pj2 ; )
	    {
		i = Ai [pj] ;
		if (i < k)
		{
				    Len [i]++ ;
		    Len [j]++ ;
		    AMD_DEBUG3 (("    lower ("ID","ID") ("ID","ID")\n",
			i,j, j,i)) ;
		    pj++ ;
		}
		else if (i == k)
		{
				    pj++ ;
		    nzboth++ ;
		    break ;
		}
		else
		{
				    break ;
		}
	    }
	    Tp [j] = pj ;
	}
	Tp [k] = p ;
    }

       for (j = 0 ; j < n ; j++)
    {
	for (pj = Tp [j] ; pj < Ap [j+1] ; pj++)
	{
	    i = Ai [pj] ;
		    Len [i]++ ;
	    Len [j]++ ;
	    AMD_DEBUG3 (("    lower cleanup ("ID","ID") ("ID","ID")\n",
		i,j, j,i)) ;
	}
    }



    if (nz == nzdiag)
    {
	sym = 1 ;
    }
    else
    {
	sym = (2 * (OSQPFloat) nzboth) / ((OSQPFloat) (nz - nzdiag)) ;
    }

    nzaat = 0 ;
    for (k = 0 ; k < n ; k++)
    {
	nzaat += Len [k] ;
    }

    AMD_DEBUG1 (("AMD nz in A+A', excluding diagonal (nzaat) = %g\n",
	(OSQPFloat) nzaat)) ;
    AMD_DEBUG1 (("   nzboth: "ID" nz: "ID" nzdiag: "ID" symmetry: %g\n",
		nzboth, nz, nzdiag, sym)) ;

    if (Info != (OSQPFloat *) NULL)
    {
	Info [AMD_STATUS] = AMD_OK ;
	Info [AMD_N] = n ;
	Info [AMD_NZ] = nz ;
	Info [AMD_SYMMETRY] = sym ;
	Info [AMD_NZDIAG] = nzdiag ;
	Info [AMD_NZ_A_PLUS_AT] = nzaat ;
    }

    return (nzaat) ;
}
