
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"

GLOBAL Int AMD_valid
(
       Int n_row,
    Int n_col,
    const Int Ap [ ],
    const Int Ai [ ]
)
{
    Int nz, j, p1, p2, ilast, i, p, result = AMD_OK ;

    if (n_row < 0 || n_col < 0 || Ap == NULL || Ai == NULL)
    {
	return (AMD_INVALID) ;
    }
    nz = Ap [n_col] ;
    if (Ap [0] != 0 || nz < 0)
    {
	AMD_DEBUG0 (("column 0 pointer bad or nz < 0\n")) ;
	return (AMD_INVALID) ;
    }
    for (j = 0 ; j < n_col ; j++)
    {
	p1 = Ap [j] ;
	p2 = Ap [j+1] ;
	AMD_DEBUG2 (("\nColumn: "ID" p1: "ID" p2: "ID"\n", j, p1, p2)) ;
	if (p1 > p2)
	{
		    AMD_DEBUG0 (("column "ID" pointer bad\n", j)) ;
	    return (AMD_INVALID) ;
	}
	ilast = EMPTY ;
	for (p = p1 ; p < p2 ; p++)
	{
	    i = Ai [p] ;
	    AMD_DEBUG3 (("row: "ID"\n", i)) ;
	    if (i < 0 || i >= n_row)
	    {
			AMD_DEBUG0 (("index out of range, col "ID" row "ID"\n", j, i));
		return (AMD_INVALID) ;
	    }
	    if (i <= ilast)
	    {
			AMD_DEBUG1 (("index unsorted/dupl col "ID" row "ID"\n", j, i));
		result = AMD_OK_BUT_JUMBLED ;
	    }
	    ilast = i ;
	}
    }
    return (result) ;
}
