
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"



GLOBAL void AMD_preprocess
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],

       Int Rp [ ],
    Int Ri [ ],

    Int W [ ],
    Int Flag [ ]
)
{


    Int i, j, p, p2 ;

    ASSERT (AMD_valid (n, n, Ap, Ai) != AMD_INVALID) ;


    for (i = 0 ; i < n ; i++)
    {
	W [i] = 0 ;
	Flag [i] = EMPTY ;
    }
    for (j = 0 ; j < n ; j++)
    {
	p2 = Ap [j+1] ;
	for (p = Ap [j] ; p < p2 ; p++)
	{
	    i = Ai [p] ;
	    if (Flag [i] != j)
	    {
			W [i]++ ;
		Flag [i] = j ;
	    }
	}
    }


    Rp [0] = 0 ;
    for (i = 0 ; i < n ; i++)
    {
	Rp [i+1] = Rp [i] + W [i] ;
    }
    for (i = 0 ; i < n ; i++)
    {
	W [i] = Rp [i] ;
	Flag [i] = EMPTY ;
    }


       for (j = 0 ; j < n ; j++)
    {
	p2 = Ap [j+1] ;
	for (p = Ap [j] ; p < p2 ; p++)
	{
	    i = Ai [p] ;
	    if (Flag [i] != j)
	    {
			Ri [W [i]++] = j ;
		Flag [i] = j ;
	    }
	}
    }

#ifndef NDEBUG
    ASSERT (AMD_valid (n, n, Rp, Ri) == AMD_OK) ;
    for (j = 0 ; j < n ; j++)
    {
	ASSERT (W [j] == Rp [j+1]) ;
    }
#endif
}
