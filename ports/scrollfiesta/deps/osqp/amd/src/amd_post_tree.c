
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"

GLOBAL Int AMD_post_tree
(
    Int root,
    Int k,
    Int Child [ ],



    const Int Sibling [ ],




    Int Order [ ],


    Int Stack [ ]
#ifndef NDEBUG
    , Int nn
#endif
)
{
    Int f, head, h, i ;

#if 0

       i = root ;
    for (f = Child [i] ; f != EMPTY ; f = Sibling [f])
    {
	k = AMD_post_tree (f, k, Child, Sibling, Order, Stack, nn) ;
    }
    Order [i] = k++ ;
    return (k) ;
#endif


       head = 0 ;
    Stack [0] = root ;

    while (head >= 0)
    {
	ASSERT (head < nn) ;
	i = Stack [head] ;
	AMD_DEBUG1 (("head of stack "ID" \n", i)) ;
	ASSERT (i >= 0 && i < nn) ;

	if (Child [i] != EMPTY)
	{
					    for (f = Child [i] ; f != EMPTY ; f = Sibling [f])
	    {
		head++ ;
		ASSERT (head < nn) ;
		ASSERT (f >= 0 && f < nn) ;
	    }
	    h = head ;
	    ASSERT (head < nn) ;
	    for (f = Child [i] ; f != EMPTY ; f = Sibling [f])
	    {
		ASSERT (h > 0) ;
		Stack [h--] = f ;
		AMD_DEBUG1 (("push "ID" on stack\n", f)) ;
		ASSERT (f >= 0 && f < nn) ;
	    }
	    ASSERT (Stack [h] == i) ;

		    Child [i] = EMPTY ;
	}
	else
	{
			    head-- ;
	    AMD_DEBUG1 (("pop "ID" order "ID"\n", i, k)) ;
	    Order [i] = k++ ;
	    ASSERT (k <= nn) ;
	}

#ifndef NDEBUG
	AMD_DEBUG1 (("\nStack:")) ;
	for (h = head ; h >= 0 ; h--)
	{
	    Int j = Stack [h] ;
	    AMD_DEBUG1 ((" "ID, j)) ;
	    ASSERT (j >= 0 && j < nn) ;
	}
	AMD_DEBUG1 (("\n\n")) ;
	ASSERT (head < nn) ;
#endif

    }
    return (k) ;
}
