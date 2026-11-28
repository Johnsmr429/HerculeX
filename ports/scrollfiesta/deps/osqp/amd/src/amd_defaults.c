
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"


GLOBAL void AMD_defaults
(
    OSQPFloat Control [ ]
)
{
    Int i ;

    if (Control != (OSQPFloat *) NULL)
    {
	for (i = 0 ; i < AMD_CONTROL ; i++)
	{
	    Control [i] = 0 ;
	}
	Control [AMD_DENSE] = AMD_DEFAULT_DENSE ;
	Control [AMD_AGGRESSIVE] = AMD_DEFAULT_AGGRESSIVE ;
    }
}
