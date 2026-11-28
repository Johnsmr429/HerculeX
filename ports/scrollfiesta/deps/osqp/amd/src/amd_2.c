
/* AMD, Copyright (c) Timothy A. Davis,					     */


#include "amd_internal.h"


static Int clear_flag (Int wflg, Int wbig, Int W [ ], Int n)
{
    Int x ;
    if (wflg < 2 || wflg >= wbig)
    {
	for (x = 0 ; x < n ; x++)
	{
	    if (W [x] != 0) W [x] = 1 ;
	}
	wflg = 2 ;
    }
       return (wflg) ;
}



GLOBAL void AMD_2
(
    Int n,
    Int Pe [ ],
    Int Iw [ ],

    Int Len [ ],
    Int iwlen,
    Int pfree,

       Int Nv [ ],
    Int Next [ ],
    Int Last [ ],
    Int Head [ ],
    Int Elen [ ],
    Int Degree [ ],
    Int W [ ],

       OSQPFloat Control [ ],
    OSQPFloat Info [ ]
)
{

/*
 * Authors, and Copyright (C) 2004 by:
 * Timothy A. Davis, Patrick Amestoy, Iain S. Duff, John K. Reid.
 *
 * Acknowledgements: This work (and the UMFPACK package) was supported by the
 * National Science Foundation (ASC-9111263, DMS-9223088, and CCR-0203270).
 * The UMFPACK/MA38 approximate degree update algorithm, the unsymmetric analog
 * which forms the basis of AMD, was developed while Tim Davis was supported by
 * CERFACS (Toulouse, France) in a post-doctoral position.  This C version, and
 * the etree postorder, were written while Tim Davis was on sabbatical at
 * Stanford University and Lawrence Berkeley National Laboratory.
 */

    Int deg, degme, dext, lemax, e, elenme, eln, i, ilast, inext, j,
	jlast, jnext, k, knt1, knt2, knt3, lenj, ln, me, mindeg, nel, nleft,
	nvi, nvj, nvpiv, slenme, wbig, we, wflg, wnvi, ok, ndense, ncmpa,
	dense, aggressive ;

    unsigned Int hash ;


    OSQPFloat f, r, ndiv, s, nms_lu, nms_ldl, dmax, alpha, lnz, lnzme ;


    Int p, p1, p2, p3, p4, pdst, pend, pj, pme, pme1, pme2, pn, psrc ;



       ASSERT (iwlen >= pfree + n) ;
    ASSERT (n > 0) ;

       lnz = 0 ;
    ndiv = 0 ;
    nms_lu = 0 ;
    nms_ldl = 0 ;
    dmax = 1 ;
    me = EMPTY ;

    mindeg = 0 ;
    ncmpa = 0 ;
    nel = 0 ;
    lemax = 0 ;

       if (Control != (OSQPFloat *) NULL)
    {
	alpha = Control [AMD_DENSE] ;
	aggressive = (Control [AMD_AGGRESSIVE] != 0) ;
    }
    else
    {
	alpha = AMD_DEFAULT_DENSE ;
	aggressive = AMD_DEFAULT_AGGRESSIVE ;
    }
       if (alpha < 0)
    {
	dense = n-2 ;
    }
    else
    {
	dense = (Int) (alpha * sqrt ((OSQPFloat) n)) ;
    }
    dense = MAX (16, dense) ;
    dense = MIN (n,  dense) ;
    AMD_DEBUG1 (("\n\nAMD (debug), alpha %g, aggr. "ID"\n",
	alpha, aggressive)) ;

    for (i = 0 ; i < n ; i++)
    {
	Last [i] = EMPTY ;
	Head [i] = EMPTY ;
	Next [i] = EMPTY ;
	Nv [i] = 1 ;
	W [i] = 1 ;
	Elen [i] = 0 ;
	Degree [i] = Len [i] ;
    }

#ifndef NDEBUG
    AMD_DEBUG1 (("\n======Nel "ID" initial\n", nel)) ;
    AMD_dump (n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last,
		Head, Elen, Degree, W, -1) ;
#endif

       wbig = Int_MAX - n ;
    wflg = clear_flag (0, wbig, W, n) ;


    ndense = 0 ;

    for (i = 0 ; i < n ; i++)
    {
	deg = Degree [i] ;
	ASSERT (deg >= 0 && deg < n) ;
	if (deg == 0)
	{


	    Elen [i] = FLIP (1) ;
	    nel++ ;
	    Pe [i] = EMPTY ;
	    W [i] = 0 ;

	}
	else if (deg > dense)
	{


	    AMD_DEBUG1 (("Dense node "ID" degree "ID"\n", i, deg)) ;
	    ndense++ ;
	    Nv [i] = 0 ;
	    Elen [i] = EMPTY ;
	    nel++ ;
	    Pe [i] = EMPTY ;

	}
	else
	{


	    inext = Head [deg] ;
	    ASSERT (inext >= EMPTY && inext < n) ;
	    if (inext != EMPTY) Last [inext] = i ;
	    Next [i] = inext ;
	    Head [deg] = i ;

	}
    }


    while (nel < n)
    {

#ifndef NDEBUG
	AMD_DEBUG1 (("\n======Nel "ID"\n", nel)) ;
	if (AMD_debug >= 2)
	{
	    AMD_dump (n, Pe, Iw, Len, iwlen, pfree, Nv, Next,
		    Last, Head, Elen, Degree, W, nel) ;
	}
#endif



	ASSERT (mindeg >= 0 && mindeg < n) ;
	for (deg = mindeg ; deg < n ; deg++)
	{
	    me = Head [deg] ;
	    if (me != EMPTY) break ;
	}
	mindeg = deg ;
	ASSERT (me >= 0 && me < n) ;
	AMD_DEBUG1 (("=================me: "ID"\n", me)) ;


	inext = Next [me] ;
	ASSERT (inext >= EMPTY && inext < n) ;
	if (inext != EMPTY) Last [inext] = EMPTY ;
	Head [deg] = inext ;


	elenme = Elen [me] ;
	nvpiv = Nv [me] ;
	ASSERT (nvpiv > 0) ;
	nel += nvpiv ;



	Nv [me] = -nvpiv ;
	degme = 0 ;
	ASSERT (Pe [me] >= 0 && Pe [me] < iwlen) ;

	if (elenme == 0)
	{


	    pme1 = Pe [me] ;
	    pme2 = pme1 - 1 ;

	    for (p = pme1 ; p <= pme1 + Len [me] - 1 ; p++)
	    {
		i = Iw [p] ;
		ASSERT (i >= 0 && i < n && Nv [i] >= 0) ;
		nvi = Nv [i] ;
		if (nvi > 0)
		{


				    degme += nvi ;
		    Nv [i] = -nvi ;
		    Iw [++pme2] = i ;


		    ilast = Last [i] ;
		    inext = Next [i] ;
		    ASSERT (ilast >= EMPTY && ilast < n) ;
		    ASSERT (inext >= EMPTY && inext < n) ;
		    if (inext != EMPTY) Last [inext] = ilast ;
		    if (ilast != EMPTY)
		    {
			Next [ilast] = inext ;
		    }
		    else
		    {
					ASSERT (Degree [i] >= 0 && Degree [i] < n) ;
			Head [Degree [i]] = inext ;
		    }
		}
	    }
	}
	else
	{


	    p = Pe [me] ;
	    pme1 = pfree ;
	    slenme = Len [me] - elenme ;

	    for (knt1 = 1 ; knt1 <= elenme + 1 ; knt1++)
	    {

		if (knt1 > elenme)
		{
				    e = me ;
		    pj = p ;
		    ln = slenme ;
		    AMD_DEBUG2 (("Search sv: "ID" "ID" "ID"\n", me,pj,ln)) ;
		}
		else
		{
				    e = Iw [p++] ;
		    ASSERT (e >= 0 && e < n) ;
		    pj = Pe [e] ;
		    ln = Len [e] ;
		    AMD_DEBUG2 (("Search element e "ID" in me "ID"\n", e,me)) ;
		    ASSERT (Elen [e] < EMPTY && W [e] > 0 && pj >= 0) ;
		}
		ASSERT (ln >= 0 && (ln == 0 || (pj >= 0 && pj < iwlen))) ;


		for (knt2 = 1 ; knt2 <= ln ; knt2++)
		{
		    i = Iw [pj++] ;
		    ASSERT (i >= 0 && i < n && (i == me || Elen [i] >= EMPTY));
		    nvi = Nv [i] ;
		    AMD_DEBUG2 ((": "ID" "ID" "ID" "ID"\n",
				i, Elen [i], Nv [i], wflg)) ;

		    if (nvi > 0)
		    {


			if (pfree >= iwlen)
			{

			    AMD_DEBUG1 (("GARBAGE COLLECTION\n")) ;


			    Pe [me] = p ;
			    Len [me] -= knt1 ;
						    if (Len [me] == 0) Pe [me] = EMPTY ;
			    Pe [e] = pj ;
			    Len [e] = ln - knt2 ;
						    if (Len [e] == 0) Pe [e] = EMPTY ;

			    ncmpa++ ;

									    for (j = 0 ; j < n ; j++)
			    {
				pn = Pe [j] ;
				if (pn >= 0)
				{
				    ASSERT (pn >= 0 && pn < iwlen) ;
				    Pe [j] = Iw [pn] ;
				    Iw [pn] = FLIP (j) ;
				}
			    }

						    psrc = 0 ;
			    pdst = 0 ;
			    pend = pme1 - 1 ;

			    while (psrc <= pend)
			    {
							j = FLIP (Iw [psrc++]) ;
				if (j >= 0)
				{
				    AMD_DEBUG2 (("Got object j: "ID"\n", j)) ;
				    Iw [pdst] = Pe [j] ;
				    Pe [j] = pdst++ ;
				    lenj = Len [j] ;
								    for (knt3 = 0 ; knt3 <= lenj - 2 ; knt3++)
				    {
					Iw [pdst++] = Iw [psrc++] ;
				    }
				}
			    }

						    p1 = pdst ;
			    for (psrc = pme1 ; psrc <= pfree-1 ; psrc++)
			    {
				Iw [pdst++] = Iw [psrc] ;
			    }
			    pme1 = p1 ;
			    pfree = pdst ;
			    pj = Pe [e] ;
			    p = Pe [me] ;

			}


					degme += nvi ;
			Nv [i] = -nvi ;
			Iw [pfree++] = i ;
			AMD_DEBUG2 (("     s: "ID"     nv "ID"\n", i, Nv [i]));


			ilast = Last [i] ;
			inext = Next [i] ;
			ASSERT (ilast >= EMPTY && ilast < n) ;
			ASSERT (inext >= EMPTY && inext < n) ;
			if (inext != EMPTY) Last [inext] = ilast ;
			if (ilast != EMPTY)
			{
			    Next [ilast] = inext ;
			}
			else
			{
						    ASSERT (Degree [i] >= 0 && Degree [i] < n) ;
			    Head [Degree [i]] = inext ;
			}
		    }
		}

		if (e != me)
		{
				    AMD_DEBUG1 ((" Element "ID" => "ID"\n", e, me)) ;
		    Pe [e] = FLIP (me) ;
		    W [e] = 0 ;
		}
	    }

	    pme2 = pfree - 1 ;
	}


	Degree [me] = degme ;
	Pe [me] = pme1 ;
	Len [me] = pme2 - pme1 + 1 ;
	ASSERT (Pe [me] >= 0 && Pe [me] < iwlen) ;

	Elen [me] = FLIP (nvpiv + degme) ;

#ifndef NDEBUG
	AMD_DEBUG2 (("New element structure: length= "ID"\n", pme2-pme1+1)) ;
	for (pme = pme1 ; pme <= pme2 ; pme++) AMD_DEBUG3 ((" "ID"", Iw[pme]));
	AMD_DEBUG3 (("\n")) ;
#endif



	wflg = clear_flag (wflg, wbig, W, n) ;



	AMD_DEBUG2 (("me: ")) ;
	for (pme = pme1 ; pme <= pme2 ; pme++)
	{
	    i = Iw [pme] ;
	    ASSERT (i >= 0 && i < n) ;
	    eln = Elen [i] ;
	    AMD_DEBUG3 ((""ID" Elen "ID": \n", i, eln)) ;
	    if (eln > 0)
	    {
			nvi = -Nv [i] ;
		ASSERT (nvi > 0 && Pe [i] >= 0 && Pe [i] < iwlen) ;
		wnvi = wflg - nvi ;
		for (p = Pe [i] ; p <= Pe [i] + eln - 1 ; p++)
		{
		    e = Iw [p] ;
		    ASSERT (e >= 0 && e < n) ;
		    we = W [e] ;
		    AMD_DEBUG4 (("    e "ID" we "ID" ", e, we)) ;
		    if (we >= wflg)
		    {
					AMD_DEBUG4 (("    unabsorbed, first time seen")) ;
			we -= nvi ;
		    }
		    else if (we != 0)
		    {
							AMD_DEBUG4 (("    unabsorbed")) ;
			we = Degree [e] + wnvi ;
		    }
		    AMD_DEBUG4 (("\n")) ;
		    W [e] = we ;
		}
	    }
	}
	AMD_DEBUG2 (("\n")) ;



	for (pme = pme1 ; pme <= pme2 ; pme++)
	{
	    i = Iw [pme] ;
	    ASSERT (i >= 0 && i < n && Nv [i] < 0 && Elen [i] >= 0) ;
	    AMD_DEBUG2 (("Updating: i "ID" "ID" "ID"\n", i, Elen[i], Len [i]));
	    p1 = Pe [i] ;
	    p2 = p1 + Elen [i] - 1 ;
	    pn = p1 ;
	    hash = 0 ;
	    deg = 0 ;
	    ASSERT (p1 >= 0 && p1 < iwlen && p2 >= -1 && p2 < iwlen) ;


		    if (aggressive)
	    {
		for (p = p1 ; p <= p2 ; p++)
		{
		    e = Iw [p] ;
		    ASSERT (e >= 0 && e < n) ;
		    we = W [e] ;
		    if (we != 0)
		    {
							dext = we - wflg ;
			if (dext > 0)
			{
			    deg += dext ;
			    Iw [pn++] = e ;
			    hash += e ;
			    AMD_DEBUG4 ((" e: "ID" hash = "ID"\n",e,hash)) ;
			}
			else
			{
						    AMD_DEBUG1 ((" Element "ID" =>"ID" (aggressive)\n",
				e, me)) ;
			    ASSERT (dext == 0) ;
			    Pe [e] = FLIP (me) ;
			    W [e] = 0 ;
			}
		    }
		}
	    }
	    else
	    {
		for (p = p1 ; p <= p2 ; p++)
		{
		    e = Iw [p] ;
		    ASSERT (e >= 0 && e < n) ;
		    we = W [e] ;
		    if (we != 0)
		    {
					dext = we - wflg ;
			ASSERT (dext >= 0) ;
			deg += dext ;
			Iw [pn++] = e ;
			hash += e ;
			AMD_DEBUG4 (("	e: "ID" hash = "ID"\n",e,hash)) ;
		    }
		}
	    }

		    Elen [i] = pn - p1 + 1 ;


		    p3 = pn ;
	    p4 = p1 + Len [i] ;
	    for (p = p2 + 1 ; p < p4 ; p++)
	    {
		j = Iw [p] ;
		ASSERT (j >= 0 && j < n) ;
		nvj = Nv [j] ;
		if (nvj > 0)
		{
						    deg += nvj ;
		    Iw [pn++] = j ;
		    hash += j ;
		    AMD_DEBUG4 (("  s: "ID" hash "ID" Nv[j]= "ID"\n",
				j, hash, nvj)) ;
		}
	    }


		    ASSERT (IMPLIES (aggressive, (deg==0) == (Elen[i]==1 && p3==pn))) ;

	    if (Elen [i] == 1 && p3 == pn)
	    {



		AMD_DEBUG1 (("  MASS i "ID" => parent e "ID"\n", i, me)) ;
		Pe [i] = FLIP (me) ;
		nvi = -Nv [i] ;
		degme -= nvi ;
		nvpiv += nvi ;
		nel += nvi ;
		Nv [i] = 0 ;
		Elen [i] = EMPTY ;

	    }
	    else
	    {



		Degree [i] = MIN (Degree [i], deg) ;


			Iw [pn] = Iw [p3] ;
			Iw [p3] = Iw [p1] ;
			Iw [p1] = me ;
			Len [i] = pn - p1 + 1 ;


			hash = hash % n ;
		ASSERT (((Int) hash) >= 0 && ((Int) hash) < n) ;

			j = Head [hash] ;
		if (j <= EMPTY)
		{
				    Next [i] = FLIP (j) ;
		    Head [hash] = FLIP (i) ;
		}
		else
		{
				    Next [i] = Last [j] ;
		    Last [j] = i ;
		}


		Last [i] = hash ;
	    }
	}

	Degree [me] = degme ;


	lemax =  MAX (lemax, degme) ;
	wflg += lemax ;
	wflg = clear_flag (wflg, wbig, W, n) ;


	AMD_DEBUG1 (("Detecting supervariables:\n")) ;
	for (pme = pme1 ; pme <= pme2 ; pme++)
	{
	    i = Iw [pme] ;
	    ASSERT (i >= 0 && i < n) ;
	    AMD_DEBUG2 (("Consider i "ID" nv "ID"\n", i, Nv [i])) ;
	    if (Nv [i] < 0)
	    {


			ASSERT (Last [i] >= 0 && Last [i] < n) ;
		hash = Last [i] ;

			j = Head [hash] ;
		if (j == EMPTY)
		{
				    i = EMPTY ;
		}
		else if (j < EMPTY)
		{
				    i = FLIP (j) ;
		    Head [hash] = EMPTY ;
		}
		else
		{
				    i = Last [j] ;
		    Last [j] = EMPTY ;
		}


		ASSERT (i >= EMPTY && i < n) ;
		AMD_DEBUG2 (("----i "ID" hash "ID"\n", i, hash)) ;

		while (i != EMPTY && Next [i] != EMPTY)
		{


		    ln = Len [i] ;
		    eln = Elen [i] ;
		    ASSERT (ln >= 0 && eln >= 0) ;
		    ASSERT (Pe [i] >= 0 && Pe [i] < iwlen) ;
				    for (p = Pe [i] + 1 ; p <= Pe [i] + ln - 1 ; p++)
		    {
			ASSERT (Iw [p] >= 0 && Iw [p] < n) ;
			W [Iw [p]] = wflg ;
		    }


		    jlast = i ;
		    j = Next [i] ;
		    ASSERT (j >= EMPTY && j < n) ;

		    while (j != EMPTY)
		    {

			AMD_DEBUG3 (("compare i "ID" and j "ID"\n", i,j)) ;

					ASSERT (Len [j] >= 0 && Elen [j] >= 0) ;
			ASSERT (Pe [j] >= 0 && Pe [j] < iwlen) ;
			ok = (Len [j] == ln) && (Elen [j] == eln) ;
					for (p = Pe [j] + 1 ; ok && p <= Pe [j] + ln - 1 ; p++)
			{
			    ASSERT (Iw [p] >= 0 && Iw [p] < n) ;
			    if (W [Iw [p]] != wflg) ok = 0 ;
			}
			if (ok)
			{

			    AMD_DEBUG1 (("found it! j "ID" => i "ID"\n", j,i));
			    Pe [j] = FLIP (i) ;
												    Nv [i] += Nv [j] ;
			    Nv [j] = 0 ;
			    Elen [j] = EMPTY ;
						    ASSERT (j != Next [j]) ;
			    j = Next [j] ;
			    Next [jlast] = j ;

			}
			else
			{
						    jlast = j ;
			    ASSERT (j != Next [j]) ;
			    j = Next [j] ;
			}
			ASSERT (j >= EMPTY && j < n) ;
		    }


		    wflg++ ;
		    i = Next [i] ;
		    ASSERT (i >= EMPTY && i < n) ;

		}
	    }
	}
	AMD_DEBUG2 (("detect done\n")) ;


	p = pme1 ;
	nleft = n - nel ;
	for (pme = pme1 ; pme <= pme2 ; pme++)
	{
	    i = Iw [pme] ;
	    ASSERT (i >= 0 && i < n) ;
	    nvi = -Nv [i] ;
	    AMD_DEBUG3 (("Restore i "ID" "ID"\n", i, nvi)) ;
	    if (nvi > 0)
	    {
				Nv [i] = nvi ;


		deg = Degree [i] + degme - nvi ;
		deg = MIN (deg, nleft - nvi) ;
		ASSERT (IMPLIES (aggressive, deg > 0) && deg >= 0 && deg < n) ;


		inext = Head [deg] ;
		ASSERT (inext >= EMPTY && inext < n) ;
		if (inext != EMPTY) Last [inext] = i ;
		Next [i] = inext ;
		Last [i] = EMPTY ;
		Head [deg] = i ;


		mindeg = MIN (mindeg, deg) ;
		Degree [i] = deg ;


		Iw [p++] = i ;

	    }
	}
	AMD_DEBUG2 (("restore done\n")) ;


	AMD_DEBUG2 (("ME = "ID" DONE\n", me)) ;
	Nv [me] = nvpiv ;
	Len [me] = p - pme1 ;
	if (Len [me] == 0)
	{
			    Pe [me] = EMPTY ;
	    W [me] = 0 ;
	}
	if (elenme != 0)
	{
			    pfree = p ;
	}


	if (Info != (OSQPFloat *) NULL)
	{
	    f = nvpiv ;
	    r = degme + ndense ;
	    dmax = MAX (dmax, f + r) ;

		    lnzme = f*r + (f-1)*f/2 ;
	    lnz += lnzme ;

		    ndiv += lnzme ;

		    s = f*r*r + r*(f-1)*f + (f-1)*f*(2*f-1)/6 ;
	    nms_lu += s ;

		    nms_ldl += (s + lnzme)/2 ;
	}

#ifndef NDEBUG
	AMD_DEBUG2 (("finalize done nel "ID" n "ID"\n   ::::\n", nel, n)) ;
	for (pme = Pe [me] ; pme <= Pe [me] + Len [me] - 1 ; pme++)
	{
	      AMD_DEBUG3 ((" "ID"", Iw [pme])) ;
	}
	AMD_DEBUG3 (("\n")) ;
#endif

    }


    if (Info != (OSQPFloat *) NULL)
    {

	f = ndense ;
	dmax = MAX (dmax, (OSQPFloat) ndense) ;

	lnzme = (f-1)*f/2 ;
	lnz += lnzme ;

	ndiv += lnzme ;

	s = (f-1)*f*(2*f-1)/6 ;
	nms_lu += s ;

	nms_ldl += (s + lnzme)/2 ;

	Info [AMD_LNZ] = lnz ;

	Info [AMD_NDIV] = ndiv ;

	Info [AMD_NMULTSUBS_LDL] = nms_ldl ;

	Info [AMD_NMULTSUBS_LU] = nms_lu ;

	Info [AMD_NDENSE] = ndense ;

	Info [AMD_DMAX] = dmax ;

	Info [AMD_NCMPA] = ncmpa ;

	Info [AMD_STATUS] = AMD_OK ;
    }



       for (i = 0 ; i < n ; i++)
    {
	Pe [i] = FLIP (Pe [i]) ;
    }

       for (i = 0 ; i < n ; i++)
    {
	Elen [i] = FLIP (Elen [i]) ;
    }


#ifndef NDEBUG
    AMD_DEBUG2 (("\nTree:\n")) ;
    for (i = 0 ; i < n ; i++)
    {
	AMD_DEBUG2 ((" "ID" parent: "ID"   ", i, Pe [i])) ;
	ASSERT (Pe [i] >= EMPTY && Pe [i] < n) ;
	if (Nv [i] > 0)
	{
		    e = i ;
	    AMD_DEBUG2 ((" element, size is "ID"\n", Elen [i])) ;
	    ASSERT (Elen [e] > 0) ;
	}
	AMD_DEBUG2 (("\n")) ;
    }
    AMD_DEBUG2 (("\nelements:\n")) ;
    for (e = 0 ; e < n ; e++)
    {
	if (Nv [e] > 0)
	{
	    AMD_DEBUG3 (("Element e= "ID" size "ID" nv "ID" \n", e,
		Elen [e], Nv [e])) ;
	}
    }
    AMD_DEBUG2 (("\nvariables:\n")) ;
    for (i = 0 ; i < n ; i++)
    {
	Int cnt ;
	if (Nv [i] == 0)
	{
	    AMD_DEBUG3 (("i unordered: "ID"\n", i)) ;
	    j = Pe [i] ;
	    cnt = 0 ;
	    AMD_DEBUG3 (("  j: "ID"\n", j)) ;
	    if (j == EMPTY)
	    {
		AMD_DEBUG3 (("	i is a dense variable\n")) ;
	    }
	    else
	    {
		ASSERT (j >= 0 && j < n) ;
		while (Nv [j] == 0)
		{
		    AMD_DEBUG3 (("	j : "ID"\n", j)) ;
		    j = Pe [j] ;
		    AMD_DEBUG3 (("	j:: "ID"\n", j)) ;
		    cnt++ ;
		    if (cnt > n) break ;
		}
		e = j ;
		AMD_DEBUG3 (("	got to e: "ID"\n", e)) ;
	    }
	}
    }
#endif


    for (i = 0 ; i < n ; i++)
    {
	if (Nv [i] == 0)
	{


	    AMD_DEBUG1 (("Path compression, i unordered: "ID"\n", i)) ;
	    j = Pe [i] ;
	    ASSERT (j >= EMPTY && j < n) ;
	    AMD_DEBUG3 (("	j: "ID"\n", j)) ;
	    if (j == EMPTY)
	    {
			AMD_DEBUG3 (("      i is a dense variable\n")) ;
		continue ;
	    }

		    while (Nv [j] == 0)
	    {
		AMD_DEBUG3 (("		j : "ID"\n", j)) ;
		j = Pe [j] ;
		AMD_DEBUG3 (("		j:: "ID"\n", j)) ;
		ASSERT (j >= 0 && j < n) ;
	    }
		    e = j ;
	    AMD_DEBUG3 (("got to e: "ID"\n", e)) ;


	    j = i ;
		    while (Nv [j] == 0)
	    {
		jnext = Pe [j] ;
		AMD_DEBUG3 (("j "ID" jnext "ID"\n", j, jnext)) ;
		Pe [j] = e ;
		j = jnext ;
		ASSERT (j >= 0 && j < n) ;
	    }
	}
    }


    AMD_postorder (n, Pe, Nv, Elen,
	W,
	Head, Next, Last) ;



    for (k = 0 ; k < n ; k++)
    {
	Head [k] = EMPTY ;
	Next [k] = EMPTY ;
    }
    for (e = 0 ; e < n ; e++)
    {
	k = W [e] ;
	ASSERT ((k == EMPTY) == (Nv [e] == 0)) ;
	if (k != EMPTY)
	{
	    ASSERT (k >= 0 && k < n) ;
	    Head [k] = e ;
	}
    }

       nel = 0 ;
    for (k = 0 ; k < n ; k++)
    {
	e = Head [k] ;
	if (e == EMPTY) break ;
	ASSERT (e >= 0 && e < n && Nv [e] > 0) ;
	Next [e] = nel ;
	nel += Nv [e] ;
    }
    ASSERT (nel == n - ndense) ;

       for (i = 0 ; i < n ; i++)
    {
	if (Nv [i] == 0)
	{
	    e = Pe [i] ;
	    ASSERT (e >= EMPTY && e < n) ;
	    if (e != EMPTY)
	    {
			ASSERT (Next [i] == EMPTY && Nv [e] > 0) ;
		Next [i] = Next [e] ;
		Next [e]++ ;
	    }
	    else
	    {
			Next [i] = nel++ ;
	    }
	}
    }
    ASSERT (nel == n) ;

    AMD_DEBUG2 (("\n\nPerm:\n")) ;
    for (i = 0 ; i < n ; i++)
    {
	k = Next [i] ;
	ASSERT (k >= 0 && k < n) ;
	Last [k] = i ;
	AMD_DEBUG2 (("   perm ["ID"] = "ID"\n", k, i)) ;
    }
}
