#if defined(_MSC_VER)
#define GENMMD_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__)
#define GENMMD_THREAD_LOCAL __thread
#else
#define GENMMD_THREAD_LOCAL
#endif

#include "f2c.h"









 int genmmd_(neqns, xadj, adjncy, invp, perm, delta, dhead,
	qsize, llist, marker, maxint, nofsub)
integer *neqns, *xadj, *adjncy, *invp, *perm, *delta, *dhead, *qsize, *llist,
	*marker, *maxint, *nofsub;
{
       integer i__1;

       static GENMMD_THREAD_LOCAL integer mdeg, ehead, i, mdlmt, mdnode;
    extern  int mmdelm_(), mmdupd_(), mmdint_(), mmdnum_();
    static GENMMD_THREAD_LOCAL integer nextmd, tag, num;





       --marker;
    --llist;
    --qsize;
    --dhead;
    --perm;
    --invp;
    --adjncy;
    --xadj;

       if (*neqns <= 0) {
	return 0;
    }

    *nofsub = 0;
    mmdint_(neqns, &xadj[1], &adjncy[1], &dhead[1], &invp[1], &perm[1], &
	    qsize[1], &llist[1], &marker[1]);

    num = 1;

    nextmd = dhead[1];
L100:
    if (nextmd <= 0) {
	goto L200;
    }
    mdnode = nextmd;
    nextmd = invp[mdnode];
    marker[mdnode] = *maxint;
    invp[mdnode] = -num;
    ++num;
    goto L100;

L200:
    if (num > *neqns) {
	goto L1000;
    }
    tag = 1;
    dhead[1] = 0;
    mdeg = 2;
L300:
    if (dhead[mdeg] > 0) {
	goto L400;
    }
    ++mdeg;
    goto L300;
L400:
       mdlmt = *delta > *neqns - mdeg ? *neqns : mdeg + *delta;
    ehead = 0;

L500:
    mdnode = dhead[mdeg];
    if (mdnode > 0) {
	goto L600;
    }
    ++mdeg;
    if (mdeg > mdlmt) {
	goto L900;
    }
    goto L500;
L600:
    nextmd = invp[mdnode];
    dhead[mdeg] = nextmd;
    if (nextmd > 0) {
	perm[nextmd] = -mdeg;
    }
    invp[mdnode] = -num;
    *nofsub = *nofsub + mdeg + qsize[mdnode] - 2;
    if (num + qsize[mdnode] > *neqns) {
	goto L1000;
    }
    ++tag;
    if (tag < *maxint) {
	goto L800;
    }
    tag = 1;
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
	if (marker[i] < *maxint) {
	    marker[i] = 0;
	}
    }
L800:
    mmdelm_(&mdnode, &xadj[1], &adjncy[1], &dhead[1], &invp[1], &perm[1], &
	    qsize[1], &llist[1], &marker[1], maxint, &tag);
    num += qsize[mdnode];
    llist[mdnode] = ehead;
    ehead = mdnode;
    if (*delta >= 0) {
	goto L500;
    }
L900:
    if (num > *neqns) {
	goto L1000;
    }
    mmdupd_(&ehead, neqns, &xadj[1], &adjncy[1], delta, &mdeg, &dhead[1], &
	    invp[1], &perm[1], &qsize[1], &llist[1], &marker[1], maxint, &tag)
	    ;
    goto L300;

L1000:
    mmdnum_(neqns, &perm[1], &invp[1], &qsize[1]);
    return 0;

}







 int mmdint_(neqns, xadj, adjncy, dhead, dforw, dbakw, qsize,
	llist, marker)
integer *neqns, *xadj, *adjncy, *dhead, *dforw, *dbakw, *qsize, *llist, *
	marker;
{
       integer i__1;

       static GENMMD_THREAD_LOCAL integer ndeg, node, fnode;





       --marker;
    --llist;
    --qsize;
    --dbakw;
    --dforw;
    --dhead;
    --adjncy;
    --xadj;

       i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
	dhead[node] = 0;
	qsize[node] = 1;
	marker[node] = 0;
	llist[node] = 0;
    }
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
	ndeg = xadj[node + 1] - xadj[node] + 1;
	fnode = dhead[ndeg];
	dforw[node] = fnode;
	dhead[ndeg] = node;
	if (fnode > 0) {
	    dbakw[fnode] = node;
	}
	dbakw[node] = -ndeg;
    }
    return 0;

}







 int mmdelm_(mdnode, xadj, adjncy, dhead, dforw, dbakw, qsize,
	 llist, marker, maxint, tag)
integer *mdnode, *xadj, *adjncy, *dhead, *dforw, *dbakw, *qsize, *llist, *
	marker, *maxint, *tag;
{
       integer i__1, i__2;

       static GENMMD_THREAD_LOCAL integer node, link, rloc, rlmt, i, j, nabor, rnode, elmnt, xqnbr,
	    istop, jstop, istrt, jstrt, nxnode, pvnode, nqnbrs, npv;





       --marker;
    --llist;
    --qsize;
    --dbakw;
    --dforw;
    --dhead;
    --adjncy;
    --xadj;

       marker[*mdnode] = *tag;
    istrt = xadj[*mdnode];
    istop = xadj[*mdnode + 1] - 1;
    elmnt = 0;
    rloc = istrt;
    rlmt = istop;
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
	nabor = adjncy[i];
	if (nabor == 0) {
	    goto L300;
	}
	if (marker[nabor] >= *tag) {
	    goto L200;
	}
	marker[nabor] = *tag;
	if (dforw[nabor] < 0) {
	    goto L100;
	}
	adjncy[rloc] = nabor;
	++rloc;
	goto L200;
L100:
	llist[nabor] = elmnt;
	elmnt = nabor;
L200:
	;
    }
L300:
    if (elmnt <= 0) {
	goto L1000;
    }
    adjncy[rlmt] = -elmnt;
    link = elmnt;
L400:
    jstrt = xadj[link];
    jstop = xadj[link + 1] - 1;
    i__1 = jstop;
    for (j = jstrt; j <= i__1; ++j) {
	node = adjncy[j];
	link = -node;
	if (node < 0) {
	    goto L400;
	} else if (node == 0) {
	    goto L900;
	} else {
	    goto L500;
	}
L500:
	if (marker[node] >= *tag || dforw[node] < 0) {
	    goto L800;
	}
	marker[node] = *tag;
L600:
	if (rloc < rlmt) {
	    goto L700;
	}
	link = -adjncy[rlmt];
	rloc = xadj[link];
	rlmt = xadj[link + 1] - 1;
	goto L600;
L700:
	adjncy[rloc] = node;
	++rloc;
L800:
	;
    }
L900:
    elmnt = llist[elmnt];
    goto L300;
L1000:
    if (rloc <= rlmt) {
	adjncy[rloc] = 0;
    }
    link = *mdnode;
L1100:
    istrt = xadj[link];
    istop = xadj[link + 1] - 1;
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
	rnode = adjncy[i];
	link = -rnode;
	if (rnode < 0) {
	    goto L1100;
	} else if (rnode == 0) {
	    goto L1800;
	} else {
	    goto L1200;
	}
L1200:
	pvnode = dbakw[rnode];
	if (pvnode == 0 || pvnode == -(*maxint)) {
	    goto L1300;
	}
	nxnode = dforw[rnode];
	if (nxnode > 0) {
	    dbakw[nxnode] = pvnode;
	}
	if (pvnode > 0) {
	    dforw[pvnode] = nxnode;
	}
	npv = -pvnode;
	if (pvnode < 0) {
	    dhead[npv] = nxnode;
	}
L1300:
	jstrt = xadj[rnode];
	jstop = xadj[rnode + 1] - 1;
	xqnbr = jstrt;
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
	    nabor = adjncy[j];
	    if (nabor == 0) {
		goto L1500;
	    }
	    if (marker[nabor] >= *tag) {
		goto L1400;
	    }
	    adjncy[xqnbr] = nabor;
	    ++xqnbr;
L1400:
	    ;
	}
L1500:
	nqnbrs = xqnbr - jstrt;
	if (nqnbrs > 0) {
	    goto L1600;
	}
	qsize[*mdnode] += qsize[rnode];
	qsize[rnode] = 0;
	marker[rnode] = *maxint;
	dforw[rnode] = -(*mdnode);
	dbakw[rnode] = -(*maxint);
	goto L1700;
L1600:
	dforw[rnode] = nqnbrs + 1;
	dbakw[rnode] = 0;
	adjncy[xqnbr] = *mdnode;
	++xqnbr;
	if (xqnbr <= jstop) {
	    adjncy[xqnbr] = 0;
	}

L1700:
	;
    }
L1800:
    return 0;

}







 int mmdupd_(ehead, neqns, xadj, adjncy, delta, mdeg, dhead,
	dforw, dbakw, qsize, llist, marker, maxint, tag)
integer *ehead, *neqns, *xadj, *adjncy, *delta, *mdeg, *dhead, *dforw, *dbakw,
	 *qsize, *llist, *marker, *maxint, *tag;
{
       integer i__1, i__2;

       static GENMMD_THREAD_LOCAL integer node, mtag, link, mdeg0, i, j, enode, fnode, nabor, elmnt,
	    istop, jstop, q2head, istrt, jstrt, qxhead, iq2, deg, deg0;





       --marker;
    --llist;
    --qsize;
    --dbakw;
    --dforw;
    --dhead;
    --adjncy;
    --xadj;

       mdeg0 = *mdeg + *delta;
    elmnt = *ehead;
L100:
    if (elmnt <= 0) {
	return 0;
    }
    mtag = *tag + mdeg0;
    if (mtag < *maxint) {
	goto L300;
    }
    *tag = 1;
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
	if (marker[i] < *maxint) {
	    marker[i] = 0;
	}
    }
    mtag = *tag + mdeg0;
L300:
    q2head = 0;
    qxhead = 0;
    deg0 = 0;
    link = elmnt;
L400:
    istrt = xadj[link];
    istop = xadj[link + 1] - 1;
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
	enode = adjncy[i];
	link = -enode;
	if (enode < 0) {
	    goto L400;
	} else if (enode == 0) {
	    goto L800;
	} else {
	    goto L500;
	}

L500:
	if (qsize[enode] == 0) {
	    goto L700;
	}
	deg0 += qsize[enode];
	marker[enode] = mtag;
	if (dbakw[enode] != 0) {
	    goto L700;
	}
	if (dforw[enode] == 2) {
	    goto L600;
	}
	llist[enode] = qxhead;
	qxhead = enode;
	goto L700;
L600:
	llist[enode] = q2head;
	q2head = enode;
L700:
	;
    }
L800:
    enode = q2head;
    iq2 = 1;
L900:
    if (enode <= 0) {
	goto L1500;
    }
    if (dbakw[enode] != 0) {
	goto L2200;
    }
    ++(*tag);
    deg = deg0;
    istrt = xadj[enode];
    nabor = adjncy[istrt];
    if (nabor == elmnt) {
	nabor = adjncy[istrt + 1];
    }
    link = nabor;
    if (dforw[nabor] < 0) {
	goto L1000;
    }
    deg += qsize[nabor];
    goto L2100;
L1000:
    istrt = xadj[link];
    istop = xadj[link + 1] - 1;
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
	node = adjncy[i];
	link = -node;
	if (node == enode) {
	    goto L1400;
	}
	if (node < 0) {
	    goto L1000;
	} else if (node == 0) {
	    goto L2100;
	} else {
	    goto L1100;
	}

L1100:
	if (qsize[node] == 0) {
	    goto L1400;
	}
	if (marker[node] >= *tag) {
	    goto L1200;
	}
	marker[node] = *tag;
	deg += qsize[node];
	goto L1400;
L1200:
	if (dbakw[node] != 0) {
	    goto L1400;
	}
	if (dforw[node] != 2) {
	    goto L1300;
	}
	qsize[enode] += qsize[node];
	qsize[node] = 0;
	marker[node] = *maxint;
	dforw[node] = -enode;
	dbakw[node] = -(*maxint);
	goto L1400;
L1300:
	if (dbakw[node] == 0) {
	    dbakw[node] = -(*maxint);
	}
L1400:
	;
    }
    goto L2100;
L1500:
    enode = qxhead;
    iq2 = 0;
L1600:
    if (enode <= 0) {
	goto L2300;
    }
    if (dbakw[enode] != 0) {
	goto L2200;
    }
    ++(*tag);
    deg = deg0;
    istrt = xadj[enode];
    istop = xadj[enode + 1] - 1;
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
	nabor = adjncy[i];
	if (nabor == 0) {
	    goto L2100;
	}
	if (marker[nabor] >= *tag) {
	    goto L2000;
	}
	marker[nabor] = *tag;
	link = nabor;
	if (dforw[nabor] < 0) {
	    goto L1700;
	}
	deg += qsize[nabor];
	goto L2000;
L1700:
	jstrt = xadj[link];
	jstop = xadj[link + 1] - 1;
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
	    node = adjncy[j];
	    link = -node;
	    if (node < 0) {
		goto L1700;
	    } else if (node == 0) {
		goto L2000;
	    } else {
		goto L1800;
	    }

L1800:
	    if (marker[node] >= *tag) {
		goto L1900;
	    }
	    marker[node] = *tag;
	    deg += qsize[node];
L1900:
	    ;
	}
L2000:
	;
    }
L2100:
    deg = deg - qsize[enode] + 1;
    fnode = dhead[deg];
    dforw[enode] = fnode;
    dbakw[enode] = -deg;
    if (fnode > 0) {
	dbakw[fnode] = enode;
    }
    dhead[deg] = enode;
    if (deg < *mdeg) {
	*mdeg = deg;
    }
L2200:
    enode = llist[enode];
    if (iq2 == 1) {
	goto L900;
    }
    goto L1600;
L2300:
    *tag = mtag;
    elmnt = llist[elmnt];
    goto L100;

}








 int mmdnum_(neqns, perm, invp, qsize)
integer *neqns, *perm, *invp, *qsize;
{
       integer i__1;

       static GENMMD_THREAD_LOCAL integer node, root, nextf, father, nqsize, num;





       --qsize;
    --invp;
    --perm;

       i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
	nqsize = qsize[node];
	if (nqsize <= 0) {
	    perm[node] = invp[node];
	}
	if (nqsize > 0) {
	    perm[node] = -invp[node];
	}
    }
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
	if (perm[node] > 0) {
	    goto L500;
	}
	father = node;
L200:
	if (perm[father] > 0) {
	    goto L300;
	}
	father = -perm[father];
	goto L200;
L300:
	root = father;
	num = perm[root] + 1;
	invp[node] = -num;
	perm[root] = num;
	father = node;
L400:
	nextf = -perm[father];
	if (nextf <= 0) {
	    goto L500;
	}
	perm[father] = -root;
	father = nextf;
	goto L400;
L500:
	;
    }
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
	num = -invp[node];
	invp[node] = num;
	perm[num] = node;
    }
    return 0;

}
