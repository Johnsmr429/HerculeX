
/*
    This is the colamd include file,

	http://www.cise.ufl.edu/~davis/colamd/colamd.h

    for use in the colamd.c, colamdmex.c, and symamdmex.c files located at

	http://www.cise.ufl.edu/~davis/colamd/

    See those files for a description of colamd and symamd, and for the
    copyright notice, which also applies to this file.

    August 3, 1998.  Version 1.0.
*/


#define COLAMD_KNOBS 20

#define COLAMD_STATS 20

#define COLAMD_DENSE_ROW 0

#define COLAMD_DENSE_COL 1

#define COLAMD_DEFRAG_COUNT 2

#define COLAMD_JUMBLED_COLS 3


int colamd_recommended
(
    int nnz,
    int n_row,
    int n_col
) ;

void colamd_set_defaults
(
    double knobs [COLAMD_KNOBS]
) ;

int colamd
(
    int n_row,
    int n_col,
    int Alen,
    int A [],
    int p [],
    double knobs [COLAMD_KNOBS]
) ;
