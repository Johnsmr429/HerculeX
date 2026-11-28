#include "kkt.h"


static void _kkt_shifts_param1(OSQPCscMatrix* KKT,
                               OSQPFloat      param1,
                               OSQPInt        n,
                               OSQPInt        format) {
  OSQPInt i;
  OSQPInt offset = format == 0 ? 1 : 0;
  for(i = 0; i < n; i++){ KKT->x[KKT->p[i+offset]-offset] += param1;}
  return;
}

static void _kkt_shifts_param2(OSQPCscMatrix* KKT,
                               OSQPFloat*     param2,
                               OSQPFloat      param2_sc,
                               OSQPInt        startcol,
                               OSQPInt        blockwidth,
                               OSQPInt        format) {

  OSQPInt i;
  OSQPInt offset = format == 0 ? 1 : 0;

  if(param2){
    for(i = 0; i < blockwidth; i++){ KKT->x[KKT->p[i + startcol + offset]-offset] -= param2[i];}
  }else{
    for(i = 0; i < blockwidth; i++){ KKT->x[KKT->p[i + startcol + offset]-offset] -= param2_sc;}
  }
}

#ifndef OSQP_EMBEDDED_MODE


static void _kkt_colcount_diag(OSQPCscMatrix* K,
                               OSQPInt        initcol,
                               OSQPInt        blockcols) {

    OSQPInt j;
    for(j = initcol; j < (initcol + blockcols); j++){
        K->p[j]++;
    }
    return;
}

static void _kkt_colcount_missing_diag(OSQPCscMatrix* K,
                                       OSQPCscMatrix* M,
                                       OSQPInt        initcol) {

    OSQPInt j;
    for (j = 0; j < M->n; j++){
               if((M->p[j] == M->p[j+1]) || (M->i[M->p[j+1]-1] != j)) {
            K->p[j + initcol]++;
        }
    }
    return;
}

static void _kkt_colcount_block(OSQPCscMatrix* K,
                                OSQPCscMatrix* M,
                                OSQPInt        initcol,
                                OSQPInt        istranspose) {

    OSQPInt nnzM, j;

    if(istranspose){
      nnzM = M->p[M->n];
      for (j = 0; j < nnzM; j++){
        K->p[M->i[j] + initcol]++;
      }
    }
    else {
           for (j = 0; j < M->n; j++){
          K->p[j + initcol] += M->p[j+1] - M->p[j];
      }
    }
    return;
}


static void _kkt_fill_block(OSQPCscMatrix* K,
                            OSQPCscMatrix* M,
                            OSQPInt*       MtoKKT,
                            OSQPInt        initrow,
                            OSQPInt        initcol,
                            OSQPInt        istranspose) {
    OSQPInt ii, jj, row, col, dest;

    for(ii=0; ii < M->n; ii++){
        for(jj = M->p[ii]; jj < M->p[ii+1]; jj++){
            if(istranspose){
                col = M->i[jj] + initcol;
                row = ii + initrow;
            }
            else {
                col = ii + initcol;
                row = M->i[jj] + initrow;
            }

            dest       = K->p[col]++;
            K->i[dest] = row;
            K->x[dest] = M->x[jj];
            if(MtoKKT != OSQP_NULL){MtoKKT[jj] = dest;}
        }
    }
    return;
}

static void _kkt_fill_diag_zeros(OSQPCscMatrix* K,
                                 OSQPInt*       rhotoKKT,
                                 OSQPInt        offset,
                                 OSQPInt        blockdim) {

    OSQPInt j, dest, col;
    for(j = 0; j < blockdim; j++){
        col         = j + offset;
        dest        = K->p[col];
        K->i[dest]  = col;
        K->x[dest]  = 0.0;
        K->p[col]++;
        if(rhotoKKT != OSQP_NULL){rhotoKKT[j] = dest;}
    }
    return;
}

static void _kkt_fill_missing_diag_zeros(OSQPCscMatrix* K,
                                         OSQPCscMatrix* M,
                                         OSQPInt        offset) {

    OSQPInt j, dest;
    for(j = 0; j < M->n; j++){
                      if((M->p[j] == M->p[j+1]) ||
           (M->i[M->p[j+1]-1] != j))
        {
            dest           = K->p[j + offset];
            K->i[dest]  = j + offset;
            K->x[dest]  = 0.0;
            K->p[j]++;;
        }
    }
    return;
}

static void _kkt_colcount_to_colptr(OSQPCscMatrix* K) {

    OSQPInt j, count;
    OSQPInt currentptr = 0;

    for(j = 0; j <= K->n; j++){
       count        = K->p[j];
       K->p[j]      = currentptr;
       currentptr  += count;
    }
    return;
}

static void _kkt_backshift_colptrs(OSQPCscMatrix* K) {

    int j;
    for(j = K->n; j > 0; j--){
        K->p[j] = K->p[j-1];
    }
    K->p[0] = 0;

    return;
}

static OSQPInt _count_diagonal_entries(OSQPCscMatrix* P) {

  OSQPInt j;
  OSQPInt count = 0;

  for(j = 0; j < P->n; j++){
          if((P->p[j+1] != P->p[j]) && (P->i[P->p[j+1]-1] == j) ){
      count++;
    }
  }
  return count;

}


static void _kkt_assemble_csr(OSQPCscMatrix* K,
                              OSQPInt*       PtoKKT,
                              OSQPInt*       AtoKKT,
                              OSQPInt*       rhotoKKT,
                              OSQPCscMatrix* P,
                              OSQPCscMatrix* A) {

          OSQPInt j;
    OSQPInt m = A->m;
    OSQPInt n = P->n;

          for(j=0; j <= (m+n); j++){K->p[j] = 0;}
    _kkt_colcount_missing_diag(K,P,0);
    _kkt_colcount_block(K,P,0,1);
    _kkt_colcount_block(K,A,0,0);
    _kkt_colcount_diag(K,n,m);

       _kkt_colcount_to_colptr(K);

       _kkt_fill_missing_diag_zeros(K,P,0);
    _kkt_fill_block(K,P,PtoKKT,0,0,1);

       _kkt_fill_block(K,A,AtoKKT,n,0,0);

       _kkt_fill_diag_zeros(K,rhotoKKT,n,m);

       _kkt_backshift_colptrs(K);

    return;
}

static void _kkt_assemble_csc(OSQPCscMatrix* K,
                              OSQPInt*       PtoKKT,
                              OSQPInt*       AtoKKT,
                              OSQPInt*       rhotoKKT,
                              OSQPCscMatrix* P,
                              OSQPCscMatrix* A) {

    OSQPInt j;
    OSQPInt m = A->m;
    OSQPInt n = P->n;

          for(j=0; j <= (m+n); j++){K->p[j] = 0;}
    _kkt_colcount_block(K,P,0,0);
    _kkt_colcount_missing_diag(K,P,0);
    _kkt_colcount_block(K,A,n,1);
    _kkt_colcount_diag(K,n,m);

       _kkt_colcount_to_colptr(K);

       _kkt_fill_block(K,P,PtoKKT,0,0,0);
    _kkt_fill_missing_diag_zeros(K,P,0);

       _kkt_fill_block(K,A,AtoKKT,0,n,1);

       _kkt_fill_diag_zeros(K,rhotoKKT,n,m);

       _kkt_backshift_colptrs(K);

    return;
}


OSQPCscMatrix* form_KKT(OSQPCscMatrix* P,
                        OSQPCscMatrix* A,
                        OSQPInt        format,
                        OSQPFloat      param1,
                        OSQPFloat*     param2,
                        OSQPFloat      param2_sc,
                        OSQPInt*       PtoKKT,
                        OSQPInt*       AtoKKT,
                        OSQPInt*       rhotoKKT) {

  OSQPInt   m,n;
  OSQPInt  nKKT, nnzKKT;
  OSQPInt  ndiagP;

  OSQPCscMatrix* KKT;

   m   = A->m;
  n   = P->n;
  nKKT = m + n;

   ndiagP = _count_diagonal_entries(P);

   nnzKKT = P->p[n] +
           n -
           ndiagP +
           A->p[n] +
           m;

   KKT = csc_spalloc(nKKT, nKKT, nnzKKT, 1, 0);
  if (!KKT) return OSQP_NULL;

  if(format == 0){
    _kkt_assemble_csc(KKT,PtoKKT,AtoKKT,rhotoKKT,P,A);
  }
  else {
    _kkt_assemble_csr(KKT,PtoKKT,AtoKKT,rhotoKKT,P,A);
  }
   _kkt_shifts_param1(KKT,param1,n,format);

    _kkt_shifts_param2(KKT,param2,param2_sc,n,m,format);

  return KKT;
}

#endif


#if OSQP_EMBEDDED_MODE != 1

void update_KKT_P(OSQPCscMatrix* KKT,
                  OSQPCscMatrix* P,
                  const OSQPInt* Px_new_idx,
                  OSQPInt        P_new_n,
                  OSQPInt*       PtoKKT,
                  OSQPFloat      param1,
                  OSQPInt        format) {
  OSQPInt j, Pidx, Kidx, row, offset, doall;

  if(P_new_n <= 0){return;}

    doall  = Px_new_idx == OSQP_NULL ? 1 : 0;
  offset = format == 0 ? 1 : 0;

  for (j = 0; j < P_new_n; j++) {
    Pidx = doall ? j : Px_new_idx[j];
    Kidx = PtoKKT[Pidx];
    KKT->x[Kidx] = P->x[Pidx];

          row  = P->i[Pidx];
    if((P->p[row] < P->p[row+1]) && ((P->p[row+offset] - offset) == Pidx)){
      KKT->x[Kidx] += param1;
    }
  }
  return;
}

void update_KKT_A(OSQPCscMatrix* KKT,
                  OSQPCscMatrix* A,
                  const OSQPInt* Ax_new_idx,
                  OSQPInt        A_new_n,
                  OSQPInt*       AtoKKT) {

  OSQPInt j, Aidx, Kidx, doall;

  if(A_new_n <= 0){return;}

    doall  = Ax_new_idx == OSQP_NULL ? 1 : 0;

   for (j = 0; j < A_new_n; j++) {
    Aidx = doall ? j : Ax_new_idx[j];
    Kidx = AtoKKT[Aidx];
    KKT->x[Kidx] = A->x[Aidx];
  }
  return;
}


void update_KKT_param2(OSQPCscMatrix* KKT,
                       OSQPFloat*     param2,
                       OSQPFloat      param2_sc,
                       OSQPInt*       param2toKKT,
                       OSQPInt        m) {
  OSQPInt i;

   if (param2) {
    for (i = 0; i < m; i++) {
      KKT->x[param2toKKT[i]] = -param2[i];
    }
  }
  else {
    for (i = 0; i < m; i++) {
      KKT->x[param2toKKT[i]] = -param2_sc;
    }
  }
}

#endif
