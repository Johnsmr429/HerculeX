#include "printing.h"
#include "csc_utils.h"


OSQPInt csc_is_eq(OSQPCscMatrix* A,
                  OSQPCscMatrix* B,
                  OSQPFloat      tol) {
  OSQPInt j, i;

   if (A->n != B->n) {
      return 0;
  }

  for (j = 0; j < A->n; j++) {
          if (A->p[j+1] != B->p[j+1]) {
        return 0;
    }

    for (i = A->p[j]; i < A->p[j + 1]; i++) {
      if ((A->i[i] != B->i[i]) ||
          (c_absval(A->x[i] - B->x[i]) > tol)) {
        return 0;
      }
    }
  }
  return 1;
}



#ifndef OSQP_EMBEDDED_MODE

static void* csc_malloc(OSQPInt n, OSQPInt size) {
  return c_malloc(n * size);
}

static void* csc_calloc(OSQPInt n, OSQPInt size) {
  return c_calloc(n, size);
}

#endif

static void prea_int_vec_copy(const OSQPInt* a, OSQPInt* b, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) b[i] = a[i];
}

static void prea_vec_copy(const OSQPFloat* a, OSQPFloat* b, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++)  b[i] = a[i];
}

static void float_vec_set_scalar(OSQPFloat* a, OSQPFloat sc, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) a[i] = sc;
}

static void int_vec_set_scalar(OSQPInt* a, OSQPInt sc, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) a[i] = sc;
}

OSQPInt csc_cumsum(OSQPInt* p, OSQPInt* c, OSQPInt n) {
  OSQPInt i, nz = 0;

  if (!p || !c) return -1;

  for (i = 0; i < n; i++)
  {
    p[i] = nz;
    nz  += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz;
}





#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* csc_spalloc(OSQPInt m,
                           OSQPInt n,
                           OSQPInt nzmax,
                           OSQPInt values,
                           OSQPInt triplet) {
  OSQPCscMatrix* A = c_calloc(1, sizeof(OSQPCscMatrix));

  if (!A) return OSQP_NULL;

  A->m     = m;
  A->n     = n;
  A->nzmax = nzmax = c_max(nzmax, 0);
  A->nz    = triplet ? 0 : -1;
  A->p     = csc_malloc(triplet ? nzmax : n + 1, sizeof(OSQPInt));
  A->i     = values ? csc_malloc(nzmax,  sizeof(OSQPInt)) : OSQP_NULL;
  A->x     = values ? csc_malloc(nzmax,  sizeof(OSQPFloat)) : OSQP_NULL;
  if (!A->p || (values && !A->i ) || (values && !A->x)){
    csc_spfree(A);
    return OSQP_NULL;
  } else return A;
}

void csc_spfree(OSQPCscMatrix* A) {
  if (A){
    if (A->p) c_free(A->p);
    if (A->i) c_free(A->i);
    if (A->x) c_free(A->x);
    c_free(A);
  }
}

OSQPCscMatrix* csc_submatrix_byrows(const OSQPCscMatrix* A,
                                          OSQPInt*       rows){

  OSQPInt        j;
  OSQPCscMatrix* R;
  OSQPInt        nzR = 0;
  OSQPInt        An = A->n;
  OSQPInt        Am = A->m;
  OSQPInt*       Ap = A->p;
  OSQPInt*       Ai = A->i;
  OSQPFloat*     Ax = A->x;
  OSQPInt*       Rp;
  OSQPInt*       Ri;
  OSQPFloat*     Rx;
  OSQPInt        Rm = 0;
  OSQPInt        ptr;
  OSQPInt*       rridx;

  rridx = (OSQPInt*)c_malloc(Am * sizeof(OSQPInt));
  if(!rridx) return OSQP_NULL;

     Rm    = 0;
  for(j = 0; j < Am; j++){
     if(rows[j]) rridx[j] = Rm++;
  }

   for (j = 0; j < Ap[An]; j++) {
    if(rows[A->i[j]]) nzR++;
  }

   R = csc_spalloc(Rm, An, nzR, 1, 0);
  if (!R) {
       c_free(rridx);
    return OSQP_NULL;
  }

   if (Rm == 0) {
    int_vec_set_scalar(R->p, 0, An + 1);
  }

  else{
    nzR = 0;
    Rp = R->p;
    Ri = R->i;
    Rx = R->x;

    for (j = 0; j < An; j++) {
      Rp[j] = nzR;
      for (ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
               if (rows[A->i[ptr]]) {
          Ri[nzR] = rridx[Ai[ptr]];
          Rx[nzR] = Ax[ptr];
          nzR++;
    }}}
       Rp[An] = nzR;
  }

  c_free(rridx);

  return R;
}

OSQPCscMatrix* triplet_to_csc(const OSQPCscMatrix* T, OSQPInt* TtoC) {

  OSQPInt    m, n, nz, p, k;
  OSQPInt*   Cp;
  OSQPInt*   Ci;
  OSQPInt*   w;
  OSQPInt*   Ti;
  OSQPInt*   Tj;
  OSQPFloat* Cx;
  OSQPFloat* Tx;
  OSQPCscMatrix* C;

  m  = T->m;
  n  = T->n;
  Ti = T->i;
  Tj = T->p;
  Tx = T->x;
  nz = T->nz;
  C  = csc_spalloc(m, n, nz, Tx != OSQP_NULL, 0);
  w  = csc_calloc(n, sizeof(OSQPInt));

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

  for (k = 0; k < nz; k++) w[Tj[k]]++;
  csc_cumsum(Cp, w, n);

  for (k = 0; k < nz; k++) {
    Ci[p = w[Tj[k]]++] = Ti[k];

    if (Cx) {
      Cx[p] = Tx[k];

      if (TtoC != OSQP_NULL) TtoC[k] = p;
    }
  }
  return csc_done(C, w, OSQP_NULL, 1);
}

OSQPCscMatrix* triplet_to_csr(const OSQPCscMatrix* T, OSQPInt* TtoC) {
  OSQPInt    m, n, nz, p, k;
  OSQPInt*   Cp;
  OSQPInt*   Cj;
  OSQPInt*   w;
  OSQPInt*   Ti;
  OSQPInt*   Tj;
  OSQPFloat* Cx;
  OSQPFloat* Tx;
  OSQPCscMatrix* C;

  m  = T->m;
  n  = T->n;
  Ti = T->i;
  Tj = T->p;
  Tx = T->x;
  nz = T->nz;
  C  = csc_spalloc(m, n, nz, Tx != OSQP_NULL, 0);
  w  = csc_calloc(m, sizeof(OSQPInt));

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);

  Cp = C->p;
  Cj = C->i;
  Cx = C->x;

  for (k = 0; k < nz; k++) w[Ti[k]]++;
  csc_cumsum(Cp, w, m);

  for (k = 0; k < nz; k++) {
    Cj[p = w[Ti[k]]++] = Tj[k];

    if (Cx) {
      Cx[p] = Tx[k];

      if (TtoC != OSQP_NULL) TtoC[k] = p;
    }
  }
  return csc_done(C, w, OSQP_NULL, 1);
}

#endif

void csc_extract_diag(const OSQPCscMatrix* A,
                            OSQPFloat*     d) {
  OSQPInt    i, ptr;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

   float_vec_set_scalar(d, 0.0, n);

   for(i = 0; i < n; i++) {
    for (ptr = Ap[i]; ptr < Ap[i + 1]; ptr++) {
      if (Ai[ptr] == i)
        d[i] = Ax[ptr];
    }
  }
}

#ifndef OSQP_EMBEDDED_MODE

OSQPInt* csc_pinv(const OSQPInt* p, OSQPInt n) {
  OSQPInt  k;
  OSQPInt* pinv;

  if (!p) return OSQP_NULL;

  pinv = csc_malloc(n, sizeof(OSQPInt));

  if (!pinv) return OSQP_NULL;

  for (k = 0; k < n; k++) pinv[p[k]] = k;
  return pinv;
}

OSQPCscMatrix* csc_symperm(const OSQPCscMatrix* A,
                           const OSQPInt*       pinv,
                                 OSQPInt*       AtoC,
                                 OSQPInt        values) {
  OSQPInt    i, j, p, q, i2, j2, n;
  OSQPInt*   Ap;
  OSQPInt*   Ai;
  OSQPInt*   Cp;
  OSQPInt*   Ci;
  OSQPInt*   w;
  OSQPFloat* Cx;
  OSQPFloat* Ax;
  OSQPCscMatrix* C;

  n  = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  C  = csc_spalloc(n, n, Ap[n], values && (Ax != OSQP_NULL),
                   0);
  w = csc_calloc(n, sizeof(OSQPInt));

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

  for (j = 0; j < n; j++)
  {
    j2 = pinv ? pinv[j] : j;

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;
      i2 = pinv ? pinv[i] : i;
      w[c_max(i2, j2)]++;
    }
  }
  csc_cumsum(Cp, w, n);

  for (j = 0; j < n; j++) {
    j2 = pinv ? pinv[j] : j;

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;

      i2                         = pinv ? pinv[i] : i;

      Ci[q = w[c_max(i2, j2)]++] = c_min(i2, j2);

      if (Cx) Cx[q] = Ax[p];

      if (AtoC) {
        AtoC[p] = q;
      }
    }
  }
  return csc_done(C, w, OSQP_NULL, 1);
}

OSQPCscMatrix* csc_copy(const OSQPCscMatrix* A) {
  OSQPCscMatrix* B = csc_spalloc(A->m, A->n, A->p[A->n], (A->x != OSQP_NULL), 0);

  if (!B) return OSQP_NULL;

  prea_int_vec_copy(A->p, B->p, A->n + 1);
  prea_int_vec_copy(A->i, B->i, A->p[A->n]);
  prea_vec_copy(A->x, B->x, A->p[A->n]);

  return B;
}




OSQPFloat* csc_to_dns(OSQPCscMatrix* M)
{
  OSQPInt i, j = 0;
  OSQPInt idx;

   OSQPFloat* A = (OSQPFloat *)c_calloc(M->m * M->n, sizeof(OSQPFloat));
  if (!A) return OSQP_NULL;

   for (idx = 0; idx < M->p[M->n]; idx++)
  {
       i = M->i[idx];

       while (M->p[j + 1] <= idx) {
      j++;
    }

       A[j * (M->m) + i] = M->x[idx];
  }
  return A;
}

OSQPCscMatrix* csc_done(OSQPCscMatrix* C,
                        void*          w,
                        void*          x,
                        OSQPInt        ok) {
  c_free(w);
  c_free(x);
  if (ok) return C;
  else {
    csc_spfree(C);
    return OSQP_NULL;
  }
}

#endif


















#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* triu_to_csc(OSQPCscMatrix* M) {
    OSQPCscMatrix* M_trip;
    OSQPCscMatrix* M_symm;
    OSQPInt        n;
    OSQPInt        ptr, i, j;
    OSQPInt        z_M = 0;

    if (M->m != M->n) {
        c_eprint("Matrix M not square");
        return OSQP_NULL;
    }
    n = M->n;

       M_trip = csc_spalloc(n, n, 2 * M->p[n], 1, 1);
    if (!M_trip) {
        c_eprint("Matrix extraction failed (out of memory)");
        return OSQP_NULL;
    }

    for (j = 0; j < n; j++) {
        for (ptr = M->p[j]; ptr < M->p[j+1]; ptr++) {
            i = M->i[ptr];
            M_trip->i[z_M] = i;
            M_trip->p[z_M] = j;
            M_trip->x[z_M] = M->x[ptr];
            z_M++;

                                  if (i < j) {
                M_trip->i[z_M] = j;
                M_trip->p[z_M] = i;
                M_trip->x[z_M] = M->x[ptr];
                z_M++;
            }
        }
    }
    M_trip->nz = z_M;

       M_symm = triplet_to_csc(M_trip, OSQP_NULL);
    M_symm->nzmax = z_M;

    csc_spfree(M_trip);
    return M_symm;
}

OSQPCscMatrix* vstack(OSQPCscMatrix* A, OSQPCscMatrix* B) {
    OSQPCscMatrix* M_trip;
    OSQPCscMatrix* M;
    OSQPInt        m1, m2;
    OSQPInt        n;
    OSQPInt        ptr, i, j;
    OSQPInt        z_M = 0;

    if (A->n != B->n) {
        c_eprint("Matrix A and B do not have the same number of columns");
        return OSQP_NULL;
    }
    m1 = A->m;
    m2 = B->m;
    n = A->n;

       M_trip = csc_spalloc(m1 + m2, n, A->nzmax + B->nzmax, 1, 1);
    if (!M_trip) {
        c_eprint("Matrix allocation failed (out of memory)");
        return OSQP_NULL;
    }

    for (j = 0; j < n; j++) {
        for (ptr = A->p[j]; ptr < A->p[j+1]; ptr++) {
            i = A->i[ptr];
            M_trip->i[z_M] = i;
            M_trip->p[z_M] = j;
            M_trip->x[z_M] = A->x[ptr];
            z_M++;
        }
    }
    for (j = 0; j < n; j++) {
        for (ptr = B->p[j]; ptr < B->p[j+1]; ptr++) {
            i = B->i[ptr] + m1;
            M_trip->i[z_M] = i;
            M_trip->p[z_M] = j;
            M_trip->x[z_M] = B->x[ptr];
            z_M++;
        }
    }
    M_trip->nz = z_M;

       M = triplet_to_csc(M_trip, OSQP_NULL);
    M->nzmax = z_M;

    csc_spfree(M_trip);
    return M;
}

#endif
