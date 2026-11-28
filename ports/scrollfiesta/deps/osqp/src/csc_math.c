
#include "glob_opts.h"
#include "osqp.h"


void vec_set_scalar(OSQPFloat* v, OSQPFloat val, OSQPInt n){
  OSQPInt i;
  for(i = 0; i< n; i++) v[i] = val;
}

void vec_mult_scalar(OSQPFloat* v, OSQPFloat val, OSQPInt n){
  OSQPInt i;
  for(i = 0; i< n; i++) v[i] *= val;
}

void vec_negate(OSQPFloat* v, OSQPInt n){
  OSQPInt i;
  for(i = 0; i< n; i++) v[i] = -v[i];
}




void csc_update_values(OSQPCscMatrix*   M,
                       const OSQPFloat* Mx_new,
                       const OSQPInt*   Mx_new_idx,
                             OSQPInt    M_new_n) {

  OSQPInt i;

   if (Mx_new_idx) {
    for (i = 0; i < M_new_n; i++) {
      M->x[Mx_new_idx[i]] = Mx_new[i];
    }
  }
  else{
    for (i = 0; i < M_new_n; i++) {
      M->x[i] = Mx_new[i];
    }
  }
}



void csc_scale(OSQPCscMatrix* A, OSQPFloat sc){
  OSQPInt i, nnzA;
  nnzA = A->p[A->n];
  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}


void csc_lmult_diag(OSQPCscMatrix* A, const OSQPFloat* d){

  OSQPInt    j, i;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

  for (j = 0; j < n; j++) {
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      Ax[i] *= d[Ai[i]];
                                             }
  }
}


void csc_rmult_diag(OSQPCscMatrix* A, const OSQPFloat* d){

  OSQPInt    j, i;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPFloat* Ax = A->x;

  for (j = 0; j < n; j++) {
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      Ax[i] *= d[j];
                                              }
  }
}

void csc_AtDA_extract_diag(const OSQPCscMatrix* A,
                           const OSQPFloat*     D,
                                 OSQPFloat*     d) {
  OSQPInt    j, i;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

   for (j = 0; j < n; j++) {
    d[j] = 0;
       for (i = Ap[j]; i < Ap[j + 1]; i++) {
      d[j] += Ax[i]*Ax[i]*D[Ai[i]];
    }
  }
}

void csc_Axpy_sym_triu(const OSQPCscMatrix* A,
                       const OSQPFloat*     x,
                             OSQPFloat*     y,
                             OSQPFloat      alpha,
                             OSQPFloat      beta) {

    OSQPInt    i, j;
    OSQPInt*   Ap = A->p;
    OSQPInt*   Ai = A->i;
    OSQPInt    An = A->n;
    OSQPInt    Am = A->m;
    OSQPFloat* Ax = A->x;

       if (beta == 0)        vec_set_scalar(y, 0.0, Am);
    else if (beta ==  1)  ;
    else if (beta == -1)  vec_negate(y, Am);
    else vec_mult_scalar(y,beta, Am);


       if (Ap[An] == 0 || alpha == 0.0) return;

    if (alpha == -1) {
               for (j = 0; j < An; j++) {
            for (i = Ap[j]; i < Ap[j + 1]; i++) {
                y[Ai[i]] -= Ax[i] * x[j];
                if(Ai[i] != j){
                    y[j]     -= Ax[i] * x[Ai[i]];
                }
    }}}

    else if(alpha == 1){
               for (j = 0; j < An; j++) {
            for (i = Ap[j]; i < Ap[j + 1]; i++) {
                y[Ai[i]] += Ax[i] * x[j];
                if(Ai[i] != j){
                    y[j]     += Ax[i] * x[Ai[i]];
                }
    }}}

    else{
               for (j = 0; j < An; j++) {
            for (i = Ap[j]; i < Ap[j + 1]; i++) {
                y[Ai[i]] += alpha*Ax[i] * x[j];
                if(Ai[i] != j){
                    y[j]     += alpha*Ax[i] * x[Ai[i]];
                }
    }}}
}

void csc_Axpy(const OSQPCscMatrix* A,
              const OSQPFloat*     x,
                    OSQPFloat*     y,
                    OSQPFloat      alpha,
                    OSQPFloat      beta) {

  OSQPInt    i, j;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPInt    An = A->n;
  OSQPInt    Am = A->m;
  OSQPFloat* Ax = A->x;

   if (beta == 0)        vec_set_scalar(y, 0.0, Am);
  else if (beta ==  1)  ;
  else if (beta == -1)  vec_negate(y, Am);
  else vec_mult_scalar(y,beta, Am);


   if (Ap[An] == 0 || alpha == 0.0) return;

  if (alpha == -1) {
       for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        y[Ai[i]] -= Ax[i] * x[j];
    }}}

  else if(alpha == +1){
       for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        y[Ai[i]] += Ax[i] * x[j];
    }}}

  else{
       for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        y[Ai[i]] += alpha*Ax[i] * x[j];
    }}}
}


void csc_Atxpy(const OSQPCscMatrix* A,
               const OSQPFloat*     x,
                     OSQPFloat*     y,
                     OSQPFloat      alpha,
                     OSQPFloat      beta) {
  OSQPInt    j, k;
  OSQPInt    An = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

   if (beta == 0)        vec_set_scalar(y, 0.0, An);
  else if (beta ==  1)  ;
  else if (beta == -1)  vec_negate(y, An);
  else vec_mult_scalar(y,beta, An);

   if (Ap[An] == 0 || alpha == 0.0) {
    return;
  }

    if(alpha == -1){
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          y[j] -= Ax[k] * x[Ai[k]];
    }}}

    else if(alpha == +1){
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          y[j] += Ax[k] * x[Ai[k]];
    }}}

    else{
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          y[j] += alpha*Ax[k] * x[Ai[k]];
    }}}
}









void csc_col_norm_inf(const OSQPCscMatrix* M, OSQPFloat* E) {

  OSQPInt    j, ptr;
  OSQPInt*   Mp = M->p;
  OSQPInt    Mn = M->n;
  OSQPFloat* Mx = M->x;

   vec_set_scalar(E, 0.0, Mn);

   for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      E[j] = c_max(c_absval(Mx[ptr]), E[j]);
    }
  }
}


void csc_row_norm_inf(const OSQPCscMatrix* M, OSQPFloat* E) {

  OSQPInt    i, j, ptr;
  OSQPInt*   Mp = M->p;
  OSQPInt*   Mi = M->i;
  OSQPInt    Mn = M->n;
  OSQPInt    Mm = M->m;
  OSQPFloat* Mx = M->x;

   vec_set_scalar(E, 0.0, Mm);

   for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      i    = Mi[ptr];
      E[i] = c_max(c_absval(Mx[ptr]), E[i]);
    }
  }
}


void csc_row_norm_inf_sym_triu(const OSQPCscMatrix* M, OSQPFloat* E) {

  OSQPInt   i, j, ptr;
  OSQPInt*   Mp = M->p;
  OSQPInt*   Mi = M->i;
  OSQPInt    Mn = M->n;
  OSQPInt    Mm = M->m;
  OSQPFloat* Mx = M->x;
  OSQPFloat  abs_x;

   vec_set_scalar(E, 0.0, Mm);

      for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      i     = Mi[ptr];
      abs_x = c_absval(Mx[ptr]);
      E[j]  = c_max(abs_x, E[j]);

      if (i != j) {
        E[i] = c_max(abs_x, E[i]);
      }
    }
  }
}
