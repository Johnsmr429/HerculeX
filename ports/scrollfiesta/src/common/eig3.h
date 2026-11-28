#ifndef EIG3_INCLUDED
#define EIG3_INCLUDED

void Eig3_sym(const double a[3][3], double evals[3], double evecs[3][3]);

void Eig3_smallest(const double a[3][3], double *out_eval, double out_vec[3]);

int Eig3_selftest(void);

#endif
