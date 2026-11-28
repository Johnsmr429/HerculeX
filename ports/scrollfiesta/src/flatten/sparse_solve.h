#ifndef SPARSE_SOLVE_INCLUDED
#define SPARSE_SOLVE_INCLUDED

typedef enum {
    SPARSE_SPD = 0,
    SPARSE_SYM_INDEFINITE
} SparseMode;

int Sparse_solve_sym(int n, int nt,
                     const int *rows, const int *cols, const double *vals,
                     const double *b, double *x, SparseMode mode);

int Sparse_spd_dense_limit(void);

typedef struct SparseFactor *SparseFactor_T;

int  Sparse_factor_spd(int n, int nt,
                       const int *rows, const int *cols, const double *vals,
                       SparseFactor_T *out);
int  Sparse_factor_solve(SparseFactor_T f, const double *b, double *x);
int  Sparse_factor_solve_multi(SparseFactor_T f, const double *B, double *X,
                               int nrhs);
void Sparse_factor_free(SparseFactor_T *f);
int  Sparse_factor_spd_ccs(int n, const int *colptr, const int *rowind,
                           const double *values, SparseFactor_T *out);
int  Sparse_factor_refactor_ccs(SparseFactor_T f, const double *values);
typedef struct SparseRepeat *SparseRepeat_T;
int  Sparse_repeat_solve(SparseRepeat_T *cache, int n, int nt, const int *rows, const int *cols, const double *vals,
                         const double *b, double *x);
void Sparse_repeat_free(SparseRepeat_T *cache);
void Sparse_factor_cache_stats(unsigned long long *hits, unsigned long long *misses);

int Sparse_selftest(void);
int Sparse_benchmark(int side, int systems, int threads);

#endif
