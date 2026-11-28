#ifndef VES_OMP_H
#define VES_OMP_H

#ifdef _OPENMP

extern int  omp_get_max_threads(void);
extern int  omp_get_thread_num(void);
extern int  omp_get_num_threads(void);
extern int  omp_get_num_procs(void);
extern int  omp_in_parallel(void);
extern void omp_set_num_threads(int);
extern void omp_set_dynamic(int);
extern double omp_get_wtime(void);

#else

#include <time.h>

static inline int  omp_get_max_threads(void) { return 1; }
static inline int  omp_get_thread_num(void)  { return 0; }
static inline int  omp_get_num_threads(void) { return 1; }
static inline int  omp_get_num_procs(void)   { return 1; }
static inline int  omp_in_parallel(void)     { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
static inline void omp_set_dynamic(int n)     { (void)n; }
static inline double omp_get_wtime(void)  { return (double)clock() / CLOCKS_PER_SEC; }

#endif

#endif
