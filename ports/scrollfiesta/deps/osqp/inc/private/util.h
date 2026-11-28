#ifndef UTIL_H
#define UTIL_H


# include "osqp.h"
# include "types.h"

#define OSQP_UnusedVar(x)  (void)(x)



#ifdef __cplusplus
extern "C" {
#endif

# ifndef OSQP_EMBEDDED_MODE

OSQPSettings* copy_settings(const OSQPSettings* settings);

# endif

void c_strcpy(char       dest[],
              const char source[]);


# ifdef OSQP_ENABLE_PRINTING

void print_setup_header(const OSQPSolver* solver);

void print_header(void);

void print_summary(OSQPSolver* solver);

void print_polish(OSQPSolver* solver);

void print_footer(OSQPInfo* info,
                  OSQPInt   polishing);


# endif





#if defined(OSQP_ENABLE_DEBUG) && defined(OSQP_ENABLE_PRINTING)

#  include <stdio.h>

void print_csc_matrix(const OSQPCscMatrix* M,
                      const char*          name);

void dump_csc_matrix(const OSQPCscMatrix* M,
                     const char*          file_name);

void print_trip_matrix(const OSQPCscMatrix* M,
                       const char*          name);

void print_dns_matrix(const OSQPFloat* M,
                      OSQPInt          m,
                      OSQPInt          n,
                      const char*      name);

void print_vec(const OSQPFloat* v,
               OSQPInt          n,
               const char*      name);

void dump_vec(const OSQPFloat* v,
              OSQPInt          len,
              const char*      file_name);

void print_vec_int(const OSQPInt* x,
                   OSQPInt        n,
                   const char*    name);

# endif


#ifdef __cplusplus
}
#endif

#endif
