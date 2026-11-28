#ifndef TAUCS_CONFIG_TESTS_H
#define TAUCS_CONFIG_TESTS_H

#define TAUCS_BLAS_UNDERSCORE

#if defined(_WIN32)
#  ifndef OSTYPE_win32
#    define OSTYPE_win32
#  endif
#else
#  ifndef OSTYPE_linux
#    define OSTYPE_linux
#  endif
#endif


#endif
