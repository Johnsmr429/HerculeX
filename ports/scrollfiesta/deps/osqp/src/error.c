#include "glob_opts.h"
#include "error.h"
#include "printing.h"
#include "osqp_api_constants.h"

const char *OSQP_ERROR_MESSAGE[] = {
  "Problem data validation.",
  "Solver settings validation.",
   "Linear system solver initialization.",
  "KKT matrix factorization.\nThe problem seems to be non-convex.",
  "Memory allocation.",
  "Solver workspace not initialized.",
  "Algebra libraries not loaded.",
  "Unable to open file for writing.",
  "Invalid defines for codegen",
  "Vector/matrix not initialized.",
  "Function not implemented.",

   "Unknown error code."
};


OSQPInt _osqp_error(enum osqp_error_type error_code,
		                const char*          function_name) {

   if (error_code != OSQP_NO_ERROR)
    c_print("ERROR in %s: %s\n", function_name, OSQP_ERROR_MESSAGE[error_code-1]);

  return (OSQPInt)error_code;
}

OSQPInt _osqp_error_line(enum osqp_error_type error_code,
                         const char*          function_name,
                         const char*          filename,
                         OSQPInt              line_number) {

   if (error_code != OSQP_NO_ERROR)
    c_print("ERROR in %s (%s:%" OSQP_INT_FMT "): %s\n", function_name, filename, line_number, OSQP_ERROR_MESSAGE[error_code-1]);

  return (OSQPInt)error_code;
}
