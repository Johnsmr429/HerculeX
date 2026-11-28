#ifndef VES_ACTIVE_SET_QP_INCLUDED
#define VES_ACTIVE_SET_QP_INCLUDED
#include <stddef.h>

enum {
    ASQP_REASON_OK = 0,
    ASQP_REASON_INPUT,
    ASQP_REASON_MEMORY,
    ASQP_REASON_FACTOR,
    ASQP_REASON_SOLVE,
    ASQP_REASON_INFEASIBLE,
    ASQP_REASON_STEPS,
    ASQP_REASON_CAP,
    ASQP_REASON_DENSE,
    ASQP_REASON_ILLPOSED,
    ASQP_REASON_TIME
};

typedef struct ActiveSetQpStats {
    int    iterations;
    int    active;
    int    dropped;
    int    ridge_retries;
    int    reason;
    int    warm;
    double factor_seconds;
    double solve_seconds;
} ActiveSetQpStats;

const char *ActiveSetQp_reason(int reason);

int ActiveSetQp_solve(int n, int hn, const int *hr, const int *hc, const double *hv,
                      const double *target, int m, int an, const int *ar, const int *ac,
                      const double *av, const double *upper, double *result);

typedef struct ActiveSetQpFactor *ActiveSetQpFactor_T;
int  ActiveSetQp_factor(int n, int hn, const int *hr, const int *hc, const double *hv,
                        ActiveSetQpFactor_T *out, ActiveSetQpStats *stats);
int  ActiveSetQp_solve_factored(ActiveSetQpFactor_T factor, const double *target, int m, int an,
                                const int *ar, const int *ac, const double *av, const double *upper,
                                double *result, ActiveSetQpStats *stats);
int  ActiveSetQp_solve_warm(ActiveSetQpFactor_T factor, const double *target, int m, int an,
                            const int *ar, const int *ac, const double *av, const double *upper,
                            double *result, ActiveSetQpStats *stats, int *rows, int *count, int capacity);
void ActiveSetQp_free(ActiveSetQpFactor_T *factor);
int  ActiveSetQp_model(ActiveSetQpFactor_T factor, const double *target, double *model);
void ActiveSetQp_limit(ActiveSetQpFactor_T factor, double limit);
void ActiveSetQp_deadline(double deadline);

int ActiveSetQp_solve_stats(int n, int hn, const int *hr, const int *hc, const double *hv,
                            const double *target, int m, int an, const int *ar, const int *ac,
                            const double *av, const double *upper, double *result,
                            ActiveSetQpStats *stats);

int ActiveSetQp_selftest(void);

#endif
