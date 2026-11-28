#ifndef RUN_CTX_INCLUDED
#define RUN_CTX_INCLUDED

#include <stddef.h>

#include "except.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const Except_T Sf_Cancelled;

typedef int  (*SfRunProgressFn)(void *user, const char *stage, double fraction);
typedef void (*SfRunLogFn)(void *user, const char *line);

typedef struct SfRunKV {
    const char *key;
    const char *value;
} SfRunKV;

typedef struct SfRunCtx {
    SfRunProgressFn progress;
    void           *progress_user;
    SfRunLogFn      log;
    void           *log_user;
    volatile int    cancel_requested;
    double          deadline_sec;
    int             timed_out;
    const SfRunKV  *tuning;
    size_t          n_tuning;
} SfRunCtx;

void      RunCtx_push(SfRunCtx *ctx);
void      RunCtx_pop(void);
SfRunCtx *RunCtx_current(void);

int  RunCtx_should_stop(void);

void RunCtx_check(void);

int  RunCtx_progress(const char *stage, double fraction);

void RunCtx_progress_check(const char *stage, double fraction);

void RunCtx_log(const char *fmt, ...);

const char *sf_env(const char *name);

#ifdef __cplusplus
}
#endif

#endif
