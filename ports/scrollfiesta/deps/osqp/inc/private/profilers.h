#ifndef PROFILERS_H_
#define PROFILERS_H_

#include "osqp_configure.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
       OSQP_PROFILER_SEC_SETUP,
    OSQP_PROFILER_SEC_SCALE,
    OSQP_PROFILER_SEC_OPT_SOLVE,
    OSQP_PROFILER_SEC_ADMM_ITER,
    OSQP_PROFILER_SEC_ADMM_KKT_SOLVE,
    OSQP_PROFILER_SEC_ADMM_UPDATE,
    OSQP_PROFILER_SEC_ADMM_PROJ,
    OSQP_PROFILER_SEC_POLISH,

       OSQP_PROFILER_SEC_LINSYS_INIT,
    OSQP_PROFILER_SEC_LINSYS_SOLVE,
    OSQP_PROFILER_SEC_LINSYS_SYM_FAC,
    OSQP_PROFILER_SEC_LINSYS_NUM_FAC,
    OSQP_PROFILER_SEC_LINSYS_BACKSOLVE,
    OSQP_PROFILER_SEC_LINSYS_MVM,

       OSQP_PROFILER_SEC_ARRAY_LAST
} OSQPProfilerSection;

typedef enum {
       OSQP_PROFILER_EVENT_RHO_UPDATE,

       OSQP_PROFILER_EVENT_ARRAY_LAST
} OSQPProfilerEvent;

typedef struct {
    const char* name;
    const char* desc;
    int level;
} OSQPProfilerItemInfo;

void _osqp_profiler_init(int level);

void _osqp_profiler_update_level(int level);

void _osqp_profiler_sec_push(OSQPProfilerSection section);

void _osqp_profiler_sec_pop(OSQPProfilerSection section);

void _osqp_profiler_event_mark(OSQPProfilerEvent event);

#ifdef OSQP_PROFILER_ANNOTATIONS
#define osqp_profiler_init(level)         _osqp_profiler_init(level)
#define osqp_profiler_update_level(level) _osqp_profiler_update_level(level)
#define osqp_profiler_sec_push(sec)       _osqp_profiler_sec_push(sec)
#define osqp_profiler_sec_pop(sec)        _osqp_profiler_sec_pop(sec)
#define osqp_profiler_event_mark(event)   _osqp_profiler_event_mark(event)

extern OSQPProfilerItemInfo osqp_profiler_sections[];

extern OSQPProfilerItemInfo osqp_profiler_events[];

#else
#define osqp_profiler_init(level)
#define osqp_profiler_update_level(level)
#define osqp_profiler_sec_push(sec)
#define osqp_profiler_sec_pop(sec)
#define osqp_profiler_event_mark(event)
#endif


#ifdef __cplusplus
}
#endif

#endif
