#ifndef VES_POOL_H
#define VES_POOL_H
#include <stddef.h>

typedef void (*VesPoolFn)(void *ctx, size_t begin, size_t end, int slot);

void VesPool_configure(int cores);
int  VesPool_cores(void);
void VesPool_job_enter(void);
void VesPool_job_leave(void);
int  VesPool_share(void);
int  VesPool_run(size_t n, size_t chunk, VesPoolFn fn, void *ctx, int max_helpers);
void VesPool_counts(size_t *runs, size_t *helpers);
int  VesPool_selftest(void);
#endif
