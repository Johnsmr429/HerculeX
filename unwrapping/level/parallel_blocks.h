// A loop whose rounds run at once:  EACH_BLOCK(index, count) { ... } END_BLOCKS;
// On macOS it is the dispatch_apply call it replaces. Elsewhere (Linux, where cc has no blocks) it is an OpenMP loop;
// runner/compiled_kernels.py builds with -fopenmp there. A body may read the variables around it and write through
// pointers, and must not return.

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#define EACH_BLOCK(index, count) dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t index)
#define END_BLOCKS )
#else
#include <stddef.h>
#define EACH_BLOCK(index, count) _Pragma("omp parallel for schedule(dynamic)") for (size_t index = 0; index < (size_t)(count); index++)
#define END_BLOCKS
#endif
