// A loop whose rounds run at once, EACH_BLOCK(index, count) { ... } END_BLOCKS;, as an OpenMP loop.
// A body may read the variables around it and write through pointers, and must not return.

#include <stddef.h>
#define EACH_BLOCK(index, count)                                                                                       \
    _Pragma("omp parallel for schedule(dynamic)") for (size_t index = 0; index < (size_t)(count); index++)
#define END_BLOCKS
