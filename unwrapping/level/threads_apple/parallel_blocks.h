// A loop whose rounds run at once, EACH_BLOCK(index, count) { ... } END_BLOCKS;, as a dispatch_apply call.
// A body may read the variables around it and write through pointers, and must not return.

#include <dispatch/dispatch.h>
#define EACH_BLOCK(index, count) dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t index)
#define END_BLOCKS )
