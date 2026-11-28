#include <dispatch/dispatch.h>

static void each_block(size_t count, void *context, void (*run)(void *, size_t)) {
    dispatch_apply_f(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), context, run);
}
