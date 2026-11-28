#include <stddef.h>

static void each_block(size_t count, void *context, void (*run)(void *, size_t)) {
#pragma omp parallel for schedule(dynamic)
    for (size_t block = 0; block < count; block++)
        run(context, block);
}
