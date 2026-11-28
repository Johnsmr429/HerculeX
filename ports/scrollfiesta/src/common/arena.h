#ifndef ARENA_INCLUDED
#define ARENA_INCLUDED

#include <stddef.h>

#include "except.h"

typedef struct Arena_T *Arena_T;

struct Arena_chunk;

typedef struct {
    char               *avail;
    struct Arena_chunk  *chunk;
} Arena_Mark;

Arena_T Arena_new(void);
Arena_T Arena_new_sized(size_t chunk_bytes);
void    Arena_dispose(Arena_T *ap);
void    Arena_free(Arena_T arena);

void   *Arena_alloc(Arena_T arena, size_t nbytes, const char *file, int line);
void   *Arena_calloc(Arena_T arena, size_t count, size_t size,
                     const char *file, int line);

Arena_Mark Arena_save(Arena_T arena);
void       Arena_restore(Arena_T arena, Arena_Mark mark);

#define ARENA_ALLOC(a, nbytes) \
    Arena_alloc((a), (size_t)(nbytes), __FILE__, __LINE__)
#define ARENA_CALLOC(a, count, size) \
    Arena_calloc((a), (size_t)(count), (size_t)(size), __FILE__, __LINE__)
#define ARENA_NEW(a, p) \
    ((p) = Arena_alloc((a), sizeof *(p), __FILE__, __LINE__))

#endif
