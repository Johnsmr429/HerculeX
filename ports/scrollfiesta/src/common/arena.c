#include "arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>

union align {
    double      d;
    long        l;
    void       *p;
    long double ld;
};

#define ALIGNMENT (sizeof(union align))

#define CHUNK_SIZE ((size_t)(64 * 1024 * 1024))

#define MAX_FREE_CHUNKS 10

struct Arena_chunk {
    struct Arena_chunk *prev;
    char               *avail;
    char               *limit;
};

struct Arena_T {
    struct Arena_chunk *current;
    struct Arena_chunk *freelist;
    int                 free_count;
    size_t              chunk_size;
};

#define MIN_CHUNK_SIZE ((size_t)4096)

static volatile int arena_trace_state = -1;
static size_t arena_trace_bytes = 0;
static volatile long long arena_live = 0, arena_peak = 0, arena_peak_mark = 0;
static volatile long long arena_parked = 0;

#ifdef _MSC_VER
#include <intrin.h>
static long long arena_add(volatile long long *p, long long d)
{
    return _InterlockedExchangeAdd64((volatile __int64 *)p, d) + d;
}
static int arena_raise(volatile long long *p, long long v)
{
    long long old = *p;
    while (v > old) {
        long long seen = _InterlockedCompareExchange64((volatile __int64 *)p, v, old);
        if (seen == old) return 1;
        old = seen;
    }
    return 0;
}
#else
static long long arena_add(volatile long long *p, long long d)
{
    return __atomic_add_fetch(p, d, __ATOMIC_RELAXED);
}
static int arena_raise(volatile long long *p, long long v)
{
    long long old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (v > old)
        if (__atomic_compare_exchange_n(p, &old, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return 1;
    return 0;
}
#endif

static int arena_tracing(void)
{
    if (arena_trace_state < 0) {
        const char *env = getenv("VESUVIUS_ARENA_TRACE_MB");
        long mb = env ? strtol(env, NULL, 10) : 0;
        arena_trace_bytes = mb > 0 ? (size_t)mb << 20 : 0;
        arena_trace_state = mb > 0;
    }
    return arena_trace_state > 0;
}

static double arena_gb(long long bytes) { return (double)bytes / (double)(1u << 30); }

static void arena_trace(long long bytes, int sign, long long parked, const char *what, const char *file, int line)
{
    long long live = arena_add(&arena_live, sign * bytes);
    long long idle = arena_add(&arena_parked, parked);
    if (sign > 0 && arena_raise(&arena_peak, live)) {
        long long step = (long long)arena_trace_bytes;
        long long mark = arena_peak_mark;
        if (live >= mark + step && arena_raise(&arena_peak_mark, live - live % step))
            fprintf(stderr, "[arena] peak %.2f GB at %s:%d\n", arena_gb(live), file ? file : "?", line);
    }
    if ((size_t)bytes >= arena_trace_bytes)
        fprintf(stderr, "[arena] %s %.3f GB at %s:%d (live %.2f GB, peak %.2f GB, free lists %.2f GB)\n", what,
                arena_gb(bytes), file ? file : "?", line, arena_gb(live), arena_gb(arena_peak), arena_gb(idle));
    {
        static volatile long long last_status = 0;
        long long now = (long long)time(NULL), last = last_status;
        if (now - last >= 30 && arena_raise(&last_status, now))
            fprintf(stderr, "[arena] status live %.2f GB, free lists %.2f GB, peak %.2f GB\n",
                    arena_gb(live), arena_gb(idle), arena_gb(arena_peak));
    }
}

static long long chunk_bytes(const struct Arena_chunk *c)
{
    return (long long)(c->limit - (const char *)c);
}

#if defined(__SANITIZE_ADDRESS__)
#  define ARENA_ASAN 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ARENA_ASAN 1
#  endif
#endif
#ifndef ARENA_ASAN
#  define ARENA_ASAN 0
#endif
#if ARENA_ASAN
#  include <sanitizer/asan_interface.h>
#  define ARENA_REDZONE ((size_t)32)
#else
#  define ARENA_REDZONE ((size_t)0)
#endif

static inline void arena_poison(const void *p, size_t n)
{
#if ARENA_ASAN
    ASAN_POISON_MEMORY_REGION(p, n);
#else
    (void)p; (void)n;
#endif
}

static inline void arena_unpoison(const void *p, size_t n)
{
#if ARENA_ASAN
    ASAN_UNPOISON_MEMORY_REGION(p, n);
#else
    (void)p; (void)n;
#endif
}

static size_t roundup(size_t nbytes)
{
    size_t rem = nbytes % ALIGNMENT;
    if (rem != 0) {
        nbytes += ALIGNMENT - rem;
    }
    return nbytes;
}

#if defined(_MSC_VER) && !defined(__SANITIZE_ADDRESS__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define ARENA_HEAP_HEADER 16
static __declspec(thread) HANDLE arena_thread_heap;
static void *arena_block_alloc(size_t bytes)
{
    if (!arena_thread_heap) arena_thread_heap = HeapCreate(0, 0, 0);
    if (!arena_thread_heap || bytes > (size_t)-1 - ARENA_HEAP_HEADER) return NULL;
    char *p = (char *)HeapAlloc(arena_thread_heap, 0, bytes + ARENA_HEAP_HEADER);
    if (!p) return NULL;
    *(HANDLE *)p = arena_thread_heap;
    return p + ARENA_HEAP_HEADER;
}
static void arena_block_free(void *ptr)
{
    if (!ptr) return;
    char *p = (char *)ptr - ARENA_HEAP_HEADER;
    HeapFree(*(HANDLE *)p, 0, p);
}
#else
static void *arena_block_alloc(size_t bytes) { return malloc(bytes); }
static void arena_block_free(void *ptr) { free(ptr); }
#endif

static struct Arena_chunk *chunk_new(size_t min_bytes, size_t chunk_size)
{
    size_t header_size = roundup(sizeof(struct Arena_chunk));
    size_t data_size = min_bytes > chunk_size ? min_bytes : chunk_size;
    size_t total = header_size + data_size;
    struct Arena_chunk *c = arena_block_alloc(total);
    if (c == NULL) {
        return NULL;
    }
    c->prev  = NULL;
    c->avail = (char *)c + header_size;
    c->limit = (char *)c + total;
    arena_poison(c->avail, data_size);
    return c;
}

static void chunk_reset(struct Arena_chunk *c)
{
    size_t header_size = roundup(sizeof(struct Arena_chunk));
    c->avail = (char *)c + header_size;
    arena_poison(c->avail, (size_t)(c->limit - c->avail));
}

static void chunk_free(struct Arena_chunk *c)
{
    arena_unpoison(c, (size_t)(c->limit - (char *)c));
    arena_block_free(c);
}

static void *chunk_carve(struct Arena_chunk *c, size_t nbytes, size_t req)
{
    void *ptr = c->avail;
    c->avail += nbytes;
    arena_unpoison(ptr, req);
    return ptr;
}

Arena_T Arena_new(void)
{
    Arena_T arena = malloc(sizeof(*arena));
    if (arena == NULL) {
        RAISE(Arena_Failed);
        return NULL;
    }
    arena->current    = NULL;
    arena->freelist   = NULL;
    arena->free_count = 0;
    arena->chunk_size = CHUNK_SIZE;
    return arena;
}

Arena_T Arena_new_sized(size_t chunk_bytes)
{
    Arena_T arena = Arena_new();
    arena->chunk_size = roundup(chunk_bytes > MIN_CHUNK_SIZE ? chunk_bytes : MIN_CHUNK_SIZE);
    return arena;
}

void Arena_dispose(Arena_T *ap)
{
    assert(ap && *ap);
    Arena_free(*ap);

    struct Arena_chunk *c = (*ap)->freelist;
    while (c != NULL) {
        struct Arena_chunk *prev = c->prev;
        if (arena_trace_state > 0) arena_trace(chunk_bytes(c), -1, -chunk_bytes(c), "free", "Arena_dispose", 0);
        chunk_free(c);
        c = prev;
    }

    free(*ap);
    *ap = NULL;
}

void Arena_free(Arena_T arena)
{
    assert(arena);
    struct Arena_chunk *c = arena->current;
    while (c != NULL) {
        struct Arena_chunk *prev = c->prev;
        if (arena->free_count < MAX_FREE_CHUNKS) {
            chunk_reset(c);
            c->prev  = arena->freelist;
            arena->freelist = c;
            arena->free_count++;
            if (arena_trace_state > 0) arena_trace(chunk_bytes(c), 0, chunk_bytes(c), "kept on free list", "Arena_free", 0);
        } else {
            if (arena_trace_state > 0) arena_trace(chunk_bytes(c), -1, 0, "free", "Arena_free", 0);
            chunk_free(c);
        }
        c = prev;
    }
    arena->current = NULL;
}

void *Arena_alloc(Arena_T arena, size_t nbytes, const char *file, int line)
{
    assert(arena);
    if (nbytes == 0) { nbytes = 1; }

    const size_t req = nbytes;
    nbytes = roundup(nbytes) + ARENA_REDZONE;

    struct Arena_chunk *c = arena->current;
    if (c != NULL && (size_t)(c->limit - c->avail) >= nbytes) {
        return chunk_carve(c, nbytes, req);
    }

    struct Arena_chunk **pp = &arena->freelist;
    while (*pp != NULL) {
        struct Arena_chunk *fc = *pp;
        if ((size_t)(fc->limit - fc->avail) >= nbytes) {
            *pp = fc->prev;
            arena->free_count--;
            if (arena_trace_state > 0) arena_add(&arena_parked, -chunk_bytes(fc));
            fc->prev = arena->current;
            arena->current = fc;
            return chunk_carve(fc, nbytes, req);
        }
        pp = &(*pp)->prev;
    }

    struct Arena_chunk *nc = chunk_new(nbytes, arena->chunk_size);
    if (nc == NULL) {
        if (file) {
            fprintf(stderr, "Arena_alloc: OOM at %s:%d (%zu bytes)\n",
                    file, line, nbytes);
        }
        RAISE(Arena_Failed);
        return NULL;
    }
    if (arena_tracing()) arena_trace(chunk_bytes(nc), +1, 0, "malloc", file, line);
    nc->prev = arena->current;
    arena->current = nc;
    return chunk_carve(nc, nbytes, req);
}

void *Arena_calloc(Arena_T arena, size_t count, size_t size,
                   const char *file, int line)
{
    if (size != 0 && count > SIZE_MAX / size) {
        if (file) {
            fprintf(stderr, "Arena_calloc: size overflow at %s:%d "
                    "(%zu * %zu)\n", file, line, count, size);
        }
        RAISE(Arena_Failed);
        return NULL;
    }
    size_t nbytes = count * size;
    void *ptr = Arena_alloc(arena, nbytes, file, line);
    memset(ptr, 0, nbytes);
    return ptr;
}

Arena_Mark Arena_save(Arena_T arena)
{
    Arena_Mark mark;
    assert(arena);
    if (arena->current == NULL) {
        struct Arena_chunk *nc = chunk_new(0, arena->chunk_size);
        if (nc == NULL) {
            RAISE(Arena_Failed);
        }
        if (arena_tracing()) arena_trace(chunk_bytes(nc), +1, 0, "malloc", "Arena_save", 0);
        nc->prev = NULL;
        arena->current = nc;
    }
    mark.avail = arena->current->avail;
    mark.chunk = arena->current;
    return mark;
}

void Arena_restore(Arena_T arena, Arena_Mark mark)
{
    assert(arena);
    while (arena->current != mark.chunk) {
        struct Arena_chunk *c = arena->current;
        assert(c != NULL);
        arena->current = c->prev;
        if (arena->free_count < MAX_FREE_CHUNKS) {
            chunk_reset(c);
            c->prev  = arena->freelist;
            arena->freelist = c;
            arena->free_count++;
            if (arena_trace_state > 0) arena_trace(chunk_bytes(c), 0, chunk_bytes(c), "kept on free list", "Arena_restore", 0);
        } else {
            if (arena_trace_state > 0) arena_trace(chunk_bytes(c), -1, 0, "free", "Arena_restore", 0);
            chunk_free(c);
        }
    }
    arena_poison(mark.avail, (size_t)(arena->current->avail - mark.avail));
    arena->current->avail = mark.avail;
}
