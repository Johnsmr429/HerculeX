#ifndef __BUF_H_
#define __BUF_H_


#include <string.h>
#include <stdlib.h>

#ifndef BUF_STANDALONE
#include "../core/allocators.h"
#endif

#ifdef _WIN32
#pragma warning(disable:4200)
#endif


#ifdef BUF_STANDALONE
struct Allocator
{
	void*	(*alloc)(size_t s);
	void*	(*realloc)(void* ptr, size_t new_size);
	void	(*free)(void* ptr);
};

static inline void* buf__std_alloc(size_t s) { return malloc(s); }
static inline void* buf__std_realloc(void* p, size_t s) { return realloc(p, s); }
static inline void  buf__std_free(void* p) { free(p); }

static inline Allocator* buf__get_default_allocator(void)
{
	static Allocator a = { buf__std_alloc, buf__std_realloc, buf__std_free };
	return &a;
}
#define allocator_get_cpp_allocator() buf__get_default_allocator()
#endif


#ifndef buf_offsetof
#define buf_offsetof(TYPE, MEMBER) ((size_t)&((TYPE *)0)->MEMBER)
#endif

#ifndef buf_containerof
#define buf_containerof(PTR, TYPE, MEMBER) \
	((TYPE *)((char *)(PTR) - buf_offsetof(TYPE, MEMBER)))
#endif

struct BufHeader
{
	size_t length;
	size_t capacity;
	struct Allocator* allocator;
	char data[];
};

#define buf_len(a)		((a) ? buf_get_internals(a)->length : 0)

#define buf_cap(a)		((a) ? buf_get_internals(a)->capacity : 0)

#define buf_free(a)		do { if (a) { buf_get_internals(a)->allocator->free(buf_get_internals(a)); a = nullptr; } } while(0)
#define buf__new_cap(old_cap) ((3 * (old_cap) / 2) < 16 ? 16 : (3 * (old_cap) / 2))

#define buf__needsgrow(a,n) ((a) == nullptr || buf_len(a) + (n) >= buf_cap(a))

#define buf__maybegrow(a,n) (buf__needsgrow(a,(n)) ? buf__sbgrow(a,buf__new_cap(buf_cap(a)+(n))) : 0)
#define buf__sbgrow(a,n)      (*((void **)&(a)) = buf_do_realloc_((a), (n), sizeof(*(a))))


#ifndef BUF_STANDALONE
void* buf_alloc_ex(struct Allocator* a);
void* buf_alloc(void);
#endif
static inline void* buf_do_realloc_(void *a, size_t nr, size_t sz);

void* buf_clone_(const void* src, size_t elem_size);

void buf_push_string(char** buf, const char* s);

void buf_printf(char** buf, const char* fmt, ...);


#define buf_for_each(it, a)						\
	for (auto __bfe_buf = (a), it = __bfe_buf; (it) < __bfe_buf + buf_len(__bfe_buf); (it)++)

#define buf_last(a)		((a)[buf_len(a) - 1])

#define buf_push(a,v) (buf__maybegrow(a,1), (a)[buf_get_internals(a)->length++] = (v))

#define buf_pop(a) (buf_get_internals(a)->length--, (a)[buf_get_internals(a)->length])

#define buf_erase(a,i)      buf_erasen(a,i,1)
#define buf_erasen(a,i,n)   (memmove(&(a)[i], &(a)[(i)+(n)], sizeof *(a) * (buf_get_internals(a)->length-(n)-(i))), buf_get_internals(a)->length -= (n))

#define buf_resize(a,n) (buf__sbgrow(a,(n)),buf_get_internals(a)->length=(n))

#define buf_resize_zeroed(a,n) do { \
	size_t __brz_old = buf_len(a); \
	buf_resize((a), (n)); \
	if ((size_t)(n) > __brz_old) \
		memset(&(a)[__brz_old], 0, sizeof(*(a)) * ((size_t)(n) - __brz_old)); \
} while(0)

#define buf_reserve(a,n) (*((void **)&(a)) = buf_do_realloc_((a), (n), sizeof(*(a))))

#define buf_reserve_at_least(a,n) do { \
	if (buf_cap(a) < (size_t)(n)) buf_reserve((a),(n)); \
} while(0)

#define buf_clear(a) ((a) ? (buf_get_internals(a)->length = 0) : 0)

#define buf_truncate(a,n) ((a) ? (buf_get_internals(a)->length = (size_t)(n)) : (size_t)0)

#define buf_swap_erase(a,i) do { (a)[i] = (a)[buf_get_internals(a)->length - 1]; buf_get_internals(a)->length--; } while(0)

#define buf_insert(a,i,v) do { buf__maybegrow(a,1); memmove(&(a)[(i)+1], &(a)[i], sizeof(*(a)) * (buf_get_internals(a)->length - (i))); (a)[i] = (v); buf_get_internals(a)->length++; } while(0)

#define buf_push_n(a,ptr,n) do { buf__maybegrow(a,n); memcpy(&(a)[buf_get_internals(a)->length], (ptr), sizeof(*(a)) * (n)); buf_get_internals(a)->length += (n); } while(0)

#define buf_concat(dst, src) do { \
	size_t __bc_n = buf_len(src); \
	if (__bc_n > 0) buf_push_n((dst), (src), __bc_n); \
} while(0)

#define buf_get_internals(a)							\
	((a) ? ((struct BufHeader *)(buf_containerof(a, struct BufHeader, data))) : (struct BufHeader *)nullptr)

#define buf_clone(src) ((decltype(src))buf_clone_((src), sizeof(*(src))))


#ifdef BUF_IMPLEMENTATION

static inline void* buf_do_realloc_(void *a, size_t nr, size_t sz)
{
	struct BufHeader* b;
	Allocator* allocator = a ? buf_get_internals(a)->allocator : allocator_get_cpp_allocator();

	b = (struct BufHeader*)allocator->realloc(buf_get_internals(a),
		sizeof(struct BufHeader) + nr * sz);
	if (!b) { abort(); }
	b->capacity = nr;
	b->allocator = allocator;
	if (!a) b->length = 0;
	return (void *)&(b->data);
}

#endif

#endif
