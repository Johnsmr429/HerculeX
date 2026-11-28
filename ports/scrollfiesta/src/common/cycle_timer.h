#ifndef CYCLE_TIMER_INCLUDED
#define CYCLE_TIMER_INCLUDED

#include <stdint.h>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
static inline uint64_t cycle_now(void) { return (uint64_t)__rdtsc(); }
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t cycle_now(void) { return (uint64_t)__rdtsc(); }
#else
#include "ves_platform.h"
static inline uint64_t cycle_now(void)
{
    return (uint64_t)(ves_clock_sec() * 1.0e9);
}
#endif

#endif
