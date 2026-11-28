#ifndef __BIT_SCAN_H_
#define __BIT_SCAN_H_


#include <stdint.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static inline int32_t bit_leading_u32(uint32_t value)
{
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, value);
    return (int32_t)index;
#else
    static const int32_t table[32] =
    {
         0,  9,  1, 10, 13, 21,  2, 29,
        11, 14, 16, 18, 22, 25,  3, 30,
         8, 12, 20, 28, 15, 17, 24,  7,
        19, 27, 23,  6, 26,  5,  4, 31
    };
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return table[(value * 0x07C4ACDDu) >> 27];
#endif
}

static inline int32_t bit_trailing_u32(uint32_t value)
{
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, value);
    return (int32_t)index;
#else
    static const int32_t table[32] =
    {
         0,  1, 28,  2, 29, 14, 24,  3,
        30, 22, 20, 15, 25, 17,  4,  8,
        31, 27, 13, 23, 21, 19, 16,  7,
        26, 12, 18,  6, 11,  5, 10,  9
    };
    uint32_t key = (uint32_t)((value & (0u - value)) * 0x077CB531u) >> 27;
    return table[key];
#endif
}

static inline int32_t bit_leading_u64(uint64_t value)
{
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long index;
    _BitScanReverse64(&index, value);
    return (int32_t)index;
#else
    uint32_t hi = (uint32_t)(value >> 32);
    if (hi != 0)
        return bit_leading_u32(hi) + 32;
    return bit_leading_u32((uint32_t)value);
#endif
}

static inline int32_t bit_trailing_u64(uint64_t value)
{
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long index;
    _BitScanForward64(&index, value);
    return (int32_t)index;
#else
    uint32_t lo = (uint32_t)value;
    if (lo != 0)
        return bit_trailing_u32(lo);
    return bit_trailing_u32((uint32_t)(value >> 32)) + 32;
#endif
}

static inline bool bit_is_pow2_u32(uint32_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

static inline uint64_t bit_round_up_pow2_u32(uint32_t value)
{
    if (value == 0)
        return 1ull;
    int32_t leading = bit_leading_u32(value);
    uint32_t mask = (1u << leading);
    if ((value & ~mask) == 0)
        return (uint64_t)value;
    return ((uint64_t)mask << 1);
}

static inline uint32_t bit_round_down_pow2_u32(uint32_t value)
{
    if (value == 0)
        return 0;
    return (1u << bit_leading_u32(value));
}

#endif
