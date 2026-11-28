#include "ieee_binary.h"
#include "bit_scan.h"


static const uint32_t F32_NUM_TRAILING_BITS    = 23;
static const uint32_t F32_EXPONENT_BIAS        = 127;
static const uint32_t F32_MAX_BIASED_EXPONENT  = 255;
static const uint32_t F32_SIGN_MASK            = 0x80000000u;
static const uint32_t F32_NOT_SIGN_MASK        = 0x7FFFFFFFu;
static const uint32_t F32_BIASED_EXPONENT_MASK = 0x7F800000u;
static const uint32_t F32_TRAILING_MASK        = 0x007FFFFFu;

static const uint32_t F16_AVR_MIN_SUBNORMAL_ZERO   = 0x33000000u;
static const uint32_t F16_MIN_SUBNORMAL            = 0x33800000u;
static const uint32_t F16_MIN_NORMAL               = 0x38800000u;
static const uint32_t F16_MAX_NORMAL               = 0x477FE000u;
static const uint32_t F16_AVR_MAX_NORMAL_INFINITY  = 0x477FF000u;

static const uint32_t DIFF_NUM_ENCODING_BITS = 16;
static const uint32_t DIFF_NUM_TRAILING_BITS = 13;
static const uint32_t DIFF_PAYLOAD_SHIFT     = 13;
static const uint32_t INT_PART_MASK          = 0x007FE000u;
static const uint32_t FRC_PART_MASK          = 0x00001FFFu;
static const uint32_t FRC_HALF               = 0x00001000u;

static uint16_t ieee_convert_32_to_16(uint32_t inEncoding)
{
    uint32_t sign32     = (inEncoding & F32_SIGN_MASK);
    uint32_t biased32   = ((inEncoding & F32_BIASED_EXPONENT_MASK) >> F32_NUM_TRAILING_BITS);
    uint32_t trailing32 = (inEncoding & F32_TRAILING_MASK);
    uint32_t nonneg32   = (inEncoding & F32_NOT_SIGN_MASK);

    uint16_t sign16 = (uint16_t)(sign32 >> DIFF_NUM_ENCODING_BITS);
    uint16_t biased16, trailing16;
    uint32_t frcpart;

    if (biased32 == 0)
    {
               return sign16;
    }

    if (biased32 < F32_MAX_BIASED_EXPONENT)
    {
        if (nonneg32 <= F16_AVR_MIN_SUBNORMAL_ZERO)
            return sign16;

        if (nonneg32 <= F16_MIN_SUBNORMAL)
            return sign16 | IEEEBinary16::MIN_SUBNORMAL;

        if (nonneg32 < F16_MIN_NORMAL)
        {
                       trailing16 = (uint16_t)((trailing32 & INT_PART_MASK) >> DIFF_NUM_TRAILING_BITS);
            frcpart = (trailing32 & FRC_PART_MASK);
            if (frcpart > FRC_HALF || (frcpart == FRC_HALF && (trailing16 & 1)))
                ++trailing16;
            return sign16 | trailing16;
        }

        if (nonneg32 <= F16_MAX_NORMAL)
        {
                       biased16 = (uint16_t)((biased32 - F32_EXPONENT_BIAS + IEEEBinary16::EXPONENT_BIAS)
                << IEEEBinary16::NUM_TRAILING_BITS);
            trailing16 = (uint16_t)((trailing32 & INT_PART_MASK) >> DIFF_NUM_TRAILING_BITS);
            frcpart = (trailing32 & FRC_PART_MASK);
            if (frcpart > FRC_HALF || (frcpart == FRC_HALF && (trailing16 & 1)))
                ++trailing16;
            return sign16 | (biased16 + trailing16);
        }

        if (nonneg32 < F16_AVR_MAX_NORMAL_INFINITY)
            return sign16 | IEEEBinary16::MAX_NORMAL;

        return sign16 | IEEEBinary16::POS_INFINITY;
    }

    if (trailing32 == 0)
        return sign16 | IEEEBinary16::POS_INFINITY;

       uint16_t maskPayload = (uint16_t)((trailing32 & 0x007FE000u) >> 13);
    return sign16 | IEEEBinary16::EXPONENT_MASK | maskPayload;
}

static uint32_t ieee_convert_16_to_32(uint16_t inEncoding)
{
    uint16_t sign16     = (inEncoding & IEEEBinary16::SIGN_MASK);
    uint16_t biased16   = (uint16_t)((inEncoding & IEEEBinary16::EXPONENT_MASK) >> IEEEBinary16::NUM_TRAILING_BITS);
    uint16_t trailing16 = (inEncoding & IEEEBinary16::TRAILING_MASK);

    uint32_t sign32 = ((uint32_t)sign16 << DIFF_NUM_ENCODING_BITS);
    uint32_t biased32, trailing32;

    if (biased16 == 0)
    {
        if (trailing16 == 0)
            return sign32;

               trailing32 = (uint32_t)trailing16;
        int32_t leading = bit_leading_u32(trailing32);
        int32_t shift = 23 - leading;
        biased32 = (uint32_t)(F32_EXPONENT_BIAS - 1 - shift);
        trailing32 = (trailing32 << shift) & F32_TRAILING_MASK;
        return sign32 | (biased32 << F32_NUM_TRAILING_BITS) | trailing32;
    }

    if (biased16 < IEEEBinary16::MAX_BIASED_EXPONENT)
    {
               biased32 = (uint32_t)(biased16 - IEEEBinary16::EXPONENT_BIAS + F32_EXPONENT_BIAS);
        trailing32 = ((uint32_t)trailing16 << DIFF_NUM_TRAILING_BITS);
        return sign32 | (biased32 << F32_NUM_TRAILING_BITS) | trailing32;
    }

    if (trailing16 == 0)
        return sign32 | F32_BIASED_EXPONENT_MASK;

       uint32_t maskPayload = (((uint32_t)trailing16 & IEEEBinary16::TRAILING_MASK) << DIFF_PAYLOAD_SHIFT);
    return sign32 | F32_BIASED_EXPONENT_MASK | maskPayload;
}

IEEEBinary16::IEEEBinary16(float inNumber)
{
    union { float n; uint32_t e; } temp;
    temp.n = inNumber;
    encoding = ieee_convert_32_to_16(temp.e);
}

IEEEBinary16::IEEEBinary16(double inNumber)
{
    union { float n; uint32_t e; } temp;
    temp.n = (float)inNumber;
    encoding = ieee_convert_32_to_16(temp.e);
}

IEEEBinary16::operator float() const
{
    union { uint32_t e; float n; } temp;
    temp.e = ieee_convert_16_to_32(encoding);
    return temp.n;
}

IEEEBinary16::operator double() const
{
    union { uint32_t e; float n; } temp;
    temp.e = ieee_convert_16_to_32(encoding);
    return (double)temp.n;
}
