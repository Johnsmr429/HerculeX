#ifndef SF_AVX2_NEON_H
#define SF_AVX2_NEON_H

#include <arm_neon.h>
#include <stdint.h>

typedef struct { float64x2_t lo, hi; } __m256d;
typedef struct { uint64x2_t lo, hi; } __m256i;
typedef float32x4_t __m128;

#define _CMP_LT_OQ 0x11
#define _CMP_GT_OQ 0x1e

static inline __m256d sf_m256d(float64x2_t lo, float64x2_t hi) { __m256d r; r.lo = lo; r.hi = hi; return r; }
static inline __m256d sf_bits(uint64x2_t lo, uint64x2_t hi) { return sf_m256d(vreinterpretq_f64_u64(lo), vreinterpretq_f64_u64(hi)); }

static inline __m256d _mm256_setzero_pd(void) { return sf_m256d(vdupq_n_f64(0.0), vdupq_n_f64(0.0)); }
static inline __m256d _mm256_set1_pd(double a) { return sf_m256d(vdupq_n_f64(a), vdupq_n_f64(a)); }
static inline __m256d _mm256_loadu_pd(const double *p) { return sf_m256d(vld1q_f64(p), vld1q_f64(p + 2)); }
static inline void _mm256_storeu_pd(double *p, __m256d a) { vst1q_f64(p, a.lo); vst1q_f64(p + 2, a.hi); }

static inline __m256d _mm256_add_pd(__m256d a, __m256d b) { return sf_m256d(vaddq_f64(a.lo, b.lo), vaddq_f64(a.hi, b.hi)); }
static inline __m256d _mm256_sub_pd(__m256d a, __m256d b) { return sf_m256d(vsubq_f64(a.lo, b.lo), vsubq_f64(a.hi, b.hi)); }
static inline __m256d _mm256_mul_pd(__m256d a, __m256d b) { return sf_m256d(vmulq_f64(a.lo, b.lo), vmulq_f64(a.hi, b.hi)); }
static inline __m256d _mm256_div_pd(__m256d a, __m256d b) { return sf_m256d(vdivq_f64(a.lo, b.lo), vdivq_f64(a.hi, b.hi)); }
static inline __m256d _mm256_sqrt_pd(__m256d a) { return sf_m256d(vsqrtq_f64(a.lo), vsqrtq_f64(a.hi)); }
static inline __m256d _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c) { return sf_m256d(vfmaq_f64(c.lo, a.lo, b.lo), vfmaq_f64(c.hi, a.hi, b.hi)); }

static inline __m256d sf_cmp_pd(__m256d a, __m256d b, int op)
{
    if (op == _CMP_LT_OQ) return sf_bits(vcltq_f64(a.lo, b.lo), vcltq_f64(a.hi, b.hi));
    return sf_bits(vcgtq_f64(a.lo, b.lo), vcgtq_f64(a.hi, b.hi));
}
#define _mm256_cmp_pd(a, b, op) sf_cmp_pd((a), (b), (op))

static inline __m256d _mm256_and_pd(__m256d a, __m256d b)
{
    return sf_bits(vandq_u64(vreinterpretq_u64_f64(a.lo), vreinterpretq_u64_f64(b.lo)),
                   vandq_u64(vreinterpretq_u64_f64(a.hi), vreinterpretq_u64_f64(b.hi)));
}

static inline int _mm256_movemask_pd(__m256d a)
{
    uint64x2_t lo = vshrq_n_u64(vreinterpretq_u64_f64(a.lo), 63);
    uint64x2_t hi = vshrq_n_u64(vreinterpretq_u64_f64(a.hi), 63);
    return (int)(vgetq_lane_u64(lo, 0) | (vgetq_lane_u64(lo, 1) << 1) | (vgetq_lane_u64(hi, 0) << 2) | (vgetq_lane_u64(hi, 1) << 3));
}

static inline int _mm_popcnt_u32(unsigned int a) { return __builtin_popcount(a); }

static inline __m256i _mm256_loadu_si256(const __m256i *p)
{
    const uint64_t *q = (const uint64_t *)p;
    __m256i r; r.lo = vld1q_u64(q); r.hi = vld1q_u64(q + 2); return r;
}
static inline __m256d _mm256_castsi256_pd(__m256i a) { return sf_bits(a.lo, a.hi); }

static inline __m128 _mm_loadu_ps(const float *p) { return vld1q_f32(p); }
static inline __m256d _mm256_cvtps_pd(__m128 a) { return sf_m256d(vcvt_f64_f32(vget_low_f32(a)), vcvt_high_f64_f32(a)); }

#endif
