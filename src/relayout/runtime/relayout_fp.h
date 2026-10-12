/* Copyright 2026 ETH Zurich and the relayout authors.
 *
 * Element conversions for relayout's conversion kernels, one source for the host (C, OpenMP) and the
 * device (CUDA / HIP through NVRTC / hipRTC). Every narrow float format is stored as its bits and
 * narrowed from a double in one rounding (round to nearest, ties to even); the caller rounds a double
 * to float first where the reference does (ml_dtypes' bfloat16 and fp8, not numpy's float16). Widening
 * is exact. Formats: IEEE-like (an exponent field of
 * all ones is Inf / NaN: float16, bfloat16, float8_e5m2) and finite-only (all ones in exponent AND
 * mantissa is the only NaN, overflow is NaN: float8_e4m3fn), as numpy and ml_dtypes define them.
 */
#ifndef RELAYOUT_FP_H
#define RELAYOUT_FP_H

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__)
#define RELAYOUT_HD __host__ __device__
#define RELAYOUT_MEMCPY __builtin_memcpy
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef int int32_t;
#else
#include <stdint.h>
#include <string.h>
#define RELAYOUT_HD
#define RELAYOUT_MEMCPY memcpy
#endif

typedef struct { float re, im; } relayout_c64;
typedef struct { double re, im; } relayout_c128;

static inline RELAYOUT_HD relayout_c64 relayout_make_c64(float re, float im) {
    relayout_c64 z;
    z.re = re;
    z.im = im;
    return z;
}

static inline RELAYOUT_HD relayout_c128 relayout_make_c128(double re, double im) {
    relayout_c128 z;
    z.re = re;
    z.im = im;
    return z;
}

/* 2**k as a double, for k in the normal range. */
static inline RELAYOUT_HD double relayout_pow2(int k) {
    uint64_t u = (uint64_t)(k + 1023) << 52;
    double d;
    RELAYOUT_MEMCPY(&d, &u, 8);
    return d;
}

/* The bits of x in the format with ebits exponent bits, mbits mantissa bits, finite-only or not. */
static inline RELAYOUT_HD uint32_t relayout_narrow(double x, int ebits, int mbits, int finite) {
    uint64_t u;
    RELAYOUT_MEMCPY(&u, &x, 8);
    uint32_t sign = (uint32_t)(u >> 63) << (ebits + mbits);
    uint64_t a = u & 0x7fffffffffffffffull;
    int bias = (1 << (ebits - 1)) - 1;
    uint32_t ones = (uint32_t)((1 << ebits) - 1) << mbits;
    uint32_t nan = finite ? ones | ((1u << mbits) - 1u) : ones | (1u << (mbits - 1));
    uint32_t inf = finite ? nan : ones;
    uint32_t largest = finite ? nan - 1u : ones - 1u;
    if (a >= 0x7ff0000000000000ull) return sign | (a > 0x7ff0000000000000ull ? nan : inf);
    if ((a >> 52) == 0) return sign; /* a double subnormal is far below every format's least step */
    int e = (int)(a >> 52) - 1023;
    uint64_t m = (a & 0xfffffffffffffull) | (1ull << 52);
    int shift = 52 - mbits + (e < 1 - bias ? 1 - bias - e : 0);
    if (shift > 54) return sign; /* below half the least subnormal */
    uint64_t r = m >> shift, rest = m & ((1ull << shift) - 1ull), half = 1ull << (shift - 1);
    if (rest > half || (rest == half && (r & 1ull))) r += 1ull;
    uint64_t bits = e < 1 - bias ? r : ((uint64_t)(e + bias - 1) << mbits) + r;
    return sign | (bits > largest ? inf : (uint32_t)bits);
}

/* The double the bits b of that format hold, exactly. */
static inline RELAYOUT_HD double relayout_widen(uint32_t b, int ebits, int mbits, int finite) {
    int bias = (1 << (ebits - 1)) - 1;
    uint32_t e = (b >> mbits) & ((1u << ebits) - 1u), m = b & ((1u << mbits) - 1u);
    double sign = (b >> (ebits + mbits)) & 1u ? -1.0 : 1.0;
    if (e == (1u << ebits) - 1u && (!finite || m == (1u << mbits) - 1u)) {
        uint64_t special = m || finite ? 0x7ff8000000000000ull : 0x7ff0000000000000ull;
        double d;
        RELAYOUT_MEMCPY(&d, &special, 8);
        return sign * d;
    }
    if (e == 0) return sign * (double)m * relayout_pow2(1 - bias - mbits);
    return sign * (double)((1u << mbits) | m) * relayout_pow2((int)e - bias - mbits);
}

#endif
