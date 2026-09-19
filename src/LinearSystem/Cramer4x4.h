/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Cramer4x4.h
 * @brief Cramer's rule implementation for 4x4 linear systems
 *
 * @details Solves A * x = b for a dense 4x4 system in row-major layout:
 * A[0..3] = row 0, A[4..7] = row 1, A[8..11] = row 2, A[12..15] = row 3.
 * Computes the adjugate matrix via 2x2 subdeterminant reuse and applies
 * architecture-specific SIMD for the matrix-vector product, with separate
 * paths for single and double precision.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Project headers
#include "Scalar.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) \
   || defined(__i386__)   || defined(_M_IX86)
    #include <immintrin.h>
#else
    #error "Target architecture requires ARM NEON or x86 SSE/AVX support."
#endif

// ********************************* Functions ********************************

/// Solve a dense 4x4 linear system A * x = b via Cramer's rule
inline void solve4x4
(
    const Scalar* __restrict__ A,
    const Scalar* __restrict__ b,
    Scalar* __restrict__ x
)
{
    // Compute 2x2 subdeterminants of rows 0-1 (s)
    const Scalar s0 = A[0] * A[5] - A[1] * A[4];
    const Scalar s1 = A[0] * A[6] - A[2] * A[4];
    const Scalar s2 = A[0] * A[7] - A[3] * A[4];
    const Scalar s3 = A[1] * A[6] - A[2] * A[5];
    const Scalar s4 = A[1] * A[7] - A[3] * A[5];
    const Scalar s5 = A[2] * A[7] - A[3] * A[6];

    // Compute 2x2 subdeterminants of rows 2-3 (c)
    const Scalar c0 = A[8] * A[13] - A[9] * A[12];
    const Scalar c1 = A[8] * A[14] - A[10] * A[12];
    const Scalar c2 = A[8] * A[15] - A[11] * A[12];
    const Scalar c3 = A[9] * A[14] - A[10] * A[13];
    const Scalar c4 = A[9] * A[15] - A[11] * A[13];
    const Scalar c5 = A[10] * A[15] - A[11] * A[14];

    // Full 4x4 determinant via Laplace expansion
    const Scalar det =
        s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    const Scalar inv_det = S(1.0) / det;

// ---- ARM NEON ----
#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#ifdef TURBLYZE_DOUBLE_PRECISION
    // ---------------------------------------------------------
    // AArch64 NEON — double precision (float64x2_t, 2-wide)
    // ---------------------------------------------------------

    // Load rows (2 registers per row)
    const float64x2_t r0_lo = vld1q_f64(A + 0);
    const float64x2_t r0_hi = vld1q_f64(A + 2);
    const float64x2_t r1_lo = vld1q_f64(A + 4);
    const float64x2_t r1_hi = vld1q_f64(A + 6);
    const float64x2_t r2_lo = vld1q_f64(A + 8);
    const float64x2_t r2_hi = vld1q_f64(A + 10);
    const float64x2_t r3_lo = vld1q_f64(A + 12);
    const float64x2_t r3_hi = vld1q_f64(A + 14);
    const float64x2_t b_lo  = vld1q_f64(b + 0);
    const float64x2_t b_hi  = vld1q_f64(b + 2);

    // Alternating sign patterns
    const float64x2_t sign_np = {-1.0, 1.0};
    const float64x2_t sign_pn = { 1.0,-1.0};

    // Apply signs: v0=r1*odd, v1=r0*even, v2=r3*odd, v3=r2*even
    const float64x2_t v0_lo = vmulq_f64(r1_lo, sign_np);
    const float64x2_t v0_hi = vmulq_f64(r1_hi, sign_np);
    const float64x2_t v1_lo = vmulq_f64(r0_lo, sign_pn);
    const float64x2_t v1_hi = vmulq_f64(r0_hi, sign_pn);
    const float64x2_t v2_lo = vmulq_f64(r3_lo, sign_np);
    const float64x2_t v2_hi = vmulq_f64(r3_hi, sign_np);
    const float64x2_t v3_lo = vmulq_f64(r2_lo, sign_pn);
    const float64x2_t v3_hi = vmulq_f64(r2_hi, sign_pn);

    // 4x4 transpose via 2-wide interleaves
    // Low halves (elements 0,1) of each row
    const float64x2_t C0_lo = vtrn1q_f64(v0_lo, v1_lo);
    const float64x2_t C1_lo = vtrn2q_f64(v0_lo, v1_lo);
    const float64x2_t C0_hi = vtrn1q_f64(v2_lo, v3_lo);
    const float64x2_t C1_hi = vtrn2q_f64(v2_lo, v3_lo);
    // High halves (elements 2,3) of each row
    const float64x2_t C2_lo = vtrn1q_f64(v0_hi, v1_hi);
    const float64x2_t C3_lo = vtrn2q_f64(v0_hi, v1_hi);
    const float64x2_t C2_hi = vtrn1q_f64(v2_hi, v3_hi);
    const float64x2_t C3_hi = vtrn2q_f64(v2_hi, v3_hi);

    // Packed coefficients: k_lo = {c, c}, k_hi = {s, s}
    const float64x2_t k0_lo = vdupq_n_f64(c0);
    const float64x2_t k0_hi = vdupq_n_f64(s0);
    const float64x2_t k1_lo = vdupq_n_f64(c1);
    const float64x2_t k1_hi = vdupq_n_f64(s1);
    const float64x2_t k2_lo = vdupq_n_f64(c2);
    const float64x2_t k2_hi = vdupq_n_f64(s2);
    const float64x2_t k3_lo = vdupq_n_f64(c3);
    const float64x2_t k3_hi = vdupq_n_f64(s3);
    const float64x2_t k4_lo = vdupq_n_f64(c4);
    const float64x2_t k4_hi = vdupq_n_f64(s4);
    const float64x2_t k5_lo = vdupq_n_f64(c5);
    const float64x2_t k5_hi = vdupq_n_f64(s5);

    // Rows of adj(A) via FMA (lo and hi halves separately)
    // R0 = Col1*k5 + Col2*k4 + Col3*k3
    float64x2_t R0_lo = vmulq_f64(C1_lo, k5_lo);
    R0_lo = vfmaq_f64(R0_lo, C2_lo, k4_lo);
    R0_lo = vfmaq_f64(R0_lo, C3_lo, k3_lo);
    float64x2_t R0_hi = vmulq_f64(C1_hi, k5_hi);
    R0_hi = vfmaq_f64(R0_hi, C2_hi, k4_hi);
    R0_hi = vfmaq_f64(R0_hi, C3_hi, k3_hi);

    // R1 = Col0*k5 - Col2*k2 - Col3*k1
    float64x2_t R1_lo = vmulq_f64(C0_lo, k5_lo);
    R1_lo = vfmsq_f64(R1_lo, C2_lo, k2_lo);
    R1_lo = vfmsq_f64(R1_lo, C3_lo, k1_lo);
    float64x2_t R1_hi = vmulq_f64(C0_hi, k5_hi);
    R1_hi = vfmsq_f64(R1_hi, C2_hi, k2_hi);
    R1_hi = vfmsq_f64(R1_hi, C3_hi, k1_hi);

    // R2 = Col3*k0 - Col0*k4 - Col1*k2
    float64x2_t R2_lo = vmulq_f64(C3_lo, k0_lo);
    R2_lo = vfmsq_f64(R2_lo, C0_lo, k4_lo);
    R2_lo = vfmsq_f64(R2_lo, C1_lo, k2_lo);
    float64x2_t R2_hi = vmulq_f64(C3_hi, k0_hi);
    R2_hi = vfmsq_f64(R2_hi, C0_hi, k4_hi);
    R2_hi = vfmsq_f64(R2_hi, C1_hi, k2_hi);

    // R3 = Col0*k3 + Col1*k1 + Col2*k0
    float64x2_t R3_lo = vmulq_f64(C0_lo, k3_lo);
    R3_lo = vfmaq_f64(R3_lo, C1_lo, k1_lo);
    R3_lo = vfmaq_f64(R3_lo, C2_lo, k0_lo);
    float64x2_t R3_hi = vmulq_f64(C0_hi, k3_hi);
    R3_hi = vfmaq_f64(R3_hi, C1_hi, k1_hi);
    R3_hi = vfmaq_f64(R3_hi, C2_hi, k0_hi);

    // Element-wise multiply each adj row by b
    const float64x2_t p0_lo = vmulq_f64(R0_lo, b_lo);
    const float64x2_t p0_hi = vmulq_f64(R0_hi, b_hi);
    const float64x2_t p1_lo = vmulq_f64(R1_lo, b_lo);
    const float64x2_t p1_hi = vmulq_f64(R1_hi, b_hi);
    const float64x2_t p2_lo = vmulq_f64(R2_lo, b_lo);
    const float64x2_t p2_hi = vmulq_f64(R2_hi, b_hi);
    const float64x2_t p3_lo = vmulq_f64(R3_lo, b_lo);
    const float64x2_t p3_hi = vmulq_f64(R3_hi, b_hi);

    // Horizontal reduction: sum each 4-element dot product
    const float64x2_t h0 = vaddq_f64(p0_lo, p0_hi);
    const float64x2_t h1 = vaddq_f64(p1_lo, p1_hi);
    const float64x2_t h2 = vaddq_f64(p2_lo, p2_hi);
    const float64x2_t h3 = vaddq_f64(p3_lo, p3_hi);
    const float64x2_t dots_lo = vpaddq_f64(h0, h1);
    const float64x2_t dots_hi = vpaddq_f64(h2, h3);

    // Scale by 1/det and store
    const float64x2_t x_lo = vmulq_n_f64(dots_lo, inv_det);
    const float64x2_t x_hi = vmulq_n_f64(dots_hi, inv_det);
    vst1q_f64(x + 0, x_lo);
    vst1q_f64(x + 2, x_hi);

#else // single precision
    // ---------------------------------------------------------
    // AArch64 NEON — single precision (float32x4_t, 4-wide)
    // ---------------------------------------------------------
    const float32x4_t r0 = vld1q_f32(A + 0);
    const float32x4_t r1 = vld1q_f32(A + 4);
    const float32x4_t r2 = vld1q_f32(A + 8);
    const float32x4_t r3 = vld1q_f32(A + 12);
    const float32x4_t b_vec = vld1q_f32(b);

    // Apply alternating sign pattern to rows
    const float32x4_t sign_odd  = {-1.0f,  1.0f, -1.0f,  1.0f};
    const float32x4_t sign_even = { 1.0f, -1.0f,  1.0f, -1.0f};

    const float32x4_t v0 = vmulq_f32(r1, sign_odd);
    const float32x4_t v1 = vmulq_f32(r0, sign_even);
    const float32x4_t v2 = vmulq_f32(r3, sign_odd);
    const float32x4_t v3 = vmulq_f32(r2, sign_even);

    // 4x4 transpose to extract signed column vectors
    const float32x4x2_t t0 = vtrnq_f32(v0, v1);
    const float32x4x2_t t1 = vtrnq_f32(v2, v3);
    const float32x4_t Col0 =
        vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
    const float32x4_t Col1 =
        vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
    const float32x4_t Col2 =
        vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
    const float32x4_t Col3 =
        vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));

    // Packed coefficients [c, c, s, s]
    auto make_k = [](float c, float s) -> float32x4_t
    {
        return vcombine_f32(vdup_n_f32(c), vdup_n_f32(s));
    };
    const float32x4_t k0 = make_k(c0, s0);
    const float32x4_t k1 = make_k(c1, s1);
    const float32x4_t k2 = make_k(c2, s2);
    const float32x4_t k3 = make_k(c3, s3);
    const float32x4_t k4 = make_k(c4, s4);
    const float32x4_t k5 = make_k(c5, s5);

    // Rows of adj(A) via FMA
    float32x4_t R0 = vmulq_f32(Col1, k5);
    R0 = vfmaq_f32(R0, Col2, k4);
    R0 = vfmaq_f32(R0, Col3, k3);

    float32x4_t R1 = vmulq_f32(Col0, k5);
    R1 = vfmsq_f32(R1, Col2, k2);
    R1 = vfmsq_f32(R1, Col3, k1);

    float32x4_t R2 = vmulq_f32(Col3, k0);
    R2 = vfmsq_f32(R2, Col0, k4);
    R2 = vfmsq_f32(R2, Col1, k2);

    float32x4_t R3 = vmulq_f32(Col0, k3);
    R3 = vfmaq_f32(R3, Col1, k1);
    R3 = vfmaq_f32(R3, Col2, k0);

    // Multiply each adj row by b
    const float32x4_t p0 = vmulq_f32(R0, b_vec);
    const float32x4_t p1 = vmulq_f32(R1, b_vec);
    const float32x4_t p2 = vmulq_f32(R2, b_vec);
    const float32x4_t p3 = vmulq_f32(R3, b_vec);

    // Pairwise reduction (horizontal addition)
    const float32x4_t sum01 = vpaddq_f32(p0, p1);
    const float32x4_t sum23 = vpaddq_f32(p2, p3);
    const float32x4_t dots  = vpaddq_f32(sum01, sum23);

    // Scale by 1/det and store
    const float32x4_t x_res = vmulq_n_f32(dots, inv_det);
    vst1q_f32(x, x_res);
#endif // precision

// ---- x86 SSE / AVX ----
#elif defined(__x86_64__) || defined(_M_X64) \
   || defined(__i386__)   || defined(_M_IX86)

#ifdef TURBLYZE_DOUBLE_PRECISION

#if !defined(__AVX__)
    #error "Double-precision Cramer4x4 on x86 requires AVX (-mavx)."
#endif
    // ---------------------------------------------------------
    // x86 AVX — double precision (__m256d, 4-wide)
    // ---------------------------------------------------------
    __m256d r0 = _mm256_loadu_pd(A + 0);
    __m256d r1 = _mm256_loadu_pd(A + 4);
    __m256d r2 = _mm256_loadu_pd(A + 8);
    __m256d r3 = _mm256_loadu_pd(A + 12);
    const __m256d b_vec = _mm256_loadu_pd(b);

    const __m256d sign_odd  =
        _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);
    const __m256d sign_even =
        _mm256_setr_pd( 1.0,-1.0,  1.0,-1.0);

    __m256d v0 = _mm256_mul_pd(r1, sign_odd);
    __m256d v1 = _mm256_mul_pd(r0, sign_even);
    __m256d v2 = _mm256_mul_pd(r3, sign_odd);
    __m256d v3 = _mm256_mul_pd(r2, sign_even);

    // 4x4 transpose via unpack + 128-bit lane permute
    const __m256d u0 = _mm256_unpacklo_pd(v0, v1);
    const __m256d u1 = _mm256_unpackhi_pd(v0, v1);
    const __m256d u2 = _mm256_unpacklo_pd(v2, v3);
    const __m256d u3 = _mm256_unpackhi_pd(v2, v3);
    const __m256d Col0 = _mm256_permute2f128_pd(u0, u2, 0x20);
    const __m256d Col1 = _mm256_permute2f128_pd(u1, u3, 0x20);
    const __m256d Col2 = _mm256_permute2f128_pd(u0, u2, 0x31);
    const __m256d Col3 = _mm256_permute2f128_pd(u1, u3, 0x31);

    // _mm256_set_pd is (d3,d2,d1,d0) → [d0,d1,d2,d3]
    #define MAKE_K(c, s) _mm256_set_pd(s, s, c, c)
    const __m256d k0 = MAKE_K(c0, s0);
    const __m256d k1 = MAKE_K(c1, s1);
    const __m256d k2 = MAKE_K(c2, s2);
    const __m256d k3 = MAKE_K(c3, s3);
    const __m256d k4 = MAKE_K(c4, s4);
    const __m256d k5 = MAKE_K(c5, s5);
    #undef MAKE_K

#if defined(__FMA__)
    __m256d R0 = 
        _mm256_fmadd_pd
        (
            Col3,
            k3,
            _mm256_fmadd_pd(Col2, k4, _mm256_mul_pd(Col1, k5))
        );
    __m256d R1 =
        _mm256_fnmadd_pd
        (
            Col3,
            k1,
            _mm256_fnmadd_pd(Col2, k2, _mm256_mul_pd(Col0, k5))
    );
    __m256d R2 =
        _mm256_fnmadd_pd
        (
            Col1,
            k2,
            _mm256_fnmadd_pd(Col0, k4, _mm256_mul_pd(Col3, k0))
        );
    __m256d R3 =
        _mm256_fmadd_pd
        (
            Col2,
            k0,
            _mm256_fmadd_pd(Col1, k1, _mm256_mul_pd(Col0, k3))
        );
#else
    __m256d R0 =
        _mm256_add_pd
        (
            _mm256_mul_pd(Col1, k5),
            _mm256_add_pd(_mm256_mul_pd(Col2, k4),_mm256_mul_pd(Col3, k3))
        );
    __m256d R1 =
        _mm256_sub_pd
        (
            _mm256_mul_pd(Col0, k5),
            _mm256_add_pd(_mm256_mul_pd(Col2, k2), _mm256_mul_pd(Col3, k1))
        );
    __m256d R2 =
        _mm256_sub_pd
        (
            _mm256_mul_pd(Col3, k0),
            _mm256_add_pd(_mm256_mul_pd(Col0, k4), _mm256_mul_pd(Col1, k2))
        );
    __m256d R3 =
        _mm256_add_pd
        (
            _mm256_mul_pd(Col0, k3),
            _mm256_add_pd(_mm256_mul_pd(Col1, k1), _mm256_mul_pd(Col2, k0))
        );
#endif // __FMA__

    const __m256d p0 = _mm256_mul_pd(R0, b_vec);
    const __m256d p1 = _mm256_mul_pd(R1, b_vec);
    const __m256d p2 = _mm256_mul_pd(R2, b_vec);
    const __m256d p3 = _mm256_mul_pd(R3, b_vec);

    // Horizontal sum: hadd gives partial sums in 128-bit lanes
    const __m256d h01 = _mm256_hadd_pd(p0, p1);
    const __m256d h23 = _mm256_hadd_pd(p2, p3);

    // Add low and high 128-bit lanes to complete the reduction
    const __m128d lo01 = _mm256_castpd256_pd128(h01);
    const __m128d hi01 = _mm256_extractf128_pd(h01, 1);
    const __m128d lo23 = _mm256_castpd256_pd128(h23);
    const __m128d hi23 = _mm256_extractf128_pd(h23, 1);
    const __m128d dots01 = _mm_add_pd(lo01, hi01);
    const __m128d dots23 = _mm_add_pd(lo23, hi23);

    // Reassemble, scale by 1/det, and store
    const __m256d dots = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(dots01), dots23, 1
    );
    const __m256d x_res =
        _mm256_mul_pd(dots, _mm256_set1_pd(inv_det));
    _mm256_storeu_pd(x, x_res);

#else // single precision

#if !defined(__SSE3__)
    #error "Single-precision Cramer4x4 on x86 requires SSE3."
#endif
    // ---------------------------------------------------------
    // x86 SSE — single precision (__m128, 4-wide)
    // ---------------------------------------------------------
    __m128 r0 = _mm_loadu_ps(A + 0);
    __m128 r1 = _mm_loadu_ps(A + 4);
    __m128 r2 = _mm_loadu_ps(A + 8);
    __m128 r3 = _mm_loadu_ps(A + 12);
    const __m128 b_vec = _mm_loadu_ps(b);

    const __m128 sign_odd  =
        _mm_setr_ps(-1.0f,  1.0f, -1.0f,  1.0f);
    const __m128 sign_even =
        _mm_setr_ps( 1.0f, -1.0f,  1.0f, -1.0f);

    __m128 v0 = _mm_mul_ps(r1, sign_odd);
    __m128 v1 = _mm_mul_ps(r0, sign_even);
    __m128 v2 = _mm_mul_ps(r3, sign_odd);
    __m128 v3 = _mm_mul_ps(r2, sign_even);

    _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
    const __m128 Col0 = v0;
    const __m128 Col1 = v1;
    const __m128 Col2 = v2;
    const __m128 Col3 = v3;

    #define MAKE_K(c, s) _mm_set_ps(s, s, c, c)
    const __m128 k0 = MAKE_K(c0, s0);
    const __m128 k1 = MAKE_K(c1, s1);
    const __m128 k2 = MAKE_K(c2, s2);
    const __m128 k3 = MAKE_K(c3, s3);
    const __m128 k4 = MAKE_K(c4, s4);
    const __m128 k5 = MAKE_K(c5, s5);
    #undef MAKE_K

#if defined(__FMA__)
    __m128 R0 =
        _mm_fmadd_ps
        (
            Col3,
            k3,
            _mm_fmadd_ps(Col2, k4, _mm_mul_ps(Col1, k5))
        );
    __m128 R1 =
        _mm_fnmadd_ps
        (
            Col3,
            k1,
            _mm_fnmadd_ps(Col2, k2, _mm_mul_ps(Col0, k5))
        );
    __m128 R2 =
        _mm_fnmadd_ps
        (
            Col1,
            k2,
            _mm_fnmadd_ps(Col0, k4, _mm_mul_ps(Col3, k0))
        );
    __m128 R3 =
        _mm_fmadd_ps
        (
            Col2,
            k0,
            _mm_fmadd_ps(Col1, k1, _mm_mul_ps(Col0, k3))
        );
#else
    __m128 R0 =
        _mm_add_ps
        (
            _mm_mul_ps(Col1, k5),
            _mm_add_ps(_mm_mul_ps(Col2, k4), _mm_mul_ps(Col3, k3))
        );
    __m128 R1 =
        _mm_sub_ps
        (
            _mm_mul_ps(Col0, k5),
            _mm_add_ps(_mm_mul_ps(Col2, k2), _mm_mul_ps(Col3, k1))
        );

    __m128 R2 =
        _mm_sub_ps
        (
            _mm_mul_ps(Col3, k0),
            _mm_add_ps(_mm_mul_ps(Col0, k4), _mm_mul_ps(Col1, k2))
        );
    __m128 R3 =
        _mm_add_ps
        (
            _mm_mul_ps(Col0, k3),
            _mm_add_ps(_mm_mul_ps(Col1, k1), _mm_mul_ps(Col2, k0))
        );
#endif // __FMA__

    const __m128 p0 = _mm_mul_ps(R0, b_vec);
    const __m128 p1 = _mm_mul_ps(R1, b_vec);
    const __m128 p2 = _mm_mul_ps(R2, b_vec);
    const __m128 p3 = _mm_mul_ps(R3, b_vec);

    const __m128 sum01 = _mm_hadd_ps(p0, p1);
    const __m128 sum23 = _mm_hadd_ps(p2, p3);
    const __m128 dots  = _mm_hadd_ps(sum01, sum23);

    const __m128 x_res =
        _mm_mul_ps(dots, _mm_set1_ps(inv_det));
    _mm_storeu_ps(x, x_res);

#endif // TURBLYZE_DOUBLE_PRECISION
#endif // architecture
}
