/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm_kernel_neon.cpp

Abstract:

    This module implements the fp32 layer normalization kernel for ARM64 NEON.

    Two passes over the row: a vectorized mean, then a vectorized centered
    variance (numerically stable), followed by a fused normalize/scale/bias
    pass. Matches the semantics of the generic implementation in
    onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc.

--*/

#include "mlasi.h"

#include <arm_neon.h>
#include <cmath>

namespace {

inline float
HorizontalSum(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
{
    return vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
}

}  // namespace

void
MLASCALL
MlasLayerNormKernelNeon(
    const float* Input,
    const float* Scale,
    const float* Bias,
    float* Output,
    float* MeanOut,
    float* InvStdDevOut,
    size_t NormSize,
    float Epsilon,
    bool Simplified
)
{
    const size_t n = NormSize;
    const float inv_n = 1.0f / static_cast<float>(n);

    // The mean is always computed and reported through MeanOut (the generic
    // path and the MLAS unit test do the same); RMSNorm just does not center.
    float mean = 0.0f;
    {
        float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            s0 = vaddq_f32(s0, vld1q_f32(Input + i));
            s1 = vaddq_f32(s1, vld1q_f32(Input + i + 4));
            s2 = vaddq_f32(s2, vld1q_f32(Input + i + 8));
            s3 = vaddq_f32(s3, vld1q_f32(Input + i + 12));
        }
        for (; i + 4 <= n; i += 4) {
            s0 = vaddq_f32(s0, vld1q_f32(Input + i));
        }
        float sum = HorizontalSum(s0, s1, s2, s3);
        for (; i < n; i++) {
            sum += Input[i];
        }
        mean = sum * inv_n;
    }

    // Centered sum of squares (for RMSNorm the "center" is zero).
    const float center = Simplified ? 0.0f : mean;
    const float32x4_t vmean = vdupq_n_f32(center);
    float32x4_t q0 = vdupq_n_f32(0.0f), q1 = q0, q2 = q0, q3 = q0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(Input + i), vmean);
        float32x4_t d1 = vsubq_f32(vld1q_f32(Input + i + 4), vmean);
        float32x4_t d2 = vsubq_f32(vld1q_f32(Input + i + 8), vmean);
        float32x4_t d3 = vsubq_f32(vld1q_f32(Input + i + 12), vmean);
        q0 = vfmaq_f32(q0, d0, d0);
        q1 = vfmaq_f32(q1, d1, d1);
        q2 = vfmaq_f32(q2, d2, d2);
        q3 = vfmaq_f32(q3, d3, d3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(Input + i), vmean);
        q0 = vfmaq_f32(q0, d0, d0);
    }
    float sumsq = HorizontalSum(q0, q1, q2, q3);
    for (; i < n; i++) {
        const float d = Input[i] - center;
        sumsq += d * d;
    }

    const float inv_std = 1.0f / sqrtf(sumsq * inv_n + Epsilon);
    const float32x4_t vinv = vdupq_n_f32(inv_std);
    // y = (x - mean) * inv_std * scale + bias == x * (inv_std * scale) + (bias - mean * inv_std * scale)
    // Computed as ((x - mean) * inv_std) * scale + bias to mirror the generic path's rounding order.
    i = 0;
    if (Bias != nullptr && !Simplified) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x = vmulq_f32(vsubq_f32(vld1q_f32(Input + i), vmean), vinv);
            vst1q_f32(Output + i, vfmaq_f32(vld1q_f32(Bias + i), x, vld1q_f32(Scale + i)));
        }
        for (; i < n; i++) {
            Output[i] = (Input[i] - center) * inv_std * Scale[i] + Bias[i];
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x = vmulq_f32(vsubq_f32(vld1q_f32(Input + i), vmean), vinv);
            vst1q_f32(Output + i, vmulq_f32(x, vld1q_f32(Scale + i)));
        }
        for (; i < n; i++) {
            Output[i] = (Input[i] - center) * inv_std * Scale[i];
        }
    }

    if (MeanOut != nullptr) {
        *MeanOut = mean;
    }
    if (InvStdDevOut != nullptr) {
        *InvStdDevOut = inv_std;
    }
}
