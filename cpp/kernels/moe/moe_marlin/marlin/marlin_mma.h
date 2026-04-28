/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Adapted from https://github.com/vllm-project/vllm/blob/v0.14.0/csrc/quantization/gptq_marlin/marlin_mma.h
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#pragma once
#include "common/cudaMacros.h"
#include "marlin_dtypes.cuh"

// CUDA 11.4 nvcc has a bug with `if constexpr` in device template code when
// the condition involves constexpr member functions (`*this`). IF_CONSTEXPR
// falls back to plain `if` on CUDA < 11.8; the compiler still optimizes away
// dead branches since conditions are compile-time constants (template params).
#if defined(__CUDACC__) && defined(CUDA_VERSION) && (CUDA_VERSION < 11080)
#define IF_CONSTEXPR if
#else
#define IF_CONSTEXPR if constexpr
#endif

namespace MARLIN_NAMESPACE_NAME
{

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <trt_edgellm::marlin_dtypes::ScalarTypeId type_id, int k_size = 16>
__device__ inline void mma(typename MarlinScalarType<type_id>::FragA const& a_frag,
    typename MarlinScalarType<type_id>::FragB const& frag_b, typename MarlinScalarType<type_id>::FragC& frag_c,
    int idx = 0)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);
    using scalar_t = typename MarlinScalarType<type_id>::scalar_t;

    static_assert(k_size == 16 || k_size == 32, "FP16/BF16 support MMA_k == 16, E4M3 supports MMA_k == 16 or 32.");

    IF_CONSTEXPR(k_size == 16)
    {
        IF_CONSTEXPR(std::is_same<scalar_t, half>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                "f"(c[3]));
        }
        else IF_CONSTEXPR(std::is_same<scalar_t, nv_bfloat16>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                "f"(c[3]));
        }
#if SUPPORTS_FP8
        else IF_CONSTEXPR(std::is_same<scalar_t, __nv_fp8_e4m3>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(a[idx * 2]), "r"(a[idx * 2 + 1]), "r"(b[idx]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
        }
#endif
    }
    else IF_CONSTEXPR(k_size == 32)
    {
#if SUPPORTS_FP8
        IF_CONSTEXPR(std::is_same<scalar_t, __nv_fp8_e4m3>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                "f"(c[3]));
        }
#endif
    }
}

template <trt_edgellm::marlin_dtypes::ScalarTypeId type_id, int k_size = 16>
__device__ inline void mma_trans(typename MarlinScalarType<type_id>::FragA const& a_frag,
    typename MarlinScalarType<type_id>::FragB const& frag_b, typename MarlinScalarType<type_id>::FragB const& frag_b2,
    typename MarlinScalarType<type_id>::FragC& frag_c)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);
    uint32_t const* b2 = reinterpret_cast<uint32_t const*>(&frag_b2);
    float* c = reinterpret_cast<float*>(&frag_c);
    using scalar_t = typename MarlinScalarType<type_id>::scalar_t;

    static_assert(k_size == 16 || k_size == 32, "FP16/BF16 support MMA_k == 16, E4M3 supports MMA_k == 16 or 32.");

    IF_CONSTEXPR(k_size == 16)
    {
        IF_CONSTEXPR(std::is_same<scalar_t, half>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                "f"(c[3]));
        }
        else IF_CONSTEXPR(std::is_same<scalar_t, nv_bfloat16>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                "f"(c[3]));
        }
#if SUPPORTS_FP8
        else IF_CONSTEXPR(std::is_same<scalar_t, __nv_fp8_e4m3>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
        }
#endif
    }
    else IF_CONSTEXPR(k_size == 32)
    {
#if SUPPORTS_FP8
        IF_CONSTEXPR(std::is_same<scalar_t, __nv_fp8_e4m3>::value)
        {
            float* c = reinterpret_cast<float*>(&frag_c);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
                "f"(c[3]));
        }
#endif
    }
}

} // namespace MARLIN_NAMESPACE_NAME