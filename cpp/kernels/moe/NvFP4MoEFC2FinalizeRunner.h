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

// C++ runner for the FC2 finalize kernel (grouped GEMM + scatter-reduce).
//
// As of plugin v5 this drives the **N-major** AOT FC2 kernel: the weight
// buffer ``[L, K, N/2]`` bytes (N innermost), symmetric to the FC1 N-major
// flip that v4 introduced. The kernel does an in-flight SMEM nibble transpose
// to feed the K-major B operand that ``tcgen05.mma`` requires. The AOT
// function symbols are unchanged from the v4 K-major variants, so the
// dispatch code is identical — only the caller's weight byte layout contract
// changes.
//
// Performs the FC2 output projection in the decomposed MoE pipeline:
// - FP4xFP4 blockscaled grouped GEMM (with in-flight SMEM nibble transpose)
// - Per-expert alpha scaling
// - Per-token router weight scaling
// - Atomic scatter-reduce to output buffer
//
// Like the contiguous FC1 runner, uses runtime lookup tables and needs
// only 2 AOT variants (n128, n256) × 2 output dtypes (bf16, fp16).

#pragma once

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
#include "cutedsl_nvfp4_moe_all.h"
#endif

#include "NvFP4MoEUtils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_moe
{

class NvFP4MoEFC2FinalizeRunner
{
public:
    /// @param numLocalExperts  Number of local experts (L)
    /// @param topK             Routing factor
    /// @param n                Hidden size (N — FC2 output dimension)
    /// @param k                Intermediate size (K — FC2 input dimension)
    /// @param outDtype         Output element type (selects the AOT variant).
    NvFP4MoEFC2FinalizeRunner(
        int32_t numLocalExperts, int32_t topK, int32_t n, int32_t k, OutputDType outDtype = OutputDType::kBF16);

    static bool loadKernelModules();
    static void unloadKernelModules();

    /// Run the FC2 finalize kernel (grouped GEMM + scatter-reduce).
    ///
    /// @param inputFP4          [permutedM, K/2] float4_e2m1fn_x2 on device
    /// @param weight            [L, K, N/2] float4_e2m1fn_x2 on device (3D stacked,
    ///                          **N-major** byte layout — N axis innermost, 2 FP4
    ///                          nibbles per byte along N = hidden_size). Matches the
    ///                          plugin v5 fc_down_qweights shape and the Marlin
    ///                          decode layout.
    /// @param inputSF           atom-layout SF buffer on device (input A scales)
    /// @param weightSF          atom-layout SF buffer on device (weight B scales,
    ///                          prefill-friendly M=N=H, K=I/16 — unchanged from v4)
    /// @param output            [numTokens, N] bfloat16 on device (pre-zeroed, scatter target)
    /// @param alpha             [L] float32 on device (per-expert scaling)
    /// @param layout            MoE layout (tile metadata + permutation indices)
    /// @param tokenFinalScales  [numTokens, topK] float32 on device (router weights)
    /// @param permutedM         Total permuted rows
    /// @param numTokens         Number of original tokens
    /// @param stream            CUDA stream
    void run(void const* inputFP4, void const* weight, void const* inputSF, void const* weightSF, void* output,
        void const* alpha, MoELayout const& layout, void const* tokenFinalScales, int64_t permutedM, int64_t numTokens,
        cudaStream_t stream);

private:
    int32_t mNumLocalExperts;
    int32_t mTopK;
    int32_t mN;
    int32_t mK;
    OutputDType mOutDtype;

    static int32_t selectTactic(int64_t n, int64_t k);

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    static nvfp4_moe_fc2_n128_bf16_Kernel_Module_t sFC2N128_bf16;
    static nvfp4_moe_fc2_n256_bf16_Kernel_Module_t sFC2N256_bf16;
    static nvfp4_moe_fc2_n128_fp16_Kernel_Module_t sFC2N128_fp16;
    static nvfp4_moe_fc2_n256_fp16_Kernel_Module_t sFC2N256_fp16;
#endif
    static bool sLoaded;
    static std::mutex sMutex;
};

} // namespace nvfp4_moe
} // namespace kernel
} // namespace trt_edgellm
