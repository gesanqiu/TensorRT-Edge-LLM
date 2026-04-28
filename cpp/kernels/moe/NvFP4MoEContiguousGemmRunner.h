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

// C++ runner for the contiguous grouped GEMM kernel (FC1 in decomposed pipeline).
//
// As of plugin v4 this drives the **N-major** AOT FC1 kernel: the weight buffer
// ``[L, K, N/2]`` bytes (N innermost), which matches the Marlin decode byte
// layout so a single on-device copy serves both decode and prefill. The kernel
// does an in-flight SMEM nibble transpose to feed the K-major B operand that
// ``tcgen05.mma`` requires. The AOT function symbols are unchanged from the
// v3 K-major variants, so the dispatch code is identical — only the caller's
// weight byte layout contract changes.
//
// Replaces the bucketed NvFP4MoEGroupedGemmRunner with a simpler runner that:
// - Takes contiguous gathered input + 3D stacked weights
// - Uses runtime lookup tables (no compile-time group_count)
// - Needs only 2 AOT variants per activation (n128, n256) vs 16 bucketed
// - Produces bit-identical output to the K-major grouped GEMM kernel on
//   the same logical weight values

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

class NvFP4MoEContiguousGemmRunner
{
public:
    /// @param numLocalExperts  Number of local experts (L)
    /// @param topK             Routing factor
    /// @param n                Intermediate size (N)
    /// @param k                Hidden size (K)
    /// @param tileSize         Tile size (128)
    /// @param activation       Activation function (compiled into the AOT binary).
    ///                         Only Relu2 and Swiglu are compiled; Identity is not
    ///                         exported as it has no production use for FC1.
    /// @param outDtype         Output element type (selects the AOT variant).
    NvFP4MoEContiguousGemmRunner(int32_t numLocalExperts, int32_t topK, int32_t n, int32_t k, int32_t tileSize = 128,
        Activation activation = Activation::kRelu2, OutputDType outDtype = OutputDType::kBF16);

    static bool loadKernelModules();
    static void unloadKernelModules();

    /// Run the contiguous grouped GEMM with fused alpha + activation.
    ///
    /// Unlike the bucketed runner, this takes the layout directly — no
    /// per-group metadata construction needed.  Alpha scaling and activation
    /// are applied inside the kernel epilogue in float32.
    ///
    /// @param gatheredFP4    [permutedM, K/2] float4_e2m1fn_x2 on device
    /// @param weight         [L, K, N/2] float4_e2m1fn_x2 on device (3D stacked,
    ///                       **N-major** byte layout — N axis innermost, 2 FP4
    ///                       nibbles per byte along N). Matches the plugin v4
    ///                       fc_up_qweights shape and the Marlin decode layout.
    /// @param gatheredSF     atom-layout SF buffer on device (input A scales)
    /// @param weightSF       atom-layout SF buffer on device (weight B scales,
    ///                       prefill-friendly M=N, K=K/16 — unchanged from v3)
    /// @param output         [permutedM, N_out] bfloat16 on device (output)
    /// @param alpha          [L] float32 per-expert scaling on device
    /// @param layout         MoE layout (tile metadata + permutation indices)
    /// @param permutedM      Total permuted rows
    /// @param stream         CUDA stream
    void run(void const* gatheredFP4, void const* weight, void const* gatheredSF, void const* weightSF, void* output,
        void const* alpha, MoELayout const& layout, int64_t permutedM, cudaStream_t stream);

private:
    int32_t mNumLocalExperts;
    int32_t mTopK;
    int32_t mN;
    int32_t mK;
    int32_t mTileSize;
    Activation mActivation;
    OutputDType mOutDtype;

    static int32_t selectTactic(int64_t n, int64_t k);

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    static nvfp4_moe_fc1_relu2_n128_bf16_Kernel_Module_t sRelu2N128_bf16;
    static nvfp4_moe_fc1_relu2_n256_bf16_Kernel_Module_t sRelu2N256_bf16;
    static nvfp4_moe_fc1_swiglu_n128_bf16_Kernel_Module_t sSwigluN128_bf16;
    static nvfp4_moe_fc1_swiglu_n256_bf16_Kernel_Module_t sSwigluN256_bf16;
    static nvfp4_moe_fc1_relu2_n128_fp16_Kernel_Module_t sRelu2N128_fp16;
    static nvfp4_moe_fc1_relu2_n256_fp16_Kernel_Module_t sRelu2N256_fp16;
    static nvfp4_moe_fc1_swiglu_n128_fp16_Kernel_Module_t sSwigluN128_fp16;
    static nvfp4_moe_fc1_swiglu_n256_fp16_Kernel_Module_t sSwigluN256_fp16;
#endif
    static bool sLoaded;
    static std::mutex sMutex;
};

} // namespace nvfp4_moe
} // namespace kernel
} // namespace trt_edgellm
