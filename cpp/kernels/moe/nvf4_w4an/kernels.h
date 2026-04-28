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

#pragma once

#include <cstdint>

#include "common/cudaMacros.h"
#include "kernels/moe/moe_marlin/marlin/nvfp4_tensor.cuh"

namespace trt_edgellm
{

//! Default CUDA block size for W4A4 decode: one block size must divide both \p hidden_dim (up strips) and \p
//! inter_dim (down strips); see \ref nemotronMoeW4A4DecodeThreadBlockSizeForDims. \p hidden_dim must be divisible
//! by 64.
inline constexpr int kDefaultMlpW4a4DecodeThreadBlockSize = 128;

//! Upper bound on warps per block for MoE decode GEMV shared scratch (\c accumulateNvfp4GemvTileWarpReduce in
//! \c marlin_template.cuh).
//! Must be at least \c 256 / 32 for current largest \p thread_block_size.
inline constexpr int kMaxDecodingKernelWarpCount = 16;

//! Intermediate activation after the up-proj dot-product (before post-nonlinearity scaling in decode kernels).
enum class MoEActivationKind : int8_t
{
    kReLU2 = 0, //!< ``max(z,0)^2``
    kSiLU = 1,  //!< ``z * sigmoid(z)``
};

//! Supported \p thread_block_size values for W4A4 decode GEMV kernels (multiple of warp size).
inline bool moeW4a4DecodeIsValidThreadBlockSize(int const thread_block_size) noexcept
{
    return thread_block_size == 64 || thread_block_size == 96 || thread_block_size == 128 || thread_block_size == 192
        || thread_block_size == 256;
}

//! Picks W4A16 decode block size from \p inter_dim: largest of 256, 128, 96, 64 that divides \p inter_dim.
//! \return 0 if \p inter_dim is not divisible by 64.
inline int nemotronMoeW4A16DecodeThreadBlockSizeForInterDim(int const inter_dim) noexcept
{
    if (inter_dim % 256 == 0)
    {
        return 256;
    }
    if (inter_dim % 128 == 0)
    {
        return 128;
    }
    if (inter_dim % 96 == 0)
    {
        return 96;
    }
    if (inter_dim % 64 == 0)
    {
        return 64;
    }
    return 0;
}

//! Picks a \p thread_block_size valid for \ref moeW4a4DecodeIsValidThreadBlockSize that divides both \p hidden_dim
//! and \p inter_dim. W4A4 decode uses one block size for up (strips along \p hidden_dim) and down (strips along
//! \p inter_dim). \return 0 if no candidate divides both dimensions.
inline int nemotronMoeW4A4DecodeThreadBlockSizeForDims(int const hidden_dim, int const inter_dim) noexcept
{
    int const candidates[] = {256, 192, 128, 96, 64};
    for (int const tb : candidates)
    {
        if (moeW4a4DecodeIsValidThreadBlockSize(tb) && hidden_dim % tb == 0 && inter_dim % tb == 0)
        {
            return tb;
        }
    }
    return 0;
}

//! Grid x-dimension for MoE decode GEMV with explicit top-k routing: one block per (token row, top-k slot,
//! \p thread_block_size-sized strip of \p inter_dim). \p inter_dim must be a positive multiple of \p thread_block_size.
//! Token rows are \c batch * seq_len (flattened row-major \c [batch, seq_len, ...]).
inline int moeDecodeGemvTopkGridDim(
    int const batch_size, int const top_k, int const inter_dim, int const thread_block_size) noexcept
{
    return batch_size * top_k * (inter_dim / thread_block_size);
}

//! Same as \ref moeDecodeGemvTopkGridDim with \p num_tokens = \p batch * \p seq_len. Pass \p strip_dim = \p hidden_dim
//! for W4A16 up (strips along hidden) or \p inter_dim for W4A16 down (strips along intermediate).
inline int moeDecodeGemvTopkGridDimBatchSeq(
    int const batch, int const seq_len, int const top_k, int const strip_dim, int const thread_block_size) noexcept
{
    return moeDecodeGemvTopkGridDim(batch * seq_len, top_k, strip_dim, thread_block_size);
}

//! Grid x-dimension for W4A4 decode \b up kernel: strips along \p hidden_dim (same as
//! \ref launchNemotronMoeW4A4DecodeUpGemvCuda). \p num_tokens = \p batch * \p seq_len (flattened row-major tokens).
inline int moeW4a4DecodeUpGridDim(
    int const batch, int const seq_len, int const top_k, int const hidden_dim, int const thread_block_size) noexcept
{
    return moeDecodeGemvTopkGridDimBatchSeq(batch, seq_len, top_k, hidden_dim, thread_block_size);
}

//! Per-intermediate-index MACs for top-k MoE decode GEMV: \c batch_size * top_k * inter_dim.
inline int moeDecodeGemvTopkThreads(int const batch_size, int const top_k, int const inter_dim) noexcept
{
    return batch_size * top_k * inter_dim;
}

#if SUPPORTS_FP4

//! Host launch wrapper (TensorRT plugins, experiments). Runs split \ref launchNemotronMoeW4A4DecodeUpGemvCuda then
//! \ref launchNemotronMoeW4A4DecodeDownGemvCuda. \p inter_fp16_scratch must hold
//! \ref nemotronMoeW4A16UpFp16ScratchBytes(\p batch * \p seq_len, top_k, inter_dim) bytes.
//! Up writes FP16 dots with \c activation.global_scale[0] applied to the activation; down loads FP16 \c z, converts to
//! FP32 for \ref moeActivation (then applies router and \c down.global_scale), then accumulates into FP16 \p output
//! (same path as W4A16 down).
//! \p thread_block_size applies to both kernels (inter strips). \p inter_dim must be divisible by it; \p hidden_dim
//! by 64.
//! \p num_chunks is legacy (ignored).
void launchNemotronMoeW4a4DecodeGemvCuda(int batch, int seq_len, int hidden_dim, int inter_dim, int num_chunks,
    int num_experts, int top_k, int32_t const* expert_ids, float const* topk_weights, NVFP4Tensor const activation,
    NVFP4Tensor const up, NVFP4Tensor const down, uint8_t const* up_decode_sf, uint8_t const* down_decode_sf,
    __half* inter_fp16_scratch, __half* output, cudaStream_t stream,
    int thread_block_size = kDefaultMlpW4a4DecodeThreadBlockSize,
    MoEActivationKind activation_kind = MoEActivationKind::kReLU2);

//! W4A4 up-proj: grid \ref moeW4a4DecodeUpGridDim; inner loop over \c hidden_dim/64 Marlin chunks; FP16 scratch
//! (zeroed). \p batch / \p seq_len match row-major \c [batch, seq_len, ...] activations; \p num_tokens = batch*seq_len.
//! \p up_decode_sf is row-major \c [E, H/16, I] FP8 block scales.
void launchNemotronMoeW4A4DecodeUpGemvCuda(int batch, int seq_len, int hidden_dim, int inter_dim, int num_experts,
    int top_k, int32_t const* expert_ids, NVFP4Tensor const activation, NVFP4Tensor const up,
    uint8_t const* up_decode_sf, __half* inter_fp16_out, cudaStream_t stream,
    int thread_block_size = kDefaultMlpW4a4DecodeThreadBlockSize);

//! W4A4 down-proj: reads FP16 \p inter_in (up output; already scaled by \c activation.global_scale[0]); converts to
//! FP32 for nonlinearity and down GEMV. Accumulates into FP16 \p output (zeroed here before atomics; same as W4A16
//! down). \p down_decode_sf is row-major \c [E, I/16, H] FP8 block scales.
void launchNemotronMoeW4A4DecodeDownGemvCuda(int batch, int seq_len, int hidden_dim, int inter_dim, int hidden_chunks,
    int num_experts, int top_k, int32_t const* expert_ids, float const* topk_weights, __half const* inter_in,
    NVFP4Tensor const down, uint8_t const* down_decode_sf, __half* output, cudaStream_t stream, int thread_block_size,
    MoEActivationKind activation_kind = MoEActivationKind::kReLU2);

//! FP16 activation \c [batch * seq_len, hidden_dim] row-major; NVFP4 up/down; FP16 output \c [batch * seq_len,
//! hidden_dim] (same layout as W4A4).
//! \p expert_ids and \p topk_weights are row-major \c [batch * seq_len, top_k] (device). Internally runs split up (\ref
//! launchNemotronMoeW4A16DecodeUpGemvCuda) then down (\ref launchNemotronMoeW4A16DecodeDownGemvCuda). \p
//! inter_fp16_scratch must be sized with \ref nemotronMoeW4A16UpFp16ScratchBytes (FP16 \c [batch * seq_len, top_k,
//! inter_dim]
//! between the two passes). Up strips use \ref nemotronMoeW4A16DecodeThreadBlockSizeForInterDim(\p hidden_dim); down
//! strips use the same helper on
//! \p inter_dim. \p num_chunks is legacy (ignored); chunk counts are \c inter_dim/64 (up inner loop) and \c
//! hidden_dim/64 (down inner loop). Per-expert weight scales live in \c up.global_scale / \c down.global_scale (device,
//! length \p num_experts). Down-proj accumulates directly into \p output (FP16); the down launch zeroes \p output
//! before atomics.
void launchNemotronMoeW4A16DecodeGemvCuda(int batch, int seq_len, int hidden_dim, int inter_dim, int num_chunks,
    int num_experts, int top_k, int32_t const* expert_ids, float const* topk_weights, __half const* activation,
    NVFP4Tensor const up, NVFP4Tensor const down, uint8_t const* up_decode_sf, uint8_t const* down_decode_sf,
    __half* inter_fp16_scratch, __half* output, cudaStream_t stream,
    MoEActivationKind activation_kind = MoEActivationKind::kReLU2);

//! Up-proj split W4A16 decode: grid \c batch × seq_len × top_k × (hidden_dim / block_size) (hidden strips). Partial
//! dots
//! \ref atomicAddHalf2Aligned into \p inter_fp16_out (zeroed by this launch); result is FP16 \c z for \ref
//! launchNemotronMoeW4A16DecodeDownGemvCuda. Block size is \ref nemotronMoeW4A16DecodeThreadBlockSizeForInterDim(\p
//! hidden_dim). Inner loop count is \c inter_dim/64
//! (\p num_chunks is ignored). \p topk_weights is unused (nonlinearity×router are applied in the down pass).
//! \p up_decode_sf is row-major \c [E, H/16, I] FP8 block scales.
void launchNemotronMoeW4A16DecodeUpGemvCuda(int batch, int seq_len, int hidden_dim, int inter_dim, int num_chunks,
    int num_experts, int top_k, int32_t const* expert_ids, float const* topk_weights, __half const* activation,
    NVFP4Tensor const up, uint8_t const* up_decode_sf, __half* inter_fp16_out, cudaStream_t stream);

//! Down-proj: reads FP16 \p inter_in row-major \c [batch * seq_len, top_k, inter_dim] (same as up output); converts to
//! FP32 for nonlinearity and \p topk_weights, NVFP4 matmul with FP16 tile accumulate (\ref
//! accumulateNvfp4GemvTileWarpReduce) into \p output (row-major \c [batch * seq_len, hidden_dim]; zeroed by this launch
//! before atomics).
//! Grid: \c batch * seq_len * top_k * (inter_dim / block_size). Inner loop over hidden tiles uses \c hidden_dim/64
//! (\p num_chunks is ignored). \p down_decode_sf is row-major \c [E, I/16, H] FP8 block scales.
void launchNemotronMoeW4A16DecodeDownGemvCuda(int batch, int seq_len, int hidden_dim, int inter_dim, int num_chunks,
    int num_experts, int top_k, int32_t const* expert_ids, float const* topk_weights, __half const* inter_in,
    NVFP4Tensor const down, uint8_t const* down_decode_sf, __half* output, cudaStream_t stream,
    MoEActivationKind activation_kind = MoEActivationKind::kReLU2);

#endif // SUPPORTS_FP4

//! Element count for W4A16/W4A4 split intermediate tensor \c [batch * seq_len, top_k, inter_dim] (row-major); stored as
//! FP16 between up and down.
inline int64_t nemotronMoeW4A16InterBufferNumElems(int batch_size, int top_k, int inter_dim) noexcept
{
    return static_cast<int64_t>(batch_size) * static_cast<int64_t>(top_k) * static_cast<int64_t>(inter_dim);
}

//! Device scratch bytes: FP16 row-major \c [batch * seq_len, top_k, inter_dim] (up-proj dot \c z), passed to down-proj
//! as \c __half*.
inline int64_t nemotronMoeW4A16UpFp16ScratchBytes(int batch_size, int top_k, int inter_dim) noexcept
{
    return nemotronMoeW4A16InterBufferNumElems(batch_size, top_k, inter_dim) * static_cast<int64_t>(sizeof(uint16_t));
}

} // namespace trt_edgellm
