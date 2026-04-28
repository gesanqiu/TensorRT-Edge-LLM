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

#include "kernels/moe/moe_marlin/marlin/nvfp4_tensor.cuh"
#include "kernels/moe/nvf4_w4an/kernels.h"
#include "kernels/moe/nvf4_w4an/marlin_template.cuh"
#include <cassert>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if SUPPORTS_FP4

namespace trt_edgellm
{

namespace
{
template <int threadBlockSize>
void launchMoeW4A4UpGemvTemplated(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const inter_chunks, int const num_experts, int const top_k, int32_t const* expert_ids,
    NVFP4Tensor const activation, NVFP4Tensor const up, uint8_t const* up_decode_sf, __half* inter_fp16_out,
    cudaStream_t stream)
{
    assert(hidden_dim % 64 == 0);
    assert(hidden_dim % threadBlockSize == 0);
    int const grid = moeDecodeGemvTopkGridDimBatchSeq(batch, seq_len, top_k, hidden_dim, threadBlockSize);
    moeW4A4DecodeUpGemvKernel<threadBlockSize><<<grid, threadBlockSize, 0, stream>>>(batch, seq_len, hidden_dim,
        inter_dim, inter_chunks, num_experts, top_k, expert_ids, activation, up, up_decode_sf, inter_fp16_out);
}

template <int threadBlockSize, MoEActivationKind kAct>
void launchMoeW4A4DownGemvTemplated(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const hidden_chunks, int const num_experts, int const top_k, int32_t const* expert_ids,
    float const* topk_weights, __half const* inter_in, NVFP4Tensor const down, uint8_t const* down_decode_sf,
    __half* output, cudaStream_t stream)
{
    int const grid = moeDecodeGemvTopkGridDimBatchSeq(batch, seq_len, top_k, inter_dim, threadBlockSize);
    moeW4A4DecodeDownGemvKernel<threadBlockSize, kAct><<<grid, threadBlockSize, 0, stream>>>(batch, seq_len, hidden_dim,
        inter_dim, hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output);
}

template <int threadBlockSize>
void launchMoeW4A16UpGemvTemplated(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const inter_chunks, int const num_experts, int const top_k, int32_t const* expert_ids, __half const* activation,
    NVFP4Tensor const up, uint8_t const* up_decode_sf, __half* inter_fp16_out, cudaStream_t stream)
{
    int const grid = moeDecodeGemvTopkGridDimBatchSeq(batch, seq_len, top_k, hidden_dim, threadBlockSize);
    moeW4A16DecodeUpGemvKernel<threadBlockSize><<<grid, threadBlockSize, 0, stream>>>(batch, seq_len, hidden_dim,
        inter_dim, inter_chunks, num_experts, top_k, expert_ids, activation, up, up_decode_sf, inter_fp16_out);
}

template <int threadBlockSize, MoEActivationKind kAct>
void launchMoeW4A16DownGemvTemplated(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const hidden_chunks, int const num_experts, int const top_k, int32_t const* expert_ids,
    float const* topk_weights, __half const* inter_in, NVFP4Tensor const down, uint8_t const* down_decode_sf,
    __half* output, cudaStream_t stream)
{
    int const grid = moeDecodeGemvTopkGridDimBatchSeq(batch, seq_len, top_k, inter_dim, threadBlockSize);
    moeW4A16DecodeDownGemvKernel<threadBlockSize, kAct><<<grid, threadBlockSize, 0, stream>>>(batch, seq_len,
        hidden_dim, inter_dim, hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down,
        down_decode_sf, output);
}
} // namespace

void launchNemotronMoeW4A4DecodeUpGemvCuda(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const num_experts, int const top_k, int32_t const* expert_ids,
    NVFP4Tensor const activation, NVFP4Tensor const up, uint8_t const* up_decode_sf, __half* inter_fp16_out,
    cudaStream_t stream, int const thread_block_size)
{
    assert(batch >= 1);
    assert(seq_len >= 1);
    assert(hidden_dim % 64 == 0);
    assert(inter_dim % 64 == 0);
    assert(moeW4a4DecodeIsValidThreadBlockSize(thread_block_size));
    assert(hidden_dim % thread_block_size == 0);
    assert(top_k > 0);
    assert(expert_ids != nullptr);
    assert(up_decode_sf != nullptr);
    assert(inter_fp16_out != nullptr);

    int const inter_chunks = inter_dim / 64;
    int const num_tokens = batch * seq_len;
    int64_t const scratch_bytes = nemotronMoeW4A16UpFp16ScratchBytes(num_tokens, top_k, inter_dim);
    cudaMemsetAsync(inter_fp16_out, 0, static_cast<size_t>(scratch_bytes), stream);

    switch (thread_block_size)
    {
    case 64:
        launchMoeW4A4UpGemvTemplated<64>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 96:
        launchMoeW4A4UpGemvTemplated<96>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 128:
        launchMoeW4A4UpGemvTemplated<128>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 192:
        launchMoeW4A4UpGemvTemplated<192>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 256:
        launchMoeW4A4UpGemvTemplated<256>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    default: assert(false && "unsupported thread_block_size"); break;
    }
}

void launchNemotronMoeW4A4DecodeDownGemvCuda(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const hidden_chunks, int const num_experts, int const top_k, int32_t const* expert_ids,
    float const* topk_weights, __half const* inter_in, NVFP4Tensor const down, uint8_t const* down_decode_sf,
    __half* output, cudaStream_t stream, int const thread_block_size, MoEActivationKind const activation_kind)
{
    assert(batch >= 1);
    assert(seq_len >= 1);
    assert(hidden_dim % 64 == 0);
    assert(moeW4a4DecodeIsValidThreadBlockSize(thread_block_size));
    assert(inter_dim % thread_block_size == 0);
    assert(top_k > 0);
    assert(expert_ids != nullptr);
    assert(topk_weights != nullptr);
    assert(inter_in != nullptr);
    assert(down_decode_sf != nullptr);
    assert(output != nullptr);

    int const num_tokens = batch * seq_len;
    int64_t const out_elems = static_cast<int64_t>(num_tokens) * static_cast<int64_t>(hidden_dim);
    size_t const out_bytes = static_cast<size_t>(out_elems) * sizeof(__half);
    cudaMemsetAsync(output, 0, out_bytes, stream);

    switch (thread_block_size)
    {
    case 64:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A4DownGemvTemplated<64, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A4DownGemvTemplated<64, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 96:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A4DownGemvTemplated<96, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A4DownGemvTemplated<96, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 128:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A4DownGemvTemplated<128, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A4DownGemvTemplated<128, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 192:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A4DownGemvTemplated<192, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A4DownGemvTemplated<192, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 256:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A4DownGemvTemplated<256, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A4DownGemvTemplated<256, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    default: assert(false && "unsupported thread_block_size"); break;
    }
}

void launchNemotronMoeW4a4DecodeGemvCuda(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const num_chunks, int const num_experts, int const top_k, int32_t const* expert_ids, float const* topk_weights,
    NVFP4Tensor const activation, NVFP4Tensor const up, NVFP4Tensor const down, uint8_t const* up_decode_sf,
    uint8_t const* down_decode_sf, __half* inter_fp16_scratch, __half* output, cudaStream_t stream,
    int const thread_block_size, MoEActivationKind const activation_kind)
{
    (void) num_chunks;
    assert(batch >= 1);
    assert(seq_len >= 1);
    assert(inter_fp16_scratch != nullptr);
    assert(output != nullptr);
    assert(hidden_dim % 64 == 0);
    int const hidden_chunks = hidden_dim / 64;
    launchNemotronMoeW4A4DecodeUpGemvCuda(batch, seq_len, hidden_dim, inter_dim, num_experts, top_k, expert_ids,
        activation, up, up_decode_sf, inter_fp16_scratch, stream, thread_block_size);
    launchNemotronMoeW4A4DecodeDownGemvCuda(batch, seq_len, hidden_dim, inter_dim, hidden_chunks, num_experts, top_k,
        expert_ids, topk_weights, inter_fp16_scratch, down, down_decode_sf, output, stream, thread_block_size,
        activation_kind);
}

void launchNemotronMoeW4A16DecodeGemvCuda(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const num_chunks, int const num_experts, int const top_k, int32_t const* expert_ids, float const* topk_weights,
    __half const* activation, NVFP4Tensor const up, NVFP4Tensor const down, uint8_t const* up_decode_sf,
    uint8_t const* down_decode_sf, __half* inter_fp16_scratch, __half* output, cudaStream_t stream,
    MoEActivationKind const activation_kind)
{
    (void) num_chunks;
    assert(inter_fp16_scratch != nullptr);
    assert(output != nullptr);
    assert(inter_dim % 64 == 0);
    assert(hidden_dim % 64 == 0);
    assert(batch >= 1);
    assert(seq_len >= 1);
    int const num_chunks_up = inter_dim / 64;
    int const num_chunks_down = hidden_dim / 64;
    launchNemotronMoeW4A16DecodeUpGemvCuda(batch, seq_len, hidden_dim, inter_dim, num_chunks_up, num_experts, top_k,
        expert_ids, topk_weights, activation, up, up_decode_sf, inter_fp16_scratch, stream);
    launchNemotronMoeW4A16DecodeDownGemvCuda(batch, seq_len, hidden_dim, inter_dim, num_chunks_down, num_experts, top_k,
        expert_ids, topk_weights, inter_fp16_scratch, down, down_decode_sf, output, stream, activation_kind);
}

void launchNemotronMoeW4A16DecodeUpGemvCuda(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const inter_chunks, int const num_experts, int const top_k, int32_t const* expert_ids,
    float const* topk_weights, __half const* activation, NVFP4Tensor const up, uint8_t const* up_decode_sf,
    __half* inter_fp16_out, cudaStream_t stream)
{
    (void) topk_weights;
    assert(inter_dim % 64 == 0);
    assert(batch >= 1);
    assert(seq_len >= 1);
    // Grid strips along \p hidden_dim (\ref moeW4A16DecodeUpGemvKernel), not \p inter_dim.
    int const thread_block_size = nemotronMoeW4A16DecodeThreadBlockSizeForInterDim(hidden_dim);
    assert(thread_block_size > 0);
    assert(moeW4a4DecodeIsValidThreadBlockSize(thread_block_size));
    assert(hidden_dim % thread_block_size == 0);
    assert(top_k > 0);
    assert(expert_ids != nullptr);
    assert(activation != nullptr);
    assert(up_decode_sf != nullptr);
    assert(inter_fp16_out != nullptr);

    int64_t const num_tokens = static_cast<int64_t>(batch) * static_cast<int64_t>(seq_len);
    int64_t const scratch_bytes = nemotronMoeW4A16UpFp16ScratchBytes(num_tokens, top_k, inter_dim);
    cudaMemsetAsync(inter_fp16_out, 0, static_cast<size_t>(scratch_bytes), stream);

    switch (thread_block_size)
    {
    case 64:
        launchMoeW4A16UpGemvTemplated<64>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 96:
        launchMoeW4A16UpGemvTemplated<96>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 128:
        launchMoeW4A16UpGemvTemplated<128>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 192:
        launchMoeW4A16UpGemvTemplated<192>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    case 256:
        launchMoeW4A16UpGemvTemplated<256>(batch, seq_len, hidden_dim, inter_dim, inter_chunks, num_experts, top_k,
            expert_ids, activation, up, up_decode_sf, inter_fp16_out, stream);
        break;
    default: assert(false && "unsupported thread_block_size"); break;
    }
}

void launchNemotronMoeW4A16DecodeDownGemvCuda(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const hidden_chunks, int const num_experts, int const top_k, int32_t const* expert_ids,
    float const* topk_weights, __half const* inter_in, NVFP4Tensor const down, uint8_t const* down_decode_sf,
    __half* output, cudaStream_t stream, MoEActivationKind const activation_kind)
{
    assert(hidden_dim % 64 == 0);
    assert(batch >= 1);
    assert(seq_len >= 1);
    int const thread_block_size = nemotronMoeW4A16DecodeThreadBlockSizeForInterDim(inter_dim);
    assert(thread_block_size > 0);
    assert(moeW4a4DecodeIsValidThreadBlockSize(thread_block_size));
    assert(inter_dim % thread_block_size == 0);
    assert(top_k > 0);
    assert(expert_ids != nullptr);
    assert(topk_weights != nullptr);
    assert(inter_in != nullptr);
    assert(down_decode_sf != nullptr);
    assert(output != nullptr);

    int64_t const num_tokens = static_cast<int64_t>(batch) * static_cast<int64_t>(seq_len);
    int64_t const out_elems = num_tokens * static_cast<int64_t>(hidden_dim);
    size_t const out_bytes = static_cast<size_t>(out_elems) * sizeof(__half);
    cudaMemsetAsync(output, 0, out_bytes, stream);

    switch (thread_block_size)
    {
    case 64:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A16DownGemvTemplated<64, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A16DownGemvTemplated<64, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 96:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A16DownGemvTemplated<96, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A16DownGemvTemplated<96, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 128:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A16DownGemvTemplated<128, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A16DownGemvTemplated<128, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 192:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A16DownGemvTemplated<192, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A16DownGemvTemplated<192, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    case 256:
        if (activation_kind == MoEActivationKind::kSiLU)
        {
            launchMoeW4A16DownGemvTemplated<256, MoEActivationKind::kSiLU>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        else
        {
            launchMoeW4A16DownGemvTemplated<256, MoEActivationKind::kReLU2>(batch, seq_len, hidden_dim, inter_dim,
                hidden_chunks, num_experts, top_k, expert_ids, topk_weights, inter_in, down, down_decode_sf, output,
                stream);
        }
        break;
    default: assert(false && "unsupported thread_block_size"); break;
    }
}

} // namespace trt_edgellm

#endif // SUPPORTS_FP4
