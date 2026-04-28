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

#include "utilKernels.h"

#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace kernel
{

__global__ void calCuQCuKVSeqLensAndKVEndIdxsKernel(int32_t const* inputSeqLen, int32_t const* kvCacheStartIndices,
    int32_t* cuQSeqlen, int32_t* cuKVSeqLens, int32_t* kvCacheEndIndices, int32_t* paddedCuKVSeqLens,
    int32_t runtimeSeqLen, int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cuQSeqlen[0] = 0;
        cuKVSeqLens[0] = 0;
        if (paddedCuKVSeqLens != nullptr)
        {
            paddedCuKVSeqLens[0] = 0;
        }

        int32_t runningCuSeqLen = 0;
        int32_t runningCuKvCacheLen = 0;
        int32_t runningPaddedCuKvLen = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            runningCuSeqLen += inputSeqLen[i];
            cuQSeqlen[i + 1] = runningCuSeqLen;

            int32_t kvCacheStartIdx = 0;
            if (kvCacheStartIndices != nullptr)
            {
                kvCacheStartIdx = kvCacheStartIndices[i];
            }

            runningCuKvCacheLen += (kvCacheStartIdx + inputSeqLen[i]);
            cuKVSeqLens[i + 1] = runningCuKvCacheLen;
            // To keep semantic consistency with the packed QKV layout for RoPE, use runtimeSeqLen here.
            int32_t const kvEndIdx = kvCacheStartIdx + runtimeSeqLen;
            kvCacheEndIndices[i] = kvEndIdx;

            if (paddedCuKVSeqLens != nullptr)
            {
                runningPaddedCuKvLen += kvEndIdx;
                paddedCuKVSeqLens[i + 1] = runningPaddedCuKvLen;
            }
        }
    }
}

void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs,
    rt::OptionalOutputTensor paddedCuKVSeqLens, int32_t const runtimeSeqLen, cudaStream_t stream)
{
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputSeqLen.getShape()[0]);

    // Perform necessary shape checks.
    check::check(cuQSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuQSeqLens shall have shape [B+1].");
    check::check(cuKVSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuKVSeqLens shall have shape [B+1].");
    check::check(kvCacheEndIdxs.getShape()[0] == runtimeBatchSize, "kvCacheEndIdxs shall have shape [B].");

    if (!kvCacheStartIndices.isEmpty())
    {
        check::check(
            kvCacheStartIndices.getShape()[0] == runtimeBatchSize, "KVCacheStartIndices tensor shall have shape [B].");
    }
    else
    {
        // We rely on this nullptr behavior to indicate whether kvCacheStartIndices is available in the kernel.
        check::check(kvCacheStartIndices.rawPointer() == nullptr,
            "KVCacheStartIndices tensor shall be nullptr when it is empty.");
    }

    int32_t* paddedPtr = nullptr;
    if (paddedCuKVSeqLens.has_value())
    {
        rt::Tensor& paddedTensor = paddedCuKVSeqLens.value().get();
        check::check(paddedTensor.getShape()[0] == (runtimeBatchSize + 1), "paddedCuKVSeqLens shall have shape [B+1].");
        paddedPtr = paddedTensor.dataPointer<int32_t>();
    }

    calCuQCuKVSeqLensAndKVEndIdxsKernel<<<1, 1, 0, stream>>>(inputSeqLen.dataPointer<int32_t>(),
        kvCacheStartIndices.dataPointer<int32_t>(), cuQSeqLens.dataPointer<int32_t>(),
        cuKVSeqLens.dataPointer<int32_t>(), kvCacheEndIdxs.dataPointer<int32_t>(), paddedPtr, runtimeSeqLen,
        runtimeBatchSize);
}

// ===== kernel: produce separate K [B, S, H, D] and V [B, S, H, D] =====
//! Splits the interleaved [B, 2, H, S, D] KV cache into two independent tensors.
//! Splits the [B, 2, H, S, D] KV cache into separate K [B, S, H, D] and V [B, S, H, D] tensors.
template <typename T>
__global__ void cvtKVLayoutBHSDToSplitKVKernel(T const* __restrict__ src, // [B, 2, H, S, D]
    half* __restrict__ kDst,                                              // [B, S, H, D]
    half* __restrict__ vDst,                                              // [B, S, H, D]
    float const* __restrict__ kScaleQuantOrig, float const* __restrict__ vScaleQuantOrig, int32_t B, int32_t S,
    int32_t H, int32_t D)
{
    uint32_t const token = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. S-1
    uint32_t const d = blockIdx.x * blockDim.x + threadIdx.x;     // 0 .. D-1

    uint32_t const numHpBlocks = (2 * H + blockDim.z - 1) / blockDim.z;
    uint32_t const batch = blockIdx.z / numHpBlocks;
    uint32_t const hpTile = blockIdx.z % numHpBlocks;
    uint32_t const headPair = hpTile * blockDim.z + threadIdx.z; // 0 .. 2*H-1

    if (batch >= B || headPair >= 2 * H || d >= D || token >= S)
        return;

    uint32_t const kv = headPair / H; // 0 = K, 1 = V
    uint32_t const h = headPair % H;

    // src layout: [B, 2, H, S, D]
    size_t const srcIdx = (((((size_t) batch * 2 + kv) * H + h) * S + token) * D + d);
    // dst layout: [B, S, H, D]
    size_t const dstIdx = ((((size_t) batch * S + token) * H + h) * D + d);

    half* dst = (kv == 0) ? kDst : vDst;

#if SUPPORTS_FP8
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        float const scale = (kv == 0) ? kScaleQuantOrig[0] : vScaleQuantOrig[0];
        dst[dstIdx] = __float2half(static_cast<float>(src[srcIdx]) * scale);
    }
    else
#endif
    {
        dst[dstIdx] = src[srcIdx];
    }
}

void cvtKVLayoutBHSDToSplitKV(
    rt::Tensor const& src, rt::Tensor& kDst, rt::Tensor& vDst, rt::Tensor const& kvScaleQuantOrig, cudaStream_t stream)
{
    rt::Coords const srcShape = src.getShape();
    int32_t const B = static_cast<int32_t>(srcShape[0]);
    int32_t const H = static_cast<int32_t>(srcShape[2]);
    int32_t const S = static_cast<int32_t>(srcShape[3]);
    int32_t const D = static_cast<int32_t>(srcShape[4]);

    check::check(srcShape[1] == 2, "Source tensor must have shape [B, 2, H, S, D].");
    check::check(kDst.getDataType() == nvinfer1::DataType::kHALF, "kDst must be FP16.");
    check::check(vDst.getDataType() == nvinfer1::DataType::kHALF, "vDst must be FP16.");

    rt::Coords const kShape = kDst.getShape();
    rt::Coords const vShape = vDst.getShape();
    check::check(
        kShape[0] == B && kShape[1] == S && kShape[2] == H && kShape[3] == D, "kDst must have shape [B, S, H, D].");
    check::check(
        vShape[0] == B && vShape[1] == S && vShape[2] == H && vShape[3] == D, "vDst must have shape [B, S, H, D].");

    // Block config with safe thread count (≤ 1024)
    uint32_t const tx = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    uint32_t const ty = 4; // token dimension per block
    uint32_t const tz = 1; // process one head-pair per thread in z
    dim3 block(tx, ty, tz);

    // Grid config
    uint32_t const hpTilesPerBatch = (2 * H + tz - 1) / tz; // z-blocks needed per batch for head-pairs
    dim3 grid((D + tx - 1) / tx,                            // x : feature dim
        (S + ty - 1) / ty,                                  // y : token dim
        hpTilesPerBatch * B);                               // z : (batch, headPair)

    if (src.getDataType() == nvinfer1::DataType::kHALF)
    {
        cvtKVLayoutBHSDToSplitKVKernel<half><<<grid, block, 0, stream>>>(
            src.dataPointer<half>(), kDst.dataPointer<half>(), vDst.dataPointer<half>(), nullptr, nullptr, B, S, H, D);
    }
#if SUPPORTS_FP8
    else if (src.getDataType() == nvinfer1::DataType::kFP8)
    {
        check::check(!kvScaleQuantOrig.isEmpty(), "kvScaleQuantOrig is required for FP8 KV cache");
        check::check(kvScaleQuantOrig.getDataType() == nvinfer1::DataType::kFLOAT, "kvScaleQuantOrig must be FP32.");
        check::check(kvScaleQuantOrig.getShape().getNumDims() == 1 && kvScaleQuantOrig.getShape()[0] == 2,
            "kvScaleQuantOrig shall have shape [2] with layout [kScaleQuantOrig, vScaleQuantOrig].");
        float const* const scales = kvScaleQuantOrig.dataPointer<float>();
        float const* const kScaleQuantOrigPtr = scales + 0;
        float const* const vScaleQuantOrigPtr = scales + 1;
        cvtKVLayoutBHSDToSplitKVKernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(src.dataPointer<__nv_fp8_e4m3>(),
            kDst.dataPointer<half>(), vDst.dataPointer<half>(), kScaleQuantOrigPtr, vScaleQuantOrigPtr, B, S, H, D);
    }
#endif
    else
    {
        check::check(false, "Unsupported KV cache dtype");
    }
}

} // namespace kernel
} // namespace trt_edgellm
