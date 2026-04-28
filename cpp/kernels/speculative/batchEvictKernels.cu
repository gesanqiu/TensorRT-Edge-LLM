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

#include "batchEvictKernels.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/stringUtils.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cstdint>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

//=============================================================================
// KV Cache Compaction Kernel
//=============================================================================

// Configuration struct to reduce kernel parameter count
struct KVCacheConfig
{
    int32_t numLayers;
    int32_t maxBatchSize;
    int32_t numKVHeads;
    int32_t maxSeqLen;
    int32_t oldActiveBatch;
};

template <typename T, int32_t HEAD_DIM>
__global__ void compactKVCacheKernel(int32_t const* batchMapping, // [oldActiveBatch] - const input
    int32_t const* srcKVLengths,                                  // [oldActiveBatch] - const input
    KVCacheConfig const config,                                   // Const config struct
    T* kvCache,                                                   // [L, maxB, 2, H, S, D] - in-place output
    int32_t* dstKVLengths)                                        // [newActiveBatch] - output
{
    // The kernel doesn't deal with leftovers because of the nice alignment.
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512,
        "Only HEAD_DIM = 64, 128, 256, or 512 are supported");

    // Unpack config
    int32_t const numLayers = config.numLayers;
    int32_t const maxBatchSize = config.maxBatchSize;
    int32_t const numKVHeads = config.numKVHeads;
    int32_t const maxSeqLen = config.maxSeqLen;
    int32_t const oldActiveBatch = config.oldActiveBatch;

    // Grid: numLayers * numKVHeads * 2 CTAs
    // Each CTA handles: (layerIdx, kvIdx, kvHeadIdx) slice
    int32_t const ctaIdx = blockIdx.x;
    int32_t const totalKVHeads = numKVHeads * 2;
    int32_t const layerIdx = ctaIdx / totalKVHeads;
    int32_t const remainder = ctaIdx % totalKVHeads;
    int32_t const kvIdx = remainder / numKVHeads;
    int32_t const kvHeadIdx = remainder % numKVHeads;

    // Early exit if this CTA is beyond the valid range
    if (layerIdx >= numLayers)
    {
        return;
    }

    // Calculate strides
    // Layout: [layer, batch, kv, head, seq, dim]
    int64_t const seqStride = HEAD_DIM;
    int64_t const headStride = maxSeqLen * seqStride;
    int64_t const kvStride = numKVHeads * headStride;
    int64_t const batchStride = 2 * kvStride;
    int64_t const layerStride = maxBatchSize * batchStride;

    using Vec = DVec<T>;
    constexpr int32_t VEC_SIZE = Vec::vec_size;
    int32_t const threadsPerBlock = blockDim.x;

    for (int32_t oldBatchIdx = 0; oldBatchIdx < oldActiveBatch; ++oldBatchIdx)
    {
        int32_t const newBatchIdx = batchMapping[oldBatchIdx];

        if (newBatchIdx < 0 || newBatchIdx >= maxBatchSize)
        {
            continue;
        }

        if (oldBatchIdx == newBatchIdx)
        {
            continue;
        }

        int32_t const seqLen = srcKVLengths[oldBatchIdx];
        if (seqLen == 0)
        {
            continue;
        }

        int32_t const elemsPerKV = seqLen * HEAD_DIM;
        int64_t const srcBatchOffset
            = layerIdx * layerStride + oldBatchIdx * batchStride + kvIdx * kvStride + kvHeadIdx * headStride;
        int64_t const dstBatchOffset
            = layerIdx * layerStride + newBatchIdx * batchStride + kvIdx * kvStride + kvHeadIdx * headStride;

        T const* srcPtr = kvCache + srcBatchOffset;
        T* dstPtr = kvCache + dstBatchOffset;

        int32_t const numVecs = elemsPerKV / VEC_SIZE;
        for (int32_t vecIdx = threadIdx.x; vecIdx < numVecs; vecIdx += threadsPerBlock)
        {
            Vec vec;
            vec.load(srcPtr + vecIdx * VEC_SIZE);
            vec.store(dstPtr + vecIdx * VEC_SIZE);
        }

        // Update kvCacheLengths (only first thread of first layer, K side, head 0)
        if (dstKVLengths != nullptr && threadIdx.x == 0 && layerIdx == 0 && kvIdx == 0 && kvHeadIdx == 0)
        {
            dstKVLengths[newBatchIdx] = seqLen;
        }
    }
}

void compactKVCacheSingleLayer(rt::Tensor& kvCacheLayer, rt::Tensor const& batchMapping,
    rt::Tensor const& kvCacheLengths, rt::Tensor& dstKVCacheLengths, int32_t oldActiveBatch, int32_t newActiveBatch,
    bool updateLengths, cudaStream_t stream)
{
    check::check(kvCacheLayer.getDeviceType() == rt::DeviceType::kGPU, "KV cache layer must be on GPU");
    check::check(batchMapping.getDeviceType() == rt::DeviceType::kGPU, "Batch mapping must be on GPU");

    auto const& kvShape = kvCacheLayer.getShape();
    check::check(kvShape.getNumDims() == 5 && kvShape[1] == 2, "Single-layer KV cache must be 5D: [maxB, 2, H, S, D]");

    int32_t const maxBatchSize = kvShape[0];
    int32_t const numKVHeads = kvShape[2];
    int32_t const maxSeqLen = kvShape[3];
    int32_t const headDim = kvShape[4];

    check::check(oldActiveBatch <= maxBatchSize, "oldActiveBatch exceeds maxBatchSize");
    check::check(newActiveBatch <= oldActiveBatch, "newActiveBatch must be <= oldActiveBatch");

    if (oldActiveBatch == newActiveBatch)
    {
        return;
    }

    // Grid: numKVHeads * 2 CTAs (single layer, so no layer dimension)
    int32_t const numCTAs = numKVHeads * 2;
    int32_t const threadsPerBlock = 256;

    int32_t const* batchMappingPtr = batchMapping.dataPointer<int32_t>();
    int32_t const* srcKVLengthsPtr = kvCacheLengths.dataPointer<int32_t>();
    int32_t* dstKVLengthsPtr = updateLengths ? dstKVCacheLengths.dataPointer<int32_t>() : nullptr;

    KVCacheConfig const config{1, maxBatchSize, numKVHeads, maxSeqLen, oldActiveBatch};

    switch (headDim)
    {
    case 64:
        compactKVCacheKernel<half, 64><<<numCTAs, threadsPerBlock, 0, stream>>>(
            batchMappingPtr, srcKVLengthsPtr, config, kvCacheLayer.dataPointer<half>(), dstKVLengthsPtr);
        break;
    case 128:
        compactKVCacheKernel<half, 128><<<numCTAs, threadsPerBlock, 0, stream>>>(
            batchMappingPtr, srcKVLengthsPtr, config, kvCacheLayer.dataPointer<half>(), dstKVLengthsPtr);
        break;
    case 256:
        compactKVCacheKernel<half, 256><<<numCTAs, threadsPerBlock, 0, stream>>>(
            batchMappingPtr, srcKVLengthsPtr, config, kvCacheLayer.dataPointer<half>(), dstKVLengthsPtr);
        break;
    case 512:
        compactKVCacheKernel<half, 512><<<numCTAs, threadsPerBlock, 0, stream>>>(
            batchMappingPtr, srcKVLengthsPtr, config, kvCacheLayer.dataPointer<half>(), dstKVLengthsPtr);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("compactKVCacheSingleLayer: Unsupported headDim=%d. Only 64, 128, 256, or 512.", headDim));
    }

    CUDA_CHECK(cudaGetLastError());
}

//=============================================================================
// Generic Tensor Compaction Kernel
//=============================================================================

template <typename T>
__global__ void compactTensorBatchKernel(
    T const* src, int32_t const* batchMapping, T* dst, int32_t oldActiveBatch, int32_t batchStride)
{
    // Each CTA handles all elements (no batch-specific assignment)
    int32_t const elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemIdx >= batchStride)
    {
        return;
    }

    for (int32_t oldBatchIdx = 0; oldBatchIdx < oldActiveBatch; ++oldBatchIdx)
    {
        int32_t const newBatchIdx = batchMapping[oldBatchIdx];

        if (newBatchIdx < 0 || newBatchIdx >= oldActiveBatch)
        {
            continue;
        }

        if (oldBatchIdx == newBatchIdx)
        {
            continue;
        }

        int64_t const srcIdx = static_cast<int64_t>(oldBatchIdx) * batchStride + elemIdx;
        int64_t const dstIdx = static_cast<int64_t>(newBatchIdx) * batchStride + elemIdx;
        dst[dstIdx] = src[srcIdx];
    }
}

void compactTensorBatch(rt::Tensor const& src, rt::Tensor const& batchMapping, rt::Tensor& dst, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream)
{
    check::check(dst.getDeviceType() == rt::DeviceType::kGPU, "Destination tensor must be on GPU");
    check::check(src.getDeviceType() == rt::DeviceType::kGPU, "Source tensor must be on GPU");
    check::check(batchMapping.getDeviceType() == rt::DeviceType::kGPU, "Batch mapping must be on GPU");

    auto const& srcShape = src.getShape();
    check::check(srcShape.getNumDims() >= 1, "Tensor must have at least 1 dimension");
    check::check(srcShape[0] == oldActiveBatch, "First dimension must match oldActiveBatch");

    int64_t batchStride = 1;
    for (int32_t i = 1; i < srcShape.getNumDims(); ++i)
    {
        batchStride *= srcShape[i];
    }

    check::check(batchStride <= std::numeric_limits<int32_t>::max(), "Batch stride too large for int32_t");

    auto const batchStrideInt = static_cast<int32_t>(batchStride);

    if (batchStrideInt == 0)
    {
        return;
    }

    int32_t const threadsPerBlock = 512;
    int32_t const numBlocks = (batchStrideInt + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocks);
    dim3 blockDim(threadsPerBlock);

    int32_t const* batchMappingPtr = batchMapping.dataPointer<int32_t>();

    // Get data type and dispatch to appropriate kernel
    nvinfer1::DataType const dataType = src.getDataType();

    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        compactTensorBatchKernel<half><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<half>(), batchMappingPtr, dst.dataPointer<half>(), oldActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kFLOAT:
        compactTensorBatchKernel<float><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<float>(), batchMappingPtr, dst.dataPointer<float>(), oldActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT32:
        compactTensorBatchKernel<int32_t><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<int32_t>(), batchMappingPtr, dst.dataPointer<int32_t>(), oldActiveBatch, batchStrideInt);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("compactTensorBatch: Unsupported data type=%d. Only HALF, FLOAT, and INT32 are supported.",
                static_cast<int>(dataType)));
    }

    CUDA_CHECK(cudaGetLastError());
}

//=============================================================================
// Batched KV Cache Compaction Kernel
//=============================================================================

template <typename T, int32_t HEAD_DIM>
__global__ void compactKVCacheBatchedKernel(KVLayerInfo const* __restrict__ layerInfos,
    int32_t const* __restrict__ batchMapping, int32_t const* __restrict__ srcKVLengths, int32_t maxBatchSize,
    int32_t oldActiveBatch)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512,
        "Only HEAD_DIM = 64, 128, 256, or 512 are supported");

    // 2D grid: x = kv * head (within layer), y = layer index
    int32_t const layerIdx = blockIdx.y;
    KVLayerInfo const info = layerInfos[layerIdx];

    int32_t const numKVHeads = info.numKVHeads;
    int32_t const maxSeqLen = info.maxSeqLen;
    int32_t const totalKVHeads = numKVHeads * 2;

    // x dimension: kv * head within this layer
    int32_t const localCTAIdx = blockIdx.x;
    if (localCTAIdx >= totalKVHeads)
    {
        return;
    }

    int32_t const kvIdx = localCTAIdx / numKVHeads;
    int32_t const kvHeadIdx = localCTAIdx % numKVHeads;

    // Strides for this layer (no layer stride — each layer has its own buffer)
    int64_t const headStride = static_cast<int64_t>(maxSeqLen) * HEAD_DIM;
    int64_t const kvStride = static_cast<int64_t>(numKVHeads) * headStride;
    int64_t const batchStride = 2 * kvStride;

    T* kvCache = static_cast<T*>(info.data);

    using Vec = DVec<T>;
    constexpr int32_t VEC_SIZE = Vec::vec_size;
    int32_t const threadsPerBlock = blockDim.x;

    for (int32_t oldBatchIdx = 0; oldBatchIdx < oldActiveBatch; ++oldBatchIdx)
    {
        int32_t const newBatchIdx = batchMapping[oldBatchIdx];
        if (newBatchIdx < 0 || newBatchIdx >= maxBatchSize || oldBatchIdx == newBatchIdx)
        {
            continue;
        }

        int32_t const seqLen = srcKVLengths[oldBatchIdx];
        if (seqLen == 0)
        {
            continue;
        }

        int32_t const elemsPerKV = seqLen * HEAD_DIM;
        int64_t const srcOffset = oldBatchIdx * batchStride + kvIdx * kvStride + kvHeadIdx * headStride;
        int64_t const dstOffset = newBatchIdx * batchStride + kvIdx * kvStride + kvHeadIdx * headStride;

        T const* srcPtr = kvCache + srcOffset;
        T* dstPtr = kvCache + dstOffset;

        int32_t const numVecs = elemsPerKV / VEC_SIZE;
        for (int32_t vecIdx = threadIdx.x; vecIdx < numVecs; vecIdx += threadsPerBlock)
        {
            Vec vec;
            vec.load(srcPtr + vecIdx * VEC_SIZE);
            vec.store(dstPtr + vecIdx * VEC_SIZE);
        }
    }
}

void compactKVCacheBatched(KVLayerInfo const* layerInfos, rt::Tensor const& batchMapping,
    rt::Tensor const& kvCacheLengths, int32_t numLayers, int32_t headDim, nvinfer1::DataType kvCacheType,
    int32_t maxKVHeads, int32_t maxBatchSize, int32_t oldActiveBatch, int32_t newActiveBatch, cudaStream_t stream)
{
    if (oldActiveBatch == newActiveBatch || numLayers == 0)
    {
        return;
    }

    // Grid: (maxKVHeads * 2, numLayers). CTAs with blockIdx.x >= layer's totalKVHeads will early-exit.
    dim3 grid(maxKVHeads * 2, numLayers);
    dim3 block(256);

    int32_t const* batchMappingPtr = batchMapping.dataPointer<int32_t>();
    int32_t const* srcKVLengthsPtr = kvCacheLengths.dataPointer<int32_t>();

    // Dispatch on (kvCacheType, headDim).
    //
    // The kernel template parameter T controls pointer arithmetic and vectorised load/store size.
    // Using the wrong T for a given buffer dtype doubles (or halves) all computed byte offsets,
    // causing out-of-bounds accesses.  KV caches can be either FP16 (2 bytes) or FP8 (1 byte).
    //
    //   kHALF → T = half     (DVec<half>    : vec_size=8,  16 bytes per load)
    //   kFP8  → T = uint8_t  (DVec<uint8_t> : vec_size=16, 16 bytes per load)
    //
    // uint8_t is used for FP8 because it has the correct element size (1 byte) and is a plain
    // POD type, which avoids FP8 hardware requirements while keeping memcpy semantics correct.

#define LAUNCH_COMPACT_KV_KERNEL(T)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        switch (headDim)                                                                                               \
        {                                                                                                              \
        case 64:                                                                                                       \
            compactKVCacheBatchedKernel<T, 64><<<grid, block, 0, stream>>>(                                            \
                layerInfos, batchMappingPtr, srcKVLengthsPtr, maxBatchSize, oldActiveBatch);                           \
            break;                                                                                                     \
        case 128:                                                                                                      \
            compactKVCacheBatchedKernel<T, 128><<<grid, block, 0, stream>>>(                                           \
                layerInfos, batchMappingPtr, srcKVLengthsPtr, maxBatchSize, oldActiveBatch);                           \
            break;                                                                                                     \
        case 256:                                                                                                      \
            compactKVCacheBatchedKernel<T, 256><<<grid, block, 0, stream>>>(                                           \
                layerInfos, batchMappingPtr, srcKVLengthsPtr, maxBatchSize, oldActiveBatch);                           \
            break;                                                                                                     \
        case 512:                                                                                                      \
            compactKVCacheBatchedKernel<T, 512><<<grid, block, 0, stream>>>(                                           \
                layerInfos, batchMappingPtr, srcKVLengthsPtr, maxBatchSize, oldActiveBatch);                           \
            break;                                                                                                     \
        default:                                                                                                       \
            throw std::invalid_argument(                                                                               \
                format::fmtstr("compactKVCacheBatched: Unsupported headDim=%d. Only 64, 128, 256, or 512.", headDim)); \
        }                                                                                                              \
    } while (0)

    switch (kvCacheType)
    {
    case nvinfer1::DataType::kHALF: LAUNCH_COMPACT_KV_KERNEL(half); break;
    case nvinfer1::DataType::kFP8: LAUNCH_COMPACT_KV_KERNEL(uint8_t); break;
    default:
        throw std::invalid_argument(
            format::fmtstr("compactKVCacheBatched: Unsupported kvCacheType=%d. Only kHALF and kFP8 are supported.",
                static_cast<int>(kvCacheType)));
    }

#undef LAUNCH_COMPACT_KV_KERNEL

    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
