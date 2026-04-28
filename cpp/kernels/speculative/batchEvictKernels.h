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

#include <common/tensor.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Per-layer KV cache metadata for batched kernel operations.
struct KVLayerInfo
{
    void* data;         //!< Pointer to this layer's KV buffer [maxB, 2, H, S, D]
    int32_t numKVHeads; //!< Number of KV heads for this layer
    int32_t maxSeqLen;  //!< Max sequence length for this layer
};

/**
 * @brief Compact a single layer's KV cache by removing evicted batches.
 *
 * Single-layer variant of compactKVCache for per-layer heterogeneous KV cache.
 *
 * @param kvCacheLayer      [maxBatch, 2, numKVHeads, maxSeq, headDim] single-layer buffer (in/out)
 * @param batchMapping      [oldActiveBatch] GPU tensor, mapping[i] = newBatchIdx or -1 (evict)
 * @param kvCacheLengths    [maxBatch] GPU tensor of sequence lengths (const input)
 * @param dstKVCacheLengths [maxBatch] GPU tensor for compacted lengths (output, may alias kvCacheLengths)
 * @param oldActiveBatch    Number of batches before eviction
 * @param newActiveBatch    Number of batches after eviction
 * @param updateLengths     If true, update dstKVCacheLengths (only first layer should do this)
 * @param stream            CUDA stream
 */
void compactKVCacheSingleLayer(rt::Tensor& kvCacheLayer, rt::Tensor const& batchMapping,
    rt::Tensor const& kvCacheLengths, rt::Tensor& dstKVCacheLengths, int32_t oldActiveBatch, int32_t newActiveBatch,
    bool updateLengths, cudaStream_t stream);

/**
 * @brief Generic tensor compaction along batch dimension
 *
 * This kernel compacts a tensor by removing evicted batches.
 *
 * @param src               Source tensor (const input)
 * @param batchMapping      [oldActiveBatch] GPU tensor (const input), mapping[i] = newBatchIdx or -1
 * @param dst               Destination tensor (output, can be same as src for in-place operation)
 * @param oldActiveBatch    Number of batches before eviction
 * @param newActiveBatch    Number of batches after eviction
 * @param stream            CUDA stream
 *
 * @note Assumes batch dimension is the first dimension (dim 0)
 * @note For in-place operation, pass the same tensor as both src and dst
 * @throws std::runtime_error if tensors are not located on the GPU, or tensor shapes are invalid
 */
void compactTensorBatch(rt::Tensor const& src, rt::Tensor const& batchMapping, rt::Tensor& dst, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream);

/**
 * @brief Batched compaction across multiple layers in a single kernel launch.
 *
 * All layers in the batch must share the same headDim (template-selected).
 * Layers may have different numKVHeads and maxSeqLen.
 *
 * @param layerInfos      [numLayers] GPU array of KVLayerInfo
 * @param batchMapping    [oldActiveBatch] GPU tensor
 * @param kvCacheLengths  [maxBatch] GPU tensor of sequence lengths
 * @param numLayers       Number of layers in this batch
 * @param headDim         Head dimension (same for all layers in batch)
 * @param kvCacheType     KV cache storage dtype (kHALF or kFP8); controls element size for stride calculation
 * @param maxKVHeads      Maximum numKVHeads across all layers (for grid sizing)
 * @param maxBatchSize    Max batch size
 * @param oldActiveBatch  Batches before eviction
 * @param newActiveBatch  Batches after eviction
 * @param stream          CUDA stream
 */
void compactKVCacheBatched(KVLayerInfo const* layerInfos, rt::Tensor const& batchMapping,
    rt::Tensor const& kvCacheLengths, int32_t numLayers, int32_t headDim, nvinfer1::DataType kvCacheType,
    int32_t maxKVHeads, int32_t maxBatchSize, int32_t oldActiveBatch, int32_t newActiveBatch, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
