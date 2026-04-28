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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "embeddingKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cuda_fp16.h>
#if SUPPORTS_FP8
#include <cuda_fp8.h>
#endif

namespace trt_edgellm
{
namespace kernel
{

namespace
{

//! \brief FP16 embedding loader for template-based kernel
struct Fp16EmbeddingLoader
{
    static constexpr uint32_t vecSize = DVec<half>::vec_size;
    half const* table{nullptr};

    __device__ __forceinline__ void load(int32_t tokenId, int64_t hiddenSize, uint32_t offset, DVec<half>& out) const
    {
        out.load(table + static_cast<int64_t>(tokenId) * hiddenSize + offset);
    }
};

#if SUPPORTS_FP8
//! \brief FP8 embedding loader with per-group dequantization
struct Fp8EmbeddingLoader
{
    static constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
    __nv_fp8_e4m3 const* table{nullptr};
    float const* scales{nullptr};
    int64_t blockSize{0};
    int64_t nGroups{0};

    __device__ __forceinline__ void load(int32_t tokenId, int64_t hiddenSize, uint32_t offset, DVec<half>& out) const
    {
        // Determine which scale group this offset belongs to
        int64_t const group = static_cast<int64_t>(offset) / blockSize;
        float const scale = scales[static_cast<int64_t>(tokenId) * nGroups + group];

        // Load FP8 values and dequantize to FP16
        DVec<__nv_fp8_e4m3> in;
        in.load(table + static_cast<int64_t>(tokenId) * hiddenSize + offset);
#pragma unroll
        for (uint32_t i = 0; i < vecSize; ++i)
        {
            out[i] = __float2half(static_cast<float>(in[i]) * scale);
        }
    }
};
#endif

//! \brief Template-based embedding lookup kernel
//! \tparam TLoader Embedding loader type (Fp16EmbeddingLoader or Fp8EmbeddingLoader)
template <typename TLoader>
__global__ void embeddingLookupKernelImpl(int32_t const* inputIds, TLoader loader, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes vecSize elements via 128-bit loads (8 FP16 or 16 FP8 elements)
    constexpr uint32_t vecSize = TLoader::vecSize;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;

    // Get token ID and check bounds
    int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];
    bool const isValidToken = (tokenId >= 0 && tokenId < vocabSize);

    // Calculate base indices for this warp's work
    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isValidToken)
        {
            loader.load(tokenId, hiddenSize, offset, embeddingVec);
        }
        else
        {
            // Use zero embedding for out-of-bounds tokens
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

//! \brief Template-based embedding lookup kernel with legacy image insertion
//! For legacy multimodal models (Qwen2-VL, InternVL) where tokenId > vocabSize indicates image tokens
//! Text tokens use the template loader (FP16 or FP8), image tokens always use FP16 imageEmbeds
//! \tparam TLoader Embedding loader type (Fp16EmbeddingLoader or Fp8EmbeddingLoader)
template <typename TLoader>
__global__ void embeddingLookupWithImageInsertionKernelImpl(int32_t const* inputIds, TLoader loader,
    half const* imageEmbeds, half* output, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    int64_t imageTokenLen)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes vecSize elements
    constexpr uint32_t vecSize = TLoader::vecSize;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;

    // Get token ID
    int32_t const tokenId = inputIds[batchIdx * seqLen + tokenIdx];

    // Check if this is an image token (tokenId > vocabSize - 1)
    bool const isImageToken = tokenId > (vocabSize - 1);

    // Determine validity of token
    bool isValidTextToken = false;
    bool isValidImageToken = false;
    int32_t visualTokenId = 0;

    if (isImageToken)
    {
        visualTokenId = tokenId - vocabSize;
        isValidImageToken = (visualTokenId >= 0 && visualTokenId < imageTokenLen);
    }
    else
    {
        isValidTextToken = (tokenId >= 0 && tokenId < vocabSize);
    }

    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isValidTextToken)
        {
            // Use template loader for text tokens (supports FP16 or FP8)
            loader.load(tokenId, hiddenSize, offset, embeddingVec);
        }
        else if (isValidImageToken)
        {
            // Load from FP16 imageEmbeds directly
            uint32_t const embeddingOffset = visualTokenId * hiddenSize + offset;
            embeddingVec.load(imageEmbeds + embeddingOffset);
        }
        else
        {
            // Use zero embedding for out-of-bounds tokens
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// Template helper function to launch embedding lookup kernel
template <typename TLoader>
void launchEmbeddingLookupKernel(int32_t const* inputIds, TLoader const& loader, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, cudaStream_t stream)
{
    constexpr uint32_t vecSize = TLoader::vecSize;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupKernelImpl<<<gridSize, threadsPerBlock, 0, stream>>>(
        inputIds, loader, output, batchSize, seqLen, vocabSize, hiddenSize);
}

// Template helper function to launch embedding lookup with image insertion kernel
template <typename TLoader>
void launchEmbeddingLookupWithImageInsertionKernel(int32_t const* inputIds, TLoader const& loader,
    half const* imageEmbeds, half* output, int64_t batchSize, int64_t seqLen, int32_t vocabSize, int64_t hiddenSize,
    int64_t imageTokenLen, cudaStream_t stream)
{
    constexpr uint32_t vecSize = TLoader::vecSize;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupWithImageInsertionKernelImpl<<<gridSize, threadsPerBlock, 0, stream>>>(
        inputIds, loader, imageEmbeds, output, batchSize, seqLen, vocabSize, hiddenSize, imageTokenLen);
}

// CUDA kernel for assembling deepstack embeddings (FP16 only)
// Extracts image token embeddings from deepstack features based on token IDs
// Token IDs >= vocabSize are mapped to deepstack features, others get zero embeddings
__global__ void assembleDeepstackEmbeddingKernel(int32_t const* inputIds, half const* deepstackFeatures, half* output,
    int64_t batchSize, int64_t seqLen, int32_t vocabSize, int32_t imageTokenId, int64_t hiddenSize,
    int64_t numImageTokens, int32_t const* multimodalIndices = nullptr)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Get token ID (warpId == batchIdx * seqLen + tokenIdx)
    int64_t const pos = static_cast<int64_t>(warpId);
    int32_t const tokenId = inputIds[pos];

    // Determine if this is an image/multimodal token:
    // - Legacy path: tokenId >= vocabSize (Qwen2.5-VL where image tokens start at vocabSize)
    // - Explicit path: tokenId == imageTokenId (Qwen3-Omni where image tokens are within vocab)
    bool const isImageToken = (tokenId >= vocabSize) || (imageTokenId > 0 && tokenId == imageTokenId);

    // Calculate base indices for this warp's work
    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isImageToken)
        {
            // Calculate the index into deepstackFeatures:
            // - If multimodalIndices is provided, use it (Qwen3-Omni: all image tokens share same ID)
            // - Otherwise, fall back to tokenId - vocabSize (Qwen2.5-VL legacy)
            int32_t deepstackIdx;
            if (multimodalIndices != nullptr)
            {
                deepstackIdx = multimodalIndices[pos];
            }
            else
            {
                deepstackIdx = tokenId - vocabSize;
            }

            // Validate that deepstackIdx is within bounds
            if (deepstackIdx >= 0 && deepstackIdx < numImageTokens)
            {
                // Load embedding data from deepstack features
                uint32_t const embeddingOffset = deepstackIdx * hiddenSize + offset;
                embeddingVec.load(deepstackFeatures + embeddingOffset);
            }
            else
            {
                // Out-of-bounds image token, use zero embedding
#pragma unroll
                for (uint32_t i = 0; i < vecSize; ++i)
                {
                    embeddingVec[i] = __float2half(0.0f);
                }
            }
        }
        else
        {
            // Not an image token, use zero embedding
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

//! \brief Template-based multimodal embedding lookup kernel for Qwen3-Omni
//! Text tokens use the template loader (FP16 or FP8), image/audio tokens use FP16 embeds
//! \tparam TLoader Embedding loader type (Fp16EmbeddingLoader or Fp8EmbeddingLoader)
template <typename TLoader>
__global__ void embeddingLookupMultimodalKernelImpl(int32_t const* inputIds, TLoader loader,
    int32_t const* multimodalIndices, int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen,
    int32_t audioTokenId, half const* audioEmbeds, int64_t audioTokenLen, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize)
{
    // Each warp handles one hidden state (one token's embedding)
    constexpr uint32_t vecSize = TLoader::vecSize;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Calculate token indices
    uint32_t const batchIdx = warpId / seqLen;
    uint32_t const tokenIdx = warpId % seqLen;
    uint32_t const linearIdx = batchIdx * seqLen + tokenIdx;

    // Get token ID
    int32_t const tokenId = inputIds[linearIdx];

    // Determine token type
    bool isImageToken = (imageEmbeds != nullptr && tokenId == imageTokenId);
    bool isAudioToken = (audioEmbeds != nullptr && tokenId == audioTokenId);
    bool isValidTextToken = false;
    bool isValidImageToken = false;
    bool isValidAudioToken = false;
    int32_t multimodalIdx = 0;

    if (isImageToken)
    {
        multimodalIdx = multimodalIndices[linearIdx];
        isValidImageToken = (multimodalIdx >= 0 && multimodalIdx < imageTokenLen);
    }
    else if (isAudioToken)
    {
        multimodalIdx = multimodalIndices[linearIdx];
        isValidAudioToken = (multimodalIdx >= 0 && multimodalIdx < audioTokenLen);
    }
    else
    {
        isValidTextToken = (tokenId >= 0 && tokenId < vocabSize);
    }

    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isValidTextToken)
        {
            // Use template loader for text tokens (supports FP16 or FP8)
            loader.load(tokenId, hiddenSize, offset, embeddingVec);
        }
        else if (isValidImageToken)
        {
            // Load from FP16 imageEmbeds directly
            uint32_t const embeddingOffset = multimodalIdx * hiddenSize + offset;
            embeddingVec.load(imageEmbeds + embeddingOffset);
        }
        else if (isValidAudioToken)
        {
            // Load from FP16 audioEmbeds directly
            uint32_t const embeddingOffset = multimodalIdx * hiddenSize + offset;
            embeddingVec.load(audioEmbeds + embeddingOffset);
        }
        else
        {
            // Use zero embedding for out-of-bounds tokens
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

// Template helper function to launch multimodal embedding lookup kernel
template <typename TLoader>
void launchEmbeddingLookupMultimodalKernel(int32_t const* inputIds, TLoader const& loader,
    int32_t const* multimodalIndices, int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen,
    int32_t audioTokenId, half const* audioEmbeds, int64_t audioTokenLen, half* output, int64_t batchSize,
    int64_t seqLen, int32_t vocabSize, int64_t hiddenSize, cudaStream_t stream)
{
    constexpr uint32_t vecSize = TLoader::vecSize;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    embeddingLookupMultimodalKernelImpl<<<gridSize, threadsPerBlock, 0, stream>>>(inputIds, loader, multimodalIndices,
        imageTokenId, imageEmbeds, imageTokenLen, audioTokenId, audioEmbeds, audioTokenLen, output, batchSize, seqLen,
        vocabSize, hiddenSize);
}

} // namespace

void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    rt::Tensor& output, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int32_t const vocabSize = static_cast<int32_t>(embeddingShape[0]);
    int64_t const hiddenSize = embeddingShape[1];

    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate common data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half* outputPtr = output.dataPointer<half>();

    // Dispatch based on embedding table datatype
    bool const isFP8Embedding = (embeddingTable.getDataType() == nvinfer1::DataType::kFP8);
    if (isFP8Embedding)
    {
#if SUPPORTS_FP8
        constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
        check::check(scales.has_value(), "scales must be provided for FP8 embedding table");
        auto const& scalesTensor = scales.value().get();
        auto const scaleShape = scalesTensor.getShape();

        check::check(scaleShape.getNumDims() == 2, "scales must be 2D tensor [vocabSize, hiddenSize / blockSize]");
        check::check(scaleShape[0] == vocabSize, "Scale vocab size must match embeddingTable vocab size");
        check::check(scaleShape[1] > 0, "Scale second dimension must be positive");
        check::check(hiddenSize % scaleShape[1] == 0, "hiddenSize must be divisible by number of scale groups");
        check::check(scalesTensor.getDataType() == nvinfer1::DataType::kFLOAT, "scales must be FP32");

        int64_t const nGroups = scaleShape[1];
        int64_t const blockSize = hiddenSize / nGroups;
        check::check(hiddenSize % vecSize == 0,
            format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));
        check::check(blockSize % vecSize == 0,
            format::fmtstr("blockSize must be a multiple of %d for efficient vectorized FP8 dequantization", vecSize));

        __nv_fp8_e4m3 const* tablePtr = embeddingTable.dataPointer<__nv_fp8_e4m3>();
        float const* scalePtr = scalesTensor.dataPointer<float>();

        Fp8EmbeddingLoader loader{tablePtr, scalePtr, blockSize, nGroups};
        launchEmbeddingLookupKernel(inputIdsPtr, loader, outputPtr, batchSize, seqLen, vocabSize, hiddenSize, stream);
#else
        check::check(false, "FP8 embedding lookup is unavailable: build does not support FP8 (SUPPORTS_FP8=0)");
#endif
    }
    else
    {
        check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16 or FP8");
        half const* embeddingTablePtr = embeddingTable.dataPointer<half>();

        Fp16EmbeddingLoader loader{embeddingTablePtr};
        launchEmbeddingLookupKernel(inputIdsPtr, loader, outputPtr, batchSize, seqLen, vocabSize, hiddenSize, stream);
    }
}

void embeddingLookupWithImageInsertion(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::OptionalInputTensor scales, rt::Tensor const& imageEmbeds, rt::Tensor& output, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const imageShape = imageEmbeds.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int32_t const vocabSize = static_cast<int32_t>(embeddingShape[0]);
    int64_t const hiddenSize = embeddingShape[1];
    int64_t const imageTokenLen = imageShape[0];

    check::check(embeddingShape[1] == imageShape[1], "Hidden size mismatch between embeddingTable and imageEmbeds");
    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate common data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(imageEmbeds.getDataType() == nvinfer1::DataType::kHALF, "imageEmbeds must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* imageEmbedsPtr = imageEmbeds.dataPointer<half>();
    half* outputPtr = output.dataPointer<half>();

    // Dispatch based on embedding table datatype
    bool const isFP8Embedding = (embeddingTable.getDataType() == nvinfer1::DataType::kFP8);
    if (isFP8Embedding)
    {
#if SUPPORTS_FP8
        constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
        check::check(scales.has_value(), "scales must be provided for FP8 embedding table");
        auto const& scalesTensor = scales.value().get();
        auto const scaleShape = scalesTensor.getShape();

        check::check(scaleShape.getNumDims() == 2, "scales must be 2D tensor [vocabSize, hiddenSize / blockSize]");
        check::check(scaleShape[0] == vocabSize, "Scale vocab size must match embeddingTable vocab size");
        check::check(scaleShape[1] > 0, "Scale second dimension must be positive");
        check::check(hiddenSize % scaleShape[1] == 0, "hiddenSize must be divisible by number of scale groups");
        check::check(scalesTensor.getDataType() == nvinfer1::DataType::kFLOAT, "scales must be FP32");

        int64_t const nGroups = scaleShape[1];
        int64_t const blockSize = hiddenSize / nGroups;
        check::check(hiddenSize % vecSize == 0,
            format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));
        check::check(blockSize % vecSize == 0,
            format::fmtstr("blockSize must be a multiple of %d for efficient vectorized FP8 dequantization", vecSize));

        __nv_fp8_e4m3 const* tablePtr = embeddingTable.dataPointer<__nv_fp8_e4m3>();
        float const* scalePtr = scalesTensor.dataPointer<float>();

        Fp8EmbeddingLoader loader{tablePtr, scalePtr, blockSize, nGroups};
        launchEmbeddingLookupWithImageInsertionKernel(inputIdsPtr, loader, imageEmbedsPtr, outputPtr, batchSize, seqLen,
            vocabSize, hiddenSize, imageTokenLen, stream);
#else
        check::check(false,
            "FP8 embedding lookup with image insertion is unavailable: build does not support FP8 (SUPPORTS_FP8=0)");
#endif
    }
    else
    {
        check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16 or FP8");
        half const* embeddingTablePtr = embeddingTable.dataPointer<half>();

        Fp16EmbeddingLoader loader{embeddingTablePtr};
        launchEmbeddingLookupWithImageInsertionKernel(inputIdsPtr, loader, imageEmbedsPtr, outputPtr, batchSize, seqLen,
            vocabSize, hiddenSize, imageTokenLen, stream);
    }
}

void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures, int32_t vocabSize,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream, int32_t imageTokenId, rt::OptionalInputTensor multimodalIndices)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const featuresShape = deepstackFeatures.getShape();
    auto const outputShape = deepstackEmbeds.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(featuresShape.getNumDims() == 2, "deepstackFeatures must be 2D tensor [numImageTokens, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "deepstackEmbeds must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const numImageTokens = featuresShape[0];
    int64_t const hiddenSize = featuresShape[1];

    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(deepstackFeatures.getDataType() == nvinfer1::DataType::kHALF, "deepstackFeatures must be FP16");
    check::check(deepstackEmbeds.getDataType() == nvinfer1::DataType::kHALF, "deepstackEmbeds must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* deepstackFeaturesPtr = deepstackFeatures.dataPointer<half>();
    half* outputPtr = deepstackEmbeds.dataPointer<half>();

    // Multimodal indices (optional, for Qwen3-Omni where image tokens share same ID)
    int32_t const* multimodalIndicesPtr = nullptr;
    if (multimodalIndices.has_value())
    {
        multimodalIndicesPtr = multimodalIndices.value().get().dataPointer<int32_t>();
    }

    // Launch kernel
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    assembleDeepstackEmbeddingKernel<<<gridSize, threadsPerBlock, 0, stream>>>(inputIdsPtr, deepstackFeaturesPtr,
        outputPtr, batchSize, seqLen, vocabSize, imageTokenId, hiddenSize, numImageTokens, multimodalIndicesPtr);
}

void embeddingLookupMultimodal(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::OptionalInputTensor scales, rt::OptionalInputTensor multimodalIndices, std::optional<int32_t> imageTokenId,
    rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId, rt::OptionalInputTensor audioEmbeds,
    rt::Tensor& output, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const embeddingShape = embeddingTable.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(embeddingShape.getNumDims() == 2, "embeddingTable must be 2D tensor [vocabSize, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "output must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int32_t const vocabSize = static_cast<int32_t>(embeddingShape[0]);
    int64_t const hiddenSize = embeddingShape[1];

    // Validate output shape
    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate common data types for required inputs
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Handle optional image parameters
    bool const hasImage = imageTokenId.has_value() && imageEmbeds.has_value();
    if (hasImage)
    {
        auto const imageShape = imageEmbeds->get().getShape();
        check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
        check::check(imageShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and imageEmbeds");
        check::check(imageEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "imageEmbeds must be FP16");
    }

    // Handle optional audio parameters
    bool const hasAudio = audioTokenId.has_value() && audioEmbeds.has_value();
    if (hasAudio)
    {
        auto const audioShape = audioEmbeds->get().getShape();
        check::check(audioShape.getNumDims() == 2, "audioEmbeds must be 2D tensor [audioTokenLen, hiddenSize]");
        check::check(audioShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and audioEmbeds");
        check::check(audioEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "audioEmbeds must be FP16");
    }

    // Validate that imageTokenId and audioTokenId are different when both are present
    if (hasImage && hasAudio)
    {
        check::check(*imageTokenId != *audioTokenId, "imageTokenId and audioTokenId must be different");
    }

    // Validate multimodalIndices if any multimodal input is present
    bool const hasMultimodal = hasImage || hasAudio;
    if (hasMultimodal)
    {
        check::check(
            multimodalIndices.has_value(), "multimodalIndices is required when image or audio inputs are provided");
        auto const multimodalIndicesShape = multimodalIndices->get().getShape();
        check::check(
            multimodalIndicesShape.getNumDims() == 2, "multimodalIndices must be 2D tensor [batchSize, seqLen]");
        check::check(multimodalIndicesShape[0] == batchSize, "multimodalIndices batch size mismatch");
        check::check(multimodalIndicesShape[1] == seqLen, "multimodalIndices sequence length mismatch");
        check::check(
            multimodalIndices->get().getDataType() == nvinfer1::DataType::kINT32, "multimodalIndices must be INT32");
    }

    // Extract values for kernel - use safe defaults when modalities are absent
    int32_t const imageTokenIdValue = hasImage ? *imageTokenId : -1;
    half const* imageEmbedsPtr = hasImage ? imageEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const imageTokenLen = hasImage ? imageEmbeds->get().getShape()[0] : 0;

    int32_t const audioTokenIdValue = hasAudio ? *audioTokenId : -1;
    half const* audioEmbedsPtr = hasAudio ? audioEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const audioTokenLen = hasAudio ? audioEmbeds->get().getShape()[0] : 0;

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    int32_t const* multimodalIndicesPtr = hasMultimodal ? multimodalIndices->get().dataPointer<int32_t>() : nullptr;
    half* outputPtr = output.dataPointer<half>();

    // Dispatch based on embedding table datatype
    bool const isFP8Embedding = (embeddingTable.getDataType() == nvinfer1::DataType::kFP8);
    if (isFP8Embedding)
    {
#if SUPPORTS_FP8
        constexpr uint32_t vecSize = DVec<__nv_fp8_e4m3>::vec_size;
        check::check(scales.has_value(), "scales must be provided for FP8 embedding table");
        auto const& scalesTensor = scales.value().get();
        auto const scaleShape = scalesTensor.getShape();

        check::check(scaleShape.getNumDims() == 2, "scales must be 2D tensor [vocabSize, hiddenSize / blockSize]");
        check::check(scaleShape[0] == vocabSize, "Scale vocab size must match embeddingTable vocab size");
        check::check(scaleShape[1] > 0, "Scale second dimension must be positive");
        check::check(hiddenSize % scaleShape[1] == 0, "hiddenSize must be divisible by number of scale groups");
        check::check(scalesTensor.getDataType() == nvinfer1::DataType::kFLOAT, "scales must be FP32");

        int64_t const nGroups = scaleShape[1];
        int64_t const blockSize = hiddenSize / nGroups;
        check::check(hiddenSize % vecSize == 0,
            format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));
        check::check(blockSize % vecSize == 0,
            format::fmtstr("blockSize must be a multiple of %d for efficient vectorized FP8 dequantization", vecSize));

        __nv_fp8_e4m3 const* tablePtr = embeddingTable.dataPointer<__nv_fp8_e4m3>();
        float const* scalePtr = scalesTensor.dataPointer<float>();

        Fp8EmbeddingLoader loader{tablePtr, scalePtr, blockSize, nGroups};
        launchEmbeddingLookupMultimodalKernel(inputIdsPtr, loader, multimodalIndicesPtr, imageTokenIdValue,
            imageEmbedsPtr, imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize,
            seqLen, vocabSize, hiddenSize, stream);
#else
        check::check(
            false, "FP8 multimodal embedding lookup is unavailable: build does not support FP8 (SUPPORTS_FP8=0)");
#endif
    }
    else
    {
        check::check(embeddingTable.getDataType() == nvinfer1::DataType::kHALF, "embeddingTable must be FP16 or FP8");
        half const* embeddingTablePtr = embeddingTable.dataPointer<half>();

        Fp16EmbeddingLoader loader{embeddingTablePtr};
        launchEmbeddingLookupMultimodalKernel(inputIdsPtr, loader, multimodalIndicesPtr, imageTokenIdValue,
            imageEmbedsPtr, imageTokenLen, audioTokenIdValue, audioEmbedsPtr, audioTokenLen, outputPtr, batchSize,
            seqLen, vocabSize, hiddenSize, stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
