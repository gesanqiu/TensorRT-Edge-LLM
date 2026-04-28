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

#include "common/cudaUtils.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "testUtils.h"
#include <chrono>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

// ============================================================================
// Helper Functions
// ============================================================================

// Flush L2 cache by reading/writing large buffer
void flushL2Cache()
{
    // 40MB L2 cache size
    constexpr size_t L2_SIZE = 40 * 1024 * 1024;
    static void* flushBuffer = nullptr;

    if (flushBuffer == nullptr)
    {
        CUDA_CHECK(cudaMalloc(&flushBuffer, L2_SIZE));
    }

    // Read and write to flush cache
    CUDA_CHECK(cudaMemset(flushBuffer, 0, L2_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Initialize a single KV cache layer with distinct FP16 values for testing
// Layout: [maxBatchSize, 2, numKVHeads, maxSeqLen, headDim]
void initializeKVCacheLayer(std::vector<half>& kvCache, int32_t layerIdx, int32_t maxBatchSize, int32_t numKVHeads,
    int32_t maxSeqLen, int32_t headDim, std::vector<int32_t> const& kvLengths)
{
    size_t totalSize = static_cast<size_t>(maxBatchSize) * 2 * numKVHeads * maxSeqLen * headDim;
    kvCache.resize(totalSize);

    // Use simple positive integers for valid positions
    // -1.0 is reserved for invalid positions
    size_t batchStride = static_cast<size_t>(2) * numKVHeads * maxSeqLen * headDim;
    size_t seqStride = static_cast<size_t>(headDim);

    for (size_t idx = 0; idx < totalSize; ++idx)
    {
        int32_t b = static_cast<int32_t>((idx / batchStride) % maxBatchSize);
        int32_t seqLen = (b < static_cast<int32_t>(kvLengths.size())) ? kvLengths[b] : 0;

        size_t idxInBatch = idx % batchStride;
        int32_t s = static_cast<int32_t>((idxInBatch / seqStride) % maxSeqLen);

        if (s < seqLen)
        {
            // Incorporate layerIdx into the value to make each layer distinct
            int32_t value = static_cast<int32_t>((static_cast<size_t>(layerIdx) * totalSize + idx) % 1000);
            kvCache[idx] = __float2half(static_cast<float>(value));
        }
        else
        {
            // Mark invalid positions with -1.0
            kvCache[idx] = __float2half(-1.0f);
        }
    }
}

// Verify a single KV cache layer after compaction
bool verifyKVCacheLayer(std::vector<half> const& kvCache, int32_t layerIdx, int32_t maxBatchSize, int32_t numKVHeads,
    int32_t maxSeqLen, int32_t headDim, std::vector<int32_t> const& batchMapping,
    std::vector<int32_t> const& oldKVLengths, std::vector<int32_t> const& newKVLengths)
{
    size_t batchStride = static_cast<size_t>(2) * numKVHeads * maxSeqLen * headDim;
    size_t layerTotalSize = static_cast<size_t>(maxBatchSize) * batchStride;

    for (int32_t oldIdx = 0; oldIdx < static_cast<int32_t>(batchMapping.size()); ++oldIdx)
    {
        int32_t newIdx = batchMapping[oldIdx];
        if (newIdx < 0)
        {
            continue; // Evicted batch
        }

        int32_t seqLen = oldKVLengths[oldIdx];
        if (seqLen == 0)
        {
            continue;
        }

        // Verify KV cache lengths match
        if (newKVLengths[newIdx] != seqLen)
        {
            std::cerr << "Length mismatch: newIdx=" << newIdx << " expected=" << seqLen
                      << " got=" << newKVLengths[newIdx] << std::endl;
            return false;
        }

        // Verify data using linear index - iterate through all positions in this batch
        size_t oldBatchStart = static_cast<size_t>(oldIdx) * batchStride;
        size_t newBatchStart = static_cast<size_t>(newIdx) * batchStride;
        size_t seqStride = static_cast<size_t>(headDim);

        for (size_t offset = 0; offset < batchStride; ++offset)
        {
            // Extract sequence position to check validity
            size_t idxInBatch = offset;
            int32_t s = static_cast<int32_t>((idxInBatch / seqStride) % maxSeqLen);

            if (s >= seqLen)
            {
                continue;
            }

            size_t oldLinearIdx = oldBatchStart + offset;
            size_t newLinearIdx = newBatchStart + offset;

            // Expected value based on original linear index (incorporating layerIdx)
            int32_t expectedIntValue
                = static_cast<int32_t>((static_cast<size_t>(layerIdx) * layerTotalSize + oldLinearIdx) % 1000);
            float expectedValue = static_cast<float>(expectedIntValue);
            float actualValue = __half2float(kvCache[newLinearIdx]);

            // For integer values in FP16 range, we expect exact match
            if (actualValue != expectedValue)
            {
                // Extract detailed position for error reporting
                int32_t kv = static_cast<int32_t>(offset / (numKVHeads * maxSeqLen * headDim));
                int32_t h = static_cast<int32_t>((offset / (maxSeqLen * headDim)) % numKVHeads);
                int32_t d = static_cast<int32_t>(offset % headDim);

                std::cerr << "Data mismatch at layer=" << layerIdx << " newIdx=" << newIdx << " oldIdx=" << oldIdx
                          << " kv=" << kv << " head=" << h << " seq=" << s << " dim=" << d
                          << " expected=" << expectedValue << " got=" << actualValue
                          << " (oldLinearIdx=" << oldLinearIdx << ")" << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Helper to compact KV cache across multiple layers using the single-layer API
void compactKVCacheAllLayers(std::vector<rt::Tensor>& kvCacheLayers, rt::Tensor const& batchMapping,
    rt::Tensor& kvCacheLengths, rt::Tensor& dstKVCacheLengths, int32_t oldActiveBatch, int32_t newActiveBatch,
    cudaStream_t stream)
{
    // Compact KV data per-layer without updating lengths (in-place length update would
    // corrupt source lengths that subsequent layers still read).
    for (int32_t l = 0; l < static_cast<int32_t>(kvCacheLayers.size()); ++l)
    {
        compactKVCacheSingleLayer(kvCacheLayers[l], batchMapping, kvCacheLengths, dstKVCacheLengths, oldActiveBatch,
            newActiveBatch, /*updateLengths=*/false, stream);
    }
    // Compact lengths separately after all layers are done.
    compactTensorBatch(kvCacheLengths, batchMapping, dstKVCacheLengths, oldActiveBatch, newActiveBatch, stream);
}

// ============================================================================
// Test 1: compactKVCacheSingleLayer - Basic Functionality
// ============================================================================
TEST(BatchEvictKernels, CompactKVCacheBasicEviction)
{
    cudaStream_t stream = nullptr;

    // Test configuration
    int32_t numLayers = 2;
    int32_t maxBatchSize = 4;
    int32_t numKVHeads = 4;
    int32_t maxSeqLen = 128;
    int32_t headDim = 64;

    // Test case: [0, -1, 1, 2] - batch 1 is evicted
    int32_t oldActiveBatch = 4;
    int32_t newActiveBatch = 3;
    std::vector<int32_t> batchMapping = {0, -1, 1, 2};
    std::vector<int32_t> oldKVLengths = {64, 32, 48, 80};

    // Initialize per-layer KV caches
    std::vector<std::vector<half>> kvCacheHostLayers(numLayers);
    std::vector<rt::Tensor> kvCacheDeviceLayers;
    kvCacheDeviceLayers.reserve(numLayers);

    for (int32_t l = 0; l < numLayers; ++l)
    {
        initializeKVCacheLayer(kvCacheHostLayers[l], l, maxBatchSize, numKVHeads, maxSeqLen, headDim, oldKVLengths);
        kvCacheDeviceLayers.emplace_back(
            rt::Tensor({maxBatchSize, 2, numKVHeads, maxSeqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF));
        copyHostToDevice(kvCacheDeviceLayers[l], kvCacheHostLayers[l]);
    }

    // Create GPU tensors for lengths and mapping
    rt::Tensor kvLengthsDevice({maxBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(kvLengthsDevice, oldKVLengths);
    copyHostToDevice(batchMappingDevice, batchMapping);

    // Execute kernel across all layers
    compactKVCacheAllLayers(kvCacheDeviceLayers, batchMappingDevice, kvLengthsDevice, kvLengthsDevice, oldActiveBatch,
        newActiveBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back results and verify
    auto const newKVLengths = copyDeviceToHost<int32_t>(kvLengthsDevice);
    for (int32_t l = 0; l < numLayers; ++l)
    {
        auto const kvCacheHost = copyDeviceToHost<half>(kvCacheDeviceLayers[l]);
        EXPECT_TRUE(verifyKVCacheLayer(
            kvCacheHost, l, maxBatchSize, numKVHeads, maxSeqLen, headDim, batchMapping, oldKVLengths, newKVLengths));
    }
}

// ============================================================================
// Test 2: compactKVCacheSingleLayer - Multiple Evictions
// ============================================================================
TEST(BatchEvictKernels, CompactKVCacheMultipleEvictions)
{
    cudaStream_t stream = nullptr;

    int32_t numLayers = 2;
    int32_t maxBatchSize = 8;
    int32_t numKVHeads = 8;
    int32_t maxSeqLen = 256;
    int32_t headDim = 128;

    // Test case: evict batches 1, 3, 5 - mapping [0, -1, 1, -1, 2, -1, 3, 4]
    int32_t oldActiveBatch = 8;
    int32_t newActiveBatch = 5;
    std::vector<int32_t> batchMapping = {0, -1, 1, -1, 2, -1, 3, 4};
    std::vector<int32_t> oldKVLengths = {64, 32, 48, 80, 96, 72, 120, 88};

    // Initialize per-layer KV caches
    std::vector<std::vector<half>> kvCacheHostLayers(numLayers);
    std::vector<rt::Tensor> kvCacheDeviceLayers;
    kvCacheDeviceLayers.reserve(numLayers);

    for (int32_t l = 0; l < numLayers; ++l)
    {
        initializeKVCacheLayer(kvCacheHostLayers[l], l, maxBatchSize, numKVHeads, maxSeqLen, headDim, oldKVLengths);
        kvCacheDeviceLayers.emplace_back(
            rt::Tensor({maxBatchSize, 2, numKVHeads, maxSeqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF));
        copyHostToDevice(kvCacheDeviceLayers[l], kvCacheHostLayers[l]);
    }

    rt::Tensor kvLengthsDevice({maxBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(kvLengthsDevice, oldKVLengths);
    copyHostToDevice(batchMappingDevice, batchMapping);

    // Execute
    compactKVCacheAllLayers(kvCacheDeviceLayers, batchMappingDevice, kvLengthsDevice, kvLengthsDevice, oldActiveBatch,
        newActiveBatch, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify
    auto const newKVLengths = copyDeviceToHost<int32_t>(kvLengthsDevice);
    for (int32_t l = 0; l < numLayers; ++l)
    {
        auto const kvCacheHost = copyDeviceToHost<half>(kvCacheDeviceLayers[l]);
        EXPECT_TRUE(verifyKVCacheLayer(
            kvCacheHost, l, maxBatchSize, numKVHeads, maxSeqLen, headDim, batchMapping, oldKVLengths, newKVLengths));
    }
}

// ============================================================================
// Test 3: compactKVCacheSingleLayer - Edge Cases
// ============================================================================
TEST(BatchEvictKernels, CompactKVCacheSmallTestCases)
{
    cudaStream_t stream = nullptr;
    int32_t numLayers = 1;
    int32_t maxBatchSize = 4;
    int32_t numKVHeads = 2;
    int32_t maxSeqLen = 64;
    int32_t headDim = 64;

    // Test 1: No eviction (all batches stay in place)
    {
        std::vector<int32_t> batchMapping = {0, 1, 2, 3};
        std::vector<int32_t> oldKVLengths = {32, 48, 24, 56};

        std::vector<half> kvCacheHost;
        initializeKVCacheLayer(kvCacheHost, 0, maxBatchSize, numKVHeads, maxSeqLen, headDim, oldKVLengths);
        std::vector<half> kvCacheOriginal = kvCacheHost; // Backup

        rt::Tensor kvCacheDevice(
            {maxBatchSize, 2, numKVHeads, maxSeqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor kvLengthsDevice({maxBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor batchMappingDevice({4}, rt::DeviceType::kGPU, DataType::kINT32);

        copyHostToDevice(kvCacheDevice, kvCacheHost);
        copyHostToDevice(kvLengthsDevice, oldKVLengths);
        copyHostToDevice(batchMappingDevice, batchMapping);

        compactKVCacheSingleLayer(
            kvCacheDevice, batchMappingDevice, kvLengthsDevice, kvLengthsDevice, 4, 4, true, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify data unchanged
        kvCacheHost = copyDeviceToHost<half>(kvCacheDevice);

        // Data should be identical
        for (size_t i = 0; i < kvCacheHost.size(); ++i)
        {
            EXPECT_EQ(__half2float(kvCacheHost[i]), __half2float(kvCacheOriginal[i]));
        }
    }

    // Test 2: All batches evicted except one
    {
        std::vector<int32_t> batchMapping = {-1, -1, 0, -1};
        std::vector<int32_t> oldKVLengths = {32, 48, 24, 56};

        std::vector<half> kvCacheHost;
        initializeKVCacheLayer(kvCacheHost, 0, maxBatchSize, numKVHeads, maxSeqLen, headDim, oldKVLengths);

        rt::Tensor kvCacheDevice(
            {maxBatchSize, 2, numKVHeads, maxSeqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor kvLengthsDevice({maxBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor batchMappingDevice({4}, rt::DeviceType::kGPU, DataType::kINT32);

        copyHostToDevice(kvCacheDevice, kvCacheHost);
        copyHostToDevice(kvLengthsDevice, oldKVLengths);
        copyHostToDevice(batchMappingDevice, batchMapping);

        compactKVCacheSingleLayer(
            kvCacheDevice, batchMappingDevice, kvLengthsDevice, kvLengthsDevice, 4, 1, true, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        kvCacheHost = copyDeviceToHost<half>(kvCacheDevice);
        auto const newKVLengths = copyDeviceToHost<int32_t>(kvLengthsDevice);

        EXPECT_TRUE(verifyKVCacheLayer(
            kvCacheHost, 0, maxBatchSize, numKVHeads, maxSeqLen, headDim, batchMapping, oldKVLengths, newKVLengths));
    }
}

// ============================================================================
// Test 4: compactTensorBatch - In-place operation for all types
// ============================================================================
TEST(BatchEvictKernels, CompactTensorBatchInPlace)
{
    cudaStream_t stream = nullptr;

    int32_t oldActiveBatch = 6;
    int32_t newActiveBatch = 4;
    int32_t dim1 = 16;
    int32_t dim2 = 32;

    // Batch mapping: evict batches 1 and 4
    std::vector<int32_t> batchMapping = {0, -1, 1, 2, -1, 3};

    // Prepare shared batchMapping on GPU (used by all sub-tests)
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(batchMappingDevice, batchMapping);

    // Test 1: half type (common for hidden states, logits)
    {
        std::vector<half> srcData(oldActiveBatch * dim1 * dim2);
        for (int32_t b = 0; b < oldActiveBatch; ++b)
        {
            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                srcData[b * dim1 * dim2 + i] = __float2half(static_cast<float>(b * 100 + i) + 0.5f);
            }
        }
        std::vector<half> srcDataBackup = srcData;

        rt::Tensor tensorDevice({oldActiveBatch, dim1, dim2}, rt::DeviceType::kGPU, DataType::kHALF);

        copyHostToDevice(tensorDevice, srcData);

        // In-place compaction
        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const resultData = copyDeviceToHost<half>(tensorDevice);

        // Verify
        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t newIdx = batchMapping[oldIdx];
            if (newIdx < 0)
            {
                continue;
            }

            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                float expected = __half2float(srcDataBackup[oldIdx * dim1 * dim2 + i]);
                float actual = __half2float(resultData[newIdx * dim1 * dim2 + i]);
                EXPECT_NEAR(actual, expected, 1e-2f); // half precision tolerance
            }
        }
    }

    // Test 2: float type (common for RoPE cache, scores)
    {
        std::vector<float> srcData(oldActiveBatch * dim1 * dim2);
        for (int32_t b = 0; b < oldActiveBatch; ++b)
        {
            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                srcData[b * dim1 * dim2 + i] = static_cast<float>(b * 1000 + i) + 0.123f;
            }
        }
        std::vector<float> srcDataBackup = srcData;

        rt::Tensor tensorDevice({oldActiveBatch, dim1, dim2}, rt::DeviceType::kGPU, DataType::kFLOAT);

        copyHostToDevice(tensorDevice, srcData);

        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const resultData = copyDeviceToHost<float>(tensorDevice);

        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t newIdx = batchMapping[oldIdx];
            if (newIdx < 0)
            {
                continue;
            }

            for (int32_t i = 0; i < dim1 * dim2; ++i)
            {
                EXPECT_FLOAT_EQ(resultData[newIdx * dim1 * dim2 + i], srcDataBackup[oldIdx * dim1 * dim2 + i]);
            }
        }
    }

    // Test 3: int32_t type (common for token IDs, indices)
    {
        int32_t dim1_int = 64; // Typical for token sequences
        std::vector<int32_t> srcData(oldActiveBatch * dim1_int);
        for (int32_t b = 0; b < oldActiveBatch; ++b)
        {
            for (int32_t i = 0; i < dim1_int; ++i)
            {
                srcData[b * dim1_int + i] = b * 10000 + i;
            }
        }
        std::vector<int32_t> srcDataBackup = srcData;

        rt::Tensor tensorDevice({oldActiveBatch, dim1_int}, rt::DeviceType::kGPU, DataType::kINT32);

        copyHostToDevice(tensorDevice, srcData);

        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const resultData = copyDeviceToHost<int32_t>(tensorDevice);

        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t newIdx = batchMapping[oldIdx];
            if (newIdx < 0)
            {
                continue;
            }

            for (int32_t i = 0; i < dim1_int; ++i)
            {
                EXPECT_EQ(resultData[newIdx * dim1_int + i], srcDataBackup[oldIdx * dim1_int + i]);
            }
        }
    }
}

// ============================================================================
// Performance Test: compactKVCacheSingleLayer
// ============================================================================
TEST(BatchEvictKernels, DISABLED_CompactKVCachePerformance)
{
    cudaStream_t stream = nullptr;

    // Realistic configuration for EAGLE
    int32_t numLayers = 32;
    int32_t maxBatchSize = 8;
    int32_t numKVHeads = 8;
    int32_t maxSeqLen = 2048;
    int32_t headDim = 128;

    int32_t oldActiveBatch = 8;
    int32_t newActiveBatch = 6;
    std::vector<int32_t> batchMapping = {0, -1, 1, 2, 3, -1, 4, 5};
    std::vector<int32_t> oldKVLengths = {512, 256, 768, 1024, 384, 640, 896, 448};

    // Initialize per-layer data
    std::vector<std::vector<half>> kvCacheHostLayers(numLayers);
    std::vector<rt::Tensor> kvCacheDeviceLayers;
    kvCacheDeviceLayers.reserve(numLayers);

    for (int32_t l = 0; l < numLayers; ++l)
    {
        initializeKVCacheLayer(kvCacheHostLayers[l], l, maxBatchSize, numKVHeads, maxSeqLen, headDim, oldKVLengths);
        kvCacheDeviceLayers.emplace_back(
            rt::Tensor({maxBatchSize, 2, numKVHeads, maxSeqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF));
        copyHostToDevice(kvCacheDeviceLayers[l], kvCacheHostLayers[l]);
    }

    rt::Tensor kvLengthsDevice({maxBatchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(kvLengthsDevice, oldKVLengths);
    copyHostToDevice(batchMappingDevice, batchMapping);

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        compactKVCacheAllLayers(kvCacheDeviceLayers, batchMappingDevice, kvLengthsDevice, kvLengthsDevice,
            oldActiveBatch, newActiveBatch, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Performance test with cold cache
    constexpr int numIterations = 100;
    std::vector<float> timings;
    timings.reserve(numIterations);

    for (int iter = 0; iter < numIterations; ++iter)
    {
        // Flush L2 cache to ensure cold data
        flushL2Cache();

        // Reset data
        for (int32_t l = 0; l < numLayers; ++l)
        {
            copyHostToDevice(kvCacheDeviceLayers[l], kvCacheHostLayers[l]);
        }
        copyHostToDevice(kvLengthsDevice, oldKVLengths);

        // Time the kernel
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        compactKVCacheAllLayers(kvCacheDeviceLayers, batchMappingDevice, kvLengthsDevice, kvLengthsDevice,
            oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Calculate statistics
    float sum = 0.0f;
    float minTime = timings[0];
    float maxTime = timings[0];
    for (float t : timings)
    {
        sum += t;
        minTime = std::min(minTime, t);
        maxTime = std::max(maxTime, t);
    }
    float avgTime = sum / numIterations;

    // Calculate actual data moved based on batchMapping and actual seqLengths
    // Count batches that need to be moved: newIdx >= 0 && oldIdx != newIdx
    int32_t numBatchesMoved = 0;
    size_t totalElements = 0;
    for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
    {
        int32_t newIdx = batchMapping[oldIdx];
        if (newIdx >= 0 && oldIdx != newIdx)
        {
            numBatchesMoved++;
            // Each moved batch: numLayers * 2(K/V) * numKVHeads * actualSeqLen * headDim
            totalElements += static_cast<size_t>(numLayers) * 2 * numKVHeads * oldKVLengths[oldIdx] * headDim;
        }
    }
    size_t totalBytes = totalElements * sizeof(half);
    float avgBandwidthGB = (totalBytes / (1024.0f * 1024.0f * 1024.0f)) / (avgTime / 1000.0f);

    std::cout << "\n=== compactKVCacheSingleLayer Performance (Cold Cache) ===" << std::endl;
    std::cout << "Configuration: " << numLayers << " layers, " << maxBatchSize << " max batch, " << numKVHeads
              << " heads, " << maxSeqLen << " max seq, " << headDim << " dim" << std::endl;
    std::cout << "Eviction: " << oldActiveBatch << " -> " << newActiveBatch << " batches" << std::endl;
    std::cout << "Batches actually moved: " << numBatchesMoved << std::endl;
    std::cout << "Average time: " << avgTime << " ms" << std::endl;
    std::cout << "Min time: " << minTime << " ms" << std::endl;
    std::cout << "Max time: " << maxTime << " ms" << std::endl;
    std::cout << "Average bandwidth: " << avgBandwidthGB << " GB/s" << std::endl;
    std::cout << "Data moved: " << (totalBytes / (1024.0f * 1024.0f)) << " MB" << std::endl;
}

// ============================================================================
// Performance Test: compactTensorBatch (In-place)
// ============================================================================
TEST(BatchEvictKernels, DISABLED_CompactTensorBatchPerformance)
{
    cudaStream_t stream = nullptr;

    // Test large tensor in-place compaction (realistic EAGLE scenario)
    int32_t oldActiveBatch = 8;
    int32_t newActiveBatch = 6;
    int32_t dim1 = 4096;
    int32_t dim2 = 4096;

    std::vector<int32_t> batchMapping = {0, -1, 1, 2, 3, -1, 4, 5};

    std::vector<half> srcData(oldActiveBatch * dim1 * dim2);
    uniformFloatInitialization(srcData, -1.0f, 1.0f);

    rt::Tensor tensorDevice({oldActiveBatch, dim1, dim2}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor batchMappingDevice({oldActiveBatch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(batchMappingDevice, batchMapping);

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        copyHostToDevice(tensorDevice, srcData);
        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Performance test with cold cache
    constexpr int numIterations = 100;
    std::vector<float> timings;
    timings.reserve(numIterations);

    for (int iter = 0; iter < numIterations; ++iter)
    {
        // Flush L2 cache to ensure cold data
        flushL2Cache();

        // Reset source data
        copyHostToDevice(tensorDevice, srcData);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        // In-place compaction
        compactTensorBatch(tensorDevice, batchMappingDevice, tensorDevice, oldActiveBatch, newActiveBatch, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Calculate statistics
    float sum = 0.0f;
    float minTime = timings[0];
    float maxTime = timings[0];
    for (float t : timings)
    {
        sum += t;
        minTime = std::min(minTime, t);
        maxTime = std::max(maxTime, t);
    }
    float avgTime = sum / numIterations;

    // Calculate actual data moved based on batchMapping
    // Count batches that need to be moved: newIdx >= 0 && oldIdx != newIdx
    int32_t numBatchesMoved = 0;
    for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
    {
        int32_t newIdx = batchMapping[oldIdx];
        if (newIdx >= 0 && oldIdx != newIdx)
        {
            numBatchesMoved++;
        }
    }
    size_t totalBytes = static_cast<size_t>(numBatchesMoved) * dim1 * dim2 * sizeof(half);
    float avgBandwidthGB = (totalBytes / (1024.0f * 1024.0f * 1024.0f)) / (avgTime / 1000.0f);

    std::cout << "\n=== compactTensorBatch In-place Performance (Cold Cache) ===" << std::endl;
    std::cout << "Configuration: " << oldActiveBatch << " -> " << newActiveBatch << " batches, [" << dim1 << ", "
              << dim2 << "] per batch" << std::endl;
    std::cout << "Batches actually moved: " << numBatchesMoved << std::endl;
    std::cout << "Average time: " << avgTime << " ms" << std::endl;
    std::cout << "Min time: " << minTime << " ms" << std::endl;
    std::cout << "Max time: " << maxTime << " ms" << std::endl;
    std::cout << "Average bandwidth: " << avgBandwidthGB << " GB/s" << std::endl;
    std::cout << "Data moved: " << (totalBytes / (1024.0f * 1024.0f)) << " MB" << std::endl;
}
