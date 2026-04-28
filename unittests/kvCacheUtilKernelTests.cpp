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
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "runtime/kvCacheManager.h"
#include "testUtils.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;
using namespace nvinfer1;

struct KVCacheParameters
{
    int32_t numDecoderLayers;
    int32_t maxBatchSize;
    int32_t maxSequenceLength;
    int32_t numKVHead;
    int32_t headDim;
};

void TestKVCacheCopyWithTensor(KVCacheParameters const& cacheParams, int32_t copyBatchIdx, int32_t copySequenceLen)
{
    cudaStream_t stream{nullptr};

    // Build per-layer configs (all layers uniform for these tests)
    std::vector<rt::KVLayerConfig> layerConfigs(
        cacheParams.numDecoderLayers, rt::KVLayerConfig{cacheParams.numKVHead, cacheParams.headDim});

    rt::KVCacheManager kvManager(rt::KVCacheManager::Config{cacheParams.numDecoderLayers, cacheParams.maxBatchSize,
                                     cacheParams.maxSequenceLength, layerConfigs, DataType::kHALF},
        stream);

    // Per-layer saved tensor shape: [2, numKVHead, copySequenceLen, headDim]
    // Test each layer independently using single-layer kernel variants
    for (int32_t idxL = 0; idxL < cacheParams.numDecoderLayers; idxL++)
    {
        rt::Tensor cacheTensor = rt::Tensor(
            {2, cacheParams.numKVHead, copySequenceLen, cacheParams.headDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor& kvCacheLayer = kvManager.getCombinedKVCache(idxL);

        // Instantiate the cache tensor with random data
        std::vector<half> cacheDataHost(cacheTensor.getShape().volume(), 0.0f);
        uniformFloatInitialization(cacheDataHost);

        // Copy the cache tensor to the KVCache layer
        CUDA_CHECK(cudaMemcpy(
            cacheTensor.rawPointer(), cacheDataHost.data(), cacheTensor.getMemoryCapacity(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(kvCacheLayer.rawPointer(), 0, kvCacheLayer.getMemoryCapacity()));

        // Perform the copy from tensor to Cache and pull the data back to host.
        std::vector<half> kvCacheLayerHost(kvCacheLayer.getShape().volume(), 0.0f);
        kernel::instantiateKVCacheLayerFromTensor(kvCacheLayer, cacheTensor, copyBatchIdx, stream);
        CUDA_CHECK(cudaMemcpyAsync(kvCacheLayerHost.data(), kvCacheLayer.rawPointer(), kvCacheLayer.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify the data in the KVCache layer
        auto compareCacheAndTensorData = [&]() {
            KvCacheIndexer indexer(
                cacheParams.maxBatchSize, cacheParams.numKVHead, cacheParams.maxSequenceLength, cacheParams.headDim);
            // tensorLayerOffset is 0 since the saved tensor is now per-layer [2, numKVHead, seqLen, headDim]
            for (int32_t idxS = 0; idxS < copySequenceLen; idxS++)
            {
                for (int32_t idxKV = 0; idxKV < cacheParams.numKVHead; idxKV++)
                {
                    for (int32_t idxD = 0; idxD < cacheParams.headDim; idxD++)
                    {
                        // First compare K then V.
                        // Saved tensor layout: [2, numKVHead, sequenceLength, headDim]
                        int64_t srcKOffset
                            = idxKV * copySequenceLen * cacheParams.headDim + idxS * cacheParams.headDim + idxD;
                        int64_t dstKOffset = indexer.indexK(copyBatchIdx, idxKV, idxS, idxD);
                        if (!isclose(kvCacheLayerHost[dstKOffset], cacheDataHost[srcKOffset], 1e-5, 1e-5))
                        {
                            std::cout << "Mismatch at layer " << idxL << ", sequence " << idxS << ", KV head " << idxKV
                                      << ", dim " << idxD << std::endl;
                            std::cout << "kvCacheLayerHost[dstKOffset]: " << __half2float(kvCacheLayerHost[dstKOffset])
                                      << ", cacheDataHost[srcKOffset]: " << __half2float(cacheDataHost[srcKOffset])
                                      << std::endl;
                        }
                        ASSERT_TRUE(isclose(kvCacheLayerHost[dstKOffset], cacheDataHost[srcKOffset], 1e-5, 1e-5));

                        int64_t srcVOffset = (cacheParams.numKVHead + idxKV) * copySequenceLen * cacheParams.headDim
                            + idxS * cacheParams.headDim + idxD;
                        int64_t dstVOffset = indexer.indexV(copyBatchIdx, idxKV, idxS, idxD);
                        ASSERT_TRUE(isclose(kvCacheLayerHost[dstVOffset], cacheDataHost[srcVOffset], 1e-5, 1e-5));
                    }
                }
            }
        };

        compareCacheAndTensorData();
        std::cout << "Tested copy from tensor to cache layer " << idxL << " with batchIdx " << copyBatchIdx
                  << ", sequence length " << copySequenceLen
                  << ", KVCacheLayer shape ([maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]): "
                  << kvCacheLayer.getShape().formatString() << std::endl;

        // cudaMemset the cache tensor and test from the other direction.
        CUDA_CHECK(cudaMemsetAsync(cacheTensor.rawPointer(), 0, cacheTensor.getMemoryCapacity(), stream));
        kernel::saveKVCacheLayerIntoTensor(cacheTensor, kvCacheLayer, copyBatchIdx, stream);
        CUDA_CHECK(cudaMemcpyAsync(cacheDataHost.data(), cacheTensor.rawPointer(), cacheTensor.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        compareCacheAndTensorData();

        std::cout << "Tested copy from cache to tensor layer " << idxL << " with batchIdx " << copyBatchIdx
                  << ", sequence length " << copySequenceLen
                  << ", saved kvCacheTensor shape ([2, numKVHeads, sequenceLength, headDim]): "
                  << cacheTensor.getShape().formatString() << std::endl;
    }
}

TEST(KVCacheUtilKernelTests, TestKVCacheCopyWithTensor)
{
    // KVCache: 3 decoder layers, 8 max batch size, 1024 max sequence length, 4 KV heads, 128 head dim.
    // Copy to batchIdx 0 with sequence length 128.
    TestKVCacheCopyWithTensor({3, 8, 1024, 4, 128}, 0, 128);
    // KVCache: 3 decoder layers, 8 max batch size, 1024 max sequence length, 4 KV heads, 128 head dim.
    // Copy to batchIdx 1 with sequence length 97, which is not divisible by 2
    TestKVCacheCopyWithTensor({3, 8, 1024, 4, 128}, 1, 97);
    // KVCache: 3 decoder layers, 4 max batch size, 512 max sequence length, 7 KV heads, 64 head dim.
    // Copy to batchIdx 0 with sequence length 96.
    TestKVCacheCopyWithTensor({3, 4, 512, 7, 64}, 0, 96);
    // KVCache: 3 decoder layers, 4 max batch size, 512 max sequence length, 7 KV heads, 64 head dim.
    // Copy to batchIdx 1 with sequence length 47, which is not divisible by 4.
    TestKVCacheCopyWithTensor({3, 4, 512, 7, 64}, 1, 47);
    // KVCache: 28 decoder layers, 4 max batch size, 2048 max sequence length, 4 KV heads, 128 head dim.
    // Copy to batchIdx 0 with sequence length 1010, simulate the Qwen2-VL config..
    TestKVCacheCopyWithTensor({28, 1, 1024, 4, 128}, 0, 255);
}
