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

#include "runtime/mambaCacheManager.h"
#include "common/checkMacros.h"
#include "common/logger.h"

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

MambaCacheManager::MambaCacheManager(Config const& config, cudaStream_t stream)
    : mConfig(config)
{
    if (mConfig.numRecurrentLayers == 0)
    {
        return;
    }

    check::check(mConfig.maxBatchSize > 0, "maxBatchSize must be positive.");
    check::check(mConfig.recurrentStateNumHeads > 0, "recurrentStateNumHeads must be positive.");
    check::check(mConfig.recurrentStateHeadDim > 0, "recurrentStateHeadDim must be positive.");
    check::check(mConfig.recurrentStateSize > 0, "recurrentStateSize must be positive.");
    check::check(mConfig.convDim > 0, "convDim must be positive.");
    check::check(mConfig.convKernel > 0, "convKernel must be positive.");

    size_t totalBytes = 0;
    size_t const recurrentElemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mConfig.convStateType);

    mRecurrentStates.reserve(mConfig.numRecurrentLayers);
    mConvStates.reserve(mConfig.numRecurrentLayers);

    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        // Allocate recurrent state: [maxBatchSize, numHeads, headDim, stateSize]
        int64_t const recurrentVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.recurrentStateNumHeads
            * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
        size_t const recurrentBytes = static_cast<size_t>(recurrentVolume) * recurrentElemSize;

        mRecurrentStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.recurrentStateNumHeads,
                                                     mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
            DeviceType::kGPU, mConfig.recurrentStateType, "MambaCacheManager::recurrentState_" + std::to_string(i)));
        CUDA_CHECK(cudaMemsetAsync(mRecurrentStates.back().rawPointer(), 0, recurrentBytes, stream));

        // Allocate conv state: [maxBatchSize, convDim, convKernel]
        int64_t const convVolume = static_cast<int64_t>(mConfig.maxBatchSize) * mConfig.convDim * mConfig.convKernel;
        size_t const convBytes = static_cast<size_t>(convVolume) * convElemSize;

        mConvStates.emplace_back(rt::Tensor({mConfig.maxBatchSize, mConfig.convDim, mConfig.convKernel},
            DeviceType::kGPU, mConfig.convStateType, "MambaCacheManager::convState_" + std::to_string(i)));
        CUDA_CHECK(cudaMemsetAsync(mConvStates.back().rawPointer(), 0, convBytes, stream));

        totalBytes += recurrentBytes + convBytes;
    }

    LOG_DEBUG("MambaCacheManager(layers=%d) allocated %.2f MB total GPU memory", mConfig.numRecurrentLayers,
        static_cast<float>(totalBytes) / (1024.0f * 1024.0f));
}

MambaCacheManager::~MambaCacheManager() noexcept {}

MambaCacheManager::MambaCacheManager(MambaCacheManager&& other) noexcept
{
    mConfig = std::move(other.mConfig);
    mRecurrentStates = std::move(other.mRecurrentStates);
    mConvStates = std::move(other.mConvStates);

    other.mConfig = Config{};
}

MambaCacheManager& MambaCacheManager::operator=(MambaCacheManager&& other) noexcept
{
    if (this != &other)
    {
        mConfig = std::move(other.mConfig);
        mRecurrentStates = std::move(other.mRecurrentStates);
        mConvStates = std::move(other.mConvStates);

        other.mConfig = Config{};
    }
    return *this;
}

rt::Tensor& MambaCacheManager::getRecurrentState(int32_t recurrentLayerIdx) noexcept
{
    return mRecurrentStates[recurrentLayerIdx];
}

rt::Tensor& MambaCacheManager::getConvState(int32_t recurrentLayerIdx) noexcept
{
    return mConvStates[recurrentLayerIdx];
}

void MambaCacheManager::clearStates(cudaStream_t stream)
{
    for (int32_t i = 0; i < mConfig.numRecurrentLayers; ++i)
    {
        CUDA_CHECK(
            cudaMemsetAsync(mRecurrentStates[i].rawPointer(), 0, mRecurrentStates[i].getMemoryCapacity(), stream));
        CUDA_CHECK(cudaMemsetAsync(mConvStates[i].rawPointer(), 0, mConvStates[i].getMemoryCapacity(), stream));
    }
}

std::vector<rt::Tensor> MambaCacheManager::captureRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    if (mConfig.numRecurrentLayers == 0)
    {
        return result;
    }

    size_t const elemSize = rt::utils::getTypeSize(mConfig.recurrentStateType);
    int64_t const perBatchElems
        = mConfig.recurrentStateNumHeads * mConfig.recurrentStateHeadDim * mConfig.recurrentStateSize;
    size_t const perBatchBytes = static_cast<size_t>(perBatchElems) * elemSize;

    result.reserve(mConfig.numRecurrentLayers);
    for (int32_t layer = 0; layer < mConfig.numRecurrentLayers; ++layer)
    {
        void const* src = static_cast<char const*>(mRecurrentStates[layer].rawPointer())
            + static_cast<size_t>(batchIdx * perBatchElems) * elemSize;
        rt::Tensor saved({1, mConfig.recurrentStateNumHeads, mConfig.recurrentStateHeadDim, mConfig.recurrentStateSize},
            DeviceType::kGPU, mConfig.recurrentStateType,
            "MambaCacheManager::capturedRecurrentState_" + std::to_string(layer));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(), src, perBatchBytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

std::vector<rt::Tensor> MambaCacheManager::captureConvStates(int32_t batchIdx, cudaStream_t stream)
{
    std::vector<rt::Tensor> result;
    if (mConfig.numRecurrentLayers == 0)
    {
        return result;
    }

    size_t const elemSize = rt::utils::getTypeSize(mConfig.convStateType);
    int64_t const perBatchElems = mConfig.convDim * mConfig.convKernel;
    size_t const perBatchBytes = static_cast<size_t>(perBatchElems) * elemSize;

    result.reserve(mConfig.numRecurrentLayers);
    for (int32_t layer = 0; layer < mConfig.numRecurrentLayers; ++layer)
    {
        void const* src = static_cast<char const*>(mConvStates[layer].rawPointer())
            + static_cast<size_t>(batchIdx * perBatchElems) * elemSize;
        rt::Tensor saved({1, mConfig.convDim, mConfig.convKernel}, DeviceType::kGPU, mConfig.convStateType,
            "MambaCacheManager::capturedConvState_" + std::to_string(layer));
        CUDA_CHECK(cudaMemcpyAsync(saved.rawPointer(), src, perBatchBytes, cudaMemcpyDeviceToDevice, stream));
        result.push_back(std::move(saved));
    }
    return result;
}

int32_t MambaCacheManager::numLayers() const noexcept
{
    return mConfig.numRecurrentLayers;
}

MambaCacheManager::Config const& MambaCacheManager::getConfig() const noexcept
{
    return mConfig;
}

} // namespace rt
} // namespace trt_edgellm
