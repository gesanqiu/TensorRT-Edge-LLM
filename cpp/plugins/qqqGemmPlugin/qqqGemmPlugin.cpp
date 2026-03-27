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

#include "qqqGemmPlugin.h"
#include "kernels/qqqGemmKernels/qqqGemm.h"

#include <cassert>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{

constexpr char const* kQQQ_GEMM_PLUGIN_VERSION{"1"};
constexpr char const* kQQQ_GEMM_PLUGIN_NAME{"QqqGemmPlugin"};
constexpr int32_t kMaxPar = 16;
constexpr size_t kAlignment = 128;

inline size_t alignUp(size_t x, size_t a)
{
    return (x + a - 1) & ~(a - 1);
}

} // namespace

PluginFieldCollection QqqGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> QqqGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(QqqGemmPluginCreator);

// ============================================================================
// Plugin implementation
// ============================================================================

QqqGemmPlugin::QqqGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mGroupSize(groupSize)
{
}

QqqGemmPlugin::QqqGemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "gemm_n")
        {
            mGemmN = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "group_size")
        {
            mGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
}

QqqGemmPlugin::~QqqGemmPlugin() {}

IPluginCapability* QqqGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
            return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME)
            return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* QqqGemmPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new QqqGemmPlugin(mLayerName, mGemmN, mGemmK, mGroupSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

char const* QqqGemmPlugin::getPluginName() const noexcept
{
    return kQQQ_GEMM_PLUGIN_NAME;
}

char const* QqqGemmPlugin::getPluginVersion() const noexcept
{
    return kQQQ_GEMM_PLUGIN_VERSION;
}

char const* QqqGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void QqqGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t QqqGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t QqqGemmPlugin::getOutputDataTypes(
    DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs, DataType const*, int32_t) const noexcept
{
    try
    {
        assert(nbOutputs == 1);
        outputTypes[0] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t QqqGemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t, DimsExprs* outputs, [[maybe_unused]] int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbInputs == expectedNbInputs());
        assert(nbOutputs == 1);
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0]; // batch
        outputs[0].d[1] = inputs[0].d[1]; // seq
        outputs[0].d[2] = exprBuilder.constant(mGemmN);
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool QqqGemmPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        int32_t const expectedInputs = expectedNbInputs();
        assert(nbInputs == expectedInputs && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos].desc;
        bool status = true;

        switch (pos)
        {
        case 0: // input: FP16 [B, S, K]
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmK;
            break;
        case 1: // B weights: INT32 [K/16, 2*N] (Marlin-packed)
            status &= tensorDesc.type == DataType::kINT32;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmK / 16;
            status &= tensorDesc.dims.d[1] == 2 * mGemmN;
            break;
        case 2: // s_channel: FP32 [1, N] (per-channel weight scales)
            status &= tensorDesc.type == DataType::kFLOAT;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[1] == mGemmN;
            break;
        case 3: // s_group: FP16 [K/gs, N] (per-group weight scales, only when isPerGroup())
            if (isPerGroup())
            {
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == PluginFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 2;
                status &= tensorDesc.dims.d[0] == mGemmK / mGroupSize;
                status &= tensorDesc.dims.d[1] == mGemmN;
            }
            else
            {
                // pos==3 is the output when per-channel (3 inputs + 1 output)
                status &= tensorDesc.type == DataType::kHALF;
                status &= tensorDesc.format == PluginFormat::kLINEAR;
                status &= tensorDesc.dims.nbDims == 3;
                status &= tensorDesc.dims.d[2] == mGemmN;
            }
            break;
        case 4: // output: FP16 [B, S, N] (only when isPerGroup)
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmN;
            break;
        default: break;
        }
        return status;
    }
    catch (std::exception const& e)
    {
        return false;
    }
}

int32_t QqqGemmPlugin::configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
    int32_t) noexcept
{
    return 0;
}

size_t QqqGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t,
    DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    int32_t const mMax
        = inputs[0].max.d[0] * inputs[0].max.d[1]; // max batch * max seq

    // quantized activations A: int8 [M, K]
    size_t quantASize = alignUp(static_cast<size_t>(mMax) * mGemmK * sizeof(int8_t), kAlignment);
    // per-token scales s1: float [M]
    size_t s1Size = alignUp(static_cast<size_t>(mMax) * sizeof(float), kAlignment);
    // reduce buffer C: int32 [max_par * 64, N]
    size_t reduceBufSize = alignUp(static_cast<size_t>(kMaxPar) * 64 * mGemmN * sizeof(int32_t), kAlignment);
    // lock array: int32 [ceil(N/128) * max_par]
    size_t lockSize = alignUp(
        static_cast<size_t>((mGemmN + 127) / 128) * kMaxPar * sizeof(int32_t), kAlignment);

    return quantASize + s1Size + reduceBufSize + lockSize;
}

int32_t QqqGemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t const M = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
        int32_t const K = mGemmK;
        int32_t const N = mGemmN;

        half const* inputFP16 = static_cast<half const*>(inputs[0]);
        void const* weights = inputs[1];
        void const* s_channel = inputs[2];
        void const* s_group = isPerGroup() ? inputs[3] : nullptr;
        half* output = static_cast<half*>(outputs[0]);

        if (M == 0)
            return 0;

        char* wsPtr = static_cast<char*>(workspace);
        size_t offset = 0;

        int8_t* quantA = reinterpret_cast<int8_t*>(wsPtr + offset);
        offset += alignUp(static_cast<size_t>(M) * K * sizeof(int8_t), kAlignment);

        float* s1 = reinterpret_cast<float*>(wsPtr + offset);
        offset += alignUp(static_cast<size_t>(M) * sizeof(float), kAlignment);

        int32_t* reduceBuffer = reinterpret_cast<int32_t*>(wsPtr + offset);
        offset += alignUp(static_cast<size_t>(kMaxPar) * 64 * N * sizeof(int32_t), kAlignment);

        int32_t* locks = reinterpret_cast<int32_t*>(wsPtr + offset);
        size_t lockBytes = alignUp(
            static_cast<size_t>((N + 127) / 128) * kMaxPar * sizeof(int32_t), kAlignment);

        // Locks must be zeroed before each GEMM
        cudaMemsetAsync(locks, 0, lockBytes, stream);

        // Step 1: Dynamic per-token quantization (FP16 → INT8 + FP32 scales)
        trt_edgellm::kernel::dynamicQuantize(inputFP16, quantA, s1, M, K, stream);

        // Step 2: W4A8 GEMM (INT8 activations × Marlin-packed INT4 weights → FP16)
        int groupsize = isPerGroup() ? mGroupSize : -1;
        int err = trt_edgellm::kernel::qqqGemmForward(quantA, weights, reduceBuffer, output, s1, s_channel,
            s_group, M, N, K, locks, groupsize, /*dev=*/0, stream);

        return err;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t QqqGemmPlugin::onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept
{
    return 0;
}

IPluginV3* QqqGemmPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}

PluginFieldCollection const* QqqGemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("group_size", &mGroupSize, PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// Plugin Creator
// ============================================================================

QqqGemmPluginCreator::QqqGemmPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* QqqGemmPluginCreator::getPluginName() const noexcept
{
    return kQQQ_GEMM_PLUGIN_NAME;
}

char const* QqqGemmPluginCreator::getPluginVersion() const noexcept
{
    return kQQQ_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* QqqGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* QqqGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void QqqGemmPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

IPluginV3* QqqGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept
{
    try
    {
        auto* plugin = new QqqGemmPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
