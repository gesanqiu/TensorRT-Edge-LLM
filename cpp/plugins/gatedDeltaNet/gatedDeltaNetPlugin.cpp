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

#include "gatedDeltaNetPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
#ifdef CUTE_DSL_GDN_ENABLED
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#include "kernels/gdnKernels/gdnKernelUtils.cuh"
#endif

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kGDN_PLUGIN_VERSION{"1"};
constexpr char const* kGDN_PLUGIN_NAME{"gated_delta_net"};

constexpr int32_t kIN_Q{0};
constexpr int32_t kIN_K{1};
constexpr int32_t kIN_V{2};
constexpr int32_t kIN_A{3};
constexpr int32_t kIN_B{4};
constexpr int32_t kIN_A_LOG{5};
constexpr int32_t kIN_DT_BIAS{6};
constexpr int32_t kIN_H0_SOURCE{7};
constexpr int32_t kIN_CONTEXT_LENGTHS{8};
constexpr int32_t kOUT_O{0};
constexpr int32_t kOUT_H0_SOURCE{1};
constexpr int32_t kNUM_INPUTS{9};
constexpr int32_t kNUM_OUTPUTS{2};
} // namespace

PluginFieldCollection GatedDeltaNetPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> GatedDeltaNetPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedDeltaNetPluginCreator);

// ---------------------------------------------------------------------------
// Plugin constructor — only this block is compilation-guarded.
// When CUTE_DSL_GDN_ENABLED is not set the constructor throws immediately so
// the object can never be constructed; all other methods are shared.
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mSMVersion(getSMVersion())
{
    if (!CuteDslGDNRunner::canImplement(mKDim, mVDim, mSMVersion))
    {
        LOG_ERROR(
            "Cannot implement GatedDeltaNetPlugin (CuTe DSL): k_dim=%d v_dim=%d SM=%d. "
            "CuTe DSL GDN is only built for k=v=128 and requires SM>=80 (Ampere+). "
            "Use k_dim=v_dim=128 on a supported GPU, or rebuild without CuTe DSL GDN if applicable.",
            mKDim, mVDim, mSMVersion);
        throw std::runtime_error("Cannot implement the GatedDeltaNetPlugin configuration (CuTe DSL GDN).");
    }

    if (!CuteDslGDNRunner::loadKernelModules())
    {
        LOG_ERROR(
            "Failed to load CuTe DSL GDN kernel modules (gdn_decode / gdn_prefill AOT). "
            "Check that the engine was built with ENABLE_CUTE_DSL=gdn (or ALL), AOT .o/.h are present and match the "
            "exported API, and the CUDA driver is compatible.");
        throw std::runtime_error("Cannot load CuTe DSL GDN kernel modules for GatedDeltaNetPlugin.");
    }
}
#else
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
{
    LOG_ERROR("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
    throw std::runtime_error("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
}
#endif // CUTE_DSL_GDN_ENABLED

GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    auto const* d = static_cast<char const*>(data);
    std::memcpy(&mKDim, d, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(&mVDim, d, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(&mSMVersion, d, sizeof(int32_t));

#ifdef CUTE_DSL_GDN_ENABLED
    CuteDslGDNRunner::loadKernelModules();
#endif
}

GatedDeltaNetPlugin::~GatedDeltaNetPlugin() = default;

// ---------------------------------------------------------------------------
// IPluginV2DynamicExt
// ---------------------------------------------------------------------------

IPluginV2DynamicExt* GatedDeltaNetPlugin::clone() const noexcept
{
    try
    {
        auto* p = new GatedDeltaNetPlugin(mLayerName, mKDim, mVDim);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

int32_t GatedDeltaNetPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

DataType GatedDeltaNetPlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    if (index == kOUT_O)
        return inputTypes[kIN_Q];
    // kOUT_H0_SOURCE
    return inputTypes[kIN_H0_SOURCE];
}

DimsExprs GatedDeltaNetPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs, IExprBuilder&) noexcept
{
    if (outputIndex == kOUT_O)
        return inputs[kIN_V]; // o has same shape as v: [n, seq_len, hv, v]
    // h0_out has same shape as h0_source: [n, hv, k, v]
    return inputs[kIN_H0_SOURCE];
}

bool GatedDeltaNetPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
        return false;
    if (inOut[pos].format != TensorFormat::kLINEAR)
        return false;
    if (pos == kIN_A_LOG || pos == kIN_H0_SOURCE)
        return inOut[pos].type == DataType::kFLOAT;
    if (pos == kIN_CONTEXT_LENGTHS)
        return inOut[pos].type == DataType::kINT32;
    if (pos == kNUM_INPUTS + kOUT_H0_SOURCE)
        return inOut[pos].type == DataType::kFLOAT;
    return inOut[pos].type == DataType::kHALF;
}

void GatedDeltaNetPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS)
    {
        LOG_ERROR("gated_delta_net: expected %d inputs, got %d", kNUM_INPUTS, nbInputs);
    }
    if (in[kIN_Q].desc.type != DataType::kHALF || in[kIN_V].desc.type != DataType::kHALF)
    {
        LOG_ERROR("gated_delta_net: Q and V must be FP16");
    }
    if (in[kIN_Q].desc.dims.nbDims != 4 || in[kIN_V].desc.dims.nbDims != 4)
    {
        LOG_ERROR("gated_delta_net: Q and V must be 4D");
    }
    if (in[kIN_CONTEXT_LENGTHS].desc.type != DataType::kINT32 || in[kIN_CONTEXT_LENGTHS].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: context_lengths must be 1D INT32");
    }
}

size_t GatedDeltaNetPlugin::getWorkspaceSize([[maybe_unused]] PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // V2 plugin receives contiguous buffers — no a/b compaction workspace needed.
    size_t total = 0;

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    int32_t const maxN = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS].dims.d[0]);
    int32_t const maxHv = static_cast<int32_t>(inputs[kIN_H0_SOURCE].dims.d[1]);
    int32_t const kDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE].dims.d[2]);
    int32_t const vDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE].dims.d[3]);

    // cu_seqlens [maxN+1] int32, padded to 128-byte alignment.
    size_t const cuSeqBytes = static_cast<size_t>(maxN + 1) * sizeof(int32_t);
    size_t const cuSeqPadded = (cuSeqBytes + 127u) & ~static_cast<size_t>(127u);
    // h0 scratch [maxN, maxHv, kDim, vDim] f32 — separate buffer for Blackwell h0_out.
    size_t const h0ScratchBytes = static_cast<size_t>(maxN) * maxHv * kDim * vDim * sizeof(float);

    total = cuSeqPadded + h0ScratchBytes;
#endif

    return total;
}

// ---------------------------------------------------------------------------
// enqueue — only this block is compilation-guarded.
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    CuteDslGDNRunner::loadKernelModules();

    int64_t const* qDims = inputDesc[kIN_Q].dims.d;
    int32_t const n = static_cast<int32_t>(qDims[0]);
    int32_t const seq_len = static_cast<int32_t>(qDims[1]);
    int32_t const h = static_cast<int32_t>(qDims[2]);
    int32_t const k_dim = static_cast<int32_t>(qDims[3]);

    int64_t const* vDims = inputDesc[kIN_V].dims.d;
    int32_t const hv = static_cast<int32_t>(vDims[2]);
    int32_t const v_dim = static_cast<int32_t>(vDims[3]);

    // h0 is batch-dense [n, hv, k, v]
    size_t const h0Bytes = static_cast<size_t>(n) * hv * static_cast<size_t>(k_dim) * v_dim * sizeof(float);
    void* h0Out = outputs[kOUT_H0_SOURCE];
    if (h0Out != inputs[kIN_H0_SOURCE])
    {
        cudaMemcpyAsync(h0Out, inputs[kIN_H0_SOURCE], h0Bytes, cudaMemcpyDeviceToDevice, stream);
    }

    // V2 plugin: buffers are contiguous, no a/b compaction needed.
    GDNParams params{};
    params.q = const_cast<void*>(inputs[kIN_Q]);
    params.k = const_cast<void*>(inputs[kIN_K]);
    params.v = const_cast<void*>(inputs[kIN_V]);
    params.a = const_cast<void*>(inputs[kIN_A]);
    params.b = const_cast<void*>(inputs[kIN_B]);
    params.A_log = const_cast<void*>(inputs[kIN_A_LOG]);
    params.dt_bias = const_cast<void*>(inputs[kIN_DT_BIAS]);
    params.h0_source = h0Out;
    params.context_lengths = const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS]);
    params.o = outputs[kOUT_O];
    params.n = n;
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.k_dim = k_dim;
    params.v_dim = v_dim;
    params.smVersion = mSMVersion;

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    // Blackwell prefill: workspace layout (no a/b compaction buffers):
    //   [cu_seqlens: (n+1)*int32, pad to 128B] [h0_scratch: n*hv*k*v*f32]
    if (seq_len > 1 && mSMVersion >= 100)
    {
        size_t const cuSeqBytes = static_cast<size_t>(n + 1) * sizeof(int32_t);
        size_t const cuSeqPadded = (cuSeqBytes + 127u) & ~static_cast<size_t>(127u);

        char* bwBase = static_cast<char*>(workspace);
        launchGdnCalCuSeqLens(inputs[kIN_CONTEXT_LENGTHS], bwBase, n, stream);
        params.cu_seqlens = bwBase;
        params.h0_scratch = bwBase + cuSeqPadded;
    }
#endif

    CuteDslGDNRunner runner;
    int ret = runner.run(params, stream);

    return (ret == 0) ? 0 : -1;
}
#else
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* /* inputDesc */, PluginTensorDesc const* /* outputDesc */,
    void const* const* /* inputs */, void* const* /* outputs */, void* /* workspace */,
    cudaStream_t /* stream */) noexcept
{
    // Constructor already threw; this path should be unreachable.
    return -1;
}
#endif // CUTE_DSL_GDN_ENABLED

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

size_t GatedDeltaNetPlugin::getSerializationSize() const noexcept
{
    return 3 * sizeof(int32_t); // mKDim, mVDim, mSMVersion
}

void GatedDeltaNetPlugin::serialize(void* buffer) const noexcept
{
    auto* d = static_cast<char*>(buffer);
    std::memcpy(d, &mKDim, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(d, &mVDim, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(d, &mSMVersion, sizeof(int32_t));
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

char const* GatedDeltaNetPlugin::getPluginType() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPlugin::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

char const* GatedDeltaNetPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

int32_t GatedDeltaNetPlugin::initialize() noexcept
{
    return 0;
}

void GatedDeltaNetPlugin::terminate() noexcept {}

void GatedDeltaNetPlugin::destroy() noexcept
{
    delete this;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

GatedDeltaNetPluginCreator::GatedDeltaNetPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("k_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("v_dim", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedDeltaNetPluginCreator::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPluginCreator::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedDeltaNetPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* GatedDeltaNetPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

IPluginV2* GatedDeltaNetPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        int32_t kDim = parsePluginScalarField<int32_t>("k_dim", fc).value_or(128);
        int32_t vDim = parsePluginScalarField<int32_t>("v_dim", fc).value_or(128);
        auto* plugin = new GatedDeltaNetPlugin(name, kDim, vDim);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("GatedDeltaNetPluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

IPluginV2* GatedDeltaNetPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* plugin = new GatedDeltaNetPlugin(name, serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to deserialize GatedDeltaNetPlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
