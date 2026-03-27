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

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/*!
 * @brief TensorRT plugin for QQQ W4A8 GEMM (INT4 weights, INT8 activations).
 *
 * Accepts FP16 activations, performs dynamic per-token INT8 quantization internally,
 * then executes W4A8 GEMM using INT8 Tensor Cores with Marlin-packed INT4 weights.
 * Output is FP16.
 *
 * Inputs (per-channel, group_size == -1):
 *   [0] input:     FP16 [B, S, K]      — activation tensor
 *   [1] weights:   INT32 [K/16, 2*N]   — Marlin-packed INT4 weights
 *   [2] s_channel: FP32 [1, N]         — per-channel weight scales
 *
 * Inputs (per-group, group_size == 128):
 *   [0] input:     FP16 [B, S, K]      — activation tensor
 *   [1] weights:   INT32 [K/16, 2*N]   — Marlin-packed INT4 weights
 *   [2] s_channel: FP32 [1, N]         — per-channel weight scales
 *   [3] s_group:   FP16 [K/gs, N]      — per-group weight scales
 *
 * Output:
 *   [0] output:    FP16 [B, S, N]
 */
class QqqGemmPlugin : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime
{
public:
    QqqGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize);
    QqqGemmPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    QqqGemmPlugin() = delete;
    QqqGemmPlugin(QqqGemmPlugin const&) = delete;
    ~QqqGemmPlugin() override;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    //! Whether this instance uses per-group quantization (has 4 inputs instead of 3)
    bool isPerGroup() const noexcept
    {
        return mGroupSize > 0 && mGroupSize < mGemmK;
    }

    int32_t expectedNbInputs() const noexcept
    {
        return isPerGroup() ? 4 : 3;
    }

    std::string mLayerName;
    std::string mNamespace;

    int32_t mGemmN{};
    int32_t mGemmK{};
    int32_t mGroupSize{}; //!< -1 for per-channel, 128 for per-group

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

/*!
 * @brief Factory for creating QqqGemmPlugin instances.
 */
class QqqGemmPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    QqqGemmPluginCreator();
    ~QqqGemmPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
