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

#include "common/tensor.h"

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace kernel
{

enum class Activation : int32_t
{
    kIdentity = 0,
    kRelu2 = 1,
    kSwiglu = 2,
};

/// CPU-side layout result. Populated by the test-only CPU reference builder.
struct MoELayout
{
    rt::Tensor tileIdxToGroupIdx;        // [maxTiles], INT32, GPU
    rt::Tensor tileIdxToMnLimit;         // [maxTiles], INT32, GPU
    rt::Tensor permutedIdxToExpandedIdx; // [maxMPadded], INT32, GPU
    rt::Tensor numNonExitingTiles;       // [1], INT32, GPU
    int32_t numTiles{};
    int32_t mPadded{};

    std::vector<int32_t> tileIdxToGroupIdxHost;
    std::vector<int32_t> tileIdxToMnLimitHost;
};

/// Pre-allocated GPU buffers for the layout builder kernel.
/// maxTiles = tileIdxToGroupIdx.getShape()[0],
/// maxMPadded = permutedIdxToExpandedIdx.getShape()[0].
struct MoELayoutBuffers
{
    rt::Tensor tileIdxToGroupIdx;        // [maxTiles], INT32, GPU
    rt::Tensor tileIdxToMnLimit;         // [maxTiles], INT32, GPU
    rt::Tensor permutedIdxToExpandedIdx; // [maxMPadded], INT32, GPU
    rt::Tensor numNonExitingTiles;       // [1], INT32, GPU
};

} // namespace kernel
} // namespace trt_edgellm
