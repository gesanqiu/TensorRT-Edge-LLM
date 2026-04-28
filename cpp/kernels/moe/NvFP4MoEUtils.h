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

// Shared types for the NvFP4 MoE CuteDSL kernel runners.

#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_moe
{

// Activation modes supported by the FC1 kernel.
enum class Activation : int32_t
{
    kIdentity = 0,
    kRelu2 = 1,
    kSwiglu = 2,
};

// Output element type for FC1 / FC2 runners. Selects which AOT-compiled
// kernel variant is invoked; the C-ABI wrapper signatures are identical
// across dtypes, only the compiled binary differs.
enum class OutputDType : int32_t
{
    kBF16 = 0,
    kFP16 = 1,
};

// Runtime tile metadata produced by the layout builder and consumed by
// both the FC1 and FC2 AOT kernel wrappers.
struct MoELayout
{
    int32_t* tileIdxToGroupIdx{};
    int32_t* tileIdxToMnLimit{};
    int32_t* permutedIdxToExpandedIdx{};
    int32_t* numNonExitingTiles{};
    int32_t numTiles{};
    int32_t mPadded{};

    // Host-side vectors (for extractExpertGroups + alpha+activation)
    std::vector<int32_t> tileIdxToGroupIdxHost;
    std::vector<int32_t> tileIdxToMnLimitHost;
};

} // namespace nvfp4_moe
} // namespace kernel
} // namespace trt_edgellm
