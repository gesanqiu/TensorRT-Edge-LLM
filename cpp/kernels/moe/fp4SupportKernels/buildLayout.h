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

#include "nvfp4MoeTypes.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// GPU-side layout builder via single-CTA kernel (~3-5 us).
/// All device pointers in `buffers` must be pre-allocated by the caller.
/// tokenSelectedExperts must contain LOCAL expert indices in [0, L).
void buildLayoutGpu(MoELayoutBuffers& buffers, int32_t const* tokenSelectedExperts, int32_t numTokens, int32_t topK,
    int32_t localNumExperts, int32_t tileSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
