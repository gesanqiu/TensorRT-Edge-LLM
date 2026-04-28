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
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Launch the MoE gather kernel.
///
/// Permutes FP4 data + atom-layout SF from token order to expert-grouped order.
/// One CTA per output row, 256 threads. Caller must pre-zero dstSF.
///
/// @param srcFP4       Packed FP4 source data (viewed as int32_t*)
/// @param dstFP4       Packed FP4 output data (viewed as int32_t*)
/// @param srcSF        Atom-layout SF buffer (source, viewed as int32_t*)
/// @param dstSF        Atom-layout SF buffer (dest, viewed as int32_t*, must be pre-zeroed)
/// @param permuteMap   INT32 permutation map (-1 = padding). May be sized
///                     larger than \p permutedM; only the first \p permutedM
///                     entries are read.
/// @param permutedM    Number of dst rows to process (must fit dst buffer shape)
/// @param topK         Experts per token
/// @param hiddenSize   Hidden dimension K
/// @param stream       CUDA stream
void launchMoeGather(rt::Tensor const& srcFP4, rt::Tensor& dstFP4, rt::Tensor const& srcSF, rt::Tensor& dstSF,
    rt::Tensor const& permuteMap, int32_t permutedM, int32_t topK, int32_t hiddenSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
