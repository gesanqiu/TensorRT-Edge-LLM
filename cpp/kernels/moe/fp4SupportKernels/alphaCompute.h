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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Compute per-expert FC1 alpha for the grouped-GEMM FP32 epilogue.
///
/// Forward-scale contract: \c actGs and \c weightGs are forward-direction global SFs.
/// The kernel writes `alpha[i] = (*actGs) * weightGs[i]` — exactly the scalar applied
/// before the activation inside the FC1 kernel (`out[m,n] = act(alpha[e(m)] * acc)`).
///
/// @param actGs              [1] float32 on device — forward-direction activation GS.
/// @param weightGs           [L] float32 on device — forward-direction per-expert weight GS.
/// @param alpha              [L] float32 on device (output).
/// @param numLocalExperts    L — number of local experts.
/// @param stream             CUDA stream.
void computeFC1Alpha(
    float const* actGs, float const* weightGs, float* alpha, int32_t numLocalExperts, cudaStream_t stream);

/// Compute per-expert FC2 alpha. Same shape as \c computeFC1Alpha; distinct symbol
/// provided for clarity at callsites.
void computeFC2Alpha(
    float const* actGs, float const* weightGs, float* alpha, int32_t numLocalExperts, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
