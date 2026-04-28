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

// Per-expert alpha = actGs * weightGs[e]. Both inputs are forward-direction global SFs.
// Single-CTA kernel; block size rounded up to warp boundary for launch efficiency.

#include "alphaCompute.h"

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

__global__ void computeAlphaKernel(float const* __restrict__ actGs, float const* __restrict__ weightGs,
    float* __restrict__ alpha, int32_t numLocalExperts)
{
    int32_t const tid = threadIdx.x;
    if (tid >= numLocalExperts)
    {
        return;
    }
    float const act = *actGs;
    alpha[tid] = act * weightGs[tid];
}

void launchComputeAlpha(
    float const* actGs, float const* weightGs, float* alpha, int32_t numLocalExperts, cudaStream_t stream)
{
    if (numLocalExperts <= 0)
    {
        return;
    }
    int32_t const blockSize = ((numLocalExperts + 31) / 32) * 32;
    computeAlphaKernel<<<1, blockSize, 0, stream>>>(actGs, weightGs, alpha, numLocalExperts);
}

} // namespace

void computeFC1Alpha(
    float const* actGs, float const* weightGs, float* alpha, int32_t numLocalExperts, cudaStream_t stream)
{
    launchComputeAlpha(actGs, weightGs, alpha, numLocalExperts, stream);
}

void computeFC2Alpha(
    float const* actGs, float const* weightGs, float* alpha, int32_t numLocalExperts, cudaStream_t stream)
{
    launchComputeAlpha(actGs, weightGs, alpha, numLocalExperts, stream);
}

} // namespace kernel
} // namespace trt_edgellm
