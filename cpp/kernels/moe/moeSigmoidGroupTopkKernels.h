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
namespace trt_edgellm
{
namespace kernel
{

/**
 * @brief MoE Sigmoid Group TopK kernel implementing HuggingFace NemotronH routing
 *
 * This kernel implements the grouped top-k routing algorithm from NemotronHMoE:
 * 1. Applies sigmoid to router logits: scores = sigmoid(logits)
 * 2. Adds optional correction bias: biased = scores + bias
 * 3. Groups experts, finds top-2 per group, sums -> groupScores
 * 4. Selects topkGroup groups with highest groupScores
 * 5. Masks experts NOT in selected groups
 * 6. Selects topK experts from masked biased scores
 * 7. Gathers weights from ORIGINAL sigmoid scores (not biased)
 * 8. Optionally renormalizes weights to sum to 1
 * 9. Scales weights by routedScalingFactor
 *
 * @param gatingOutput Input router logits [numTokens, numExperts] (FP32, GPU)
 * @param topkWeights Output selected expert weights [numTokens, topK] (FP32, GPU)
 * @param topkIndices Output selected expert indices [numTokens, topK] (INT32, GPU)
 * @param topK Number of experts to select per token
 * @param nGroup Number of expert groups
 * @param topkGroup Number of groups to select
 * @param normTopkProb Whether to renormalize topK weights to sum to 1
 * @param routedScalingFactor Scaling factor applied to final weights
 * @param stream CUDA stream for execution
 * @param correctionBias Optional bias tensor [numExperts] for expert load balancing (FP32, GPU)
 */
void moeSigmoidGroupTopk(rt::Tensor const& gatingOutput, rt::Tensor& topkWeights, rt::Tensor& topkIndices, int32_t topK,
    int32_t nGroup, int32_t topkGroup, bool normTopkProb, float routedScalingFactor, cudaStream_t stream,
    rt::OptionalInputTensor correctionBias = std::nullopt);

} // namespace kernel
} // namespace trt_edgellm
