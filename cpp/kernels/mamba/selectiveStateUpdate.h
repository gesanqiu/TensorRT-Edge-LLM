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

/*
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Extracted selective state update kernel interface for TensorRT Edge-LLM integration
 * - Added explicit stride parameters for non-contiguous/padded memory layouts
 * - Renamed namespace from flashinfer::mamba to mamba_ssm
 */

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace mamba_ssm
{

/*!
 * \brief Launch the decode selective state update kernel (seq_len == 1).
 *
 * Computes:
 *   new_state = state * exp(A * dt) + B * dt * x
 *   output    = sum_i(new_state_i * C_i) + D * x
 *   if z is present: output *= silu(z)
 *
 * x:       [batch, nheads, dim]
 * A:       [nheads], FP32
 * B, C:    [batch, ngroups, dstate]
 * dt:      [batch, nheads]
 * dt_bias: [nheads] (optional)
 * D:       [nheads] (optional skip connection)
 * z:       same shape as x (optional SiLU gate)
 * state:   [batch, nheads, dim, dstate], updated in-place
 * output:  [batch, nheads, dim]
 */
void invokeSelectiveStateUpdate(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, cudaStream_t stream);

/*!
 * \brief Launch the prefill selective state update kernel (seq_len > 1).
 *
 * Processes all seq_len tokens in a single kernel launch. x must be 4D:
 * [batch, seq_len, nheads, dim].
 */
void invokeSelectiveStateUpdatePrefill(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& A,
    trt_edgellm::rt::Tensor const& B, trt_edgellm::rt::Tensor const& C, trt_edgellm::rt::Tensor const& dt,
    trt_edgellm::rt::OptionalInputTensor dt_bias, trt_edgellm::rt::OptionalInputTensor D,
    trt_edgellm::rt::OptionalInputTensor z, trt_edgellm::rt::Tensor& state, trt_edgellm::rt::Tensor& output,
    bool dt_softplus, trt_edgellm::rt::OptionalInputTensor contextLengths, cudaStream_t stream);

} // namespace mamba_ssm
