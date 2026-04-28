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

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Quantize BF16/FP16 to packed FP4 with swizzled FP8 E4M3 scale factors.
/// SM100+: hardware E2M1 conversion. Pre-SM100: software fallback.
/// input shape [M, N]: M must be multiple of 128, N must be multiple of 16.
/// input dataType must be kBF16 or kHALF.
///
/// \c globalSF is the forward-direction activation global scale (e.g. `max|x|/(448*6)`);
/// the reciprocal consumed by the FP4 mapping is computed inside the kernel via a single
/// IEEE divide per thread.
void fp4Quantize(rt::Tensor const& input, rt::Tensor const& globalSF, rt::Tensor& outputFP4, rt::Tensor& outputSF,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
