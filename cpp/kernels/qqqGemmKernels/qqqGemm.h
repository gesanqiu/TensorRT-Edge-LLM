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
 * Adapted from https://github.com/HandH1998/QQQ
 * Copyright (C) 2024 HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief Dynamic per-token INT8 quantization of FP16 activations.
 *
 * For each row i: scale[i] = max(|input[i,:]|) / 127, output[i,:] = round(input[i,:] / scale[i]) clamped to [-128,127]
 *
 * @param input   FP16 activation matrix [M, K]
 * @param output  INT8 quantized output [M, K]
 * @param scales  FP32 per-token scales [M]
 * @param m       Number of rows (tokens)
 * @param k       Number of columns (hidden dim)
 * @param stream  CUDA stream
 */
void dynamicQuantize(
    half const* input, int8_t* output, float* scales, int m, int k, cudaStream_t stream) noexcept;

/*!
 * @brief W4A8 GEMM: INT8 activations x Marlin-packed INT4 weights -> FP16 output.
 *
 * Computes D = dequant( A_int8 @ B_int4 ) using INT8 Tensor Cores, with on-the-fly INT4->INT8
 * weight conversion. Output is dequantized to FP16 using per-token activation scales (s1),
 * per-channel weight scales (s2), and optional per-group weight scales (s3).
 *
 * @param A         INT8 quantized activations [M, K]
 * @param B         Marlin-packed INT4 weights [K/16, 2*N] as int32 (8 nibbles per int32)
 * @param C         INT32 global reduce buffer [max_par*64, N]
 * @param D         FP16 output [M, N]
 * @param s1        FP32 per-token activation scales [M]
 * @param s2        FP32 per-channel weight scales [N]  (already incorporates /16 for per-channel mode)
 * @param s3        FP16 per-group weight scales [K/groupsize, N], nullptr for per-channel mode
 * @param prob_m    Batch dimension M
 * @param prob_n    Output dimension N
 * @param prob_k    Reduction dimension K
 * @param workspace INT32 lock array for cross-SM synchronization, size >= ceil(N/128)*max_par, must be zeroed
 * @param groupsize Weight quantization group size (-1 for per-channel, 128 for per-group)
 * @param dev       CUDA device ordinal
 * @param stream    CUDA stream
 * @param thread_k  User override for tile K dimension (-1 for auto)
 * @param thread_n  User override for tile N dimension (-1 for auto)
 * @param sms       Number of SMs to use (-1 for auto)
 * @param max_par   Maximum parallelism factor for large M (default 16)
 * @return 0 on success, non-zero on shape/config error
 */
int qqqGemmForward(void const* A, void const* B, void* C, void* D, void const* s1, void const* s2, void const* s3,
    int prob_m, int prob_n, int prob_k, void* workspace, int groupsize = -1, int dev = 0,
    cudaStream_t stream = nullptr, int thread_k = -1, int thread_n = -1, int sms = -1, int max_par = 16) noexcept;

} // namespace kernel
} // namespace trt_edgellm
