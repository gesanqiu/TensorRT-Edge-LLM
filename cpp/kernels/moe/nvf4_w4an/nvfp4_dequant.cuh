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

// NVFP4 (E2M1) dequantization: converts packed FP4 values to FP16.
// Separated from moe_marlin/marlin/dequant.h because FP4 support is
// specific to the NVFP4 MoE decode path (requires CUDA >= 12.8).

#pragma once
#include "common/cudaMacros.h"
#include "kernels/moe/moe_marlin/marlin/dequant.h"

#if SUPPORTS_FP4

namespace MARLIN_NAMESPACE_NAME
{

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kFE2M1f.id()>(int q, half2* frag_b)
{
    // FP4 is packed as 2 values per byte. The Marlin layout used here puts the relevant bytes in positions 1 and 3.
    // Note: reverse indexing is intentional because weights are permuted.
    __nv_fp4x2_storage_t const fp4x2_0 = static_cast<__nv_fp4x2_storage_t>((q) & 0xFF);
    __nv_fp4x2_storage_t const fp4x2_1 = static_cast<__nv_fp4x2_storage_t>((q >> 8) & 0xFF);
    __nv_fp4x2_storage_t const fp4x2_2 = static_cast<__nv_fp4x2_storage_t>((q >> 16) & 0xFF);
    __nv_fp4x2_storage_t const fp4x2_3 = static_cast<__nv_fp4x2_storage_t>((q >> 24) & 0xFF);

    __half2_raw const h2_0 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_0, __NV_E2M1);
    __half2_raw const h2_1 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_1, __NV_E2M1);
    __half2_raw const h2_2 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_2, __NV_E2M1);
    __half2_raw const h2_3 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_3, __NV_E2M1);

    frag_b[0] = *reinterpret_cast<half2 const*>(&h2_0);
    frag_b[1] = *reinterpret_cast<half2 const*>(&h2_1);
    frag_b[2] = *reinterpret_cast<half2 const*>(&h2_2);
    frag_b[3] = *reinterpret_cast<half2 const*>(&h2_3);
}

} // namespace MARLIN_NAMESPACE_NAME

#endif // SUPPORTS_FP4
