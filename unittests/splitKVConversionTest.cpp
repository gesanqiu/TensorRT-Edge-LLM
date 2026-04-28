/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Unit tests for cvtKVLayoutBHSDToSplitKV.
//
// The kernel converts a KV-cache tensor [B, 2, H, S, D] into two separate
// tensors kDst [B, S, H, D] and vDst [B, S, H, D].  Two test strategies:
//
//   1. CPU reference  – fill src with known values on the host, compute the
//      expected K/V layout by hand, compare against GPU output.
//   2. FP8 dequant    – verify that scale factors are applied correctly when
//      the source is FP8 (compiled in only when SUPPORTS_FP8 == 1).

#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// Returns the flat index into a [B, 2, H, S, D] src tensor.
size_t srcIdx(int32_t b, int32_t kv, int32_t h, int32_t s, int32_t d, int32_t H, int32_t S, int32_t D)
{
    return (((((size_t) b * 2 + kv) * H + h) * S + s) * D + d);
}

// Returns the flat index into a [B, S, H, D] dst tensor.
size_t dstIdx(int32_t b, int32_t s, int32_t h, int32_t d, int32_t S, int32_t H, int32_t D)
{
    return ((((size_t) b * S + s) * H + h) * D + d);
}

struct SplitKVParams
{
    int32_t B, H, S, D;
};

} // namespace

// ===== 1. CPU reference test (FP16) ==============================================

class SplitKVCpuReferenceTest : public ::testing::TestWithParam<SplitKVParams>
{
};

TEST_P(SplitKVCpuReferenceTest, MatchesCpuReference)
{
    auto [B, H, S, D] = GetParam();

    cudaStream_t stream{nullptr};

    // Fill source on the host with random FP16 values.
    size_t const srcVol = (size_t) B * 2 * H * S * D;
    std::vector<half> srcHost(srcVol);
    uniformFloatInitialization(srcHost, -4.f, 4.f);

    // Upload to device.
    rt::Tensor srcTensor({B, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(srcTensor.rawPointer(), srcHost.data(), srcVol * sizeof(half), cudaMemcpyHostToDevice));

    // Allocate output tensors.
    rt::Tensor kTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor emptyScale{};
    kernel::cvtKVLayoutBHSDToSplitKV(srcTensor, kTensor, vTensor, emptyScale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pull results back to host.
    size_t const dstVol = (size_t) B * S * H * D;
    std::vector<half> kHost(dstVol), vHost(dstVol);
    CUDA_CHECK(cudaMemcpy(kHost.data(), kTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vHost.data(), vTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));

    // Verify against CPU reference.
    auto const [rtol, atol] = getTolerance<half>();
    for (int32_t b = 0; b < B; ++b)
        for (int32_t s = 0; s < S; ++s)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t d = 0; d < D; ++d)
                {
                    half expectedK = srcHost[srcIdx(b, 0, h, s, d, H, S, D)];
                    half expectedV = srcHost[srcIdx(b, 1, h, s, d, H, S, D)];
                    size_t outIdx = dstIdx(b, s, h, d, S, H, D);

                    ASSERT_TRUE(isclose(kHost[outIdx], expectedK, rtol, atol))
                        << "K mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d
                        << ": got=" << __half2float(kHost[outIdx]) << " expected=" << __half2float(expectedK);

                    ASSERT_TRUE(isclose(vHost[outIdx], expectedV, rtol, atol))
                        << "V mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d
                        << ": got=" << __half2float(vHost[outIdx]) << " expected=" << __half2float(expectedV);
                }
}

INSTANTIATE_TEST_SUITE_P(SplitKVShapes, SplitKVCpuReferenceTest,
    ::testing::Values(SplitKVParams{1, 4, 16, 64}, // small single-batch
        SplitKVParams{2, 8, 32, 128},              // multi-batch, head-size 128
        SplitKVParams{4, 2, 64, 64},               // larger sequence
        SplitKVParams{1, 1, 8, 64}                 // minimal dims
        ));

// ===== 2. FP8 dequantization test =================================================

#if SUPPORTS_FP8
TEST(SplitKVFP8Test, DequantizesWithScale)
{
    // Small shape for a targeted FP8 → FP16 dequant check.
    int32_t const B = 1, H = 2, S = 4, D = 8;

    cudaStream_t stream{nullptr};

    size_t const srcVol = (size_t) B * 2 * H * S * D;

    // Build FP8 source from known FP32 values so we can predict the dequant output.
    std::vector<float> srcFP32(srcVol);
    uniformFloatInitialization(srcFP32, 0.5f, 2.f); // positive range avoids FP8 sign edge cases

    std::vector<__nv_fp8_e4m3> srcFP8(srcVol);
    for (size_t i = 0; i < srcVol; ++i)
        srcFP8[i] = __nv_fp8_e4m3(srcFP32[i]);

    rt::Tensor srcTensor({B, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kFP8);
    CUDA_CHECK(
        cudaMemcpy(srcTensor.rawPointer(), srcFP8.data(), srcVol * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    // Scale factors: K scale = 2.0, V scale = 0.5
    float const kScale = 2.f, vScale = 0.5f;
    std::vector<float> scalesHost = {kScale, vScale};
    rt::Tensor scaleTensor({2}, rt::DeviceType::kGPU, DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpy(scaleTensor.rawPointer(), scalesHost.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));

    rt::Tensor kTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    kernel::cvtKVLayoutBHSDToSplitKV(srcTensor, kTensor, vTensor, scaleTensor, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const dstVol = (size_t) B * S * H * D;
    std::vector<half> kHost(dstVol), vHost(dstVol);
    CUDA_CHECK(cudaMemcpy(kHost.data(), kTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vHost.data(), vTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));

    // Expected: dequant(fp8_value) * scale, where dequant recovers the stored fp8 value.
    // FP8 → float → scale → half.  Use loose tolerance to account for FP8 quantization error.
    float const rtol = 0.1f, atol = 0.05f;
    for (int32_t b = 0; b < B; ++b)
        for (int32_t s = 0; s < S; ++s)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t d = 0; d < D; ++d)
                {
                    size_t const outIdx = dstIdx(b, s, h, d, S, H, D);
                    float const fp8ValK = static_cast<float>(srcFP8[srcIdx(b, 0, h, s, d, H, S, D)]);
                    float const fp8ValV = static_cast<float>(srcFP8[srcIdx(b, 1, h, s, d, H, S, D)]);

                    ASSERT_TRUE(isclose(kHost[outIdx], __float2half(fp8ValK * kScale), rtol, atol))
                        << "FP8 K dequant mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d;
                    ASSERT_TRUE(isclose(vHost[outIdx], __float2half(fp8ValV * vScale), rtol, atol))
                        << "FP8 V dequant mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d;
                }
}
#endif // SUPPORTS_FP8
