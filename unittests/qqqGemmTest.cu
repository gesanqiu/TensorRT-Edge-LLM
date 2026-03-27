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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/qqqGemmKernels/qqqGemm.h"
#include "naiveGemmReference.h"
#include "testUtils.h"

using namespace trt_edgellm;

// ============================================================================
// Marlin weight packing reference (per-channel mode)
// ============================================================================

namespace
{

// Build the Marlin tile permutation array (size 1024).
// interleave_idx selects per-channel [4,0,5,1,6,2,7,3] or per-group [0,2,4,6,1,3,5,7].
std::vector<int> buildMarlinPerm(bool perChannel)
{
    int interleave[8];
    if (perChannel)
    {
        int tmp[] = {4, 0, 5, 1, 6, 2, 7, 3};
        std::copy(tmp, tmp + 8, interleave);
    }
    else
    {
        int tmp[] = {0, 2, 4, 6, 1, 3, 5, 7};
        std::copy(tmp, tmp + 8, interleave);
    }

    std::vector<int> perm;
    perm.reserve(1024);
    for (int i = 0; i < 32; ++i)
    {
        std::vector<int> perm1;
        int col = i / 4;
        for (int block : {0, 1})
        {
            for (int d = 0; d < 4; ++d)
            {
                int row = 4 * (i % 4) + d;
                perm1.push_back(16 * row + col + 8 * block);
            }
        }
        for (int j = 0; j < 4; ++j)
            for (int p : perm1)
                perm.push_back(p + 256 * j);
    }

    // Apply interleave to each group of 8
    std::vector<int> result(perm.size());
    for (size_t g = 0; g < perm.size() / 8; ++g)
    {
        for (int k = 0; k < 8; ++k)
            result[g * 8 + k] = perm[g * 8 + interleave[k]];
    }
    return result;
}

// Build scale_perm_single (for per-channel scale permutation), size 32.
std::vector<int> buildScalePermSingle()
{
    std::vector<int> sp;
    sp.reserve(32);
    for (int i = 0; i < 4; ++i)
        for (int j : {0, 1, 8, 9, 16, 17, 24, 25})
            sp.push_back(2 * i + j);
    return sp;
}

// Pack INT4 weights [K, N] (values in [-7, 7]) into Marlin format [K/16, 2*N] as int32.
// Also produces s_channel_out [N] = s_original[n] / 16, permuted by scale_perm_single.
// This is the per-channel mode reference packer.
void marlinPackPerChannel(int8_t const* w_KxN, float const* s_original_N, int K, int N,
    int32_t* packed_out, float* s_channel_out)
{
    auto perm = buildMarlinPerm(/*perChannel=*/true);
    auto scale_perm_single = buildScalePermSingle();

    // Step 1: Tile permutation
    // w[K, N] → reshape[K/16, 16, N/16, 16] → permute[K/16, N/16, 16, 16] → reshape[K/16, N*16]
    int const KT = K / 16;
    int const NT = N / 16;
    std::vector<int8_t> tiled(static_cast<size_t>(KT) * N * 16);
    for (int kt = 0; kt < KT; ++kt)
        for (int nt = 0; nt < NT; ++nt)
            for (int ti = 0; ti < 16; ++ti)
                for (int tj = 0; tj < 16; ++tj)
                {
                    int src_k = kt * 16 + ti;
                    int src_n = nt * 16 + tj;
                    int dst_col = nt * 16 * 16 + ti * 16 + tj;
                    tiled[static_cast<size_t>(kt) * (N * 16) + dst_col]
                        = w_KxN[static_cast<size_t>(src_k) * N + src_n];
                }

    // Step 2: Apply perm in chunks of 1024
    int const rowLen = N * 16;
    std::vector<int8_t> permuted(tiled.size());
    for (int r = 0; r < KT; ++r)
    {
        for (int chunk = 0; chunk < rowLen / 1024; ++chunk)
        {
            for (int i = 0; i < 1024; ++i)
            {
                permuted[r * rowLen + chunk * 1024 + i]
                    = tiled[r * rowLen + chunk * 1024 + perm[i]];
            }
        }
    }

    // Step 3: Pack 8 consecutive nibbles per int32.
    // Python: q |= (res[:, i::8] & 0xF) << 4*i  →  res[:, i::8][r, c] = res[r, 8*c + i]
    int const packCols = rowLen / 8; // = 2*N
    for (int r = 0; r < KT; ++r)
    {
        for (int c = 0; c < packCols; ++c)
        {
            uint32_t q = 0;
            for (int i = 0; i < 8; ++i)
            {
                uint32_t val = static_cast<uint32_t>(permuted[r * rowLen + 8 * c + i]) & 0xF;
                q |= val << (4 * i);
            }
            packed_out[r * packCols + c] = static_cast<int32_t>(q);
        }
    }

    // Step 4: Permute and scale s_channel
    // s_stored = s_original / 16, then apply scale_perm_single in groups of 32
    std::vector<float> s_div16(N);
    for (int n = 0; n < N; ++n)
        s_div16[n] = s_original_N[n] / 16.0f;

    for (int base = 0; base < N; base += 32)
    {
        for (int i = 0; i < 32 && base + i < N; ++i)
            s_channel_out[base + i] = s_div16[base + scale_perm_single[i]];
    }
}

// Build scale_perm (for per-group scale permutation), size 64.
std::vector<int> buildScalePerm()
{
    std::vector<int> sp;
    sp.reserve(64);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            sp.push_back(i + 8 * j);
    return sp;
}

// Pack unsigned INT4 weights [K, N] (values in [0, 15]) into Marlin per-group format.
// s_extra_N [N]: per-channel FP32 scale (permuted by scale_perm_single → s2_out)
// s_group_raw [K/gs, N]: per-group FP16 scale (permuted by scale_perm → s3_out)
void marlinPackPerGroup(uint8_t const* w_KxN, float const* s_extra_N, half const* s_group_raw,
    int K, int N, int groupSize, int32_t* packed_out, float* s2_out, half* s3_out)
{
    auto perm = buildMarlinPerm(/*perChannel=*/false);
    auto scale_perm = buildScalePerm();
    auto scale_perm_single = buildScalePermSingle();

    int const KT = K / 16;
    int const NT = N / 16;
    int const numGroups = K / groupSize;

    // Step 1: Tile permutation (identical to per-channel)
    std::vector<uint8_t> tiled(static_cast<size_t>(KT) * N * 16);
    for (int kt = 0; kt < KT; ++kt)
        for (int nt = 0; nt < NT; ++nt)
            for (int ti = 0; ti < 16; ++ti)
                for (int tj = 0; tj < 16; ++tj)
                {
                    int src_k = kt * 16 + ti;
                    int src_n = nt * 16 + tj;
                    int dst_col = nt * 256 + ti * 16 + tj;
                    tiled[static_cast<size_t>(kt) * (N * 16) + dst_col]
                        = w_KxN[static_cast<size_t>(src_k) * N + src_n];
                }

    // Step 2: Apply perm in chunks of 1024
    int const rowLen = N * 16;
    std::vector<uint8_t> permuted(tiled.size());
    for (int r = 0; r < KT; ++r)
        for (int chunk = 0; chunk < rowLen / 1024; ++chunk)
            for (int i = 0; i < 1024; ++i)
                permuted[r * rowLen + chunk * 1024 + i]
                    = tiled[r * rowLen + chunk * 1024 + perm[i]];

    // Step 3: Pack 8 nibbles per int32 (same stride-8 layout as per-channel)
    int const packCols = rowLen / 8;
    for (int r = 0; r < KT; ++r)
    {
        for (int c = 0; c < packCols; ++c)
        {
            uint32_t q = 0;
            for (int i = 0; i < 8; ++i)
            {
                uint32_t val = static_cast<uint32_t>(permuted[r * rowLen + 8 * c + i]) & 0xF;
                q |= val << (4 * i);
            }
            packed_out[r * packCols + c] = static_cast<int32_t>(q);
        }
    }

    // Step 4: Permute s2 (per-channel FP32) with scale_perm_single in groups of 32
    for (int base = 0; base < N; base += 32)
        for (int i = 0; i < 32 && base + i < N; ++i)
            s2_out[base + i] = s_extra_N[base + scale_perm_single[i]];

    // Step 5: Permute s3 (per-group FP16) with scale_perm in groups of 64
    for (int g = 0; g < numGroups; ++g)
        for (int base = 0; base < N; base += 64)
            for (int i = 0; i < 64 && base + i < N; ++i)
                s3_out[g * N + base + i] = s_group_raw[g * N + base + scale_perm[i]];
}

} // anonymous namespace

// ============================================================================
// Test: Dynamic per-token INT8 quantization
// ============================================================================

void TestDynamicQuantize(int M, int K)
{
    std::vector<half> h_input(static_cast<size_t>(M) * K);
    uniformFloatInitialization(h_input, -2.0f, 2.0f);

    half* d_input = nullptr;
    int8_t* d_output = nullptr;
    float* d_scales = nullptr;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaMallocAsync(&d_input, M * K * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_output, M * K * sizeof(int8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scales, M * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), M * K * sizeof(half), cudaMemcpyHostToDevice, stream));

    kernel::dynamicQuantize(d_input, d_output, d_scales, M, K, stream);

    std::vector<int8_t> h_output(static_cast<size_t>(M) * K);
    std::vector<float> h_scales(M);
    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_output, M * K * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_scales.data(), d_scales, M * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CPU reference
    for (int m = 0; m < M; ++m)
    {
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            float v = __half2float(h_input[m * K + k]);
            max_abs = std::max(max_abs, std::fabs(v));
        }
        float ref_scale = max_abs / 127.0f;

        EXPECT_NEAR(h_scales[m], ref_scale, ref_scale * 1e-5f + 1e-10f)
            << "Scale mismatch at row " << m;

        float inv_scale = (ref_scale > 0.0f) ? (1.0f / ref_scale) : 0.0f;
        for (int k = 0; k < K; ++k)
        {
            float v = __half2float(h_input[m * K + k]);
            int ref_q = static_cast<int>(std::round(v * inv_scale));
            ref_q = std::max(-128, std::min(127, ref_q));
            int actual_q = static_cast<int>(h_output[m * K + k]);
            EXPECT_LE(std::abs(actual_q - ref_q), 1)
                << "Quant mismatch at (" << m << ", " << k << "): got " << actual_q << " vs ref " << ref_q;
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_input, stream));
    CUDA_CHECK(cudaFreeAsync(d_output, stream));
    CUDA_CHECK(cudaFreeAsync(d_scales, stream));
}

TEST(QqqGemmTest, dynamicQuantizeSmall)
{
    TestDynamicQuantize(1, 64);
    TestDynamicQuantize(4, 128);
}

TEST(QqqGemmTest, dynamicQuantizeLarge)
{
    TestDynamicQuantize(32, 4096);
    TestDynamicQuantize(128, 4096);
}

// ============================================================================
// Test: W4A8 GEMM per-channel accuracy
// ============================================================================

void TestQqqGemmPerChannel(int M, int N, int K)
{
    std::mt19937 rng(42);

    // Generate random FP16 activations
    std::vector<half> h_act(static_cast<size_t>(M) * K);
    uniformFloatInitialization(h_act, -1.0f, 1.0f);

    // Generate random INT4 weights in [-7, 7] and per-channel scales
    std::vector<int8_t> w_int4(static_cast<size_t>(K) * N);
    std::uniform_int_distribution<int> wdist(-7, 7);
    for (auto& v : w_int4)
        v = static_cast<int8_t>(wdist(rng));

    std::vector<float> s_original(N);
    std::uniform_real_distribution<float> sdist(0.01f, 0.1f);
    for (auto& v : s_original)
        v = sdist(rng);

    // Pack weights into Marlin format
    std::vector<int32_t> h_packed(static_cast<size_t>(K / 16) * 2 * N);
    std::vector<float> h_s_channel(N);
    marlinPackPerChannel(w_int4.data(), s_original.data(), K, N, h_packed.data(), h_s_channel.data());

    // Build FP16 reference weight: W_ref[k,n] = w_int4[k,n] * s_original[n]
    std::vector<half> w_ref_fp16(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            w_ref_fp16[k * N + n] = __float2half(
                static_cast<float>(w_int4[k * N + n]) * s_original[n]);

    cudaStream_t stream = nullptr;
    constexpr int kMaxPar = 16;

    // Allocate device memory
    half *d_act = nullptr, *d_out = nullptr, *d_out_ref = nullptr, *d_w_ref = nullptr;
    int32_t *d_packed = nullptr, *d_reduce = nullptr, *d_locks = nullptr;
    float *d_s_channel = nullptr;
    int8_t *d_quant_a = nullptr;
    float *d_s1 = nullptr;

    size_t packedBytes = h_packed.size() * sizeof(int32_t);
    size_t reduceBytes = static_cast<size_t>(kMaxPar) * 64 * N * sizeof(int32_t);
    size_t lockBytes = static_cast<size_t>((N + 127) / 128) * kMaxPar * sizeof(int32_t);

    CUDA_CHECK(cudaMallocAsync(&d_act, M * K * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_packed, packedBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_s_channel, N * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_quant_a, M * K * sizeof(int8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_s1, M * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_reduce, reduceBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_locks, lockBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, M * N * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_w_ref, K * N * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_ref, M * N * sizeof(half), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_act, h_act.data(), M * K * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_packed, h_packed.data(), packedBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_s_channel, h_s_channel.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w_ref, w_ref_fp16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_locks, 0, lockBytes, stream));

    // Step 1: Dynamic quantization
    kernel::dynamicQuantize(d_act, d_quant_a, d_s1, M, K, stream);

    // Step 2: W4A8 GEMM
    int err = kernel::qqqGemmForward(
        d_quant_a, d_packed, d_reduce, d_out, d_s1, d_s_channel, nullptr,
        M, N, K, d_locks, /*groupsize=*/-1, /*dev=*/0, stream);
    ASSERT_EQ(err, 0) << "qqqGemmForward returned error " << err;

    // Step 3: Reference FP16 GEMM
    naive_gemm_forward(d_act, d_w_ref, d_out_ref, M, N, K, stream);

    // Copy results back
    std::vector<half> h_out(static_cast<size_t>(M) * N);
    std::vector<half> h_out_ref(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpyAsync(h_out.data(), d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_ref.data(), d_out_ref, M * N * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare with tolerance (INT8 quantization introduces rounding error)
    int const total = M * N;
    float maxAbsErr = 0.0f;
    float maxRelErr = 0.0f;
    int failCount = 0;
    for (int i = 0; i < total; ++i)
    {
        float o = __half2float(h_out[i]);
        float r = __half2float(h_out_ref[i]);
        float absErr = std::fabs(o - r);
        float relErr = (std::fabs(r) > 1e-6f) ? (absErr / std::fabs(r)) : absErr;
        maxAbsErr = std::max(maxAbsErr, absErr);
        maxRelErr = std::max(maxRelErr, relErr);
        if (!isclose(o, r, 0.15f, 0.15f))
            failCount++;
    }
    std::printf("[M=%d, N=%d, K=%d per-channel] maxAbsErr=%.4f maxRelErr=%.4f fails=%d/%d\n",
        M, N, K, maxAbsErr, maxRelErr, failCount, total);
    EXPECT_LT(failCount, total / 10) << "More than 10% elements exceed tolerance";

    CUDA_CHECK(cudaFreeAsync(d_act, stream));
    CUDA_CHECK(cudaFreeAsync(d_packed, stream));
    CUDA_CHECK(cudaFreeAsync(d_s_channel, stream));
    CUDA_CHECK(cudaFreeAsync(d_quant_a, stream));
    CUDA_CHECK(cudaFreeAsync(d_s1, stream));
    CUDA_CHECK(cudaFreeAsync(d_reduce, stream));
    CUDA_CHECK(cudaFreeAsync(d_locks, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaFreeAsync(d_w_ref, stream));
    CUDA_CHECK(cudaFreeAsync(d_out_ref, stream));
}

TEST(QqqGemmTest, perChannelAccuracy)
{
    TestQqqGemmPerChannel(16, 128, 128);
    TestQqqGemmPerChannel(64, 256, 256);
}

TEST(QqqGemmTest, perChannelLargeBatch)
{
    TestQqqGemmPerChannel(128, 256, 256);
}

// ============================================================================
// Test: W4A8 GEMM per-group accuracy
// ============================================================================

void TestQqqGemmPerGroup(int M, int N, int K, int gs)
{
    std::mt19937 rng(123);

    // FP16 activations
    std::vector<half> h_act(static_cast<size_t>(M) * K);
    uniformFloatInitialization(h_act, -1.0f, 1.0f);

    // Unsigned INT4 weights [0, 15]
    std::vector<uint8_t> w_uint4(static_cast<size_t>(K) * N);
    std::uniform_int_distribution<int> wdist(0, 15);
    for (auto& v : w_uint4)
        v = static_cast<uint8_t>(wdist(rng));

    // Per-group FP16 scale s3 [K/gs, N].
    // Keep values moderate so round((v-8)*s3) stays within INT8 range.
    int const numGroups = K / gs;
    std::vector<half> s3_raw(static_cast<size_t>(numGroups) * N);
    std::uniform_real_distribution<float> s3dist(0.5f, 10.0f);
    for (auto& v : s3_raw)
        v = __float2half(s3dist(rng));

    // Per-channel FP32 extra scale s2 [N]
    std::vector<float> s2_raw(N);
    std::uniform_real_distribution<float> s2dist(0.01f, 0.1f);
    for (auto& v : s2_raw)
        v = s2dist(rng);

    // Pack into Marlin format
    std::vector<int32_t> h_packed(static_cast<size_t>(K / 16) * 2 * N);
    std::vector<float> h_s2(N);
    std::vector<half> h_s3(static_cast<size_t>(numGroups) * N);
    marlinPackPerGroup(w_uint4.data(), s2_raw.data(), s3_raw.data(),
        K, N, gs, h_packed.data(), h_s2.data(), h_s3.data());

    // Build FP16 reference weight:
    // w_ref[k,n] = (w_uint4[k,n] - 8) * float(s3_raw[k/gs, n]) * s2_raw[n]
    std::vector<half> w_ref_fp16(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            w_ref_fp16[k * N + n] = __float2half(
                (static_cast<int>(w_uint4[k * N + n]) - 8)
                * __half2float(s3_raw[(k / gs) * N + n])
                * s2_raw[n]);

    cudaStream_t stream = nullptr;
    constexpr int kMaxPar = 16;

    // Device allocations
    half *d_act = nullptr, *d_out = nullptr, *d_out_ref = nullptr, *d_w_ref = nullptr;
    int32_t *d_packed = nullptr, *d_reduce = nullptr, *d_locks = nullptr;
    float *d_s2 = nullptr, *d_s1 = nullptr;
    half *d_s3 = nullptr;
    int8_t *d_quant_a = nullptr;

    size_t packedBytes = h_packed.size() * sizeof(int32_t);
    size_t reduceBytes = static_cast<size_t>(kMaxPar) * 64 * N * sizeof(int32_t);
    size_t lockBytes = static_cast<size_t>((N + 127) / 128) * kMaxPar * sizeof(int32_t);
    size_t s3Bytes = h_s3.size() * sizeof(half);

    CUDA_CHECK(cudaMallocAsync(&d_act, M * K * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_packed, packedBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_s2, N * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_s3, s3Bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_quant_a, M * K * sizeof(int8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_s1, M * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_reduce, reduceBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_locks, lockBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, M * N * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_w_ref, K * N * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_ref, M * N * sizeof(half), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_act, h_act.data(), M * K * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_packed, h_packed.data(), packedBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_s2, h_s2.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_s3, h_s3.data(), s3Bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_w_ref, w_ref_fp16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_locks, 0, lockBytes, stream));

    // Dynamic quantization
    kernel::dynamicQuantize(d_act, d_quant_a, d_s1, M, K, stream);

    // W4A8 GEMM (per-group)
    int err = kernel::qqqGemmForward(
        d_quant_a, d_packed, d_reduce, d_out, d_s1, d_s2, d_s3,
        M, N, K, d_locks, /*groupsize=*/gs, /*dev=*/0, stream);
    ASSERT_EQ(err, 0) << "qqqGemmForward per-group returned error " << err;

    // Reference FP16 GEMM
    naive_gemm_forward(d_act, d_w_ref, d_out_ref, M, N, K, stream);

    // Copy results
    std::vector<half> h_out(static_cast<size_t>(M) * N);
    std::vector<half> h_out_ref(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpyAsync(h_out.data(), d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_ref.data(), d_out_ref, M * N * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare — per-group has double quantization (activation INT8 + on-the-fly weight re-quant)
    // so tolerance is wider than per-channel
    int const total = M * N;
    float maxAbsErr = 0.0f;
    float maxRelErr = 0.0f;
    int failCount = 0;
    for (int i = 0; i < total; ++i)
    {
        float o = __half2float(h_out[i]);
        float r = __half2float(h_out_ref[i]);
        float absErr = std::fabs(o - r);
        float relErr = (std::fabs(r) > 1e-6f) ? (absErr / std::fabs(r)) : absErr;
        maxAbsErr = std::max(maxAbsErr, absErr);
        maxRelErr = std::max(maxRelErr, relErr);
        if (!isclose(o, r, 0.2f, 0.2f))
            failCount++;
    }
    std::printf("[M=%d, N=%d, K=%d, gs=%d per-group] maxAbsErr=%.4f maxRelErr=%.4f fails=%d/%d\n",
        M, N, K, gs, maxAbsErr, maxRelErr, failCount, total);
    EXPECT_LT(failCount, total / 10) << "More than 10% elements exceed tolerance";

    CUDA_CHECK(cudaFreeAsync(d_act, stream));
    CUDA_CHECK(cudaFreeAsync(d_packed, stream));
    CUDA_CHECK(cudaFreeAsync(d_s2, stream));
    CUDA_CHECK(cudaFreeAsync(d_s3, stream));
    CUDA_CHECK(cudaFreeAsync(d_quant_a, stream));
    CUDA_CHECK(cudaFreeAsync(d_s1, stream));
    CUDA_CHECK(cudaFreeAsync(d_reduce, stream));
    CUDA_CHECK(cudaFreeAsync(d_locks, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaFreeAsync(d_w_ref, stream));
    CUDA_CHECK(cudaFreeAsync(d_out_ref, stream));
}

TEST(QqqGemmTest, perGroupAccuracy)
{
    TestQqqGemmPerGroup(16, 128, 256, 128);
    TestQqqGemmPerGroup(64, 256, 256, 128);
}

TEST(QqqGemmTest, perGroupLargeBatch)
{
    TestQqqGemmPerGroup(128, 256, 512, 128);
}
