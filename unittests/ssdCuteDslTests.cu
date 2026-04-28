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

/// Unit tests for CuTe DSL SSD (Mamba2 chunk scan) prefill kernel.
/// Verifies correctness by comparing against the serial invokeSelectiveStateUpdatePrefill reference.

#ifdef CUTE_DSL_SSD_ENABLED

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/cuteDslSSDRunner.h"
#include "kernels/mamba/selectiveStateUpdate.h"

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// =============================================================================
// CPU Reference (token-by-token, same as invokeSelectiveStateUpdatePrefill)
// =============================================================================

float softplus(float x)
{
    return std::log(1.f + std::exp(x));
}

float thresholdedSoftplus(float x)
{
    constexpr float threshold = 20.f;
    return (x <= threshold) ? softplus(x) : x;
}

/// CPU reference for SSD prefill (sequential scan, matches selectiveStateUpdatePrefill).
void ssdPrefillReference(int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups,
    std::vector<half> const& x,       // [batch, seqLen, nheads, dim]
    std::vector<half> const& dt,      // [batch, seqLen, nheads]
    std::vector<float> const& A,      // [nheads]
    std::vector<half> const& B,       // [batch, seqLen, ngroups, dstate]
    std::vector<half> const& C,       // [batch, seqLen, ngroups, dstate]
    std::vector<float> const& D,      // [nheads]
    std::vector<float> const& dtBias, // [nheads]
    bool dtSoftplus,
    std::vector<float>& stateRef, // [batch, nheads, dim, dstate] — in/out
    std::vector<half>& outputRef  // [batch, seqLen, nheads, dim]
)
{
    int32_t const headsPerGroup = nheads / ngroups;

    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t t = 0; t < seqLen; ++t)
        {
            for (int32_t h = 0; h < nheads; ++h)
            {
                int32_t const g = h / headsPerGroup;
                float dtVal = __half2float(dt[((b * seqLen + t) * nheads) + h]) + dtBias[h];
                if (dtSoftplus)
                    dtVal = thresholdedSoftplus(dtVal);
                float const aVal = A[h];
                float const dA = std::exp(aVal * dtVal);
                float const dVal = D[h];

                for (int32_t d = 0; d < dim; ++d)
                {
                    float const xVal = __half2float(x[((b * seqLen + t) * nheads + h) * dim + d]);
                    float outVal = dVal * xVal;

                    for (int32_t ds = 0; ds < dstate; ++ds)
                    {
                        float const bVal = __half2float(B[((b * seqLen + t) * ngroups + g) * dstate + ds]);
                        float const cVal = __half2float(C[((b * seqLen + t) * ngroups + g) * dstate + ds]);

                        int64_t const sIdx
                            = static_cast<int64_t>(b) * nheads * dim * dstate + h * dim * dstate + d * dstate + ds;
                        float const newState = dA * stateRef[sIdx] + dtVal * bVal * xVal;
                        stateRef[sIdx] = newState;
                        outVal += newState * cVal;
                    }
                    outputRef[((b * seqLen + t) * nheads + h) * dim + d] = __float2half(outVal);
                }
            }
        }
    }
}

// =============================================================================
// Test fixture
// =============================================================================

struct SsdCuteDslTestConfig
{
    int32_t batch;
    int32_t seqLen;
    int32_t nheads;
    int32_t dim;
    int32_t dstate;
    int32_t ngroups;
};

class SsdCuteDslTest : public ::testing::TestWithParam<SsdCuteDslTestConfig>
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(CuteDslSSDRunner::loadKernelModules()) << "Failed to load SSD CuTe DSL kernel modules";
    }
};

TEST_P(SsdCuteDslTest, CorrectnessVsSerialReference)
{
    auto const& cfg = GetParam();
    int32_t const batch = cfg.batch;
    int32_t const seqLen = cfg.seqLen;
    int32_t const nheads = cfg.nheads;
    int32_t const dim = cfg.dim;
    int32_t const dstate = cfg.dstate;
    int32_t const ngroups = cfg.ngroups;

    if (!CuteDslSSDRunner::canImplement(dim, dstate, 80))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate;
    }

    // Allocate host data
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * seqLen * nheads * dim;
    size_t const dtSize = batch * seqLen * nheads;
    size_t const bSize = batch * seqLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;
    size_t const outSize = xSize;

    std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateHost(stateSize, 0.f);
    std::vector<half> outHost(outSize, __float2half(0.f));

    for (auto& v : xHost)
        v = __float2half(normal(rng));
    for (auto& v : dtHost)
        v = __float2half(uniform(rng));
    for (auto& v : bHost)
        v = __float2half(normal(rng));
    for (auto& v : cHost)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;

    // CPU reference
    std::vector<float> refState = stateHost;
    std::vector<half> refOut(outSize, __float2half(0.f));
    ssdPrefillReference(batch, seqLen, nheads, dim, dstate, ngroups, xHost, dtHost, aHost, bHost, cHost, dHost,
        dtBiasHost, true, refState, refOut);

    // GPU: allocate and copy
    void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutput;
    CUDA_CHECK(cudaMalloc(&dX, xSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDt, dtSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, outSize * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, dHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dState, stateHost.data(), stateSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0, outSize * sizeof(half)));

    // Allocate workspace for chunk scan intermediates
    size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
    void* dWorkspace = nullptr;
    if (wsSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&dWorkspace, wsSize));
        CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
    }

    // Run CuteDSL kernel
    SSDParams params{};
    params.x = dX;
    params.dt = dDt;
    params.A = dA;
    params.B = dB;
    params.C = dC;
    params.D = dD;
    params.dt_bias = dDtBias;
    params.z = nullptr;
    params.state = dState;
    params.output = dOutput;
    params.workspace = dWorkspace;
    params.batch = batch;
    params.seq_len = seqLen;
    params.nheads = nheads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = ngroups;
    params.smVersion = 80;
    params.dt_softplus = true;
    params.has_D = true;
    params.has_z = false;

    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.run(params, nullptr), 0) << "CuteDslSSDRunner::run failed";
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and compare
    std::vector<half> gpuOut(outSize);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOutput, outSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < outSize; ++i)
    {
        float const a = __half2float(gpuOut[i]);
        float const b = __half2float(refOut[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "Relative error " << relErr << " exceeds threshold. "
                             << "maxDiff=" << maxDiff << " refMax=" << refMax;

    // Cleanup
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dDt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dOutput));
    if (dWorkspace)
    {
        CUDA_CHECK(cudaFree(dWorkspace));
    }
}

// SM80 test configurations — all D×N combos: {128,64} × {128,64}
INSTANTIATE_TEST_SUITE_P(SsdCuteDslSM80, SsdCuteDslTest,
    ::testing::Values(
        // batch, seqLen, nheads, dim, dstate, ngroups
        // D=128, N=128
        SsdCuteDslTestConfig{1, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 1024, 8, 128, 128, 1},
        SsdCuteDslTestConfig{4, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{4, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 256, 64, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 64, 128, 128, 8},
        // D=64, N=128
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 128, 1},
        // D=128, N=64
        SsdCuteDslTestConfig{1, 128, 8, 128, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 128, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 128, 64, 1},
        // D=64, N=64
        SsdCuteDslTestConfig{1, 128, 8, 64, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 64, 1}),
    [](testing::TestParamInfo<SsdCuteDslTestConfig> const& info) {
        auto const& c = info.param;
        return "b" + std::to_string(c.batch) + "_s" + std::to_string(c.seqLen) + "_h" + std::to_string(c.nheads) + "_d"
            + std::to_string(c.dim) + "_ds" + std::to_string(c.dstate) + "_g" + std::to_string(c.ngroups);
    });

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED

// =============================================================================
// Blackwell test fixture (SM100+, dim=64, dstate=128)
// =============================================================================
// The Blackwell kernel is a single persistent kernel that takes pre-computed
// cumsum_delta and dt_processed as inputs (unlike the SM80 kernel which does
// cumsum internally). This test verifies end-to-end correctness via
// CuteDslSSDRunner::run() with smVersion=100.

class SsdCuteDslBlackwellTest : public ::testing::TestWithParam<SsdCuteDslTestConfig>
{
protected:
    void SetUp() override
    {
        // Check that GPU actually supports SM100+
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 10)
        {
            GTEST_SKIP() << "Blackwell tests require SM100+ GPU (got SM" << prop.major << prop.minor << ")";
        }
        ASSERT_TRUE(CuteDslSSDRunner::loadKernelModules()) << "Failed to load SSD CuTe DSL kernel modules";
    }
};

TEST_P(SsdCuteDslBlackwellTest, CorrectnessVsSerialReference)
{
    auto const& cfg = GetParam();
    int32_t const batch = cfg.batch;
    int32_t const seqLen = cfg.seqLen;
    int32_t const nheads = cfg.nheads;
    int32_t const dim = cfg.dim;
    int32_t const dstate = cfg.dstate;
    int32_t const ngroups = cfg.ngroups;

    if (!CuteDslSSDRunner::canImplement(dim, dstate, 100))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate << " for SM100";
    }

    // Allocate host data
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * seqLen * nheads * dim;
    size_t const dtSize = batch * seqLen * nheads;
    size_t const bSize = batch * seqLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;
    size_t const outSize = xSize;

    std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateHost(stateSize, 0.f);
    std::vector<half> outHost(outSize, __float2half(0.f));

    for (auto& v : xHost)
        v = __float2half(normal(rng));
    for (auto& v : dtHost)
        v = __float2half(uniform(rng));
    for (auto& v : bHost)
        v = __float2half(normal(rng));
    for (auto& v : cHost)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;

    // CPU reference
    std::vector<float> refState = stateHost;
    std::vector<half> refOut(outSize, __float2half(0.f));
    ssdPrefillReference(batch, seqLen, nheads, dim, dstate, ngroups, xHost, dtHost, aHost, bHost, cHost, dHost,
        dtBiasHost, true, refState, refOut);

    // GPU: allocate and copy
    void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutput;
    CUDA_CHECK(cudaMalloc(&dX, xSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDt, dtSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, outSize * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, dHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dState, stateHost.data(), stateSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0, outSize * sizeof(half)));

    size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
    void* dWorkspace = nullptr;
    if (wsSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&dWorkspace, wsSize));
        CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
    }

    SSDParams params{};
    params.x = dX;
    params.dt = dDt;
    params.A = dA;
    params.B = dB;
    params.C = dC;
    params.D = dD;
    params.dt_bias = dDtBias;
    params.z = nullptr;
    params.state = dState;
    params.output = dOutput;
    params.workspace = dWorkspace;
    params.batch = batch;
    params.seq_len = seqLen;
    params.nheads = nheads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = ngroups;
    params.smVersion = 100; // Force Blackwell path
    params.dt_softplus = true;
    params.has_D = true;
    params.has_z = false;

    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.run(params, nullptr), 0) << "CuteDslSSDRunner::run (Blackwell) failed";
    CUDA_CHECK(cudaDeviceSynchronize());

    // Runner now transposes y to params.output [B, S, H, D] and copies fstate to params.state.
    std::vector<half> gpuOut(outSize);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOutput, outSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < outSize; ++i)
    {
        float const a = __half2float(gpuOut[i]);
        float const b = __half2float(refOut[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "Relative error " << relErr << " exceeds threshold. "
                             << "maxDiff=" << maxDiff << " refMax=" << refMax;

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dDt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dOutput));
    if (dWorkspace)
    {
        CUDA_CHECK(cudaFree(dWorkspace));
    }
}

// Blackwell test configurations: D=64 (Blackwell TMA kernel) + D=128/N=64 (SM80 fallback)
INSTANTIATE_TEST_SUITE_P(SsdCuteDslBlackwell, SsdCuteDslBlackwellTest,
    ::testing::Values(
        // batch, seqLen, nheads, dim, dstate, ngroups
        // D=64, N=128: Blackwell persistent kernel (native)
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 1024, 8, 64, 128, 1},
        SsdCuteDslTestConfig{4, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 64, 64, 128, 1},
        SsdCuteDslTestConfig{1, 256, 64, 64, 128, 8},
        // D=64, N=64: Blackwell persistent kernel (native)
        SsdCuteDslTestConfig{1, 128, 8, 64, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 64, 1},
        // D=128, N=128: SM80 cp.async kernel running on Blackwell GPU (fallback)
        SsdCuteDslTestConfig{1, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 1024, 8, 128, 128, 1},
        // D=128, N=64: SM80 fallback
        SsdCuteDslTestConfig{1, 128, 8, 128, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 64, 1}),
    [](testing::TestParamInfo<SsdCuteDslTestConfig> const& info) {
        auto const& c = info.param;
        return "b" + std::to_string(c.batch) + "_s" + std::to_string(c.seqLen) + "_h" + std::to_string(c.nheads) + "_d"
            + std::to_string(c.dim) + "_ds" + std::to_string(c.dstate) + "_g" + std::to_string(c.ngroups);
    });

#endif // CUTE_DSL_SSD_BLACKWELL_ENABLED

} // anonymous namespace

#endif // CUTE_DSL_SSD_ENABLED
