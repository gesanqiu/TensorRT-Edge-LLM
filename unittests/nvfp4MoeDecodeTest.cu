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

// NVFP4 MoE W4A16 and W4A4 decode kernel correctness tests.
// Tier-2 methodology: same quantized inputs, FP32 reference — isolates kernel error from quantization error.

#include "kernels/moe/nvf4_w4an/kernels.h"
#include "testUtils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#if SUPPORTS_FP4

using namespace trt_edgellm;

// ============================================================================
// Standalone tile dequant (avoids including marlin_template.cuh / dequant.h
// which requires --expt-relaxed-constexpr). Uses CUDA FP4 intrinsics directly.
// Matches the bit-exact dequant path of the production kernels.
// ============================================================================

// Dequant 4 packed E4M3 block scales from one int32 → 4 half values.
// Matches marlin::dequant_fp8_scales<half2, kFE4M3fn> layout.
__device__ __forceinline__ void dequantFp8ScalesToHalf(int q, __half* __restrict__ out4)
{
    // Marlin layout: reverse indexing (scales 2,3 from low bytes, 0,1 from high)
    int const out1 = (q & 0xFF00FF00) >> 1;
    int const shifted = q << 8;
    int const out2 = (shifted & 0xFF00FF00) >> 1;

    half2 const h2_1 = *reinterpret_cast<half2 const*>(&out1);
    half2 const h2_0 = *reinterpret_cast<half2 const*>(&out2);

    out4[0] = __low2half(h2_0);
    out4[1] = __high2half(h2_0);
    out4[2] = __low2half(h2_1);
    out4[3] = __high2half(h2_1);
}

// Dequant 8 packed FP4 E2M1 values from one uint32 → 4 half2 fragments.
// Matches marlin::dequant<half2, kFE2M1f> layout.
__device__ __forceinline__ void dequantFp4ToHalf2(uint32_t q, half2* __restrict__ frag4)
{
    __nv_fp4x2_storage_t const fp4x2_0 = static_cast<__nv_fp4x2_storage_t>((q) & 0xFF);
    __nv_fp4x2_storage_t const fp4x2_1 = static_cast<__nv_fp4x2_storage_t>((q >> 8) & 0xFF);
    __nv_fp4x2_storage_t const fp4x2_2 = static_cast<__nv_fp4x2_storage_t>((q >> 16) & 0xFF);
    __nv_fp4x2_storage_t const fp4x2_3 = static_cast<__nv_fp4x2_storage_t>((q >> 24) & 0xFF);

    __half2_raw const h2_0 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_0, __NV_E2M1);
    __half2_raw const h2_1 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_1, __NV_E2M1);
    __half2_raw const h2_2 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_2, __NV_E2M1);
    __half2_raw const h2_3 = __nv_cvt_fp4x2_to_halfraw2(fp4x2_3, __NV_E2M1);

    frag4[0] = *reinterpret_cast<half2 const*>(&h2_0);
    frag4[1] = *reinterpret_cast<half2 const*>(&h2_1);
    frag4[2] = *reinterpret_cast<half2 const*>(&h2_2);
    frag4[3] = *reinterpret_cast<half2 const*>(&h2_3);
}

// Core dequant: 64-element NVFP4 tile → 64 half values, given a pre-read block-scale word.
__device__ void testDequantNvfp4TileToHalfScaledImpl(NVFP4Tensor const& tensor, Dim3 const tile,
    int const globalScaleIndex, int const scaleWord, __half* __restrict__ out64)
{
    __half scaleH[4];
    dequantFp8ScalesToHalf(scaleWord, scaleH);

    int outIdx = 0;
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 blk;
        tensor.loadTileUint4(tile, chunk, blk);
        uint32_t const lanes[4] = {blk.x, blk.y, blk.z, blk.w};

        for (int lane = 0; lane < 4; ++lane)
        {
            int const base = (chunk * 4 + lane) * 8;
            int const sb = base / 16;
            __half const s = scaleH[sb];

            half2 frag[4];
            dequantFp4ToHalf2(lanes[lane], frag);

            for (int k = 0; k < 4; ++k)
            {
                out64[outIdx++] = __hmul(s, __low2half(frag[k]));
                out64[outIdx++] = __hmul(s, __high2half(frag[k]));
            }
        }
    }

    // Apply global scale
    __half const gs = __float2half(nvfp4TensorScaleAt(tensor.global_scale, globalScaleIndex));
    for (int i = 0; i < kNvfp4ElemsPerTile; ++i)
    {
        out64[i] = __hmul(out64[i], gs);
    }
}

// Dequant using atom-layout scale read (for weights).
__device__ void testDequantNvfp4TileToHalfScaled(
    NVFP4Tensor const& tensor, Dim3 const tile, int const globalScaleIndex, __half* __restrict__ out64)
{
    testDequantNvfp4TileToHalfScaledImpl(tensor, tile, globalScaleIndex, tensor.readBlockScaleWord(tile), out64);
}

// Dequant using linear scale read (for activations — plain Marlin-packed layout).
__device__ void testDequantNvfp4TileToHalfScaledLinear(
    NVFP4Tensor const& tensor, Dim3 const tile, int const globalScaleIndex, __half* __restrict__ out64)
{
    testDequantNvfp4TileToHalfScaledImpl(tensor, tile, globalScaleIndex, tensor.readBlockScaleWordLinear(tile), out64);
}

// ============================================================================
// Accuracy metrics (matching MoE.md methodology)
// ============================================================================

struct AccuracyMetrics
{
    double cosine;
    double magRatio;
    double medRelErr;
    double p95RelErr;
    int nanCount;
    int infCount;
};

static AccuracyMetrics computeMetrics(std::vector<float> const& actual, std::vector<float> const& expected)
{
    AccuracyMetrics m{};
    size_t const n = actual.size();

    double dotAE = 0.0, dotAA = 0.0, dotEE = 0.0;
    std::vector<double> relErrs;
    relErrs.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
        float const a = actual[i];
        float const e = expected[i];

        if (std::isnan(a))
        {
            ++m.nanCount;
        }
        if (std::isinf(a))
        {
            ++m.infCount;
        }

        dotAE += static_cast<double>(a) * static_cast<double>(e);
        dotAA += static_cast<double>(a) * static_cast<double>(a);
        dotEE += static_cast<double>(e) * static_cast<double>(e);

        double const denom = std::max(std::abs(static_cast<double>(e)), 1e-12);
        relErrs.push_back(std::abs(static_cast<double>(a) - static_cast<double>(e)) / denom);
    }

    double const normA = std::sqrt(dotAA);
    double const normE = std::sqrt(dotEE);
    m.cosine = (normA > 0.0 && normE > 0.0) ? (dotAE / (normA * normE)) : 0.0;
    m.magRatio = (normE > 0.0) ? (normA / normE) : 0.0;

    std::sort(relErrs.begin(), relErrs.end());
    m.medRelErr = relErrs.empty() ? 0.0 : relErrs[relErrs.size() / 2];
    m.p95RelErr = relErrs.empty() ? 0.0 : relErrs[static_cast<size_t>(relErrs.size() * 0.95)];

    return m;
}

static void printMetrics(char const* label, AccuracyMetrics const& m)
{
    printf("  [%s] cosine=%.6f  magRatio=%.4f  medRelErr=%.4e  p95RelErr=%.4e  NaN=%d  Inf=%d\n", label, m.cosine,
        m.magRatio, m.medRelErr, m.p95RelErr, m.nanCount, m.infCount);
}

// ============================================================================
// GPU dequant kernel: NVFP4 weight tiles → FP16 dense matrix
// ============================================================================

// Dequant up weights: tile layout [E, hidden_pos, inter_chunk] → dense [E, hidden_dim, inter_dim]
__global__ void dequantUpWeightsToFp16Kernel(
    NVFP4Tensor up, int hiddenDim, int interDim, int numExperts, __half* __restrict__ out)
{
    int const interChunks = interDim / kNvfp4ElemsPerTile;
    int const totalTiles = numExperts * hiddenDim * interChunks;
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalTiles)
    {
        return;
    }

    int const e = tid / (hiddenDim * interChunks);
    int const rem = tid % (hiddenDim * interChunks);
    int const j = rem / interChunks;
    int const c = rem % interChunks;

    __half tile64[kNvfp4ElemsPerTile];
    Dim3 const tile = make_int3(e, j, c);
    testDequantNvfp4TileToHalfScaled(up, tile, e, tile64);

    int64_t const baseIdx
        = static_cast<int64_t>(e) * hiddenDim * interDim + static_cast<int64_t>(j) * interDim + c * kNvfp4ElemsPerTile;
    for (int i = 0; i < kNvfp4ElemsPerTile; ++i)
    {
        out[baseIdx + i] = tile64[i];
    }
}

// Dequant down weights: tile layout [E, inter_pos, hidden_chunk] → dense [E, inter_dim, hidden_dim]
__global__ void dequantDownWeightsToFp16Kernel(
    NVFP4Tensor down, int hiddenDim, int interDim, int numExperts, __half* __restrict__ out)
{
    int const hiddenChunks = hiddenDim / kNvfp4ElemsPerTile;
    int const totalTiles = numExperts * interDim * hiddenChunks;
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalTiles)
    {
        return;
    }

    int const e = tid / (interDim * hiddenChunks);
    int const rem = tid % (interDim * hiddenChunks);
    int const j = rem / hiddenChunks;
    int const c = rem % hiddenChunks;

    __half tile64[kNvfp4ElemsPerTile];
    Dim3 const tile = make_int3(e, j, c);
    testDequantNvfp4TileToHalfScaled(down, tile, e, tile64);

    int64_t const baseIdx
        = static_cast<int64_t>(e) * interDim * hiddenDim + static_cast<int64_t>(j) * hiddenDim + c * kNvfp4ElemsPerTile;
    for (int i = 0; i < kNvfp4ElemsPerTile; ++i)
    {
        out[baseIdx + i] = tile64[i];
    }
}

// ============================================================================
// GPU dequant kernel: NVFP4 activation tiles → FP16 dense matrix
// ============================================================================

// Dequant activation: tile layout [numTokens, hiddenTile, 0] → dense [numTokens, hiddenDim]
__global__ void dequantActivationToFp16Kernel(NVFP4Tensor act, int hiddenDim, int numTokens, __half* __restrict__ out)
{
    int const numHiddenTiles = hiddenDim / kNvfp4ElemsPerTile;
    int const totalTiles = numTokens * numHiddenTiles;
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalTiles)
    {
        return;
    }

    int const t = tid / numHiddenTiles;
    int const tileIdx = tid % numHiddenTiles;

    __half tile64[kNvfp4ElemsPerTile];
    Dim3 const tile = make_int3(t, tileIdx, 0);
    testDequantNvfp4TileToHalfScaledLinear(act, tile, 0, tile64);

    int64_t const baseIdx = static_cast<int64_t>(t) * hiddenDim + tileIdx * kNvfp4ElemsPerTile;
    for (int i = 0; i < kNvfp4ElemsPerTile; ++i)
    {
        out[baseIdx + i] = tile64[i];
    }
}

// ============================================================================
// CPU reference MoE GEMV (FP32 math)
// ============================================================================

static float cpuActivation(float z, MoEActivationKind act)
{
    if (act == MoEActivationKind::kSiLU)
    {
        float const zc = std::fmin(std::fmax(z, -50.f), 50.f);
        return zc / (1.f + std::exp(-zc));
    }
    else
    {
        float const t = std::fmax(z, 0.f);
        return t * t;
    }
}

// Up-proj reference: out[t, k, n] = sum_j act[t, j] * upW[expert[t,k], j, n]
static void cpuUpProjRef(std::vector<float> const& actFp32, std::vector<float> const& upWFp32,
    std::vector<int32_t> const& expertIds, int numTokens, int topK, int hiddenDim, int interDim, int numExperts,
    std::vector<float>& interOut)
{
    interOut.assign(static_cast<size_t>(numTokens) * topK * interDim, 0.f);
    for (int t = 0; t < numTokens; ++t)
    {
        for (int k = 0; k < topK; ++k)
        {
            int const e = expertIds[t * topK + k];
            if (e < 0 || e >= numExperts)
            {
                continue;
            }
            for (int n = 0; n < interDim; ++n)
            {
                double acc = 0.0;
                for (int j = 0; j < hiddenDim; ++j)
                {
                    float const a = actFp32[t * hiddenDim + j];
                    float const w = upWFp32[static_cast<size_t>(e) * hiddenDim * interDim + j * interDim + n];
                    acc += static_cast<double>(a) * static_cast<double>(w);
                }
                interOut[(t * topK + k) * interDim + n] = static_cast<float>(acc);
            }
        }
    }
}

// Down-proj reference: out[t, h] = sum_k score[t,k] * sum_j act(inter[t,k,j]) * downW[expert[t,k], j, h]
static void cpuDownProjRef(std::vector<float> const& interFp32, std::vector<float> const& downWFp32,
    std::vector<int32_t> const& expertIds, std::vector<float> const& topkWeights, int numTokens, int topK,
    int hiddenDim, int interDim, int numExperts, MoEActivationKind actKind, std::vector<float>& outFp32)
{
    outFp32.assign(static_cast<size_t>(numTokens) * hiddenDim, 0.f);
    for (int t = 0; t < numTokens; ++t)
    {
        for (int k = 0; k < topK; ++k)
        {
            int const e = expertIds[t * topK + k];
            float const score = topkWeights[t * topK + k];
            if (e < 0 || e >= numExperts)
            {
                continue;
            }
            for (int h = 0; h < hiddenDim; ++h)
            {
                double acc = 0.0;
                for (int j = 0; j < interDim; ++j)
                {
                    float const z = interFp32[(t * topK + k) * interDim + j];
                    float const a = cpuActivation(z, actKind);
                    float const w = downWFp32[static_cast<size_t>(e) * interDim * hiddenDim + j * hiddenDim + h];
                    acc += static_cast<double>(a) * static_cast<double>(w);
                }
                outFp32[t * hiddenDim + h] += static_cast<float>(score * acc);
            }
        }
    }
}

// Full E2E reference
static void cpuE2ERef(std::vector<float> const& actFp32, std::vector<float> const& upWFp32,
    std::vector<float> const& downWFp32, std::vector<int32_t> const& expertIds, std::vector<float> const& topkWeights,
    int numTokens, int topK, int hiddenDim, int interDim, int numExperts, MoEActivationKind actKind,
    std::vector<float>& outFp32)
{
    std::vector<float> interFp32;
    cpuUpProjRef(actFp32, upWFp32, expertIds, numTokens, topK, hiddenDim, interDim, numExperts, interFp32);
    cpuDownProjRef(interFp32, downWFp32, expertIds, topkWeights, numTokens, topK, hiddenDim, interDim, numExperts,
        actKind, outFp32);
}

// ============================================================================
// Test fixture
// ============================================================================

class Nvfp4MoeDecodeTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    // Generate random NVFP4 payload bytes
    void generatePayload(std::vector<uint8_t>& bytes, size_t numBytes, std::mt19937& rng)
    {
        std::uniform_int_distribution<int> dist(0, 255);
        bytes.resize(numBytes);
        for (size_t i = 0; i < numBytes; ++i)
        {
            bytes[i] = static_cast<uint8_t>(dist(rng));
        }
    }

    // Generate random FP8 E4M3 block scales in a reasonable range
    void generateBlockScales(std::vector<uint8_t>& bytes, size_t numScaleInts, std::mt19937& rng)
    {
        std::uniform_int_distribution<int> dist(0x28, 0x50);
        bytes.resize(numScaleInts * 4);
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<uint8_t>(dist(rng));
        }
    }

    // Generate random FP32 global scales in [0.5, 2.0]
    void generateGlobalScales(std::vector<float>& scales, int numExperts, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> dist(0.5f, 2.0f);
        scales.resize(numExperts);
        for (int i = 0; i < numExperts; ++i)
        {
            scales[i] = dist(rng);
        }
    }

    struct WeightBuffers
    {
        int4* dPayload = nullptr;
        int* dBlockScale = nullptr;
        float* dGlobalScale = nullptr;
        size_t payloadBytes = 0;
        size_t scaleBytes = 0;
        size_t globalScaleBytes = 0;
        // Host-side copies for decode SF generation.
        std::vector<uint8_t> hostScales;
        int wDim0 = 0;
        int wDim1 = 0;
        int wNumExperts = 0;

        void free()
        {
            if (dPayload)
            {
                cudaFree(dPayload);
            }
            if (dBlockScale)
            {
                cudaFree(dBlockScale);
            }
            if (dGlobalScale)
            {
                cudaFree(dGlobalScale);
            }
        }
    };

    //! Atom-layout 128×4 byte offset for scale factor at logical (mIdx, kIdx) within one expert plane.
    //! Matches fp4Quantize.cu get_sf_out_offset_128x4.
    static size_t atomSfByteOffset(int mIdx, int kIdx, int numSfCols)
    {
        int const innerK = kIdx % 4;
        int const innerM = (mIdx % 128) / 32;
        int const outerM = mIdx % 32;
        int const kTile = kIdx / 4;
        int const numKTiles = (numSfCols + 3) / 4;
        int const mTile = mIdx / 128;
        return static_cast<size_t>(mTile) * numKTiles * 512 + static_cast<size_t>(kTile) * 512
            + static_cast<size_t>(outerM) * 16 + innerM * 4 + innerK;
    }

    //! Atom-layout buffer size in bytes for one expert's scale factor plane.
    static size_t atomSfBytesPerExpert(int mDim, int kDim, int sfVec = 16)
    {
        int const numSfCols = kDim / sfVec;
        int const paddedSfCols = ((numSfCols + 3) / 4) * 4;
        int const paddedM = ((mDim + 127) / 128) * 128;
        return static_cast<size_t>(paddedM) * paddedSfCols;
    }

    WeightBuffers allocateNvfp4Weight(int numExperts, int dim0, int dim1, std::mt19937& rng)
    {
        WeightBuffers buf;
        int const numTiles = numExperts * dim0 * (dim1 / kNvfp4ElemsPerTile);
        buf.payloadBytes = static_cast<size_t>(numTiles) * kNvfp4Int4PerTilePayload * sizeof(int4);

        // Atom-layout scale buffer: padded per-expert planes
        size_t const sfBytesPerEx = atomSfBytesPerExpert(dim0, dim1);
        // int32-aligned: round up to multiple of 4 bytes
        size_t const sfInt32PerEx = (sfBytesPerEx + 3) / 4;
        buf.scaleBytes = static_cast<size_t>(numExperts) * sfInt32PerEx * sizeof(int);
        buf.globalScaleBytes = static_cast<size_t>(numExperts) * sizeof(float);

        std::vector<uint8_t> payload;
        std::vector<float> globalScales;
        generatePayload(payload, buf.payloadBytes, rng);
        generateGlobalScales(globalScales, numExperts, rng);

        // Generate atom-layout block scales: write individual FP8 bytes at swizzled positions.
        // Weight tiles are [E, dim0_row, dim1_chunk] where dim1_chunk = dim1/64.
        // M = dim0, K = dim1; numSfCols = dim1/16; each tile has 4 SF columns.
        int const numSfCols = dim1 / 16;
        int const dim1Chunks = dim1 / kNvfp4ElemsPerTile;

        std::vector<uint8_t> scalesBuf(buf.scaleBytes, 0);
        std::uniform_int_distribution<int> scaleDist(0x28, 0x50);

        for (int e = 0; e < numExperts; ++e)
        {
            size_t const exByteBase = static_cast<size_t>(e) * sfInt32PerEx * 4; // byte offset of expert e
            for (int mRow = 0; mRow < dim0; ++mRow)
            {
                for (int c = 0; c < dim1Chunks; ++c)
                {
                    // Each tile (mRow, c) has 4 SF columns: c*4 .. c*4+3
                    for (int g = 0; g < 4; ++g)
                    {
                        int const sfCol = c * 4 + g;
                        size_t const byteOff = atomSfByteOffset(mRow, sfCol, numSfCols);
                        if (mRow % 16 == 0)
                        {
                            // First row in quant group: generate random scale.
                            scalesBuf[exByteBase + byteOff] = static_cast<uint8_t>(scaleDist(rng));
                        }
                        else
                        {
                            // Copy from the first row in the same quant group so all 16 rows share one scale.
                            int const groupLeader = (mRow / 16) * 16;
                            size_t const leaderOff = atomSfByteOffset(groupLeader, sfCol, numSfCols);
                            scalesBuf[exByteBase + byteOff] = scalesBuf[exByteBase + leaderOff];
                        }
                    }
                }
            }
        }

        buf.hostScales = scalesBuf;
        buf.wDim0 = dim0;
        buf.wDim1 = dim1;
        buf.wNumExperts = numExperts;

        CUDA_CHECK(cudaMalloc(&buf.dPayload, buf.payloadBytes));
        CUDA_CHECK(cudaMalloc(&buf.dBlockScale, buf.scaleBytes));
        CUDA_CHECK(cudaMalloc(&buf.dGlobalScale, buf.globalScaleBytes));

        CUDA_CHECK(cudaMemcpy(buf.dPayload, payload.data(), buf.payloadBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.dBlockScale, scalesBuf.data(), buf.scaleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.dGlobalScale, globalScales.data(), buf.globalScaleBytes, cudaMemcpyHostToDevice));

        return buf;
    }

    NVFP4Tensor makeNvfp4Tensor(
        WeightBuffers const& buf, int64_t s0, int64_t s1, int64_t s2, int mDim, int kDim, int numExperts)
    {
        NVFP4Tensor t{};
        t.quantized_data = buf.dPayload;
        t.block_scale = buf.dBlockScale;
        t.global_scale = buf.dGlobalScale;
        t.strides[0] = s0;
        t.strides[1] = s1;
        t.strides[2] = s2;
        // Atom-layout fields for weights: Dim3 = (expert=x, M_row=y, K_chunk=z)
        t.scaleMDimIdx = 1;
        t.scaleKDimIdx = 2;
        int const numSfCols = kDim / 16;
        t.scaleNumKTiles = (numSfCols + 3) / 4;
        int const numMTiles = (mDim + 127) / 128;
        t.scaleExpertStride = static_cast<int64_t>(numMTiles) * t.scaleNumKTiles * 128;
        return t;
    }

    // Dequant weights on GPU and copy to host as FP32
    std::vector<float> dequantWeightsToHostFp32(
        NVFP4Tensor const& tensor, int numExperts, int dim0, int dim1, bool isUp)
    {
        size_t const totalElems = static_cast<size_t>(numExperts) * dim0 * dim1;
        __half* dFp16 = nullptr;
        CUDA_CHECK(cudaMalloc(&dFp16, totalElems * sizeof(__half)));

        int const tilesPerDim = dim1 / kNvfp4ElemsPerTile;
        int const totalTiles = numExperts * dim0 * tilesPerDim;
        int const threads = 256;
        int const blocks = (totalTiles + threads - 1) / threads;

        if (isUp)
        {
            dequantUpWeightsToFp16Kernel<<<blocks, threads, 0, mStream>>>(tensor, dim0, dim1, numExperts, dFp16);
        }
        else
        {
            dequantDownWeightsToFp16Kernel<<<blocks, threads, 0, mStream>>>(tensor, dim1, dim0, numExperts, dFp16);
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));

        std::vector<__half> hostFp16(totalElems);
        CUDA_CHECK(cudaMemcpy(hostFp16.data(), dFp16, totalElems * sizeof(__half), cudaMemcpyDeviceToHost));
        cudaFree(dFp16);

        std::vector<float> hostFp32(totalElems);
        for (size_t i = 0; i < totalElems; ++i)
        {
            hostFp32[i] = __half2float(hostFp16[i]);
        }
        return hostFp32;
    }

    struct ActivationBuffers
    {
        int4* dPayload = nullptr;
        int* dBlockScale = nullptr;
        float* dGlobalScale = nullptr;
        size_t payloadBytes = 0;
        size_t scaleBytes = 0;

        void free()
        {
            if (dPayload)
            {
                cudaFree(dPayload);
            }
            if (dBlockScale)
            {
                cudaFree(dBlockScale);
            }
            if (dGlobalScale)
            {
                cudaFree(dGlobalScale);
            }
        }
    };

    ActivationBuffers allocateNvfp4Activation(int numTokens, int hiddenDim, std::mt19937& rng)
    {
        ActivationBuffers buf;
        int const numHiddenTiles = hiddenDim / kNvfp4ElemsPerTile;
        int const numTiles = numTokens * numHiddenTiles;
        buf.payloadBytes = static_cast<size_t>(numTiles) * kNvfp4Int4PerTilePayload * sizeof(int4);

        // Plain linear scale buffer: one Marlin-packed int32 per tile.
        buf.scaleBytes = static_cast<size_t>(numTiles) * sizeof(int);

        std::vector<uint8_t> payload;
        generatePayload(payload, buf.payloadBytes, rng);

        // Generate linear Marlin-packed activation block scales (one int32 per tile).
        std::vector<int> scalesHost(numTiles, 0);
        std::uniform_int_distribution<int> scaleDist(0x28, 0x50);
        for (int i = 0; i < numTiles; ++i)
        {
            // Pack 4 random FP8 E4M3 bytes in Marlin order {s0, s2, s1, s3}.
            uint8_t const s0 = static_cast<uint8_t>(scaleDist(rng));
            uint8_t const s1 = static_cast<uint8_t>(scaleDist(rng));
            uint8_t const s2 = static_cast<uint8_t>(scaleDist(rng));
            uint8_t const s3 = static_cast<uint8_t>(scaleDist(rng));
            // Marlin byte order: {s0, s2, s1, s3}
            scalesHost[i] = static_cast<int>(static_cast<uint32_t>(s0) | (static_cast<uint32_t>(s2) << 8)
                | (static_cast<uint32_t>(s1) << 16) | (static_cast<uint32_t>(s3) << 24));
        }

        std::uniform_real_distribution<float> dist(0.5f, 2.0f);
        float const globalScale = dist(rng);

        CUDA_CHECK(cudaMalloc(&buf.dPayload, buf.payloadBytes));
        CUDA_CHECK(cudaMalloc(&buf.dBlockScale, buf.scaleBytes));
        CUDA_CHECK(cudaMalloc(&buf.dGlobalScale, sizeof(float)));

        CUDA_CHECK(cudaMemcpy(buf.dPayload, payload.data(), buf.payloadBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.dBlockScale, scalesHost.data(), buf.scaleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(buf.dGlobalScale, &globalScale, sizeof(float), cudaMemcpyHostToDevice));

        return buf;
    }

    NVFP4Tensor makeNvfp4ActivationTensor(ActivationBuffers const& buf, int hiddenDim)
    {
        int const numHiddenTiles = hiddenDim / kNvfp4ElemsPerTile;
        int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
        NVFP4Tensor t{};
        t.quantized_data = buf.dPayload;
        t.block_scale = buf.dBlockScale;
        t.global_scale = buf.dGlobalScale;
        t.strides[0] = static_cast<int64_t>(numHiddenTiles) * int4PerTile;
        t.strides[1] = int4PerTile;
        t.strides[2] = 0;
        // Activation uses plain linear scale layout (readBlockScaleWordLinear) — no atom fields needed.
        return t;
    }

    //! Generate row-major decode SF [E, dim0/16, dim1] from atom-swizzled block scales.
    //! For up: dim0=H, dim1=I → decode SF [E, H/16, I].
    //! For down: dim0=I, dim1=H → decode SF [E, I/16, H].
    //! Each byte at decode_sf[e, group, pos] equals the atom byte at the group-leader row (group*16).
    uint8_t* generateDecodeSf(WeightBuffers const& wb)
    {
        int const dim0 = wb.wDim0;
        int const dim1 = wb.wDim1;
        int const numExperts = wb.wNumExperts;
        int const numSfCols = dim1 / 16;
        int const dim0Groups = dim0 / 16;
        size_t const sfBytesPerEx = atomSfBytesPerExpert(dim0, dim1);
        size_t const sfInt32PerEx = (sfBytesPerEx + 3) / 4;
        size_t const totalBytes = static_cast<size_t>(numExperts) * dim0Groups * dim1;

        std::vector<uint8_t> decodeSf(totalBytes, 0);
        for (int e = 0; e < numExperts; ++e)
        {
            size_t const exAtomBase = static_cast<size_t>(e) * sfInt32PerEx * 4;
            size_t const exDecBase = static_cast<size_t>(e) * dim0Groups * dim1;
            for (int g = 0; g < dim0Groups; ++g)
            {
                int const leaderRow = g * 16;
                for (int pos = 0; pos < dim1; ++pos)
                {
                    int const sfCol = pos / 16;
                    size_t const atomOff = atomSfByteOffset(leaderRow, sfCol, numSfCols);
                    decodeSf[exDecBase + static_cast<size_t>(g) * dim1 + pos] = wb.hostScales[exAtomBase + atomOff];
                }
            }
        }

        uint8_t* dDecodeSf = nullptr;
        CUDA_CHECK(cudaMalloc(&dDecodeSf, totalBytes));
        CUDA_CHECK(cudaMemcpy(dDecodeSf, decodeSf.data(), totalBytes, cudaMemcpyHostToDevice));
        return dDecodeSf;
    }

    std::vector<float> dequantActivationToHostFp32(NVFP4Tensor const& act, int numTokens, int hiddenDim)
    {
        size_t const totalElems = static_cast<size_t>(numTokens) * hiddenDim;
        __half* dFp16 = nullptr;
        CUDA_CHECK(cudaMalloc(&dFp16, totalElems * sizeof(__half)));

        int const numHiddenTiles = hiddenDim / kNvfp4ElemsPerTile;
        int const totalTiles = numTokens * numHiddenTiles;
        int const threads = 256;
        int const blocks = (totalTiles + threads - 1) / threads;

        dequantActivationToFp16Kernel<<<blocks, threads, 0, mStream>>>(act, hiddenDim, numTokens, dFp16);
        CUDA_CHECK(cudaStreamSynchronize(mStream));

        std::vector<__half> hostFp16(totalElems);
        CUDA_CHECK(cudaMemcpy(hostFp16.data(), dFp16, totalElems * sizeof(__half), cudaMemcpyDeviceToHost));
        cudaFree(dFp16);

        std::vector<float> hostFp32(totalElems);
        for (size_t i = 0; i < totalElems; ++i)
        {
            hostFp32[i] = __half2float(hostFp16[i]);
        }
        return hostFp32;
    }

    cudaStream_t mStream = nullptr;
};

// ============================================================================
// Test: Up-proj only (Tier-2, small dims)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_UpProj_Tier2_Small)
{
    int const H = 256;
    int const N = 256;
    int const E = 2;
    int const topK = 1;
    int const numTokens = 1;
    double const cosineThreshold = 0.99;

    std::mt19937 rng(42);

    // Activation: FP16 in [-1, 1]
    std::vector<__half> actHost(numTokens * H);
    {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& v : actHost)
        {
            v = __float2half(dist(rng));
        }
    }
    __half* dAct = nullptr;
    CUDA_CHECK(cudaMalloc(&dAct, actHost.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dAct, actHost.data(), actHost.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Up weights: tile grid [E, hidden_pos, inter_chunk]
    int const interChunks = N / kNvfp4ElemsPerTile;
    int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
    int64_t const upS0 = static_cast<int64_t>(H) * interChunks * int4PerTile;
    int64_t const upS1 = interChunks * int4PerTile;
    int64_t const upS2 = int4PerTile;

    auto upBuf = allocateNvfp4Weight(E, H, N, rng);
    NVFP4Tensor up = makeNvfp4Tensor(upBuf, upS0, upS1, upS2, H, N, E);

    // Expert IDs and topk weights
    std::vector<int32_t> expertIdsHost = {0};
    std::vector<float> topkWeightsHost = {1.0f};
    int32_t* dExpertIds = nullptr;
    float* dTopkWeights = nullptr;
    CUDA_CHECK(cudaMalloc(&dExpertIds, expertIdsHost.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dTopkWeights, topkWeightsHost.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(dExpertIds, expertIdsHost.data(), expertIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dTopkWeights, topkWeightsHost.data(), topkWeightsHost.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Decode SF for up: [E, H/16, N]
    uint8_t* dUpDecodeSf = generateDecodeSf(upBuf);

    // Inter scratch (FP16, zeroed by launch)
    int64_t const interElems = nemotronMoeW4A16InterBufferNumElems(numTokens, topK, N);
    __half* dInter = nullptr;
    CUDA_CHECK(cudaMalloc(&dInter, interElems * sizeof(__half)));

    // Launch up-proj kernel
    launchNemotronMoeW4A16DecodeUpGemvCuda(
        numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dTopkWeights, dAct, up, dUpDecodeSf, dInter, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Get kernel output
    std::vector<__half> interHost(interElems);
    CUDA_CHECK(cudaMemcpy(interHost.data(), dInter, interElems * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> kernelOut(interElems);
    for (int64_t i = 0; i < interElems; ++i)
    {
        kernelOut[i] = __half2float(interHost[i]);
    }

    // CPU reference: dequant weights → FP32, then FP32 GEMV
    std::vector<float> upWFp32 = dequantWeightsToHostFp32(up, E, H, N, true);
    std::vector<float> actFp32(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        actFp32[i] = __half2float(actHost[i]);
    }

    std::vector<float> refOut;
    cpuUpProjRef(actFp32, upWFp32, expertIdsHost, numTokens, topK, H, N, E, refOut);

    AccuracyMetrics m = computeMetrics(kernelOut, refOut);
    printMetrics("UpProj_Tier2_Small", m);

    EXPECT_EQ(m.nanCount, 0);
    EXPECT_EQ(m.infCount, 0);
    EXPECT_GT(m.cosine, cosineThreshold) << "Cosine similarity too low: " << m.cosine;

    cudaFree(dAct);
    cudaFree(dInter);
    cudaFree(dUpDecodeSf);
    cudaFree(dExpertIds);
    cudaFree(dTopkWeights);
    upBuf.free();
}

// ============================================================================
// Test: Down-proj only (Tier-2, small dims)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_DownProj_Tier2_Small)
{
    int const H = 256;
    int const N = 256;
    int const E = 2;
    int const topK = 1;
    int const numTokens = 1;
    double const cosineThreshold = 0.99;

    std::mt19937 rng(123);

    // Generate "intermediate" FP16 values (simulating up-proj output)
    int64_t const interElems = nemotronMoeW4A16InterBufferNumElems(numTokens, topK, N);
    std::vector<__half> interHost(interElems);
    {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& v : interHost)
        {
            v = __float2half(dist(rng));
        }
    }
    __half* dInter = nullptr;
    CUDA_CHECK(cudaMalloc(&dInter, interElems * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dInter, interHost.data(), interElems * sizeof(__half), cudaMemcpyHostToDevice));

    // Down weights: tile grid [E, inter_pos, hidden_chunk]
    int const hiddenChunks = H / kNvfp4ElemsPerTile;
    int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
    int64_t const dnS0 = static_cast<int64_t>(N) * (H / 32);
    int64_t const dnS1 = H / 32;
    int64_t const dnS2 = int4PerTile;

    auto dnBuf = allocateNvfp4Weight(E, N, H, rng);
    NVFP4Tensor dn = makeNvfp4Tensor(dnBuf, dnS0, dnS1, dnS2, N, H, E);

    // Expert IDs and topk weights
    std::vector<int32_t> expertIdsHost = {0};
    std::vector<float> topkWeightsHost = {1.0f};
    int32_t* dExpertIds = nullptr;
    float* dTopkWeights = nullptr;
    CUDA_CHECK(cudaMalloc(&dExpertIds, expertIdsHost.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dTopkWeights, topkWeightsHost.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(dExpertIds, expertIdsHost.data(), expertIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dTopkWeights, topkWeightsHost.data(), topkWeightsHost.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Decode SF for down: [E, N/16, H]
    uint8_t* dDnDecodeSf = generateDecodeSf(dnBuf);

    // Output buffer
    __half* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dOutput, numTokens * H * sizeof(__half)));

    // Launch down-proj kernel
    launchNemotronMoeW4A16DecodeDownGemvCuda(numTokens, 1, H, N, hiddenChunks, E, topK, dExpertIds, dTopkWeights,
        dInter, dn, dDnDecodeSf, dOutput, mStream, MoEActivationKind::kReLU2);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Get kernel output
    std::vector<__half> outHost(numTokens * H);
    CUDA_CHECK(cudaMemcpy(outHost.data(), dOutput, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> kernelOut(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        kernelOut[i] = __half2float(outHost[i]);
    }

    // CPU reference
    std::vector<float> downWFp32 = dequantWeightsToHostFp32(dn, E, N, H, false);
    std::vector<float> interFp32(interElems);
    for (int64_t i = 0; i < interElems; ++i)
    {
        interFp32[i] = __half2float(interHost[i]);
    }

    std::vector<float> refOut;
    cpuDownProjRef(interFp32, downWFp32, expertIdsHost, topkWeightsHost, numTokens, topK, H, N, E,
        MoEActivationKind::kReLU2, refOut);

    AccuracyMetrics m = computeMetrics(kernelOut, refOut);
    printMetrics("DownProj_Tier2_Small", m);

    EXPECT_EQ(m.nanCount, 0);
    EXPECT_EQ(m.infCount, 0);
    EXPECT_GT(m.cosine, cosineThreshold) << "Cosine similarity too low: " << m.cosine;

    cudaFree(dInter);
    cudaFree(dOutput);
    cudaFree(dDnDecodeSf);
    cudaFree(dExpertIds);
    cudaFree(dTopkWeights);
    dnBuf.free();
}

// ============================================================================
// Helper: run full E2E W4A16 MoE decode and return metrics
// ============================================================================

struct E2EResult
{
    AccuracyMetrics metrics;
    std::vector<float> kernelOut;
    std::vector<float> refOut;
};

static E2EResult runE2E(int H, int N, int E, int topK, int numTokens, MoEActivationKind actKind, unsigned seed,
    cudaStream_t stream, Nvfp4MoeDecodeTest* self)
{
    std::mt19937 rng(seed);

    // Activation: FP16 in [-1, 1]
    std::vector<__half> actHost(numTokens * H);
    {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& v : actHost)
        {
            v = __float2half(dist(rng));
        }
    }
    __half* dAct = nullptr;
    CUDA_CHECK(cudaMalloc(&dAct, actHost.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dAct, actHost.data(), actHost.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Up weights
    int const interChunks = N / kNvfp4ElemsPerTile;
    int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
    int64_t const upS0 = static_cast<int64_t>(H) * interChunks * int4PerTile;
    int64_t const upS1 = interChunks * int4PerTile;
    int64_t const upS2 = int4PerTile;

    auto upBuf = self->allocateNvfp4Weight(E, H, N, rng);
    NVFP4Tensor up = self->makeNvfp4Tensor(upBuf, upS0, upS1, upS2, H, N, E);

    // Down weights
    int64_t const dnS0 = static_cast<int64_t>(N) * (H / 32);
    int64_t const dnS1 = H / 32;
    int64_t const dnS2 = int4PerTile;

    auto dnBuf = self->allocateNvfp4Weight(E, N, H, rng);
    NVFP4Tensor dn = self->makeNvfp4Tensor(dnBuf, dnS0, dnS1, dnS2, N, H, E);

    // Expert IDs: round-robin assignment
    std::vector<int32_t> expertIdsHost(numTokens * topK);
    for (int t = 0; t < numTokens; ++t)
    {
        for (int k = 0; k < topK; ++k)
        {
            expertIdsHost[t * topK + k] = (t * topK + k) % E;
        }
    }

    // Topk weights: uniform in [0.1, 0.5], normalized per token
    std::vector<float> topkWeightsHost(numTokens * topK);
    {
        std::uniform_real_distribution<float> dist(0.1f, 0.5f);
        for (int t = 0; t < numTokens; ++t)
        {
            float sum = 0.f;
            for (int k = 0; k < topK; ++k)
            {
                topkWeightsHost[t * topK + k] = dist(rng);
                sum += topkWeightsHost[t * topK + k];
            }
            for (int k = 0; k < topK; ++k)
            {
                topkWeightsHost[t * topK + k] /= sum;
            }
        }
    }

    int32_t* dExpertIds = nullptr;
    float* dTopkWeights = nullptr;
    CUDA_CHECK(cudaMalloc(&dExpertIds, expertIdsHost.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dTopkWeights, topkWeightsHost.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(dExpertIds, expertIdsHost.data(), expertIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dTopkWeights, topkWeightsHost.data(), topkWeightsHost.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Decode SF
    uint8_t* dUpDecodeSf = self->generateDecodeSf(upBuf);
    uint8_t* dDnDecodeSf = self->generateDecodeSf(dnBuf);

    // Scratch and output
    int64_t const interElems = nemotronMoeW4A16InterBufferNumElems(numTokens, topK, N);
    __half* dInter = nullptr;
    __half* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInter, interElems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOutput, numTokens * H * sizeof(__half)));

    // Launch E2E
    launchNemotronMoeW4A16DecodeGemvCuda(numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dTopkWeights, dAct, up,
        dn, dUpDecodeSf, dDnDecodeSf, dInter, dOutput, stream, actKind);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Get kernel output
    std::vector<__half> outHost(numTokens * H);
    CUDA_CHECK(cudaMemcpy(outHost.data(), dOutput, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> kernelOut(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        kernelOut[i] = __half2float(outHost[i]);
    }

    // CPU reference using GPU-dequanted weights
    std::vector<float> upWFp32 = self->dequantWeightsToHostFp32(up, E, H, N, true);
    std::vector<float> dnWFp32 = self->dequantWeightsToHostFp32(dn, E, N, H, false);

    std::vector<float> actFp32(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        actFp32[i] = __half2float(actHost[i]);
    }

    std::vector<float> refOut;
    cpuE2ERef(actFp32, upWFp32, dnWFp32, expertIdsHost, topkWeightsHost, numTokens, topK, H, N, E, actKind, refOut);

    E2EResult result;
    result.metrics = computeMetrics(kernelOut, refOut);
    result.kernelOut = std::move(kernelOut);
    result.refOut = std::move(refOut);

    // Cleanup
    cudaFree(dAct);
    cudaFree(dInter);
    cudaFree(dOutput);
    cudaFree(dUpDecodeSf);
    cudaFree(dDnDecodeSf);
    cudaFree(dExpertIds);
    cudaFree(dTopkWeights);
    upBuf.free();
    dnBuf.free();

    return result;
}

// ============================================================================
// Test: Full E2E (Tier-2, small dims, ReLU2)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_Tier2_Small)
{
    auto result = runE2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, 42, mStream, this);
    printMetrics("E2E_Tier2_Small", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0);
    EXPECT_EQ(result.metrics.infCount, 0);
    EXPECT_GT(result.metrics.cosine, 0.99) << "Cosine similarity too low: " << result.metrics.cosine;
}

// ============================================================================
// Test: NaN/Inf check
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_NaNInf_Small)
{
    auto result = runE2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, 77, mStream, this);
    printMetrics("E2E_NaNInf_Small", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0) << "Found " << result.metrics.nanCount << " NaN values in output";
    EXPECT_EQ(result.metrics.infCount, 0) << "Found " << result.metrics.infCount << " Inf values in output";
}

// ============================================================================
// Test: Determinism (5 relaunches, byte-exact match)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_Determinism)
{
    int const numRelaunches = 5;
    std::vector<std::vector<float>> outputs(numRelaunches);

    for (int i = 0; i < numRelaunches; ++i)
    {
        auto result = runE2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, 42, mStream, this);
        outputs[i] = std::move(result.kernelOut);
    }

    for (int i = 1; i < numRelaunches; ++i)
    {
        ASSERT_EQ(outputs[0].size(), outputs[i].size());
        bool exact = std::memcmp(outputs[0].data(), outputs[i].data(), outputs[0].size() * sizeof(float)) == 0;
        EXPECT_TRUE(exact) << "Relaunch " << i << " differs from relaunch 0";
    }
}

// ============================================================================
// Test: Production dims (Nemotron-like: H=2688, N=1856, E=16, topk=6)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_Tier2_Nemotron)
{
    auto result = runE2E(2688, 1856, 16, 6, 1, MoEActivationKind::kReLU2, 42, mStream, this);
    printMetrics("E2E_Tier2_Nemotron", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0);
    EXPECT_EQ(result.metrics.infCount, 0);
    EXPECT_GT(result.metrics.cosine, 0.98) << "Cosine similarity too low: " << result.metrics.cosine;
}

// ============================================================================
// Test: SiLU activation path
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_SiLU)
{
    auto result = runE2E(256, 256, 4, 2, 1, MoEActivationKind::kSiLU, 42, mStream, this);
    printMetrics("E2E_SiLU", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0);
    EXPECT_EQ(result.metrics.infCount, 0);
    EXPECT_GT(result.metrics.cosine, 0.99) << "Cosine similarity too low: " << result.metrics.cosine;
}

// ============================================================================
// Test: Multi-seed stability (MoE.md: std ≈ 0 across seeds)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_MultiSeedStability)
{
    constexpr int kNumSeeds = 5;
    unsigned const seeds[kNumSeeds] = {42, 123, 7777, 31415, 99999};
    double cosines[kNumSeeds];
    double magRatios[kNumSeeds];

    for (int s = 0; s < kNumSeeds; ++s)
    {
        auto result = runE2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, seeds[s], mStream, this);
        cosines[s] = result.metrics.cosine;
        magRatios[s] = result.metrics.magRatio;
        EXPECT_EQ(result.metrics.nanCount, 0) << "Seed " << seeds[s] << " produced NaN";
        EXPECT_EQ(result.metrics.infCount, 0) << "Seed " << seeds[s] << " produced Inf";
        EXPECT_GT(cosines[s], 0.99) << "Seed " << seeds[s] << " cosine too low: " << cosines[s];
        EXPECT_NEAR(magRatios[s], 1.0, 0.05) << "Seed " << seeds[s] << " mag ratio off: " << magRatios[s];
    }

    // Compute mean and stddev of cosine across seeds
    double mean = 0.0;
    for (int s = 0; s < kNumSeeds; ++s)
    {
        mean += cosines[s];
    }
    mean /= kNumSeeds;

    double variance = 0.0;
    for (int s = 0; s < kNumSeeds; ++s)
    {
        double const d = cosines[s] - mean;
        variance += d * d;
    }
    double const stddev = std::sqrt(variance / kNumSeeds);

    printf("  [MultiSeed] cosines:");
    for (int s = 0; s < kNumSeeds; ++s)
    {
        printf(" %.6f", cosines[s]);
    }
    printf("  mean=%.6f  std=%.6e\n", mean, stddev);

    EXPECT_LT(stddev, 0.01) << "Multi-seed cosine stddev too high: " << stddev;
}

// ============================================================================
// Test: Router scale linearity (MoE.md: 2x weight → 2x output)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A16_E2E_RouterScaleLinearity)
{
    int const H = 256;
    int const N = 256;
    int const E = 2;
    int const topK = 1;
    int const numTokens = 1;

    std::mt19937 rng(42);

    // Activation: FP16 in [-1, 1]
    std::vector<__half> actHost(numTokens * H);
    {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& v : actHost)
        {
            v = __float2half(dist(rng));
        }
    }
    __half* dAct = nullptr;
    CUDA_CHECK(cudaMalloc(&dAct, actHost.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dAct, actHost.data(), actHost.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Up weights
    int const interChunks = N / kNvfp4ElemsPerTile;
    int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
    auto upBuf = allocateNvfp4Weight(E, H, N, rng);
    NVFP4Tensor up = makeNvfp4Tensor(
        upBuf, static_cast<int64_t>(H) * interChunks * int4PerTile, interChunks * int4PerTile, int4PerTile, H, N, E);

    // Down weights
    auto dnBuf = allocateNvfp4Weight(E, N, H, rng);
    NVFP4Tensor dn = makeNvfp4Tensor(dnBuf, static_cast<int64_t>(N) * (H / 32), H / 32, int4PerTile, N, H, E);

    // Decode SF
    uint8_t* dUpDecodeSf = generateDecodeSf(upBuf);
    uint8_t* dDnDecodeSf = generateDecodeSf(dnBuf);

    // Expert IDs: token 0 → expert 0
    std::vector<int32_t> expertIdsHost = {0};
    int32_t* dExpertIds = nullptr;
    CUDA_CHECK(cudaMalloc(&dExpertIds, sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(dExpertIds, expertIdsHost.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

    // Scratch and output buffers
    int64_t const interElems = nemotronMoeW4A16InterBufferNumElems(numTokens, topK, N);
    __half* dInter = nullptr;
    __half* dOutput1 = nullptr;
    __half* dOutput2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dInter, interElems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOutput1, numTokens * H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOutput2, numTokens * H * sizeof(__half)));

    // Run with router weight = 1.0
    float const w1 = 1.0f;
    float* dW1 = nullptr;
    CUDA_CHECK(cudaMalloc(&dW1, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW1, &w1, sizeof(float), cudaMemcpyHostToDevice));
    launchNemotronMoeW4A16DecodeGemvCuda(numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dW1, dAct, up, dn,
        dUpDecodeSf, dDnDecodeSf, dInter, dOutput1, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Run with router weight = 2.0
    float const w2 = 2.0f;
    float* dW2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dW2, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW2, &w2, sizeof(float), cudaMemcpyHostToDevice));
    launchNemotronMoeW4A16DecodeGemvCuda(numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dW2, dAct, up, dn,
        dUpDecodeSf, dDnDecodeSf, dInter, dOutput2, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Read back both outputs
    std::vector<__half> out1Host(numTokens * H);
    std::vector<__half> out2Host(numTokens * H);
    CUDA_CHECK(cudaMemcpy(out1Host.data(), dOutput1, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out2Host.data(), dOutput2, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));

    // Check: out2 ≈ 2 * out1 using cosine similarity and magnitude ratio
    std::vector<float> actual(numTokens * H);
    std::vector<float> expected(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        actual[i] = __half2float(out2Host[i]);
        expected[i] = 2.0f * __half2float(out1Host[i]);
    }

    AccuracyMetrics m = computeMetrics(actual, expected);
    printMetrics("RouterScale_2x", m);

    EXPECT_EQ(m.nanCount, 0);
    EXPECT_EQ(m.infCount, 0);
    EXPECT_GT(m.cosine, 0.999) << "Router scale linearity: cosine too low: " << m.cosine;
    EXPECT_NEAR(m.magRatio, 1.0, 0.01) << "Router scale linearity: magnitude ratio off: " << m.magRatio;

    // Cleanup
    cudaFree(dAct);
    cudaFree(dInter);
    cudaFree(dOutput1);
    cudaFree(dOutput2);
    cudaFree(dUpDecodeSf);
    cudaFree(dDnDecodeSf);
    cudaFree(dExpertIds);
    cudaFree(dW1);
    cudaFree(dW2);
    upBuf.free();
    dnBuf.free();
}

// ============================================================================
// Helper: run full E2E W4A4 MoE decode and return metrics
// ============================================================================

static E2EResult runW4A4E2E(int H, int N, int E, int topK, int numTokens, MoEActivationKind actKind, unsigned seed,
    cudaStream_t stream, Nvfp4MoeDecodeTest* self)
{
    std::mt19937 rng(seed);

    // Activation: NVFP4 tensor [numTokens, hiddenTiles, 0]
    auto actBuf = self->allocateNvfp4Activation(numTokens, H, rng);
    NVFP4Tensor actNvfp4 = self->makeNvfp4ActivationTensor(actBuf, H);

    // Up weights: tile grid [E, hidden_pos, inter_chunk]
    int const interChunks = N / kNvfp4ElemsPerTile;
    int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
    int64_t const upS0 = static_cast<int64_t>(H) * interChunks * int4PerTile;
    int64_t const upS1 = interChunks * int4PerTile;
    int64_t const upS2 = int4PerTile;

    auto upBuf = self->allocateNvfp4Weight(E, H, N, rng);
    NVFP4Tensor up = self->makeNvfp4Tensor(upBuf, upS0, upS1, upS2, H, N, E);

    // Down weights: tile grid [E, inter_pos, hidden_chunk]
    int64_t const dnS0 = static_cast<int64_t>(N) * (H / 32);
    int64_t const dnS1 = H / 32;
    int64_t const dnS2 = int4PerTile;

    auto dnBuf = self->allocateNvfp4Weight(E, N, H, rng);
    NVFP4Tensor dn = self->makeNvfp4Tensor(dnBuf, dnS0, dnS1, dnS2, N, H, E);

    // Expert IDs: round-robin assignment
    std::vector<int32_t> expertIdsHost(numTokens * topK);
    for (int t = 0; t < numTokens; ++t)
    {
        for (int k = 0; k < topK; ++k)
        {
            expertIdsHost[t * topK + k] = (t * topK + k) % E;
        }
    }

    // Topk weights: uniform in [0.1, 0.5], normalized per token
    std::vector<float> topkWeightsHost(numTokens * topK);
    {
        std::uniform_real_distribution<float> dist(0.1f, 0.5f);
        for (int t = 0; t < numTokens; ++t)
        {
            float sum = 0.f;
            for (int k = 0; k < topK; ++k)
            {
                topkWeightsHost[t * topK + k] = dist(rng);
                sum += topkWeightsHost[t * topK + k];
            }
            for (int k = 0; k < topK; ++k)
            {
                topkWeightsHost[t * topK + k] /= sum;
            }
        }
    }

    int32_t* dExpertIds = nullptr;
    float* dTopkWeights = nullptr;
    CUDA_CHECK(cudaMalloc(&dExpertIds, expertIdsHost.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dTopkWeights, topkWeightsHost.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(dExpertIds, expertIdsHost.data(), expertIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        dTopkWeights, topkWeightsHost.data(), topkWeightsHost.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Decode SF
    uint8_t* dUpDecodeSf = self->generateDecodeSf(upBuf);
    uint8_t* dDnDecodeSf = self->generateDecodeSf(dnBuf);

    // Scratch and output
    int64_t const interElems = nemotronMoeW4A16InterBufferNumElems(numTokens, topK, N);
    __half* dInter = nullptr;
    __half* dOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&dInter, interElems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOutput, numTokens * H * sizeof(__half)));

    // Thread block size for W4A4
    int const tbSize = nemotronMoeW4A4DecodeThreadBlockSizeForDims(H, N);

    // Launch E2E W4A4
    launchNemotronMoeW4a4DecodeGemvCuda(numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dTopkWeights, actNvfp4,
        up, dn, dUpDecodeSf, dDnDecodeSf, dInter, dOutput, stream, tbSize, actKind);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Get kernel output
    std::vector<__half> outHost(numTokens * H);
    CUDA_CHECK(cudaMemcpy(outHost.data(), dOutput, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> kernelOut(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        kernelOut[i] = __half2float(outHost[i]);
    }

    // CPU reference: dequant activation and weights → FP32
    std::vector<float> actFp32 = self->dequantActivationToHostFp32(actNvfp4, numTokens, H);
    std::vector<float> upWFp32 = self->dequantWeightsToHostFp32(up, E, H, N, true);
    std::vector<float> dnWFp32 = self->dequantWeightsToHostFp32(dn, E, N, H, false);

    std::vector<float> refOut;
    cpuE2ERef(actFp32, upWFp32, dnWFp32, expertIdsHost, topkWeightsHost, numTokens, topK, H, N, E, actKind, refOut);

    E2EResult result;
    result.metrics = computeMetrics(kernelOut, refOut);
    result.kernelOut = std::move(kernelOut);
    result.refOut = std::move(refOut);

    // Cleanup
    cudaFree(dInter);
    cudaFree(dOutput);
    cudaFree(dUpDecodeSf);
    cudaFree(dDnDecodeSf);
    cudaFree(dExpertIds);
    cudaFree(dTopkWeights);
    actBuf.free();
    upBuf.free();
    dnBuf.free();

    return result;
}

// ============================================================================
// Test: W4A4 Full E2E (Tier-2, small dims, ReLU2)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_Tier2_Small)
{
    auto result = runW4A4E2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, 42, mStream, this);
    printMetrics("W4A4_E2E_Tier2_Small", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0);
    EXPECT_EQ(result.metrics.infCount, 0);
    EXPECT_GT(result.metrics.cosine, 0.99) << "Cosine similarity too low: " << result.metrics.cosine;
}

// ============================================================================
// Test: W4A4 NaN/Inf check
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_NaNInf_Small)
{
    auto result = runW4A4E2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, 77, mStream, this);
    printMetrics("W4A4_E2E_NaNInf_Small", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0) << "Found " << result.metrics.nanCount << " NaN values in output";
    EXPECT_EQ(result.metrics.infCount, 0) << "Found " << result.metrics.infCount << " Inf values in output";
}

// ============================================================================
// Test: W4A4 Determinism (5 relaunches, byte-exact match)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_Determinism)
{
    int const numRelaunches = 5;
    std::vector<std::vector<float>> outputs(numRelaunches);

    for (int i = 0; i < numRelaunches; ++i)
    {
        auto result = runW4A4E2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, 42, mStream, this);
        outputs[i] = std::move(result.kernelOut);
    }

    for (int i = 1; i < numRelaunches; ++i)
    {
        ASSERT_EQ(outputs[0].size(), outputs[i].size());
        bool exact = std::memcmp(outputs[0].data(), outputs[i].data(), outputs[0].size() * sizeof(float)) == 0;
        EXPECT_TRUE(exact) << "Relaunch " << i << " differs from relaunch 0";
    }
}

// ============================================================================
// Test: W4A4 Production dims (Nemotron-like: H=2688, N=1856, E=16, topk=6)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_Tier2_Nemotron)
{
    auto result = runW4A4E2E(2688, 1856, 16, 6, 1, MoEActivationKind::kReLU2, 42, mStream, this);
    printMetrics("W4A4_E2E_Tier2_Nemotron", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0);
    EXPECT_EQ(result.metrics.infCount, 0);
    EXPECT_GT(result.metrics.cosine, 0.98) << "Cosine similarity too low: " << result.metrics.cosine;
}

// ============================================================================
// Test: W4A4 SiLU activation path
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_SiLU)
{
    auto result = runW4A4E2E(256, 256, 4, 2, 1, MoEActivationKind::kSiLU, 42, mStream, this);
    printMetrics("W4A4_E2E_SiLU", result.metrics);

    EXPECT_EQ(result.metrics.nanCount, 0);
    EXPECT_EQ(result.metrics.infCount, 0);
    EXPECT_GT(result.metrics.cosine, 0.99) << "Cosine similarity too low: " << result.metrics.cosine;
}

// ============================================================================
// Test: W4A4 Multi-seed stability (std ≈ 0 across seeds)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_MultiSeedStability)
{
    constexpr int kNumSeeds = 5;
    unsigned const seeds[kNumSeeds] = {42, 123, 7777, 31415, 99999};
    double cosines[kNumSeeds];
    double magRatios[kNumSeeds];

    for (int s = 0; s < kNumSeeds; ++s)
    {
        auto result = runW4A4E2E(256, 256, 4, 2, 1, MoEActivationKind::kReLU2, seeds[s], mStream, this);
        cosines[s] = result.metrics.cosine;
        magRatios[s] = result.metrics.magRatio;
        EXPECT_EQ(result.metrics.nanCount, 0) << "Seed " << seeds[s] << " produced NaN";
        EXPECT_EQ(result.metrics.infCount, 0) << "Seed " << seeds[s] << " produced Inf";
        EXPECT_GT(cosines[s], 0.99) << "Seed " << seeds[s] << " cosine too low: " << cosines[s];
        EXPECT_NEAR(magRatios[s], 1.0, 0.05) << "Seed " << seeds[s] << " mag ratio off: " << magRatios[s];
    }

    // Compute mean and stddev of cosine across seeds
    double mean = 0.0;
    for (int s = 0; s < kNumSeeds; ++s)
    {
        mean += cosines[s];
    }
    mean /= kNumSeeds;

    double variance = 0.0;
    for (int s = 0; s < kNumSeeds; ++s)
    {
        double const d = cosines[s] - mean;
        variance += d * d;
    }
    double const stddev = std::sqrt(variance / kNumSeeds);

    printf("  [W4A4_MultiSeed] cosines:");
    for (int s = 0; s < kNumSeeds; ++s)
    {
        printf(" %.6f", cosines[s]);
    }
    printf("  mean=%.6f  std=%.6e\n", mean, stddev);

    EXPECT_LT(stddev, 0.01) << "Multi-seed cosine stddev too high: " << stddev;
}

// ============================================================================
// Test: W4A4 Router scale linearity (2x weight → 2x output)
// ============================================================================

TEST_F(Nvfp4MoeDecodeTest, W4A4_E2E_RouterScaleLinearity)
{
    int const H = 256;
    int const N = 256;
    int const E = 2;
    int const topK = 1;
    int const numTokens = 1;

    std::mt19937 rng(42);

    // Activation: NVFP4 tensor
    auto actBuf = allocateNvfp4Activation(numTokens, H, rng);
    NVFP4Tensor actNvfp4 = makeNvfp4ActivationTensor(actBuf, H);

    // Up weights
    int const interChunks = N / kNvfp4ElemsPerTile;
    int64_t const int4PerTile = kNvfp4Int4PerTilePayload;
    auto upBuf = allocateNvfp4Weight(E, H, N, rng);
    NVFP4Tensor up = makeNvfp4Tensor(
        upBuf, static_cast<int64_t>(H) * interChunks * int4PerTile, interChunks * int4PerTile, int4PerTile, H, N, E);

    // Down weights
    auto dnBuf = allocateNvfp4Weight(E, N, H, rng);
    NVFP4Tensor dn = makeNvfp4Tensor(dnBuf, static_cast<int64_t>(N) * (H / 32), H / 32, int4PerTile, N, H, E);

    // Decode SF
    uint8_t* dUpDecodeSf = generateDecodeSf(upBuf);
    uint8_t* dDnDecodeSf = generateDecodeSf(dnBuf);

    // Expert IDs: token 0 → expert 0
    std::vector<int32_t> expertIdsHost = {0};
    int32_t* dExpertIds = nullptr;
    CUDA_CHECK(cudaMalloc(&dExpertIds, sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(dExpertIds, expertIdsHost.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

    // Scratch and output buffers
    int64_t const interElems = nemotronMoeW4A16InterBufferNumElems(numTokens, topK, N);
    __half* dInter = nullptr;
    __half* dOutput1 = nullptr;
    __half* dOutput2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dInter, interElems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOutput1, numTokens * H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOutput2, numTokens * H * sizeof(__half)));

    int const tbSize = nemotronMoeW4A4DecodeThreadBlockSizeForDims(H, N);

    // Run with router weight = 1.0
    float const w1 = 1.0f;
    float* dW1 = nullptr;
    CUDA_CHECK(cudaMalloc(&dW1, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW1, &w1, sizeof(float), cudaMemcpyHostToDevice));
    launchNemotronMoeW4a4DecodeGemvCuda(numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dW1, actNvfp4, up, dn,
        dUpDecodeSf, dDnDecodeSf, dInter, dOutput1, mStream, tbSize);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Run with router weight = 2.0
    float const w2 = 2.0f;
    float* dW2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dW2, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW2, &w2, sizeof(float), cudaMemcpyHostToDevice));
    launchNemotronMoeW4a4DecodeGemvCuda(numTokens, 1, H, N, interChunks, E, topK, dExpertIds, dW2, actNvfp4, up, dn,
        dUpDecodeSf, dDnDecodeSf, dInter, dOutput2, mStream, tbSize);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Read back both outputs
    std::vector<__half> out1Host(numTokens * H);
    std::vector<__half> out2Host(numTokens * H);
    CUDA_CHECK(cudaMemcpy(out1Host.data(), dOutput1, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out2Host.data(), dOutput2, numTokens * H * sizeof(__half), cudaMemcpyDeviceToHost));

    // Check: out2 ≈ 2 * out1 using cosine similarity and magnitude ratio
    std::vector<float> actual(numTokens * H);
    std::vector<float> expected(numTokens * H);
    for (int i = 0; i < numTokens * H; ++i)
    {
        actual[i] = __half2float(out2Host[i]);
        expected[i] = 2.0f * __half2float(out1Host[i]);
    }

    AccuracyMetrics m = computeMetrics(actual, expected);
    printMetrics("W4A4_RouterScale_2x", m);

    EXPECT_EQ(m.nanCount, 0);
    EXPECT_EQ(m.infCount, 0);
    EXPECT_GT(m.cosine, 0.999) << "Router scale linearity: cosine too low: " << m.cosine;
    EXPECT_NEAR(m.magRatio, 1.0, 0.01) << "Router scale linearity: magnitude ratio off: " << m.magRatio;

    // Cleanup
    cudaFree(dInter);
    cudaFree(dOutput1);
    cudaFree(dOutput2);
    cudaFree(dUpDecodeSf);
    cudaFree(dDnDecodeSf);
    cudaFree(dExpertIds);
    cudaFree(dW1);
    cudaFree(dW2);
    actBuf.free();
    upBuf.free();
    dnBuf.free();
}

#endif // SUPPORTS_FP4
