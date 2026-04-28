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

//! MoE W4AN decode GEMV kernels: NVFP4 tile dequant / GEMV helpers and launch templates.
//! Includes \ref nvfp4_tensor.cuh (tensor view) plus device helpers (\c dequantNvfp4TileToHalf, \c nvfp4GemvTileDot,
//! \c accumulateNvfp4GemvTileWarpReduce, etc.). Include after \ref kernels.h (needs \ref MoEActivationKind and
//! \ref kMaxDecodingKernelWarpCount).
#include "kernels/moe/moe_marlin/marlin/nvfp4_tensor.cuh"
#include "kernels/moe/moe_marlin/marlin/scalar_type.hpp"
#include "kernels/moe/nvf4_w4an/nvfp4_dequant.cuh"

#if SUPPORTS_FP4

namespace trt_edgellm
{

//! Convert one FP8 E4M3 byte to float using the same bit trick as Marlin \c dequant_fp8_scales.
//! Result is the raw FP8 value divided by 256; the caller's \c global_scale compensates.
__device__ __forceinline__ float decodeSfByteToFloat(uint8_t const b)
{
    uint16_t const h = static_cast<uint16_t>(b) << 7u;
    return __half2float(*reinterpret_cast<__half const*>(&h));
}

//! Apply \p kAct nonlinearity (see \ref MoEActivationKind) to up-proj accumulator \p z before decode post-scaling.
template <MoEActivationKind kAct>
__device__ __forceinline__ float moeActivation(float z)
{
    if constexpr (kAct == MoEActivationKind::kSiLU)
    {
        float const zc = fminf(fmaxf(z, -50.f), 50.f);
        return zc / (1.f + expf(-zc));
    }
    else
    {
        static_assert(kAct == MoEActivationKind::kReLU2, "moeActivation: unsupported MoEActivationKind");
        float const t = fmaxf(z, 0.f);
        return t * t;
    }
}

//! Sum warp-partial \c float2 pairs over \p kWarpCount warps for one \c (chunk,\c lane,\c k) slot.
template <int kWarpCount>
__device__ __forceinline__ float2 block_reduce_float2(
    float2 const (*__restrict__ s_warp_lane_k)[2][4][4], int const chunk, int const lane, int const k)
{
    float block0 = 0.f;
    float block1 = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpCount; ++w)
    {
        float2 const p = s_warp_lane_k[w][chunk][lane][k];
        block0 += p.x;
        block1 += p.y;
    }
    return make_float2(block0, block1);
}

//! Sum warp-partial \c half2 pairs over \p kWarpCount warps for one \c (chunk,\c lane,\c k) slot.
template <int kWarpCount>
__device__ __forceinline__ half2 block_reduce_half2(
    half2 const (*__restrict__ s_warp_lane_k)[2][4][4], int const chunk, int const lane, int const k)
{
    half2 acc = __float2half2_rn(0.f);
#pragma unroll
    for (int w = 0; w < kWarpCount; ++w)
    {
        acc = __hadd2(acc, s_warp_lane_k[w][chunk][lane][k]);
    }
    return acc;
}

// ============================================================================
// Compile-time type-id constants for use as template arguments inside kernels.
// (.id() is constexpr host-only; these namespace-scope constants work on device.)
// ============================================================================

namespace nvfp4_tensor_detail
{
constexpr marlin_dtypes::ScalarTypeId kFe4m3fnTypeId = marlin_dtypes::kFE4M3fn.id();
constexpr marlin_dtypes::ScalarTypeId kFe2m1fTypeId = marlin_dtypes::kFE2M1f.id();
} // namespace nvfp4_tensor_detail

// ============================================================================
// Tile-level NVFP4 → FP16 dequantization
// ============================================================================

//! Dequantize one 64-element NVFP4 tile to 64 contiguous \c half values.
//! Applies per-block FP8 E4M3 scales but does \b NOT apply \ref NVFP4Tensor::global_scale.
//! Single-thread; all data fits in registers.
//!
//! Output order matches tile memory order: chunk 0 lanes 0-3 (elements 0-31), chunk 1 lanes 0-3 (elements 32-63).
__device__ __forceinline__ void dequantNvfp4TileToHalf(
    NVFP4Tensor const& tensor, Dim3 const tile, __half* __restrict__ out64)
{
    // Decode the 4 packed E4M3 block scales (one int32 → 4 half values).
    half2 scale2[2];
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(tensor.readBlockScaleWord(tile), scale2);
    __half const scale_h[4] = {
        __low2half(scale2[0]),
        __high2half(scale2[0]),
        __low2half(scale2[1]),
        __high2half(scale2[1]),
    };

    int out_idx = 0;
#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 blk;
        tensor.loadTileUint4(tile, chunk, blk);
        uint32_t const lanes[4] = {blk.x, blk.y, blk.z, blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            int const base = (chunk * 4 + lane) * 8;
            int const sb = base / 16; // which of the 4 block scales
            __half const s = scale_h[sb];

            half2 frag[4]; // 4 × half2 = 8 FP16 values from 8 packed FP4 values
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(lanes[lane]), frag);

#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                out64[out_idx++] = __hmul(s, __low2half(frag[k]));
                out64[out_idx++] = __hmul(s, __high2half(frag[k]));
            }
        }
    }
}

//! Dequantize one NVFP4 activation scalar within a 64-element tile: one \c uint32 lane load plus block-scale decode.
//! Returns FP32 (NVFP4 path is computed in FP16 then widened). \p elem_idx_in_tile is \c 0..\c kNvfp4ElemsPerTile-1 in
//! the same order as \ref dequantNvfp4TileToHalf. Does \b not apply \ref NVFP4Tensor::global_scale.
//! Uses \ref readBlockScaleWordLinear — activation scale factors use plain linear layout.
__device__ __forceinline__ float dequantNvfp4TileElemToFloat(
    NVFP4Tensor const& tensor, Dim3 const tile, int const elem_idx_in_tile)
{
    int const chunk = elem_idx_in_tile >> 5;
    int const sub2 = elem_idx_in_tile & 31;
    int const lane = sub2 >> 3;
    int const pos = sub2 & 7;

    half2 scale2[2];
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(
        tensor.readBlockScaleWordLinear(tile), scale2);
    __half const scale_h[4] = {
        __low2half(scale2[0]),
        __high2half(scale2[0]),
        __low2half(scale2[1]),
        __high2half(scale2[1]),
    };

    int const base = (chunk * 4 + lane) * 8;
    int const sb = base / 16;
    __half const s = scale_h[sb];

    uint32_t const lane_word = tensor.loadTileUint32Lane(tile, chunk, lane);
    half2 frag[4];
    marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(lane_word), frag);
    int const pair = pos >> 1;
    __half const h = (pos & 1) != 0 ? __high2half(frag[pair]) : __low2half(frag[pair]);
    return __half2float(__hmul(s, h));
}

//! Same as \ref dequantNvfp4TileToHalf but also multiplies by
//! \c tensor.global_scale[\p global_scale_index].
__device__ __forceinline__ void dequantNvfp4TileToHalfScaled(
    NVFP4Tensor const& tensor, Dim3 const tile, int const global_scale_index, __half* __restrict__ out64)
{
    dequantNvfp4TileToHalf(tensor, tile, out64);

    __half const gs = __float2half(nvfp4TensorScaleAt(tensor.global_scale, global_scale_index));
#pragma unroll
    for (int i = 0; i < kNvfp4ElemsPerTile; ++i)
    {
        out64[i] = __hmul(out64[i], gs);
    }
}

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

//! Sum \p v across all 32 lanes of the warp in FP16.
__device__ __forceinline__ __half warp_reduce_half(__half v)
{
    v = __hadd(v, __shfl_down_sync(0xFFFFFFFFu, v, 16));
    v = __hadd(v, __shfl_down_sync(0xFFFFFFFFu, v, 8));
    v = __hadd(v, __shfl_down_sync(0xFFFFFFFFu, v, 4));
    v = __hadd(v, __shfl_down_sync(0xFFFFFFFFu, v, 2));
    v = __hadd(v, __shfl_down_sync(0xFFFFFFFFu, v, 1));
    return v;
}

//! Sum \p v across all 32 lanes of the warp in FP32.
__device__ __forceinline__ float warp_reduce_float(float v)
{
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    return v;
}

// ============================================================================
// Tile-level NVFP4 × NVFP4 dot product (GEMV building block)
// ============================================================================

//! Dot product of two 64-element NVFP4 tiles with full block-scale and global-scale dequantization.
//!
//! Computes: \c Σ_k  act_dequant[k] × weight_dequant[k]  over one 64-element tile pair.
//!
//! For GEMV \c y[n] = Σ_k x[k] × W[n,k]: the \b weight tile selects one row of \c W
//! (MMA A operand) and the \b act tile provides the activation vector chunk (MMA B operand).
//! The operand swap is transparent to the caller — just pass the right tiles.
//!
//! \param act      Activation tensor (A vector). \c global_scale[0] is used.
//! \param tile_a   Tile coordinates for the activation.
//! \param weight   Weight tensor (B matrix). \c global_scale[\p expert_index] is used.
//! \param tile_w   Tile coordinates for the weight.
//! \param expert_index  Expert id for weight global scale lookup.
//! \return FP32 partial dot product (caller accumulates across K-tiles).
__device__ __forceinline__ float nvfp4GemvTileDot(
    NVFP4Tensor const& act, Dim3 const tile_a, NVFP4Tensor const& weight, Dim3 const tile_w, int const expert_index)
{
    float const ga = nvfp4TensorScaleAt(act.global_scale, 0);
    float const gw = nvfp4TensorScaleAt(weight.global_scale, expert_index);

    __half acc = __float2half(0.f);

    half2 a_scale2[2];
    half2 w_scale2[2];
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(
        act.readBlockScaleWordLinear(tile_a), a_scale2);
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(weight.readBlockScaleWord(tile_w), w_scale2);

    __half const a_scale_h[4] = {
        __low2half(a_scale2[0]),
        __high2half(a_scale2[0]),
        __low2half(a_scale2[1]),
        __high2half(a_scale2[1]),
    };
    __half const w_scale_h[4] = {
        __low2half(w_scale2[0]),
        __high2half(w_scale2[0]),
        __low2half(w_scale2[1]),
        __high2half(w_scale2[1]),
    };

#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 a_blk, w_blk;
        act.loadTileUint4(tile_a, chunk, a_blk);
        weight.loadTileUint4(tile_w, chunk, w_blk);
        uint32_t const a32[4] = {a_blk.x, a_blk.y, a_blk.z, a_blk.w};
        uint32_t const w32[4] = {w_blk.x, w_blk.y, w_blk.z, w_blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            int const base = (chunk * 4 + lane) * 8;
            int const sb = base / 16;
            __half const scale_prod = __hmul(a_scale_h[sb], w_scale_h[sb]);

            half2 a_frag[4];
            half2 w_frag[4];
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(a32[lane]), a_frag);
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(w32[lane]), w_frag);

#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                acc = __hadd(acc, __hmul(__hmul(__low2half(a_frag[k]), __low2half(w_frag[k])), scale_prod));
                acc = __hadd(acc, __hmul(__hmul(__high2half(a_frag[k]), __high2half(w_frag[k])), scale_prod));
            }
        }
    }
    return __half2float(acc) * (ga * gw);
}

//! Same as \ref nvfp4GemvTileDot but omits the activation global scale (\c ga).
//! Callers that need \c activation.global_scale[0] fuse it into the scalar passed to \ref
//! accumulateNvfp4GemvTileWarpReduceToHalf (W4A4 up-proj) or multiply before the nonlinearity as appropriate.
__device__ __forceinline__ float nvfp4GemvTileDotNoActGlobal(
    NVFP4Tensor const& act, Dim3 const tile_a, NVFP4Tensor const& weight, Dim3 const tile_w, int const expert_index)
{
    float const gw = nvfp4TensorScaleAt(weight.global_scale, expert_index);

    __half acc = __float2half(0.f);

    half2 a_scale2[2];
    half2 w_scale2[2];
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(
        act.readBlockScaleWordLinear(tile_a), a_scale2);
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(weight.readBlockScaleWord(tile_w), w_scale2);

    __half const a_scale_h[4] = {
        __low2half(a_scale2[0]),
        __high2half(a_scale2[0]),
        __low2half(a_scale2[1]),
        __high2half(a_scale2[1]),
    };
    __half const w_scale_h[4] = {
        __low2half(w_scale2[0]),
        __high2half(w_scale2[0]),
        __low2half(w_scale2[1]),
        __high2half(w_scale2[1]),
    };

#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 a_blk, w_blk;
        act.loadTileUint4(tile_a, chunk, a_blk);
        weight.loadTileUint4(tile_w, chunk, w_blk);
        uint32_t const a32[4] = {a_blk.x, a_blk.y, a_blk.z, a_blk.w};
        uint32_t const w32[4] = {w_blk.x, w_blk.y, w_blk.z, w_blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            int const base = (chunk * 4 + lane) * 8;
            int const sb = base / 16;
            __half const scale_prod = __hmul(a_scale_h[sb], w_scale_h[sb]);

            half2 a_frag[4];
            half2 w_frag[4];
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(a32[lane]), a_frag);
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(w32[lane]), w_frag);

#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                acc = __hadd(acc, __hmul(__hmul(__low2half(a_frag[k]), __low2half(w_frag[k])), scale_prod));
                acc = __hadd(acc, __hmul(__hmul(__high2half(a_frag[k]), __high2half(w_frag[k])), scale_prod));
            }
        }
    }
    return __half2float(acc) * gw;
}

// ============================================================================
// Thread-block level GEMV: y[output_idx] = Σ_k act[k] × weight[output_idx, k]
// ============================================================================

//! Compute one output element of an NVFP4 × NVFP4 GEMV by looping over all K-tiles.
//!
//! Each thread independently computes one output value by accumulating \ref nvfp4GemvTileDot across K-tiles.
//! The caller launches enough threads to cover the output (N) dimension and maps each \c threadIdx to an
//! \p output_idx.
//!
//! \b Tensor layout convention (MoE up-proj example):
//!   - \c weight tiles: \c [expert, k_tile, output_idx] → \c tile = make_int3(expert, k_tile, output_idx)
//!   - \c act tiles:    \c [batch, k_tile, 0]           → \c tile = make_int3(batch, k_tile, 0)
//!
//! \param weight      Weight tensor tiled as \c [x=expert, y=k_tile, z=output_idx].
//! \param act         Activation tensor tiled as \c [x=batch, y=k_tile, z=0].
//! \param batch_idx   Batch index for activation (tile.x).
//! \param expert_idx  Expert index for weight (tile.x and global_scale lookup).
//! \param output_idx  Output element index (tile.z for weight).
//! \param K_tiles     Number of tiles in the K dimension (\c hidden_dim / 64).
//! \return FP32 accumulated GEMV output for \c y[\p output_idx].
__device__ __forceinline__ float nvfp4BlockGemv(NVFP4Tensor const& weight, NVFP4Tensor const& act, int const batch_idx,
    int const expert_idx, int const output_idx, int const K_tiles)
{
    float z = 0.f;
    for (int kt = 0; kt < K_tiles; ++kt)
    {
        Dim3 const tile_a = make_int3(batch_idx, kt, 0);
        Dim3 const tile_w = make_int3(expert_idx, kt, output_idx);
        z += nvfp4GemvTileDotNoActGlobal(act, tile_a, weight, tile_w, expert_idx);
    }
    return z;
}

// ============================================================================
// Down-proj style: warp-reduce dequanted tile × scalar → atomicAdd FP16 output
// ============================================================================

//! Add \p add into adjacent FP16 elements \p addr_even and \p addr_even+1 (\p addr_even must be 4-byte aligned).
//! On sm_70+ uses two \c atomicAdd(\c __half*) (CUDA native FP16 atomics). Older SMs use a 32-bit \c atomicCAS on the
//! packed \c half2 word.
__device__ __forceinline__ void atomicAddHalf2Aligned(__half* addr_even, half2 add)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(addr_even, __low2half(add));
    atomicAdd(addr_even + 1, __high2half(add));
#else
    auto* uiaddr = reinterpret_cast<unsigned int*>(addr_even);
    unsigned int old = *uiaddr;
    unsigned int assumed;
    do
    {
        assumed = old;
        union
        {
            unsigned int u;
            half2 h;
        } cur;
        cur.u = assumed;
        half2 const sum = __hadd2(cur.h, add);
        union
        {
            unsigned int u;
            half2 h;
        } outv;
        outv.h = sum;
        old = atomicCAS(uiaddr, assumed, outv.u);
    } while (assumed != old);
#endif
}

//! Dequantize one NVFP4 weight tile, multiply each element by a pre-computed scalar \p act_scale,
//! warp-reduce and block-reduce across warps in shared memory, then \ref atomicAddHalf2Aligned
//! into the FP16 output buffer.
//!
//! \p act_scale should already include all global scales and activation value (e.g.
//! \c moeActivation(z) × score × weight_global_scale[expert]).
//!
//! \p kThreadBlockSize must match the launch config and be a multiple of 32.
//! Requires \c __shared__ scratch of size <tt>kMaxWarpCount × 2 × 4 × 4 × sizeof(half2)</tt>.
//!
//! \param weight     Weight tensor.
//! \param tile       Tile coordinates in weight tensor.
//! \param act_scale  Pre-multiplied scalar for all 64 dequanted weight values.
//! \param output     FP16 output buffer (atomicAdd target, must be zero-initialized by caller).
//! \param out_base   Offset into \p output for element 0 of this tile.
//! \param kMaxWarpCount  Maximum warp count across all launch configs (for shared memory sizing).
//!
//! Warp shuffle uses \ref warp_reduce_half on dequantized weights; \c half2 shared scratch and \ref block_reduce_half2
//! across warps; \c atomicAddHalf2Aligned packs FP16. \c act_scale × block-scale is still applied in FP32 before
//! narrowing to \c half2 (same rationale as \ref accumulateNvfp4GemvTileWarpReduceToHalf): \p act_scale can exceed FP16
//! range while still finite as FP32, and early \c __float2half would overflow to Inf.
template <int kThreadBlockSize, int kMaxWarpCount = 8>
__device__ __forceinline__ void accumulateNvfp4GemvTileWarpReduce(
    NVFP4Tensor const& weight, Dim3 const tile, float const act_scale, __half* __restrict__ output, int const out_base)
{
    static_assert(kThreadBlockSize % 32 == 0, "kThreadBlockSize must be a multiple of warp size.");
    static constexpr int kWarpCount = kThreadBlockSize / 32;
    static_assert(kWarpCount <= kMaxWarpCount, "kWarpCount exceeds kMaxWarpCount.");

    // Shared memory for cross-warp reduction: [warp][chunk][lane][k] of half2 partials.
    static __shared__ half2 s_warp_chunk[kMaxWarpCount][kNvfp4Int4PerTilePayload][4][4];

    half2 scale2[2];
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(weight.readBlockScaleWord(tile), scale2);
    __half const scale_h[4] = {
        __low2half(scale2[0]),
        __high2half(scale2[0]),
        __low2half(scale2[1]),
        __high2half(scale2[1]),
    };
    __half const act_global_h = __float2half_rn(act_scale);

#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 blk;
        weight.loadTileUint4(tile, chunk, blk);
        uint32_t const w32[4] = {blk.x, blk.y, blk.z, blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            int const base = (chunk * 4 + lane) * 8;
            int const sb = base / 16;
            __half const lane_scale_h = act_global_h * scale_h[sb];

            half2 frag[4];
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(w32[lane]), frag);

#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                __half const wsum0 = warp_reduce_half(lane_scale_h * __low2half(frag[k]));
                __half const wsum1 = warp_reduce_half(lane_scale_h * __high2half(frag[k]));
                int const warp_id = static_cast<int>(threadIdx.x) / 32;
                if ((threadIdx.x & 31) == 0)
                {
                    s_warp_chunk[warp_id][chunk][lane][k] = make_half2(wsum0, wsum1);
                }
            }
        }
    }
    __syncthreads();

    // Final cross-warp reduction + atomic store. 32 output slots = 2 chunks × 4 lanes × 4 k.
    int const tid = static_cast<int>(threadIdx.x);
    if (tid < 32)
    {
        int const chunk = tid / 16;
        int const lane_k = tid - 16 * chunk;
        int const lane = lane_k / 4;
        int const k = lane_k - lane * 4;

        half2 const reduced = block_reduce_half2<kWarpCount>(s_warp_chunk, chunk, lane, k);
        int const i0 = (chunk * 4 + lane) * 8 + 2 * k;
        atomicAddHalf2Aligned(output + out_base + i0, reduced);
    }
    __syncthreads();
}

//! Like \ref accumulateNvfp4GemvTileWarpReduce but keeps \c half2 shared scratch and \ref atomicAddHalf2Aligned into
//! FP16 (up-proj path; intermediate \c z between up and down).
//! Warp shuffle uses \ref warp_reduce_half with \c act_scale × block-scale in FP32 before the multiply. Also applies
//! \c wt.global_scale[tile.x] into the scalar (down path folds scale into \p act_scale).
template <int kThreadBlockSize>
__device__ __forceinline__ void accumulateNvfp4GemvTileWarpReduceToHalf(NVFP4Tensor const& wt, Dim3 const tile,
    float const act_scale, __half* __restrict__ output, int const out_chunk_base)
{
    static_assert(kThreadBlockSize == 64 || kThreadBlockSize == 96 || kThreadBlockSize == 128 || kThreadBlockSize == 192
            || kThreadBlockSize == 256,
        "accumulateNvfp4GemvTileWarpReduceToHalf: kThreadBlockSize must be 64, 96, 128, 192, or 256");
    static_assert(kThreadBlockSize % 32 == 0,
        "accumulateNvfp4GemvTileWarpReduceToHalf: kThreadBlockSize must be a multiple of warp size.");
    static constexpr int kWarpCount = kThreadBlockSize / 32;

    static __shared__ half2 floatAccumWarpReduceChunk[kMaxDecodingKernelWarpCount][kNvfp4Int4PerTilePayload][4][4];

    half2 scale2[2];
    marlin::dequant_fp8_scales<half2, nvfp4_tensor_detail::kFe4m3fnTypeId>(wt.readBlockScaleWord(tile), scale2);
    __half const scale_h[4] = {
        __low2half(scale2[0]),
        __high2half(scale2[0]),
        __low2half(scale2[1]),
        __high2half(scale2[1]),
    };
    float const ds = nvfp4TensorScaleAt(wt.global_scale, tile.x);
    __half const act_global_h = __float2half_rn(act_scale * ds);

#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 a_blk;
        wt.loadTileUint4(tile, chunk, a_blk);
        uint32_t const w32[4] = {a_blk.x, a_blk.y, a_blk.z, a_blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            uint32_t const wpack = w32[lane];
            int const base = (chunk * 4 + lane) * 8;
            int const sb = base / 16;
            __half const lane_scale_h = act_global_h * scale_h[sb];
            half2 frag[4];
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(wpack), frag);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                __half const wsum0 = warp_reduce_half(lane_scale_h * __low2half(frag[k]));
                __half const wsum1 = warp_reduce_half(lane_scale_h * __high2half(frag[k]));
                int const warp_id = static_cast<int>(threadIdx.x) / 32;
                if ((threadIdx.x & 31) == 0)
                {
                    floatAccumWarpReduceChunk[warp_id][chunk][lane][k] = make_half2(wsum0, wsum1);
                }
            }
        }
    }
    __syncthreads();

    int const tid = static_cast<int>(threadIdx.x);
    static_assert(kNvfp4Int4PerTilePayload == 2, "kNvfp4Int4PerTilePayload must be 2");
    if (tid < 32)
    {
        int const chunk = tid / 16;
        int const lane_k = tid - 16 * chunk;
        int const lane = lane_k / 4;
        int const k = lane_k - lane * 4;
        int const i0 = (chunk * 4 + lane) * 8 + 2 * k;
        half2 const reduced = block_reduce_half2<kWarpCount>(floatAccumWarpReduceChunk, chunk, lane, k);
        atomicAddHalf2Aligned(output + out_chunk_base + i0, reduced);
    }
    __syncthreads();
}

//! Like \ref accumulateNvfp4GemvTileWarpReduce but reads per-element FP8 scales from \p sf_base
//! (decode-friendly row-major layout \c [E, K/16, N]) instead of the NVFP4Tensor's atom-swizzled
//! block_scale. \p sf_base points at 64 contiguous scale bytes for this tile — one per N-axis
//! element. Each thread's H-group determines the SF row; the 64 consecutive I (or H for down)
//! positions provide per-element scales.
template <int kThreadBlockSize, int kMaxWarpCount = 8>
__device__ __forceinline__ void accumulateNvfp4GemvTileWarpReduceDecodeSf(NVFP4Tensor const& weight, Dim3 const tile,
    float const act_scale, uint8_t const* __restrict__ sf_base, __half* __restrict__ output, int const out_base)
{
    static_assert(kThreadBlockSize % 32 == 0, "kThreadBlockSize must be a multiple of warp size.");
    static constexpr int kWarpCount = kThreadBlockSize / 32;
    static_assert(kWarpCount <= kMaxWarpCount, "kWarpCount exceeds kMaxWarpCount.");

    static __shared__ half2 s_warp_chunk[kMaxWarpCount][kNvfp4Int4PerTilePayload][4][4];

    __half const act_global_h = __float2half_rn(act_scale);

#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 blk;
        weight.loadTileUint4(tile, chunk, blk);
        uint32_t const w32[4] = {blk.x, blk.y, blk.z, blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            int const base = (chunk * 4 + lane) * 8;
            half2 frag[4];
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(w32[lane]), frag);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                __half const sf_lo = __float2half_rn(decodeSfByteToFloat(sf_base[base + 2 * k]));
                __half const sf_hi = __float2half_rn(decodeSfByteToFloat(sf_base[base + 2 * k + 1]));
                __half const wsum0 = warp_reduce_half(act_global_h * sf_lo * __low2half(frag[k]));
                __half const wsum1 = warp_reduce_half(act_global_h * sf_hi * __high2half(frag[k]));
                int const warp_id = static_cast<int>(threadIdx.x) / 32;
                if ((threadIdx.x & 31) == 0)
                {
                    s_warp_chunk[warp_id][chunk][lane][k] = make_half2(wsum0, wsum1);
                }
            }
        }
    }
    __syncthreads();

    int const tid = static_cast<int>(threadIdx.x);
    if (tid < 32)
    {
        int const chunk = tid / 16;
        int const lane_k = tid - 16 * chunk;
        int const lane = lane_k / 4;
        int const k = lane_k - lane * 4;

        half2 const reduced = block_reduce_half2<kWarpCount>(s_warp_chunk, chunk, lane, k);
        int const i0 = (chunk * 4 + lane) * 8 + 2 * k;
        atomicAddHalf2Aligned(output + out_base + i0, reduced);
    }
    __syncthreads();
}

//! Like \ref accumulateNvfp4GemvTileWarpReduceToHalf but reads per-element FP8 scales from
//! \p sf_base (decode-friendly row-major layout). Also applies \c wt.global_scale[tile.x].
//! \p sf_base points at 64 contiguous scale bytes — one per element in the tile's N-axis.
template <int kThreadBlockSize>
__device__ __forceinline__ void accumulateNvfp4GemvTileWarpReduceToHalfDecodeSf(NVFP4Tensor const& wt, Dim3 const tile,
    float const act_scale, uint8_t const* __restrict__ sf_base, __half* __restrict__ output, int const out_chunk_base)
{
    static_assert(kThreadBlockSize == 64 || kThreadBlockSize == 96 || kThreadBlockSize == 128 || kThreadBlockSize == 192
            || kThreadBlockSize == 256,
        "accumulateNvfp4GemvTileWarpReduceToHalfDecodeSf: kThreadBlockSize must be 64, 96, 128, 192, or 256");
    static_assert(kThreadBlockSize % 32 == 0,
        "accumulateNvfp4GemvTileWarpReduceToHalfDecodeSf: kThreadBlockSize must be a multiple of warp size.");
    static constexpr int kWarpCount = kThreadBlockSize / 32;

    static __shared__ half2 floatAccumWarpReduceChunk[kMaxDecodingKernelWarpCount][kNvfp4Int4PerTilePayload][4][4];

    float const ds = nvfp4TensorScaleAt(wt.global_scale, tile.x);
    __half const act_global_h = __float2half_rn(act_scale * ds);

#pragma unroll
    for (int chunk = 0; chunk < static_cast<int>(kNvfp4Int4PerTilePayload); ++chunk)
    {
        uint4 a_blk;
        wt.loadTileUint4(tile, chunk, a_blk);
        uint32_t const w32[4] = {a_blk.x, a_blk.y, a_blk.z, a_blk.w};

#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            uint32_t const wpack = w32[lane];
            int const base = (chunk * 4 + lane) * 8;
            half2 frag[4];
            marlin::dequant<half2, nvfp4_tensor_detail::kFe2m1fTypeId>(static_cast<int>(wpack), frag);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                __half const sf_lo = __float2half_rn(decodeSfByteToFloat(sf_base[base + 2 * k]));
                __half const sf_hi = __float2half_rn(decodeSfByteToFloat(sf_base[base + 2 * k + 1]));
                __half const wsum0 = warp_reduce_half(act_global_h * sf_lo * __low2half(frag[k]));
                __half const wsum1 = warp_reduce_half(act_global_h * sf_hi * __high2half(frag[k]));
                int const warp_id = static_cast<int>(threadIdx.x) / 32;
                if ((threadIdx.x & 31) == 0)
                {
                    floatAccumWarpReduceChunk[warp_id][chunk][lane][k] = make_half2(wsum0, wsum1);
                }
            }
        }
    }
    __syncthreads();

    int const tid = static_cast<int>(threadIdx.x);
    static_assert(kNvfp4Int4PerTilePayload == 2, "kNvfp4Int4PerTilePayload must be 2");
    if (tid < 32)
    {
        int const chunk = tid / 16;
        int const lane_k = tid - 16 * chunk;
        int const lane = lane_k / 4;
        int const k = lane_k - lane * 4;
        int const i0 = (chunk * 4 + lane) * 8 + 2 * k;
        half2 const reduced = block_reduce_half2<kWarpCount>(floatAccumWarpReduceChunk, chunk, lane, k);
        atomicAddHalf2Aligned(output + out_chunk_base + i0, reduced);
    }
    __syncthreads();
}

//! W4A4 decode — up-proj only. Grid = \c batch × seq_len × top_k × (hidden_dim / kThreadBlockSize) (hidden strips).
//! FP16 scratch row-major \c [batch * seq_len, top_k, inter_dim] (same as \ref moeW4A16DecodeUpGemvKernel).
//! Each thread dequants its NVFP4 activation scalar at hidden position \c j, multiplies by \c
//! activation.global_scale[0], then loops over \c inter_chunks calling \ref accumulateNvfp4GemvTileWarpReduceToHalf
//! (warp reduce + atomicAddHalf2Aligned).
template <int kThreadBlockSize>
__global__ void moeW4A4DecodeUpGemvKernel(int const batch, int const seq_len, int const hidden_dim, int const inter_dim,
    int const inter_chunks, int const num_experts, int const top_k, int32_t const* __restrict__ expert_ids,
    NVFP4Tensor activation, NVFP4Tensor up, uint8_t const* __restrict__ up_decode_sf,
    __half* __restrict__ inter_fp16_accum)
{
    static_assert(kThreadBlockSize == 64 || kThreadBlockSize == 96 || kThreadBlockSize == 128 || kThreadBlockSize == 192
            || kThreadBlockSize == 256,
        "moeW4A4DecodeUpGemvKernel: kThreadBlockSize must be 64, 96, 128, 192, or 256");
    static_assert(
        kThreadBlockSize % 32 == 0, "moeW4A4DecodeUpGemvKernel: kThreadBlockSize must be a multiple of warp size.");

    int const num_tokens = batch * seq_len;
    int const num_strips = hidden_dim / kThreadBlockSize;

    int const bid = blockIdx.x;
    int const blocks_per_token = top_k * num_strips;
    int const token_idx = bid / blocks_per_token;
    if (token_idx >= num_tokens)
    {
        return;
    }

    int const rem = bid % blocks_per_token;
    int const k_slot = rem / num_strips;
    int const strip_id = rem % num_strips;

    size_t const tk_idx = static_cast<size_t>(token_idx) * static_cast<size_t>(top_k) + static_cast<size_t>(k_slot);
    int const e = expert_ids[tk_idx];
    if (e < 0 || e >= num_experts)
    {
        return;
    }

    int const jb = strip_id * kThreadBlockSize;
    int const j = jb + static_cast<int>(threadIdx.x); // hidden position

    // Dequant this thread's NVFP4 activation scalar: one uint32 lane + block-scale (no global_scale).
    int const act_tile_idx = j / kNvfp4ElemsPerTile;
    int const act_elem_idx = j % kNvfp4ElemsPerTile;
    float const act_val = dequantNvfp4TileElemToFloat(activation, make_int3(token_idx, act_tile_idx, 0), act_elem_idx)
        * nvfp4TensorScaleAt(activation.global_scale, 0);

    // Decode SF base for this thread: up_decode_sf[E, H/16, I], row-major.
    int const h_group = j / 16;
    int const sf_expert_offset = e * (hidden_dim / 16) * inter_dim;
    int const sf_h_offset = h_group * inter_dim;

    for (int c = 0; c < inter_chunks; ++c)
    {
        Dim3 const upTile = make_int3(e, j, c);
        int64_t const inter_row = (static_cast<int64_t>(token_idx) * top_k + static_cast<int64_t>(k_slot))
            * static_cast<int64_t>(inter_dim);
        int const out_base = static_cast<int>(inter_row) + c * 64;
        uint8_t const* sf_base = up_decode_sf + sf_expert_offset + sf_h_offset + c * 64;
        accumulateNvfp4GemvTileWarpReduceToHalfDecodeSf<kThreadBlockSize>(
            up, upTile, act_val, sf_base, inter_fp16_accum, out_base);
    }
}

//! W4A4 decode — down-proj only. Grid = \c batch × seq_len × top_k × (inter_dim / kThreadBlockSize) (inter strips; same
//! as
//! \ref moeW4A16DecodeDownGemvKernel). Reads FP16 pre-nonlinearity \p inter_in from the up pass (already includes
//! \c activation.global_scale[0] in the dot); converts to FP32 for \ref moeActivation, then router \c score, and \c
//! down.global_scale[e], then inner loop over \c hidden_dim / 64. Accumulates via \ref
//! accumulateNvfp4GemvTileWarpReduce into FP16 \p output row-major \c [batch * seq_len, hidden_dim] (caller zeros; pass
//! \c down with \c global_scale cleared in the helper path so expert scale is not applied twice — it is folded into the
//! scalar \c t).
template <int kThreadBlockSize, MoEActivationKind kAct>
__global__ void moeW4A4DecodeDownGemvKernel(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const hidden_chunks, int const num_experts, int const top_k,
    int32_t const* __restrict__ expert_ids, float const* __restrict__ topk_weights, __half const* __restrict__ inter_in,
    NVFP4Tensor down, uint8_t const* __restrict__ down_decode_sf, __half* __restrict__ output)
{
    static_assert(kThreadBlockSize == 64 || kThreadBlockSize == 96 || kThreadBlockSize == 128 || kThreadBlockSize == 192
            || kThreadBlockSize == 256,
        "moeW4A4DecodeDownGemvKernel: kThreadBlockSize must be 64, 96, 128, 192, or 256");
    static_assert(kThreadBlockSize % 32 == 0,
        "moeW4A4DecodeDownGemvKernel: kThreadBlockSize must be a multiple of warp size for shuffle reduce.");

    int const num_tokens = batch * seq_len;
    int const num_strips = inter_dim / kThreadBlockSize;

    int const bid = blockIdx.x;
    int const blocks_per_token = top_k * num_strips;
    int const token_idx = bid / blocks_per_token;
    if (token_idx >= num_tokens)
    {
        return;
    }

    int const rem = bid % blocks_per_token;
    int const k_slot = rem / num_strips;
    int const strip_id = rem % num_strips;

    size_t const tk_idx = static_cast<size_t>(token_idx) * static_cast<size_t>(top_k) + static_cast<size_t>(k_slot);
    int const e = expert_ids[tk_idx];
    float const score = topk_weights[tk_idx];
    if (e < 0 || e >= num_experts)
    {
        return;
    }

    int const jb = strip_id * kThreadBlockSize;
    int const j = jb + static_cast<int>(threadIdx.x);

    size_t const in_idx = (static_cast<size_t>(token_idx) * static_cast<size_t>(top_k) + static_cast<size_t>(k_slot))
            * static_cast<size_t>(inter_dim)
        + static_cast<size_t>(j);

    float const z = __half2float(inter_in[in_idx]);
    float const down_gs = nvfp4TensorScaleAt(down.global_scale, e);
    float const t = moeActivation<kAct>(z) * score * down_gs;

    // Decode SF base: down_decode_sf[E, I/16, H], row-major.
    int const i_group = j / 16;
    int const sf_expert_offset = e * (inter_dim / 16) * hidden_dim;
    int const sf_i_offset = i_group * hidden_dim;

    for (int c = 0; c < hidden_chunks; ++c)
    {
        Dim3 const d_tile = make_int3(e, j, c);
        int const out_base = token_idx * hidden_dim + c * 64;
        uint8_t const* sf_base = down_decode_sf + sf_expert_offset + sf_i_offset + c * 64;
        accumulateNvfp4GemvTileWarpReduceDecodeSf<kThreadBlockSize, kMaxDecodingKernelWarpCount>(
            down, d_tile, t, sf_base, output, out_base);
    }
}

//! W4A16 decode — up-proj only. Grid = \c batch × seq_len × top_k × (hidden_dim / kThreadBlockSize) (hidden strips).
//! FP16 \p input row-major \c [batch * seq_len, hidden_dim]; partials \ref atomicAddHalf2Aligned into \p
//! inter_fp16_accum row-major \c [batch * seq_len, top_k, inter_dim] (caller zeros before launch).
template <int kThreadBlockSize>
__global__ void moeW4A16DecodeUpGemvKernel(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const inter_chunks, int const num_experts, int const top_k,
    int32_t const* __restrict__ expert_ids, __half const* __restrict__ input, NVFP4Tensor up,
    uint8_t const* __restrict__ up_decode_sf, __half* __restrict__ inter_fp16_accum)
{
    static_assert(kThreadBlockSize == 64 || kThreadBlockSize == 96 || kThreadBlockSize == 128 || kThreadBlockSize == 192
            || kThreadBlockSize == 256,
        "moeW4A16DecodeUpGemvKernel: kThreadBlockSize must be 64, 96, 128, 192, or 256");
    static_assert(kThreadBlockSize % 32 == 0,
        "moeW4A16DecodeUpGemvKernel: kThreadBlockSize must be a multiple of warp size for shuffle reduce.");

    int const num_tokens = batch * seq_len;
    int const num_strips = hidden_dim / kThreadBlockSize;

    int const bid = blockIdx.x;
    int const blocks_per_token = top_k * num_strips;
    int const token_idx = bid / blocks_per_token;
    if (token_idx >= num_tokens)
    {
        return;
    }

    int const rem = bid % blocks_per_token;
    int const k_slot = rem / num_strips;
    int const strip_id = rem % num_strips;

    size_t const tk_idx = static_cast<size_t>(token_idx) * static_cast<size_t>(top_k) + static_cast<size_t>(k_slot);
    int const e = expert_ids[tk_idx];
    if (e < 0 || e >= num_experts)
    {
        return;
    }

    int const jb = strip_id * kThreadBlockSize;
    int const j = jb + static_cast<int>(threadIdx.x);

    size_t const in_idx = static_cast<size_t>(token_idx) * static_cast<size_t>(hidden_dim) + static_cast<size_t>(j);

    float const hidden = __half2float(input[in_idx]);

    // Decode SF base: up_decode_sf[E, H/16, I], row-major.
    int const h_group = j / 16;
    int const sf_expert_offset = e * (hidden_dim / 16) * inter_dim;
    int const sf_h_offset = h_group * inter_dim;

    for (int c = 0; c < inter_chunks; ++c)
    {
        Dim3 const dTile = make_int3(e, j, c);
        int64_t const inter_row = (static_cast<int64_t>(token_idx) * top_k + static_cast<int64_t>(k_slot))
            * static_cast<int64_t>(inter_dim);
        int const out_base = static_cast<int>(inter_row) + c * 64;
        uint8_t const* sf_base = up_decode_sf + sf_expert_offset + sf_h_offset + c * 64;
        accumulateNvfp4GemvTileWarpReduceToHalfDecodeSf<kThreadBlockSize>(
            up, dTile, hidden, sf_base, inter_fp16_accum, out_base);
    }
}

//! W4A16 decode — down-proj only. Grid = \c batch × seq_len × top_k × (inter_dim / kThreadBlockSize) (inter strips),
//! unlike up
//! partial. Reads FP16 \p inter_in row-major \c [batch * seq_len, top_k, inter_dim] (up-proj dot \c z before
//! nonlinearity); converts to FP32 for \ref moeActivation, router \c score, and \c down.global_scale[e], then NVFP4
//! down weights \c [E, inter, hidden/2] tiles. Accumulates via \ref accumulateNvfp4GemvTileWarpReduce into FP16 \p
//! output row-major \c [batch * seq_len, hidden_dim] (caller must zero before launch).
template <int kThreadBlockSize, MoEActivationKind kAct>
__global__ void moeW4A16DecodeDownGemvKernel(int const batch, int const seq_len, int const hidden_dim,
    int const inter_dim, int const hidden_chunks, int const num_experts, int const top_k,
    int32_t const* __restrict__ expert_ids, float const* __restrict__ topk_weights, __half const* __restrict__ inter_in,
    NVFP4Tensor down, uint8_t const* __restrict__ down_decode_sf, __half* __restrict__ output)
{
    static_assert(kThreadBlockSize == 64 || kThreadBlockSize == 96 || kThreadBlockSize == 128 || kThreadBlockSize == 192
            || kThreadBlockSize == 256,
        "moeW4A16DecodeDownGemvKernel: kThreadBlockSize must be 64, 96, 128, 192, or 256");
    static_assert(kThreadBlockSize % 32 == 0,
        "moeW4A16DecodeDownGemvKernel: kThreadBlockSize must be a multiple of warp size for shuffle reduce.");

    int const num_tokens = batch * seq_len;
    int const num_strips = inter_dim / kThreadBlockSize;

    int const bid = blockIdx.x;
    int const blocks_per_token = top_k * num_strips;
    int const token_idx = bid / blocks_per_token;
    if (token_idx >= num_tokens)
    {
        return;
    }

    int const rem = bid % blocks_per_token;
    int const k_slot = rem / num_strips;
    int const strip_id = rem % num_strips;

    size_t const tk_idx = static_cast<size_t>(token_idx) * static_cast<size_t>(top_k) + static_cast<size_t>(k_slot);
    int const e = expert_ids[tk_idx];
    float const score = topk_weights[tk_idx];
    if (e < 0 || e >= num_experts)
    {
        return;
    }

    int const jb = strip_id * kThreadBlockSize;
    int const j = jb + static_cast<int>(threadIdx.x);

    size_t const in_idx = (static_cast<size_t>(token_idx) * static_cast<size_t>(top_k) + static_cast<size_t>(k_slot))
            * static_cast<size_t>(inter_dim)
        + static_cast<size_t>(j);

    float const inter = __half2float(inter_in[in_idx]);
    float const down_gs = nvfp4TensorScaleAt(down.global_scale, e);
    float const act = moeActivation<kAct>(inter) * score * down_gs;

    // Decode SF base: down_decode_sf[E, I/16, H], row-major.
    int const i_group = j / 16;
    int const sf_expert_offset = e * (inter_dim / 16) * hidden_dim;
    int const sf_i_offset = i_group * hidden_dim;

    for (int c = 0; c < hidden_chunks; ++c)
    {
        Dim3 const dTile = make_int3(e, j, c);
        int const out_base = token_idx * hidden_dim + c * 64;
        uint8_t const* sf_base = down_decode_sf + sf_expert_offset + sf_i_offset + c * 64;
        accumulateNvfp4GemvTileWarpReduceDecodeSf<kThreadBlockSize, kMaxDecodingKernelWarpCount>(
            down, dTile, act, sf_base, output, out_base);
    }
}

} // namespace trt_edgellm

#endif // SUPPORTS_FP4
