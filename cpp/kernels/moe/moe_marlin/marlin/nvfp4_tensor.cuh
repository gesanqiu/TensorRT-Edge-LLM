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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{

//! Packed NVFP4 (E2M1) payload for one 64-element tile: 8×uint32 lanes, 8 FP4 values per uint32.
constexpr int kNvfp4ElemsPerTile = 64;
//! Number of \c int4 vectors per tile payload (8×uint32 lanes == 2×\c int4); \c int64_t for mixed \c int4 / \c int64
//! stride math.
constexpr int64_t kNvfp4Int4PerTilePayload = 2;

//! Three tile indices for \ref NVFP4Tensor: \c x → \c strides[0] (slowest), \c y → \c strides[1], \c z → \c strides[2]
//! (fastest).
using Dim3 = int3;

//! Device view of an NVFP4 tensor stored in 64-element tiles with per-tile FP8 block scales and optional
//! \ref global_scale (applied after block dequantization).
//!
//! \ref global_scale is a device pointer to \c E floats from the MoE plugin (per-expert scales) when the tensor is
//! stacked expert weights (slow tile index \c x is the expert id); kernels use \c global_scale[tile.x].
//! \c nullptr means a factor of \c 1.f.
//! MoE W4A4 \b activation: \c global_scale[\c 0] scales the activation in the up-proj GEMV (after NVFP4 dequant).
//! Down-proj decode does not read the activation tensor. \c nullptr means \c 1.f for index \c 0.
//!
//! Logical layout is 3D in **tiles** (each tile is \ref kNvfp4ElemsPerTile NVFP4 values + one packed scale int).
//! Index order is row-major C: \p z is the fastest-varying tile index, \p x the slowest.
//!
//! \p quantized_data points at the first \c int4 of the tile at logical origin (0,0,0). Each tile occupies
//! \ref kNvfp4Int4PerTilePayload contiguous \c int4 vectors. \p strides[d] is the offset in \c int4 elements
//! when logical tile index component \c d increases by 1 (NumPy-style / sizeof(int4)).
//!
//! \p block_scale uses **atom-layout 128×4 swizzle** (matching Blackwell \c tcgen05.mma block-scaled MMA and TMA
//! scale-factor multicasts). Each FP8 E4M3 byte is stored at a byte offset computed by \ref tileScaleIndex; four
//! consecutive bytes (innerK 0..3) form one \c int32 word read by \ref readBlockScaleWord.
//!
//! Atom-layout byte offset for scale factor at logical (mRow, kChunk):
//!   innerK = kChunk % 4;  innerM = (mRow % 128) / 32;  outerM = mRow % 32;
//!   kTile = kChunk / 4;   mTile = mRow / 128;
//!   byteOffset = mTile * numKTiles * 512 + kTile * 512 + outerM * 16 + innerM * 4 + innerK
//!
//! The int32 word index (for \c block_scale pointer) drops innerK:
//!   mTile * numKTiles * 128 + kTile * 128 + outerM * 4 + innerM
//!
//! \p scaleMDimIdx / \p scaleKDimIdx select which \c Dim3 component maps to mRow / kChunk.
//! \p scaleNumKTiles is \c ceil(numSfCols / 4) where \c numSfCols = K / 16.
//! \p scaleExpertStride is the int32 element stride between experts (\c ceil(M/128) * numKTiles * 128).
struct NVFP4Tensor
{
    int4* quantized_data;
    int* block_scale;
    //! Device pointer to \c E per-expert scales (MoE plugin inputs), or \c nullptr. See struct comment.
    float* global_scale;
    //! Stride in \c int4 elements per +1 in tile index dim 0..2 (\c z fastest).
    int64_t strides[3];

    //! Atom-layout scale factor metadata.
    int32_t scaleNumKTiles;    //!< K-tiles for atom swizzle: ceil(numSfCols / 4) where numSfCols = K_dim / 16.
    int64_t scaleExpertStride; //!< int32 elements between experts: ceil(M/128) * scaleNumKTiles * 128.
    int8_t scaleMDimIdx;       //!< Which Dim3 component is the M-row (0 for act, 1 for weights).
    int8_t scaleKDimIdx;       //!< Which Dim3 component is the K-chunk (1 for act, 2 for weights).

    __device__ __forceinline__ int64_t tileOffsetInt4(Dim3 const c) const
    {
        return static_cast<int64_t>(c.x) * strides[0] + static_cast<int64_t>(c.y) * strides[1]
            + static_cast<int64_t>(c.z) * strides[2];
    }

    //! Atom-layout 128×4 swizzle: returns \c int32 index into \c block_scale for the tile at \p c.
    __device__ __forceinline__ int64_t tileScaleIndex(Dim3 const c) const
    {
        int32_t const dims[3] = {c.x, c.y, c.z};
        int32_t const mRow = dims[scaleMDimIdx];
        int32_t const kChunk = dims[scaleKDimIdx];
        // Expert dim is the remaining one (neither M nor K).
        int32_t expertIdx = 0;
        for (int i = 0; i < 3; ++i)
        {
            if (i != scaleMDimIdx && i != scaleKDimIdx)
            {
                expertIdx = dims[i];
                break;
            }
        }

        int32_t const mTile = mRow / 128;
        int32_t const innerM = (mRow % 128) / 32;
        int32_t const outerM = mRow % 32;
        return static_cast<int64_t>(expertIdx) * scaleExpertStride + static_cast<int64_t>(mTile) * scaleNumKTiles * 128
            + static_cast<int64_t>(kChunk) * 128 + static_cast<int64_t>(outerM) * 4 + innerM;
    }

    //! Read block-scale int32 word with byte-swap for Marlin dequant compatibility.
    //! Atom stores sequential FP8 bytes {s0,s1,s2,s3}; Marlin \c dequant_fp8_scales expects {s0,s2,s1,s3}.
    __device__ __forceinline__ int readBlockScaleWord(Dim3 const c) const
    {
        int64_t const idx = tileScaleIndex(c);
        uint32_t const w = static_cast<uint32_t>(block_scale[idx]);
        // Swap bytes 1 and 2: {s0,s1,s2,s3} → {s0,s2,s1,s3}
        return static_cast<int>((w & 0xFF0000FFu) | ((w & 0x0000FF00u) << 8) | ((w & 0x00FF0000u) >> 8));
    }

    //! Read block-scale int32 word using plain linear tile indexing (one scale word per tile payload).
    //! The word is already in Marlin-packed byte order — no byte-swap. Used for activation scale factors
    //! whose layout matches \c TRT_DynamicQuantize output (contiguous, not atom-swizzled).
    __device__ __forceinline__ int readBlockScaleWordLinear(Dim3 const c) const
    {
        int64_t const idx = tileOffsetInt4(c) / kNvfp4Int4PerTilePayload;
        return block_scale[idx];
    }

    //! Loads one \c int4 chunk (\c 4× \c uint32 NVFP4 lane packs) from the tile payload.
    //! \param chunk \c 0 or \c 1 — tile payload is \ref kNvfp4Int4PerTilePayload \c int4 vectors.
    __device__ __forceinline__ void loadTileUint4(Dim3 const tile, int const chunk, uint4& out) const
    {
        int64_t const o = tileOffsetInt4(tile);
        int4 const* const p = quantized_data + o;
        uint4 const* const u = reinterpret_cast<uint4 const*>(p);
        out = u[chunk];
    }

    //! Loads one \c uint32 lane (8 packed NVFP4 values) from the tile payload.
    //! \param chunk \c 0 or \c 1; \param lane \c 0..\c 3 within that chunk (see \ref loadTileUint4).
    __device__ __forceinline__ uint32_t loadTileUint32Lane(Dim3 const tile, int const chunk, int const lane) const
    {
        int64_t const o = tileOffsetInt4(tile);
        uint32_t const* const p = reinterpret_cast<uint32_t const*>(quantized_data + o);
        return p[chunk * 4 + lane];
    }
};

//! \p global_scale[\p idx] if \p global_scale is non-null, else \c 1.f.
__device__ __forceinline__ float nvfp4TensorScaleAt(float const* global_scale, int const idx)
{
    return global_scale != nullptr ? global_scale[idx] : 1.f;
}

} // namespace trt_edgellm
