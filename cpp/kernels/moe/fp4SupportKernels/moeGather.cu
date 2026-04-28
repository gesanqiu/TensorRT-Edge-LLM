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

// MoE gather kernel: permutes FP4 data + atom-layout SF to expert-grouped order.
// One CTA per output row, 256 threads. int4-vectorized FP4 copy.

#include "moeGather.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int kGatherBlockSize = 256;
constexpr int kSfVecSize = 16;

inline int32_t padUp(int32_t a, int32_t b)
{
    return ((a + b - 1) / b) * b;
}

inline __device__ int32_t atomSFIndex(int32_t row, int32_t tileSfInt32s)
{
    int32_t mTile = row / 128;
    int32_t rowInTile = row % 128;
    int32_t rowGroup = rowInTile / 32;
    int32_t rowInGroup = rowInTile % 32;
    return mTile * tileSfInt32s + rowInGroup * 4 + rowGroup;
}

__launch_bounds__(kGatherBlockSize) __global__
    void moe_gather_kernel(int32_t const* __restrict__ srcData, int32_t* __restrict__ dstData,
        int32_t const* __restrict__ srcSF, int32_t* __restrict__ dstSF, int32_t const* __restrict__ permuteMap,
        int32_t topK, int32_t dataRowInt32s, int32_t numKTiles, int32_t tileSfInt32s)
{
    int32_t const row = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int32_t const expandedIdx = permuteMap[row];
    bool const isValid = (expandedIdx >= 0);
    int32_t const srcRow = isValid ? (expandedIdx / topK) : 0;

    int64_t const srcDataBase = static_cast<int64_t>(srcRow) * dataRowInt32s;
    int64_t const dstDataBase = static_cast<int64_t>(row) * dataRowInt32s;
    int32_t const dataRowInt4s = dataRowInt32s / 4;
    int32_t const dataRowTail = dataRowInt32s - dataRowInt4s * 4;

    if (isValid)
    {
        int4 const* __restrict__ srcVec = reinterpret_cast<int4 const*>(srcData + srcDataBase);
        int4* __restrict__ dstVec = reinterpret_cast<int4*>(dstData + dstDataBase);
        for (int32_t i = tid; i < dataRowInt4s; i += kGatherBlockSize)
            dstVec[i] = srcVec[i];
        int32_t const tailStart = dataRowInt4s * 4;
        for (int32_t i = tid; i < dataRowTail; i += kGatherBlockSize)
            dstData[dstDataBase + tailStart + i] = srcData[srcDataBase + tailStart + i];
    }
    else
    {
        int4* __restrict__ dstVec = reinterpret_cast<int4*>(dstData + dstDataBase);
        int4 const zero4 = make_int4(0, 0, 0, 0);
        for (int32_t i = tid; i < dataRowInt4s; i += kGatherBlockSize)
            dstVec[i] = zero4;
        int32_t const tailStart = dataRowInt4s * 4;
        for (int32_t i = tid; i < dataRowTail; i += kGatherBlockSize)
            dstData[dstDataBase + tailStart + i] = 0;
    }

    int32_t const dstBase = atomSFIndex(row, tileSfInt32s);
    int32_t const srcBase = atomSFIndex(srcRow, tileSfInt32s);

    if (isValid)
    {
        for (int32_t i = tid; i < numKTiles; i += kGatherBlockSize)
            dstSF[dstBase + i * 128] = srcSF[srcBase + i * 128];
    }
    else
    {
        for (int32_t i = tid; i < numKTiles; i += kGatherBlockSize)
            dstSF[dstBase + i * 128] = 0;
    }
}

} // namespace

void launchMoeGather(rt::Tensor const& srcFP4, rt::Tensor& dstFP4, rt::Tensor const& srcSF, rt::Tensor& dstSF,
    rt::Tensor const& permuteMap, int32_t permutedM, int32_t topK, int32_t hiddenSize, cudaStream_t stream)
{
    if (permutedM <= 0)
        return;

    int32_t const K = hiddenSize;
    int32_t const dataRowInt32s = K / 8;
    int32_t const sfCols = K / kSfVecSize;
    int32_t const paddedSfCols = padUp(sfCols, 4);
    int32_t const numKTiles = paddedSfCols / 4;
    int32_t const tileSfInt32s = 32 * paddedSfCols;

    dim3 const grid(permutedM, 1, 1);
    dim3 const block(kGatherBlockSize, 1, 1);

    moe_gather_kernel<<<grid, block, 0, stream>>>(static_cast<int32_t const*>(srcFP4.rawPointer()),
        static_cast<int32_t*>(dstFP4.rawPointer()), static_cast<int32_t const*>(srcSF.rawPointer()),
        static_cast<int32_t*>(dstSF.rawPointer()), permuteMap.dataPointer<int32_t>(), topK, dataRowInt32s, numKTiles,
        tileSfInt32s);
}

} // namespace kernel
} // namespace trt_edgellm
