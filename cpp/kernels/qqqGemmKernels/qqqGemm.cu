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
 * Adapted from https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
 * https://github.com/HandH1998/QQQ/blob/main/csrc/qqq_gemm.cu
 * Copyright (C) 2024 HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 */

#include "qqqGemm.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// Internal implementation details — Marlin W4A8 kernel
// ============================================================================
namespace trt_edgellm
{
namespace kernel
{
namespace qqq_detail
{

constexpr int ceildiv(int a, int b)
{
    return (a + b - 1) / b;
}

template <typename T, int n>
struct Vec
{
    T elems[n];

    __device__ T& operator[](int i)
    {
        return elems[i];
    }
};

using I4 = Vec<int, 4>;

// Matrix fragments for mma m16n8k16 with integer type
using FragA = Vec<uint32_t, 2>;
using FragB = Vec<uint32_t, 1>;
using FragC = Vec<int, 4>;
using FragS_GROUP = Vec<half2, 1>;
using FragS_CHANNEL = Vec<float, 2>;

__device__ inline void cp_async4_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    int const BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 "   .reg .pred p;\n"
                 "   setp.ne.b32 p, %0, 0;\n"
                 "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int) pred),
        "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, void const* glob_ptr)
{
    int const BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 "   cp.async.cg.shared.global [%0], [%1], %2;\n"
                 "}\n" ::"r"(smem),
        "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async1(void* smem_ptr, void const* glob_ptr)
{
    int const BYTES = 4;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("{\n"
                 "   cp.async.ca.shared.global [%0], [%1], %2;\n"
                 "}\n" ::"r"(smem),
        "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 INT8 tensor core MMA: s32 = s8 * s8 + s32
__device__ inline void mma(FragA const& a_frag, FragB const& frag_b, FragC& frag_c)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);
    int* c = reinterpret_cast<int*>(&frag_c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__device__ inline void ldsm4(FragA& frag_a, void const* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(a[0]), "=r"(a[1]) : "r"(smem));
}

inline __device__ half2 float2_to_half2(float2 f)
{
    uint32_t res;
    uint16_t h0, h1;
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h0) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h1) : "f"(f.y));
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(res) : "h"(h0), "h"(h1));
    return reinterpret_cast<half2&>(res);
}

inline __device__ float int32_to_float(int h)
{
    float res;
    asm volatile("cvt.rn.f32.s32 %0, %1;\n" : "=f"(res) : "r"(h));
    return res;
}

// Per-channel dequant: keep upper 4 bits of each byte in-place (acts as left-shift-by-4 for FastINT4toINT8)
__device__ inline FragB dequant_per_channel(int q)
{
    static constexpr int MASK = 0xf0f0f0f0;
    FragB frag_b;
    frag_b[0] = (q & MASK);
    return frag_b;
}

template <int lut>
__device__ inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Per-group dequant: INT4 -> FP16 -> dequant*group_scale -> requant_to_INT8 (FusedDequantQuant + FastFP16toINT8)
__device__ inline FragB dequant_per_group(int q, FragS_GROUP& frag_s, int i)
{
    static constexpr uint32_t LO = 0x000f000f;
    static constexpr uint32_t HI = 0x00f000f0;
    static constexpr uint32_t EX = 0x64006400;
    uint32_t t0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    uint32_t t1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    static constexpr uint32_t SUB = 0x64086408;
    static constexpr uint32_t MUL = 0x2c002c00;
    static constexpr uint32_t ADD = 0xd480d480;
    *reinterpret_cast<half2*>(&t0) = __hsub2(*reinterpret_cast<half2*>(&t0), *reinterpret_cast<half2 const*>(&SUB));
    *reinterpret_cast<half2*>(&t1)
        = __hfma2(*reinterpret_cast<half2*>(&t1), *reinterpret_cast<half2 const*>(&MUL), *reinterpret_cast<half2 const*>(&ADD));

    uint16_t s = reinterpret_cast<uint16_t*>(&frag_s)[i];
    uint32_t double_s;
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(double_s) : "h"(s), "h"(s));
    static constexpr uint32_t MAGIC_NUM = 0x64806480;
    *reinterpret_cast<half2*>(&t0) = __hfma2(
        *reinterpret_cast<half2*>(&t0), *reinterpret_cast<half2*>(&double_s), *reinterpret_cast<half2 const*>(&MAGIC_NUM));
    *reinterpret_cast<half2*>(&t1) = __hfma2(
        *reinterpret_cast<half2*>(&t1), *reinterpret_cast<half2*>(&double_s), *reinterpret_cast<half2 const*>(&MAGIC_NUM));
    FragB frag_b;
    uint32_t uint8s;
    static constexpr uint32_t MASK_0246 = 0x6420;
    static constexpr uint32_t UINT8s_TO_INT8s_MASK = 0x80808080;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(uint8s) : "r"(t0), "r"(t1), "n"(MASK_0246));
    frag_b[0] = (uint8s ^ UINT8s_TO_INT8s_MASK);
    return frag_b;
}

__device__ inline void barrier_acquire(int* lock, int count)
{
    if (threadIdx.x == 0)
    {
        int state = -1;
        do
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        while (state != count);
    }
    __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
            lock[0] = 0;
            return;
        }
        int val = 1;
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}

// ============================================================================
// Marlin W4A8 GEMM kernel template
// ============================================================================
template <int const threads, int const thread_m_blocks, int const thread_n_blocks, int const thread_k_blocks,
    int const stages, int const group_blocks = -1>
__global__ void Marlin(int4 const* __restrict__ A, int4 const* __restrict__ B, int4* __restrict__ C,
    int4* __restrict__ D, float const* __restrict__ s1, int4 const* __restrict__ s2, int4 const* __restrict__ s3,
    int prob_m, int prob_n, int prob_k, int* locks)
{
    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks)
    {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m = 16 * thread_m_blocks;
    }

    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

    if constexpr (group_blocks != -1)
        iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

    int slice_row = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col = slice_col_par;
    int slice_iters;
    int slice_count = 0;
    int slice_idx;

    if (slice_col_par >= n_tiles)
    {
        A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 16;
        C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 4;
        D += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
        s1 += (slice_col_par / n_tiles) * 16 * thread_m_blocks;
        locks += (slice_col_par / n_tiles) * n_tiles;
        slice_col = slice_col_par % n_tiles;
    }

    auto init_slice = [&]()
    {
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
            slice_iters = 0;
        if (slice_iters == 0)
            return;
        if (slice_row + slice_iters > k_tiles)
            slice_iters = k_tiles - slice_row;
        slice_count = 1;
        slice_idx = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1))
        {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0)
                slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0))
                slice_idx = slice_count - 1;
            else
            {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0)
                    slice_idx--;
            }
        }
        if (slice_col == n_tiles)
        {
            A += 16 * thread_m_blocks * prob_k / 16;
            C += 16 * thread_m_blocks * prob_n / 4;
            D += 16 * thread_m_blocks * prob_n / 8;
            s1 += 16 * thread_m_blocks;
            locks += n_tiles;
            slice_col = 0;
        }
    };
    init_slice();

    int a_gl_stride = prob_k / 16;
    constexpr int a_sh_stride = 16 * thread_k_blocks / 16;
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 16;
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_rd_delta_o = 1 * ((threads / 32) / (thread_n_blocks / 4));
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

    int b_gl_stride = 16 * prob_n / 32;
    constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
    constexpr int b_sh_wr_delta = threads;
    constexpr int b_sh_rd_delta = threads;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

    constexpr int s1_sh_stride = 16 * thread_m_blocks;
    constexpr int s2_sh_stride = 16 * thread_n_blocks / 4;
    int s3_gl_stride = prob_n / 8;
    constexpr int s3_sh_stride = 16 * thread_n_blocks / 8;
    constexpr int s3_sh_stage = s3_sh_stride;
    int s3_gl_rd_delta = s3_gl_stride;

    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16);
    a_sh_rd += 1 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s1_gl_rd = threadIdx.x;
    int s1_sh_wr = (threadIdx.x / 16) * 16 + (threadIdx.x % 8) * 2 + (threadIdx.x % 16) / 8;
    int s1_sh_rd = (threadIdx.x % 32) / 4;
    bool s1_sh_wr_pred = threadIdx.x < prob_m;

    int s2_gl_rd = s2_sh_stride * slice_col + threadIdx.x;
    int s2_sh_wr = threadIdx.x;
    int s2_sh_rd = 16 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + 2 * ((threadIdx.x % 32) % 4);
    bool s2_sh_wr_pred = threadIdx.x < s2_sh_stride;

    int s3_gl_rd, s3_sh_wr, s3_sh_rd;
    bool s3_sh_wr_pred;
    if constexpr (group_blocks != -1)
    {
        s3_gl_rd
            = s3_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s3_sh_stride * slice_col + threadIdx.x;
        s3_sh_wr = threadIdx.x;
        s3_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
        s3_sh_wr_pred = threadIdx.x < s3_sh_stride;
    }

    bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

    auto transform_a = [&](int i)
    {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };
    int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
    {
#pragma unroll
        for (int j = 0; j < thread_m_blocks; j++)
            a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
    }

    int4 const* B_ptr[b_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    extern __shared__ int4 sh[];
    int4* sh_a = sh;
    int4* sh_b = sh_a + (stages * a_sh_stage);
    int4* sh_s1 = sh_b + (stages * b_sh_stage);
    int4* sh_s2 = sh_s1 + s1_sh_stride;
    int4* sh_s3 = sh_s2 + s2_sh_stride;

    FragA frag_a[2][thread_m_blocks];
    I4 frag_b_quant[2];
    FragC frag_c[thread_m_blocks][4][2];
    FragS_GROUP frag_s3[2][4];
    FragS_CHANNEL frag_s1[thread_m_blocks];
    FragS_CHANNEL frag_s2[2][4];

    auto zero_accums = [&]()
    {
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
            reinterpret_cast<int*>(frag_c)[i] = 0;
    };

    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true)
    {
        if (pred)
        {
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++)
            {
                cp_async4_pred(
                    &sh_a_stage[a_sh_wr_trans[i]], &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                    a_sh_wr_pred[i]);
            }
            int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++)
            {
                cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            if constexpr (group_blocks != -1)
            {
                if (pipe % (group_blocks / thread_k_blocks) == 0)
                {
                    int4* sh_s3_stage = sh_s3 + s3_sh_stage * pipe;
                    if (s3_sh_wr_pred)
                        cp_async4(&sh_s3_stage[s3_sh_wr], &s3[s3_gl_rd]);
                    s3_gl_rd += s3_gl_rd_delta;
                }
            }
        }
        cp_async_fence();
    };

    auto wait_for_stage = [&]()
    {
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    auto fetch_to_registers = [&](int k, int pipe)
    {
        if constexpr (group_blocks != -1)
        {
            int4* sh_s3_stage
                = sh_s3 + s3_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_s3[k % 2])[0] = sh_s3_stage[s3_sh_rd];
        }
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
            ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4* sh_b_stage = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    auto matmul = [&](int k)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            int b_quant = frag_b_quant[k % 2][j];
            FragB frag_b0, frag_b1;
            if constexpr (group_blocks != -1)
            {
                int b_quant_shift = b_quant >> 8;
                frag_b0 = dequant_per_group(b_quant, frag_s3[k % 2][j], 0);
                frag_b1 = dequant_per_group(b_quant_shift, frag_s3[k % 2][j], 1);
            }
            else
            {
                int b_quant_shift = b_quant << 4;
                frag_b0 = dequant_per_channel(b_quant);
                frag_b1 = dequant_per_channel(b_quant_shift);
            }
#pragma unroll
            for (int i = 0; i < thread_m_blocks; i++)
            {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    auto thread_block_reduce = [&]()
    {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1)
        {
            int red_idx = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta = b_sh_stride;
            int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

#pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++)
            {
#pragma unroll
                for (int i = red_off; i > 0; i /= 2)
                {
                    if (i <= red_idx && red_idx < 2 * i)
                    {
#pragma unroll
                        for (int j = 0; j < 4 * 2; j++)
                        {
                            int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                            if (i < red_off)
                            {
                                int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * j + red_sh_rd]);
                                int* c_wr = reinterpret_cast<int*>(&sh[red_sh_wr]);
#pragma unroll
                                for (int k = 0; k < 4; k++)
                                    reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0)
                {
#pragma unroll
                    for (int i = 0; i < 4 * 2; i++)
                    {
                        int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
                    }
                }
                __syncthreads();
            }
        }
    };

    auto global_reduce = [&](bool first = false, bool last = false)
    {
        constexpr int active_threads = 32 * thread_n_blocks / 4;
        if (threadIdx.x < active_threads)
        {
            int c_gl_stride = prob_n / 4;
            int c_gl_wr_delta_o = 8 * c_gl_stride;
            int c_gl_wr_delta_i = 8 * (active_threads / 32);
            int c_gl_wr
                = c_gl_stride * ((threadIdx.x % 32) / 4) + 8 * (threadIdx.x / 32) + (threadIdx.x % 4) * 2;
            c_gl_wr += (4 * thread_n_blocks) * slice_col;
            constexpr int c_sh_wr_delta = active_threads * 2;
            int c_sh_wr = 2 * threadIdx.x;
            int row = (threadIdx.x % 32) / 4;

            if (!first)
            {
#pragma unroll
                for (int i = 0; i < thread_m_blocks * 4; i++)
                {
                    cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i],
                        &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
                        i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
                    cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i + 1],
                        &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1],
                        i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
                }
                cp_async_fence();
                cp_async_wait<0>();
            }

#pragma unroll
            for (int i = 0; i < thread_m_blocks * 4; i++)
            {
                if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m)
                {
                    if (!first)
                    {
                        int4 d_red1 = sh[c_sh_wr + i * c_sh_wr_delta];
                        int4 d_red2 = sh[c_sh_wr + i * c_sh_wr_delta + 1];
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]
                                += reinterpret_cast<int*>(&d_red1)[j];
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)]
                                += reinterpret_cast<int*>(&d_red2)[j];
                    }
                    if (!last)
                    {
                        int4 d1, d2;
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<int*>(&d1)[j]
                                = reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)];
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<int*>(&d2)[j]
                                = reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)];
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = d1;
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1] = d2;
                    }
                }
            }
        }
    };

    auto write_result = [&]()
    {
        int d_gl_stride = prob_n / 8;
        constexpr int d_sh_stride = 2 * thread_n_blocks + 1;
        int d_gl_wr_delta = d_gl_stride * (threads / (2 * thread_n_blocks));
        constexpr int d_sh_rd_delta = d_sh_stride * (threads / (2 * thread_n_blocks));

        int d_gl_wr
            = d_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
        d_gl_wr += (2 * thread_n_blocks) * slice_col;
        int d_sh_wr = (4 * d_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
        d_sh_wr += 32 * (threadIdx.x / 32);
        int d_sh_rd
            = d_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
        int d_gl_wr_end = d_gl_stride * prob_m;

        auto write = [&](int idx, int c0, int c1, float a_s, FragS_CHANNEL& w_s)
        {
            float2 deq_res;
            deq_res.x = int32_to_float(c0) * w_s[0] * a_s;
            deq_res.y = int32_to_float(c1) * w_s[1] * a_s;
            ((half2*) sh)[idx] = float2_to_half2(deq_res);
        };

        if (threadIdx.x / 32 < thread_n_blocks / 4)
        {
#pragma unroll
            for (int i = 0; i < thread_m_blocks; i++)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    int wr = d_sh_wr + 8 * j;
                    write(wr + (4 * d_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s1[i][0],
                        frag_s2[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * d_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s1[i][1],
                        frag_s2[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * d_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s1[i][0],
                        frag_s2[j / 2][2 * (j % 2) + 1]);
                    write(wr + (4 * d_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s1[i][1],
                        frag_s2[j / 2][2 * (j % 2) + 1]);
                }
                d_sh_wr += 16 * (4 * d_sh_stride);
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++)
        {
            if (d_gl_wr < d_gl_wr_end)
            {
                D[d_gl_wr] = sh[d_sh_rd];
                d_gl_wr += d_gl_wr_delta;
                d_sh_rd += d_sh_rd_delta;
            }
        }
    };

    auto start_pipes = [&]()
    {
#pragma unroll
        for (int i = 0; i < stages - 1; i++)
            fetch_to_shared(i, i, i < slice_iters);
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    while (slice_iters)
    {
#pragma unroll
        for (int pipe = 0; pipe < stages;)
        {
#pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++)
            {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2)
                {
                    fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0)
                break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;

        if (slice_iters == 0)
        {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;
            if (last)
            {
                if (s1_sh_wr_pred)
                    cp_async1(&sh_s1[s1_sh_wr], &s1[s1_gl_rd]);
                if (s2_sh_wr_pred)
                    cp_async4(&sh_s2[s2_sh_wr], &s2[s2_gl_rd]);
                cp_async_fence();
            }
            thread_block_reduce();
            if (last)
            {
                cp_async_wait<0>();
                __syncthreads();
                if (threadIdx.x / 32 < thread_n_blocks / 4)
                {
#pragma unroll
                    for (int i = 0; i < thread_m_blocks; i++)
                    {
                        frag_s1[i][0] = *reinterpret_cast<float*>(&sh_s1[16 * i + 2 * s1_sh_rd]);
                        frag_s1[i][1] = *reinterpret_cast<float*>(&sh_s1[16 * i + 2 * s1_sh_rd + 1]);
                    }
                    reinterpret_cast<int4*>(&frag_s2)[0] = sh_s2[s2_sh_rd + 0];
                    reinterpret_cast<int4*>(&frag_s2)[1] = sh_s2[s2_sh_rd + 1];
                    reinterpret_cast<int4*>(&frag_s2)[2] = sh_s2[s2_sh_rd + 8];
                    reinterpret_cast<int4*>(&frag_s2)[3] = sh_s2[s2_sh_rd + 9];
                }
            }
            if (slice_count > 1)
            {
                barrier_acquire(&locks[slice_col], slice_idx);
                global_reduce(slice_idx == 0, last);
                barrier_release(&locks[slice_col], last);
            }
            if (last)
                write_result();
            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters)
            {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
                for (int i = 0; i < b_sh_wr_iters; i++)
                    B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
                if (slice_col == 0)
                {
#pragma unroll
                    for (int i = 0; i < b_sh_wr_iters; i++)
                        B_ptr[i] -= b_gl_stride;
                }
                s3_gl_rd = s3_sh_stride * slice_col + threadIdx.x;
                s2_gl_rd = s2_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}

// ============================================================================
// Dispatch helpers
// ============================================================================

static constexpr int USER_THREADS = 256;
static constexpr int STAGES = 4;
static constexpr int kMinThreadN = 64;
static constexpr int kMinThreadK = 64;

struct ThreadConfig
{
    int thread_k;
    int thread_n;
    int num_threads;
};

static constexpr ThreadConfig kSmallBatchConfigs[] = {
    {128, 128, 256},
    {128, 64, 128},
    {64, 256, 256},
    {64, 128, 128},
};

static constexpr ThreadConfig kLargeBatchConfigs[] = {
    {64, 256, 256},
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

inline bool isValidConfig(ThreadConfig const& cfg, int prob_m, int prob_n, int prob_k)
{
    if (cfg.thread_k == -1 || cfg.thread_n == -1 || cfg.num_threads == -1)
        return false;
    if (prob_k % cfg.thread_k != 0 || prob_n % cfg.thread_n != 0)
        return false;
    if (cfg.thread_k != 128 && cfg.thread_k != 64)
        return false;
    if (cfg.thread_n < kMinThreadN || cfg.thread_k < kMinThreadK)
        return false;
    if (cfg.num_threads < 128)
        return false;
    return true;
}

inline ThreadConfig determineThreadConfig(int prob_m, int prob_n, int prob_k)
{
    if (prob_m <= 16)
    {
        for (auto const& cfg : kSmallBatchConfigs)
            if (isValidConfig(cfg, prob_m, prob_n, prob_k))
                return cfg;
    }
    else
    {
        for (auto const& cfg : kLargeBatchConfigs)
            if (isValidConfig(cfg, prob_m, prob_n, prob_k))
                return cfg;
    }
    return {-1, -1, -1};
}

static constexpr int kErrProbShape = 1;
static constexpr int kErrKernShape = 2;

#define QQQ_CALL_IF_INNER(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, NUM_THREADS) \
    else if (thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS                   \
        && thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS                            \
        && num_threads == NUM_THREADS)                                                                   \
    {                                                                                                    \
        cudaFuncSetAttribute(                                                                            \
            Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);                                \
        Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>    \
            <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                                           \
                A_ptr, B_ptr, C_ptr, D_ptr, s1_ptr, s2_ptr, s3_ptr, prob_m, prob_n, prob_k, locks);      \
    }

#define QQQ_CALL_IF(N_BLOCKS, K_BLOCKS, NUM_THREADS)                   \
    QQQ_CALL_IF_INNER(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)         \
    QQQ_CALL_IF_INNER(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)          \
    QQQ_CALL_IF_INNER(2, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)         \
    QQQ_CALL_IF_INNER(2, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)          \
    QQQ_CALL_IF_INNER(3, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)         \
    QQQ_CALL_IF_INNER(3, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)          \
    QQQ_CALL_IF_INNER(4, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)         \
    QQQ_CALL_IF_INNER(4, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)

inline int qqqDispatch(void const* A, void const* B, void* C, void* D, void const* s1, void const* s2,
    void const* s3, int prob_m, int prob_n, int prob_k, void* workspace, int groupsize, int dev,
    cudaStream_t stream, int thread_k, int thread_n, int sms, int max_par)
{
    int tot_m = prob_m;
    int tot_m_blocks = ceildiv(tot_m, 16);
    int pad = 16 * tot_m_blocks - tot_m;

    if (sms == -1)
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    ThreadConfig th_config;
    if (thread_k != -1 && thread_n != -1)
    {
        th_config = {thread_k, thread_n, USER_THREADS};
    }
    else
    {
        th_config = determineThreadConfig(prob_m, prob_n, prob_k);
    }
    int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
    if (!isValidConfig(th_config, prob_m, prob_n, prob_k)
        || (group_blocks != -1 && prob_k % group_blocks != 0))
        return kErrProbShape;

    int num_threads = th_config.num_threads;
    thread_k = th_config.thread_k;
    thread_n = th_config.thread_n;
    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
    int blocks = sms;

    if (groupsize == -1)
    {
        if (s3 != nullptr)
            return kErrProbShape;
    }
    if (prob_m == 0 || prob_n == 0 || prob_k == 0)
        return 0;

    int4 const* A_ptr = static_cast<int4 const*>(A);
    int4 const* B_ptr = static_cast<int4 const*>(B);
    int4* C_ptr = static_cast<int4*>(C);
    int4* D_ptr = static_cast<int4*>(D);
    float const* s1_ptr = static_cast<float const*>(s1);
    int4 const* s2_ptr = static_cast<int4 const*>(s2);
    int4 const* s3_ptr = static_cast<int4 const*>(s3);
    int* locks = static_cast<int*>(workspace);

    int ret = 0;
    for (int i = 0; i < tot_m_blocks; i += 4)
    {
        int thread_m_blocks = tot_m_blocks - i;
        prob_m = tot_m - 16 * i;
        int par = 1;
        if (thread_m_blocks > 4)
        {
            par = (16 * thread_m_blocks - pad) / 64;
            if (par > max_par)
                par = max_par;
            prob_m = 64 * par;
            i += 4 * (par - 1);
            thread_m_blocks = 4;
        }

        // clang-format off
        if (false) {}
        QQQ_CALL_IF(8, 8, 256)
        QQQ_CALL_IF(16, 4, 256)
        QQQ_CALL_IF(8, 4, 128)
        QQQ_CALL_IF(4, 8, 128)
        else
            ret = kErrKernShape;
        // clang-format on

        A_ptr += 16 * thread_m_blocks * (prob_k / 16) * par;
        D_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
        s1_ptr += 16 * thread_m_blocks * par;
    }

    return ret;
}

#undef QQQ_CALL_IF_INNER
#undef QQQ_CALL_IF

// ============================================================================
// Dynamic per-token INT8 quantization kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void dynamicQuantKernel(
    half const* __restrict__ input, int8_t* __restrict__ output, float* __restrict__ scales, int K)
{
    int row = blockIdx.x;
    half const* row_in = input + static_cast<int64_t>(row) * K;
    int8_t* row_out = output + static_cast<int64_t>(row) * K;

    __shared__ float smem[BLOCK_SIZE];

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K; i += BLOCK_SIZE)
    {
        float val = __half2float(row_in[i]);
        local_max = fmaxf(local_max, fabsf(val));
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();

#pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }

    float scale = smem[0] / 127.0f;
    if (threadIdx.x == 0)
        scales[row] = scale;

    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    for (int i = threadIdx.x; i < K; i += BLOCK_SIZE)
    {
        float val = __half2float(row_in[i]);
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<int8_t>(q);
    }
}

} // namespace qqq_detail

// ============================================================================
// Public API
// ============================================================================

void dynamicQuantize(half const* input, int8_t* output, float* scales, int m, int k, cudaStream_t stream) noexcept
{
    if (m == 0 || k == 0)
        return;
    constexpr int kBlockSize = 256;
    qqq_detail::dynamicQuantKernel<kBlockSize><<<m, kBlockSize, 0, stream>>>(input, output, scales, k);
}

int qqqGemmForward(void const* A, void const* B, void* C, void* D, void const* s1, void const* s2, void const* s3,
    int prob_m, int prob_n, int prob_k, void* workspace, int groupsize, int dev, cudaStream_t stream, int thread_k,
    int thread_n, int sms, int max_par) noexcept
{
    return qqq_detail::qqqDispatch(
        A, B, C, D, s1, s2, s3, prob_m, prob_n, prob_k, workspace, groupsize, dev, stream, thread_k, thread_n, sms,
        max_par);
}

} // namespace kernel
} // namespace trt_edgellm
