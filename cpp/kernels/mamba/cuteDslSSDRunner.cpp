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

#ifdef CUTE_DSL_SSD_ENABLED

#include "cuteDslSSDRunner.h"

#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace trt_edgellm
{

ssd_prefill_d128_n128_Kernel_Module_t CuteDslSSDRunner::sD128N128Module = {};
ssd_prefill_d64_n128_Kernel_Module_t CuteDslSSDRunner::sD64N128Module = {};
ssd_prefill_d128_n64_Kernel_Module_t CuteDslSSDRunner::sD128N64Module = {};
ssd_prefill_d64_n64_Kernel_Module_t CuteDslSSDRunner::sD64N64Module = {};
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
ssd_prefill_blackwell_d64_n128_Kernel_Module_t CuteDslSSDRunner::sBlackwellD64N128Module = {};
ssd_prefill_blackwell_d64_n64_Kernel_Module_t CuteDslSSDRunner::sBlackwellD64N64Module = {};
#endif
bool CuteDslSSDRunner::sLoaded = false;

static std::mutex sSSDMutex;

// Tensor struct helpers — contiguous row-major layout.
#define SET_5D_TENSOR(tensor, data_ptr, d0, d1, d2, d3, d4)                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
        (tensor).dynamic_shapes[1] = (d1);                                                                             \
        (tensor).dynamic_shapes[2] = (d2);                                                                             \
        (tensor).dynamic_shapes[3] = (d3);                                                                             \
        (tensor).dynamic_shapes[4] = (d4);                                                                             \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(d1) * (d2) * (d3) * (d4);                                   \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(d2) * (d3) * (d4);                                          \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(d3) * (d4);                                                 \
        (tensor).dynamic_strides[3] = static_cast<int64_t>(d4);                                                        \
    } while (0)

#define SET_4D_TENSOR(tensor, data_ptr, d0, d1, d2, d3)                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
        (tensor).dynamic_shapes[1] = (d1);                                                                             \
        (tensor).dynamic_shapes[2] = (d2);                                                                             \
        (tensor).dynamic_shapes[3] = (d3);                                                                             \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(d1) * (d2) * (d3);                                          \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(d2) * (d3);                                                 \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(d3);                                                        \
    } while (0)

#define SET_3D_TENSOR(tensor, data_ptr, d0, d1, d2)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
        (tensor).dynamic_shapes[1] = (d1);                                                                             \
        (tensor).dynamic_shapes[2] = (d2);                                                                             \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(d1) * (d2);                                                 \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(d2);                                                        \
    } while (0)

#define SET_1D_TENSOR(tensor, data_ptr, d0)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (d0);                                                                             \
    } while (0)

bool CuteDslSSDRunner::canImplement(int32_t dim, int32_t dstate, int32_t smVersion)
{
    if (smVersion < 80)
        return false;
    return (dim == 64 || dim == 128) && (dstate == 64 || dstate == 128);
}

bool CuteDslSSDRunner::loadKernelModules()
{
    std::lock_guard<std::mutex> lock(sSSDMutex);
    if (sLoaded)
        return true;

    try
    {
        ssd_prefill_d128_n128_Kernel_Module_Load(&sD128N128Module);
        ssd_prefill_d64_n128_Kernel_Module_Load(&sD64N128Module);
        ssd_prefill_d128_n64_Kernel_Module_Load(&sD128N64Module);
        ssd_prefill_d64_n64_Kernel_Module_Load(&sD64N64Module);
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
        ssd_prefill_blackwell_d64_n128_Kernel_Module_Load(&sBlackwellD64N128Module);
        ssd_prefill_blackwell_d64_n64_Kernel_Module_Load(&sBlackwellD64N64Module);
        LOG_DEBUG("CuTe DSL SSD kernel modules (4 SM80 + 2 Blackwell) loaded");
#else
        LOG_DEBUG("CuTe DSL SSD kernel modules (4 SM80 variants) loaded");
#endif
        sLoaded = true;
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL SSD kernel modules");
        return false;
    }
}

void CuteDslSSDRunner::unloadKernelModules()
{
    std::lock_guard<std::mutex> lock(sSSDMutex);
    if (sLoaded)
    {
        ssd_prefill_d128_n128_Kernel_Module_Unload(&sD128N128Module);
        ssd_prefill_d64_n128_Kernel_Module_Unload(&sD64N128Module);
        ssd_prefill_d128_n64_Kernel_Module_Unload(&sD128N64Module);
        ssd_prefill_d64_n64_Kernel_Module_Unload(&sD64N64Module);
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
        ssd_prefill_blackwell_d64_n128_Kernel_Module_Unload(&sBlackwellD64N128Module);
        ssd_prefill_blackwell_d64_n64_Kernel_Module_Unload(&sBlackwellD64N64Module);
#endif
        sLoaded = false;
    }
}

int CuteDslSSDRunner::run(SSDParams const& params, cudaStream_t stream)
{
#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    // SM100-110 with D=64: use Blackwell persistent kernel (TMA/wgmma/TMEM).
    // SM120+ lacks TMEM/wgmma — falls through to SM80 kernel.
    if (params.smVersion >= 100 && params.smVersion < 120 && params.dim == 64
        && (params.dstate == 128 || params.dstate == 64))
        return runPrefillBlackwell(params, stream);
#endif
    return runPrefill(params, stream);
}

// Macro to fill tensor structs and call the SM80 wrapper for a given variant prefix.
// All SM80 variants share the same struct layout — only the type name prefix differs.
// Relies on local variables: params, n, seq_len, nheads, dim, dstate, ngroups, nchunks,
// nheads_ngroups_ratio, stream.
// clang-format off
#define CALL_SSD_PREFILL(PREFIX, MODULE)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_x_t xTensor{};                                                                                 \
        SET_4D_TENSOR(xTensor, params.x, n, seq_len, nheads, dim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_dt_in_t dtTensor{};                                                                            \
        SET_3D_TENSOR(dtTensor, params.dt, n, seq_len, nheads);                                                        \
                                                                                                                       \
        PREFIX##_Tensor_A_t aTensor{};                                                                                 \
        SET_1D_TENSOR(aTensor, params.A, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_B_t bTensor{};                                                                                 \
        bTensor.data = params.B;                                                                                       \
        bTensor.dynamic_shapes[0] = n;                                                                                 \
        bTensor.dynamic_shapes[1] = seq_len;                                                                           \
        bTensor.dynamic_shapes[2] = ngroups;                                                                           \
        bTensor.dynamic_strides[0] = static_cast<int64_t>(seq_len) * ngroups * dstate;                                 \
        bTensor.dynamic_strides[1] = static_cast<int64_t>(ngroups) * dstate;                                           \
                                                                                                                       \
        PREFIX##_Tensor_C_t cTensor{};                                                                                 \
        cTensor.data = params.C;                                                                                       \
        cTensor.dynamic_shapes[0] = n;                                                                                 \
        cTensor.dynamic_shapes[1] = seq_len;                                                                           \
        cTensor.dynamic_shapes[2] = ngroups;                                                                           \
        cTensor.dynamic_strides[0] = static_cast<int64_t>(seq_len) * ngroups * dstate;                                 \
        cTensor.dynamic_strides[1] = static_cast<int64_t>(ngroups) * dstate;                                           \
                                                                                                                       \
        PREFIX##_Tensor_D_t dTensor{};                                                                                 \
        SET_1D_TENSOR(dTensor, params.D, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_dt_bias_t dtBiasTensor{};                                                                      \
        SET_1D_TENSOR(dtBiasTensor, params.dt_bias, nheads);                                                           \
                                                                                                                       \
        PREFIX##_Tensor_z_t zTensor{};                                                                                 \
        SET_4D_TENSOR(zTensor, params.z ? params.z : params.x, n, seq_len, nheads, dim);                               \
                                                                                                                       \
        PREFIX##_Tensor_output_t outputTensor{};                                                                       \
        SET_4D_TENSOR(outputTensor, params.output, n, seq_len, nheads, dim);                                           \
                                                                                                                       \
        PREFIX##_Tensor_final_states_t finalStatesTensor{};                                                            \
        SET_4D_TENSOR(finalStatesTensor, params.state, n, nheads, dim, dstate);                                        \
                                                                                                                       \
        void* ws = params.workspace;                                                                                   \
        size_t offset = 0;                                                                                             \
                                                                                                                       \
        PREFIX##_Tensor_dA_cumsum_t dACumsumTensor{};                                                                  \
        size_t dACumsumBytes = static_cast<size_t>(n) * nheads * nchunks * 128 * sizeof(float);                        \
        SET_4D_TENSOR(dACumsumTensor, static_cast<char*>(ws) + offset, n, nheads, nchunks, 128);                       \
        offset += dACumsumBytes;                                                                                       \
                                                                                                                       \
        PREFIX##_Tensor_dt_proc_t dtProcTensor{};                                                                      \
        SET_4D_TENSOR(dtProcTensor, static_cast<char*>(ws) + offset, n, nheads, nchunks, 128);                         \
        offset += dACumsumBytes;                                                                                       \
                                                                                                                       \
        PREFIX##_Tensor_states_t statesTensor{};                                                                       \
        size_t statesBytes = static_cast<size_t>(n) * nchunks * nheads * dim * dstate * sizeof(float);                 \
        SET_5D_TENSOR(statesTensor, static_cast<char*>(ws) + offset, n, nchunks, nheads, dim, dstate);                 \
        offset += statesBytes;                                                                                         \
                                                                                                                       \
        PREFIX##_Tensor_prev_states_t prevStatesTensor{};                                                              \
        SET_5D_TENSOR(prevStatesTensor, static_cast<char*>(ws) + offset, n, nchunks, nheads, dim, dstate);             \
        offset += statesBytes;                                                                                         \
                                                                                                                       \
        PREFIX##_Tensor_CB_t cbTensor{};                                                                               \
        size_t cbBytes = static_cast<size_t>(n) * nchunks * ngroups * 128 * 128 * sizeof(float);                       \
        SET_5D_TENSOR(cbTensor, static_cast<char*>(ws) + offset, n, nchunks, ngroups, 128, 128);                       \
        offset += cbBytes;                                                                                             \
                                                                                                                       \
        cudaMemsetAsync(ws, 0, offset, stream);                                                                        \
                                                                                                                       \
        return cute_dsl_##PREFIX##_wrapper(&(MODULE), &xTensor, &dtTensor, &aTensor, &bTensor, &cTensor, &dTensor,     \
            &dtBiasTensor, &zTensor, &outputTensor, &finalStatesTensor, &dACumsumTensor, &dtProcTensor,                \
            &statesTensor, &prevStatesTensor, &cbTensor, seq_len, nchunks, n, nheads, nheads_ngroups_ratio,            \
            ngroups, stream);                                                                                          \
    } while (0)
// clang-format on

int CuteDslSSDRunner::runPrefill(SSDParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL SSD prefill kernel module not loaded.");
        return -1;
    }

    int32_t const n = params.batch;
    int32_t const seq_len = params.seq_len;
    int32_t const nheads = params.nheads;
    int32_t const dim = params.dim;
    int32_t const dstate = params.dstate;
    int32_t const ngroups = params.ngroups;
    int32_t const nchunks = (seq_len + 127) / 128; // CHUNK_SIZE = 128
    int32_t const nheads_ngroups_ratio = nheads / ngroups;

    if (dim == 128 && dstate == 128)
    {
        CALL_SSD_PREFILL(ssd_prefill_d128_n128, sD128N128Module);
    }
    else if (dim == 64 && dstate == 128)
    {
        CALL_SSD_PREFILL(ssd_prefill_d64_n128, sD64N128Module);
    }
    else if (dim == 128 && dstate == 64)
    {
        CALL_SSD_PREFILL(ssd_prefill_d128_n64, sD128N64Module);
    }
    else if (dim == 64 && dstate == 64)
    {
        CALL_SSD_PREFILL(ssd_prefill_d64_n64, sD64N64Module);
    }

    LOG_ERROR("CuTe DSL SSD prefill: unsupported dim=%d dstate=%d", dim, dstate);
    return -1;
}

#undef CALL_SSD_PREFILL

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED

// Macro to fill tensor structs and call the Blackwell wrapper for a given variant prefix.
// All Blackwell D=64 variants share the same struct layout — only the type name prefix differs.
// Relies on local variables: params, n, seq_len, nheads, dim, dstate, ngroups, nchunks, stream.
// clang-format off
#define CALL_SSD_PREFILL_BLACKWELL(PREFIX, MODULE)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_x_t xTensor{};                                                                                 \
        SET_4D_TENSOR(xTensor, params.x, n, seq_len, nheads, dim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_dt_in_t dtTensor{};                                                                            \
        SET_3D_TENSOR(dtTensor, params.dt, n, seq_len, nheads);                                                        \
                                                                                                                       \
        PREFIX##_Tensor_A_t aTensor{};                                                                                 \
        SET_1D_TENSOR(aTensor, params.A, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_B_t bTensor{};                                                                                 \
        SET_4D_TENSOR(bTensor, params.B, n, seq_len, ngroups, dstate);                                                 \
                                                                                                                       \
        PREFIX##_Tensor_C_t cTensor{};                                                                                 \
        SET_4D_TENSOR(cTensor, params.C, n, seq_len, ngroups, dstate);                                                 \
                                                                                                                       \
        PREFIX##_Tensor_D_fp32_t dTensor{};                                                                            \
        SET_1D_TENSOR(dTensor, params.D, nheads);                                                                      \
                                                                                                                       \
        PREFIX##_Tensor_dt_bias_t dtBiasTensor{};                                                                      \
        SET_1D_TENSOR(dtBiasTensor, params.dt_bias, nheads);                                                           \
                                                                                                                       \
        PREFIX##_Tensor_output_t outputTensor{};                                                                       \
        SET_4D_TENSOR(outputTensor, params.output, n, seq_len, nheads, dim);                                           \
                                                                                                                       \
        PREFIX##_Tensor_state_t stateTensor{};                                                                         \
        SET_4D_TENSOR(stateTensor, params.state, n, nheads, dim, dstate);                                              \
                                                                                                                       \
        void* ws = params.workspace;                                                                                   \
        size_t offset = 0;                                                                                             \
                                                                                                                       \
        size_t cumsumBytes = static_cast<size_t>(n) * nheads * nchunks * 128 * sizeof(float);                          \
        PREFIX##_Tensor_dA_cumsum_t cumsumTensor{};                                                                    \
        SET_4D_TENSOR(cumsumTensor, static_cast<char*>(ws) + offset, n, nheads, nchunks, 128);                         \
        offset += cumsumBytes;                                                                                         \
                                                                                                                       \
        size_t dtProcBytes = static_cast<size_t>(n) * nheads * nchunks * 128 * sizeof(__half);                         \
        PREFIX##_Tensor_dt_proc_t dtProcTensor{};                                                                      \
        SET_4D_TENSOR(dtProcTensor, static_cast<char*>(ws) + offset, n, nheads, nchunks, 128);                         \
        offset += dtProcBytes;                                                                                         \
                                                                                                                       \
        size_t yBytes = static_cast<size_t>(n) * nheads * dim * nchunks * 128 * sizeof(__half);                        \
        PREFIX##_Tensor_y_ws_t yTensor{};                                                                              \
        SET_5D_TENSOR(yTensor, static_cast<char*>(ws) + offset, n, nheads, dim, nchunks, 128);                         \
        offset += yBytes;                                                                                              \
                                                                                                                       \
        size_t fstateBytes = static_cast<size_t>(n) * nheads * dim * dstate * sizeof(__half);                          \
        PREFIX##_Tensor_fstate_ws_t fstateTensor{};                                                                    \
        SET_4D_TENSOR(fstateTensor, static_cast<char*>(ws) + offset, n, nheads, dim, dstate);                          \
        offset += fstateBytes;                                                                                         \
                                                                                                                       \
        size_t dFp16Bytes = static_cast<size_t>(nheads) * sizeof(__half);                                              \
        PREFIX##_Tensor_D_fp16_t dFp16Tensor{};                                                                        \
        SET_1D_TENSOR(dFp16Tensor, static_cast<char*>(ws) + offset, nheads);                                           \
        offset += dFp16Bytes;                                                                                          \
                                                                                                                       \
        cudaMemsetAsync(ws, 0, offset, stream);                                                                        \
                                                                                                                       \
        return cute_dsl_##PREFIX##_wrapper(&(MODULE), &xTensor, &dtTensor, &aTensor, &bTensor, &cTensor, &dTensor,     \
            &dtBiasTensor, &outputTensor, &stateTensor,                                                                \
            &cumsumTensor, &dtProcTensor, &yTensor, &fstateTensor, &dFp16Tensor,                                       \
            seq_len, nchunks, stream);                                                                                 \
    } while (0)
// clang-format on

int CuteDslSSDRunner::runPrefillBlackwell(SSDParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL SSD Blackwell prefill kernel module not loaded.");
        return -1;
    }

    int32_t const n = params.batch;
    int32_t const seq_len = params.seq_len;
    int32_t const nheads = params.nheads;
    int32_t const dim = params.dim;
    int32_t const dstate = params.dstate;
    int32_t const ngroups = params.ngroups;
    int32_t const nchunks = (seq_len + 127) / 128;

    if (dstate == 128)
    {
        CALL_SSD_PREFILL_BLACKWELL(ssd_prefill_blackwell_d64_n128, sBlackwellD64N128Module);
    }
    else if (dstate == 64)
    {
        CALL_SSD_PREFILL_BLACKWELL(ssd_prefill_blackwell_d64_n64, sBlackwellD64N64Module);
    }

    LOG_ERROR("CuTe DSL SSD Blackwell prefill: unsupported dstate=%d", dstate);
    return -1;
}

#undef CALL_SSD_PREFILL_BLACKWELL
#endif

size_t CuteDslSSDRunner::getWorkspaceSize(
    int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups)
{
    int32_t const nchunks = (seqLen + 127) / 128;
    size_t size = 0;
    // SM80 intermediates (cumsum, dt_proc, states, prev_states, CB)
    size += static_cast<size_t>(batch) * nheads * nchunks * 128 * sizeof(float);          // dA_cumsum
    size += static_cast<size_t>(batch) * nheads * nchunks * 128 * sizeof(float);          // dt_proc
    size += static_cast<size_t>(batch) * nchunks * nheads * dim * dstate * sizeof(float); // states
    size += static_cast<size_t>(batch) * nchunks * nheads * dim * dstate * sizeof(float); // prev_states
    size += static_cast<size_t>(batch) * nchunks * ngroups * 128 * 128 * sizeof(float);   // CB

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED
    // Blackwell workspace: cumsum_delta(fp32) + delta(fp16) + y(fp16) + fstate(fp16)
    size_t bwSize = 0;
    bwSize += static_cast<size_t>(batch) * nheads * nchunks * 128 * sizeof(float);        // cumsum_delta
    bwSize += static_cast<size_t>(batch) * nheads * nchunks * 128 * sizeof(__half);       // delta
    bwSize += static_cast<size_t>(batch) * nheads * dim * nchunks * 128 * sizeof(__half); // y
    bwSize += static_cast<size_t>(batch) * nheads * dim * dstate * sizeof(__half);        // fstate
    bwSize += static_cast<size_t>(nheads) * sizeof(__half);                               // D fp16
    size = std::max(size, bwSize);
#endif
    return size;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_SSD_ENABLED
