/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Prebuilt CuTe DSL archives call cudaLibrary* / cudaKernelSetAttributeForDevice
 * (CUDA runtime library management ABI). Some toolchains / libcudart builds
 * (e.g. CUDA 12.6 on embedded) omit these exports while the driver still
 * implements cuLibrary* / cuKernelSetAttribute. Provide weak definitions that
 * forward to the driver so linking succeeds; a strong libcudart symbol wins when
 * present.
 *
 * Parameter types use CUDA Driver API names from cuda.h so this compiles when
 * cuda_runtime.h lacks cudaJitOption / cudaLibraryOption / cudaFuncAttribute.
 * Layout matches the runtime ABI used by the prebuilt archive.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#if defined(__GNUC__)
#define TRT_EDGELLM_CUTEDSL_SHIM_WEAK __attribute__((weak))
#else
#define TRT_EDGELLM_CUTEDSL_SHIM_WEAK
#endif
/* CUresult and cudaError_t have different enum values for non-zero codes.
 * Map success to cudaSuccess and everything else to cudaErrorUnknown to avoid
 * propagating driver error codes as runtime error codes. */
#define SHIM_RETURN(r) return (r) == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown

TRT_EDGELLM_CUTEDSL_SHIM_WEAK cudaError_t cudaLibraryLoadData(CUlibrary* library, void const* code,
    CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions,
    void** libraryOptionValues, unsigned int numLibraryOptions)
{
    CUresult const r = cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
        libraryOptionValues, numLibraryOptions);
    SHIM_RETURN(r);
}
TRT_EDGELLM_CUTEDSL_SHIM_WEAK cudaError_t cudaLibraryUnload(CUlibrary library)
{
    CUresult const r = cuLibraryUnload(library);
    SHIM_RETURN(r);
}
TRT_EDGELLM_CUTEDSL_SHIM_WEAK cudaError_t cudaLibraryGetKernel(CUkernel* pKernel, CUlibrary library, char const* name)
{
    CUresult const r = cuLibraryGetKernel(pKernel, library, name);
    SHIM_RETURN(r);
}
TRT_EDGELLM_CUTEDSL_SHIM_WEAK cudaError_t cudaKernelSetAttributeForDevice(
    CUkernel kernel, CUfunction_attribute attr, int value, int device)
{
    CUresult const r = cuKernelSetAttribute(attr, value, kernel, (CUdevice) device);
    SHIM_RETURN(r);
}
#undef SHIM_RETURN

#if defined(CUTEDSL_WRAP_LAUNCH_KERNEL_EX)
/* cudaLaunchKernelExC(config, func, args) expects 'func' to be a host-side runtime
 * kernel stub registered via fat-binary tables.  Kernels loaded through cuLibraryLoadData
 * are never registered that way, so passing the CUfunction returned by cuKernelGetFunction
 * to cudaLaunchKernelExC causes a silent no-op (output stays all-zero).
 *
 * Instead, build a CUlaunchConfig from the runtime cudaLaunchConfig_t and call
 * cuLaunchKernelEx (driver API) directly.  The driver accepts any CUfunction regardless
 * of how its parent library was loaded.
 *
 * Struct layout notes (CUDA 12.6):
 *   cudaLaunchConfig_t.dynamicSmemBytes  is size_t (8 B)
 *   CUlaunchConfig.sharedMemBytes        is unsigned int (4 B)  — truncate, safe for typical smem sizes.
 *   cudaLaunchAttribute / CUlaunchAttribute have the same binary layout; pointer cast is safe.
 */
cudaError_t __wrap__cudaLaunchKernelEx(cudaLaunchConfig_t const* config, cudaKernel_t kernel, void** kernelParams)
{
    CUfunction fn = NULL;
    CUresult cr = cuKernelGetFunction(&fn, (CUkernel) kernel);
    if (cr != CUDA_SUCCESS)
    {
        return (cudaError_t) cr;
    }
    CUlaunchConfig cu_cfg;
    cu_cfg.gridDimX = config->gridDim.x;
    cu_cfg.gridDimY = config->gridDim.y;
    cu_cfg.gridDimZ = config->gridDim.z;
    cu_cfg.blockDimX = config->blockDim.x;
    cu_cfg.blockDimY = config->blockDim.y;
    cu_cfg.blockDimZ = config->blockDim.z;
    cu_cfg.sharedMemBytes = (unsigned int) config->dynamicSmemBytes;
    cu_cfg.hStream = (CUstream) config->stream;
    cu_cfg.attrs = (CUlaunchAttribute*) config->attrs;
    cu_cfg.numAttrs = config->numAttrs;
    cr = cuLaunchKernelEx(&cu_cfg, fn, kernelParams, NULL);
    return (cudaError_t) cr;
}
#endif /* CUTEDSL_WRAP_LAUNCH_KERNEL_EX */
