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

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED

#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "kernels/moe/NvFP4MoEContiguousGemmRunner.h"
#include "kernels/moe/NvFP4MoEFC2FinalizeRunner.h"

using namespace trt_edgellm::kernel::nvfp4_moe;
using trt_edgellm::getSMVersion;

namespace
{

bool isSupportedSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

} // namespace

// Verify that FC1 kernel modules can be loaded and unloaded without error.
TEST(NvFP4MoERunnerTest, fc1LoadUnload)
{
    int32_t const smVersion = getSMVersion();
    if (!isSupportedSm(smVersion))
    {
        GTEST_SKIP() << "NvFP4 MoE kernels require SM100/101/110. Current SM=" << smVersion;
    }

    bool const loaded = NvFP4MoEContiguousGemmRunner::loadKernelModules();
    ASSERT_TRUE(loaded) << "Failed to load FC1 kernel modules";

    // Verify runner construction with Nemotron dimensions.
    NvFP4MoEContiguousGemmRunner fc1Runner(
        /*numLocalExperts=*/16, /*topK=*/6, /*n=*/1856, /*k=*/2688,
        /*tileSize=*/128, Activation::kRelu2);

    NvFP4MoEContiguousGemmRunner::unloadKernelModules();
    SUCCEED();
}

// Verify that FC2 kernel modules can be loaded and unloaded without error.
TEST(NvFP4MoERunnerTest, fc2LoadUnload)
{
    int32_t const smVersion = getSMVersion();
    if (!isSupportedSm(smVersion))
    {
        GTEST_SKIP() << "NvFP4 MoE kernels require SM100/101/110. Current SM=" << smVersion;
    }

    bool const loaded = NvFP4MoEFC2FinalizeRunner::loadKernelModules();
    ASSERT_TRUE(loaded) << "Failed to load FC2 kernel modules";

    // Verify runner construction with Nemotron dimensions.
    NvFP4MoEFC2FinalizeRunner fc2Runner(
        /*numLocalExperts=*/16, /*topK=*/6, /*n=*/2688, /*k=*/1856);

    NvFP4MoEFC2FinalizeRunner::unloadKernelModules();
    SUCCEED();
}

// Verify all three activation modes can construct FC1 runners.
TEST(NvFP4MoERunnerTest, fc1AllActivations)
{
    int32_t const smVersion = getSMVersion();
    if (!isSupportedSm(smVersion))
    {
        GTEST_SKIP() << "NvFP4 MoE kernels require SM100/101/110. Current SM=" << smVersion;
    }

    bool const loaded = NvFP4MoEContiguousGemmRunner::loadKernelModules();
    ASSERT_TRUE(loaded) << "Failed to load FC1 kernel modules";

    // ReLU2 (Nemotron)
    NvFP4MoEContiguousGemmRunner relu2Runner(16, 6, 1856, 2688, 128, Activation::kRelu2);

    // SwiGLU (Qwen3) — note N is doubled for interleaved gate+up
    NvFP4MoEContiguousGemmRunner swigluRunner(128, 8, 1536, 2048, 128, Activation::kSwiglu);

    NvFP4MoEContiguousGemmRunner::unloadKernelModules();
    SUCCEED();
}

// Verify FC1 runners can be constructed with FP16 output dtype for every
// compiled activation. Guards against regressions in AOT variant wiring
// for the fp16 kernel modules.
TEST(NvFP4MoERunnerTest, fc1Fp16Output)
{
    int32_t const smVersion = getSMVersion();
    if (!isSupportedSm(smVersion))
    {
        GTEST_SKIP() << "NvFP4 MoE kernels require SM100/101/110. Current SM=" << smVersion;
    }

    bool const loaded = NvFP4MoEContiguousGemmRunner::loadKernelModules();
    ASSERT_TRUE(loaded) << "Failed to load FC1 kernel modules";

    NvFP4MoEContiguousGemmRunner relu2Fp16Runner(16, 6, 1856, 2688, 128, Activation::kRelu2, OutputDType::kFP16);

    NvFP4MoEContiguousGemmRunner swigluFp16Runner(128, 8, 1536, 2048, 128, Activation::kSwiglu, OutputDType::kFP16);

    NvFP4MoEContiguousGemmRunner::unloadKernelModules();
    SUCCEED();
}

// Verify FC2 runner can be constructed with FP16 output dtype.
TEST(NvFP4MoERunnerTest, fc2Fp16Output)
{
    int32_t const smVersion = getSMVersion();
    if (!isSupportedSm(smVersion))
    {
        GTEST_SKIP() << "NvFP4 MoE kernels require SM100/101/110. Current SM=" << smVersion;
    }

    bool const loaded = NvFP4MoEFC2FinalizeRunner::loadKernelModules();
    ASSERT_TRUE(loaded) << "Failed to load FC2 kernel modules";

    NvFP4MoEFC2FinalizeRunner fc2Fp16Runner(
        /*numLocalExperts=*/16, /*topK=*/6, /*n=*/2688, /*k=*/1856, OutputDType::kFP16);

    NvFP4MoEFC2FinalizeRunner::unloadKernelModules();
    SUCCEED();
}

#endif // CUTE_DSL_NVFP4_MOE_ENABLED
