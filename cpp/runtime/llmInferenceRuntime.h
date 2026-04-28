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

// Compatibility header for in-flight branches (Qwen3-VL, Mamba, LLM Loader).
// LLMInferenceRuntime has been unified into LLMInferenceSpecDecodeRuntime.
// This typedef allows existing code to compile without immediate changes.
// Remove in Phase 2 when the class is renamed to LLMRuntime.

#include "runtime/llmInferenceSpecDecodeRuntime.h"

namespace trt_edgellm
{
namespace rt
{

//! @brief Compatibility typedef — LLMInferenceRuntime is now LLMInferenceSpecDecodeRuntime.
//! The unified runtime supports both vanilla (no draft model) and Eagle spec-decode modes.
//! Construct without EagleDraftingConfig for vanilla-only behavior identical to the old LLMInferenceRuntime.
using LLMInferenceRuntime = LLMInferenceSpecDecodeRuntime;
} // namespace rt
} // namespace trt_edgellm
