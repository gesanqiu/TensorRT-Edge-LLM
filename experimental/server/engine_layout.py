# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standard directory layouts for TensorRT Edge-LLM.

Defines the file naming conventions expected by the C++ runtime.
Used by ``LLM`` (HLAPI), CI tests, and documentation.

ONNX directory (input to engine build)::

    {onnx_dir}/
        model.onnx
        config.json
        tokenizer.json / tokenizer_config.json
        embedding.safetensors

LLM engine directory::

    {engine_dir}/
        llm.engine
        config.json
        tokenizer.json / tokenizer_config.json
        processed_chat_template.json
        embedding.safetensors

VLM (LLM + Visual encoder)::

    {llm_engine_dir}/              # Same as LLM above
    {visual_engine_dir}/
        visual.engine
        config.json

EAGLE speculative decoding::

    {eagle_engine_dir}/
        eagle_base.engine / eagle_draft.engine
        base_config.json / draft_config.json
        d2t.safetensors
        tokenizer.json / tokenizer_config.json
        embedding.safetensors
"""

import enum
import os
from typing import Optional


class EngineType(enum.Enum):
    """Detected engine directory type."""

    LLM = "llm"
    VLM = "vlm"
    EAGLE = "eagle"
    UNKNOWN = "unknown"


ONNX_MODEL_FILE = "model.onnx"
LLM_ENGINE_FILE = "llm.engine"
VISUAL_ENGINE_FILE = "visual.engine"
EAGLE_BASE_ENGINE_FILE = "eagle_base.engine"
EAGLE_DRAFT_ENGINE_FILE = "eagle_draft.engine"
CONFIG_FILE = "config.json"


def validate_onnx_dir(onnx_dir: str) -> bool:
    """Check that a directory contains an ONNX model."""
    return os.path.isfile(os.path.join(onnx_dir, ONNX_MODEL_FILE))


def detect_engine_type(engine_dir: str) -> EngineType:
    """Detect the engine type from directory contents."""
    if not os.path.isdir(engine_dir):
        return EngineType.UNKNOWN
    if os.path.isfile(os.path.join(engine_dir, EAGLE_BASE_ENGINE_FILE)):
        return EngineType.EAGLE
    if os.path.isfile(os.path.join(engine_dir, LLM_ENGINE_FILE)):
        return EngineType.LLM
    if os.path.isfile(os.path.join(engine_dir, VISUAL_ENGINE_FILE)):
        return EngineType.VLM
    return EngineType.UNKNOWN


def validate_llm_engine_dir(engine_dir: str) -> bool:
    """Check that an LLM engine directory has the required files."""
    return os.path.isfile(os.path.join(engine_dir, LLM_ENGINE_FILE))


def validate_visual_engine_dir(engine_dir: str) -> bool:
    """Check that a visual engine directory has the required files.

    Matches the C++ runtime convention: checks ``visual/visual.engine``
    first, then ``visual.engine`` at root as a fallback.
    """
    return (os.path.isfile(
        os.path.join(engine_dir, "visual", VISUAL_ENGINE_FILE))
            or os.path.isfile(os.path.join(engine_dir, VISUAL_ENGINE_FILE)))


def validate_eagle_engine_dir(engine_dir: str) -> bool:
    """Check that an EAGLE engine directory has both engines."""
    return (os.path.isfile(os.path.join(engine_dir, EAGLE_BASE_ENGINE_FILE))
            and os.path.isfile(
                os.path.join(engine_dir, EAGLE_DRAFT_ENGINE_FILE)))


def find_visual_engine_dir(
    llm_engine_dir: str,
    model_name: str = "",
) -> Optional[str]:
    """Auto-detect a sibling visual engine directory.

    Searches for a sibling of the LLM engine directory that contains
    ``visual/visual.engine`` or ``visual.engine``.
    """
    parent = os.path.dirname(llm_engine_dir)
    if not os.path.isdir(parent):
        return None
    for entry in os.listdir(parent):
        candidate = os.path.join(parent, entry)
        if candidate == llm_engine_dir:
            continue
        if validate_visual_engine_dir(candidate):
            return candidate
    return None
