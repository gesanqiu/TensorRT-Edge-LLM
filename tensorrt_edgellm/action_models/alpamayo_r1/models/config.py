# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Adapted from https://github.com/NVlabs/alpamayo/blob/2d2b511ccd626b97aab0c9c211861c0e768551aa/src/alpamayo_r1/models/base_model.py to extract out only the model configuration from ReasoningVLA.
"""

from typing import Any

from transformers import PretrainedConfig


class AlpamayoR1Config(PretrainedConfig):
    """
    Configuration for the Alpamayo R1 release model.
    """

    model_type = "alpamayo_r1"

    def __init__(
        self,
        vlm_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        vlm_backend: str = "qwenvl3",
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_vocab_size: int = 768,
        tokens_per_history_traj: int = 16,
        tokens_per_future_traj: int = 64,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_special_tokens: bool = False,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend.lower()
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation
        self.traj_tokenizer_cfg = traj_tokenizer_cfg
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_special_tokens = add_special_tokens
        self.diffusion_cfg = diffusion_cfg
        self.action_space_cfg = action_space_cfg
        self.action_in_proj_cfg = action_in_proj_cfg
        self.action_out_proj_cfg = action_out_proj_cfg
        self.expert_cfg = expert_cfg
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention
