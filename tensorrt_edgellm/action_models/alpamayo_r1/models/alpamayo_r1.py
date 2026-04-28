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
Adapted from https://github.com/NVlabs/alpamayo/blob/2d2b511ccd626b97aab0c9c211861c0e768551aa/src/alpamayo_r1/models/alpamayo_r1.py

Extracts out the action expert logic from AlpamayoR1 to its own AlpamayoR1ActionExpert class, and initializes the VLM from Qwen3-VL directly.
"""

import copy
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoModelForImageTextToText,
                          PreTrainedModel)

from ..action_space.unicycle_accel_curvature import \
    UnicycleAccelCurvatureActionSpace
from ..diffusion.flow_matching import FlowMatching
from .action_in_proj import PerWaypointActionInProjV2
from .config import AlpamayoR1Config
from .tokenizers import AlpamayoR1TokenizerMixin

logger = logging.getLogger(__name__)


def kwargs_from_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Build kwargs from a config dict, omitting Hydra private attributes."""
    if cfg is None:
        return {}
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


class AlpamayoR1ActionExpert(AlpamayoR1TokenizerMixin, PreTrainedModel):
    """
    Expert model for reasoning VLA.

    Adapted from https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/models/alpamayo_r1.py
    to extract out only the action expert.
    """

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1Config,
    ):
        super().__init__(config)

        self.expert = AutoModel.from_config(config.expert_cfg)
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens

        action_space_kw = kwargs_from_cfg(config.action_space_cfg)
        self.action_space = UnicycleAccelCurvatureActionSpace(
            **action_space_kw)

        diffusion_kw = kwargs_from_cfg(config.diffusion_cfg)
        diffusion_kw["x_dims"] = self.action_space.get_action_space_dims()
        self.diffusion = FlowMatching(**diffusion_kw)

        in_proj_kw = kwargs_from_cfg(config.action_in_proj_cfg)
        in_proj_kw["in_dims"] = self.action_space.get_action_space_dims()
        in_proj_kw["out_dim"] = config.expert_cfg.hidden_size
        self.action_in_proj = PerWaypointActionInProjV2(**in_proj_kw)

        out_proj_kw = kwargs_from_cfg(config.action_out_proj_cfg)
        out_proj_kw["in_features"] = config.expert_cfg.hidden_size
        out_proj_kw["out_features"] = self.action_space.get_action_space_dims(
        )[-1]
        self.action_out_proj = nn.Linear(**out_proj_kw)

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()


class AlpamayoR1(AlpamayoR1ActionExpert):
    """
    Alpamayo R1 model, including the action expert and VLM.
    """

    def __init__(self,
                 config: AlpamayoR1Config,
                 torch_dtype: Optional[torch.dtype] = None):
        vlm_config = AutoConfig.from_pretrained(
            config.vlm_name_or_path,
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
        )
        vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.vocab_size = config.vocab_size
        vlm = AutoModelForImageTextToText.from_config(
            vlm_config,
            dtype=torch_dtype,
        )

        expert_config = copy.deepcopy(vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        config.expert_cfg = expert_config

        super().__init__(config)
        self.vlm = vlm
