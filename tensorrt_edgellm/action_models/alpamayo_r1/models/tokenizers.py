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
Adapted from https://github.com/NVlabs/alpamayo/blob/2d2b511ccd626b97aab0c9c211861c0e768551aa/src/alpamayo_r1/models/base_model.py to extract out only the tokenizer initialization logic from ReasoningVLA.
"""

import logging

from transformers import AutoProcessor, AutoTokenizer

from .config import AlpamayoR1Config

logger = logging.getLogger(__name__)

# Constants
IGNORE_INDEX = -100
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}


class AlpamayoR1TokenizerMixin():
    """
    Tokenizer mixin for Alpamayo R1 model.
    """

    def __init__(self, config: AlpamayoR1Config) -> None:
        super().__init__(config)
        self.tokenizer = self._build_tokenizer(config)
        self.vocab_size = len(self.tokenizer)

    def _build_tokenizer(self, config: AlpamayoR1Config) -> AutoTokenizer:
        """Build the tokenizer with trajectory tokens."""
        processor_kwargs = {}
        if config.min_pixels is not None:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels

        processor = AutoProcessor.from_pretrained(config.vlm_name_or_path,
                                                  **processor_kwargs)
        tokenizer = processor.tokenizer

        # Add traj tokens to the tokenizer
        if config.traj_vocab_size is not None:
            discrete_tokens = [
                f"<i{v}>" for v in range(config.traj_vocab_size)
            ]
            num_new_tokens = tokenizer.add_tokens(discrete_tokens)
            assert len(discrete_tokens) == num_new_tokens
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids(
                "<i0>")
            tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(
                f"<i{config.traj_vocab_size - 1}>")

        if config.add_special_tokens:
            special_tokens = list(SPECIAL_TOKENS.values())
            tokenizer.add_tokens(special_tokens, special_tokens=True)
        else:
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()),
                                 special_tokens=True)

        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v)
            for k, v in TRAJ_TOKEN.items()
        }

        return tokenizer
