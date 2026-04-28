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
"""FP8 embedding quantization utilities.

This module provides functions to quantize embedding tables to FP8 format
with per-row-per-block scaling for reduced memory bandwidth during inference.

Key design:
- Block size: 128 (fixed, matches C++ kernel granularity)
- Scaling: Per-row, per-block max-abs scaling
- Format: FP8 E4M3
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

from tensorrt_edgellm.quantization import FP8_E4M3_MAX

# Fixed block size matching C++ kernel (kFP8EmbeddingBlockSize)
FP8_EMBEDDING_BLOCK_SIZE = 128


def quantize_embedding_to_fp8(
    embedding_weight: torch.Tensor,
    block_size: int = FP8_EMBEDDING_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize embedding table to FP8 with per-row-per-block scales.

    Args:
        embedding_weight: FP16/FP32 embedding table [vocab_size, hidden_size]
        block_size: Number of elements per scale group (default: 128)

    Returns:
        Tuple of:
            - embedding_fp8: FP8 E4M3 quantized embedding [vocab_size, hidden_size]
            - scales: FP32 per-group scales [vocab_size, hidden_size // block_size]

    Raises:
        ValueError: If hidden_size is not divisible by block_size
    """
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")

    vocab_size, hidden_size = embedding_weight.shape

    if hidden_size % block_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by block size {block_size}"
        )

    num_groups = hidden_size // block_size

    # Convert to FP32 for quantization computation
    weight_fp32 = embedding_weight.float()

    # Reshape to [vocab_size, num_groups, block_size] for per-block processing
    weight_reshaped = weight_fp32.view(vocab_size, num_groups, block_size)

    # Compute max abs per block: [vocab_size, num_groups]
    amax = weight_reshaped.abs().amax(dim=-1)

    # Avoid division by zero
    amax = amax.clamp(min=1e-4)

    # Compute scales: scale = amax / FP8_E4M3_MAX
    scales = amax / FP8_E4M3_MAX

    # Quantize: quantized = weight / scale
    # Expand scales for broadcasting: [vocab_size, num_groups, 1]
    scales_expanded = scales.unsqueeze(-1)
    quantized = weight_reshaped / scales_expanded

    # Clamp to FP8 E4M3 range and convert
    quantized = quantized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Reshape back to [vocab_size, hidden_size]
    quantized = quantized.view(vocab_size, hidden_size)

    # Convert to FP8 E4M3
    embedding_fp8 = quantized.to(torch.float8_e4m3fn)

    logger.info(f"Quantized embedding to FP8: [{vocab_size}, {hidden_size}], "
                f"scales: [{vocab_size}, {num_groups}]")

    return embedding_fp8, scales
