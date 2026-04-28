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
"""
Qwen3-VL visual model wrapper and export functionality.

This module provides wrapper classes and export functions for Qwen3-family vision encoders
(``qwen3_vl``, ``qwen3_omni``, ``qwen3_5`` vision encoder), enabling ONNX
export with proper attention mechanism handling. Deepstack side outputs are included only when
the checkpoint defines ``deepstack_visual_indexes`` and matching merger modules.

TODO: Input/output names have been aligned with the old multimodal_export.py for compatibility.
      Future refactoring should consider more descriptive names while maintaining backward compatibility.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import \
    apply_rotary_pos_emb_vision

from ..llm_models.layers.attention_plugin import (
    register_attention_plugin_onnx_symbolic_functions, vit_attention_plugin)
from ..onnx_export.onnx_utils import export_onnx


class Qwen3VLVisionAttentionPatch(nn.Module):
    """
    Patched version of Qwen3-VL vision attention for ONNX export.
    Uses vit attention plugin to support ragged attention via cu_seqlens.
    """

    def __init__(self, attention_module: nn.Module) -> None:
        """
        Initialize the patched attention module.
        
        Args:
            attention_module: Original attention module to extract components from
        """
        super().__init__()
        self.qkv = attention_module.qkv
        self.proj = attention_module.proj
        self.num_heads = attention_module.num_heads
        self.head_dim = attention_module.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with custom attention implementation.
        
        Args:
            hidden_states: Input hidden states
            cu_seqlens: Prefix sum of sequence lengths
            max_seqlen_carrier: Shape-only input carrying max sequence length for FMHA launch
            position_embeddings: Position embeddings for rotary attention

        Returns:
            Attention output
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                  self.num_heads,
                                                  -1).permute(1, 0, 2,
                                                              3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Convert to FP16 for plugin compatibility
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        # Use ViT attention plugin with separate Q, K, V
        # q, k, v are already in shape [total_S, H, D]
        attn_output = vit_attention_plugin(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen_carrier,
            num_heads=self.num_heads,
            head_size=self.head_dim,
        )

        # Plugin output layout is [total_S, H, D], reshape to [total_S, H * D]
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionModelPatch(nn.Module):
    """
    Patched vision tower for ONNX export.

    Wraps Hugging Face Qwen3-family vision modules (e.g. ``Qwen3VLVisionModel``,
    ``Qwen3OmniVisionEncoder``, ``Qwen3_5VisionModel``) while reusing weights and replacing
    attention with :class:`Qwen3VLVisionAttentionPatch`.
    """

    def __init__(self, original_model: nn.Module) -> None:
        """
        Initialize the patched vision transformer from original model.
        
        Args:
            original_model: Loaded Qwen3-family vision encoder (see module docstring).
        """
        super().__init__()
        self.config = original_model.config
        self.num_grid_per_side = original_model.num_grid_per_side

        # Reuse all original components
        self.patch_embed = original_model.patch_embed
        self.pos_embed = original_model.pos_embed
        self.blocks = original_model.blocks
        self.merger = original_model.merger

        self.deepstack_visual_indexes = list(
            getattr(original_model, "deepstack_visual_indexes", []))
        if self.deepstack_visual_indexes:
            if hasattr(original_model, "deepstack_merger_list"):
                self.deepstack_merger_list = original_model.deepstack_merger_list
            elif hasattr(original_model, "merger_list"):
                # qwen3_omni exposes deepstack mergers as merger_list.
                self.deepstack_merger_list = original_model.merger_list
            else:
                raise ValueError(
                    "Deepstack visual indexes exist but no deepstack merger list was found."
                )

        # Replace attention modules, reusing existing components to preserve quantization
        for block in self.blocks:
            block.attn = Qwen3VLVisionAttentionPatch(block.attn)

    @property
    def device(self):
        return next(self.parameters()).device

    def fast_pos_embed_interpolate_optimized(self, grid_thw):
        """
        Optimized version of `fast_pos_embed_interpolate` in Qwen3VLVisionModel.
        The original `fast_pos_embed_interpolate` workflow permutes after embedding,
            which is inefficient and hard to implement in TensorRT.
        We permute the index and weight tensor first during initialization and take them as model inputs.
        """
        grid_ts, grid_hs, grid_ws = \
            grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side -
                                                  1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side -
                                                  1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # Permute indices and weights first so no need to permute after embedding.
            # From [h, w] to [h // merge_size, w // merge_size, merge_size, merge_size]
            merge_size = self.config.spatial_merge_size
            merged_h, merged_w = h // merge_size, w // merge_size

            indices = [
                (base_h.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h_ceil.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
                (base_h_ceil.reshape(merged_h, 1, merge_size, 1) +
                 w_idxs_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
            ]

            weights = [
                ((1 - dh).reshape(merged_h, 1, merge_size, 1) *
                 (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
                ((1 - dh).reshape(merged_h, 1, merge_size, 1) *
                 dw.reshape(1, merged_w, 1, merge_size)).flatten(),
                (dh.reshape(merged_h, 1, merge_size, 1) *
                 (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
                (dh.reshape(merged_h, 1, merge_size, 1) *
                 dw.reshape(1, merged_w, 1, merge_size)).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list,
                                  dtype=torch.long,
                                  device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list,
                                     dtype=self.pos_embed.weight.dtype,
                                     device=self.pos_embed.weight.device)
        return idx_tensor, weight_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,
        fast_pos_embed_weight: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the vision transformer.
        
        Args:
            hidden_states: Input hidden states [seq_len, input_dim]
            rotary_pos_emb: Rotary position embeddings [seq_len, rotary_pos_emb_dim]
            cu_seqlens: Prefix sum of sequence lengths
            max_seqlen_carrier: Shape-only input carrying max sequence length for FMHA launch
            fast_pos_embed_idx: Fast position index tensor [4, seq_len]
            fast_pos_embed_weight: Fast position weight tensor [4, seq_len]

        Returns:
            `torch.Tensor`: hidden_states.
            (Optional) list of `torch.Tensor`: deepstack_feature_lists.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.pos_embed(
            fast_pos_embed_idx) * fast_pos_embed_weight[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[
            2] + pos_embeds[3]
        # No need to permute after embedding.
        hidden_states = hidden_states + patch_pos_embeds

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen_carrier=max_seqlen_carrier,
                position_embeddings=position_embeddings,
            )
            if self.deepstack_visual_indexes and layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)](
                        hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        if not self.deepstack_visual_indexes:
            return hidden_states
        return hidden_states, deepstack_feature_lists


def export_qwen3_vl_visual(
    model: Qwen3VLVisionModelPatch,
    output_dir: str,
    torch_dtype: torch.dtype,
) -> None:
    """
    Export Qwen3-family visual model to ONNX format.
    
    This function takes a patched vision encoder, prepares dummy inputs for ONNX export,
    and saves the model in ONNX format.
    
    Args:
        model: Patched Qwen3-family vision transformer model
        output_dir: Directory to save the exported ONNX model
        torch_dtype: PyTorch data type for the model
    """
    num_deepstack = len(model.deepstack_visual_indexes)

    # Prepare dummy inputs for ONNX export
    hw = 16  # Height * width for the input
    in_chans = model.config.in_channels
    temporal_patch_size = model.config.temporal_patch_size
    patch_size = model.config.patch_size
    rotary_pos_emb_dim = model.config.hidden_size // model.config.num_heads // 2

    # Create input tensors with appropriate shapes and dtypes
    pixel_values = torch.randn(
        (hw, in_chans * temporal_patch_size * patch_size * patch_size),
        dtype=torch_dtype,
        device=model.device)
    rotary_pos_emb = torch.randn(
        (hw, rotary_pos_emb_dim),
        dtype=torch.float32,  # Keep as float32 for rotary embeddings
        device=model.device)
    cu_seqlens = torch.tensor([0, hw], dtype=torch.int32, device=model.device)
    max_seqlen_carrier = torch.zeros(hw,
                                     dtype=torch.int32,
                                     device=model.device)
    fast_pos_embed_idx = torch.arange(hw,
                                      dtype=torch.int64,
                                      device=model.device).unsqueeze(0).repeat(
                                          4, 1)
    fast_pos_embed_weight = torch.randn((4, hw),
                                        dtype=torch_dtype,
                                        device=model.device)

    inputs = (pixel_values, rotary_pos_emb, cu_seqlens, max_seqlen_carrier,
              fast_pos_embed_idx, fast_pos_embed_weight)

    input_names = [
        "input", "rotary_pos_emb", "cu_seqlens", "max_seqlen_carrier",
        "fast_pos_embed_idx", "fast_pos_embed_weight"
    ]
    output_names = ["output"] + [
        f"deepstack_features_{i}" for i in range(num_deepstack)
    ]

    # Define dynamic axes for variable input sizes
    dynamic_axes = {
        # Model inputs
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'cu_seqlens': {
            0: 'batch_size + 1'
        },
        'max_seqlen_carrier': {
            0: 'max_seqlen'
        },
        'fast_pos_embed_idx': {
            1: 'hw'
        },
        'fast_pos_embed_weight': {
            1: 'hw'
        },
        # Model outputs
        'output': {
            0: 'image_token_len'
        },
    }
    for i in range(num_deepstack):
        dynamic_axes[f"deepstack_features_{i}"] = {0: 'image_token_len'}

    register_attention_plugin_onnx_symbolic_functions()
    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
