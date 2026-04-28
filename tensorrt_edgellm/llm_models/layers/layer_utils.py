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
"""Utility layers for attention mechanisms in TensorRT Edge LLM models.

Provides Q/K/V projection and normalization layers that adapt to different model architectures.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class EdgeLLMQKVProj(nn.Module):
    """Q/K/V projection layer supporting both fused and separate projections.
    
    Automatically adapts to use either separate (q_proj, k_proj, v_proj) or 
    fused (qkv_proj) projections based on the attention module.
    
    Args:
        attention_module: Attention module with either separate or fused QKV projections.
    """

    def __init__(self, attention_module: nn.Module, eagle3_draft: bool):
        super().__init__()
        num_attention_heads: int = attention_module.config.num_attention_heads
        num_key_value_heads: int = attention_module.config.num_key_value_heads
        if hasattr(attention_module.config, 'head_dim'):
            self.head_dim: int = attention_module.config.head_dim
        else:
            self.head_dim: int = attention_module.config.hidden_size // num_attention_heads

        # Copy projection layers from original attention module
        # Phi4MM uses a fused qkv_proj; we support both split and fused Q/K/V paths for compatibility.
        self.has_qwen3_5_gate = False
        self.q_dim = num_attention_heads * self.head_dim
        if hasattr(attention_module, 'q_proj'):
            assert hasattr(attention_module, 'k_proj') and hasattr(attention_module, 'v_proj'), \
                "q_proj, k_proj, and v_proj must be present"
            self.fused_qkv_proj = False
            self.q_proj = attention_module.q_proj
            self.k_proj = attention_module.k_proj
            self.v_proj = attention_module.v_proj
            # Qwen3.5 full attention packs (query, gate) into q_proj output.
            if (getattr(attention_module.config, 'model_type',
                        '') == 'qwen3_5_text'):
                assert attention_module.q_proj.out_features == 2 * self.q_dim, \
                    "Qwen3.5 full attention q_proj output dimension must be 2 * q_dim"
                self.has_qwen3_5_gate = True
        elif hasattr(attention_module, 'qkv_proj'):
            self.fused_qkv_proj = True
            self.kv_dim = num_key_value_heads * self.head_dim
            self.qkv_proj = attention_module.qkv_proj
        else:
            assert False

        # Eagle3 draft: double the input dimension for the attention module
        if eagle3_draft:
            assert hasattr(attention_module, 'q_proj') and hasattr(attention_module, 'k_proj') and hasattr(attention_module, 'v_proj'), \
                "q_proj, k_proj, and v_proj must be present"
            self.q_proj = nn.Linear(attention_module.q_proj.in_features * 2,
                                    attention_module.q_proj.out_features,
                                    bias=attention_module.q_proj.bias
                                    is not None)
            self.k_proj = nn.Linear(attention_module.k_proj.in_features * 2,
                                    attention_module.k_proj.out_features,
                                    bias=attention_module.k_proj.bias
                                    is not None)
            self.v_proj = nn.Linear(attention_module.v_proj.in_features * 2,
                                    attention_module.v_proj.out_features,
                                    bias=attention_module.v_proj.bias
                                    is not None)

    def forward(
        self, hidden_states: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        """Apply Q, K, V projections to hidden states.
        
        Args:
            hidden_states: Input hidden states.
        
        Returns:
            Tuple of (query_states, key_states, value_states, gate_states).
        """
        gate_states: Optional[torch.Tensor] = None
        # Apply Q, K, V projections
        if self.fused_qkv_proj:
            # Fused qkv_proj path (for Phi4MM)
            qkv_out = self.qkv_proj(hidden_states)
            query_states = qkv_out[..., :self.q_dim]
            key_states = qkv_out[..., self.q_dim:self.q_dim + self.kv_dim]
            value_states = qkv_out[..., self.q_dim + self.kv_dim:]
        else:
            # Separate q/k/v projections path
            query_states = self.q_proj(hidden_states)
            if self.has_qwen3_5_gate:
                input_shape = hidden_states.shape[:-1]
                query_states, gate_states = torch.chunk(query_states.view(
                    *input_shape, -1, self.head_dim * 2),
                                                        2,
                                                        dim=-1)
                query_states = query_states.reshape(*input_shape, -1)
                gate_states = gate_states.reshape(*input_shape, -1)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states, gate_states


class EdgeLLMQKNorm(nn.Module):
    """Query and key normalization layer for multiple model architectures.
    
    Supports separate normalization (Qwen3) or shared normalization (Llama4).
    
    Args:
        attention_module: Attention module with optional q_norm, k_norm, or qk_norm.
    """

    def __init__(self, attention_module: nn.Module):
        super().__init__()
        # Qwen3 models have individual Q and K normalization layers
        self.q_norm = getattr(attention_module, 'q_norm', None)
        self.k_norm = getattr(attention_module, 'k_norm', None)

        # Llama4 models have shared QK normalization layer
        self.qk_norm = getattr(attention_module, 'qk_norm', None)

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor,
                norm_shape):
        """Apply normalization to query and key states.
        
        Args:
            query_states: Query states to normalize.
            key_states: Key states to normalize.
            norm_shape: Shape for normalization (states are reshaped then restored).
        
        Returns:
            Tuple of (normalized query_states, normalized key_states).
        """
        q_shape = query_states.shape
        k_shape = key_states.shape
        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.view(norm_shape)).contiguous().view(q_shape)
        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.view(norm_shape)).contiguous().view(k_shape)

        if self.qk_norm is not None:
            query_states = self.qk_norm(
                query_states.view(norm_shape)).contiguous().view(q_shape)
            key_states = self.qk_norm(
                key_states.view(norm_shape)).contiguous().view(k_shape)

        return query_states, key_states
