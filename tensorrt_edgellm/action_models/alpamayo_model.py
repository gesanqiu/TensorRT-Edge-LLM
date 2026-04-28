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
Alpamayo 1 action expert model wrapper and ONNX export.

This module provides patched Qwen3-VL text model components (attention, decoder layer,
text model) for the Alpamayo 1 action expert, plus a wrapper for one flow-matching
denoising step and ONNX export.
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention, Qwen3VLTextDecoderLayer, Qwen3VLTextModel)

from ..llm_models.layers.attention_trt import (
    EdgeLLMAttentionTRTNative,
    register_trt_native_attention_onnx_symbolic_functions)
from ..onnx_export.onnx_utils import export_onnx
from .alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class Alpamayo1ExpertDecoderLayerPatch(Qwen3VLTextDecoderLayer):
    """
    Patched Qwen3-VL text decoder layer for Alpamayo 1 action expert.

    Uses EdgeLLMAttentionTRTNative (initialized from a Qwen3VLTextAttention)
    with non-causal attention via a mask of all zeros.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        source_attention: Qwen3VLTextAttention,
    ) -> None:
        super().__init__(config, layer_idx)
        self.self_attn = EdgeLLMAttentionTRTNative(source_attention,
                                                   eagle3_draft=False)
        self.source_attention = source_attention
        self.source_attention.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_indexing_position_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        past_seq_length = kvcache_start_index[0]
        non_causal_mask = torch.full(
            (1, 1, q_len, q_len + past_seq_length),
            0.0,
            device=hidden_states.device,
            dtype=torch.float16,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_k_cache, present_v_cache = self.self_attn(
            hidden_states=hidden_states,
            k_cache=k_cache,
            v_cache=v_cache,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            attention_mask=non_causal_mask,
            position_ids=rope_indexing_position_ids,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_k_cache, present_v_cache


class Alpamayo1ExpertTextModelPatch(Qwen3VLTextModel):
    """
    Patched Qwen3-VL text model for Alpamayo 1 action expert ONNX export.
    """

    def __init__(
        self,
        config: Any,
        source_expert: Qwen3VLTextModel,
        n_diffusion_tokens: int,
    ) -> None:
        super().__init__(config)

        self.layers = nn.ModuleList([
            Alpamayo1ExpertDecoderLayerPatch(
                config,
                layer_idx,
                source_attention=source_expert.layers[layer_idx].self_attn,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        if "embed_tokens" in self._modules:
            del self._modules["embed_tokens"]
        self.n_diffusion_tokens = n_diffusion_tokens

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        kvcache_start_index: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        rope_indexing_position_ids: torch.Tensor,
        k_caches: Tuple[torch.Tensor, ...],
        v_caches: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor,
                                                             ...]]:
        """
        Forward with inputs_embeds and separate k_caches/v_caches.
        Returns (hidden_states, present_k_caches, present_v_caches).
        """
        hidden_states = inputs_embeds

        context_lengths = torch.tensor(
            [self.n_diffusion_tokens],
            dtype=torch.int32,
            device=k_caches[0].device,
        ).expand(1)

        present_k_caches: List[torch.Tensor] = []
        present_v_caches: List[torch.Tensor] = []
        for decoder_layer, k_cache, v_cache in zip(self.layers, k_caches,
                                                   v_caches):
            hidden_states, present_k, present_v = decoder_layer(
                hidden_states=hidden_states,
                k_cache=k_cache,
                v_cache=v_cache,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                rope_indexing_position_ids=rope_indexing_position_ids,
                kvcache_start_index=kvcache_start_index,
            )
            present_k_caches.append(present_k)
            present_v_caches.append(present_v)

        hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(present_k_caches), tuple(present_v_caches)


class Alpamayo1ActionExpertPatch(nn.Module):
    """
    Wraps the Alpamayo1 action expert and one flow-matching denoising step for ONNX export.
    Initialized directly from a loaded AlpamayoR1 model.
    """

    def __init__(
        self,
        model: AlpamayoR1,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Build from a full AlpamayoR1 model (e.g. from load_hf_model)."""
        super().__init__()
        self.diffusion = model.diffusion
        self.action_space_dims = model.action_space.get_action_space_dims()
        self.n_diffusion_tokens = self.action_space_dims[0]
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        patched_expert = Alpamayo1ExpertTextModelPatch(
            model.expert.config,
            source_expert=model.expert,
            n_diffusion_tokens=self.n_diffusion_tokens)
        patched_expert.load_state_dict(model.expert.state_dict(), strict=False)
        patched_expert.eval().to(device).to(torch_dtype)
        self.expert = patched_expert

    def _step_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        rope_indexing_position_ids: torch.Tensor,
        k_caches: Tuple[torch.Tensor, ...],
        v_caches: Tuple[torch.Tensor, ...],
        kvcache_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor,
                                                             ...]]:
        """Single flow-matching step: action_in_proj -> expert -> action_out_proj."""
        b_star = x.shape[0]
        future_token_embeds = self.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(
                b_star, self.n_diffusion_tokens, -1)
        last_hidden, present_k_caches, present_v_caches = self.expert(
            future_token_embeds,
            kvcache_start_index,
            rope_rotary_cos_sin,
            rope_indexing_position_ids,
            k_caches,
            v_caches,
        )
        last_hidden = last_hidden[:, -self.n_diffusion_tokens:]
        pred = self.action_out_proj(last_hidden).view(-1,
                                                      *self.action_space_dims)
        return pred, present_k_caches, present_v_caches

    def forward(
        self,
        noise_trajectory: torch.Tensor,
        time_steps_t0: torch.Tensor,
        time_steps_t1: torch.Tensor,
        seq_length: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        rope_indexing_position_ids: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor,
                                                             ...]]:
        n_layers = self.expert.config.num_hidden_layers
        assert len(cache_tensors) == 2 * n_layers, (
            f"Expected 2*{n_layers} cache tensors, got {len(cache_tensors)}")
        k_caches = tuple(cache_tensors[:n_layers])
        v_caches = tuple(cache_tensors[n_layers:])
        batch_size = 1
        noise_trajectory = noise_trajectory.to(torch.float16)
        time_steps_t0 = time_steps_t0.to(torch.float16)
        time_steps_t1 = time_steps_t1.to(torch.float16)
        n_dim = len(self.diffusion.x_dims)
        dt = time_steps_t1 - time_steps_t0
        dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
        t_start = time_steps_t0.view(1, *[1] * n_dim).expand(
            batch_size, *[1] * n_dim)

        v, present_k_caches, present_v_caches = self._step_fn(
            x=noise_trajectory,
            t=t_start,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            rope_indexing_position_ids=rope_indexing_position_ids,
            k_caches=k_caches,
            v_caches=v_caches,
            kvcache_start_index=seq_length,
        )
        denoised = (noise_trajectory + dt * v).to(torch.float32)
        return (denoised, *present_k_caches, *present_v_caches)


def export_alpamayo1_action(
    model: Alpamayo1ActionExpertPatch,
    output_dir: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> None:
    """
    Export the Alpamayo 1 action expert (one flow-matching step) to ONNX.

    Args:
        model: Alpamayo1ActionExpertPatch.
        output_dir: Directory to save model.onnx.
        device: Device for dummy tensors and export.
        torch_dtype: Export dtype (fp16).
    """

    n_diffusion_tokens = model.n_diffusion_tokens
    max_kv_cache_capacity = 3424  # TRT KVCacheUpdate layer requires a static value for capacity
    dummy_seq_length = 3027
    inference_step = 10  # from https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/diffusion/flow_matching.py

    expert_config = model.expert.config
    n_layers = expert_config.num_hidden_layers
    n_heads = expert_config.num_key_value_heads
    n_head_dims = getattr(expert_config, "head_dim", 128)

    batch_size = 1

    x = torch.randn(
        batch_size,
        *model.diffusion.x_dims,
        device=device,
        dtype=torch_dtype,
    ).to(torch.float32)

    time_steps = torch.linspace(0.0, 1.0, inference_step + 1,
                                device=device).to(torch.float32)

    seq_length = torch.full((batch_size, ),
                            dummy_seq_length,
                            device=device,
                            dtype=torch.int64)

    cache_shape = (batch_size, n_heads, max_kv_cache_capacity, n_head_dims)
    k_caches = tuple(
        torch.randn(cache_shape, device=device, dtype=torch_dtype)
        for _ in range(n_layers))
    v_caches = tuple(
        torch.randn(cache_shape, device=device, dtype=torch_dtype)
        for _ in range(n_layers))

    # Only 64 waypoints; rope cache is (1, n_diffusion_tokens, head_dim).
    rope_rotary_cos_sin = torch.randn(
        batch_size,
        n_diffusion_tokens,
        n_head_dims,
        device=device,
        dtype=torch.float32,
    )
    rope_indexing_position_ids = torch.arange(n_diffusion_tokens,
                                              device=device,
                                              dtype=torch.int32).unsqueeze(0)

    inputs = [
        x,
        time_steps[[0]],
        time_steps[[1]],
        seq_length,
        rope_rotary_cos_sin,
        rope_indexing_position_ids,
    ] + list(k_caches) + list(v_caches)

    input_names = ([
        "noise_trajectory",
        "time_steps_t0",
        "time_steps_t1",
        "seq_length",
        "rope_rotary_cos_sin",
        "rope_indexing_position_ids",
    ] + [f"k_cache_{i}"
         for i in range(n_layers)] + [f"v_cache_{i}" for i in range(n_layers)])
    output_names = (["denoised_trajectory"] +
                    [f"present_k_cache_{i}" for i in range(n_layers)] +
                    [f"present_v_cache_{i}" for i in range(n_layers)])

    register_trt_native_attention_onnx_symbolic_functions()
    model.eval()
    export_onnx(
        model,
        tuple(inputs),
        output_dir,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
    )
