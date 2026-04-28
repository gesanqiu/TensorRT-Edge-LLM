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
Patches the Nemotron-H model to allow it to load and export without mamba_ssm CUDA extensions.
Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/a1964bcbbcbbe1d6f4e0750ec5ff4d58ca7e81fb/tensorrt_llm/_torch/auto_deploy/models/patches/nemotron_h.py
"""
import contextlib
import importlib.util
import sys
import types
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM

from ...onnx_export.config_export import _patch_nemotron_h_config


# Forked from:
# https://github.com/state-spaces/mamba/blob/6b32be06d026e170b3fdaf3ae6282c5a6ff57b06/mamba_ssm/ops/triton/layernorm_gated.py
# NOTES:
# 1. At time of writing (09/25/2025), the nano nemotron v2 modeling code expects `mamba_ssm`
#    to be installed so as to be able to make use of its grouped gated RMS norm operation.
#    We therefore replace it with one that uses einops + pytorch.
def _rms_norm_ref(x,
                  weight,
                  bias,
                  z=None,
                  eps=1e-6,
                  group_size=None,
                  norm_before_gate=True,
                  upcast=True):
    dtype = x.dtype
    # N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd *
                                                                   weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) +
                              eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# The original implementation looks at `cache_position[0]` to decide what to do which does not
# play well with export. Plus, we do not want it to be updated anyway.
def _nemotron_h_model_update_mamba_mask(self, attention_mask, cache_position):
    return None


def _nemotron_h_model_update_causal_mask(self, attention_mask, input_tensor,
                                         cache_position):
    # Force attention to use causal mode without explicit masks
    return None


def _nemotron_h_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    device = hidden_states.device
    with contextlib.ExitStack() as stack:
        if device.type == "cuda":
            stack.enter_context(
                torch.cuda.stream(torch.cuda.default_stream(device)))
        # * Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
        residual = hidden_states
        hidden_states = self.norm(
            hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states,
                                       cache_params=cache_params,
                                       cache_position=cache_position)
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states,
                                       cache_position=cache_position)
            hidden_states = hidden_states[0]
        elif self.block_type in ["mlp", "moe"]:
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        hidden_states = residual + hidden_states
        return hidden_states


def _nemotron_h_moe_dense(self, hidden_states: torch.Tensor,
                          topk_indices: torch.Tensor,
                          topk_weights: torch.Tensor) -> torch.Tensor:
    """Dense MoE computation safe for ONNX export.

    Replaces the original sparse routing (``torch.where`` + ``index_add_``)
    with a fully-traceable dense loop: every expert runs on every token, and
    contributions are gated by the soft routing weights.  Numerically
    equivalent to the sparse version; slower but export-correct.

    Avoids:
    - ``if token_indices.numel() > 0`` (TracerWarning – Python bool from tensor)
    - ``hidden_states[token_indices]`` (``aten::index`` duplicate-values warning)
    - ``index_add_`` (in-place scatter, breaks ONNX graph)
    """
    n_experts = len(self.experts)
    orig_dtype = hidden_states.dtype
    acc = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

    for expert_idx in range(n_experts):
        expert = self.experts[expert_idx]
        # Weight this expert contributes to each token: sum over top-k slots
        # where that slot selected expert_idx.
        # topk_indices: [n_tokens, top_k]  topk_weights: [n_tokens, top_k]
        expert_weight = ((topk_indices == expert_idx).to(topk_weights.dtype) *
                         topk_weights).sum(dim=-1,
                                           keepdim=True)  # [n_tokens, 1]

        # Run expert on ALL tokens (dense) then mask by routing weight
        expert_out = expert(hidden_states.to(orig_dtype))  # [n_tokens, hidden]
        acc = acc + expert_out.to(topk_weights.dtype) * expert_weight

    return acc.to(orig_dtype)


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, List[Tuple[str, Callable]]] = {
    "NemotronHModel": [
        ("_update_causal_mask", _nemotron_h_model_update_causal_mask),
        ("_update_mamba_mask", _nemotron_h_model_update_mamba_mask),
    ],
    "NemotronHBlock": [("forward", _nemotron_h_block_forward)],
    "NemotronHMOE": [("moe", _nemotron_h_moe_dense)],
}


def get_model_from_config_patched(config, **kwargs):
    # Patch NemotronHConfig before model creation so that 'E' in
    # hybrid_override_pattern is recognised as "moe" when NemotronHModel
    # iterates over config.layers_block_type to build its decoder layers.
    if type(config).__name__ == "NemotronHConfig":
        _patch_nemotron_h_config(config)

    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if (module_name :=
                type(module).__name__) in CUSTOM_MODULE_PATCHES.keys():
            patches = CUSTOM_MODULE_PATCHES[module_name]
            for method_name, method_patch in patches:
                setattr(module, method_name,
                        types.MethodType(method_patch, module))

    return model


# Module-level constants for mamba_ssm stub
_mamba_ssm_module = "mamba_ssm"
_mamba_ssm_submodule = f"{_mamba_ssm_module}.ops.triton.layernorm_gated"


def apply():
    """
    Apply Nemotron-H patches so the model can load and export without mamba_ssm CUDA extensions.
    Idempotent; safe to call multiple times.
    - Patches AutoModelForCausalLM.from_config to inject custom forward/mask behavior.
    - Stubs mamba_ssm.ops.triton.layernorm_gated with a pure-PyTorch RMS norm when mamba_ssm is missing.
    """
    AutoModelForCausalLM.from_config = get_model_from_config_patched
    if importlib.util.find_spec(_mamba_ssm_module) is None:
        stub_mod = types.ModuleType(_mamba_ssm_submodule)
        stub_mod.rmsnorm_fn = _rms_norm_ref
        sys.modules[_mamba_ssm_submodule] = stub_mod
