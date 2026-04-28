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
From-scratch InternVL3.5 vision encoder (model_type: ``internvl``).

Architecture (HuggingFace ``transformers.models.internvl``):
    InternVL3_5VisionEmbeddings
        cls_token  +  patch_embeddings.projection Conv2d
        + position_embeddings [1, N+1, H]  (learnable)
    → num_hidden_layers × InternVL3_5VisionLayer
        layernorm_before  →  attention(q/k/v_proj, projection_layer) × lambda_1  →  residual
        layernorm_after   →  mlp(fc1, GELU, fc2) × lambda_2  →  residual
    → pixel_shuffle downsampling (downsample_ratio)
    → multi_modal_projector: layer_norm → linear_1 → GELU → linear_2

Checkpoint weight key prefixes:
    Vision encoder:  ``vision_tower.*``
    Projector:       ``multi_modal_projector.*``

Reference: ``transformers/models/internvl/modeling_internvl.py``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _make_norm(norm_type: str, hidden_size: int, eps: float) -> nn.Module:
    if norm_type == "rms_norm":
        return _RMSNorm(hidden_size, eps)
    return nn.LayerNorm(hidden_size, eps=eps)


class _RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class _InternVL3_5PatchEmbeddings(nn.Module):
    """Conv2d patch projection.

    Checkpoint key: ``vision_tower.embeddings.patch_embeddings.projection.*``
    """

    def __init__(self, num_channels: int, hidden_size: int,
                 patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(num_channels,
                                    hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.projection(pixel_values.to(self.projection.weight.dtype))
        return x.flatten(2).transpose(1, 2)


class InternVL3_5VisionEmbeddings(nn.Module):
    """CLS token + patch embedding + absolute position embedding.

    Checkpoint keys (under ``vision_tower.embeddings.``):
        ``cls_token``                                    [1, 1, hidden_size]
        ``patch_embeddings.projection.{weight,bias}``
        ``position_embeddings``                          [1, num_patches + 1, hidden_size]
    """

    def __init__(self, num_channels: int, hidden_size: int, patch_size: int,
                 image_size: int, use_abs_pos_embed: bool) -> None:
        super().__init__()
        num_patches = (image_size // patch_size)**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embeddings = _InternVL3_5PatchEmbeddings(
            num_channels, hidden_size, patch_size)
        if use_abs_pos_embed:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches + 1, hidden_size))
        else:
            self.position_embeddings = None  # type: ignore[assignment]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.size(0)
        patch_embeds = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patch_embeds), dim=1)
        if self.position_embeddings is not None:
            x = x + self.position_embeddings
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class InternVL3_5VisionAttention(nn.Module):
    """Separate-projection MHA for InternVL3.5.

    Checkpoint keys (under ``vision_tower.encoder.layer.N.attention.``):
        ``q_proj.*``, ``k_proj.*``, ``v_proj.*``
        ``projection_layer.*``
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 attention_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.embed_dim = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)
        self.projection_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(B, N, self.num_heads,
                                               self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).reshape(B, N, self.num_heads,
                                               self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).reshape(B, N, self.num_heads,
                                               self.head_dim).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale,
                         dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(
            B, N, self.embed_dim)
        return self.projection_layer(out)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class InternVL3_5VisionMLP(nn.Module):
    """Two-layer GELU FFN.

    Checkpoint keys: ``mlp.fc1.*``, ``mlp.fc2.*``
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        act_map = {
            "gelu": F.gelu,
            "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "relu": F.relu,
            "silu": F.silu,
        }
        self.act_fn = act_map.get(hidden_act, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


# ---------------------------------------------------------------------------
# Vision Layer
# ---------------------------------------------------------------------------


class InternVL3_5VisionLayer(nn.Module):
    """Single InternVL3.5 vision transformer layer with layer-scale (lambda_1, lambda_2).

    Checkpoint keys (under ``vision_tower.encoder.layer.N.``):
        ``layernorm_before.*``, ``attention.*``, ``lambda_1``
        ``layernorm_after.*``,  ``mlp.*``,       ``lambda_2``
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 intermediate_size: int, hidden_act: str, attention_bias: bool,
                 norm_type: str, layer_norm_eps: float,
                 layer_scale_init: float) -> None:
        super().__init__()
        self.layernorm_before = _make_norm(norm_type, hidden_size,
                                           layer_norm_eps)
        self.attention = InternVL3_5VisionAttention(hidden_size, num_heads,
                                                    attention_bias)
        self.lambda_1 = nn.Parameter(layer_scale_init *
                                     torch.ones(hidden_size),
                                     requires_grad=True)
        self.layernorm_after = _make_norm(norm_type, hidden_size,
                                          layer_norm_eps)
        self.mlp = InternVL3_5VisionMLP(hidden_size, intermediate_size,
                                        hidden_act)
        self.lambda_2 = nn.Parameter(layer_scale_init *
                                     torch.ones(hidden_size),
                                     requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.lambda_1 * self.attention(
            self.layernorm_before(hidden_states))
        hidden_states = hidden_states + self.lambda_2 * self.mlp(
            self.layernorm_after(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------


class InternVL3_5VisionEncoder(nn.Module):
    """Stack of InternVL3_5VisionLayer.

    Checkpoint keys: ``vision_tower.encoder.layer.N.*``
    """

    def __init__(self, num_hidden_layers: int, hidden_size: int,
                 num_heads: int, intermediate_size: int, hidden_act: str,
                 attention_bias: bool, norm_type: str, layer_norm_eps: float,
                 layer_scale_init: float) -> None:
        super().__init__()
        self.layer = nn.ModuleList([
            InternVL3_5VisionLayer(hidden_size, num_heads, intermediate_size,
                                   hidden_act, attention_bias, norm_type,
                                   layer_norm_eps, layer_scale_init)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Multi-modal projector
# ---------------------------------------------------------------------------


class _InternVL3_5Projector(nn.Module):
    """layer_norm → linear_1 → GELU → linear_2.

    Checkpoint keys (under ``multi_modal_projector.``):
        ``layer_norm.*``, ``linear_1.*``, ``linear_2.*``
    """

    def __init__(self, in_dim: int, text_hidden_size: int,
                 projector_hidden_act: str) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear_1 = nn.Linear(in_dim, text_hidden_size)
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size)
        act_map = {
            "gelu": F.gelu,
            "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "relu": F.relu,
        }
        self.act_fn = act_map.get(projector_hidden_act, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        return x


# ---------------------------------------------------------------------------
# Pixel shuffle
# ---------------------------------------------------------------------------


def _pixel_shuffle(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    B, H, W, C = x.shape
    scale = int(1.0 / scale_factor)
    new_H, new_W = H // scale, W // scale
    x = x.view(B, new_H, scale, new_W, scale, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, new_H, new_W, scale * scale * C)
    return x


# ---------------------------------------------------------------------------
# Top-level visual model
# ---------------------------------------------------------------------------


class InternVL3_5VisualModel(nn.Module):
    """From-scratch InternVL3.5 visual encoder + projector (model_type: internvl).

    Expects the full ``config.json`` dict (not just ``vision_config``), because it
    needs ``text_config.hidden_size`` and ``downsample_ratio`` from the top level.

    Output: ``[total_patches * tokens_per_patch, text_hidden_size]``
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        vc = config["vision_config"]
        tc = config["text_config"]
        text_hidden_size: int = tc["hidden_size"]
        self.downsample_ratio: float = float(config["downsample_ratio"])

        image_size = vc["image_size"]
        if isinstance(image_size, list):
            image_size = image_size[0]
        patch_size = vc["patch_size"]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]

        hidden_size: int = vc["hidden_size"]
        # Precompute spatial side length (patches per side) so the forward pass
        # never calls int(math.isqrt(N)) on a runtime tensor shape, which would
        # specialize the batch dimension during torch.export tracing.
        self._feat_side: int = image_size // patch_size

        self.vision_tower = nn.Module.__new__(nn.Module)
        nn.Module.__init__(self.vision_tower)

        # Build embeddings + encoder under vision_tower sub-module
        # so weight keys like vision_tower.embeddings.* map correctly
        self.vision_tower.embeddings = InternVL3_5VisionEmbeddings(
            num_channels=vc.get("num_channels", 3),
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_size=image_size,
            use_abs_pos_embed=vc.get("use_absolute_position_embeddings", True),
        )
        self.vision_tower.encoder = InternVL3_5VisionEncoder(
            num_hidden_layers=vc["num_hidden_layers"],
            hidden_size=hidden_size,
            num_heads=vc["num_attention_heads"],
            intermediate_size=vc["intermediate_size"],
            hidden_act=vc.get("hidden_act", "gelu"),
            attention_bias=vc.get("attention_bias", True),
            norm_type=vc.get("norm_type", "layer_norm"),
            layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
            layer_scale_init=float(vc.get("layer_scale_init_value", 0.1)),
        )

        scale = int(1.0 / self.downsample_ratio)
        in_dim = hidden_size * scale * scale

        self.multi_modal_projector = _InternVL3_5Projector(
            in_dim=in_dim,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=config.get("projector_hidden_act", "gelu"),
        )
        self._text_hidden_size = text_hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [num_patches, C, H, W]

        Returns:
            image_features: [num_patches * tokens_per_patch, text_hidden_size]
        """
        # Vision tower: [B, N+1, H]
        x = self.vision_tower.embeddings(pixel_values)
        x = self.vision_tower.encoder(x)

        # Remove CLS token: [B, N, H]
        x = x[:, 1:, :]

        B = x.shape[0]
        C = x.shape[2]
        feat_side = self._feat_side  # precomputed in __init__, avoids int() on SymInt
        x = x.reshape(B, feat_side, feat_side, C)
        x = _pixel_shuffle(x, self.downsample_ratio)
        x = x.reshape(B, -1, x.shape[-1])
        x = self.multi_modal_projector(x)
        return x.reshape(-1, self._text_hidden_size)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes) for ONNX export."""
        vc = config.get("vision_config", config)
        image_size = vc["image_size"]
        if isinstance(image_size, list):
            image_size = image_size[0]
        num_channels = vc.get("num_channels", 3)
        # Use batch=2 so the tracer sees a non-unit batch and keeps it symbolic.
        # A batch=1 sample causes torch.export to specialize the batch dim to 1.
        pixel_values = torch.zeros(2,
                                   num_channels,
                                   image_size,
                                   image_size,
                                   dtype=torch.float16,
                                   device=device)
        args = (pixel_values, )
        input_names = ["input"]
        output_names = ["output"]
        N = torch.export.Dim("num_images", min=1)
        dynamic_shapes = {"pixel_values": {0: N}}
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_internvl3_5_weights(model: InternVL3_5VisualModel,
                              weights: dict) -> None:
    """Load vision_tower and multi_modal_projector weights."""
    import logging
    logger = logging.getLogger(__name__)

    vt_prefix = "vision_tower."
    proj_prefix = "multi_modal_projector."

    vt_state: dict = {}
    proj_state: dict = {}
    for k, v in weights.items():
        if k.startswith(vt_prefix):
            vt_state[k[len(vt_prefix):]] = v
        elif k.startswith(proj_prefix):
            proj_state[k[len(proj_prefix):]] = v

    missing_vt, _ = model.vision_tower.load_state_dict(vt_state, strict=False)
    missing_proj, _ = model.multi_modal_projector.load_state_dict(proj_state,
                                                                  strict=False)

    if missing_vt:
        logger.warning("InternVL3_5VisualModel: missing vision_tower keys: %s",
                       missing_vt[:10])
    if missing_proj:
        logger.warning("InternVL3_5VisualModel: missing projector keys: %s",
                       missing_proj[:10])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_internvl3_5_visual(
        config: dict,
        weights: dict,
        dtype: torch.dtype = torch.float16) -> InternVL3_5VisualModel:
    """Build and return an :class:`InternVL3_5VisualModel` with loaded weights.

    Args:
        config:  Full parsed ``config.json`` dict (contains ``vision_config``,
                 ``text_config``, ``downsample_ratio``).
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype (default ``float16``).
    """
    model = InternVL3_5VisualModel(config)
    model.to(dtype)
    _load_internvl3_5_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "InternVL3_5VisualModel",
    "build_internvl3_5_visual",
]
