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
From-scratch Qwen3-ASR / Qwen3-Omni audio encoder implementation.

Architecture (Whisper-style encoder):
    CNN stack:  conv2d1 → conv2d2 → conv2d3  (depthwise downsampling)
    conv_out:   Linear(downsample_hidden_size * freq_bins, d_model)
    positional_embedding:  SinusoidsPositionEmbedding (fixed, non-learned)
    layers:     32 × QwenAudioEncoderLayer
        self_attn_layer_norm, self_attn (q/k/v_proj, out_proj), final_layer_norm, fc1, fc2
    ln_post:    LayerNorm
    proj1, act, proj2:  output projection

Checkpoint weight key prefix:
    Qwen3-ASR standalone:  ``audio_tower.*``   (or no prefix if a pure audio checkpoint)
    Qwen3-Omni / Qwen3-TTS embedded:  ``thinker.audio_tower.*`` or ``audio_tower.*``

ONNX Forward I/O:
    Inputs:
        padded_feature                 [num_chunks, num_mel_bins, n_window*2]  float16
        padded_mask_after_cnn_indices  [num_attention_elems, 2]               int64
        attention_mask                 [num_attention_elems, num_attention_elems] float16
    Output:
        last_hidden_state              [num_attention_elems, output_dim]       float16

The ``padded_mask_after_cnn_indices`` approach avoids non-zero ONNX nodes that are
TensorRT-unfriendly. The C++ runtime computes this tensor during pre-processing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default architecture constants (Qwen3-ASR / Qwen3-Omni)
# ---------------------------------------------------------------------------

_D_MODEL = 1280
_NUM_LAYERS = 32
_NUM_HEADS = 20
_FFN_DIM = 5120
_NUM_MEL_BINS = 128
_MAX_SOURCE_POSITIONS = 1500
_OUTPUT_DIM = 3584  # LLM hidden size (Qwen3-7.5B)
_DOWNSAMPLE_HIDDEN = 480
_N_WINDOW = 100

# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (fixed, not learned)
# ---------------------------------------------------------------------------


class SinusoidsPositionEmbedding(nn.Module):
    """Fixed sinusoidal position encoding as used in Whisper / Qwen3-ASR.

    Checkpoint buffer: ``positional_embedding``  [max_source_positions, d_model]
    """

    def __init__(self,
                 length: int = _MAX_SOURCE_POSITIONS,
                 channels: int = _D_MODEL,
                 max_timescale: int = 10000) -> None:
        super().__init__()
        assert channels % 2 == 0
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment *
            torch.arange(channels // 2, dtype=torch.float32))
        scaled_time = torch.arange(
            length,
            dtype=torch.float32).unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(scaled_time),
                             torch.cos(scaled_time)],
                            dim=1)
        self.register_buffer("positional_embedding", pos_emb, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class QwenAudioAttention(nn.Module):
    """Multi-head self-attention for the audio encoder.

    Checkpoint keys (under encoder prefix + ``layers.N.self_attn``):
        q_proj.{weight,bias}, k_proj.{weight,bias},
        v_proj.{weight,bias}, out_proj.{weight,bias}
    """

    def __init__(self,
                 d_model: int = _D_MODEL,
                 num_heads: int = _NUM_HEADS) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [T, d_model] — ragged sequence (all chunks concatenated)
            attention_mask: [T, T] additive mask (0 = attend, -inf = ignore)
        """
        T = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(T, self.num_heads,
                                            self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(T, self.num_heads,
                                            self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(T, self.num_heads,
                                            self.head_dim).transpose(0, 1)
        # q/k/v: [num_heads, T, head_dim]
        # Explicit softmax attention (avoids SDPA op which TRT ONNX parser rejects).
        # scores: [num_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # attention_mask: [T, T] → [1, T, T] (broadcast over heads)
        scores = scores + attention_mask.unsqueeze(0)
        attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn_weights, v)
        # out: [num_heads, T, head_dim] → [T, num_heads * head_dim]
        out = out.transpose(0, 1).reshape(T, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------


class QwenAudioEncoderLayer(nn.Module):
    """Single Qwen3-ASR encoder block (pre-norm).

    Checkpoint keys (under encoder prefix + ``layers.N``):
        self_attn_layer_norm.{weight,bias}
        self_attn.*
        final_layer_norm.{weight,bias}
        fc1.{weight,bias}
        fc2.{weight,bias}
    """

    def __init__(self,
                 d_model: int = _D_MODEL,
                 num_heads: int = _NUM_HEADS,
                 ffn_dim: int = _FFN_DIM,
                 activation: str = "gelu") -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = QwenAudioAttention(d_model, num_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self._activation = activation

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_val = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_val,
                                        max=clamp_val)
        return hidden_states


# ---------------------------------------------------------------------------
# Full audio encoder
# ---------------------------------------------------------------------------


class QwenAudioEncoder(nn.Module):
    """Complete Qwen3-ASR / Qwen3-Omni audio encoder.

    ONNX forward:
        padded_feature                 [C, num_mel_bins, n_window*2]  float16
        padded_mask_after_cnn_indices  [T, 2]                        int64
        attention_mask                 [T, T]                        float16
        → last_hidden_state            [T, output_dim]               float16

    Checkpoint keys are directly under the constructor prefix stripped by
    :func:`build_qwen_audio`.
    """

    def __init__(self,
                 num_mel_bins: int = _NUM_MEL_BINS,
                 d_model: int = _D_MODEL,
                 num_layers: int = _NUM_LAYERS,
                 num_heads: int = _NUM_HEADS,
                 ffn_dim: int = _FFN_DIM,
                 max_source_positions: int = _MAX_SOURCE_POSITIONS,
                 output_dim: int = _OUTPUT_DIM,
                 downsample_hidden: int = _DOWNSAMPLE_HIDDEN) -> None:
        super().__init__()
        self.num_mel_bins = num_mel_bins

        # CNN downsampling stack
        self.conv2d1 = nn.Conv2d(1,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv2d2 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv2d3 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        # Compute frequency bins after 3 stride-2 convolutions
        freq_bins = ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(downsample_hidden * freq_bins,
                                  d_model,
                                  bias=False)
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_source_positions, d_model)
        self.layers = nn.ModuleList([
            QwenAudioEncoderLayer(d_model, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, output_dim)

    def forward(
        self,
        padded_feature: torch.Tensor,
        padded_mask_after_cnn_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            padded_feature: [num_chunks, num_mel_bins, n_window*2]
            padded_mask_after_cnn_indices: [num_attention_elems, 2]  int64
            attention_mask: [num_attention_elems, num_attention_elems]  float16

        Returns:
            [num_attention_elems, output_dim]
        """
        x = padded_feature.unsqueeze(1)  # [C, 1, mel, t]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # x: [C, D, freq, T_out]
        b, c, f, t = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        # x: [C, T_out, d_model]

        # Add positional encoding
        pos = self.positional_embedding.positional_embedding[:x.shape[
            1], :].unsqueeze(0).to(x.dtype)  # type: ignore[attr-defined]
        x = x + pos

        # Gather valid tokens using pre-computed nonzero indices
        hidden_states = x[padded_mask_after_cnn_indices[:, 0],
                          padded_mask_after_cnn_indices[:, 1]]  # [T, d_model]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_CANDIDATE_PREFIXES = (
    "audio_tower.",
    "thinker.audio_tower.",
    "model.audio_tower.",
    "",  # direct keys (pure audio checkpoint)
)


def _load_audio_weights(model: QwenAudioEncoder,
                        weights: dict,
                        prefix: str | None = None) -> None:
    """Load safetensors weights into *model*, stripping *prefix*.

    If *prefix* is ``None``, the function auto-detects from
    :data:`_CANDIDATE_PREFIXES`.
    """
    import logging
    logger = logging.getLogger(__name__)

    if prefix is None:
        for cand in _CANDIDATE_PREFIXES:
            if cand == "" or any(k.startswith(cand) for k in weights.keys()):
                prefix = cand
                break
        else:
            prefix = ""

    stripped: dict = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            stripped[k[len(prefix):]] = v

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        logger.warning("QwenAudioEncoder: missing keys: %s", missing[:10])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_qwen_audio(
    config: dict,
    weights: dict,
    dtype: torch.dtype = torch.float16,
    prefix: str | None = None,
) -> QwenAudioEncoder:
    """Build and return a :class:`QwenAudioEncoder` with loaded weights.

    Args:
        config:  Model config dict; recognized keys mirror
                 ``Qwen3ASRAudioEncoderConfig`` field names.
                 May be a top-level config (with ``audio_config`` sub-key)
                 or an audio-only sub-config directly.
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype (default ``float16``).
        prefix:  Checkpoint key prefix to strip. ``None`` = auto-detect.
    """
    # Support both top-level config with audio_config sub-key and direct audio config
    audio_cfg = config.get("audio_config", config)

    def _get(key: str, default):
        return audio_cfg.get(key, config.get(key, default))

    model = QwenAudioEncoder(
        num_mel_bins=_get("num_mel_bins", _NUM_MEL_BINS),
        d_model=_get("d_model", _D_MODEL),
        num_layers=_get("encoder_layers", _NUM_LAYERS),
        num_heads=_get("encoder_attention_heads", _NUM_HEADS),
        ffn_dim=_get("encoder_ffn_dim", _FFN_DIM),
        max_source_positions=_get("max_source_positions",
                                  _MAX_SOURCE_POSITIONS),
        output_dim=_get("output_dim", _OUTPUT_DIM),
        downsample_hidden=_get("downsample_hidden_size", _DOWNSAMPLE_HIDDEN),
    )
    _load_audio_weights(model, weights, prefix)
    model = model.to(dtype=dtype)
    model.eval()
    return model


__all__ = [
    "SinusoidsPositionEmbedding",
    "QwenAudioAttention",
    "QwenAudioEncoderLayer",
    "QwenAudioEncoder",
    "build_qwen_audio",
]
