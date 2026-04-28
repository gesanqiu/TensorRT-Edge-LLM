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
"""Marlin NVFP4 tile packing, FP8 block-scale conversion, and INT4 Marlin swizzle tables (``Nvfp4MoePlugin``, ``Int4MoePlugin``, etc.)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch


class MarlinConverter:
    """
    Marlin NVFP4 tile quantizer for MoE expert weights (matches ``Nvfp4MoePlugin`` decode).

    Encodes 64-lane weight tiles to INT32-packed E2M1 nibbles plus INT32-packed FP8 E4M3 block scales,
    using the same rules as the CUDA plugin (per-16-lane FP32 scale ``max(|w|)/6``, expert-wide
    ``S_max`` for FP8 normalization, ``global_scale = S_max / block_fp8_quant_max`` on the module).

    Marlin block-scale helpers (:meth:`marlin_f8x4_block_scale_as_i32`,
    :meth:`marlin_f8x4_block_scale_to_fp32x4`, :meth:`fp32x4_to_marlin_f8x4_block_scale`) are
    stateless utilities exposed as static methods for use without constructing a converter instance.

    INT4 Marlin tensor-core swizzle (module-level ``MARLIN_INT4_*`` indices) is exposed via
    :meth:`swizzle` and :meth:`inverse_swizzle`. GPTQ/AWQ INT4 expert weights are packed with
    :meth:`pack_int4_awq_marlin`.     Dense FP32 → NVFP4 nibble + group scales for ``Nvfp4MoePlugin`` uses
    :meth:`quantize_fp32_to_nvfp4_in_int16_with_block_scale` and
    :meth:`fp32_marlin_scales_to_fp8_e4m3_in_int8`. Single-value helpers are
    :meth:`f32_to_fp4_e2m1_nibble`.
    """

    FP8_MAX: ClassVar[float] = 448.0

    # NVFP4 E2M1 positive magnitudes (nibble codes 0..7); sign is bit 3 of the nibble (CUDA ``__NV_E2M1``).
    FP4_E2M1_POSITIVE_LEVELS: ClassVar[tuple[float,
                                             ...]] = (0.0, 0.5, 1.0, 1.5, 2.0,
                                                      3.0, 4.0, 6.0)
    # :meth:`f32_to_fp4_e2m1_nibble` returns NVFP4 E2M1 u4 codes ``0x0``..``0xF`` (``__NV_E2M1``) directly — no LUT.

    __slots__ = ("fp8_max", )

    def __init__(self, block_fp8_quant_max: float | None = None) -> None:
        self.fp8_max = float(
            type(self).FP8_MAX if block_fp8_quant_max is
            None else block_fp8_quant_max)

    @property
    def block_fp8_quant_max(self) -> float:
        return self.fp8_max

    @staticmethod
    def fp32_marlin_scales_to_fp8_e4m3_in_int8(
        scales_marlin: torch.Tensor,
        s_max_per_ex: torch.Tensor,
        k_fp8: float | None = None,
        *,
        out_device: torch.device,
    ) -> torch.Tensor:
        """Map per-expert-normalized Marlin float scales to FP8 E4M3 bytes (``int8`` view).

        ``k_fp8`` defaults to :attr:`FP8_MAX` (FP8 E4M3 finite range cap for normalized scales).
        """
        e = scales_marlin.shape[0]
        kmax = float(MarlinConverter.FP8_MAX if k_fp8 is None else k_fp8)
        norm = scales_marlin.float().to(out_device) / s_max_per_ex.to(
            out_device).view(e, 1, 1).clamp(min=1e-12) * kmax
        norm = norm.clamp(0.0, kmax)
        dt8 = getattr(torch, "float8_e4m3fn", None)
        if dt8 is None:
            raise RuntimeError(
                "torch.float8_e4m3fn is required for Nvfp4MoePlugin FP8 block scales; "
                "use a PyTorch build that supports float8 E4M3.")
        return norm.to(dt8).view(torch.int8)

    @staticmethod
    def marlin_f8x4_block_scale_as_i32(word: int) -> int:
        """
        Reinterpret one **f8x4**-packed block-scale word (four FP8 E4M3 bytes in 32 bits) as signed ``int32``
        for plugin INT8 buffers / ``torch.int32``.

        Words with the high bit set in the unsigned sense cannot be passed directly to
        ``torch.tensor(..., dtype=torch.int32)`` without overflow; CUDA still consumes the same four bytes.
        """
        return int(np.int32(np.uint32(int(word) & 0xFFFFFFFF)))

    @staticmethod
    def marlin_f8x4_block_scale_to_fp32x4(q: int) -> np.ndarray:
        """
        Decode one **f8x4** Marlin block-scale word to four FP32 lane-group scales (same as CUDA host ref / NumPy tests).

        Order matches ``scale_f[0..3]`` in ``referenceNvfp4x64TileToFp32x64`` (groups of 16 NVFP4 lanes).
        """
        qq = int(q) & 0xFFFFFFFF
        out1 = (qq & 0xFF00FF00) >> 1
        qs = (qq << 8) & 0xFFFFFFFF
        out2 = (qs & 0xFF00FF00) >> 1
        u1 = np.uint32(out1)
        u2 = np.uint32(out2)
        f1 = np.frombuffer(u1.tobytes(), dtype=np.float16).astype(np.float32)
        f2 = np.frombuffer(u2.tobytes(), dtype=np.float16).astype(np.float32)
        return np.array([f2[0], f2[1], f1[0], f1[1]], dtype=np.float32)

    @staticmethod
    def marlin_f8x4_block_scale_to_f16x2(q: int) -> tuple[int, int]:
        """Match ``marlin::dequant_fp8_scales<half2, kFE4M3fn>``: f8x4 word → two uint32 **f16x2** (``half2``) bit patterns."""
        u = np.uint32(int(q) & 0xFFFFFFFF)
        out1 = int((u & np.uint32(0xFF00FF00)) >> np.uint32(1))
        u_s = np.uint32((u << 8) & np.uint32(0xFFFFFFFF))
        out2 = int((u_s & np.uint32(0xFF00FF00)) >> np.uint32(1))
        return out1, out2

    @staticmethod
    def f16x2_in_u32_to_marlin_f8x4(out_u32: int) -> int:
        """
        Project one **f16x2** (``half2``) uint32 bit pattern onto the manifold reachable from Marlin ``dequant_fp8_scales``.

        Forward uses only odd bytes of ``(word << 1)``; mask clears the others so packing inverts decode
        without loops (see :meth:`fp32x4_to_marlin_f8x4_block_scale`).
        """
        o = np.uint32(int(out_u32) & 0xFFFFFFFF)
        u = np.uint32((np.uint64(o) << np.uint64(1)) & np.uint64(0xFFFFFFFF))
        u = u & np.uint32(0xFF00FF00)
        return int(np.uint32(u >> np.uint32(1)))

    @staticmethod
    def f16x4_in_u32x2_to_marlin_f8x4_block_scale(out1: int, out2: int) -> int:
        """
        Merge four FP16 lane values (**f16x4**) given as two ``half2`` uint32 words (**u32×2**) into one f8x4 block-scale int32.

        Closed-form inverse of :meth:`marlin_f8x4_block_scale_to_f16x2` for projected ``out1``/``out2``.
        """
        o1 = np.uint32(int(out1) & 0xFFFFFFFF)
        o2 = np.uint32(int(out2) & 0xFFFFFFFF)
        u1 = np.uint32((np.uint64(o1) << np.uint64(1)) & np.uint64(0xFFFFFFFF))
        u2 = np.uint32((np.uint64(o2) << np.uint64(1)) & np.uint64(0xFFFFFFFF))
        b1 = int((np.uint64(u1) >> 8) & np.uint64(0xFF))
        b3 = int((np.uint64(u1) >> 24) & np.uint64(0xFF))
        b0 = int((np.uint64(u2) >> 8) & np.uint64(0xFF))
        b2 = int((np.uint64(u2) >> 24) & np.uint64(0xFF))
        return int(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))

    @staticmethod
    def fp32x4_to_marlin_f8x4_block_scale(
        s0: float,
        s1: float,
        s2: float,
        s3: float,
    ) -> int:
        """
        Pack four normalized per-group scales (FP32) into one **f8x4** int32 for the plugin block-scale buffer.

        Rounds each target to FP16, forms the two **f16x2** uint32 words CUDA reinterprets after
        ``dequant_fp8_scales``, **projects** those words onto the Marlin bit manifold (no search), then
        inverts the forward shuffle in closed form so :meth:`marlin_f8x4_block_scale_to_fp32x4` matches
        CUDA. Values that are not exactly representable on that manifold (e.g. some FP16 scalings) snap
        slightly (e.g. 100 → 96) instead of decoding to unrelated magnitudes.
        """
        h16 = np.array([s0, s1, s2, s3], dtype=np.float32).astype(np.float16)
        out2_raw = int(
            np.frombuffer(h16[0:2].tobytes(), dtype="<u4", count=1)[0])
        out1_raw = int(
            np.frombuffer(h16[2:4].tobytes(), dtype="<u4", count=1)[0])
        out1 = MarlinConverter.f16x2_in_u32_to_marlin_f8x4(out1_raw)
        out2 = MarlinConverter.f16x2_in_u32_to_marlin_f8x4(out2_raw)
        return MarlinConverter.f16x4_in_u32x2_to_marlin_f8x4_block_scale(
            out1, out2)

    @staticmethod
    def f32_to_fp4_e2m1_nibble(x: float) -> int:
        mag = abs(float(x))
        sign_bit = 8 if x < 0.0 else 0
        best_i = 0
        best_d = mag
        for i, lv in enumerate(MarlinConverter.FP4_E2M1_POSITIVE_LEVELS):
            d = abs(mag - lv)
            if d < best_d:
                best_d = d
                best_i = i
        return int((sign_bit | best_i) & 0xF)

    def f32x64_to_max_f32_block_scale(self, vec64: np.ndarray) -> float:
        """
        Maximum FP32 block scale ``max(|block|)/6`` over the four 16-lane groups in one Marlin tile.
        Same definition as the per-expert maximum aggregated over all tiles in that projection.
        """
        v = np.asarray(vec64, dtype=np.float32).reshape(64)
        s_max = 1e-12
        for g in range(4):
            blk = v[g * 16:(g + 1) * 16]
            am = float(np.max(np.abs(blk)))
            s_max = max(s_max, max(am / 6.0, 1e-12))
        return float(s_max)

    @staticmethod
    def quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
        vec64: np.ndarray,
        expert_block_scale_max_fp32: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Pack one 64-lane Marlin tile to ``(payload_int32x8, block_scale_packed_int32)``.

        ``expert_block_scale_max_fp32`` is ``S_max`` for that expert's up or down tensor: normalized
        targets ``(s_g / S_max) * block_fp8_quant_max`` are rounded to FP16 and packed for Marlin
        ``dequant_fp8_scales`` (see :meth:`fp32x4_to_marlin_f8x4_block_scale`). The plugin
        uses ``global_scale = S_max / block_fp8_quant_max`` so decode × global tracks ``max(|w|)/6`` up
        to FP16 / manifold projection.
        """
        v = np.asarray(vec64, dtype=np.float32).reshape(64)
        s_max = max(float(expert_block_scale_max_fp32), 1e-12)
        scales = np.zeros(4, dtype=np.float32)
        for g in range(4):
            blk = v[g * 16:(g + 1) * 16]
            am = float(np.max(np.abs(blk)))
            scales[g] = max(am / 6.0, 1e-12)
        nib = np.empty(64, dtype=np.int32)
        for i in range(64):
            g = i // 16
            qn = float(v[i] / scales[g])
            if qn > 6.0:
                qn = 6.0
            elif qn < -6.0:
                qn = -6.0
            nib[i] = MarlinConverter.f32_to_fp4_e2m1_nibble(qn)
        out = np.zeros(8, dtype=np.uint32)
        for lane in range(8):
            w = np.uint32(0)
            for j in range(4):
                base = lane * 8 + j * 2
                lo = int(nib[base]) & 0xF
                hi = int(nib[base + 1]) & 0xF
                b = np.uint32(lo | (hi << 4))
                w |= b << (8 * j)
            out[lane] = w
        scale_q = MarlinConverter.fp32x4_to_marlin_f8x4_block_scale(
            float(scales[0]) / s_max * MarlinConverter.FP8_MAX,
            float(scales[1]) / s_max * MarlinConverter.FP8_MAX,
            float(scales[2]) / s_max * MarlinConverter.FP8_MAX,
            float(scales[3]) / s_max * MarlinConverter.FP8_MAX,
        )
        return out.astype(np.int32), int(scale_q)

    @staticmethod
    def atom_sf_offset(m_idx: int, k_idx: int, num_sf_cols: int) -> int:
        """Byte offset for atom-layout 128x4 swizzle (matches ``fp4Quantize.cu`` ``get_sf_out_offset_128x4``)."""
        inner_k = k_idx % 4
        inner_m = (m_idx % 128) // 32
        outer_m = m_idx % 32
        k_tile = k_idx // 4
        num_k_tiles = (num_sf_cols + 3) // 4
        m_tile = m_idx // 128
        return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k

    @staticmethod
    def atom_sf_bytes_per_expert(m_dim: int,
                                 k_dim: int,
                                 sf_vec: int = 16) -> int:
        """Total atom-layout scale factor buffer bytes per expert (or per M x K plane)."""
        num_sf_cols = k_dim // sf_vec
        padded_sf_cols = ((num_sf_cols + 3) // 4) * 4
        padded_m = ((m_dim + 127) // 128) * 128
        return padded_m * padded_sf_cols

    @staticmethod
    def marlin_scale_word_to_raw_fp8_bytes(scale_word: int) -> np.ndarray:
        """Extract 4 raw FP8 E4M3 bytes in natural group order from a Marlin-packed scale int32.

        Marlin byte layout is ``{s0, s2, s1, s3}``; this returns ``uint8[4]`` = ``{s0, s1, s2, s3}``.
        """
        b = np.frombuffer(np.uint32(int(scale_word) & 0xFFFFFFFF).tobytes(),
                          dtype=np.uint8)
        return np.array([b[0], b[2], b[1], b[3]], dtype=np.uint8)
