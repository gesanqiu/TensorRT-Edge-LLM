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
QqqGemmPlugin for TensorRT Integration (W4A8: INT4 weights, INT8 activations)

This module provides a custom TensorRT operation for QQQ W4A8 GEMM that can be
exported to ONNX format. It loads QQQ-quantized Marlin-format weights directly
(no repacking needed) and maps them to the QqqGemmPlugin TensorRT plugin.

The module contains:
- QqqGemmPluginModule: Custom module replacing QQQ QuantLinear for ONNX export
- qqq_gemm_plugin / qqq_gemm_plugin_grouped: Dummy TRT ops for per-channel / per-group
- replace_qqq_linear_with_plugin: Replace QQQ QuantLinear modules with QqqGemmPluginModule
- ONNX export registration utilities
"""

import torch
import torch.nn as nn
from onnx import defs as onnx_defs
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION

# ---------------------------------------------------------------------------
# ONNX Schema: per-channel (3 inputs)
# ---------------------------------------------------------------------------
_qqq_gemm_schema_perchannel = OpSchema(
    name="QqqGemmPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION,
    doc="QQQ W4A8 GEMM plugin – per-channel mode (3 inputs).",
    inputs=[
        OpSchema.FormalParameter(name="input", description="Activation tensor", type_str="T"),
        OpSchema.FormalParameter(name="weights", description="Marlin-packed INT4 weights", type_str="tensor(int32)"),
        OpSchema.FormalParameter(name="s_channel", description="Per-channel weight scales", type_str="tensor(float)"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output", description="Output tensor", type_str="T"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], "Activation / output type."),
    ],
    attributes=[
        OpSchema.Attribute(name="gemm_n", type=OpSchema.AttrType.INT, description="Output feature dimension", required=True),
        OpSchema.Attribute(name="gemm_k", type=OpSchema.AttrType.INT, description="Input feature dimension", required=True),
        OpSchema.Attribute(name="group_size", type=OpSchema.AttrType.INT, description="Group size (-1 = per-channel)", required=True),
    ],
)

# Per-group schema registered under a different opset version to avoid duplicate
# name conflict.  The TRT ONNX parser matches on (domain, op) and ignores the
# since_version, so both schemas route to the same C++ plugin.
_qqq_gemm_schema_pergroup = OpSchema(
    name="QqqGemmPlugin",
    domain="trt",
    since_version=ONNX_OPSET_VERSION + 1,
    doc="QQQ W4A8 GEMM plugin – per-group mode (4 inputs).",
    inputs=[
        OpSchema.FormalParameter(name="input", description="Activation tensor", type_str="T"),
        OpSchema.FormalParameter(name="weights", description="Marlin-packed INT4 weights", type_str="tensor(int32)"),
        OpSchema.FormalParameter(name="s_channel", description="Per-channel weight scales", type_str="tensor(float)"),
        OpSchema.FormalParameter(name="s_group", description="Per-group weight scales", type_str="tensor(float16)"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output", description="Output tensor", type_str="T"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], "Activation / output type."),
    ],
    attributes=[
        OpSchema.Attribute(name="gemm_n", type=OpSchema.AttrType.INT, description="Output feature dimension", required=True),
        OpSchema.Attribute(name="gemm_k", type=OpSchema.AttrType.INT, description="Input feature dimension", required=True),
        OpSchema.Attribute(name="group_size", type=OpSchema.AttrType.INT, description="Group size (e.g. 128)", required=True),
    ],
)

onnx_defs.register_schema(_qqq_gemm_schema_perchannel)
onnx_defs.register_schema(_qqq_gemm_schema_pergroup)


# ---------------------------------------------------------------------------
# ONNX Symbolic functions
# ---------------------------------------------------------------------------


@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
def _symbolic_qqq_gemm_perchannel(g, input, weights, s_channel, gemm_n, gemm_k, group_size):
    """Emit trt::QqqGemmPlugin node with 3 inputs (per-channel)."""
    output = g.op(
        "trt::QqqGemmPlugin",
        input, weights, s_channel,
        gemm_n_i=gemm_n, gemm_k_i=gemm_k, group_size_i=group_size,
    )
    output_sizes = _get_tensor_sizes(input)[:-1] + [gemm_n]
    output.setType(input.type().with_sizes(output_sizes))
    return output


@symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "i")
def _symbolic_qqq_gemm_pergroup(g, input, weights, s_channel, s_group,
                                gemm_n, gemm_k, group_size):
    """Emit trt::QqqGemmPlugin node with 4 inputs (per-group)."""
    output = g.op(
        "trt::QqqGemmPlugin",
        input, weights, s_channel, s_group,
        gemm_n_i=gemm_n, gemm_k_i=gemm_k, group_size_i=group_size,
    )
    output_sizes = _get_tensor_sizes(input)[:-1] + [gemm_n]
    output.setType(input.type().with_sizes(output_sizes))
    return output


# ---------------------------------------------------------------------------
# Dummy custom ops (for torch.onnx.export tracing)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::qqq_gemm_perchannel", mutates_args=())
def qqq_gemm_perchannel(
    input: torch.Tensor,
    weights: torch.Tensor,
    s_channel: torch.Tensor,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """Dummy op for per-channel QQQ GEMM (used during ONNX tracing only)."""
    B, S, _ = input.shape
    return torch.zeros(B, S, gemm_n, dtype=input.dtype, device=input.device)


@torch.library.custom_op("trt::qqq_gemm_pergroup", mutates_args=())
def qqq_gemm_pergroup(
    input: torch.Tensor,
    weights: torch.Tensor,
    s_channel: torch.Tensor,
    s_group: torch.Tensor,
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """Dummy op for per-group QQQ GEMM (used during ONNX tracing only)."""
    B, S, _ = input.shape
    return torch.zeros(B, S, gemm_n, dtype=input.dtype, device=input.device)


# ---------------------------------------------------------------------------
# QqqGemmPluginModule
# ---------------------------------------------------------------------------


class QqqGemmPluginModule(nn.Module):
    """Replaces QQQ QuantLinear for ONNX export.

    QQQ checkpoint weights (B, s_channel, s_group) are already in
    Marlin-packed format and are passed through to the TRT plugin
    without any repacking.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features

        self._is_per_group = self.group_size > 0 and self.group_size < in_features

        # Marlin-packed INT4 weights: [K/16, 2*N] as int32
        self.register_buffer(
            "B",
            torch.zeros(in_features // 16, 2 * out_features, dtype=torch.int32),
        )

        # Per-channel weight scale: [1, N] as float32
        self.register_buffer(
            "s_channel",
            torch.zeros(1, out_features, dtype=torch.float32),
        )

        # Per-group weight scale (only for per-group mode)
        if self._is_per_group:
            self.register_buffer(
                "s_group",
                torch.zeros(in_features // self.group_size, out_features, dtype=torch.float16),
            )
        else:
            self.s_group = None

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_per_group:
            output = qqq_gemm_pergroup(
                input=x,
                weights=self.B,
                s_channel=self.s_channel,
                s_group=self.s_group,
                gemm_n=self.out_features,
                gemm_k=self.in_features,
                group_size=self.group_size,
            )
        else:
            output = qqq_gemm_perchannel(
                input=x,
                weights=self.B,
                s_channel=self.s_channel,
                gemm_n=self.out_features,
                gemm_k=self.in_features,
                group_size=-1,
            )
        if self.bias is not None:
            output = output + self.bias
        return output

    def load_from_qqq_linear(self, source: nn.Module) -> None:
        """Copy buffers from a QQQ QuantLinear module (no repacking needed)."""
        device = source.B.device
        self.to(device)
        with torch.no_grad():
            self.B.copy_(source.B)
            self.s_channel.copy_(source.s_channel)
            if self._is_per_group and hasattr(source, "s_group") and source.s_group.numel() > 0:
                self.s_group.copy_(source.s_group)
            if self.bias is not None and hasattr(source, "bias") and source.bias is not None:
                self.bias.copy_(source.bias)


# ---------------------------------------------------------------------------
# Registration and replacement helpers
# ---------------------------------------------------------------------------


def register_qqq_gemm_plugin_onnx_symbolic_functions() -> None:
    """Register ONNX symbolic functions for QQQ GEMM custom ops."""
    register_custom_op_symbolic(
        "trt::qqq_gemm_perchannel", _symbolic_qqq_gemm_perchannel, ONNX_OPSET_VERSION
    )
    register_custom_op_symbolic(
        "trt::qqq_gemm_pergroup", _symbolic_qqq_gemm_pergroup, ONNX_OPSET_VERSION
    )
    print("Registered ONNX symbolic functions for QqqGemmPlugin")


def _get_qqq_quantlinear_class():
    """Import QQQ's QuantLinear, falling back to direct file load if the
    QQQ package __init__ chain has incompatible transitive imports."""
    try:
        from QQQ.gptq.qlinear.qlinear_marlin import QuantLinear
        return QuantLinear
    except ImportError:
        pass
    import importlib.util as _ilu
    import pathlib
    import QQQ
    for base in QQQ.__path__:
        candidate = pathlib.Path(base) / "QQQ" / "gptq" / "qlinear" / "qlinear_marlin.py"
        if not candidate.exists():
            candidate = pathlib.Path(base) / "gptq" / "qlinear" / "qlinear_marlin.py"
        if candidate.exists():
            spec = _ilu.spec_from_file_location("_qlinear_marlin", str(candidate))
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.QuantLinear
    raise ImportError("Cannot locate QQQ QuantLinear")


def replace_qqq_linear_with_plugin(model: nn.Module) -> nn.Module:
    """Replace all QQQ QuantLinear modules with QqqGemmPluginModule.

    Walks the model tree, finds QuantLinear instances (identified by
    having a ``B`` buffer of dtype int32 and an ``s_channel`` buffer),
    and swaps them in-place with QqqGemmPluginModule.

    Args:
        model: PyTorch model loaded from a QQQ checkpoint.

    Returns:
        The same model with QuantLinear layers replaced.
    """
    QQQQuantLinear = _get_qqq_quantlinear_class()

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, QQQQuantLinear):
            continue
        has_group = hasattr(module, "s_group") and module.s_group.numel() > 0
        group_size = module.group_size if has_group else -1
        new_module = QqqGemmPluginModule(
            in_features=module.infeatures,
            out_features=module.outfeatures,
            group_size=group_size,
            bias=module.bias is not None,
        )
        new_module.load_from_qqq_linear(module)
        replacements.append((name, new_module))

    for name, new_module in replacements:
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, new_module)

    print(f"Replaced {len(replacements)} QQQ QuantLinear modules with QqqGemmPluginModule")
    return model
