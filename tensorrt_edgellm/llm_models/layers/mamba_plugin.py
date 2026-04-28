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
Mamba Plugin Operations for TensorRT ONNX Export

Provides dummy PyTorch custom ops and ONNX symbolic functions for:
  - trt_edgellm::causal_conv1d: depthwise causal 1D convolution
  - trt_edgellm::update_ssm_state: Mamba selective state space model update

These are used during ONNX tracing to produce custom ONNX ops that map
to the C++ TensorRT plugins (CausalConv1dPlugin, MambaPlugin).
"""

from typing import Tuple

import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper

from ...common import ONNX_OPSET_VERSION

causal_conv1d_schema = OpSchema(
    name="causal_conv1d",
    domain="trt_edgellm",
    since_version=ONNX_OPSET_VERSION,
    doc="Custom causal 1D depthwise convolution plugin with persistent state.",
    inputs=[
        OpSchema.FormalParameter(name="x",
                                 description="Input tensor",
                                 type_str="T"),
        OpSchema.FormalParameter(name="weight",
                                 description="Conv weight",
                                 type_str="T"),
        OpSchema.FormalParameter(name="bias",
                                 description="Conv bias",
                                 type_str="T"),
        OpSchema.FormalParameter(name="conv_state",
                                 description="Conv state [batch, dim, kernel]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Per-batch actual token count [batch]",
            type_str="T_CL"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="Conv output",
                                 type_str="T"),
        OpSchema.FormalParameter(name="conv_state_out",
                                 description="Updated conv state",
                                 type_str="T"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="stride",
                           type=OpSchema.AttrType.INT,
                           description="Stride",
                           required=True),
        OpSchema.Attribute(name="padding",
                           type=OpSchema.AttrType.INT,
                           description="Padding",
                           required=True),
        OpSchema.Attribute(name="dilation",
                           type=OpSchema.AttrType.INT,
                           description="Dilation",
                           required=True),
        OpSchema.Attribute(name="groups",
                           type=OpSchema.AttrType.INT,
                           description="Groups",
                           required=True),
    ],
)

update_ssm_state_schema = OpSchema(
    name="update_ssm_state",
    domain="trt_edgellm",
    since_version=ONNX_OPSET_VERSION,
    doc="Mamba selective state space model update plugin.",
    inputs=[
        OpSchema.FormalParameter(name="x", description="Input x",
                                 type_str="T"),
        OpSchema.FormalParameter(name="A",
                                 description="A parameter",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="B",
                                 description="B parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="C",
                                 description="C parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="D",
                                 description="D parameter",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="dt",
                                 description="dt parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="dt_bias",
                                 description="dt_bias parameter",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="state",
                                 description="SSM state",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Per-batch actual token count [batch]",
            type_str="T_CL"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="SSM output",
                                 type_str="T"),
        OpSchema.FormalParameter(name="state_out",
                                 description="Updated SSM state",
                                 type_str="T"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
        ("T_A", ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="dt_softplus",
                           type=OpSchema.AttrType.INT,
                           description="Apply softplus to dt",
                           required=True),
        OpSchema.Attribute(name="ngroups",
                           type=OpSchema.AttrType.INT,
                           description="Number of groups",
                           required=True),
    ],
)

# ---------------------------------------------------------------------------
# Causal Conv1d
# ---------------------------------------------------------------------------


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i")
def symbolic_causal_conv1d(g, x, weight, bias, conv_state, context_lengths,
                           stride, padding, dilation, groups):
    """Map trt_edgellm::causal_conv1d to ONNX custom op."""
    output, conv_state_out = g.op(
        "trt_edgellm::causal_conv1d",
        x,
        weight,
        bias,
        conv_state,
        context_lengths,
        stride_i=stride,
        padding_i=padding,
        dilation_i=dilation,
        groups_i=groups,
        outputs=2,
    )
    output.setType(x.type())
    conv_state_out.setType(conv_state.type())
    return output, conv_state_out


@torch.library.custom_op("trt_edgellm::causal_conv1d", mutates_args=())
def causal_conv1d_plugin(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    context_lengths: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dummy causal conv1d for ONNX tracing. Not used at runtime."""
    return torch.zeros_like(x), conv_state.clone()


# ---------------------------------------------------------------------------
# Mamba SSM (selective_state_update)
# ---------------------------------------------------------------------------


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "v", "v", "i",
                            "i")
def symbolic_update_ssm_state(g, x, A, B, C, D, dt, dt_bias, state,
                              context_lengths, dt_softplus, ngroups):
    """Map trt_edgellm::update_ssm_state to ONNX custom op."""
    output, state_out = g.op(
        "trt_edgellm::update_ssm_state",
        x,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        context_lengths,
        dt_softplus_i=dt_softplus,
        ngroups_i=ngroups,
        outputs=2,
    )
    output.setType(x.type())
    state_out.setType(state.type())
    return output, state_out


@torch.library.custom_op("trt_edgellm::update_ssm_state", mutates_args=())
def update_ssm_state_plugin(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    context_lengths: torch.Tensor,
    dt_softplus: int,
    ngroups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dummy SSM update for ONNX tracing. Not used at runtime."""
    output = torch.zeros_like(x)
    state_out = state.clone()
    return output, state_out


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_mamba_plugin_onnx_symbolic_functions() -> None:
    """Register ONNX symbolic functions for Mamba custom ops."""
    register_custom_op_symbolic(
        "trt_edgellm::causal_conv1d",
        symbolic_causal_conv1d,
        ONNX_OPSET_VERSION,
    )
    register_custom_op_symbolic(
        "trt_edgellm::update_ssm_state",
        symbolic_update_ssm_state,
        ONNX_OPSET_VERSION,
    )
    print("Registered ONNX symbolic functions for Mamba plugins")
