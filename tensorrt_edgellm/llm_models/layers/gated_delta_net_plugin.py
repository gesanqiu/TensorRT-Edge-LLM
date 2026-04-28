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
"""GatedDeltaNet plugin stubs for ONNX export.

This module defines a dummy PyTorch custom op and ONNX symbolic mapping for
`trt_edgellm::gated_delta_net` so qwen3.5 export can emit the custom node
before the full linear-attention wrapper is integrated.
"""

from typing import Tuple

import torch
from onnx.defs import OpSchema
from torch.onnx import register_custom_op_symbolic, symbolic_helper

from ...common import ONNX_OPSET_VERSION

gated_delta_net_schema = OpSchema(
    name="gated_delta_net",
    domain="trt_edgellm",
    since_version=ONNX_OPSET_VERSION,
    doc="Qwen3.5 Gated Delta Net plugin with explicit recurrent state output.",
    inputs=[
        OpSchema.FormalParameter(name="q",
                                 description="Query [n, seq, h, k]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="k",
                                 description="Key [n, seq, h, k]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="v",
                                 description="Value [n, seq, hv, v]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="a",
                                 description="A gating tensor [n, seq, hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="b",
                                 description="B gating tensor [n, seq, hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="A_log",
                                 description="A_log [hv]",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="dt_bias",
                                 description="dt_bias [hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="h0_source",
            description="Recurrent state in [n, hv, k, v]",
            type_str="T_A"),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Valid token count per batch row [n]",
            type_str="T_I"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="o",
                                 description="Output [n, seq, hv, v]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="h0_out",
            description="Recurrent state out [n, hv, k, v]",
            type_str="T_A"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], ""),
        ("T_A", ["tensor(float)"], ""),
        ("T_I", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="k_dim",
                           type=OpSchema.AttrType.INT,
                           description="K head dimension",
                           required=True),
        OpSchema.Attribute(name="v_dim",
                           type=OpSchema.AttrType.INT,
                           description="V head dimension",
                           required=True),
    ],
)


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "v", "v", "v", "v", "i",
                            "i")
def symbolic_gated_delta_net(g, q, k, v, a, b, A_log, dt_bias, h0_source,
                             context_lengths, k_dim, v_dim):
    o, h0_out = g.op(
        "trt_edgellm::gated_delta_net",
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        context_lengths,
        k_dim_i=k_dim,
        v_dim_i=v_dim,
        outputs=2,
    )
    o.setType(v.type())
    h0_out.setType(h0_source.type())
    return o, h0_out


@torch.library.custom_op("trt_edgellm::gated_delta_net", mutates_args=())
def gated_delta_net_plugin(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h0_source: torch.Tensor,
    context_lengths: torch.Tensor,
    k_dim: int,
    v_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dummy op for tracing only."""
    del q, k, a, b, A_log, dt_bias, context_lengths, k_dim, v_dim
    return torch.zeros_like(v), h0_source.clone()


def register_gated_delta_net_onnx_symbolic_functions() -> None:
    register_custom_op_symbolic(
        "trt_edgellm::gated_delta_net",
        symbolic_gated_delta_net,
        ONNX_OPSET_VERSION,
    )
    print("Registered ONNX symbolic functions for GatedDeltaNet plugin")
