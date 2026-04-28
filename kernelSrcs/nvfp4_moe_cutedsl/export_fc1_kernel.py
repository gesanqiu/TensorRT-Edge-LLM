#!/usr/bin/env python3
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

"""AOT export script for the N-major contiguous grouped GEMM kernel (FC1).

Exports ``BlockScaledContiguousGroupedGemmNMajorKernel.wrapper`` with
9 pointers + 5 int64 + stream. Weight B bytes arrive in ``[L, K, N/2]``
(N innermost) — what the plugin's ``populate_prefill_plugin_buffers``
emits for ``fc_up_qweights``. The kernel performs an in-flight SMEM
nibble transpose so ``tcgen05.mma`` still sees a K-major B operand.

The kernel is compiled with ``raster_along_m=False`` (Thor-opt Phase-1
scheduling): the persistent tile scheduler walks N-first within each
M-tile, which gives ~2x FC1 speedup at Qwen3 SwiGLU T>=2048 EP=1 and
near-parity with the K-major path at Nemotron. This matches the
upstream ``cutedsl-nvfp4-moe`` repo's AOT exports, which now compile
only this scheduling variant.

Variant names: ``nvfp4_moe_fc1_{relu2,swiglu}_n{128,256}_{bf16,fp16}``.

Usage (from kernelSrcs/):
    python nvfp4_moe_cutedsl/export_fc1_kernel.py \
        --activation relu2 \
        --mma_tiler_n 128 \
        --output_dtype bf16 \
        --output_dir /tmp/staging \
        --file_name nvfp4_moe_fc1_relu2_n128_bf16 \
        --function_prefix nvfp4_moe_fc1_relu2_n128_bf16

Usage (invoked by build_cutedsl.py — PYTHONPATH set automatically).
"""

import argparse
import os
import sys

import cupy as cp


def export_contiguous_gemm_variant(args):
    """Export a single N-major contiguous grouped GEMM kernel variant."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from blockscaled_contiguous_grouped_gemm_n_major import (
        BlockScaledContiguousGroupedGemmNMajorKernel,
    )
    from common import create_dummy_pointers, get_max_active_clusters, resolve_out_dtype

    cp.cuda.Device(0).use()

    sf_vec_size = 16
    mma_tiler_mn = (128, args.mma_tiler_n)
    cluster_shape_mn = (1, 1)
    activation = args.activation
    out_dtype = resolve_out_dtype(args.output_dtype)
    verbose = getattr(args, "verbose", False)

    print(f"N-major contiguous GEMM variant: activation={activation}, "
          f"mma_tiler_mn={mma_tiler_mn}, output_dtype={args.output_dtype}")

    # Thor-opt: walk N-first within each M-tile. See module docstring.
    gemm = BlockScaledContiguousGroupedGemmNMajorKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        activation=activation,
        raster_along_m=False,
    )

    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1])

    # ---- Create typed pointers from uint8-backed GPU buffers ----
    is_swiglu = activation == "swiglu"
    dummy_n = 1856
    dummy_n_out = dummy_n // 2 if is_swiglu else dummy_n

    ptrs, _bufs = create_dummy_pointers(
        sf_vec_size=sf_vec_size,
        dummy_m=128, dummy_n=dummy_n, dummy_k=2688, dummy_l=16,
        is_swiglu=is_swiglu,
        out_dtype=out_dtype)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    print("Compiling contiguous grouped GEMM kernel via wrapper...")
    compiled = cute.compile(
        gemm.wrapper,
        ptrs["a_ptr"], ptrs["b_ptr"], ptrs["a_sf_ptr"], ptrs["b_sf_ptr"],
        ptrs["c_ptr"], ptrs["alpha_ptr"],
        ptrs["tile_group_ptr"], ptrs["tile_mn_ptr"], ptrs["num_tiles_ptr"],
        128, dummy_n, dummy_n_out, 2688, 16,
        tile_size=128,
        scaling_vector_size=sf_vec_size,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Exporting to {args.output_dir}/{args.file_name}.[h|o]")
    compiled.export_to_c(
        file_path=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )

    # Verify output files
    header = os.path.join(args.output_dir, f"{args.file_name}.h")
    obj = os.path.join(args.output_dir, f"{args.file_name}.o")
    ok = True
    for path, label in [(header, "Header"), (obj, "Object")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            ok = False
        elif os.path.getsize(path) == 0:
            print(f"ERROR: {label} is empty: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)

    print(f"  header: {os.path.getsize(header)} bytes")
    print(f"  object: {os.path.getsize(obj)} bytes")

    if verbose:
        print("\n--- Generated Header (first 100 lines) ---")
        with open(header) as f:
            for i, line in enumerate(f):
                if i >= 100:
                    print("  ... (truncated)")
                    break
                print(f"  {line.rstrip()}")

    return header, obj


def main():
    parser = argparse.ArgumentParser(
        description="Export contiguous grouped GEMM kernel (FC1) for AOT compilation"
    )
    parser.add_argument(
        "--activation", type=str, default="identity",
        choices=["identity", "relu2", "swiglu"],
        help="Activation function (default: identity for decomposed pipeline)"
    )
    parser.add_argument(
        "--mma_tiler_n", type=int, required=True,
        choices=[128, 256],
        help="N-tile size for MMA (128 or 256)"
    )
    parser.add_argument(
        "--output_dtype", type=str, default="bf16",
        choices=["bf16", "fp16"],
        help="Output element type (default: bf16)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write .h and .o files"
    )
    parser.add_argument(
        "--file_name", type=str, required=True,
        help="Base name for output files (without extension)"
    )
    parser.add_argument(
        "--function_prefix", type=str, required=True,
        help="C function name prefix"
    )
    parser.add_argument(
        "--export_only", action="store_true",
        help="Accepted for build_cutedsl.py compatibility (no-op)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print generated header content"
    )
    args = parser.parse_args()

    export_contiguous_gemm_variant(args)
    print("\nContiguous grouped GEMM kernel export completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
