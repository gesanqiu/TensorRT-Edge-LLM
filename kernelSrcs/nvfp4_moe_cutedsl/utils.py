# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is copied and modified from cutlass https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py

import ctypes
from typing import Union

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir.dialects import _nvvm_enum_gen as nvvm_enums
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type
from cutlass.cutlass_dsl import T, dsl_user_op


# WAR for CuTeDSL make_ptr implementation
class _Pointer(Pointer):
    """Represents a runtime pointer that can interoperate with various data structures,
    including numpy arrays and device memory.

    Args:
        pointer (int or pointer-like object): The pointer to the data.
        dtype (Type): Data type of the elements pointed to.
        mem_space (_cute_ir.AddressSpace, optional): Memory space where the pointer resides. Defaults to generic.
        assumed_align (int, optional): Alignment of the input pointer in bytes. Defaults to None.

    Attributes:
        _pointer: The underlying pointer.
        _dtype: Data type of the elements.
        _addr_space: Memory space of the pointer.
        _assumed_align: Alignment of the pointer in bytes.
        _desc: C-type descriptor for the pointer.
        _c_pointer: C-compatible pointer representation.
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned")

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(self._dtype.mlir_type, self._addr_space,
                                    self._assumed_align)

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        if expected_py_type is Pointer or (isinstance(
                expected_py_type, ir.Value) and expected_py_type.ty is Pointer):
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Creates a pointer from a memory address.

    Args:
        dtype (Type[Numeric]): Data type of the pointer elements.
        value (Union[int, ctypes._Pointer]): Memory address as an integer or ctypes pointer.
        mem_space (AddressSpace, optional): Memory address space. Defaults to AddressSpace.generic.
        assumed_align (int, optional): Alignment in bytes. Defaults to None.

    Returns:
        Pointer: A pointer object.

    Example:
        ```python
        import numpy as np
        import ctypes
        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)
        # Get pointer address as ctypes pointer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)
        ```
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}")

    return _Pointer(address_value,
                    dtype,
                    mem_space,
                    assumed_align=assumed_align)


def is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@dsl_user_op
def fmax(a: Union[float, cutlass.Float32],
         b: Union[float, cutlass.Float32],
         *,
         nan=False,
         loc=None,
         ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmax(
            T.f32(),
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        ))


def sigmoid_f32(a: Union[float, cutlass.Float32],
                fastmath: bool = False) -> Union[float, cutlass.Float32]:
    """
    Compute the sigmoid of the input tensor.
    """
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(a: Union[float, cutlass.Float32],
             fastmath: bool = False) -> Union[float, cutlass.Float32]:
    """
    Compute the silu of the input tensor.
    """
    return a * sigmoid_f32(a, fastmath=fastmath)


# TODO(zhichenj): try to move these to NVVM wrapper or helper functions
@dsl_user_op
def vectorized_atomic_add_bf16x8(rOut_epi_packed,
                                 scatter_out_offset,
                                 loc=None,
                                 ip=None):
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            llvm.bitcast(T.i32(), rOut_epi_packed[0, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[1, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[2, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[3, None].load().ir_value()),
        ],
        "red.global.v4.bf16x2.add.noftz [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def vectorized_atomic_add_fp16x8(rOut_epi_packed,
                                 scatter_out_offset,
                                 loc=None,
                                 ip=None):
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            llvm.bitcast(T.i32(), rOut_epi_packed[0, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[1, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[2, None].load().ir_value()),
            llvm.bitcast(T.i32(), rOut_epi_packed[3, None].load().ir_value()),
        ],
        "red.global.v4.f16x2.add.noftz [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def vectorized_atomic_add_fp32x2(rOut_epi_packed,
                                 scatter_out_offset,
                                 loc=None,
                                 ip=None):
    llvm.inline_asm(
        None,
        [
            scatter_out_offset.iterator.llvm_ptr,
            rOut_epi_packed[0].ir_value(),
            rOut_epi_packed[1].ir_value(),
        ],
        "red.global.v2.f32.add [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_func(rOut_epi_packed, scatter_out_offset, loc=None, ip=None):
    if cutlass.const_expr(rOut_epi_packed.dtype == cutlass.Float32):
        llvm.inline_asm(
            None,
            [
                scatter_out_offset.iterator.llvm_ptr,
                rOut_epi_packed.ir_value(),
            ],
            "red.global.add.f32 [$0], $1;",
            "l,f",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    elif cutlass.const_expr(rOut_epi_packed.dtype == cutlass.BFloat16):
        llvm.inline_asm(
            None,
            [
                scatter_out_offset.iterator.llvm_ptr,
                llvm.bitcast(T.i16(), rOut_epi_packed.ir_value()),
            ],
            "red.add.noftz.bf16 [$0], $1;",
            "l,h",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp32(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp16(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None) -> None:
    """
    This instruction is used to wait for the previous kernel's grid ending
    (all blocks of the previous kernel have finished and memflushed), i.e.,
    the instruction after this instruction will not be issued until the previous
    grid has finished.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.wait;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None) -> None:
    """
    Issuing the launch_dependents instruction hints a dependent kernel to launch earlier.
    launch_dependents doesn't impact the functionality but the performance:
    Launching a dependent kernel too early can compete with current kernels,
    while launching too late can lead to a long latency.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.launch_dependents;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# N-major transpose warp primitives (FC1 in-flight SMEM nibble transpose).
# Used by ``blockscaled_contiguous_grouped_gemm_n_major`` and
# ``blockscaled_contiguous_grouped_gemm_finalize_n_major``.
# ---------------------------------------------------------------------------

def prmt_b32(a_u32, b_u32, sel_u32, *, loc=None, ip=None):
    """Issue ``prmt.b32 d, a, b, sel``.

    ``d`` bytes [j] = source_byte[sel_j] of the concatenation of ``a`` and
    ``b`` (a's bytes 0..3 then b's bytes 0..3). ``sel``'s low 16 bits hold
    four 4-bit selectors s_0..s_3.
    """
    return llvm.inline_asm(
        T.i32(),
        [
            a_u32.ir_value() if hasattr(a_u32, "ir_value") else a_u32,
            b_u32.ir_value() if hasattr(b_u32, "ir_value") else b_u32,
            sel_u32.ir_value() if hasattr(sel_u32, "ir_value") else sel_u32,
        ],
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_b16(dst_sub, val_u16, *, loc=None, ip=None):
    """Issue a 16-bit aligned SMEM store (``st.shared.b16``)."""
    llvm.inline_asm(
        None,
        [
            dst_sub.iterator.llvm_ptr,
            val_u16.ir_value() if hasattr(val_u16, "ir_value") else val_u16,
        ],
        """{
            .reg .u64 addr_u64;
            .reg .u32 addr_u32;
            .reg .u16 val_u16;
            cvta.to.shared.u64 addr_u64, $0;
            cvt.u32.u64 addr_u32, addr_u64;
            cvt.u16.u32 val_u16, $1;
            st.shared.b16 [addr_u32], val_u16;
        }""",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldmatrix_m16n16_trans_b8(smem_ptr, *, loc=None, ip=None):
    """Issue ``ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8``.

    Reads a 16 row × 16 col u8 tile (= 256 bytes) from SMEM with hardware
    transpose. Returns ``(r0, r1)`` — two Uint32 registers per lane.
    """
    src_ptr_val = (
        smem_ptr.ir_value() if hasattr(smem_ptr, "ir_value") else smem_ptr
    )
    shared_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    shared_ptr = llvm.addrspacecast(
        shared_ptr_ty, src_ptr_val, loc=loc, ip=ip
    )
    struct_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    packed = nvvm.ldmatrix(
        res=struct_ty,
        ptr=shared_ptr,
        num=2,  # MLIR num=2 -> PTX .x1 for this srcFormat
        layout=nvvm_enums.MMALayout.col,
        shape=nvvm_enums.LoadShape.M16N16,
        src_format=nvvm_enums.LoadSrcFormat.B8,
        loc=loc,
        ip=ip,
    )
    r0 = cutlass.Uint32(llvm.extractvalue(T.i32(), packed, [0]))
    r1 = cutlass.Uint32(llvm.extractvalue(T.i32(), packed, [1]))
    return r0, r1


@dsl_user_op
def st_shared_b32(dst_sub, val_u32, *, loc=None, ip=None):
    """Issue a 32-bit aligned SMEM store (``st.shared.b32``)."""
    llvm.inline_asm(
        None,
        [
            dst_sub.iterator.llvm_ptr,
            val_u32.ir_value() if hasattr(val_u32, "ir_value") else val_u32,
        ],
        """{
            .reg .u64 addr_u64;
            .reg .u32 addr_u32;
            cvta.to.shared.u64 addr_u64, $0;
            cvt.u32.u64 addr_u32, addr_u64;
            st.shared.b32 [addr_u32], $1;
        }""",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
