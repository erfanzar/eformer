# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive

from eformer.jaximus import ImplicitArray, aux_field, register, ste

from . import quantization_functions as _quantization_impl
from .quantization_functions import bmm_nf4, bmm_nf4_transpose, dequantize_nf4, nf4_matmul, quantize_and_pack_nf4

Array = jax.Array


@dataclass
class ArrayNF4(ImplicitArray):
    """
    4-bit NormalFloat Quantization Class

    This class implements 4-bit NormalFloat (NF4) quantization for arrays. It quantizes the input array into 4-bit
    integers and stores the absolute maximum values for each block. The original array can be reconstructed using the
    stored packed data and absolute maximum values.

    Attributes:
        packed (jax.Array): The packed 4-bit integer array.
        absmax (jax.Array): The absolute maximum values for each block.
        block_size (int): The size of each quantization block (static).
        mesh_and_axis (Optional[Tuple]): Sharding configuration for distributed operations (static).

    Methods:
        quantize(array, block_size): Creates a quantized ArrayNF4 from input array.
        materialize(): Reconstructs the original array from the quantized data.
    """

    packed: Array
    absmax: Array
    block_size: int = aux_field()
    mesh_and_axis: tuple[jax.sharding.Mesh, int | None] | None = aux_field(default=None)

    @classmethod
    def quantize(
        cls,
        array: Array,
        block_size: int = 64,
        verbose=False,
    ):
        """
        Initializes the `ArrayNF4` object by quantizing the input array.

        Args:
            array (jax.Array): The input array to be quantized.
            block_size (int): The size of each quantization block. Defaults to 64.
            verbose (bool): Print verbose information. Defaults to False.
        """
        # Ensure last dimension is divisible by block_size
        if array.shape[-1] % block_size != 0:
            # Pad last dimension to be divisible by block_size
            pad_width = [(0, 0)] * (array.ndim - 1) + [(0, block_size - array.shape[-1] % block_size)]
            array = jnp.pad(array, pad_width, mode="constant", constant_values=0)

        # Quantize along last dimension only - preserves structure
        packed, absmax = quantize_and_pack_nf4(array, block_size)
        return cls(
            packed=packed,
            absmax=absmax,
            block_size=block_size,
            shape=array.shape,
            dtype=array.dtype,
            mesh_and_axis=None,
        )

    def materialize(self):
        """
        Reconstructs the original array from the quantized data.

        Returns:
            jax.Array: The dequantized array.
        """

        return (
            dequantize_nf4(
                self.packed.astype(jnp.uint8),
                self.absmax,
                self.block_size,
            )
            .reshape(self.shape)
            .astype(self.dtype)
        )

    def dequantize(self):
        """Alias for materialize() for compatibility."""
        return self.materialize()

    def with_mesh_and_axis(self, mesh_and_axis):
        """
        Configure sharding for distributed operations.

        Args:
            mesh_and_axis: Tuple of (mesh, axis) for sharding configuration

        Returns:
            ArrayNF4 with updated sharding configuration
        """
        import dataclasses

        # Simply update the mesh_and_axis attribute without applying sharding
        # The sharding will be applied when the array is actually used
        new_quant_matrix = dataclasses.replace(self, mesh_and_axis=mesh_and_axis)
        return new_quant_matrix

    def delete(self):
        self.packed.delete()
        self.absmax.delete()


ArrayType = Array | ArrayNF4 | Any


@register("convert_element_type")
def _(primitive: Primitive, operand: ArrayType, new_dtype: Any) -> ArrayType:
    if isinstance(operand, ArrayNF4):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("lt")
def _(primitive: Primitive, x: ArrayType, y: ArrayType, **kwargs):
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
        y = y.materialize()
    return jax.lax.lt(x, y, **kwargs)


@register("convert_element_type")
def _(primitive: Primitive, operand: ArrayType, **kwargs) -> ArrayType:
    new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
    if isinstance(operand, ArrayNF4):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def _(primitive: Primitive, x: Any, y: Any) -> Any:
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
        y = y.materialize()
    return lax.pow(x, y)


@register("integer_pow")
def _(primitive: Primitive, x: Any, **kwargs) -> Any:
    y = kwargs.get("y", 2)
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    return lax.pow(x, y)


@register("div")
def _(primitive: Primitive, x: ArrayType, y: ArrayType) -> Any:
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
        y = y.materialize()
    return lax.div(x, y)


@register("sqrt")
def _(primitive: Primitive, x: ArrayNF4) -> Any:
    x = x.materialize()
    return lax.sqrt(x)


@register("convert_element_type")
def convert_element_type_nf4_operand_pos(primitive: Primitive, operand: ArrayType, new_dtype: Any) -> ArrayType:
    if isinstance(operand, ArrayNF4):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("lt")
def lt_nf4_xy(primitive: Primitive, x: ArrayType, y: ArrayType, **kwargs):
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
        y = y.materialize()
    return jax.lax.lt(x, y, **kwargs)


@register("convert_element_type")
def convert_element_type_nf4_operand_kw(primitive: Primitive, operand: ArrayType, **kwargs) -> ArrayType:
    new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
    if isinstance(operand, ArrayNF4):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def integer_pow_nf4_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> Any:
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
        y = y.materialize()
    return lax.pow(x, y)


@register("integer_pow")
def integer_pow_nf4_x(primitive: Primitive, x: ArrayType, **kwargs) -> Any:
    y = kwargs.get("y", 2)
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    return lax.pow(x, y)


@register("div")
def div_nf4_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> Any:
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
        y = y.materialize()
    return lax.div(x, y)


@register("sqrt")
def sqrt_nf4_x(primitive: Primitive, x: ArrayNF4) -> Any:
    x = x.materialize()
    return lax.sqrt(x)


def safe_materialize(arr: ArrayType) -> tuple[ArrayType, bool]:
    """Safely materialize an array if it's ArrayNF4."""
    if isinstance(arr, ArrayNF4):
        materialized_arr = arr.materialize()
        return materialized_arr, True
    return arr, False


def safe_delete(arr: ArrayType, materialized: bool) -> None:
    """Safely delete an array if it was materialized."""

    if materialized:
        pass


def _nf4_kernels_enabled() -> bool:
    """Return True if NF4 kernels are enabled on the current device."""

    try:
        return _quantization_impl._get_kernel_state()
    except AttributeError:
        return False


def _unpack_nf4_codes(packed: jax.Array, block_size: int) -> jax.Array | None:
    """Unpack packed NF4 values (two nibbles per byte) into individual codes."""

    if packed.ndim < 2:
        return None
    num_blocks = packed.shape[-2]
    half_block = packed.shape[-1]
    if half_block * 2 != block_size:
        return None
    packed_u32 = packed.astype(jnp.uint32)
    high = ((packed_u32 >> 4) & jnp.uint32(0xF)).astype(jnp.uint8)
    low = (packed_u32 & jnp.uint32(0xF)).astype(jnp.uint8)
    codes = jnp.stack((high, low), axis=-1)
    codes = codes.reshape(*packed.shape[:-2], num_blocks, block_size)
    return codes.reshape(*packed.shape[:-2], num_blocks * block_size)


def _expand_nf4_absmax(absmax: jax.Array, block_size: int) -> jax.Array | None:
    """Broadcast absmax values so every NF4 element has a matching scale."""

    if absmax.ndim < 1:
        return None
    expanded = jnp.broadcast_to(absmax[..., None], (*absmax.shape, block_size))
    return expanded.reshape(*absmax.shape[:-1], absmax.shape[-1] * block_size)


def _prepare_nf4_kernel_tensors(weight: ArrayNF4, *, transpose: bool = False) -> tuple[jax.Array, jax.Array] | None:
    """Convert packed NF4 weights into the tensors expected by the TPU kernels."""

    packed = weight.packed
    absmax = weight.absmax
    block_size = weight.block_size

    if weight.shape is None or len(weight.shape) != 2:
        return None

    # Only handle 2D weights (packed.ndim == 3 after quantization)
    if packed.ndim != 3 or absmax.ndim != 2:
        return None
    if absmax.shape != packed.shape[:-1]:
        return None

    unpacked = _unpack_nf4_codes(packed.astype(jnp.uint8), block_size)
    scales = _expand_nf4_absmax(absmax.astype(jnp.float32), block_size)
    if unpacked is None or scales is None:
        return None

    if unpacked.shape != weight.shape:
        return None

    if transpose:
        unpacked = jnp.swapaxes(unpacked, -1, -2)
        scales = jnp.swapaxes(scales, -1, -2)

    dim0, dim1 = unpacked.shape
    quants = unpacked.reshape(dim0, 1, dim1).astype(jnp.uint8)
    scales = scales.reshape(dim0, 1, dim1).astype(jnp.float32)
    return quants, scales


def _flatten_inputs_for_kernel(lhs: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    """Reshape lhs to (batch, k) for kernel consumption."""

    if lhs.ndim == 0:
        raise ValueError("lhs must have at least 1 dimension for matmul")
    batch_shape = tuple(lhs.shape[:-1])
    k = lhs.shape[-1]
    batch_size = int(math.prod(batch_shape)) if batch_shape else 1
    inputs = lhs.reshape((batch_size, k))
    return inputs, batch_shape


def _move_axis_to_last(arr: jax.Array, axis: int) -> jax.Array:
    """Move the specified axis to the last dimension."""

    axis = axis % arr.ndim
    if axis == arr.ndim - 1:
        return arr
    return jnp.moveaxis(arr, axis, -1)


def _kernel_rhs_matmul(
    lhs: jax.Array,
    rhs: ArrayNF4,
    dimension_numbers: tuple[Any, Any],
) -> jax.Array | None:
    """Attempt to run the TPU kernel when rhs is quantized."""

    if not _nf4_kernels_enabled():
        return None

    try:
        (lhs_contract, rhs_contract), (lhs_batch_dims, rhs_batch_dims) = dimension_numbers
    except (TypeError, ValueError):
        return None

    if lhs_batch_dims or rhs_batch_dims:
        return None
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        return None

    lhs_contract_axis = lhs_contract[0]
    rhs_contract_axis = rhs_contract[0]

    if lhs_contract_axis != lhs.ndim - 1:
        return None
    if rhs.ndim != 2 or rhs_contract_axis != 0:
        return None
    if lhs.shape[lhs_contract_axis] != rhs.shape[rhs_contract_axis]:
        return None

    tensors = _prepare_nf4_kernel_tensors(rhs)
    if tensors is None:
        return None

    lhs_2d, batch_shape = _flatten_inputs_for_kernel(lhs)
    outputs = nf4_matmul(lhs_2d, *tensors, kernel=bmm_nf4)
    out_shape = (*batch_shape, rhs.shape[1])
    rhs_dtype = rhs.dtype or lhs.dtype
    result_dtype = jnp.result_type(lhs.dtype, rhs_dtype)
    return outputs.reshape(out_shape).astype(result_dtype)


def _kernel_lhs_matmul(
    lhs: ArrayNF4,
    rhs: jax.Array,
    dimension_numbers: tuple[Any, Any],
) -> jax.Array | None:
    """Attempt to run the TPU kernel when lhs is quantized."""

    if not _nf4_kernels_enabled():
        return None

    try:
        (lhs_contract, rhs_contract), (lhs_batch_dims, rhs_batch_dims) = dimension_numbers
    except (TypeError, ValueError):
        return None

    if lhs_batch_dims or rhs_batch_dims:
        return None
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        return None

    lhs_contract_axis = lhs_contract[0]
    rhs_contract_axis = rhs_contract[0]

    if lhs_contract_axis != lhs.ndim - 1:
        return None
    if rhs_contract_axis < 0:
        rhs_contract_axis += rhs.ndim
    if rhs_contract_axis != 0:
        return None

    if lhs.shape is None or len(lhs.shape) != 2:
        return None
    if lhs.shape[lhs_contract_axis] != rhs.shape[rhs_contract_axis]:
        return None

    tensors = _prepare_nf4_kernel_tensors(lhs, transpose=True)
    if tensors is None:
        return None

    rhs_reordered = _move_axis_to_last(rhs, rhs_contract_axis)
    rhs_2d, rhs_batch_shape = _flatten_inputs_for_kernel(rhs_reordered)

    outputs = nf4_matmul(rhs_2d, *tensors, kernel=bmm_nf4_transpose, backward=True)

    lhs_rows = lhs.shape[0]
    result = outputs.reshape(*rhs_batch_shape, lhs_rows)
    if rhs_batch_shape:
        result = jnp.moveaxis(result, -1, 0)
        result = result.reshape((lhs_rows, *rhs_batch_shape))
    else:
        result = result.reshape((lhs_rows,))

    lhs_dtype = lhs.dtype or rhs.dtype
    result_dtype = jnp.result_type(lhs_dtype, rhs.dtype)
    return result.astype(result_dtype)


def _maybe_kernel_dot_general(
    lhs: ArrayType,
    rhs: ArrayType,
    dimension_numbers: tuple[Any, Any] | None,
    preferred_element_type: Any,
) -> jax.Array | None:
    """Dispatch to TPU kernels when applicable."""

    if preferred_element_type is not None or dimension_numbers is None:
        return None

    if isinstance(rhs, ArrayNF4) and not isinstance(lhs, ArrayNF4) and isinstance(lhs, jax.Array):
        return _kernel_rhs_matmul(lhs, rhs, dimension_numbers)

    if isinstance(lhs, ArrayNF4) and not isinstance(rhs, ArrayNF4) and isinstance(rhs, jax.Array):
        return _kernel_lhs_matmul(lhs, rhs, dimension_numbers)

    return None


@register("dot_general")
def dot_general_nf4_lhs_rhs(
    primitive: Primitive,
    lhs: ArrayType,
    rhs: ArrayType,
    *args: Any,
    **kwargs: Any,
) -> ArrayType:
    """
    Custom handler for JAX's dot_general operation.

    Supports both kernel-based and materialization-based execution.

    Args:
      primitive: The JAX primitive being handled.
      lhs: Left-hand side array.
      rhs: Right-hand side array.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.dot_general operation.
    """
    dimension_numbers = args[0] if args else kwargs.get("dimension_numbers")
    precision = args[1] if len(args) > 1 else kwargs.get("precision")
    preferred_element_type = args[2] if len(args) > 2 else kwargs.get("preferred_element_type")

    if precision is None:
        kernel_result = _maybe_kernel_dot_general(lhs, rhs, dimension_numbers, preferred_element_type)
        if kernel_result is not None:
            return kernel_result

    # Fallback to materialization
    lhs_mat, _lhs_materialized = safe_materialize(lhs)
    rhs_mat, _rhs_materialized = safe_materialize(rhs)

    res = lax.dot_general(lhs_mat, rhs_mat, *args, **kwargs)

    return res


@register("add")
def add_nf4_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    """
    Custom handler for JAX's add operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      x: First array to add.
      y: Second array to add.

    Returns:
      The result of lax.add operation.
    """
    x_mat, _x_materialized = safe_materialize(x)
    y_mat, _y_materialized = safe_materialize(y)

    result = lax.add(x_mat, y_mat)
    return result


@register("reduce")
def reduce_nf4_operand_init_value(
    primitive: Primitive,
    operand: ArrayType,
    init_value: ArrayType,
    *args: Any,
    **kwargs: Any,
) -> ArrayType:
    """
    Custom handler for JAX's reduce operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      operand: The array to be reduced.
      init_value: The initial value for the reduction.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.reduce operation.
    """
    operand_mat, _operand_materialized = safe_materialize(operand)
    init_value_mat, _init_value_materialized = safe_materialize(init_value)

    result = lax.reduce(operand_mat, init_value_mat, *args, **kwargs)

    return result


@register("mul")
def mul_nf4_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    """
    Custom handler for JAX's mul operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      x: First array to multiply.
      y: Second array to multiply.

    Returns:
      The result of lax.mul operation.
    """
    x_mat, _x_materialized = safe_materialize(x)
    y_mat, _y_materialized = safe_materialize(y)

    result = lax.mul(x_mat, y_mat)
    return result


@register("transpose")
def transpose_nf4_operand(primitive: Primitive, operand: ArrayNF4, *args: Any, **kwargs: Any) -> ArrayType:
    """
    Custom handler for JAX's transpose operation.

    Materializes ArrayNF4 input before performing the operation.
    Re-quantizes the result if the input was ArrayNF4. Note: Original code didn't re-quantize transpose, corrected here
        based on reshape pattern.

    Args:
      primitive: The JAX primitive being handled.
      operand: The array to be transposed.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.transpose operation, potentially re-quantized.
    """

    array = operand.materialize()
    result_mat = lax.transpose(array, *args, **kwargs)
    result = ArrayNF4.quantize(result_mat, block_size=operand.block_size)
    return result


@register("conv_general_dilated")
def conv_general_dilated_nf4_lhs_rhs(
    primitive: Primitive,
    lhs: ArrayType,
    rhs: ArrayType,
    *args: Any,
    **kwargs: Any,
) -> ArrayType:
    """
    Custom handler for JAX's conv_general_dilated operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      lhs: Left-hand side array (input).
      rhs: Right-hand side array (kernel).
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.conv_general_dilated operation.
    """
    lhs_mat, _lhs_materialized = safe_materialize(lhs)
    rhs_mat, _rhs_materialized = safe_materialize(rhs)

    result = lax.conv_general_dilated(lhs_mat, rhs_mat, *args, **kwargs)

    return result


@register("max")
def max_nf4_xy(primitive: Primitive, x: ArrayType, y: ArrayType, *args: Any, **kwargs: Any) -> ArrayType:
    """
    Custom handler for JAX's max operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      x: First array for max comparison.
      y: Second array for max comparison.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.max operation.
    """
    x_mat, _x_materialized = safe_materialize(x)
    y_mat, _y_materialized = safe_materialize(y)

    result = lax.max(x_mat, y_mat, *args, **kwargs)

    return result


@register("exp")
def exp_nf4_x(primitive: Primitive, x: ArrayNF4, *args: Any, **kwargs: Any) -> ArrayType:
    """
    Custom handler for JAX's exp operation.

    Materializes ArrayNF4 input before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      x: The array to apply exponential to.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.exp operation.
    """
    x_mat, _x_materialized = safe_materialize(x)

    result = lax.exp(x_mat, *args, **kwargs)

    return result


@register("log")
def log_nf4_x(primitive: Primitive, x: ArrayNF4, **kwargs: Any) -> jnp.ndarray:
    """
    Custom handler for JAX's log operation.

    This function computes the natural logarithm of the input, handling both
    regular arrays and ArrayNF4 quantized arrays.

    Args:
      primitive: The JAX primitive being handled.
      x: The array to apply logarithm to. (Must be ArrayNF4 for this registration)
      **kwargs: Additional keyword arguments for the log operation.

    Returns:
      The result of the natural logarithm operation.

    Raises:
      RuntimeError: If the log operation fails.
    """
    x_mat, _x_materialized = safe_materialize(x)

    result = lax.log(x_mat, **kwargs)

    return result


@register("reshape")
def reshape_nf4_operand(primitive: Primitive, operand: ArrayNF4, *args, **params) -> ArrayType:
    """
    Custom handler for JAX's reshape operation.

    This function handles reshaping for ArrayNF4 quantized arrays.
    It materializes ArrayNF4 input before reshaping and re-quantizes the result.

    Args:
      primitive: The JAX primitive being handled.
      operand: The ArrayNF4 array to be reshaped.
      *args: Positional arguments for reshape (e.g., new_sizes, dimensions).
      **params: Keyword arguments/parameters for reshape.

    Returns:
      The reshaped array, re-quantized as ArrayNF4.

    Raises:
      ValueError: If the new shape is not compatible with the original array's size.
    """
    array = operand.materialize()

    subfuns, bind_params = primitive.get_bind_params(params)

    result_mat = primitive.bind(*subfuns, array, *args, **bind_params)

    result = ArrayNF4.quantize(result_mat, block_size=operand.block_size)

    return result


@register("concatenate")
def concatenate_nf4_operands(
    primitive: Primitive, operands: Sequence[ArrayType], *args: Any, **kwargs: Any
) -> ArrayType:
    """
    Custom handler for JAX's concatenate operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      operands: Sequence of arrays to concatenate.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.concatenate operation.
    """
    materialized_operands = []

    for op in operands:
        mat_op, _ = safe_materialize(op)
        materialized_operands.append(mat_op)

    result = lax.concatenate(materialized_operands, *args, **kwargs)
    return result


@register("broadcast_in_dim")
def broadcast_in_dim_nf4_operand(primitive: Primitive, operand: ArrayNF4, *args, **params) -> ArrayType:
    """Handle broadcast_in_dim for ArrayNF4."""
    array = operand.materialize()
    subfuns, bind_params = primitive.get_bind_params(params)

    result_mat = primitive.bind(*subfuns, array, *args, **bind_params)

    result = ArrayNF4.quantize(result_mat, block_size=operand.block_size)

    return result


@register("gather")
def gather_nf4_operand(primitive: Primitive, operand: ArrayNF4, *args: Any, **kwargs: Any) -> ArrayType:
    """Handle gather for ArrayNF4."""
    operand_mat, _operand_materialized = safe_materialize(operand)
    result = jax.lax.gather(operand_mat, *args, **kwargs)
    return result


@ste
def straight_through_nf4(weights: jax.Array, block_size: int = 64):
    """
    Dummy Straight-through NF4 emulator.
    """

    return ArrayNF4.quantize(weights, block_size=block_size).materialize()
