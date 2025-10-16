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


from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive

from eformer.jaximus import ImplicitArray, aux_field, register

from .quantization_functions import dequantize_nf4, quantize_and_pack_nf4

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

    Methods:
        __init__(self, array: jax.Array, block_size: int = 64): Initializes the `ArrayNF4`
            object by quantizing the input array.
        materialize(self): Reconstructs the original array from the quantized data.
    """

    packed: Array
    absmax: Array
    block_size: int = aux_field()

    @classmethod
    def quantize(cls, array: Array, block_size: int = 64, verbose=False):
        """
        Initializes the `ArrayNF4` object by quantizing the input array.

        Args:
            array (jax.Array): The input array to be quantized.
            block_size (int): The size of each quantization block. Defaults to 64.
        """
        block_size = min(block_size, array.shape[-1], array.size)

        packed, absmax = quantize_and_pack_nf4(array.reshape(-1, block_size), block_size)
        return cls(
            packed=packed,
            absmax=absmax,
            block_size=block_size,
            shape=array.shape,
            dtype=array.dtype,
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

    def delete(self):
        self.packed.delete()
        self.absmax.delete()


ArrayType = Array | ArrayNF4


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

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
      primitive: The JAX primitive being handled.
      lhs: Left-hand side array.
      rhs: Right-hand side array.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.dot_general operation.
    """
    lhs_mat, lhs_materialized = safe_materialize(lhs)
    rhs_mat, rhs_materialized = safe_materialize(rhs)

    try:
        res = lax.dot_general(lhs_mat, rhs_mat, *args, **kwargs)
    finally:
        pass
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
    x_mat, x_materialized = safe_materialize(x)
    y_mat, y_materialized = safe_materialize(y)

    try:
        result = lax.add(x_mat, y_mat)
    finally:
        pass

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
    operand_mat, operand_materialized = safe_materialize(operand)
    init_value_mat, init_value_materialized = safe_materialize(init_value)

    try:
        result = lax.reduce(operand_mat, init_value_mat, *args, **kwargs)
    finally:
        pass

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
    x_mat, x_materialized = safe_materialize(x)
    y_mat, y_materialized = safe_materialize(y)

    try:
        result = lax.mul(x_mat, y_mat)
    finally:
        pass

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

    try:
        result_mat = lax.transpose(array, *args, **kwargs)

        result = ArrayNF4.quantize(result_mat, block_size=operand.block_size)
    finally:
        pass

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
    lhs_mat, lhs_materialized = safe_materialize(lhs)
    rhs_mat, rhs_materialized = safe_materialize(rhs)

    try:
        result = lax.conv_general_dilated(lhs_mat, rhs_mat, *args, **kwargs)
    finally:
        pass

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
    x_mat, x_materialized = safe_materialize(x)
    y_mat, y_materialized = safe_materialize(y)

    try:
        result = lax.max(x_mat, y_mat, *args, **kwargs)
    finally:
        pass

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
    x_mat, x_materialized = safe_materialize(x)

    try:
        result = lax.exp(x_mat, *args, **kwargs)
    finally:
        pass

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
    x_mat, x_materialized = safe_materialize(x)

    try:
        result = lax.log(x_mat, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Log operation failed: {e!s}") from e
    finally:
        pass

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

    try:
        result = lax.concatenate(materialized_operands, *args, **kwargs)
    finally:
        pass

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
    operand_mat, operand_materialized = safe_materialize(operand)

    try:
        result = jax.lax.gather(operand_mat, *args, **kwargs)
    finally:
        pass
    return result
