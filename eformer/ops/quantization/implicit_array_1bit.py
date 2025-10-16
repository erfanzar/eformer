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


"""
Quantization Module

This module provides functionality for quantizing and dequantizing arrays using two different quantization methods:
- 1-bit quantization (`Array1B`)

These classes are designed to reduce memory usage and computational overhead while maintaining reasonable accuracy for
machine learning models. They are built on top of JAX, a high-performance numerical computing library.

Classes:
    - `Array1B`: Implements 1-bit quantization for arrays.

Usage Example:
    ```python
    import jax
    from eformer.ops.quantization import Array1B, ArrayNF4
    from eformer.jaximus import implicit

    array = jax.random.normal(jax.random.key(0), (256, 64), "f2")


    qarray = Array1B(array)


    n4array = ArrayNF4(array)



    def power(x):
      return x**2



    print(jax.jit(implicit(power))(qarray))
    print(qarray)

    print(jax.jit(implicit(power))(n4array))
    print(n4array)
    ```
"""

import typing as tp
from dataclasses import dataclass

import jax
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive

from eformer.jaximus import ImplicitArray, register

from .quantization_functions import pack_weights_1bit, unpack_weights_1bit

Array = jax.Array


@dataclass
class Array1B(ImplicitArray):
    """
    1-bit Quantization Class

    This class implements 1-bit quantization for arrays. It quantizes the input array into 1-bit integers.

    Attributes:
      weight (jax.Array): The quantized 1-bit integer array.

    Methods:
      __init__(self, array: jax.Array): Initializes the `Array1B` object by quantizing the input array.
      materialize(self): Reconstructs the original array from the quantized data.
    """

    weight: Array

    @classmethod
    def quantize(
        cls,
        array: Array,
        dtype: jnp.dtype | None = None,
        axis: int | tuple[int] | None = None,
    ):
        """
        Initializes the `Array1B` object by quantizing the input array.

        Args:
          array (jax.Array): The input array to be quantized.
        """
        if axis is None:
            axis = -1
        weight = pack_weights_1bit(quantized_weights=array)
        return cls(
            weight=weight,
            shape=array.shape,
            dtype=dtype or array.dtype,
        )

    def materialize(self):
        """
        Reconstructs the original array from the quantized data.

        Returns:
          jax.Array: The dequantized array.
        """
        return unpack_weights_1bit(self.weight, self.dtype).reshape(self.shape)

    def delete(self):
        self.weight.delete()
        self.scale.delete()


ArrayType = Array | Array1B | int | float | bool


@register("lt")
def lt_1bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType, **kwargs):
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return jax.lax.lt(x, y, **kwargs)


@register("convert_element_type")
def convert_element_type_1bit_operand_pos(primitive: Primitive, operand: Array1B, new_dtype: tp.Any) -> ArrayType:
    if isinstance(operand, Array1B):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("convert_element_type")
def convert_element_type_1bit_operand_kw(primitive: Primitive, operand: Array1B, **kwargs) -> ArrayType:
    new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
    if isinstance(operand, Array1B):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def integer_pow_1bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return lax.pow(x, y)


@register("integer_pow")
def integer_pow_1bit_x(primitive: Primitive, x: ArrayType, **kwargs) -> ArrayType:
    y = kwargs.get("y", 2)
    if isinstance(x, Array1B):
        x = x.materialize()
    return lax.pow(x, y)


@register("div")
def div_1bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return lax.div(x, y)


@register("sqrt")
def sqrt_1bit_x(primitive: Primitive, x: Array1B) -> ArrayType:
    x = x.materialize()
    return lax.sqrt(x)


@register("dot_general")
def dot_general_1bit_lhs_rhs(primitive: Primitive, lhs: ArrayType, rhs: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's dot_general operation.

    Materializes Array1B inputs before performing the operation.

    Args:

      lhs (ArrayType): Left-hand side array.
      rhs (ArrayType): Right-hand side array.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.dot_general operation.
    """
    if isinstance(lhs, Array1B):
        lhs = lhs.materialize()
    if isinstance(rhs, Array1B):
        rhs = rhs.materialize()
    return lax.dot_general(lhs, rhs, *args, **kwargs)


@register("add")
def add_1bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's add operation.

    Materializes Array1B inputs before performing the operation.

    Args:

      x (ArrayType): First array to add.
      y (ArrayType): Second array to add.

    Returns:
      The result of lax.add operation.
    """
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return lax.add(x, y)


@register("reduce")
def reduce_1bit_operand_init_value(primitive: Primitive, operand: ArrayType, init_value: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's reduce operation.

    Materializes Array1B inputs before performing the operation.

    Args:
      operand (ArrayType): The array to be reduced.
      init_value (ArrayType): The initial value for the reduction.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.reduce operation.
    """

    if isinstance(operand, Array1B):
        operand = operand.materialize()
    if isinstance(init_value, Array1B):
        init_value = init_value.materialize()
    return lax.reduce(operand, init_value, *args, **kwargs)


@register("mul")
def mul_1bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's mul operation.

    Materializes Array1B inputs before performing the operation.

    Args:
      x (ArrayType): First array to multiply.
      y (ArrayType): Second array to multiply.

    Returns:
      The result of lax.mul operation.
    """
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return lax.mul(x, y)


@register("transpose")
def transpose_1bit_operand(primitive: Primitive, operand: Array1B, *args, **kwargs):
    """
    Custom handler for JAX's transpose operation.

    Materializes Array1B input before performing the operation.
    Re-quantizes the result if the input was Array1B.

    Args:

      operand (ArrayType): The array to be transposed.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.transpose operation, potentially re-quantized.
    """
    operand = operand.materialize()
    operand = lax.transpose(operand, *args, **kwargs)
    operand = Array1B.quantize(operand, dtype=operand.dtype)
    return operand


@register("conv_general_dilated")
def conv_general_dilated_1bit_lhs_rhs(primitive: Primitive, lhs: ArrayType, rhs: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's conv_general_dilated operation.

    Materializes Array1B inputs before performing the operation.

    Args:

      lhs (ArrayType): Left-hand side array (input).
      rhs (ArrayType): Right-hand side array (kernel).
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.conv operation.
    """
    if isinstance(lhs, Array1B):
        lhs = lhs.materialize()
    if isinstance(rhs, Array1B):
        rhs = rhs.materialize()
    return lax.conv_general_dilated(lhs, rhs, *args, **kwargs)


@register("max")
def max_1bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's max operation.

    Materializes Array1B inputs before performing the operation.

    Args:

      x (ArrayType): First array for max comparison.
      y (ArrayType): Second array for max comparison.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.max operation.
    """
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return lax.max(x, y, *args, **kwargs)


@register("div")
def _(primitive: Primitive, x: ArrayType, y: ArrayType) -> tp.Any:
    if isinstance(x, Array1B):
        x = x.materialize()
    if isinstance(y, Array1B):
        y = y.materialize()
    return lax.div(x, y)


@register("exp")
def exp_1bit_x(primitive: Primitive, x: Array1B, *args, **kwargs):
    """
    Custom handler for JAX's exp operation.

    Materializes Array1B input before performing the operation.

    Args:

      x (ArrayType): The array to apply exponential to.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.exp operation.
    """

    x = x.materialize()
    return lax.exp(x, *args, **kwargs)


@register("log")
def log_1bit_x(primitive: Primitive, x: Array1B, *args, **kwargs):
    """
    Custom handler for JAX's log operation.

    Materializes Array1B input before performing the operation.

    Args:

      x (ArrayType): The array to apply logarithm to.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.log operation.
    """

    x = x.materialize()
    return lax.log(x, *args, **kwargs)


@register("reshape")
def reshape_1bit_operand(primitive: Primitive, operand: Array1B, *args, **params):
    """
    Custom handler for JAX's reshape operation.

    This function handles reshaping for both regular arrays and Array1B quantized arrays.
    It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

    Args:
      primitive (Primitive): The JAX primitive being handled.
      operand (ArrayType): The array to be reshaped.
      new_sizes (Tuple[int, ...]): The desired new shape of the array.
      dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
      **kwargs: Additional keyword arguments for the reshape operation.

    Returns:
      ArrayType: The reshaped array, potentially re-quantized if the input was Array1B.

    Raises:
      ValueError: If the new shape is not compatible with the original array's size.
    """

    array = operand.materialize()
    subfuns, bind_params = primitive.get_bind_params(params)
    result = primitive.bind(*subfuns, array, *args, **bind_params)
    result = Array1B.quantize(result, dtype=operand.dtype)
    return result


@register("concatenate")
def concatenate_1bit_operands(primitive: Primitive, operands: tp.Sequence[ArrayType], *args, **kwargs):
    """
    Custom handler for JAX's concatenate operation.

    Materializes Array1B inputs before performing the operation.

    Args:
      operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.concatenate operation.
    """
    materialized_operands = [op.materialize() if isinstance(op, Array1B) else op for op in operands]
    return lax.concatenate(materialized_operands, *args, **kwargs)


@register("broadcast_in_dim")
def broadcast_in_dim_1bit_operand(primitive: Primitive, operand: Array1B, *args, **params) -> ArrayType:
    """Handle broadcast_in_dim for Array1B."""
    array = operand.materialize()
    subfuns, bind_params = primitive.get_bind_params(params)
    result = primitive.bind(*subfuns, array, *args, **bind_params)
    result = Array1B.quantize(result, dtype=operand.dtype)
    return result


@register("gather")
def gather_1bit_operand(primitive: Primitive, operand: Array1B, *args, **kwargs) -> ArrayType:
    """Handle gather for Array1B."""
    array = operand.materialize()
    result = jax.lax.gather(array, *args, **kwargs)
    return result
