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
- 8-bit quantization (`Array8B`)

These classes are designed to reduce memory usage and computational overhead while maintaining reasonable accuracy for
machine learning models. They are built on top of JAX, a high-performance numerical computing library.

Classes:
    - `Array8B`: Implements 8-bit quantization for arrays.

Usage Example:
    ```python
    import jax
    from eformer.ops.quantization import Array8B, ArrayNF4
    from eformer.jaximus import implicit

    array = jax.random.normal(jax.random.key(0), (256, 64), "f2")


    qarray = Array8B(array)


    n4array = ArrayNF4(array)



    def power(x):
      return x**2



    print(jax.jit(implicit(power))(qarray))
    print(qarray)

    print(jax.jit(implicit(power))(n4array))
    print(n4array)
    ```
"""

from __future__ import annotations

import dataclasses
import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive
from jax.sharding import NamedSharding, PartitionSpec

from eformer.jaximus import ImplicitArray, aux_field, register, ste

from .quantization_functions import dequantize_int8, quantize_int8

Array = jax.Array
ShardingType = NamedSharding | PartitionSpec | None


@dataclasses.dataclass
class Array8B(ImplicitArray):
    """
    8-bit Quantization Class

    This class implements 8-bit quantization for arrays. It quantizes the input array into 8-bit integers and stores
    the quantization scale factor. The original array can be reconstructed (dequantized) using the stored scale factor.

    Attributes:
      scale (jax.Array): The scale factor used for quantization.
      weight (jax.Array): The quantized 8-bit integer array.
      axis (tuple[int, ...] | None): The axis used for quantization (static).
      sharding (ShardingType): The sharding specification to preserve across operations (static).

    Methods:
      quantize(array, dtype, axis): Creates a quantized Array8B from input array.
      materialize(): Reconstructs the original array from the quantized data.
      with_sharding(sharding): Returns a new Array8B with the specified sharding applied.
    """

    scale: Array
    weight: Array
    axis: tuple[int, ...] | None = aux_field(default=None)
    sharding: ShardingType = aux_field(default=None)  # noqa: RUF009

    @classmethod
    def quantize(
        cls,
        array: Array,
        dtype: jnp.dtype | None = None,
        axis: int | tuple[int] | None = None,
    ) -> Array8B:
        """
        Initializes the `Array8B` object by quantizing the input array.

        Args:
          array (jax.Array): The input array to be quantized.
          dtype (jnp.dtype | None): The dtype for materialization. Defaults to input dtype.
          axis (int | tuple[int] | None): The axis for quantization. Defaults to (-1,).

        Returns:
          Array8B: The quantized array with sharding preserved from input.
        """
        if axis is None:
            axis = (-1,)
        elif not isinstance(axis, tuple):
            axis = (axis,)

        # Capture sharding from input array (only NamedSharding, not SingleDeviceSharding)
        input_sharding = None
        if hasattr(array, "sharding") and isinstance(array.sharding, NamedSharding):
            input_sharding = array.sharding

        weight, scale = quantize_int8(x=array, axis=axis)

        return cls(
            weight=weight,
            scale=scale,
            shape=array.shape,
            dtype=dtype or array.dtype,
            axis=axis,
            sharding=input_sharding,
        )

    def materialize(self) -> Array:
        """
        Reconstructs the original array from the quantized data.

        Returns:
          jax.Array: The dequantized array with sharding constraint applied if available.
        """
        result = dequantize_int8(self.weight, self.scale).reshape(self.shape).astype(self.dtype)

        # Apply sharding constraint if available
        if self.sharding is not None:
            result = _apply_sharding(result, self.sharding)

        return result

    def with_sharding(self, sharding: ShardingType) -> Array8B:
        """
        Returns a new Array8B with the specified sharding applied to component arrays.

        This method creates a copy of the quantized array with sharding constraints
        applied to the underlying weight and scale arrays, ensuring they are properly
        distributed across devices.

        Args:
            sharding: A NamedSharding, PartitionSpec, or None. If PartitionSpec is provided,
                     it will be used directly. For NamedSharding, both the mesh and spec
                     are preserved.

        Returns:
            Array8B: A new instance with sharding applied to component arrays.
        """
        new_weight = _apply_sharding(self.weight, sharding)
        new_scale = _apply_sharding(self.scale, sharding)

        return dataclasses.replace(
            self,
            weight=new_weight,
            scale=new_scale,
            sharding=sharding,
        )

    def reshard(self, sharding: ShardingType) -> Array8B:
        """Alias for with_sharding for API consistency."""
        return self.with_sharding(sharding)

    @property
    def is_sharded(self) -> bool:
        """Returns True if this array has sharding information."""
        return self.sharding is not None

    def delete(self):
        self.weight.delete()
        self.scale.delete()


def _apply_sharding(array: Array, sharding: ShardingType) -> Array:
    """Apply sharding constraint to an array if sharding is specified."""
    if sharding is None:
        return array

    from eformer.escale import with_sharding_constraint

    return with_sharding_constraint(array, sharding)


ArrayType = Array | Array8B


@register("lt")
def lt_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType, **kwargs):
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return jax.lax.lt(x, y, **kwargs)


@register("convert_element_type")
def convert_element_type_8bit_operand_pos(primitive: Primitive, operand: Array8B, new_dtype: tp.Any) -> ArrayType:
    if isinstance(operand, Array8B):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("convert_element_type")
def convert_element_type_8bit_operand_kw(primitive: Primitive, operand: Array8B, **kwargs) -> ArrayType:
    new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
    if isinstance(operand, Array8B):
        operand.dtype = new_dtype
        return operand
    else:
        return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def integer_pow_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.pow(x, y)


@register("integer_pow")
def integer_pow_8bit_x(primitive: Primitive, x: ArrayType, **kwargs) -> ArrayType:
    y = kwargs.get("y", 2)
    if isinstance(x, Array8B):
        x = x.materialize()
    return lax.pow(x, y)


@register("div")
def div_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.div(x, y)


@register("sqrt")
def sqrt_8bit_x(primitive: Primitive, x: Array8B) -> ArrayType:
    x = x.materialize()
    return lax.sqrt(x)


def matmul_bf16_int8_weight_only(
    lhs_bf16: Array,  # (..., M, K), bfloat16
    rhs_q_int8: Array,  # (K, N), int8
    rhs_scale: Array,  # typically (K, 1) or (1, N) from quantize_int8
) -> Array:
    """
    bf16 lhs @ int8 rhs (weight-only quantization).

    We infer whether rhs_scale is per-row (K,1) or per-column (1,N)
    and place the scale in the cheapest mathematically correct spot:

      - per-column (1, N):  Y = (lhs @ W_q) * scale
      - per-row    (K, 1):  Y = (lhs * scale^T) @ W_q

    Anything else falls back to a generic dequantize-then-matmul path.
    """
    rhs_bf16 = rhs_q_int8.astype(jnp.bfloat16)
    w_shape = rhs_q_int8.shape
    s_shape = rhs_scale.shape

    if rhs_q_int8.ndim == 2 and rhs_scale.ndim == 2:
        K, N = w_shape

        if s_shape == (1, N):
            y = jnp.matmul(lhs_bf16, rhs_bf16, preferred_element_type=jnp.float32)

            extra_dims = y.ndim - 2
            scale = rhs_scale.reshape((1,) * extra_dims + (1, N))

            y = y * scale
            return y.astype(jnp.bfloat16)

        if s_shape == (K, 1):
            s_vec = rhs_scale.reshape((K,))
            bcast_shape = (1,) * (lhs_bf16.ndim - 1) + (K,)
            s_bcast = s_vec.reshape(bcast_shape)

            lhs_scaled = lhs_bf16 * s_bcast
            y = jnp.matmul(lhs_scaled, rhs_bf16, preferred_element_type=jnp.float32)
            return y.astype(jnp.bfloat16)

    rhs_deq = rhs_bf16 * rhs_scale.astype(jnp.bfloat16)
    y = jnp.matmul(lhs_bf16, rhs_deq, preferred_element_type=jnp.float32)
    return y.astype(jnp.bfloat16)


@register("dot_general")
def dot_general_8bit_lhs_rhs(primitive: Primitive, lhs: ArrayType, rhs: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's dot_general operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      lhs (ArrayType): Left-hand side array.
      rhs (ArrayType): Right-hand side array.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.dot_general operation.
    """
    if isinstance(lhs, Array) and isinstance(rhs, Array8B):
        return matmul_bf16_int8_weight_only(
            lhs_bf16=lhs,
            rhs_q_int8=rhs.weight,
            rhs_scale=rhs.scale,
        )

    if isinstance(lhs, Array8B):
        lhs = lhs.materialize()
    if isinstance(rhs, Array8B):
        rhs = rhs.materialize()
    return lax.dot_general(lhs, rhs, *args, **kwargs)


@register("add")
def add_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's add operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      x (ArrayType): First array to add.
      y (ArrayType): Second array to add.

    Returns:
      The result of lax.add operation.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.add(x, y)


@register("reduce")
def reduce_8bit_operand_init_value(primitive: Primitive, operand: ArrayType, init_value: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's reduce operation.

    Materializes Array8B inputs before performing the operation.

    Args:
      operand (ArrayType): The array to be reduced.
      init_value (ArrayType): The initial value for the reduction.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.reduce operation.
    """

    if isinstance(operand, Array8B):
        operand = operand.materialize()
    if isinstance(init_value, Array8B):
        init_value = init_value.materialize()
    return lax.reduce(operand, init_value, *args, **kwargs)


@register("mul")
def mul_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's mul operation.

    Materializes Array8B inputs before performing the operation.

    Args:
      x (ArrayType): First array to multiply.
      y (ArrayType): Second array to multiply.

    Returns:
      The result of lax.mul operation.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.mul(x, y)


@register("transpose")
def transpose_8bit_operand(primitive: Primitive, operand: Array8B, *args, **kwargs):
    """
    Custom handler for JAX's transpose operation.

    Transposes the underlying weight and scale arrays directly.
    Preserves sharding from the original array.

    Args:
      primitive: The JAX primitive being handled.
      operand (Array8B): The array to be transposed.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      Array8B: The transposed array with sharding preserved.
    """
    if "permutation" in kwargs:
        perm = kwargs["permutation"]
    elif args:
        perm = args[0]
    else:
        raise TypeError("transpose requires a `permutation` argument")

    weight_t = lax.transpose(operand.weight, perm)
    scale_t = lax.transpose(operand.scale, perm)

    if operand.axis is None:
        new_axis = None
    else:
        new_axis = tuple(perm[a] for a in operand.axis)

    result = Array8B(
        weight=weight_t,
        scale=scale_t,
        shape=weight_t.shape,
        dtype=operand.dtype,
        axis=new_axis,
        sharding=operand.sharding,  # Preserve sharding
    )
    return result


@register("conv_general_dilated")
def conv_general_dilated_8bit_lhs_rhs(primitive: Primitive, lhs: ArrayType, rhs: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's conv_general_dilated operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      lhs (ArrayType): Left-hand side array (input).
      rhs (ArrayType): Right-hand side array (kernel).
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.conv operation.
    """
    if isinstance(lhs, Array8B):
        lhs = lhs.materialize()
    if isinstance(rhs, Array8B):
        rhs = rhs.materialize()
    return lax.conv_general_dilated(lhs, rhs, *args, **kwargs)


@register("max")
def max_8bit_xy(primitive: Primitive, x: ArrayType, y: ArrayType, *args, **kwargs):
    """
    Custom handler for JAX's max operation.

    Materializes Array8B inputs before performing the operation.

    Args:

      x (ArrayType): First array for max comparison.
      y (ArrayType): Second array for max comparison.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.max operation.
    """
    if isinstance(x, Array8B):
        x = x.materialize()
    if isinstance(y, Array8B):
        y = y.materialize()
    return lax.max(x, y, *args, **kwargs)


@register("exp")
def exp_8bit_x(primitive: Primitive, x: Array8B, *args, **kwargs):
    """
    Custom handler for JAX's exp operation.

    Materializes Array8B input before performing the operation.

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
def log_8bit_x(primitive: Primitive, x: Array8B, *args, **kwargs):
    """
    Custom handler for JAX's log operation.

    Materializes Array8B input before performing the operation.

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
def reshape_8bit_operand(primitive: Primitive, operand: Array8B, *args, **params):
    """
    Custom handler for JAX's reshape operation.

    This function handles reshaping for Array8B quantized arrays.
    It materializes input before reshaping and re-quantizes the result.
    Preserves sharding from the original array.

    Args:
      primitive (Primitive): The JAX primitive being handled.
      operand (Array8B): The array to be reshaped.
      *args: Positional arguments for reshape.
      **params: Keyword arguments/parameters for reshape.

    Returns:
      Array8B: The reshaped array, re-quantized with sharding preserved.

    Raises:
      ValueError: If the new shape is not compatible with the original array's size.
    """
    array = operand.materialize()
    subfuns, bind_params = primitive.get_bind_params(params)
    result = primitive.bind(*subfuns, array, *args, **bind_params)
    result = Array8B.quantize(result, dtype=operand.dtype)
    # Preserve sharding from original operand
    if operand.sharding is not None:
        result = result.with_sharding(operand.sharding)
    return result


@register("concatenate")
def concatenate_8bit_operands(primitive: Primitive, operands: tp.Sequence[ArrayType], *args, **kwargs):
    """
    Custom handler for JAX's concatenate operation.

    Materializes Array8B inputs before performing the operation.

    Args:
      operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
      *args: Variable length argument list.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The result of lax.concatenate operation.
    """
    materialized_operands = [op.materialize() if isinstance(op, Array8B) else op for op in operands]
    return lax.concatenate(materialized_operands, *args, **kwargs)


@register("broadcast_in_dim")
def broadcast_in_dim_8bit_operand(primitive: Primitive, operand: Array8B, *args, **params) -> ArrayType:
    """Handle broadcast_in_dim for Array8B. Preserves sharding from the original array."""
    subfuns, bind_params = primitive.get_bind_params(params)

    shape = bind_params["shape"]
    broadcast_dimensions = bind_params["broadcast_dimensions"]

    weight_b = primitive.bind(*subfuns, operand.weight, *args, **bind_params)
    scale_b = primitive.bind(*subfuns, operand.scale, *args, **bind_params)

    if operand.axis is None:
        new_axis = None
    else:
        new_axis = tuple(broadcast_dimensions[a] for a in operand.axis)

    return Array8B(
        weight=weight_b,
        scale=scale_b,
        shape=shape,
        dtype=operand.dtype,
        axis=new_axis,
        sharding=operand.sharding,  # Preserve sharding
    )


@register("gather")
def gather_8bit_operand(primitive: Primitive, operand: Array8B, *args, **kwargs) -> ArrayType:
    """Handle gather for Array8B."""
    array = operand.materialize()
    result = jax.lax.gather(array, *args, **kwargs)
    return result


@ste
def straight_through_8bit(weights: jax.Array, axis: int | None = None):
    """
    Straight-through 8BIT emulator.
    """

    return Array8B.quantize(weights, axis=axis).materialize()
