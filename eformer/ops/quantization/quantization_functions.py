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


import functools
import os
from contextlib import contextmanager
from functools import partial
from math import ceil

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Global flag for kernel mode
# Automatically set to True on TPU, False elsewhere
_USE_KERNEL_ON_TPU = os.getenv("USE_NF4_KERNEL_TPU", "1")  # Lazy initialization
if _USE_KERNEL_ON_TPU.lower() in ["1", "true", "on"]:
    _USE_KERNEL_ON_TPU = None
else:
    _USE_KERNEL_ON_TPU = False


def _is_tpu():
    """Check if the current device is a TPU."""
    try:
        return jax.local_devices()[0].platform == "tpu"
    except Exception:
        return False


def _get_kernel_state():
    """
    Get the current kernel state with lazy initialization.

    On first call, automatically enables kernels on TPU, disables on other devices.
    This lazy behavior avoids triggering JAX initialization at import time.
    """
    global _USE_KERNEL_ON_TPU
    if _USE_KERNEL_ON_TPU is None:
        _USE_KERNEL_ON_TPU = _is_tpu()
    return _USE_KERNEL_ON_TPU


def is_kernel_available():
    """
    Check if NF4 kernels are available on the current device.

    Returns:
        bool: True if running on TPU (where kernels are supported), False otherwise
    """
    return _is_tpu()


@contextmanager
def nf4_use_kernel(value: bool):
    """
    Context manager to enable/disable NF4 kernel mode.

    Note: Kernels are only enabled on TPU devices. On other devices,
    this setting has no effect and the code will fall back to materialization.

    Args:
        value: Whether to enable kernel mode (only effective on TPU)

    Example:
        >>> with nf4_use_kernel(True):
        ...     result = input @ quantized_weight  # Uses kernel on TPU
    """
    global _USE_KERNEL_ON_TPU
    old = _get_kernel_state()  # Initialize if needed
    # Only enable kernel mode if on TPU
    _USE_KERNEL_ON_TPU = value and _is_tpu()
    yield
    _USE_KERNEL_ON_TPU = old


def get_nf4():
    """Get NF4 lookup table, creating it lazily to avoid JAX init at import time."""

    return jnp.asarray(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
    )


def nf4xf32_to_f32(x):
    """
    Fast polynomial approximation for NF4 dequantization.

    This is significantly faster than table lookups and provides
    accurate approximation of the NF4 codebook values.

    Args:
            x: Integer array (0-15) representing NF4 quantized values

    Returns:
            Float32 array with approximated NF4 values
    """
    x = x.astype(jnp.float32)
    return (
        x
        * (
            x * (x * (x * (1.82943132356953e-5 * x - 0.00068587779130373) + 0.0100420261313669) - 0.0722703570217226)
            + 0.346075459755188
        )
        - 0.994166218659335
    )


# Bit operation utilities
sr = jax.lax.shift_right_logical
sl = jax.lax.shift_left
ba = jax.lax.bitwise_and


def i8tou8(x):
    """Convert int8 to uint8."""
    return jnp.where(x < 0, 256 + x, x)


def u4toi4(x):
    """Convert uint4 to int4 (signed)."""
    return jnp.where(x >= 8, x - 16, x)


def i4tou4(x):
    """Convert int4 to uint4 (unsigned)."""
    return jnp.where(x < 0, 16 + x, x)


def quantize_int8(x: jax.Array, axis: int | tuple = -1):
    """
    Quantize values to 8-bit integers.

    Args:
        x (jax.Array): Input array.

    Returns:
        tuple: A tuple containing:
            - quantized_values (jax.Array): int8 array of shape (k,) containing quantized values.
            - scales (jax.Array): Array of shape (nb,) containing scaling factors.
    """
    if not isinstance(axis, tuple):
        axis = (axis,)
    axis = tuple(z % x.ndim for z in axis)
    amax = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(x.dtype).tiny).astype(x.dtype)
    quant = jnp.round(x / scale).astype(jnp.int8)
    return quant, scale


def dequantize_int8(quants, scales):
    """
    Dequantize 8-bit integers back to float32 values using blockwise scaling.

    Args:
        quants (jax.Array): int8 array of shape (k,) containing quantized values.
        scales (jax.Array): Array of shape (nb,) containing scaling factors.

    Returns:
        jax.Array: Array of shape (k,) containing dequantized float32 values.
    """

    return quants * scales


@functools.partial(jax.jit, static_argnames=["block_size"])
def single_quantize_and_pack_nf4(blocks, block_size=64):
    """
    Combined quantization and packing for better performance.
    Quantizes along the last dimension only, preserving structure.

    Args:
        blocks (jax.Array): Input array. Shape (..., features) where features % block_size == 0
        block_size (int): Size of each quantization block. Defaults to 64.

    Returns:
        tuple: A tuple containing:
            - packed (jax.Array): uint8 packed values. Shape (..., num_blocks, block_size // 2)
            - absmax (jax.Array): Absolute max per block. Shape (..., num_blocks)
    """
    orig_shape = blocks.shape

    # Reshape only last dimension: (..., features) -> (..., num_blocks, block_size)
    *batch_dims, features = orig_shape
    num_blocks = features // block_size
    blocks = blocks.reshape(*batch_dims, num_blocks, block_size)

    # Compute absmax per block along last dimension
    absmax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)  # (..., num_blocks, 1)
    normalized = blocks / (absmax + jnp.finfo(blocks.dtype).tiny)

    # Quantize using NF4 codebook
    errors = normalized[..., None] - get_nf4()  # (..., num_blocks, block_size, 16)
    quantized = jnp.argmin(jnp.abs(errors), axis=-1)  # (..., num_blocks, block_size)

    # Pack two 4-bit values into one 8-bit value
    quantized = quantized.reshape(*batch_dims, num_blocks, block_size // 2, 2)
    packed = (quantized[..., 0] << 4) | quantized[..., 1]  # (..., num_blocks, block_size // 2)

    return packed.astype(jnp.uint8), absmax.squeeze(-1)  # (..., num_blocks, block_size//2), (..., num_blocks)


@functools.partial(jax.jit, static_argnames=["block_size"])
def single_dequantize_nf4(packed_values, absmax, block_size):
    """
    Optimized dequantization combining unpacking and scaling in fewer operations.
    Preserves structure from quantization.

    Args:
        packed_values (jax.Array): uint8 packed values. Shape (..., num_blocks, block_size // 2)
        absmax (jax.Array): Absolute max per block. Shape (..., num_blocks)
        block_size (int): Size of each quantization block.

    Returns:
        jax.Array: Dequantized array. Shape (..., num_blocks * block_size)
    """
    # Unpack 4-bit values from 8-bit packed format
    high = (packed_values >> 4) & 0xF  # (..., num_blocks, block_size // 2)
    low = packed_values & 0xF  # (..., num_blocks, block_size // 2)
    unpacked = jnp.stack([high, low], axis=-1)  # (..., num_blocks, block_size // 2, 2)

    # Get shape info
    *batch_dims, num_blocks, _ = packed_values.shape
    unpacked = unpacked.reshape(*batch_dims, num_blocks, block_size)  # (..., num_blocks, block_size)

    dequantized = nf4xf32_to_f32(unpacked)

    # Scale by absmax
    scaled = dequantized * absmax[..., None]  # (..., num_blocks, block_size)

    # Flatten last two dimensions back to original feature dimension
    scaled = scaled.reshape(*batch_dims, num_blocks * block_size)  # (..., features)
    return scaled


@functools.partial(jax.jit, static_argnames=["block_size"])
def quantize_and_pack_nf4(blocks, block_size=64):
    """
    Quantize and pack an array using NF4 quantization.

    Args:
        blocks (jax.Array): Input array to be quantized and packed.
        block_size (int): Size of each quantization block. Defaults to 64.

    Returns:
        tuple: A tuple containing:
            - packed (jax.Array): uint8 array of packed quantized values.
            - absmax (jax.Array): Array of absolute maximum values for each block.
    """
    # Single function now handles all batch dimensions
    return single_quantize_and_pack_nf4(blocks, block_size)


@functools.partial(jax.jit, static_argnames=["block_size"])
def dequantize_nf4(packed_values, absmax, block_size):
    """
    Dequantize an array packed using NF4 quantization.

    Args:
        packed_values (jax.Array): uint8 array of packed quantized values.
        absmax (jax.Array): Array of absolute maximum values for each block.
        block_size (int): Size of each quantization block.

    Returns:
        jax.Array: Dequantized array of float32 values.
    """
    # Single function now handles all batch dimensions
    return single_dequantize_nf4(packed_values, absmax, block_size)


@jax.jit
def pack_weights_1bit(quantized_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Packs a JAX array of quantized weights into a compact format using 2 bits per value.

    Parameters:
    -----------
    quantized_weights : jnp.ndarray
        An array containing ternary quantized weights {-1, 0, 1}. The first dimension must be
        a multiple of 4.

    Returns:
    --------
    jnp.ndarray
        A packed jnp.uint8 array.
    """
    original_shape = quantized_weights.shape
    if original_shape[0] % 4 != 0:
        raise ValueError(f"The first dimension must be a multiple of {4}. Got shape {original_shape}.")

    unpacked = (quantized_weights + 1).astype(jnp.uint8)
    reshaped = unpacked.reshape((4, original_shape[0] // 4, *original_shape[1:]))
    shifter = jnp.arange(0, 2 * 4, 2, dtype=jnp.uint8)
    shifter = shifter.reshape((4,) + (1,) * (reshaped.ndim - 1))
    shifted_values = reshaped << shifter
    packed = jnp.sum(shifted_values, axis=0, dtype=jnp.uint8)

    return packed


@functools.partial(jax.jit, static_argnames="dtype")
def unpack_weights_1bit(packed: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Unpacks a JAX array of quantized weights, matching the logic of the PyTorch original.
    This function concatenates the unpacked bit groups.

    Parameters:
    -----------
    packed : jnp.ndarray
        A packed jnp.uint8 array.
    dtype : jnp.dtype
        The dtype of the returned array (e.g., jnp.int8). This is a static argument for JIT.

    Returns:
    --------
    jnp.ndarray
        An unpacked array with ternary values {-1, 0, 1}.
    """
    shifter = jnp.arange(0, 2 * 4, 2, dtype=jnp.uint8)
    shifter = shifter.reshape((4,) + (1,) * packed.ndim)
    unpacked_groups = (packed >> shifter) & 3
    original_row_dim = packed.shape[0] * 4
    unpacked_shape = (original_row_dim, *packed.shape[1:])
    unpacked = unpacked_groups.reshape(unpacked_shape)

    return unpacked.astype(dtype) - 1


# TPU Pallas Kernels for optimized matmul
BLOCK_OVERRIDE = None


def bmm_nf4(
    inputs_ref,
    quants_ref,
    scale_ref,
    outputs_ref,
    accum_ref,
    *,
    block_k,
):
    """
    Pallas kernel for NF4 matrix multiplication with on-the-fly dequantization.

    This kernel performs efficient matrix multiplication by dequantizing weights
    on-the-fly during computation, avoiding materialization of full-precision weights.

    Args:
            inputs_ref: Reference to input activation tensor
            quants_ref: Reference to quantized weight tensor (int4)
            scale_ref: Reference to scale factors
            outputs_ref: Reference to output tensor
            accum_ref: Reference to accumulator tensor
            block_k: Block size for K dimension (static)
    """

    @pl.when(pl.program_id(axis=2) == 0)
    def _():
        accum_ref[...] = jnp.zeros_like(accum_ref)

    quants = quants_ref[...]
    scale = scale_ref[...]

    if quants.dtype == jnp.int8:
        w1 = (quants.astype(jnp.float32) / 127.5) * scale.astype(jnp.float32)
    else:
        # Convert int4 to unsigned, then dequantize
        quants = i4tou4(quants.astype(jnp.int32))
        quants = nf4xf32_to_f32(quants)
        w1 = quants * scale

    inputs = inputs_ref[...]
    accum_ref[...] += jnp.dot(inputs, w1.reshape(block_k, -1), preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(axis=2) == (pl.num_programs(axis=2) - 1))
    def _():
        outputs_ref[...] = accum_ref[...].astype(outputs_ref.dtype)


def bmm_nf4_transpose(
    inputs_ref,
    quants_ref,
    scale_ref,
    outputs_ref,
    accum_ref,
    *,
    block_k,
):
    """
    Pallas kernel for transposed NF4 matrix multiplication.

    This kernel handles the transpose case where weights need to be accessed
    in a different order. Used for backward passes in training.

    Args:
            inputs_ref: Reference to input activation tensor
            quants_ref: Reference to quantized weight tensor (int4)
            scale_ref: Reference to scale factors
            outputs_ref: Reference to output tensor
            accum_ref: Reference to accumulator tensor
            block_k: Block size for K dimension (static)
    """
    accum_ref[...] = jnp.zeros_like(accum_ref)

    loop_iterations = max(1, inputs_ref.shape[-1] // block_k)

    def matmul_loop(i, _):
        inputs = pl.load(inputs_ref, (slice(None), pl.dslice(i * block_k, block_k)))
        quants = pl.load(quants_ref, (slice(None), slice(None), pl.dslice(i * block_k, block_k)))
        scale = pl.load(scale_ref, (slice(None), slice(None), pl.dslice(i * block_k, block_k)))

        quants = quants.astype(jnp.int32)
        quants = i8tou8(quants)

        w1 = nf4xf32_to_f32(sr(quants, 4)) * scale
        w1 = w1.reshape(-1, inputs.shape[-1])
        w2 = nf4xf32_to_f32(quants & 0b1111) * scale
        w2 = w2.reshape(-1, inputs.shape[-1])

        output1 = inputs @ w1.T
        output2 = inputs @ w2.T

        output = jnp.concatenate((output1, output2), -1)
        accum_ref[...] += output

    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None)
    accum = accum_ref[...]
    outputs_ref[...] = accum.astype(outputs_ref.dtype)


@partial(jax.jit, static_argnames=("kernel", "backward", "blocks"))
def nf4_matmul(inputs, *tensors, kernel, backward=False, blocks=None):
    """
    Fast matrix multiplication using Pallas TPU kernels.

    This function provides optimized matrix multiplication with automatic
    block size selection and padding for optimal TPU performance.

    Args:
            inputs: Input activation tensor
            *tensors: Quantized weight tensors (quants, scales)
            kernel: Pallas kernel function to use
            backward: Whether this is a backward pass
            blocks: Optional manual block size specification (block_x, block_y, block_k)

    Returns:
            Result of matrix multiplication
    """
    weight_transpose = backward

    inputs = inputs.astype(jnp.bfloat16)

    if blocks is None:
        if not backward:
            if BLOCK_OVERRIDE is not None:
                block_x, block_y, block_k = BLOCK_OVERRIDE
            else:
                block_x, block_y, block_k = 2048, 512, 256
        else:
            block_x, block_y, block_k = 256, 256, 512
    else:
        block_x, block_y, block_k = blocks

    if not weight_transpose:
        y = tensors[0].shape[2]
        quant_group_size = tensors[0].shape[1]
    else:
        quant_group_size = tensors[0].shape[1]
        y = quant_group_size * tensors[0].shape[0]

    x = inputs.shape[0]
    k = inputs.shape[1]

    # Adjust block sizes to fit dimensions
    if x < block_x:
        block_x = max(16, int(2 ** np.floor(np.log2(x))))
    if y < block_y:
        block_y = max(16, int(2 ** np.floor(np.log2(y))))
    if k < block_k:
        block_k = max(128, int(2 ** np.floor(np.log2(k))))

    # Pad inputs and tensors to match block sizes
    x_pad = (block_x - x) % block_x
    k_pad = (block_k - k) % block_k

    if x_pad or k_pad:
        inputs = jnp.pad(inputs.reshape(x, k), ((0, x_pad), (0, k_pad)))

    y_pad = (block_y - y) % block_y
    if y_pad:
        if not weight_transpose:
            tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, y_pad))) for t in tensors]
        else:
            tensors = [jnp.pad(t, ((0, y_pad // quant_group_size), (0, 0), (0, 0))) for t in tensors]
    if k_pad:
        if not weight_transpose:
            tensors = [jnp.pad(t, ((0, k_pad // quant_group_size), (0, 0), (0, 0))) for t in tensors]
        else:
            tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, k_pad))) for t in tensors]

    if not weight_transpose:
        expected = tensors[0].shape[0] * quant_group_size
        if inputs.shape[1] != expected:
            raise ValueError(f"Input shape mismatch: got {inputs.shape[1]}, expected {expected}.")
    else:
        expected = tensors[0].shape[2]
        if inputs.shape[1] != expected:
            raise ValueError(f"Input shape mismatch: got {inputs.shape[1]}, expected {expected}.")

    def kernel_call(inputs, *tensors):
        inputs_dtype = inputs.dtype
        grid = ceil(inputs.shape[0] / block_x), ceil(y / block_y), ceil(k / block_k)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(index_map=lambda i, j, k: (i, k), block_shape=(block_x, block_k)),
            ]
            + [
                pl.BlockSpec(
                    index_map=lambda i, j, k: (k, 0, j),
                    block_shape=(block_k // quant_group_size, t.shape[1], block_y),
                )
                if not weight_transpose
                else pl.BlockSpec(
                    index_map=lambda i, j, k: (j, 0, k),
                    block_shape=(block_y // quant_group_size, t.shape[1], block_k),
                )
                for t in tensors
            ],
            out_specs=pl.BlockSpec(index_map=lambda i, j, k: (i, j), block_shape=(block_x, block_y)),
            scratch_shapes=[pltpu.VMEM((block_x, block_y), jnp.float32)],
        )
        outputs = pl.pallas_call(
            partial(kernel, block_k=block_k),
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct((grid[0] * block_x, grid[1] * block_y), inputs_dtype),
            compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))),
            interpret=jax.local_devices()[0].platform == "cpu",
        )(inputs, *tensors)
        if backward:
            outputs = outputs.reshape(outputs.shape[0], -1, 2, block_y // 2).mT.reshape(outputs.shape)
        return outputs

    result = kernel_call(inputs, *tensors)
    if x_pad or y_pad:
        result = result[:x, :y]
    return result
