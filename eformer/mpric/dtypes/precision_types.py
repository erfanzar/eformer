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

import jax
import jax.extend
import jax.numpy as jnp

from eformer.jaximus import implicit

# Data type mappings
STRING_TO_DTYPE_MAP = {
    "bf16": jnp.bfloat16,
    "bfloat16": jnp.bfloat16,
    "fp16": jnp.float16,
    "float16": jnp.float16,
    "fp32": jnp.float32,
    "float32": jnp.float32,
    "fp64": jnp.float64,
    "float64": jnp.float64,
    "fp8": jnp.float8_e5m2,
    "fp8_e4m3fn": jnp.float8_e4m3fn,
    "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "fp8_e5m2": jnp.float8_e5m2,
    "fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "float8_e5m2": jnp.float8_e5m2,
    "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
}

DTYPE_TO_STRING_MAP = {
    jnp.bfloat16: "bf16",
    jnp.float16: "fp16",
    jnp.float32: "fp32",
    jnp.float64: "fp64",
    jnp.float8_e5m2: "fp8",
    jnp.float8_e4m3fn: "fp8_e4m3fn",
    jnp.float8_e4m3fnuz: "fp8_e4m3fnuz",
    jnp.float8_e4m3b11fnuz: "fp8_e4m3b11fnuz",
    jnp.float8_e5m2: "fp8_e5m2",
    jnp.float8_e5m2fnuz: "fp8_e5m2fnuz",
}


@implicit
def put_dtype(
    array: jax.Array,
    dtype: str | jnp.dtype | None,
) -> jax.Array:
    """
    Convert array to specified data type.

    Args:
        array: The input array
        dtype: Target data type (string or jnp.dtype)

    Returns:
        Array with specified dtype
    """
    if not dtype:
        return array

    if isinstance(dtype, str):
        try:
            dtype = STRING_TO_DTYPE_MAP[dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype string: {dtype}") from e

    if array.dtype in (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64):
        return array.astype(dtype)
    return array


DTYPE_MAPPING = {
    # Standard types
    "bf16": jnp.bfloat16,
    "f16": jnp.float16,
    "f32": jnp.float32,
    "f64": jnp.float64,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "float64": jnp.float64,
    "f8_e4m3": jnp.float8_e4m3fn,
    "f8_e5m2": jnp.float8_e5m2,
    "float8_e4m3": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
}


def get_platform_default_half() -> jnp.dtype:
    """Returns platform-specific half precision type."""
    platform = jax.extend.backend.get_backend().platform
    return jnp.bfloat16 if platform == "tpu" else jnp.float16
