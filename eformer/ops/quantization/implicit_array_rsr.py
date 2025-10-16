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
from dataclasses import dataclass

import jax
from jax import lax
from jax import numpy as jnp

from eformer.jaximus import ImplicitArray, aux_field

from .quantization_functions import pack_weights_1bit

Array = jax.Array


@functools.partial(jax.jit, static_argnames=["k"])
def _generate_binary_matrix(k: int) -> Array:
    """Generates the small, dense binary matrix of shape (2**k, k)."""
    indices = jnp.arange(2**k)
    exponents = jnp.arange(k - 1, -1, -1)
    return ((indices[:, None] >> exponents) & 1).astype(jnp.int32)


def _preprocess_matrix_jax(A: Array, k: int) -> tuple[Array, int]:
    """
    JAX-based preprocessing function based on the one-hot encoding method.
    This should be run once on the CPU.
    """
    n, m = A.shape
    padding = (k - (m % k)) % k

    if padding > 0:
        A_padded = jnp.pad(A, ((0, 0), (0, padding)), mode="constant")
    else:
        A_padded = A

    num_blocks = A_padded.shape[1] // k
    powers_of_2 = 2 ** jnp.arange(k - 1, -1, -1, dtype=jnp.int32)

    one_hot_maps = []
    for i in range(num_blocks):
        block = lax.dynamic_slice_in_dim(A_padded, i * k, k, axis=1)
        integer_values = block @ powers_of_2
        L = jnp.zeros((n, 2**k), dtype=A.dtype)
        L = L.at[jnp.arange(n), integer_values].set(1)
        one_hot_maps.append(L)
    return jnp.stack(one_hot_maps), padding


@functools.partial(jax.jit, static_argnames=["k", "m"])
def _rsr_v_dot_a_binary(v: Array, one_hot_maps: Array, padding: int, k: int, m: int) -> Array:
    """
    JIT-compiled core logic for v @ A using the one-hot RSR method.
    """
    segmented_sums = jnp.einsum("n,bnz->bz", v, one_hot_maps, optimize="optimal")
    binary_matrix = _generate_binary_matrix(k).astype(v.dtype)
    block_results = segmented_sums @ binary_matrix
    results_flat = block_results.flatten()
    return lax.slice(results_flat, (0,), (m,))


@dataclass
class RSROperatorBinary(ImplicitArray):
    """
    Implicit Array for a Binary Matrix using the one-hot RSR method.
    """

    one_hot_maps: Array
    k: int = aux_field()
    padding: int = aux_field()
    org_dtype: jnp.dtype = aux_field()  # noqa

    @classmethod
    def quantize(cls, A: Array, k: int = 8):
        return cls.from_matrix(pack_weights_1bit(A), k, A.dtype)

    @classmethod
    def from_matrix(cls, A: Array, k: int = 8, org_dtype: jnp.dtype = jnp.float32):
        assert A.dtype == jnp.int32 or A.dtype == jnp.int8, "Input matrix must be integer type."

        one_hot_maps, padding = _preprocess_matrix_jax(A.astype(jnp.int32), k)

        return cls(one_hot_maps=one_hot_maps, k=k, padding=padding, shape=A.shape, dtype=A.dtype, org_dtype=org_dtype)

    def dot(self, v: Array) -> Array:
        n, m = self.shape
        assert v.shape == (n,), f"Input vector shape mismatch. Expected ({n},), got {v.shape}"

        return _rsr_v_dot_a_binary(v.astype(self.one_hot_maps.dtype), self.one_hot_maps, self.padding, self.k, m)

    def materialize(self) -> Array:
        """
        Reconstructs the original dense binary matrix from the preprocessed
        one-hot maps. This is the reverse of the preprocessing step and is
        essential for debugging and verification.
        """
        n, m = self.shape
        num_blocks = self.one_hot_maps.shape[0]
        binary_matrix_B = _generate_binary_matrix(self.k).astype(self.dtype)
        reconstructed_blocks = jnp.einsum("bnz,zk->bnk", self.one_hot_maps, binary_matrix_B, optimize="optimal")
        reconstructed_padded = reconstructed_blocks.transpose((1, 0, 2))
        reconstructed_padded = reconstructed_padded.reshape((n, num_blocks * self.k))
        E = lax.slice(reconstructed_padded, (0, 0), (n, m))
        return E


@dataclass
class RSROperatorTernary(ImplicitArray):
    """This class remains the same, but also needs the materialize method."""

    rsr_b1: RSROperatorBinary
    rsr_b2: RSROperatorBinary

    @classmethod
    def from_matrix(cls, A: Array, k: int = 8):
        assert A.dtype == jnp.int32 or A.dtype == jnp.int8, "Input matrix must be integer type."

        B1 = (A == 1).astype(jnp.int32)
        B2 = (A == -1).astype(jnp.int32)

        rsr_b1 = RSROperatorBinary.from_matrix(B1, k=k)
        rsr_b2 = RSROperatorBinary.from_matrix(B2, k=k)

        return cls(rsr_b1=rsr_b1, rsr_b2=rsr_b2, shape=A.shape, dtype=A.dtype)

    def dot(self, v: Array) -> Array:
        return self.rsr_b1.dot(v) - self.rsr_b2.dot(v)

    def materialize(self) -> Array:
        """
        Reconstructs the original dense ternary matrix by materializing
        the two binary components and subtracting them.
        """
        B1 = self.rsr_b1.materialize()
        B2 = self.rsr_b2.materialize()
        return (B1 - B2).astype(self.dtype)
