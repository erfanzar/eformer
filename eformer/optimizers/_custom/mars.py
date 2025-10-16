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


from typing import Any, NamedTuple

import chex
import jax
import optax
from jax import numpy as jnp
from optax import tree_utils as otu


class ScaleByMarsState(NamedTuple):
    """State for the Mars algorithm."""

    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates
    mog: optax.Updates


def scale_by_mars(
    b1: float = 0.9,
    b2: float = 0.999,
    gamma: float = 0.05,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    max_grad_norm: float = 0.0,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    r"""Rescale updates according to the Mars algorithm.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      gamma: Decay rate for the exponentially weighted average of the gradient from the previous step.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional dtype to be used for the first order accumulator; if
        None then the dtype is inferred from params and updates.


    Returns:
      A :class:optax.GradientTransformation object.
    """

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        mog = otu.tree_zeros_like(params, dtype=mu_dtype)
        return ScaleByMarsState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, mog=mog)

    def update_fn(updates, state, params=None):
        c = jax.tree.map(
            lambda og, g: None if g is None else g + (gamma * b1 / (1 - b1)) * (g - og),
            state.mog,
            updates,
            is_leaf=lambda x: x is None,
        )
        if max_grad_norm:
            g_norm = optax.global_norm(c)
            scale = jnp.minimum(1.0, max_grad_norm / (g_norm + 1e-6))
            c = jax.tree_map(lambda g: None if g is None else g * scale, c, is_leaf=lambda x: x is None)
        mu = otu.tree_update_moment(c, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(c, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)
        return adam_updates, ScaleByMarsState(count=count_inc, mu=mu, nu=nu, mog=updates)

    return optax.GradientTransformation(init_fn, update_fn)


def mars(learning_rate: float | optax.Schedule, **kwargs) -> optax.GradientTransformation:
    """
    Mars (Matrix-wise Adaptive Regularized Scaling) optimizer.

    Args:
        learning_rate: Learning rate or schedule.
        **kwargs: Additional parameters for scale_by_mars.

    Returns:
        optax.GradientTransformation: The Mars optimizer.
    """
    return optax.chain(scale_by_mars(**kwargs), optax.scale(-learning_rate))
