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


import dataclasses
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class LossScaleConfig:
    """Configuration for loss scaling behavior."""

    initial_scale: float = 2**15
    growth_interval: int = 2000
    scale_factor: int = 2
    min_scale: float = 1.0


@dataclasses.dataclass(frozen=True)
class NoOpLossScale:
    """No-op loss scale that does nothing."""

    @property
    def loss_scale(self):
        return 1

    def scale(self, tree: T) -> T:
        return tree

    def unscale(self, tree: T) -> T:
        return tree

    def adjust(self, grads_finite: jnp.ndarray):
        return self


@dataclasses.dataclass(frozen=True)
class DynamicLossScale:
    """Dynamic loss scaling for mixed precision training."""

    loss_scale: jnp.ndarray
    counter: jnp.ndarray = dataclasses.field(default_factory=lambda: np.zeros([], np.int32))
    period: int = 2000
    factor: int = 2
    min_loss_scale: jnp.ndarray = dataclasses.field(default_factory=lambda: np.ones([], np.float32))

    def scale(self, tree: T) -> T:
        return jax.tree_util.tree_map(lambda x: x * self.loss_scale, tree)

    def unscale(self, tree: T) -> T:
        return jax.tree_util.tree_map(lambda x: x / self.loss_scale, tree)

    def adjust(self, grads_finite: jnp.ndarray) -> "DynamicLossScale":
        """Adjusts loss scale based on gradient finiteness."""

        def first_finite(a, b):
            return jax.lax.select(jnp.isfinite(a).all(), a, b)

        loss_scale = jax.lax.select(
            grads_finite,
            jax.lax.select(
                self.counter == (self.period - 1),
                first_finite(self.loss_scale * self.factor, self.loss_scale),
                self.loss_scale,
            ),
            jnp.maximum(self.min_loss_scale, self.loss_scale / self.factor),
        )

        counter = ((self.counter + 1) % self.period) * grads_finite

        return DynamicLossScale(
            loss_scale=loss_scale,
            counter=counter,
            period=self.period,
            factor=self.factor,
            min_loss_scale=self.min_loss_scale,
        )
