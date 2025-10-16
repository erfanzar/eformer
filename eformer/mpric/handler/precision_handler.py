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


from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..loss_scaling.loss_scaler import DynamicLossScale, LossScaleConfig, NoOpLossScale
from ..policy.policy import Policy


def _cast_to_dtype(tree: Any, dtype: jnp.dtype) -> Any:
    """Cast floating point values in tree to specified dtype."""

    def conditional_cast(x):
        if isinstance(x, np.ndarray | jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(conditional_cast, tree)


class PrecisionHandler:
    """Handles mixed precision operations for training and inference."""

    def __init__(
        self,
        policy: str | Policy,
        use_dynamic_scale: bool = True,
        loss_scale_config: LossScaleConfig = None,
    ):
        self.policy = policy if isinstance(policy, Policy) else Policy.from_string(policy)
        self.loss_scale_config = loss_scale_config or LossScaleConfig()

        if use_dynamic_scale:
            self.loss_scaler = DynamicLossScale(
                loss_scale=jnp.array(self.loss_scale_config.initial_scale),
                period=self.loss_scale_config.growth_interval,
                factor=self.loss_scale_config.scale_factor,
                min_loss_scale=jnp.array(self.loss_scale_config.min_scale),
            )
        else:
            self.loss_scaler = NoOpLossScale()

    @partial(jax.jit, static_argnums=(0,))
    def cast_for_compute(self, x: Any) -> Any:
        """Cast input to computation dtype."""
        return _cast_to_dtype(x, self.policy.compute_dtype)

    @partial(jax.jit, static_argnums=(0,))
    def cast_for_output(self, x: Any) -> Any:
        """Cast output to output dtype."""
        return _cast_to_dtype(x, self.policy.output_dtype)

    def cast_params(self, params: Any) -> Any:
        """Cast parameters to parameter dtype."""
        return _cast_to_dtype(params, self.policy.param_dtype)

    def training_step_wrapper(self, training_step_fn):
        """Wrap training step with precision and loss scaling handling."""

        def wrapped_step(*args, **kwargs):
            args = jax.tree_util.tree_map(self.cast_for_compute, args)
            kwargs = jax.tree_util.tree_map(self.cast_for_compute, kwargs)

            loss, grads = training_step_fn(*args, **kwargs)

            scaled_loss = self.loss_scaler.scale(loss)

            grads = self.loss_scaler.unscale(grads)

            grads_finite = jax.tree_util.tree_reduce(
                lambda x, y: x and jnp.all(jnp.isfinite(y)),
                grads,
                True,
            )

            self.loss_scaler = self.loss_scaler.adjust(grads_finite)

            unscaled_loss = self.loss_scaler.unscale(scaled_loss)
            final_loss = self.cast_for_output(unscaled_loss)
            final_grads = jax.tree_util.tree_map(self.cast_for_output, grads)

            return final_loss, final_grads, grads_finite

        return wrapped_step

    def inference_wrapper(self, inference_fn):
        """Wrap inference function with precision handling."""

        def wrapped_inference(*args, **kwargs):
            args = jax.tree_util.tree_map(self.cast_for_compute, args)
            kwargs = jax.tree_util.tree_map(self.cast_for_compute, kwargs)
            outputs = inference_fn(*args, **kwargs)
            return jax.tree_util.tree_map(self.cast_for_output, outputs)

        return wrapped_inference
