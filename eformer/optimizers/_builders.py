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

import optax

from ._base import OptimizerBuilder, SchedulerBuilder, register_optimizer, register_scheduler
from ._config import (
    AdafactorConfig,
    AdamWConfig,
    LionConfig,
    MarsConfig,
    MuonConfig,
    RMSPropConfig,
    SchedulerConfig,
    WhiteKronConfig,
)
from ._tx import mars, quad, skew


@register_scheduler("constant")
@dataclasses.dataclass
class ConstantSchedulerBuilder(SchedulerBuilder):
    """Builder for constant learning rate schedule."""

    config: SchedulerConfig

    def build(self) -> optax.Schedule:
        return optax.constant_schedule(self.config.learning_rate)


@register_scheduler("linear")
@dataclasses.dataclass
class LinearSchedulerBuilder(SchedulerBuilder):
    """Builder for linear learning rate schedule with optional warmup."""

    config: SchedulerConfig

    def build(self) -> optax.Schedule:
        if self.config.learning_rate_end is None:
            raise ValueError("Linear scheduler requires learning_rate_end")

        base_scheduler = optax.linear_schedule(
            init_value=self.config.learning_rate,
            end_value=self.config.learning_rate_end,
            transition_steps=self.config.steps,
        )

        if self.config.warmup_steps:
            warmup = optax.linear_schedule(
                init_value=1e-8,
                end_value=self.config.learning_rate,
                transition_steps=self.config.warmup_steps,
            )
            return optax.join_schedules(
                schedules=[warmup, base_scheduler],
                boundaries=[self.config.warmup_steps],
            )
        return base_scheduler


@register_scheduler("cosine")
@dataclasses.dataclass
class CosineSchedulerBuilder(SchedulerBuilder):
    """Builder for cosine learning rate schedule with optional warmup."""

    config: SchedulerConfig

    def build(self) -> optax.Schedule:
        if self.config.warmup_steps:
            return optax.warmup_cosine_decay_schedule(
                init_value=1e-8,
                peak_value=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.steps - self.config.warmup_steps,
                end_value=self.config.learning_rate_end or 0.0,
                exponent=self.config.exponent,
            )
        return optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=self.config.steps,
            alpha=self.config.learning_rate_end or 0.0,
        )


@register_optimizer("adamw")
@dataclasses.dataclass
class AdamWOptimizer(OptimizerBuilder):
    """Builder for AdamW optimizer."""

    config: AdamWConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            eps=self.config.eps,
            eps_root=self.config.eps_root,
            mu_dtype=self.config.mu_dtype,
        )


@register_optimizer("adafactor")
@dataclasses.dataclass
class AdafactorOptimizer(OptimizerBuilder):
    """Builder for Adafactor optimizer."""

    config: AdafactorConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return optax.adafactor(
            learning_rate=scheduler,
            min_dim_size_to_factor=self.config.min_dim_size_to_factor,
            decay_rate=self.config.decay_rate,
            decay_offset=self.config.decay_offset,
            multiply_by_parameter_scale=self.config.multiply_by_parameter_scale,
            clipping_threshold=self.config.clipping_threshold,
            momentum=self.config.momentum,
            dtype_momentum=self.config.dtype_momentum,
            weight_decay_rate=self.config.weight_decay_rate,
            eps=self.config.eps,
            factored=self.config.factored,
        )


@register_optimizer("lion")
@dataclasses.dataclass
class LionOptimizer(OptimizerBuilder):
    """Builder for Lion optimizer."""

    config: LionConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return optax.lion(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            mu_dtype=self.config.mu_dtype,
        )


@register_optimizer("rmsprop")
@dataclasses.dataclass
class RMSPropOptimizer(OptimizerBuilder):
    """Builder for RMSProp optimizer."""

    config: RMSPropConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return optax.rmsprop(
            learning_rate=scheduler,
            decay=self.config.decay,
            eps=self.config.eps,
            initial_scale=self.config.initial_scale,
            centered=False,
            momentum=self.config.momentum,
            nesterov=self.config.nesterov,
        )


@register_optimizer("muon")
@dataclasses.dataclass
class MuonOptimizer(OptimizerBuilder):
    """Builder for Muon optimizer."""

    config: MuonConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return optax.contrib.muon(
            learning_rate=scheduler,
            ns_steps=self.config.ns_steps,
            ns_coeffs=self.config.ns_coeffs,
            beta=self.config.beta,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            mu_dtype=self.config.mu_dtype,
            nesterov=self.config.nesterov,
            adaptive=self.config.adaptive,
            adam_b1=self.config.adam_b1,
            adam_b2=self.config.adam_b2,
            adam_eps_root=self.config.adam_eps_root,
        )


@register_optimizer("quad")
@dataclasses.dataclass
class QuadOptimizer(OptimizerBuilder):
    """Builder for Quad (White Kron Quad) optimizer."""

    config: WhiteKronConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return quad(
            learning_rate=scheduler,
            lr_style=self.config.lr_style,
            b1=self.config.b1,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            normalize_grads=self.config.normalize_grads,
            max_size_dense=self.config.max_size_dense,
            preconditioner_lr=self.config.preconditioner_lr,
            preconditioner_init_scale=self.config.preconditioner_init_scale,
            dtype=self.config.dtype,
            scanned_layers=self.config.scanned_layers,
            block_size=self.config.block_size,
            pipeline_axis_name=self.config.pipeline_axis_name,
            pipeline_axis_size=self.config.pipeline_axis_size,
            params_partition_specs=self.config.params_partition_specs,
            noise_scale=self.config.noise_scale,
        )


@register_optimizer("skew")
@dataclasses.dataclass
class SkewOptimizer(OptimizerBuilder):
    """Builder for Skew (White Kron Skew) optimizer."""

    config: WhiteKronConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return skew(
            learning_rate=scheduler,
            lr_style=self.config.lr_style,
            b1=self.config.b1,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            normalize_grads=self.config.normalize_grads,
            max_size_dense=self.config.max_size_dense,
            preconditioner_lr=self.config.preconditioner_lr,
            preconditioner_init_scale=self.config.preconditioner_init_scale,
            dtype=self.config.dtype,
            scanned_layers=self.config.scanned_layers,
            block_size=self.config.block_size,
            pipeline_axis_name=self.config.pipeline_axis_name,
            pipeline_axis_size=self.config.pipeline_axis_size,
            params_partition_specs=self.config.params_partition_specs,
            noise_scale=self.config.noise_scale,
        )


@register_optimizer("mars")
@dataclasses.dataclass
class MarsOptimizer(OptimizerBuilder):
    """Builder for Mars optimizer."""

    config: MarsConfig

    def build(self, scheduler: optax.Schedule) -> optax.GradientTransformation:
        return mars(
            learning_rate=scheduler,
            b1=self.config.beta1,
            b2=self.config.beta2,
            gamma=self.config.gamma,
            eps=self.config.epsilon,
            max_grad_norm=self.config.max_grad_norm,
        )
