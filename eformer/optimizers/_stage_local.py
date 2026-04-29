# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import dataclasses
import functools
import math
import typing as tp

import jax
import optax
from jax import numpy as jnp


@dataclasses.dataclass(frozen=True)
class StageLocalOptimizerMetadata:
    """Immutable metadata container for pipeline-parallel stage-local optimizer updates.

    This dataclass captures all hyperparameters and configuration needed to
    reconstruct and apply optimizer updates leaf-by-leaf inside a single
    pipeline stage without whole-tree cross-stage communication. It is used
    by the stage-local apply functions to correctly schedule learning rates,
    apply weight decay, and clip gradients while respecting per-parameter
    masks and accumulation settings.

    Attributes:
        scheduler: The learning-rate schedule that was paired with the optimizer
            at construction time.
        weight_decay: Global weight-decay coefficient applied externally through
            :func:`optax_add_scheduled_weight_decay`.
        weight_decay_mask: Optional pytree or callable mask controlling which
            parameters receive external weight decay. ``None`` means all
            parameters are decayed.
        gradient_accumulation_steps: Number of micro-steps accumulated before
            applying an update. Values ``> 1`` are currently unsupported for
            stage-local paths and will raise at runtime.
        clip_grad: Optional global gradient-norm clipping threshold applied
            before the base optimizer update.
        adamw_b1: AdamW first-moment decay (``b1``) when the underlying optimizer
            is AdamW-like. ``None`` for non-AdamW optimizers.
        adamw_b2: AdamW second-moment decay (``b2``) when the underlying optimizer
            is AdamW-like. ``None`` for non-AdamW optimizers.
        adamw_eps: AdamW epsilon for numerical stability. ``None`` for non-AdamW.
        adamw_eps_root: AdamW epsilon applied inside the square-root. ``None``
            for non-AdamW.
        adamw_mu_dtype: Optional dtype for the first-moment buffer. ``None``
            for non-AdamW or when the default dtype is acceptable.
        optimizer_config: The original optimizer-specific configuration object
            (e.g. ``AdamWConfig``). This gives stage-local kernels access to
            extra hyperparameters that are not explicitly mirrored above.
        extra_kwargs: Additional MPMD optimizer options forwarded by the
            builder/factory. Optimizer-specific stage-local kernels can read
            this mapping for extension arguments without changing this
            dataclass every time.
    """

    scheduler: optax.Schedule
    weight_decay: float = 0.0
    weight_decay_mask: tp.Any | None = None
    gradient_accumulation_steps: int = 1
    clip_grad: float | None = None
    adamw_b1: float | None = None
    adamw_b2: float | None = None
    adamw_eps: float | None = None
    adamw_eps_root: float | None = None
    adamw_mu_dtype: tp.Any | None = None
    optimizer_config: tp.Any | None = None
    extra_kwargs: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)


StageLocalApplyFn = tp.Callable[..., tuple[optax.Params, optax.OptState]]
"""Type alias for a stage-local apply function.

A callable with keyword-only arguments ``params``, ``grads``, ``opt_state``,
``learning_rate_fn`` (optional), and ``delete_grads`` (optional) that returns
a tuple of ``(new_params, new_opt_state)``.
"""


class StageLocalGradientTransformation(optax.GradientTransformation):
    """Optax transformation with an explicit PP stage-local apply API.

    This class wraps a standard :class:`optax.GradientTransformation` and
    attaches an additional ``apply_gradients_stage_local`` entry-point that
    pipeline-parallel training loops can call when parameters and gradients
    are partitioned per-stage. The normal :meth:`update` path is preserved,
    so the same object works seamlessly with regular Optax code paths.

    The stage-local path relies on a custom ``_eformer_stage_local_apply``
    attribute attached to the internal ``update`` callable. If that attribute
    is missing, calling :meth:`apply_gradients_stage_local` raises a clear
    :exc:`NotImplementedError`.
    """

    def apply_gradients_stage_local(
        self,
        *,
        params: optax.Params,
        grads: optax.Updates,
        opt_state: optax.OptState,
        learning_rate_fn: optax.Schedule | None = None,
        delete_grads: bool = False,
    ) -> tuple[optax.Params, optax.OptState]:
        """Apply gradients leafwise without whole-tree cross-stage math.

        This method dispatches to the stage-local kernel that was attached at
        construction time. It is intended for scheduled pipeline-parallel
        training where each stage only has a local view of parameters and
        gradients.

        Args:
            params: Current model parameters (pytree). Must be partitioned to
                the same devices as the stage-local gradient buffers.
            grads: Gradient updates (pytree) with the same structure as
                ``params``. ``None`` leaves are interpreted as zero gradients.
            opt_state: Optimizer state produced by :meth:`init` or previous
                calls to :meth:`update` / :meth:`apply_gradients_stage_local`.
            learning_rate_fn: Optional schedule override. When ``None``, the
                schedule stored in :attr:`StageLocalOptimizerMetadata.scheduler`
                is used.
            delete_grads: If ``True``, a best-effort deletion of gradient
                arrays is performed after the update to reduce peak memory
                usage.

        Returns:
            A tuple ``(new_params, new_opt_state)`` with updated values.

        Raises:
            NotImplementedError: If the underlying transformation was not built
                with a stage-local apply function.
        """

        apply_fn = getattr(self.update, "_eformer_stage_local_apply", None)
        if not callable(apply_fn):
            raise NotImplementedError("This optimizer was not built with an eFormer stage-local MPMD apply function.")
        return apply_fn(
            params=params,
            grads=grads,
            opt_state=opt_state,
            learning_rate_fn=learning_rate_fn,
            delete_grads=delete_grads,
        )


def make_stage_local_gradient_transformation(
    tx: optax.GradientTransformation,
    metadata: StageLocalOptimizerMetadata | None = None,
    apply_fn: StageLocalApplyFn | None = None,
) -> StageLocalGradientTransformation:
    """Attach an explicit stage-local MPMD apply path to an Optax transform.

    This function wraps an existing :class:`optax.GradientTransformation` so
    that it also exposes :meth:`StageLocalGradientTransformation.apply_gradients_stage_local`.
    The normal ``tx.init`` and ``tx.update`` callables are forwarded verbatim,
    while the extra ``apply_fn`` is stashed as an attribute on the internal
    update function for later retrieval.

    Optimizer builders pass their optimizer-specific stage-local kernel
    directly. When ``metadata`` is also provided, this helper injects that
    metadata and applies factory-level gradient clipping before calling the
    kernel.

    Args:
        tx: Base optax transformation to wrap.
        metadata: Hyperparameter metadata used by eFormer-provided stage-local
            kernels. Optional for fully custom ``apply_fn`` callables.
        apply_fn: Explicit stage-local apply callable.

    Returns:
        A :class:`StageLocalGradientTransformation` that behaves like ``tx``
        for standard Optax calls but also supports stage-local updates.

    Raises:
        ValueError: If ``apply_fn`` is ``None``.
    """

    if apply_fn is None:
        raise ValueError("apply_fn must be provided for a stage-local optimizer.")
    if metadata is not None:
        raw_apply_fn = apply_fn

        def apply_fn(
            *,
            params: optax.Params,
            grads: optax.Updates,
            opt_state: optax.OptState,
            learning_rate_fn: optax.Schedule | None = None,
            delete_grads: bool = False,
        ) -> tuple[optax.Params, optax.OptState]:
            return raw_apply_fn(
                metadata=metadata,
                params=params,
                grads=_maybe_clip_global_norm(grads, metadata.clip_grad),
                opt_state=opt_state,
                learning_rate_fn=learning_rate_fn,
                delete_grads=delete_grads,
            )

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.OptState]:
        return tx.update(updates, state, params)

    update_fn._eformer_stage_local_metadata = metadata  # type: ignore[attr-defined]
    update_fn._eformer_stage_local_apply = apply_fn  # type: ignore[attr-defined]
    return StageLocalGradientTransformation(tx.init, update_fn)


def make_unsupported_stage_local_gradient_transformation(
    tx: optax.GradientTransformation,
    *,
    optimizer_type: str,
    reason: str | None = None,
) -> StageLocalGradientTransformation:
    """Expose a clear PP error while preserving the normal Optax update path.

    Use this helper when an optimizer builder does **not** override
    :meth:`OptimizerBuilder.build_mpmd`. The returned transformation still
    works via the standard :meth:`optax.GradientTransformation.update`, but
    calling :meth:`StageLocalGradientTransformation.apply_gradients_stage_local`
    raises a descriptive :exc:`NotImplementedError` telling users exactly
    which optimizer is unsupported and how to fix it.

    Args:
        tx: Base optax transformation to wrap.
        optimizer_type: Registered name of the unsupported optimizer (used
            only for the error message).
        reason: Optional extra detail appended to the error message.

    Returns:
        A :class:`StageLocalGradientTransformation` whose stage-local path
        always raises :exc:`NotImplementedError`.
    """

    detail = f" Reason: {reason}" if reason else ""

    def apply_fn(
        *,
        params: optax.Params,
        grads: optax.Updates,
        opt_state: optax.OptState,
        learning_rate_fn: optax.Schedule | None = None,
        delete_grads: bool = False,
    ) -> tuple[optax.Params, optax.OptState]:
        del params, grads, opt_state, learning_rate_fn, delete_grads
        raise NotImplementedError(
            f"Optimizer {optimizer_type!r} does not implement OptimizerBuilder.build_mpmd(...), "
            "so it cannot be used with scheduled MPMD/pipeline-parallel training. "
            "Override build_mpmd in the registered optimizer builder to provide stage-local semantics."
            f"{detail}"
        )

    return make_stage_local_gradient_transformation(tx, apply_fn=apply_fn)


def _chain_parts(metadata: StageLocalOptimizerMetadata, opt_state: optax.OptState):
    """Decompose a chained optimizer state into its constituent parts.

    The standard eFormer optimizer chain is ``[clip?, base, wd?, accum?]``.
    This helper locates the base optimizer state and optional scheduled
    weight-decay state so that stage-local apply functions can mutate them
    directly.

    Args:
        metadata: Metadata describing the optimizer chain.
        opt_state: Tuple state produced by :func:`optax.chain`.

    Returns:
        A triple ``(states, base_index, weight_decay_index)`` where
        ``states`` is a mutable list view of ``opt_state``, ``base_index``
        points to the base optimizer state, and ``weight_decay_index`` is
        either an integer index or ``None``.

    Raises:
        NotImplementedError: If the state structure does not match the
            expected eFormer chain layout.
    """

    states = list(tuple(opt_state))
    base_index = 1 if metadata.clip_grad else 0
    if base_index >= len(states):
        raise NotImplementedError("Invalid eFormer optimizer state for PP stage-local update.")
    weight_decay_index = base_index + 1 if metadata.weight_decay != 0.0 else None
    if weight_decay_index is not None and weight_decay_index >= len(states):
        raise NotImplementedError("Missing scheduled weight-decay state for PP stage-local update.")
    return states, base_index, weight_decay_index


def _scheduled_scalar(
    learning_rate_fn: optax.Schedule | None,
    fallback: optax.Schedule,
    count: jax.Array,
) -> jax.Array:
    """Evaluate a learning-rate schedule at ``count``.

    Args:
        learning_rate_fn: Preferred schedule. When ``None``, ``fallback`` is
            used instead.
        fallback: Fallback schedule embedded in the optimizer metadata.
        count: Scalar integer step count.

    Returns:
        A scalar ``jax.Array`` with dtype ``float32`` representing the
        learning rate at the given step.
    """

    if learning_rate_fn is None:
        learning_rate_fn = fallback
    return jnp.asarray(learning_rate_fn(count), dtype=jnp.float32)


def _external_weight_decay(
    metadata: StageLocalOptimizerMetadata,
    learning_rate_fn: optax.Schedule | None,
    weight_decay_count: jax.Array,
) -> jax.Array:
    """Compute the effective external weight-decay coefficient at a step.

    The eFormer factory applies weight decay through a scheduled wrapper
    ``-scheduler(step) * weight_decay``. This helper reconstructs that scalar
    for stage-local application.

    Args:
        metadata: Optimizer metadata containing ``weight_decay`` and the
            fallback schedule.
        learning_rate_fn: Optional schedule override.
        weight_decay_count: Step count at which to evaluate the schedule.

    Returns:
        A scalar ``jax.Array`` equal to ``lr * weight_decay``.
    """

    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, weight_decay_count)
    return lr * jnp.asarray(metadata.weight_decay, dtype=jnp.float32)


def _mask_or_true(metadata: StageLocalOptimizerMetadata, params: optax.Params) -> tp.Any:
    """Return the weight-decay mask, defaulting to an all-true pytree.

    Args:
        metadata: Optimizer metadata whose ``weight_decay_mask`` may be a
            pytree or a callable.
        params: Parameter pytree used when the mask is callable.

    Returns:
        A pytree with the same structure as ``params`` where each leaf is
        ``True`` (apply decay) or ``False`` (skip decay).
    """

    return _resolve_mask_or_true(metadata.weight_decay_mask, params)


def _resolve_mask_or_true(mask: tp.Any, params: optax.Params) -> tp.Any:
    """Resolve a weight-decay mask against a parameter pytree.

    If ``mask`` is callable, it is invoked with ``params``. If ``mask`` is
    ``None``, an all-true pytree is generated.

    Args:
        mask: A pytree, a callable returning a pytree, or ``None``.
        params: Parameter pytree whose structure drives the default mask.

    Returns:
        A boolean pytree aligned with ``params``.
    """

    if callable(mask):
        mask = mask(params)
    if mask is None:
        return jax.tree_util.tree_map(lambda _: True, params)
    return mask


def _is_masked_node(x: tp.Any) -> bool:
    """Return whether ``x`` is an Optax ``MaskedNode`` sentinel.

    Masked nodes appear in optimizer states when :func:`optax.masked` is used
    to skip updates for certain parameters.
    """

    return type(x).__name__ == "MaskedNode"


def _global_norm_scale(tree: tp.Any, max_norm: float | None) -> jax.Array:
    """Compute a scalar scaling factor for global gradient clipping.

    The scale is ``min(1, max_norm / (norm + 1e-6))`` so that when the
    Euclidean norm of all leaves in ``tree`` exceeds ``max_norm``, the
    gradients are scaled down proportionally.

    Args:
        tree: Pytree of arrays (typically gradients).
        max_norm: Maximum allowed global norm. ``None`` disables clipping.

    Returns:
        A scalar ``jax.Array`` with value in ``[0, 1]``.
    """

    if max_norm is None:
        return jnp.asarray(1.0, dtype=jnp.float32)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
    if not leaves:
        return jnp.asarray(1.0, dtype=jnp.float32)
    sq_norm = 0.0
    for leaf in leaves:
        value = jnp.sum(jnp.square(leaf.astype(jnp.float32)))
        sq_norm += float(jax.device_get(value))
    norm = math.sqrt(sq_norm)
    scale = min(1.0, float(max_norm) / (norm + 1e-6))
    return jnp.asarray(scale, dtype=jnp.float32)


def _maybe_clip_global_norm(grads: optax.Updates, max_norm: float | None) -> optax.Updates:
    """Conditionally rescale a gradient pytree by its global norm.

    Args:
        grads: Gradient pytree.
        max_norm: Maximum allowed norm. ``None`` returns ``grads`` unchanged.

    Returns:
        A pytree with the same structure as ``grads`` where every leaf has
        been multiplied by the global-norm scale factor.
    """

    if max_norm is None:
        return grads
    scale_arr = _global_norm_scale(grads, max_norm)
    return jax.tree_util.tree_map(
        lambda g: None if g is None else g * _place_scalar_like(scale_arr, g),
        grads,
        is_leaf=lambda x: x is None,
    )


def _safe_rms(x: jax.Array, min_rms: float = 1e-3) -> jax.Array:
    """Compute a numerically safe RMS for Adafactor parameter scaling.

    When the RMS of ``x`` is below ``min_rms``, the function falls back to
    ``min_rms`` to avoid division by near-zero values during
    ``multiply_by_parameter_scale`` logic.

    Args:
        x: Input array.
        min_rms: Lower-bound clamp for the returned RMS.

    Returns:
        Scalar RMS value, at least ``min_rms``.
    """

    rms = jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32))))
    safe_x = jnp.where(rms <= min_rms, jnp.ones_like(x), x)
    return jnp.where(rms <= min_rms, min_rms, jnp.sqrt(jnp.mean(jnp.square(safe_x.astype(jnp.float32)))))


def _factored_dims(shape: tuple[int, ...], factored: bool, min_dim_size_to_factor: int) -> tuple[int, int] | None:
    """Determine whether a tensor shape supports factored second-moment estimation.

    Adafactor-style factorization requires at least two dimensions whose
    sizes are both ``>= min_dim_size_to_factor``. The two largest dimensions
    are chosen as the factored pair.

    Args:
        shape: Shape tuple of the parameter tensor.
        factored: Whether factorization is enabled globally.
        min_dim_size_to_factor: Minimum size a dimension must have to be
            considered for factorization.

    Returns:
        A pair of dimension indices ``(d0, d1)`` for factorization, or
        ``None`` if the shape is unsuitable.
    """

    if not factored or len(shape) < 2:
        return None
    sorted_dims = sorted(range(len(shape)), key=lambda i: shape[i])
    if shape[sorted_dims[-2]] < min_dim_size_to_factor:
        return None
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def _decay_rate_pow(step: jax.Array, exponent: float) -> jax.Array:
    """Adafactor-style decay rate ``1 - (step + 1) ** (-exponent)``.

    Args:
        step: Scalar step count (0-indexed).
        exponent: Decay exponent, typically ``0.8`` for Adafactor.

    Returns:
        Scalar decay-rate array.
    """

    t = jnp.asarray(step + 1, dtype=jnp.float32)
    return 1.0 - t ** (-exponent)


@functools.lru_cache(maxsize=32)
def _make_stage_local_adamw_leaf_update(
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    mu_dtype_name: str | None,
) -> tp.Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Build a per-leaf AdamW update kernel.

    The returned JIT-compiled function performs a single AdamW step on one
    parameter leaf. Donation is intentionally avoided here because the
    stage-local optimizer is called from a larger pipeline step whose pytree
    structure may still hold references to the same buffers across steps.

    Args:
        b1: First-moment decay coefficient.
        b2: Second-moment decay coefficient.
        eps: Epsilon added outside the square-root.
        eps_root: Epsilon added inside the square-root.
        mu_dtype_name: Optional dtype name for the first-moment buffer.

    Returns:
        A callable ``leaf_update(param, grad, mu, nu, count, lr, weight_decay)``
        returning ``(new_param, new_mu, new_nu)``.
    """

    mu_dtype = None if mu_dtype_name is None else jnp.dtype(mu_dtype_name)

    @jax.jit
    def leaf_update(param, grad, mu, nu, count, lr, weight_decay):
        count_inc = count + jnp.asarray(1, dtype=count.dtype)
        mu_next = (1.0 - b1) * grad + b1 * mu
        nu_next = (1.0 - b2) * (grad * grad) + b2 * nu

        bias1 = 1.0 - b1**count_inc
        bias2 = 1.0 - b2**count_inc
        mu_hat = mu_next / bias1.astype(mu_next.dtype)
        nu_hat = nu_next / bias2.astype(nu_next.dtype)
        adam_update = mu_hat / (jnp.sqrt(nu_hat + eps_root) + eps)

        update = -lr.astype(adam_update.dtype) * adam_update
        update = update - weight_decay.astype(update.dtype) * param
        new_param = jnp.asarray(param + update).astype(jnp.asarray(param).dtype)

        mu_next = mu_next.astype(mu_dtype if mu_dtype is not None else mu.dtype)
        nu_next = nu_next.astype(nu.dtype)
        return new_param, mu_next, nu_next

    return leaf_update


def _place_array_like(value: tp.Any, like: tp.Any) -> tp.Any:
    """Place ``value`` on ``like``'s sharding when both are array-like.

    This helper ensures that scalar hyperparameters and parameter buffers
    live on the same devices as the gradient leaf they will be combined
    with, preventing unnecessary cross-device transfers inside a stage-local
    JIT block.

    Args:
        value: Array-like value to potentially re-shard.
        like: Reference array whose sharding should be mirrored.

    Returns:
        ``value`` re-placed on ``like``'s sharding, or ``value`` unchanged
        when sharding information is unavailable or incompatible.
    """

    if not hasattr(value, "shape"):
        return value
    sharding = getattr(like, "sharding", None)
    if isinstance(sharding, jax.sharding.Sharding):
        if getattr(value, "ndim", None) != getattr(like, "ndim", None):
            mesh = getattr(sharding, "mesh", None)
            if mesh is not None:
                spec = jax.sharding.PartitionSpec(*([None] * getattr(value, "ndim", 0)))
                return jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            devices = getattr(like, "devices", None)
            if callable(devices):
                devices = tuple(devices())
                if len(devices) == 1:
                    return jax.device_put(value, jax.sharding.SingleDeviceSharding(devices[0]))
            return value
        return jax.device_put(value, sharding)
    return value


def _replicated_scalar_sharding_like(value: tp.Any) -> jax.sharding.Sharding | None:
    """Return a scalar-compatible sharding over ``value``'s devices.

    Args:
        value: Array-like value whose sharding should be inspected.

    Returns:
        A :class:`jax.sharding.Sharding` suitable for a 0-d scalar, or
        ``None`` if no usable sharding can be derived.
    """

    sharding = getattr(value, "sharding", None)
    mesh = getattr(sharding, "mesh", None)
    if mesh is not None:
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    devices = getattr(value, "devices", None)
    if callable(devices):
        devices = tuple(devices())
        if len(devices) == 1:
            return jax.sharding.SingleDeviceSharding(devices[0])
    return None


def _place_scalar_like(scalar: tp.Any, like: tp.Any) -> tp.Any:
    """Place an optimizer scalar on the same stage-local devices as ``like``.

    Args:
        scalar: Scalar value (e.g. learning rate or weight decay) to place.
        like: Reference array whose device assignment should be mirrored.

    Returns:
        ``scalar`` re-placed on a replicated sharding derived from ``like``,
        or ``scalar`` unchanged when sharding information is unavailable.
    """

    sharding = _replicated_scalar_sharding_like(like)
    if sharding is None:
        return scalar
    return jax.device_put(scalar, sharding)


def _delete_tree_arrays(tree: tp.Any) -> None:
    """Best-effort release of donated gradient buffers.

    Iterates over all leaves in ``tree`` and calls ``.delete()`` when
    available. Exceptions are silently swallowed so that this cleanup
    step never crashes the training loop.

    Args:
        tree: Arbitrary pytree (typically gradients).
    """

    for leaf in jax.tree_util.tree_leaves(tree):
        delete = getattr(leaf, "delete", None)
        if callable(delete):
            try:
                delete()
            except Exception:
                pass


def _is_adamw_leaf_result(x: tp.Any) -> bool:
    """Return whether ``x`` is a 3-tuple of arrays (AdamW leaf result)."""

    return isinstance(x, tuple) and len(x) == 3 and all(hasattr(item, "shape") for item in x)


def _unsupported_stage_local(metadata: StageLocalOptimizerMetadata) -> str | None:
    """Check whether the given metadata is incompatible with stage-local AdamW.

    Currently, gradient accumulation and missing AdamW hyperparameters are
    treated as unsupported because the stage-local kernel needs explicit
    ``b1``, ``b2``, ``eps``, and ``eps_root`` values to compile the
    per-leaf JIT update.

    Args:
        metadata: Optimizer metadata to validate.

    Returns:
        A human-readable reason string if unsupported, otherwise ``None``.
    """

    if metadata.gradient_accumulation_steps != 1:
        return "gradient_accumulation_steps > 1"
    missing = [
        name for name in ("adamw_b1", "adamw_b2", "adamw_eps", "adamw_eps_root") if getattr(metadata, name) is None
    ]
    if missing:
        return f"missing AdamW metadata: {', '.join(missing)}"
    return None


def _apply_adamw_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's AdamW chain leafwise for stage-local PP buffers.

    This function unpacks the standard eFormer AdamW chain state
    (``ScaleByAdamState``, ``EmptyState``, ``ScaleByScheduleState``),
    rebuilds the learning-rate and weight-decay scalars, and then maps
    a JIT leaf kernel over the parameter tree. The result is a
    fully stage-local update: no cross-leaf or cross-stage collectives
    are required.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If ``metadata`` describes unsupported settings
            (e.g. gradient accumulation).
    """

    unsupported = _unsupported_stage_local(metadata)
    if unsupported is not None:
        raise NotImplementedError(f"eFormer stage-local PP updates do not yet support {unsupported}.")

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)

    base_state = states[base_index]
    try:
        adam_state, empty_state, schedule_state = base_state
        adam_count = adam_state.count
        schedule_count = schedule_state.count
        mu = adam_state.mu
        nu = adam_state.nu
    except (TypeError, ValueError, AttributeError) as exc:
        raise NotImplementedError(
            "eFormer stage-local AdamW requires the standard optax.adamw state "
            "(ScaleByAdamState, EmptyState, ScaleByScheduleState)."
        ) from exc

    weight_decay_state = states[weight_decay_index] if weight_decay_index is not None else None
    weight_decay_count = weight_decay_state.count if weight_decay_state is not None else schedule_count

    mu_dtype = None if metadata.adamw_mu_dtype is None else jnp.dtype(metadata.adamw_mu_dtype).name
    leaf_update = _make_stage_local_adamw_leaf_update(
        float(metadata.adamw_b1),
        float(metadata.adamw_b2),
        float(metadata.adamw_eps),
        float(metadata.adamw_eps_root),
        mu_dtype,
    )

    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, schedule_count)
    weight_decay = _external_weight_decay(metadata, learning_rate_fn, weight_decay_count)
    weight_decay_mask = _mask_or_true(metadata, params)

    def update_one(param, grad, old_mu, old_nu, decay_enabled):
        if grad is None:
            return param, old_mu, old_nu
        param = _place_array_like(param, grad)
        old_mu = _place_array_like(old_mu, grad)
        old_nu = _place_array_like(old_nu, grad)
        return leaf_update(
            param,
            grad,
            old_mu,
            old_nu,
            _place_scalar_like(adam_count, param),
            _place_scalar_like(lr, param),
            _place_scalar_like(jnp.where(decay_enabled, weight_decay, 0.0), param),
        )

    updated = jax.tree_util.tree_map(
        update_one,
        params,
        grads,
        mu,
        nu,
        weight_decay_mask,
        is_leaf=lambda x: x is None,
    )
    new_params = jax.tree_util.tree_map(lambda x: x[0], updated, is_leaf=_is_adamw_leaf_result)
    new_mu = jax.tree_util.tree_map(lambda x: x[1], updated, is_leaf=_is_adamw_leaf_result)
    new_nu = jax.tree_util.tree_map(lambda x: x[2], updated, is_leaf=_is_adamw_leaf_result)

    new_adam_state = adam_state._replace(
        count=optax.safe_int32_increment(adam_count),
        mu=new_mu,
        nu=new_nu,
    )
    new_schedule_state = schedule_state._replace(count=optax.safe_int32_increment(schedule_count))
    new_base_state = (new_adam_state, empty_state, new_schedule_state)
    states[base_index] = new_base_state

    if weight_decay_state is not None:
        new_weight_decay_state = weight_decay_state._replace(count=optax.safe_int32_increment(weight_decay_state.count))
        states[weight_decay_index] = new_weight_decay_state

    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)


def _leaf_tuple_size(size: int):
    """Return an ``is_leaf`` predicate matching tuple-of-arrays of given length.

    This is used by :func:`jax.tree_util.tree_map` when unpacking leaf-level
    results produced by optimizer-specific stage-local kernels.

    Args:
        size: Expected tuple length.

    Returns:
        A predicate ``callable(x) -> bool``.
    """

    def _is_leaf(x: tp.Any) -> bool:
        return isinstance(x, tuple) and len(x) == size and all(hasattr(item, "shape") for item in x)

    return _is_leaf


def _tuple_size(size: int):
    """Return an ``is_leaf`` predicate matching tuples of a fixed length.

    Unlike :func:`_leaf_tuple_size`, this variant does not require every
    element to have a ``shape`` attribute, making it suitable for mixed
    tuple leaves (e.g. Muon updates that contain both arrays and ``None``).

    Args:
        size: Expected tuple length.

    Returns:
        A predicate ``callable(x) -> bool``.
    """

    def _is_leaf(x: tp.Any) -> bool:
        return isinstance(x, tuple) and len(x) == size

    return _is_leaf


@functools.lru_cache(maxsize=32)
def _make_stage_local_lion_leaf_update(
    b1: float,
    b2: float,
    mu_dtype_name: str | None,
    internal_weight_decay: float,
) -> tp.Callable[..., tuple[jax.Array, jax.Array]]:
    """Build a per-leaf Lion update kernel.

    Lion uses sign-based momentum updates. The returned JIT-compiled function
    performs one Lion step with an optional external weight-decay term.

    Args:
        b1: Lion first-moment (momentum) decay coefficient.
        b2: Lion second-moment (momentum) decay coefficient.
        mu_dtype_name: Optional dtype name for the momentum buffer.
        internal_weight_decay: Small fixed weight-decay coefficient baked into
            the Lion update rule.

    Returns:
        A callable ``leaf_update(param, grad, mu, lr, external_weight_decay)``
        returning ``(new_param, new_mu)``.
    """

    mu_dtype = None if mu_dtype_name is None else jnp.dtype(mu_dtype_name)

    @jax.jit
    def leaf_update(param, grad, mu, lr, external_weight_decay):
        c = b1 * mu + (1.0 - b1) * grad
        update = -lr.astype(c.dtype) * (jnp.sign(c) + jnp.asarray(internal_weight_decay, c.dtype) * param)
        update = update - external_weight_decay.astype(update.dtype) * param
        new_param = jnp.asarray(param + update).astype(jnp.asarray(param).dtype)
        mu_next = b2 * mu + (1.0 - b2) * grad
        mu_next = mu_next.astype(mu_dtype if mu_dtype is not None else mu.dtype)
        return new_param, mu_next

    return leaf_update


def _apply_lion_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's Lion chain leafwise for stage-local PP buffers.

    Unpacks the standard Lion chain state, rebuilds schedule scalars, and
    maps a JIT leaf kernel over the parameter tree.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If gradient accumulation is enabled or the Lion
            config is missing.
    """

    if metadata.gradient_accumulation_steps != 1:
        raise NotImplementedError("eFormer stage-local PP updates do not yet support gradient_accumulation_steps > 1.")
    config = metadata.optimizer_config
    if config is None:
        raise NotImplementedError("Missing Lion optimizer config for PP stage-local update.")

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)
    try:
        lion_state, empty_decay_state, schedule_state = states[base_index]
        del empty_decay_state
        mu = lion_state.mu
    except (TypeError, ValueError, AttributeError) as exc:
        raise NotImplementedError("eFormer stage-local Lion requires the standard optax.lion state.") from exc

    wd_state = states[weight_decay_index] if weight_decay_index is not None else None
    wd_count = wd_state.count if wd_state is not None else schedule_state.count
    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, schedule_state.count)
    external_wd = _external_weight_decay(metadata, learning_rate_fn, wd_count)
    external_mask = _mask_or_true(metadata, params)
    mu_dtype = None if getattr(config, "mu_dtype", None) is None else jnp.dtype(config.mu_dtype).name
    leaf_update = _make_stage_local_lion_leaf_update(
        float(config.b1),
        float(config.b2),
        mu_dtype,
        1e-3,
    )

    def update_one(param, grad, old_mu, decay_enabled):
        if grad is None:
            return param, old_mu
        param = _place_array_like(param, grad)
        old_mu = _place_array_like(old_mu, grad)
        return leaf_update(
            param,
            grad,
            old_mu,
            _place_scalar_like(lr, param),
            _place_scalar_like(jnp.where(decay_enabled, external_wd, 0.0), param),
        )

    updated = jax.tree_util.tree_map(update_one, params, grads, mu, external_mask, is_leaf=lambda x: x is None)
    new_params = jax.tree_util.tree_map(lambda x: x[0], updated, is_leaf=_leaf_tuple_size(2))
    new_mu = jax.tree_util.tree_map(lambda x: x[1], updated, is_leaf=_leaf_tuple_size(2))
    states[base_index] = (
        lion_state._replace(count=optax.safe_int32_increment(lion_state.count), mu=new_mu),
        states[base_index][1],
        schedule_state._replace(count=optax.safe_int32_increment(schedule_state.count)),
    )
    if wd_state is not None:
        states[weight_decay_index] = wd_state._replace(count=optax.safe_int32_increment(wd_count))
    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)


@functools.lru_cache(maxsize=32)
def _make_stage_local_rmsprop_leaf_update(
    decay: float,
    eps: float,
    momentum: float | None,
    nesterov: bool,
) -> tp.Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Build a per-leaf RMSProp update kernel.

    Args:
        decay: Exponential decay rate for the moving average of squared
            gradients.
        eps: Small constant for numerical stability inside the square-root.
        momentum: Optional momentum coefficient. ``None`` disables momentum.
        nesterov: Whether to apply Nesterov momentum when ``momentum`` is set.

    Returns:
        A callable ``leaf_update(param, grad, nu, trace, lr, external_weight_decay)``
        returning ``(new_param, new_nu, new_trace)``.
    """

    @jax.jit
    def leaf_update(param, grad, nu, trace, lr, external_weight_decay):
        nu_next = decay * nu + (1.0 - decay) * (grad * grad)
        scaled = grad * jax.lax.rsqrt(nu_next + eps)
        update = -lr.astype(scaled.dtype) * scaled
        if momentum is not None:
            trace_next = update + float(momentum) * trace
            update = update + float(momentum) * trace_next if nesterov else trace_next
        else:
            trace_next = trace
        update = update - external_weight_decay.astype(update.dtype) * param
        new_param = jnp.asarray(param + update).astype(jnp.asarray(param).dtype)
        return new_param, nu_next.astype(nu.dtype), trace_next.astype(trace.dtype)

    return leaf_update


def _apply_rmsprop_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's RMSProp chain leafwise for stage-local PP buffers.

    Unpacks the standard RMSProp chain state (``ScaleByRmsState``,
    ``ScaleByScheduleState``, ``TraceState``), rebuilds schedule scalars,
    and maps a JIT leaf kernel over the parameter tree.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If gradient accumulation is enabled or the
            RMSProp config is missing.
    """

    if metadata.gradient_accumulation_steps != 1:
        raise NotImplementedError("eFormer stage-local PP updates do not yet support gradient_accumulation_steps > 1.")
    config = metadata.optimizer_config
    if config is None:
        raise NotImplementedError("Missing RMSProp optimizer config for PP stage-local update.")

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)
    try:
        rms_state, schedule_state, trace_state = states[base_index]
        nu = rms_state.nu
    except (TypeError, ValueError, AttributeError) as exc:
        raise NotImplementedError("eFormer stage-local RMSProp requires the standard optax.rmsprop state.") from exc
    trace = getattr(trace_state, "trace", jax.tree_util.tree_map(jnp.zeros_like, params))

    wd_state = states[weight_decay_index] if weight_decay_index is not None else None
    wd_count = wd_state.count if wd_state is not None else schedule_state.count
    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, schedule_state.count)
    external_wd = _external_weight_decay(metadata, learning_rate_fn, wd_count)
    external_mask = _mask_or_true(metadata, params)
    leaf_update = _make_stage_local_rmsprop_leaf_update(
        float(config.decay),
        float(config.eps),
        None if config.momentum is None else float(config.momentum),
        bool(config.nesterov),
    )

    def update_one(param, grad, old_nu, old_trace, decay_enabled):
        if grad is None:
            return param, old_nu, old_trace
        param = _place_array_like(param, grad)
        old_nu = _place_array_like(old_nu, grad)
        old_trace = _place_array_like(old_trace, grad)
        return leaf_update(
            param,
            grad,
            old_nu,
            old_trace,
            _place_scalar_like(lr, param),
            _place_scalar_like(jnp.where(decay_enabled, external_wd, 0.0), param),
        )

    updated = jax.tree_util.tree_map(update_one, params, grads, nu, trace, external_mask, is_leaf=lambda x: x is None)
    new_params = jax.tree_util.tree_map(lambda x: x[0], updated, is_leaf=_leaf_tuple_size(3))
    new_nu = jax.tree_util.tree_map(lambda x: x[1], updated, is_leaf=_leaf_tuple_size(3))
    new_trace = jax.tree_util.tree_map(lambda x: x[2], updated, is_leaf=_leaf_tuple_size(3))
    new_trace_state = trace_state._replace(trace=new_trace) if hasattr(trace_state, "trace") else trace_state
    states[base_index] = (
        rms_state._replace(nu=new_nu),
        schedule_state._replace(count=optax.safe_int32_increment(schedule_state.count)),
        new_trace_state,
    )
    if wd_state is not None:
        states[weight_decay_index] = wd_state._replace(count=optax.safe_int32_increment(wd_count))
    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)


@functools.lru_cache(maxsize=32)
def _make_stage_local_adafactor_leaf_update(
    factored: bool,
    decay_rate: float,
    decay_offset: int,
    min_dim_size_to_factor: int,
    eps: float,
    clipping_threshold: float | None,
    multiply_by_parameter_scale: bool,
) -> tp.Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Build a per-leaf Adafactor update kernel.

    Adafactor uses factored second-moment estimation for memory efficiency.
    The returned JIT-compiled function handles both factored and non-factored
    parameter shapes, optional gradient clipping, and optional parameter-scale
    multiplication.

    Args:
        factored: Whether to enable factored second-moment estimation.
        decay_rate: Exponent for the moving-average decay schedule.
        decay_offset: Step offset applied before evaluating the decay rate.
        min_dim_size_to_factor: Minimum dimension size to qualify for
            factorization.
        eps: Small constant for numerical stability.
        clipping_threshold: Optional threshold for update clipping.
        multiply_by_parameter_scale: Whether to scale the update by the
            parameter's RMS.

    Returns:
        A callable ``leaf_update(param, grad, v_row, v_col, v, count, lr, external_weight_decay)``
        returning ``(new_param, new_v_row, new_v_col, new_v)``.
    """

    @jax.jit
    def leaf_update(param, grad, v_row, v_col, v, count, lr, external_weight_decay):
        decay_rate_t = _decay_rate_pow(count - decay_offset, decay_rate)
        dims = _factored_dims(param.shape, factored, min_dim_size_to_factor)
        if dims is not None:
            d1, d0 = dims
            grad_sqr = jnp.square(grad.astype(jnp.float32)) + eps
            new_v_row = decay_rate_t * v_row + (1.0 - decay_rate_t) * jnp.mean(grad_sqr, axis=d0)
            new_v_col = decay_rate_t * v_col + (1.0 - decay_rate_t) * jnp.mean(grad_sqr, axis=d1)
            reduced_d1 = d1 - 1 if d1 > d0 else d1
            row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
            row_factor = (new_v_row / row_col_mean) ** -0.5
            col_factor = new_v_col**-0.5
            update = grad * jnp.expand_dims(row_factor, axis=d0) * jnp.expand_dims(col_factor, axis=d1)
            new_v = v
        else:
            grad_sqr = jnp.square(grad.astype(jnp.float32)) + eps
            new_v = decay_rate_t * v + (1.0 - decay_rate_t) * grad_sqr
            update = grad * new_v**-0.5
            new_v_row = v_row
            new_v_col = v_col

        if clipping_threshold is not None:
            clip_denom = jnp.maximum(
                1.0,
                jnp.sqrt(jnp.mean(jnp.square(update.astype(jnp.float32)))) / clipping_threshold,
            )
            update = update / clip_denom
        update = update * lr.astype(update.dtype)
        if multiply_by_parameter_scale:
            update = update * _safe_rms(param)
        update = -update - external_weight_decay.astype(update.dtype) * param
        new_param = jnp.asarray(param + update).astype(jnp.asarray(param).dtype)
        return new_param, new_v_row.astype(v_row.dtype), new_v_col.astype(v_col.dtype), new_v.astype(v.dtype)

    return leaf_update


def _apply_adafactor_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's Adafactor chain leafwise for stage-local PP buffers.

    Unpacks the standard Adafactor chain state, rebuilds schedule scalars,
    and maps a JIT leaf kernel that supports both factored and
    non-factored second-moment estimation.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If gradient accumulation, momentum, or internal
            weight decay is enabled, since these are not yet supported in the
            stage-local path.
    """

    if metadata.gradient_accumulation_steps != 1:
        raise NotImplementedError("eFormer stage-local PP updates do not yet support gradient_accumulation_steps > 1.")
    config = metadata.optimizer_config
    if config is None:
        raise NotImplementedError("Missing Adafactor optimizer config for PP stage-local update.")
    if getattr(config, "momentum", None) is not None:
        raise NotImplementedError("eFormer stage-local Adafactor does not yet support momentum.")
    if getattr(config, "weight_decay_rate", None) is not None:
        raise NotImplementedError("eFormer stage-local Adafactor does not yet support internal weight_decay_rate.")

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)
    base_state = tuple(states[base_index])
    factored_state = base_state[0]
    schedule_state = next(
        (
            state
            for state in base_state
            if state is not factored_state and hasattr(state, "count") and not callable(getattr(state, "count", None))
        ),
        None,
    )
    if schedule_state is None:
        raise NotImplementedError("eFormer stage-local Adafactor requires a scheduled learning-rate state.")

    wd_state = states[weight_decay_index] if weight_decay_index is not None else None
    wd_count = wd_state.count if wd_state is not None else schedule_state.count
    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, schedule_state.count)
    external_wd = _external_weight_decay(metadata, learning_rate_fn, wd_count)
    external_mask = _mask_or_true(metadata, params)
    leaf_update = _make_stage_local_adafactor_leaf_update(
        bool(config.factored),
        float(config.decay_rate),
        int(config.decay_offset),
        int(config.min_dim_size_to_factor),
        float(config.eps),
        None if config.clipping_threshold is None else float(config.clipping_threshold),
        bool(config.multiply_by_parameter_scale),
    )

    def update_one(param, grad, v_row, v_col, v, decay_enabled):
        if grad is None:
            return param, v_row, v_col, v
        param = _place_array_like(param, grad)
        v_row = _place_array_like(v_row, grad)
        v_col = _place_array_like(v_col, grad)
        v = _place_array_like(v, grad)
        return leaf_update(
            param,
            grad,
            v_row,
            v_col,
            v,
            _place_scalar_like(factored_state.count, param),
            _place_scalar_like(lr, param),
            _place_scalar_like(jnp.where(decay_enabled, external_wd, 0.0), param),
        )

    updated = jax.tree_util.tree_map(
        update_one,
        params,
        grads,
        factored_state.v_row,
        factored_state.v_col,
        factored_state.v,
        external_mask,
        is_leaf=lambda x: x is None,
    )
    new_params = jax.tree_util.tree_map(lambda x: x[0], updated, is_leaf=_leaf_tuple_size(4))
    new_v_row = jax.tree_util.tree_map(lambda x: x[1], updated, is_leaf=_leaf_tuple_size(4))
    new_v_col = jax.tree_util.tree_map(lambda x: x[2], updated, is_leaf=_leaf_tuple_size(4))
    new_v = jax.tree_util.tree_map(lambda x: x[3], updated, is_leaf=_leaf_tuple_size(4))

    new_factored_state = factored_state._replace(
        count=optax.safe_int32_increment(factored_state.count),
        v_row=new_v_row,
        v_col=new_v_col,
        v=new_v,
    )
    new_base = []
    for state in base_state:
        if state is factored_state:
            new_base.append(new_factored_state)
        elif state is schedule_state:
            new_base.append(schedule_state._replace(count=optax.safe_int32_increment(schedule_state.count)))
        else:
            new_base.append(state)
    states[base_index] = tuple(new_base)
    if wd_state is not None:
        states[weight_decay_index] = wd_state._replace(count=optax.safe_int32_increment(wd_count))
    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)


@functools.lru_cache(maxsize=32)
def _make_stage_local_mars_leaf_update(
    b1: float,
    b2: float,
    gamma: float,
    eps: float,
) -> tp.Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Build a per-leaf Mars update kernel.

    Mars improves upon Adam by adding a variance-reduction term based on the
    gradient from the previous step. The returned JIT-compiled function
    performs one Mars step with an optional external weight-decay term.

    Args:
        b1: First-moment decay coefficient.
        b2: Second-moment decay coefficient.
        gamma: Coefficient controlling the gradient momentum correction term.
        eps: Small constant for numerical stability inside the square-root.

    Returns:
        A callable ``leaf_update(param, grad, mu, nu, mog, count, lr, mars_scale, external_weight_decay)``
        returning ``(new_param, new_mu, new_nu, new_mog)`` where ``new_mog``
        is the stored previous gradient.
    """

    @jax.jit
    def leaf_update(param, grad, mu, nu, mog, count, lr, mars_scale, external_weight_decay):
        c = grad + (gamma * b1 / (1.0 - b1)) * (grad - mog)
        c = c * mars_scale.astype(c.dtype)
        mu_next = (1.0 - b1) * c + b1 * mu
        nu_next = (1.0 - b2) * (c * c) + b2 * nu
        count_inc = count + jnp.asarray(1, dtype=count.dtype)
        mu_hat = mu_next / (1.0 - b1**count_inc).astype(mu_next.dtype)
        nu_hat = nu_next / (1.0 - b2**count_inc).astype(nu_next.dtype)
        update = mu_hat / (jnp.sqrt(nu_hat) + eps)
        update = -lr.astype(update.dtype) * update
        update = update - external_weight_decay.astype(update.dtype) * param
        new_param = jnp.asarray(param + update).astype(jnp.asarray(param).dtype)
        return new_param, mu_next.astype(mu.dtype), nu_next.astype(nu.dtype), grad.astype(mog.dtype)

    return leaf_update


def _apply_mars_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's Mars chain leafwise for stage-local PP buffers.

    Unpacks the standard Mars chain state, applies the Mars gradient
    correction tree-wise (to compute the ``c`` term), optionally clips the
    corrected gradients by global norm, and then maps a JIT leaf
    kernel over the parameter tree.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If gradient accumulation is enabled or the Mars
            config is missing.
    """

    if metadata.gradient_accumulation_steps != 1:
        raise NotImplementedError("eFormer stage-local PP updates do not yet support gradient_accumulation_steps > 1.")
    config = metadata.optimizer_config
    if config is None:
        raise NotImplementedError("Missing Mars optimizer config for PP stage-local update.")

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)
    try:
        mars_state, schedule_state = states[base_index]
        mu, nu, mog = mars_state.mu, mars_state.nu, mars_state.mog
    except (TypeError, ValueError, AttributeError) as exc:
        raise NotImplementedError("eFormer stage-local Mars requires the standard eFormer Mars state.") from exc

    wd_state = states[weight_decay_index] if weight_decay_index is not None else None
    wd_count = wd_state.count if wd_state is not None else schedule_state.count
    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, schedule_state.count)
    external_wd = _external_weight_decay(metadata, learning_rate_fn, wd_count)
    external_mask = _mask_or_true(metadata, params)

    coeff = float(config.gamma) * float(config.beta1) / (1.0 - float(config.beta1))
    c_tree = jax.tree_util.tree_map(
        lambda old_grad, grad: None if grad is None else grad + coeff * (grad - old_grad),
        mog,
        grads,
        is_leaf=lambda x: x is None,
    )
    max_grad_norm = getattr(config, "max_grad_norm", None)
    mars_scale = jnp.asarray(1.0, dtype=jnp.float32)
    if max_grad_norm:
        mars_scale = _global_norm_scale(c_tree, float(max_grad_norm))

    leaf_update = _make_stage_local_mars_leaf_update(
        float(config.beta1),
        float(config.beta2),
        float(config.gamma),
        float(config.epsilon),
    )

    def update_one(param, grad, old_mu, old_nu, old_mog, decay_enabled):
        if grad is None:
            return param, old_mu, old_nu, old_mog
        param = _place_array_like(param, grad)
        old_mu = _place_array_like(old_mu, grad)
        old_nu = _place_array_like(old_nu, grad)
        old_mog = _place_array_like(old_mog, grad)
        return leaf_update(
            param,
            grad,
            old_mu,
            old_nu,
            old_mog,
            _place_scalar_like(mars_state.count, param),
            _place_scalar_like(lr, param),
            _place_scalar_like(mars_scale, param),
            _place_scalar_like(jnp.where(decay_enabled, external_wd, 0.0), param),
        )

    updated = jax.tree_util.tree_map(
        update_one,
        params,
        grads,
        mu,
        nu,
        mog,
        external_mask,
        is_leaf=lambda x: x is None,
    )
    new_params = jax.tree_util.tree_map(lambda x: x[0], updated, is_leaf=_leaf_tuple_size(4))
    new_mu = jax.tree_util.tree_map(lambda x: x[1], updated, is_leaf=_leaf_tuple_size(4))
    new_nu = jax.tree_util.tree_map(lambda x: x[2], updated, is_leaf=_leaf_tuple_size(4))
    new_mog = jax.tree_util.tree_map(lambda x: x[3], updated, is_leaf=_leaf_tuple_size(4))
    states[base_index] = (
        mars_state._replace(
            count=optax.safe_int32_increment(mars_state.count),
            mu=new_mu,
            nu=new_nu,
            mog=new_mog,
        ),
        schedule_state._replace(count=optax.safe_int32_increment(schedule_state.count)),
    )
    if wd_state is not None:
        states[weight_decay_index] = wd_state._replace(count=optax.safe_int32_increment(wd_count))
    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)


def _safe_increment(count: jax.Array) -> jax.Array:
    """Increment a step counter without overflowing int32.

    This is a thin wrapper around :func:`optax.safe_int32_increment`.

    Args:
        count: Scalar integer array.

    Returns:
        ``count + 1`` wrapped safely against int32 overflow.
    """

    return optax.safe_int32_increment(count)


def _bias_correction(value: jax.Array, decay: float, count: jax.Array) -> jax.Array:
    """Apply Adam-style bias correction to ``value``.

    Computes ``value / (1 - decay ** count)`` with careful dtype promotion
    to avoid precision loss.

    Args:
        value: Array to correct (typically a momentum buffer).
        decay: Decay coefficient (e.g. ``b1`` or ``b2``).
        count: Scalar step count.

    Returns:
        Bias-corrected array with the same shape as ``value``.
    """

    return value / (1.0 - jnp.asarray(decay, value.dtype) ** count.astype(value.dtype))


def _apply_muon_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's Muon chain leafwise for stage-local PP buffers.

    Muon processes 2D parameters with Newton-Schulz orthogonalized momentum
    and falls back to Adam for non-2D parameters. This function unpacks the
    partitioned inner state (Muon + Adam), determines the appropriate update
    path per leaf, and applies it stage-locally.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple containing Muon's partitioned
            inner states.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If gradient accumulation is enabled or the Muon
            config is missing.
    """

    if metadata.gradient_accumulation_steps != 1:
        raise NotImplementedError("eFormer stage-local PP updates do not yet support gradient_accumulation_steps > 1.")
    config = metadata.optimizer_config
    if config is None:
        raise NotImplementedError("Missing Muon optimizer config for PP stage-local update.")

    from optax.contrib._muon import orthogonalize_via_newton_schulz

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)
    partition_state = states[base_index]
    try:
        inner_states = dict(partition_state.inner_states)
        muon_masked_state = inner_states["muon"]
        adam_masked_state = inner_states["adam"]
        muon_state, muon_empty_state, muon_schedule_state = muon_masked_state.inner_state
        adam_state, adam_empty_state, adam_schedule_state = adam_masked_state.inner_state
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise NotImplementedError("eFormer stage-local Muon requires the standard optax.contrib.muon state.") from exc

    wd_state = states[weight_decay_index] if weight_decay_index is not None else None
    external_wd_count = wd_state.count if wd_state is not None else muon_schedule_state.count
    external_wd = _external_weight_decay(metadata, learning_rate_fn, external_wd_count)
    external_mask = _mask_or_true(metadata, params)
    muon_wd_mask = _resolve_mask_or_true(getattr(config, "weight_decay_mask", None), params)
    muon_lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, muon_schedule_state.count)
    adam_lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, adam_schedule_state.count)
    muon_count_inc = _safe_increment(muon_state.count)
    adam_count_inc = _safe_increment(adam_state.count)
    mu_dtype = None if getattr(config, "mu_dtype", None) is None else jnp.dtype(config.mu_dtype)

    def update_one(param, grad, muon_mu, adam_mu, adam_nu, muon_decay_enabled, external_decay_enabled):
        if grad is None:
            return param, muon_mu, adam_mu, adam_nu
        param = _place_array_like(param, grad)
        if getattr(param, "ndim", 0) == 2:
            old_mu = _place_array_like(muon_mu, grad)
            new_mu = (1.0 - float(config.beta)) * grad + float(config.beta) * old_mu
            if bool(config.nesterov):
                mu_hat = float(config.beta) * _bias_correction(
                    new_mu,
                    float(config.beta),
                    _safe_increment(muon_count_inc),
                ) + (1.0 - float(config.beta)) * _bias_correction(
                    grad,
                    float(config.beta),
                    muon_count_inc,
                )
            else:
                mu_hat = _bias_correction(new_mu, float(config.beta), muon_count_inc)
            update = orthogonalize_via_newton_schulz(
                mu_hat,
                _place_array_like(muon_state.ns_coeffs, grad),
                int(config.ns_steps),
                float(config.eps),
            )
            if bool(config.adaptive):
                update = jnp.einsum("ij,ij,ab->ab", mu_hat, update, update)
            update = jnp.sqrt(jnp.maximum(1.0, update.shape[-1] / update.shape[-2])).astype(update.dtype) * update
            if float(config.weight_decay) != 0.0:
                update = (
                    update
                    + jnp.where(
                        muon_decay_enabled,
                        jnp.asarray(config.weight_decay, update.dtype),
                        jnp.asarray(0.0, update.dtype),
                    )
                    * param
                )
            update = -_place_scalar_like(muon_lr, param).astype(update.dtype) * update
            if metadata.weight_decay != 0.0:
                update = (
                    update
                    - _place_scalar_like(
                        jnp.where(external_decay_enabled, external_wd, 0.0),
                        param,
                    ).astype(update.dtype)
                    * param
                )
            if mu_dtype is not None:
                new_mu = new_mu.astype(mu_dtype)
            return jnp.asarray(param + update).astype(param.dtype), new_mu, adam_mu, adam_nu

        old_mu = _place_array_like(adam_mu, grad)
        old_nu = _place_array_like(adam_nu, grad)
        new_mu = (1.0 - float(config.adam_b1)) * grad + float(config.adam_b1) * old_mu
        new_nu = (1.0 - float(config.adam_b2)) * (grad * grad) + float(config.adam_b2) * old_nu
        if bool(config.nesterov):
            mu_hat = float(config.adam_b1) * _bias_correction(
                new_mu,
                float(config.adam_b1),
                _safe_increment(adam_count_inc),
            ) + (1.0 - float(config.adam_b1)) * _bias_correction(
                grad,
                float(config.adam_b1),
                adam_count_inc,
            )
        else:
            mu_hat = _bias_correction(new_mu, float(config.adam_b1), adam_count_inc)
        nu_hat = _bias_correction(new_nu, float(config.adam_b2), adam_count_inc)
        update = mu_hat / (jnp.sqrt(nu_hat + float(config.adam_eps_root)) + float(config.eps))
        if float(config.adam_weight_decay) != 0.0:
            update = update + jnp.asarray(config.adam_weight_decay, update.dtype) * param
        update = -_place_scalar_like(adam_lr, param).astype(update.dtype) * update
        if metadata.weight_decay != 0.0:
            update = (
                update
                - _place_scalar_like(
                    jnp.where(external_decay_enabled, external_wd, 0.0),
                    param,
                ).astype(update.dtype)
                * param
            )
        if mu_dtype is not None:
            new_mu = new_mu.astype(mu_dtype)
        return jnp.asarray(param + update).astype(param.dtype), muon_mu, new_mu, new_nu.astype(old_nu.dtype)

    updated = jax.tree_util.tree_map(
        update_one,
        params,
        grads,
        muon_state.mu,
        adam_state.mu,
        adam_state.nu,
        muon_wd_mask,
        external_mask,
        is_leaf=lambda x: x is None or _is_masked_node(x),
    )
    new_params = jax.tree_util.tree_map(lambda x: x[0], updated, is_leaf=_tuple_size(4))
    new_muon_mu = jax.tree_util.tree_map(lambda x: x[1], updated, is_leaf=_tuple_size(4))
    new_adam_mu = jax.tree_util.tree_map(lambda x: x[2], updated, is_leaf=_tuple_size(4))
    new_adam_nu = jax.tree_util.tree_map(lambda x: x[3], updated, is_leaf=_tuple_size(4))

    inner_states["muon"] = muon_masked_state._replace(
        inner_state=(
            muon_state._replace(count=muon_count_inc, mu=new_muon_mu),
            muon_empty_state,
            muon_schedule_state._replace(count=_safe_increment(muon_schedule_state.count)),
        )
    )
    inner_states["adam"] = adam_masked_state._replace(
        inner_state=(
            adam_state._replace(count=adam_count_inc, mu=new_adam_mu, nu=new_adam_nu),
            adam_empty_state,
            adam_schedule_state._replace(count=_safe_increment(adam_schedule_state.count)),
        )
    )
    states[base_index] = partition_state._replace(inner_states=inner_states)
    if wd_state is not None:
        states[weight_decay_index] = wd_state._replace(count=_safe_increment(wd_state.count))
    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)


def _apply_quad_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    return _apply_white_kron_stage_local(
        metadata=metadata,
        variant="quad",
        params=params,
        grads=grads,
        opt_state=opt_state,
        learning_rate_fn=learning_rate_fn,
        delete_grads=delete_grads,
    )


def _apply_skew_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    return _apply_white_kron_stage_local(
        metadata=metadata,
        variant="skew",
        params=params,
        grads=grads,
        opt_state=opt_state,
        learning_rate_fn=learning_rate_fn,
        delete_grads=delete_grads,
    )


def _apply_white_kron_stage_local(
    *,
    metadata: StageLocalOptimizerMetadata,
    variant: tp.Literal["quad", "skew"],
    params: optax.Params,
    grads: optax.Updates,
    opt_state: optax.OptState,
    learning_rate_fn: optax.Schedule | None,
    delete_grads: bool,
) -> tuple[optax.Params, optax.OptState]:
    """Apply eFormer's WhiteKron (Quad/Skew) chain leafwise for stage-local PP buffers.

    WhiteKron optimizers use a Kronecker-factored preconditioner. This function
    extracts the preconditioner state, runs the preconditioner update (which
    may itself contain internal collectives), applies the resulting preconditioned
    gradients leafwise with learning-rate and weight-decay scaling, and advances
    the schedule counters.

    Args:
        metadata: Immutable hyperparameter metadata.
        params: Parameter pytree local to the current pipeline stage.
        grads: Gradient pytree matching ``params``.
        opt_state: Chained optimizer state tuple.
        learning_rate_fn: Optional schedule override.
        delete_grads: Whether to delete gradient buffers after update.

    Returns:
        A tuple ``(new_params, new_opt_state)``.

    Raises:
        NotImplementedError: If gradient accumulation is enabled or the
            WhiteKron config is missing.
    """

    if metadata.gradient_accumulation_steps != 1:
        raise NotImplementedError("eFormer stage-local PP updates do not yet support gradient_accumulation_steps > 1.")
    config = metadata.optimizer_config
    if config is None:
        raise NotImplementedError("Missing WhiteKron optimizer config for PP stage-local update.")

    from ._tx import scale_by_quad, scale_by_skew

    states, base_index, weight_decay_index = _chain_parts(metadata, opt_state)
    base_state = tuple(states[base_index])
    if len(base_state) not in (2, 3):
        raise NotImplementedError("eFormer stage-local WhiteKron requires the standard eFormer chain state.")

    precond_state = base_state[0]
    schedule_state = base_state[-1]
    scale_builder = scale_by_quad if variant == "quad" else scale_by_skew
    precond_tx = scale_builder(
        lr_style=config.lr_style,
        b1=config.b1,
        normalize_grads=config.normalize_grads,
        max_size_dense=config.max_size_dense,
        preconditioner_lr=config.preconditioner_lr,
        preconditioner_init_scale=config.preconditioner_init_scale,
        dtype=config.dtype,
        scanned_layers=config.scanned_layers,
        block_size=config.block_size,
        pipeline_axis_name=config.pipeline_axis_name,
        pipeline_axis_size=config.pipeline_axis_size,
        params_partition_specs=config.params_partition_specs,
        noise_scale=config.noise_scale,
    )
    precond_updates, new_precond_state = precond_tx.update(grads, precond_state, params)

    lr = _scheduled_scalar(learning_rate_fn, metadata.scheduler, schedule_state.count)
    external_wd_state = states[weight_decay_index] if weight_decay_index is not None else None
    external_wd_count = external_wd_state.count if external_wd_state is not None else schedule_state.count
    external_wd = _external_weight_decay(metadata, learning_rate_fn, external_wd_count)
    external_mask = _mask_or_true(metadata, params)
    internal_mask = _resolve_mask_or_true(getattr(config, "weight_decay_mask", None), params)

    def apply_one(param, update, internal_decay_enabled, external_decay_enabled):
        if update is None:
            return param
        param = _place_array_like(param, update)
        if float(config.weight_decay) != 0.0:
            update = (
                update
                + jnp.where(
                    internal_decay_enabled,
                    jnp.asarray(config.weight_decay, update.dtype),
                    jnp.asarray(0.0, update.dtype),
                )
                * param
            )
        update = -_place_scalar_like(lr, param).astype(update.dtype) * update
        if metadata.weight_decay != 0.0:
            update = (
                update
                - _place_scalar_like(
                    jnp.where(external_decay_enabled, external_wd, 0.0),
                    param,
                ).astype(update.dtype)
                * param
            )
        return jnp.asarray(param + update).astype(param.dtype)

    new_params = jax.tree_util.tree_map(
        apply_one,
        params,
        precond_updates,
        internal_mask,
        external_mask,
        is_leaf=lambda x: x is None,
    )

    new_schedule_state = schedule_state._replace(count=_safe_increment(schedule_state.count))
    states[base_index] = (new_precond_state, *base_state[1:-1], new_schedule_state)
    if external_wd_state is not None:
        states[weight_decay_index] = external_wd_state._replace(count=_safe_increment(external_wd_state.count))
    if delete_grads:
        _delete_tree_arrays(grads)
    return new_params, tuple(states)
