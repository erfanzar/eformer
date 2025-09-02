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

import abc
import time
import typing as tp
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from ..partition import get_incontext_mesh


class ShardingRule(abc.ABC):
    """Base class for sharding rules."""

    @abc.abstractmethod
    def apply(self, pytree: tp.Any) -> tp.Any:
        """Apply sharding rule to a pytree."""


class AutoShardingRule(ShardingRule):
    """Automatically determines sharding based on array shapes and mesh."""

    def __init__(
        self,
        mesh: Mesh | None = None,
        axis_names: list[str] | None = None,
        min_shard_size: int | None = None,
        reverse: bool = False,
    ):
        self.mesh = mesh or get_incontext_mesh()
        self.axis_names = axis_names or list(self.mesh.axis_names)
        self.min_shard_size = min_shard_size or np.prod(self.mesh.shape)
        self.reverse = reverse

    def _get_optimal_partition(self, array_shape: tuple[int, ...]) -> PartitionSpec:
        """Determines optimal partitioning for given array shape."""
        if np.prod(array_shape) < self.min_shard_size:
            return PartitionSpec()

        partition_spec = [None] * len(array_shape)
        remaining_axes = set(self.axis_names)

        # Sort dimensions by size (descending)
        dim_order = np.argsort([-d if not self.reverse else d for d in array_shape])

        for dim_idx in dim_order:
            dim_size = array_shape[dim_idx]

            # Find best matching mesh axis
            best_axis = None
            for axis in remaining_axes:
                mesh_size = self.mesh.shape[axis]
                if dim_size % mesh_size == 0:
                    best_axis = axis
                    break

            if best_axis:
                partition_spec[dim_idx] = best_axis
                remaining_axes.remove(best_axis)

            if not remaining_axes:
                break

        return PartitionSpec(*partition_spec)

    def apply(self, pytree: tp.Any) -> tp.Any:
        return jax.tree_util.tree_map(
            lambda x: self._get_optimal_partition(x.shape),
            pytree,
        )


class CompositeShardingRule(ShardingRule):
    """Combines multiple sharding rules with priority order."""

    def __init__(self, *rules: ShardingRule):
        self.rules = rules

    def apply(self, pytree: tp.Any) -> tp.Any:
        def combine_specs(*specs):
            # Return first non-empty PartitionSpec
            for spec in specs:
                if spec != PartitionSpec():
                    return spec
            return PartitionSpec()

        results = [rule.apply(pytree) for rule in self.rules]
        return jax.tree_util.tree_map(combine_specs, *results)


class MemoryConstrainedShardingRule(ShardingRule):
    """Creates sharding based on memory constraints."""

    def __init__(
        self,
        max_memory_per_device: int,
        mesh: Mesh | None = None,
        axis_names: list[str] | None = None,
    ):
        self.max_memory_per_device = max_memory_per_device
        self.mesh = mesh or get_incontext_mesh()
        self.axis_names = axis_names or list(self.mesh.axis_names)

    def _calculate_partition_spec(self, array: jnp.ndarray) -> PartitionSpec:
        array_size = np.prod(array.shape) * array.dtype.itemsize
        if array_size <= self.max_memory_per_device:
            return PartitionSpec()

        partition_spec = [None] * len(array.shape)
        remaining_size = array_size

        # Sort axes by mesh size (descending)
        sorted_axes = sorted(self.axis_names, key=lambda x: self.mesh.shape[x], reverse=True)

        # Sort dimensions by size (descending)
        dim_order = np.argsort([-d for d in array.shape])

        for dim_idx in dim_order:
            if remaining_size <= self.max_memory_per_device:
                break

            dim_size = array.shape[dim_idx]

            for axis in sorted_axes:
                mesh_size = self.mesh.shape[axis]
                if dim_size % mesh_size == 0:
                    partition_spec[dim_idx] = axis
                    remaining_size //= mesh_size
                    sorted_axes.remove(axis)
                    break

        return PartitionSpec(*partition_spec)

    def apply(self, pytree: tp.Any) -> tp.Any:
        return jax.tree_util.tree_map(self._calculate_partition_spec, pytree)


class ShapeBasedShardingRule(ShardingRule):
    """Creates sharding based on array shape patterns."""

    def __init__(self, shape_patterns: dict[tuple[int | None, ...], PartitionSpec]):
        self.shape_patterns = shape_patterns

    def _match_shape_pattern(self, array_shape: tuple[int, ...], pattern: tuple[int | None, ...]) -> bool:
        if len(array_shape) != len(pattern):
            return False
        return all(p is None or p == s for p, s in zip(pattern, array_shape, strict=False))

    def _get_partition_spec(self, array: jnp.ndarray) -> PartitionSpec:
        for pattern, spec in self.shape_patterns.items():
            if self._match_shape_pattern(array.shape, pattern):
                return spec
        return PartitionSpec()

    def apply(self, pytree: tp.Any) -> tp.Any:
        return jax.tree_util.tree_map(self._get_partition_spec, pytree)


class ShardingAnalyzer:
    """
    Analyzes and validates sharding strategies.

    Attributes:
            mesh (Mesh): The mesh configuration for sharding. If not provided, it defaults to the physical mesh from the
                thread resources.

    Methods:
            validate_partition_specs(pytree: tp.Any, partition_specs: tp.Any) -> tp.List[str]:
                    Validates the compatibility of partition specifications with the shapes of arrays in the pytree.
                    Args:
                            pytree (tp.Any): A pytree of arrays to be validated.
                            partition_specs (tp.Any): A pytree of partition specifications corresponding to the arrays.
                    Returns:
                            tp.List[str]: A list of issues found during validation. If empty, no issues were found.

            estimate_memory_usage(pytree: tp.Any, partition_specs: tp.Any) -> tp.Dict[str, int]:
                    Estimates the memory usage per device after applying the sharding strategy.
                    Args:
                            pytree (tp.Any): A pytree of arrays for which memory usage is to be estimated.
                            partition_specs (tp.Any): A pytree of partition specifications corresponding to the arrays.
                    Returns:
                            tp.Dict[str, int]: A dictionary containing the total memory size and the size per device.
    """

    def __init__(self, mesh: Mesh | None = None):
        self.mesh = mesh or get_incontext_mesh()

    def validate_partition_specs(self, pytree: tp.Any, partition_specs: tp.Any) -> list[str]:
        """Validates compatibility of partition specs with array shapes."""
        issues = []

        def validate_leaf(array: jnp.ndarray, spec: PartitionSpec):
            if spec == PartitionSpec():
                return

            for dim, axis_name in enumerate(spec):
                if axis_name is not None:
                    if array.shape[dim] % self.mesh.shape[axis_name] != 0:
                        issues.append(
                            f"Array shape {array.shape} not divisible by mesh "
                            f"axis {axis_name} size {self.mesh.shape[axis_name]}"
                        )

        jax.tree_util.tree_map(validate_leaf, pytree, partition_specs)
        return issues

    def estimate_memory_usage(self, pytree: tp.Any, partition_specs: tp.Any) -> dict[str, int]:
        """Estimates memory usage per device after sharding."""

        def calculate_size(array: jnp.ndarray, spec: PartitionSpec):
            size = np.prod(array.shape) * array.dtype.itemsize

            if spec != PartitionSpec():
                for axis_name in spec:
                    if axis_name is not None:
                        size //= self.mesh.shape[axis_name]

            return size

        total_size = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(calculate_size, pytree, partition_specs),
        )

        return {
            "total_size": total_size,
            "size_per_device": total_size // np.prod(self.mesh.shape),
        }


def create_monitored_function(
    fn: tp.Callable,
    partition_specs: tp.Any,
    analyzer: ShardingAnalyzer,
) -> tp.Callable:
    """Creates a monitored version of a function with sharding analysis."""

    def monitored_fn(*args, **kwargs):
        start_time = time.time()

        # Validate sharding before execution
        validation_issues = analyzer.validate_partition_specs(args[0], partition_specs)
        if validation_issues:
            warnings.warn(
                f"Sharding validation issues: {validation_issues}",
                stacklevel=1,
            )

        # Execute function
        result = fn(*args, **kwargs)
        execution_time = time.time() - start_time

        # Collect metrics
        metrics = {
            "execution_time": execution_time,
            "memory_usage": analyzer.estimate_memory_usage(args[0], partition_specs),
            "validation_issues": validation_issues,
        }

        return result, metrics

    return monitored_fn


def barrier_sync(timeout: float = 200):
    """Synchronize all JAX processes at a barrier point.

    Blocks execution until all processes in the distributed JAX runtime reach
    this barrier. This is essential for ensuring consistency across distributed
    training, especially before/after collective operations or checkpointing.

    The function uses a global counter to create unique barrier names, allowing
    multiple barriers to be used sequentially without conflicts.

    Args:
        timeout: Maximum time to wait for all processes to reach the barrier,
            in seconds. Defaults to 200 seconds (3.33 minutes). If the timeout
            is exceeded, a RuntimeError will be raised by the underlying JAX
            distributed client.

    Returns:
        None

    Raises:
        RuntimeError: If the JAX distributed client is not initialized. This
            typically means JAX was not started in distributed mode or the
            distributed runtime failed to initialize.

    Note:
        - This function is a no-op when running with a single process
          (jax.process_count() == 1), allowing code to work seamlessly
          in both single and multi-process environments.
        - Each call increments a global counter to ensure unique barrier names,
          preventing conflicts when multiple barriers are used in sequence.
        - The timeout is converted to milliseconds for the underlying JAX API.

    Example:
        >>> # In distributed training
        >>> model = train_step(model, batch)
        >>> barrier_sync()  # Wait for all processes
        >>> if jax.process_index() == 0:
        ...     save_checkpoint(model)  # Only process 0 saves
        >>> barrier_sync()  # Wait before continuing

        >>> # With custom timeout for long operations
        >>> barrier_sync(timeout=600)  # Wait up to 10 minutes

    Warning:
        Ensure all processes call barrier_sync() the same number of times and
        in the same order, or deadlocks may occur. Conditional barriers based
        on process rank should be avoided.
    """
    global _sync_counter
    if jax.process_count() == 1:
        return
    import jax._src.distributed as distributed

    client = distributed.global_state.client

    if client is None:
        raise RuntimeError("barrier_sync requires jax distributed client to be initialized")

    _sync_counter += 1
    client.wait_at_barrier(f"easy_barrier_sync_{_sync_counter}", timeout_in_ms=int(timeout * 1000.0))
