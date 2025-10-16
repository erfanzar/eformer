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


"""JAX mesh creation utilities for distributed computation.

This module provides utilities for creating and managing JAX meshes for distributed
computation across multiple devices. It supports various parallelism strategies including:

- Data Parallelism (dp): Replicates model across devices, splits data
- Fully Sharded Data Parallelism (fsdp): Shards both model and data across devices
- Expert Parallelism (ep): For mixture-of-experts models
- Tensor Parallelism (tp): Splits individual tensors across devices
- Sequence Parallelism (sp): Splits sequence dimension across devices

Key Features:
    - Automatic device mesh creation for single/multi-host setups
    - Support for TPU slices and multi-process environments
    - CPU-specific utilities for debugging and testing
    - String-based mesh configuration parsing
    - Caching for efficient mesh reuse

Typical Usage:
    >>>
    >>> mesh = create_mesh(
    ...     axis_dims=(2, 4),
    ...     axis_names=('data', 'model')
    ... )
    >>> with mesh:
    ...
    ...     pass

    >>>
    >>> mesh = parse_mesh_from_string("dp:2,tp:4", ["dp", "tp"])

    >>>
    >>> with cpu_context() as mesh:
    ...
    ...     pass
"""

import functools
import os
import typing as tp

import contextlib2
import jax
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh, create_hybrid_device_mesh
from jax.sharding import Mesh

DEFAULT_SHARDING_STG = (1, -1, 1, 1, 1)
DEFAULT_NAMED_SHARDING_STG = ("dp", "fsdp", "ep", "tp", "sp")


def calculate_host_mesh_shape(
    global_mesh_shape: tp.Sequence[int],
    total_devices: int | None = None,
    num_processes: int | None = None,
):
    """Calculate the mesh shape for the local host in a distributed setting.

    Determines how to split a global mesh shape across multiple processes,
    ensuring each host gets an appropriate portion of the mesh.

    Args:
        global_mesh_shape: The desired global mesh shape across all processes.
        total_devices: Total number of devices on this host. If None, uses
            jax.local_device_count().
        num_processes: Total number of processes in the distributed setup.
            If None, uses jax.process_count().

    Returns:
        Tuple representing the mesh shape for this host.

    Raises:
        AssertionError: If mesh size doesn't match available devices or if
            the calculated host mesh doesn't use the correct number of devices.

    Example:
        >>>
        >>> calculate_host_mesh_shape((2, 4), total_devices=4, num_processes=2)
        (1, 4)
    """
    total_devices = total_devices or jax.local_device_count()
    num_processes = num_processes or jax.process_count()
    total_mesh_size = int(np.prod(global_mesh_shape))
    assert total_mesh_size == total_devices * num_processes, (
        f"Mesh size {total_mesh_size} doesn't match available devices "
        f"{total_devices * num_processes} (local x processes)"
    )
    host_mesh = list(global_mesh_shape)
    remaining_process_split = num_processes
    idx = 0

    while remaining_process_split > 1 and idx < len(host_mesh):
        dim_size = host_mesh[idx]
        if dim_size >= remaining_process_split:
            factor = remaining_process_split
            host_mesh[idx] = dim_size // factor
            remaining_process_split = 1
        else:
            factor = dim_size
            host_mesh[idx] = 1
            remaining_process_split = remaining_process_split // factor
        idx += 1
    host_total = int(np.prod(host_mesh))
    assert host_total == total_devices, (
        f"Host mesh shape {tuple(host_mesh)} uses {host_total} devices instead of {total_devices}. "
        "Ensure that num_processes factors the global mesh shape."
    )

    return tuple(host_mesh)


def _cached_mesh(
    axis_dims: tp.Sequence[int],
    axis_names: tp.Sequence[str],
    dcn_mesh_dims: tp.Sequence[int] | None = None,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = True,
    backend: str | None = None,
):
    """Wrapper that normalizes arguments and feeds the cached implementation.

    This function converts sequences to tuples for hashability and delegates
    to the cached implementation. The caching ensures that identical mesh
    configurations reuse the same mesh object, improving performance.

    Args:
        axis_dims: Dimensions for each mesh axis
        axis_names: Names for each mesh axis
        dcn_mesh_dims: Data center network mesh dimensions
        should_sort_granules_by_key: Whether to sort device granules
        allow_split_physical_axes: Whether to allow splitting physical axes
        backend: JAX backend to use

    Returns:
        Cached JAX Mesh object
    """

    axis_dims_t = tuple(axis_dims)
    axis_names_t = tuple(axis_names)
    dcn_mesh_dims_t = None if dcn_mesh_dims is None else tuple(dcn_mesh_dims)
    backend_s = backend or jax.default_backend()
    return _cached_mesh_impl(
        axis_dims=axis_dims_t,
        axis_names=axis_names_t,
        dcn_mesh_dims=dcn_mesh_dims_t,
        should_sort_granules_by_key=should_sort_granules_by_key,
        allow_split_physical_axes=allow_split_physical_axes,
        backend=backend_s,
    )


@functools.cache
def _cached_mesh_impl(
    axis_dims: tuple[int, ...],
    axis_names: tuple[str, ...],
    dcn_mesh_dims: tuple[int, ...] | None = None,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = True,
    backend: str = "cpu",
):
    """Cached implementation of mesh creation logic.

    This function handles three main scenarios:
    1. Multi-slice environments (TPU pods): Creates per-slice meshes with
       appropriate DCN configuration for inter-slice communication.
    2. Multi-process environments: Distributes mesh across processes,
       calculating DCN dimensions to map logical to physical topology.
    3. Single-process environments: Creates a simple device mesh.

    The function automatically detects the environment type and applies
    the appropriate mesh creation strategy.

    Args:
        axis_dims: Tuple of dimensions for each mesh axis
        axis_names: Tuple of names for each mesh axis
        dcn_mesh_dims: Data center network dimensions for hybrid setups
        should_sort_granules_by_key: Sort devices for consistency
        allow_split_physical_axes: Allow splitting physical device axes
        backend: Backend to use ('cpu', 'gpu', 'tpu')

    Returns:
        JAX Mesh configured for the detected environment

    Raises:
        ValueError: If mesh configuration is invalid for the environment
    """
    devices = jax.devices(backend)
    total_devices = jax.device_count(backend)
    local_devices = jax.local_device_count(backend)
    process_count = jax.process_count()

    num_slices = 1
    if devices and hasattr(devices[0], "slice_index"):
        try:
            num_slices = len({d.slice_index for d in devices})
        except Exception:
            pass
    if num_slices == 1:
        num_slices = int(os.environ.get("MEGASCALE_NUM_SLICES", num_slices))

    def fill_minus_one_to_target(shape: tuple[int, ...], target: int) -> tuple[int, ...]:
        """Replace -1 in shape with value to match target product.

        Allows using -1 as a placeholder in dcn_mesh_dims to automatically
        calculate the appropriate dimension size.

        Args:
            shape: Shape tuple potentially containing one -1
            target: Target product all dimensions should multiply to

        Returns:
            Shape tuple with -1 replaced by calculated value

        Raises:
            ValueError: If multiple -1s exist or product doesn't match target
        """
        shp = list(shape)
        minus = [i for i, v in enumerate(shp) if v == -1]
        if len(minus) > 1:
            raise ValueError("Only one -1 is supported in dcn_mesh_dims.")
        prod_known = 1
        for v in shp:
            if v != -1:
                if v <= 0:
                    raise ValueError(f"dcn_mesh_dims entries must be > 0 or -1, got {v}")
                prod_known *= v
        if minus:
            if target % prod_known != 0:
                raise ValueError(f"dcn_mesh_dims product ({prod_known}) does not divide target ({target}).")
            shp[minus[0]] = target // prod_known
        if np.prod(shp) != target:
            raise ValueError(f"dcn_mesh_dims product {int(np.prod(shp))} must equal {target}; got {tuple(shp)}")
        return tuple(int(v) for v in shp)

    if num_slices > 1:
        global_mesh_shape = np.arange(total_devices).reshape(axis_dims).shape

        dynamic_axis = next((i for i, dim in enumerate(global_mesh_shape) if dim % num_slices == 0), None)
        if dynamic_axis is None:
            raise ValueError(
                f"Multi-slice detected (num_slices={num_slices}) but no mesh axis in "
                f"{global_mesh_shape} is divisible by num_slices."
            )

        per_slice_mesh_shape = list(global_mesh_shape)
        per_slice_mesh_shape[dynamic_axis] //= num_slices
        per_slice_mesh_shape = tuple(per_slice_mesh_shape)

        if dcn_mesh_dims is None:
            dcn_list = [1] * len(axis_dims)
            dcn_list[dynamic_axis] = num_slices
            dcn = tuple(dcn_list)
        else:
            dcn = fill_minus_one_to_target(dcn_mesh_dims, num_slices)

        ndarray = create_hybrid_device_mesh(
            mesh_shape=per_slice_mesh_shape,
            dcn_mesh_shape=dcn,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
            process_is_granule=False,
            should_sort_granules_by_key=should_sort_granules_by_key,
        )

    elif process_count > 1:
        local_mesh_shape = np.arange(local_devices).reshape(axis_dims).shape

        if dcn_mesh_dims is None:
            global_mesh_shape = np.arange(total_devices).reshape(axis_dims).shape

            ratios = [int(g // le) for g, le in zip(global_mesh_shape, local_mesh_shape, strict=False)]
            if np.prod(ratios) != process_count:
                ratios = [1] * len(axis_dims)
                for i in range(len(axis_dims)):
                    ratios[i] = process_count
                    break
            dcn = tuple(ratios)
        else:
            dcn = fill_minus_one_to_target(dcn_mesh_dims, process_count)

        ndarray = create_hybrid_device_mesh(
            mesh_shape=local_mesh_shape,
            dcn_mesh_shape=dcn,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
            process_is_granule=True,
            should_sort_granules_by_key=should_sort_granules_by_key,
        )

    else:
        global_mesh_shape = np.arange(total_devices).reshape(axis_dims).shape
        ndarray = create_device_mesh(
            mesh_shape=global_mesh_shape,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )

    return Mesh(ndarray, axis_names)


def create_mesh(
    axis_dims: tp.Sequence[int] = DEFAULT_SHARDING_STG,
    axis_names: tp.Sequence[str] = DEFAULT_NAMED_SHARDING_STG,
    dcn_mesh_dims: tp.Sequence[int] | None = None,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = True,
    backend: str | None = None,
) -> Mesh:
    """Create a JAX mesh for distributed computation.

    Creates a mesh that maps logical mesh axes to physical devices, supporting
    various parallelism strategies including data, tensor, sequence, and pipeline
    parallelism.

    Args:
        axis_dims: Dimensions for each mesh axis. Default is (1, -1, 1, 1, 1)
            where -1 means use all remaining devices.
        axis_names: Names for each axis. Default is ('dp', 'fsdp', 'ep', 'tp', 'sp')
            representing data, fully-sharded data, expert, tensor, and sequence
            parallelism respectively.
        dcn_mesh_dims: Data center network mesh dimensions for hybrid device setups.
            If None, automatically calculated for multi-process environments.
        process_is_granule: Whether to treat each process as an indivisible unit
            in mesh creation.
        should_sort_granules_by_key: Whether to sort device granules for consistent
            ordering across processes.
        allow_split_physical_axes: Whether physical device axes can be split
            across logical mesh axes.
        backend: JAX backend ('cpu', 'gpu', 'tpu'). If None, uses default.

    Returns:
        JAX Mesh object ready for use with pjit and sharding specifications.

    Example:
        >>>
        >>> mesh = create_mesh(
        ...     axis_dims=(2, 4),
        ...     axis_names=('data', 'model')
        ... )
        >>>
        >>> with mesh:
        ...     sharded_fn = pjit(fn, in_shardings=..., out_shardings=...)
    """
    return _cached_mesh(
        axis_dims=axis_dims,
        axis_names=axis_names,
        dcn_mesh_dims=dcn_mesh_dims,
        should_sort_granules_by_key=should_sort_granules_by_key,
        allow_split_physical_axes=allow_split_physical_axes,
        backend=backend,
    )


def parse_mesh_from_string(
    axis_dims: str,
    names: tp.Sequence[str],
) -> Mesh:
    """Parse mesh configuration from string representation.

    Supports two formats:
    1. Named format: "dp:2,tp:4" - explicitly maps names to dimensions
    2. Positional format: "2,4" - maps dimensions to names by position

    Args:
        axis_dims: String representation of axis dimensions. Either:
            - Named: "name1:dim1,name2:dim2,..." (e.g., "dp:2,tp:4")
            - Positional: "dim1,dim2,..." (e.g., "2,4")
        names: Sequence of axis names that should appear in the mesh.

    Returns:
        JAX Mesh configured according to the string specification.

    Raises:
        AssertionError: If axis names don't match, dimensions and names have
            different lengths, or unknown axis names are used.

    Example:
        >>>
        >>> mesh = parse_mesh_from_string("dp:2,tp:4", ["dp", "tp"])
        >>>
        >>>
        >>> mesh = parse_mesh_from_string("2,4", ["data", "model"])
    """
    if ":" in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(","):
            name, dim = axis.split(":")
            assert name in names, f"Axis name '{name}' not found in provided names: {names}"
            dims.append(int(dim))
            dim_names.append(name)
        assert set(dim_names) == set(names), "Not all axis names were used in 'axis_dims'"
    else:
        dims = [int(x) for x in axis_dims.split(",")]
        dim_names = list(names)
    assert len(dims) == len(names), "Number of dimensions and names must match"

    return create_mesh(tuple(dims), tuple(dim_names))


def create_cpu_mesh(
    axis_dims: tp.Sequence[int] = DEFAULT_SHARDING_STG,
    axis_names: tp.Sequence[str] = DEFAULT_NAMED_SHARDING_STG,
) -> Mesh:
    """Create a mesh using CPU devices.

    Useful for debugging, testing, or when you want to force operations
    to run on CPU regardless of available accelerators.

    Args:
        axis_dims: Dimensions for each mesh axis. Default is (1, -1, 1, 1, 1).
            For CPU, this typically resolves to a shape matching the number of
            available CPU devices.
        axis_names: Names for each axis. Default is ('dp', 'fsdp', 'ep', 'tp', 'sp').

    Returns:
        JAX Mesh configured to use CPU device(s).

    Note:
        This uses all available CPU devices on the host and arranges them
        according to axis_dims.
    """
    return create_mesh(axis_dims=tuple(axis_dims), axis_names=tuple(axis_names), backend="cpu")


@contextlib2.contextmanager
def force_cpu():
    """Context manager that forces JAX operations to run on CPU.

    Temporarily sets the default JAX device to CPU for all operations
    within the context. Useful for debugging or when specific operations
    need to run on CPU.

    Yields:
        The CPU device being used.

    Example:
        >>> with force_cpu() as cpu_device:
        ...
        ...     result = jax.numpy.sum(array)
        ...     print(f"Running on {cpu_device}")

    Note:
        Device setting is restored when exiting the context.
    """
    cpu = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu):
        yield cpu


@contextlib2.contextmanager
def cpu_context():
    """Context manager that provides both CPU mesh and forces CPU execution.

    Combines force_cpu() and create_cpu_mesh() to provide a complete CPU
    execution environment. This ensures both that operations run on CPU
    and that they use a CPU-configured mesh.

    Yields:
        The CPU mesh created for the context.

    Example:
        >>> with cpu_context() as mesh:
        ...
        ...     @jax.jit
        ...     def fn(x):
        ...         return x * 2
        ...     result = fn(jax.numpy.ones((4, 4)))

    Note:
        This is particularly useful for:
        - Unit testing that needs deterministic CPU behavior
        - Debugging distributed code on a single machine
        - Prototyping before deploying to accelerators
    """
    mesh = create_cpu_mesh()
    with force_cpu(), mesh:
        yield mesh
