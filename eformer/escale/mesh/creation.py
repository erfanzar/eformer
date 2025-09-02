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
        >>> # With 8 global devices across 2 processes (4 devices each)
        >>> calculate_host_mesh_shape((2, 4), total_devices=4, num_processes=2)
        (1, 4)  # Each host gets half of the first dimension
    """
    total_devices = total_devices or jax.local_device_count()
    num_processes = num_processes or jax.process_count()
    total_mesh_size = np.prod(global_mesh_shape)
    assert total_mesh_size == total_devices * num_processes, (
        f"Mesh size {total_mesh_size} doesn't match available devices {total_devices * num_processes}"
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
    host_total = np.prod(host_mesh)
    assert host_total == total_devices, (
        f"Host mesh shape {tuple(host_mesh)} uses {host_total} devices instead of {total_devices}"
    )

    return tuple(host_mesh)


@functools.lru_cache
def _cached_mesh(
    axis_dims: tp.Sequence[int],
    axis_names: tp.Sequence[str],
    dcn_mesh_dims: tp.Sequence[int] | None = None,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = True,
    backend: str | None = None,
):
    """Create and cache a mesh configuration for distributed computation.

    Internal function that handles the complex logic of creating meshes for
    various distributed setups including multi-slice environments and hybrid
    device configurations. Results are cached for efficiency.

    Args:
        axis_dims: Dimensions for each mesh axis.
        axis_names: Names for each mesh axis (e.g., 'dp', 'tp', 'sp').
        dcn_mesh_dims: Data center network mesh dimensions for hybrid setups.
        process_is_granule: Whether to treat each process as a granule in
            the mesh creation.
        should_sort_granules_by_key: Whether to sort device granules by their
            keys for consistent ordering.
        allow_split_physical_axes: Whether to allow splitting physical device
            axes in the mesh.
        backend: JAX backend to use ('cpu', 'gpu', 'tpu'). If None, uses
            the default backend.

    Returns:
        JAX Mesh object configured for the specified parameters.

    Note:
        This function handles three main scenarios:
        1. Multi-slice environments (MEGASCALE_NUM_SLICES > 1)
        2. Multi-process setups with slice indices
        3. Single process or simple multi-device setups
    """
    backend = backend or jax.default_backend()
    num_devices = jax.device_count(backend)
    num_local_devices = jax.local_device_count(backend)
    if dcn_mesh_dims is None:
        mesh_shape = np.arange(num_devices).reshape(axis_dims).shape
    else:
        mesh_shape = np.arange(num_local_devices).reshape(axis_dims).shape
    num_slices = int(os.environ.get("MEGASCALE_NUM_SLICES", 1))
    multi_slice_env = num_slices > 1

    if multi_slice_env:
        if dcn_mesh_dims is None:
            dynamic_axis = None
            for i, dim in enumerate(mesh_shape):
                if dim % num_slices == 0:
                    dynamic_axis = i
                    break
            if dynamic_axis is None:
                raise ValueError("No axis in the mesh shape is divisible by num_slices")

            per_slice_mesh_shape = list(mesh_shape)
            per_slice_mesh_shape[dynamic_axis] //= num_slices
            per_slice_mesh_shape = tuple(per_slice_mesh_shape)

            dcn_mesh_dims = tuple(num_slices if i == dynamic_axis else 1 for i in range(len(mesh_shape)))
        else:
            per_slice_mesh_shape = mesh_shape
        ndarray = create_hybrid_device_mesh(
            mesh_shape=per_slice_mesh_shape,
            dcn_mesh_shape=dcn_mesh_dims,
            devices=jax.devices(backend),
            allow_split_physical_axes=allow_split_physical_axes,
            process_is_granule=process_is_granule,
            should_sort_granules_by_key=should_sort_granules_by_key,
        )

    elif jax.process_count() > 1 and hasattr(jax.devices()[0], "slice_index"):
        if dcn_mesh_dims is None:
            dcn_mesh_dims = calculate_host_mesh_shape(
                mesh_shape,
                jax.device_count(),
                jax.process_count(),
            )
        ndarray = create_hybrid_device_mesh(
            mesh_shape=mesh_shape,
            dcn_mesh_shape=dcn_mesh_dims,
            devices=jax.devices(backend),
            allow_split_physical_axes=allow_split_physical_axes,
            process_is_granule=process_is_granule,
            should_sort_granules_by_key=should_sort_granules_by_key,
        )
    else:
        ndarray = create_device_mesh(
            mesh_shape=mesh_shape,
            allow_split_physical_axes=True,
        )
    return Mesh(ndarray, axis_names)


def create_mesh(
    axis_dims: tp.Sequence[int] = DEFAULT_SHARDING_STG,
    axis_names: tp.Sequence[str] = DEFAULT_NAMED_SHARDING_STG,
    dcn_mesh_dims: tp.Sequence[int] | None = None,
    process_is_granule: bool = False,
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
        >>> # Create a simple 2D mesh for data and model parallelism
        >>> mesh = create_mesh(
        ...     axis_dims=(2, 4),
        ...     axis_names=('data', 'model')
        ... )
        >>> # Use with pjit
        >>> with mesh:
        ...     sharded_fn = pjit(fn, in_shardings=..., out_shardings=...)
    """
    return _cached_mesh(
        axis_dims=axis_dims,
        axis_names=axis_names,
        dcn_mesh_dims=dcn_mesh_dims,
        process_is_granule=process_is_granule,
        should_sort_granules_by_key=should_sort_granules_by_key,
        allow_split_physical_axes=allow_split_physical_axes,
        backend=backend,
    )


def parse_mesh_from_string(
    axis_dims: tp.Sequence[str],
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
        >>> # Named format
        >>> mesh = parse_mesh_from_string("dp:2,tp:4", ["dp", "tp"])
        >>>
        >>> # Positional format
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
        dim_names = names
    assert len(dims) == len(names), "Number of dimensions and names must match"

    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
    return create_mesh(mesh_shape, dim_names)


def create_cpu_mesh(
    axis_dims: tp.Sequence[int] = DEFAULT_SHARDING_STG,
    axis_names: tp.Sequence[str] = DEFAULT_NAMED_SHARDING_STG,
) -> Mesh:
    """Create a mesh using only CPU devices.

    Useful for debugging, testing, or when you want to force operations
    to run on CPU regardless of available accelerators.

    Args:
        axis_dims: Dimensions for each mesh axis. Default is (1, -1, 1, 1, 1).
            Note that for CPU, this typically uses just one device.
        axis_names: Names for each axis. Default is ('dp', 'fsdp', 'ep', 'tp', 'sp').

    Returns:
        JAX Mesh configured to use CPU device(s) only.

    Example:
        >>> # Create CPU mesh for testing
        >>> cpu_mesh = create_cpu_mesh()
        >>> with cpu_mesh:
        ...     # Operations here run on CPU
        ...     result = jax.jit(fn)(data)

    Note:
        This function always uses the first available CPU device and reshapes
        it according to axis_dims. Since CPU typically has one device, most
        axis dimensions should be 1.
    """
    return jax.sharding.Mesh(np.array([jax.local_devices(backend="cpu")[0]]).reshape(*axis_dims), axis_names)


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
        ...     # All JAX operations here run on CPU
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
        ...     # All operations here run on CPU with CPU mesh
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


if __name__ == "__main__":
    test_cases = [
        ((1, 1, 32), 4, 8, (1, 1, 4)),
        ((8, 4), 4, 8, (1, 4)),
        ((1, 1, 8, 4), 4, 8, (1, 1, 1, 4)),
        ((2, 4, 8), 8, 8, (1, 1, 8)),
        ((16, 4), 4, 16, (1, 4)),
    ]

    for global_mesh, devices, processes, expected in test_cases:
        mesh_size = np.prod(global_mesh)
        device_total = devices * processes
        assert mesh_size == device_total, f"Mesh size {mesh_size} must equal total devices {device_total}"
        result = calculate_host_mesh_shape(global_mesh, devices, processes)
        assert result == expected, f"Failed for {global_mesh}: expected {expected}, got {result}"
