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
import os
import re
import typing as tp
import warnings
from functools import partial

import chex
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import tree_util as tu
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from eformer.common_types import AxisType
from eformer.pytree import named_tree_map

MIN_SHARDING_SIZE = int(os.environ.get("MIN_SHARDING_SIZE", "16384"))
LOG_SHARDING_MOVE = os.environ.get("LOG_SHARDING_MOVE", "false") in [
    "true",
    "yes",
    "1",
    "on",
]


def names_in_current_mesh(*names: str) -> bool:
    """
    Check if the given names are present in the current JAX mesh.

    Args:
        *names: Variable number of axis names to check.

    Returns:
        True if all given names are present in the current mesh, False otherwise.
    """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def make_shard_and_gather_fns(
    partition_specs: dict[str, PartitionSpec],
    mesh: Mesh | None = None,
) -> tuple[dict[str, tp.Callable], dict[str, tp.Callable]]:
    """
    Create shard and gather functions based on given partition specs and mesh.

    This function generates dictionaries of shard and gather functions that can be used
    to distribute and collect arrays across a JAX mesh. The functions are specifically
    designed for use with Flax's `tu.tree_map`.

    Args:
            partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.
            mesh: The JAX mesh to use for sharding. If None, the current mesh is used.

    Returns:
            A tuple containing two dictionaries:
                    - `shard_fns`: A dictionary mapping parameter names to their corresponding shard functions.
                    - `gather_fns`: A dictionary mapping parameter names to their corresponding gather functions.
    """
    if mesh is None:
        mesh = get_incontext_mesh()

    named_shardings = tu.tree_map(
        lambda p: NamedSharding(mesh=mesh, spec=p),
        partition_specs,
    )

    def make_shard_fn(sharding: NamedSharding) -> tp.Callable:
        """
        Create a shard function for a specific partition spec.
        """
        if jax.process_count() > 1:

            @partial(jax.jit, out_shardings=sharding)
            def _self_shard(tensor):
                return jnp.asarray(tensor)

            def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
                with mesh:
                    tensor = jax.block_until_ready(_self_shard(tensor))
                    assert tensor.sharding == sharding, "sharding Failed!."
                return tensor

            return shard_fn
        else:

            def shard_fn(tensor: jnp.ndarray) -> jnp.ndarray:
                with mesh:
                    tensor = with_sharding_constraint(tensor, sharding=sharding)
                return tensor

            return shard_fn

    def make_gather_fn(sharding: NamedSharding) -> tp.Callable:
        """
        Create a gather function for a specific partition spec.
        """

        @partial(jax.jit, out_shardings=NamedSharding(mesh=mesh, spec=PartitionSpec()))
        def _self_gather(tensor):
            return jnp.asarray(tensor)

        def gather_fn(tensor: jnp.ndarray) -> jnp.ndarray:
            return jax.device_get(jax.block_until_ready(_self_gather(tensor)))

        return gather_fn

    shard_fns = tu.tree_map(make_shard_fn, named_shardings)
    gather_fns = tu.tree_map(make_gather_fn, named_shardings)
    return shard_fns, gather_fns


def get_names_from_partition_spec(
    partition_specs: dict[str, PartitionSpec],
) -> list[str]:
    """
    Extract axis names from a partition specification.

    This function recursively iterates through the provided `partition_specs`
    dictionary and extracts all unique axis names used in the sharding specifications.

    Args:
            partition_specs: A dictionary mapping parameter names to their respective `PartitionSpec`.

    Returns:
            A list of unique axis names used in the partition specs.
    """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))
    return list(names)


def array_with_sharding_constraint(arr: jax.Array, sharding: PartitionSpec | NamedSharding) -> jax.Array:
    """
    Apply sharding constraints with automatic correction based on array shape and mesh.

    This function takes a JAX array and a sharding specification (PartitionSpec or
    NamedSharding). It attempts to apply the sharding, but first checks if the
    specification is compatible with the array's shape and the current mesh configuration.

    If an axis specified in the PartitionSpec:
      - Does not exist in the mesh,
      - Corresponds to a mesh axis of size 1, or
      - Is incompatible with the array's dimension size (not divisible),
    then that part of the PartitionSpec is automatically corrected to None, effectively
    preventing sharding along that dimension.

    Args:
        arr: The JAX array to apply sharding constraints to.
        sharding: The desired sharding specification (PartitionSpec or NamedSharding).

    Returns:
        The JAX array with potentially corrected sharding constraints applied.
    """
    if not isinstance(arr, jax.Array | jnp.ndarray):
        return arr
    if isinstance(sharding, NamedSharding):
        mesh = sharding.mesh
        original_spec = sharding.spec
    elif isinstance(sharding, PartitionSpec):
        mesh = get_incontext_mesh(False)
        original_spec = sharding
    else:
        raise TypeError(f"Unsupported sharding type: {type(sharding)}")

    if mesh.empty:
        if LOG_SHARDING_MOVE:
            warnings.warn(
                "Attempted to apply sharding constraint with an empty mesh. Constraint ignored.",
                stacklevel=1,
            )
        return arr

    if len(original_spec) == 0:
        return arr
    spec_tuple = tuple(original_spec)
    if len(spec_tuple) < arr.ndim:
        spec_tuple += (None,) * (arr.ndim - len(spec_tuple))
    elif len(spec_tuple) > arr.ndim:
        if LOG_SHARDING_MOVE:
            warnings.warn(
                f"PartitionSpec length ({len(spec_tuple)}) exceeds array rank ({arr.ndim}). "
                f"Truncating spec: {original_spec} -> {spec_tuple[: arr.ndim]}",
                stacklevel=1,
            )
        spec_tuple = spec_tuple[: arr.ndim]

    corrected_spec_list = list(spec_tuple)
    mesh_axis_names = set(mesh.axis_names)

    for i, axis_spec in enumerate(spec_tuple):
        if axis_spec is None:
            continue

        current_axis_names = []
        if isinstance(axis_spec, str):
            current_axis_names.append(axis_spec)
        elif isinstance(axis_spec, tuple):
            current_axis_names.extend(axis_spec)
        else:
            if LOG_SHARDING_MOVE:
                warnings.warn(
                    f"Unexpected element type in PartitionSpec at index {i}: {axis_spec}. Treating as None.",
                    stacklevel=1,
                )
            corrected_spec_list[i] = None
            continue

        valid_axis = True
        total_mesh_size_for_dim = 1
        for axis_name in current_axis_names:
            if axis_name not in mesh_axis_names:
                if LOG_SHARDING_MOVE:
                    warnings.warn(
                        f"Axis name '{axis_name}' in PartitionSpec {original_spec} at index {i} "
                        f"not found in mesh axes {mesh_axis_names}. Correcting dimension {i} to None.",
                        stacklevel=1,
                    )
                valid_axis = False
                break
            total_mesh_size_for_dim *= mesh.shape[axis_name]
        if not valid_axis:
            corrected_spec_list[i] = None
            continue
        if total_mesh_size_for_dim > 0 and arr.shape[i] % total_mesh_size_for_dim != 0:
            if LOG_SHARDING_MOVE:
                warnings.warn(
                    f"Array dimension {i} (size {arr.shape[i]}) is not divisible by the total mesh "
                    f"size ({total_mesh_size_for_dim}) for axis spec {axis_spec} in {original_spec}. "
                    f"Correcting to None.",
                    stacklevel=1,
                )
            corrected_spec_list[i] = None
            continue
        elif total_mesh_size_for_dim == 0:
            if LOG_SHARDING_MOVE:
                warnings.warn(
                    f"Total mesh axis size for dimension {i} based on spec {axis_spec} resulted in 0. "
                    f"Correcting to None.",
                    stacklevel=1,
                )
            corrected_spec_list[i] = None
            continue
    corrected_spec = PartitionSpec(*corrected_spec_list)
    if not any(axis is not None for axis in corrected_spec):
        final_spec_to_apply = PartitionSpec()
    else:
        final_spec_to_apply = corrected_spec
    with mesh:
        arr = _with_sharding_constraint(arr, final_spec_to_apply)
    return arr


def with_sharding_constraint(
    arr: tp.Any,
    sharding: PartitionSpec | NamedSharding,
) -> tp.Any:
    """
    Apply sharding constraints with automatic correction based on array shape and mesh.

    This function takes a PyTree of JAX arrays and a sharding specification (PartitionSpec or
    NamedSharding). It attempts to apply the sharding, but first checks if the
    specification is compatible with the array's shape and the current mesh configuration.

    If an axis specified in the PartitionSpec:
      - Does not exist in the mesh,
      - Corresponds to a mesh axis of size 1, or
      - Is incompatible with the array's dimension size (not divisible),
    then that part of the PartitionSpec is automatically corrected to None, effectively
    preventing sharding along that dimension.

    Args:
        x: The PyTree of JAX arrays to apply sharding constraints to.
        sharding: PyTree of the desired sharding specification (PartitionSpec or NamedSharding).

    Returns:
        The PyTree of JAX arrays with potentially corrected sharding constraints applied.
    """
    flat_x, treedef = jax.tree_util.tree_flatten(arr)
    flat_sharding, other_treedef = jax.tree_util.tree_flatten(sharding)

    if treedef != other_treedef:
        raise ValueError(
            f"Input PyTree and sharding specification must have the same structure. "
            f"Got {treedef} and {other_treedef}."
        )
    del other_treedef

    if not all(isinstance(s, (PartitionSpec, NamedSharding)) for s in flat_sharding):
        raise TypeError(
            "All elements in the sharding specification must be either PartitionSpec or NamedSharding."
        )

    flat_x = [
        array_with_sharding_constraint(arr, sharding)
        for arr, sharding in zip(flat_x, flat_sharding)
    ]
    return jax.tree_util.tree_unflatten(treedef, flat_x)


def get_corrected_named_sharding(
    shape: tuple[int, ...],
    partition_spec: PartitionSpec,
    raise_mesh_error: bool = True,
) -> NamedSharding:
    """
    Calculates the corrected PartitionSpec based on shape and mesh, returns NamedSharding.

    This function takes an array shape and a desired PartitionSpec.
    It determines the effective PartitionSpec by correcting the input based on:
      - Axis names present in the current mesh.
      - Divisibility of array dimensions by the product of corresponding mesh axis sizes.

    It does NOT correct based on mesh axes having size 1, allowing such axes
    to persist in the spec if explicitly provided and divisibility holds.

    Args:
        shape: The shape of the target JAX array.
        partition_spec: The desired PartitionSpec.
        raise_mesh_error: If True, raises an error if no mesh is active.
                          If False, returns a replicated NamedSharding on an
                          empty mesh if no mesh is found.

    Returns:
        A NamedSharding object containing the current mesh and the corrected
        PartitionSpec.

    Raises:
        AssertionError: If no mesh is active and raise_mesh_error is True.
    """
    try:
        mesh = get_incontext_mesh(raise_error=raise_mesh_error)
    except AssertionError:
        if raise_mesh_error:
            raise
        else:
            # Create a dummy empty mesh to return replicated sharding
            mesh = Mesh(np.empty((0,), dtype=np.int32), [])
            warnings.warn(
                "No active mesh found. Returning replicated NamedSharding on empty mesh.",
                stacklevel=2,
            )
            return NamedSharding(mesh, PartitionSpec())  # Replicated on empty mesh

    if mesh.empty:
        warnings.warn(
            "Active mesh is empty. Returning replicated NamedSharding.",
            stacklevel=2,
        )
        return NamedSharding(mesh, PartitionSpec())

    ndim = len(shape)
    original_spec = partition_spec  # Keep original name for clarity

    if len(original_spec) == 0:
        return NamedSharding(mesh, PartitionSpec())

    spec_tuple = tuple(original_spec)
    if len(spec_tuple) < ndim:
        spec_tuple += (None,) * (ndim - len(spec_tuple))
    elif len(spec_tuple) > ndim:
        if LOG_SHARDING_MOVE:
            warnings.warn(
                f"PartitionSpec length ({len(spec_tuple)}) exceeds array rank ({ndim}). "
                f"Truncating spec: {original_spec} -> {spec_tuple[:ndim]}",
                stacklevel=2,
            )
        spec_tuple = spec_tuple[:ndim]

    corrected_spec_list = list(spec_tuple)
    mesh_axis_names = set(mesh.axis_names)

    for i, axis_spec in enumerate(spec_tuple):
        if axis_spec is None:
            continue

        current_axis_names = []
        if isinstance(axis_spec, str):
            current_axis_names.append(axis_spec)
        elif isinstance(axis_spec, tuple):
            current_axis_names.extend(axis_spec)
        else:
            if LOG_SHARDING_MOVE:
                warnings.warn(
                    f"Unexpected element type in PartitionSpec at index {i}: {axis_spec}. Treating as None.",
                    stacklevel=2,
                )
            corrected_spec_list[i] = None
            continue

        valid_axis = True
        total_mesh_size_for_dim = 1
        for axis_name in current_axis_names:
            if axis_name not in mesh_axis_names:
                if LOG_SHARDING_MOVE:
                    warnings.warn(
                        f"Axis name '{axis_name}' in PartitionSpec {original_spec} at index {i} "
                        f"not found in mesh axes {mesh_axis_names}. Correcting dimension {i} to None.",
                        stacklevel=2,
                    )
                valid_axis = False
                break
            total_mesh_size_for_dim *= mesh.shape[axis_name]
        if not valid_axis:
            corrected_spec_list[i] = None
            continue

        if total_mesh_size_for_dim > 0 and shape[i] % total_mesh_size_for_dim != 0:
            if LOG_SHARDING_MOVE:
                warnings.warn(
                    f"Array dimension {i} (size {shape[i]}) is not divisible by the total mesh "
                    f"size ({total_mesh_size_for_dim}) for axis spec {axis_spec} in {original_spec}. "
                    f"Correcting to None.",
                    stacklevel=2,
                )
            corrected_spec_list[i] = None
            continue
        elif total_mesh_size_for_dim == 0:
            if LOG_SHARDING_MOVE:
                warnings.warn(
                    f"Total mesh axis size for dimension {i} based on spec {axis_spec} resulted in 0. "
                    f"Correcting to None.",
                    stacklevel=2,
                )
            corrected_spec_list[i] = None
            continue

    corrected_spec = PartitionSpec(*corrected_spec_list)
    if not any(axis is not None for axis in corrected_spec):
        final_spec_to_apply = PartitionSpec()
    else:
        final_spec_to_apply = corrected_spec

    return NamedSharding(mesh, final_spec_to_apply)


def match_partition_rules(
    rules: list[tuple[str, PartitionSpec]],
    tree: dict,
    min_size: int | None = 0,
) -> dict:
    """
    Match partition rules to parameters based on their names.

    This function takes a list of partition rules (regular expressions and
    corresponding `PartitionSpec`) and applies them to a dictionary of parameters
    based on their names. It's useful for automatically defining sharding strategies.

    Args:
            rules: A list of tuples, where each tuple contains:
                             - A regular expression to match parameter names.
                             - A `PartitionSpec` to apply if the name matches.
            tree: A dictionary of parameters, where keys are parameter names.

    Returns:
            A dictionary with the same keys as `tree`, but values are replaced
            with the corresponding `PartitionSpec` based on matching rules.
    """

    min_size = min_size if min_size is not None else MIN_SHARDING_SIZE

    def get_partition_spec(name: str, leaf: jnp.ndarray) -> PartitionSpec:
        """
        Determine the partition spec for a parameter based on its name.
        """

        if not hasattr(leaf, "shape"):
            return PartitionSpec()
        size = np.prod(leaf.shape)
        if len(leaf.shape) == 0:
            """ Don't partition scalar values. """
            return PartitionSpec()

        for rule, ps in rules:
            if re.search(rule, name) is not None:
                if size < min_size:
                    if LOG_SHARDING_MOVE:
                        warnings.warn(
                            f"PartitionSpec Related to {name} was safer and faster being local array.",
                            stacklevel=1,
                        )
                    return PartitionSpec()
                if len(ps) > leaf.ndim:
                    ps = PartitionSpec(*tuple(ps[: leaf.ndim]))
                    if LOG_SHARDING_MOVE:
                        warnings.warn(
                            f"PartitionSpec Related to {name} went out of range (will be auto trimed to {ps}).",
                            stacklevel=1,
                        )
                return ps
        raise ValueError(f"Partition rule not found for param: {name}")

    return named_tree_map(get_partition_spec, tree, sep="/")


def analyze_sharding_strategy(
    pytree: tp.Any,
    partition_specs: dict[str, PartitionSpec],
    mesh: Mesh | None = None,
) -> dict:
    """
    Analyzes the effectiveness of a sharding strategy.

    Returns metrics like:
    - Memory usage per device
    - Load balance
    - Communication costs
    """
    if mesh is None:
        mesh = get_incontext_mesh()

    analysis = {
        "total_parameters": 0,
        "sharded_parameters": 0,
        "memory_per_device": {},
        "balance_score": 0.0,
        "partition_stats": {},
    }

    def analyze_leaf(path: str, array: np.ndarray, spec: PartitionSpec):
        total_size = np.prod(array.shape) * array.dtype.itemsize
        analysis["total_parameters"] += np.prod(array.shape)

        if spec != PartitionSpec():
            analysis["sharded_parameters"] += np.prod(array.shape)

        # Calculate per-device memory
        sharded_size = total_size
        for _, name in enumerate(spec):
            if name is not None:
                sharded_size //= mesh.shape[name]

        return sharded_size

    # Traverse the pytree and collect statistics
    tu.tree_map_with_path(analyze_leaf, pytree, partition_specs)

    return analysis


def create_pattern_based_partition_spec(
    pattern: str,
    mesh: Mesh | None = None,
    default_spec: PartitionSpec | None = None,
) -> tp.Callable[[str, chex.Array], PartitionSpec]:
    """
    Creates a function that returns PartitionSpec based on parameter name patterns.

    Example:
            pattern_fn = create_pattern_based_partition_spec(
                    "attention|mlp->data,hidden->model"
            )
    """
    if default_spec is None:
        default_spec = PartitionSpec()
    if mesh is None:
        mesh = get_incontext_mesh()

    rules = []
    for rule in pattern.split(","):
        if "->" in rule:
            patterns, spec = rule.split("->")
            patterns = patterns.split("|")
            spec = PartitionSpec(*spec.split("."))
            rules.extend((pattern, spec) for pattern in patterns)

    def get_partition_spec(name: str, array: chex.Array) -> PartitionSpec:
        for pattern, spec in rules:
            if re.search(pattern, name):
                return spec
        return default_spec

    return get_partition_spec


def extract_sharding_structure(pytree: tp.Any) -> tp.Any:
    """
    Extract a PyTree of NamedShardings matching the input structure.
    Returns None for leaves without shardings.
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)

    sharding_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jax.Array) and (shard := leaf.sharding) is not None:
            sharding_leaves.append(shard if isinstance(shard, NamedSharding) else None)
        else:
            sharding_leaves.append(None)

    return jax.tree_util.tree_unflatten(treedef, sharding_leaves)


def get_shardings_with_structure(pytree: tp.Any) -> tp.Any:
    """
    Returns a PyTree matching the input structure containing either:
    - NamedSharding objects where present
    - None for leaves without NamedShardings
    """
    return extract_sharding_structure(pytree)


def get_incontext_mesh(raise_error: bool = True) -> Mesh:
    """Retrieves the mesh object active in the current execution context.

    This function accesses the physical mesh defined within the thread's
    resource environment (pxla.thread_resources.env.physical_mesh).

    Returns:
        MeshType: The active mesh object for the current context.

    Raises:
        AssertionError: If no mesh is found in the current context
                        (i.e., mesh.empty is True).
    """
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
        if raise_error:
            raise AssertionError("No mesh found under this context manager.")
        else:
            return mesh
    return mesh


def get_axes_size_in_mesh(axis_names: AxisType, mesh: Mesh | None = None) -> int:
    """
    Calculates the total size of the specified mesh axes.

    If a single axis name (string) is provided, it returns the size of that
    dimension in the mesh. If a sequence (list or tuple) of axis names is
    provided, it returns the product of the sizes of all specified axes.

    If no mesh is explicitly provided, it uses the mesh active in the
    current context obtained via `get_current_mesh()`.

    Args:
        axis_names: The name of a single mesh axis (str) or a sequence
                    (list/tuple) of axis names whose sizes should be multiplied.
        mesh: The mesh object to query. If None, the current context's mesh
              is used. Defaults to None.

    Returns:
        int: The size of the single specified axis, or the product of the sizes
             of the sequence of specified axes.

    Raises:
        KeyError: If any of the specified `axis_names` are not found in the
                  mesh's dimensions.
        AssertionError: If `mesh` is None and no mesh is found in the current
                       context (raised by `get_current_mesh()`).
    """
    if mesh is None:
        mesh = get_incontext_mesh()

    # Assuming mesh.shape behaves like a dictionary {axis_name: size}
    mesh_shape: dict[str, int] = mesh.shape

    if isinstance(axis_names, str):
        # Raises KeyError if axis_names is not a valid key
        return mesh_shape[axis_names]
    elif isinstance(axis_names, list | tuple):
        product = 1
        # Iterate in the provided order, though order doesn't matter for product
        for axis in axis_names:
            # Raises KeyError if axis is not a valid key
            product *= mesh_shape[axis]
        return product
    else:
        # Handle unexpected type for axis_names
        raise TypeError(f"axis_names must be str or Sequence[str], got {type(axis_names)}")


def get_mesh_axis_names(mesh: Mesh | None = None) -> list[str]:
    """Retrieves the names of all axes defined in the mesh.

    These names typically correspond to the dimensions used for sharding or
    parallelism.

    If no mesh is explicitly provided, it uses the mesh active in the
    current context obtained via `get_current_mesh()`.

    Args:
        mesh: The mesh object to query. If None, the current context's mesh
              is used. Defaults to None.

    Returns:
        List[str]: A list containing the names of all axes in the mesh.

    Raises:
        AssertionError: If `mesh` is None and no mesh is found in the current
                       context (raised by `get_current_mesh()`).
    """
    if mesh is None:
        mesh = get_incontext_mesh()

    mesh_shape: dict[str, int] = mesh.shape
    return list(mesh_shape.keys())


def get_mesh_axis_size(axis_names: AxisType) -> int:
    """Calculates the total number of devices along the specified mesh axis or axes.

    Args:
        axis_names: The name of a single mesh axis (str) or a sequence (list/tuple)
                    of mesh axis names. The order in the sequence does not affect
                    the result (product is commutative).

    Returns:
        The total number of devices (size) in the submesh defined by the axis/axes.
        Returns 1 if axis_names is an empty sequence.

    Raises:
        TypeError: If axis_names is not a str or a sequence of str.
    """
    if isinstance(axis_names, str):
        # Size along a single axis dimension
        return lax.psum(1, axis_name=axis_names)
    elif isinstance(axis_names, list | tuple):
        if not axis_names:
            return 1  # The size of a submesh with zero dimensions is 1

        # Calculate the product of sizes along each specified axis
        product = 1
        for axis in axis_names:
            product *= lax.psum(1, axis_name=axis)
        return product
        # Alternative using math.prod (Python 3.8+)
        # return math.prod(lax.psum(1, axis_name=ax) for ax in axis_names)
    else:
        raise TypeError(f"Input 'axis_names' must be a string or sequence (list/tuple), but got type {type(axis_names)}")


def get_submesh_device_index(axis_names: AxisType) -> int:
    """
    Calculates the linear index of the current device within the specified mesh axes.

    This effectively flattens the multi-dimensional coordinates of the device
    within the submesh defined by `axis_names` into a single integer index.

    IMPORTANT: It assumes the input `axis_names` sequence is ordered from
    most major to most minor dimension. The calculation performs a
    row-major-like flattening based on this order.

    Args:
        axis_names: The name of a single mesh axis (str) or a sequence (list/tuple)
                    of mesh axis names, ordered from major to minor.

    Returns:
        The 0-based linear index of the current device within the submesh.
        Returns 0 if axis_names is an empty sequence.

    Raises:
        TypeError: If axis_names is not a str or a sequence of str.
    """
    if isinstance(axis_names, str):
        # Index along a single axis dimension
        return lax.axis_index(axis_name=axis_names)
    elif isinstance(axis_names, list | tuple):
        if not axis_names:
            return 0  # Index within a zero-dimensional submesh is 0

        linear_index = 0
        stride = 1
        # Iterate from the minor axis to the major axis (reverse of the input order)
        # This implements the formula: idx = sum(local_idx[dim] * stride[dim])
        # where stride[dim] = product(size[k] for k > dim)
        for axis in reversed(axis_names):
            index_on_axis = lax.axis_index(axis_name=axis)
            linear_index += index_on_axis * stride

            # Update stride for the next (more major) dimension
            axis_size = lax.psum(1, axis_name=axis)  # Use lax.psum, not the other func
            stride *= axis_size
        return linear_index
    else:
        raise TypeError(f"Input 'axis_names' must be a string or sequence (list/tuple), but got type {type(axis_names)}")


def extract_shardings(tree, mesh: Mesh = None):
    """
    Extracts JAX NamedSharding objects from the leaves of a PyTree.

    This function traverses the input PyTree and inspects each leaf.
    - If a leaf has a `.sharding` attribute that is already a `NamedSharding`,
      it's returned directly.
    - If a leaf has a `.sharding` attribute that is a `PartitionSpec`, it
      attempts to convert it into a `NamedSharding` using the provided `mesh`.
      If no `mesh` is provided, it tries to get one from the JAX context
      (e.g., using `get_incontext_mesh`). If no mesh is available in either
      case, an AssertionError is raised.
    - If a leaf does not have a `.sharding` attribute, or if its sharding
      is not a `NamedSharding` or convertible `PartitionSpec`, `None` is
      returned for that leaf in the output tree.

    Args:
        tree: The input PyTree (e.g., nested dictionary, list, tuple) potentially
              containing JAX arrays or other objects with sharding information.
        mesh: An optional `jax.sharding.Mesh`. If provided, it's used to convert
              `PartitionSpec` objects to `NamedSharding`. If `None`, the function
              attempts to find a mesh from the current JAX context.

    Returns:
        A PyTree with the same structure as the input `tree`. Each leaf will
        contain either a `jax.sharding.NamedSharding` object corresponding
        to the input leaf's sharding, or `None` if no valid sharding
        information was found or could be constructed.

    Raises:
        AssertionError: If a leaf has a `PartitionSpec` sharding but no `mesh`
                        is provided or found in the context.
    """
    if mesh is None:
        mesh = get_incontext_mesh()

    def cond(x):
        sharding = x.sharding if hasattr(x, "sharding") else None
        if isinstance(sharding, jax.sharding.PartitionSpec):
            assert mesh is not None, "Mesh Can not be none (use function under with `mesh`)."
            sharding = jax.sharding.NamedSharding(mesh=mesh, spec=sharding)
        if not isinstance(sharding, jax.sharding.NamedSharding):
            return None
        return sharding

    return jax.tree_util.tree_map(cond, tree)


def get_partition_spec(tree):
    """
    Retrieves the PartitionSpec for each leaf in a PyTree.

    This function traverses the input PyTree and determines the
    `jax.sharding.PartitionSpec` for each leaf based on its type:
    - If the leaf is a `jax.Array`, it returns the `PartitionSpec` from
      `leaf.sharding.spec`.
    - If the leaf is a Python scalar (`int` or `float`), it returns an
      empty `PartitionSpec()`, assuming scalars are typically replicated.
    - For any other leaf type, it raises a `ValueError`.

    Args:
        tree: The input PyTree (e.g., nested dictionary, list, tuple) containing
              JAX arrays, scalars, or potentially other types.

    Returns:
        A PyTree with the same structure as the input `tree`. Each leaf
        contains the corresponding `jax.sharding.PartitionSpec`.

    Raises:
        ValueError: If a leaf in the tree is not a `jax.Array`, `int`, or `float`.
        AttributeError: If a `jax.Array` leaf doesn't have `.sharding.spec` (which
                        would be unusual for a properly sharded array).
    """

    def _call(arr):
        if isinstance(arr, jax.Array):
            if hasattr(arr, "sharding") and hasattr(arr.sharding, "spec"):
                return arr.sharding.spec
            else:
                raise AttributeError(f"jax.Array leaf does not have expected .sharding.spec: {arr}")

        elif isinstance(arr, int | float):
            return PartitionSpec()
        else:
            raise ValueError(
                f"Unsupported leaf type for get_partition_spec: {type(arr)}. Expected jax.Array, int, or float."
            )

    return jax.tree_util.tree_map(_call, tree)
