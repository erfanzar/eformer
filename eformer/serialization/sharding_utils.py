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

"""Utilities for handling sharding in checkpoint serialization."""

import json
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from eformer.loggings import get_logger
from eformer.paths import ePath

logger = get_logger(__name__)


def create_sharding_tree_from_index(
    checkpoint_dir: str,
    mesh: Mesh | None = None,
    prefix: str | None = None,
    default_sharding: Sharding | PartitionSpec | Callable[[Any], Sharding] | None = None,
) -> dict:
    """Create a sharding tree from tensorstore index file.

    Creates a PyTree structure that matches the checkpoint structure, where each
    leaf is a sharding specification or function that can be applied to the
    corresponding array during deserialization.

    Args:
        checkpoint_dir: Directory containing the tensorstore checkpoint.
        mesh: JAX mesh for creating shardings. If None, uses replicated sharding.
        prefix: Optional prefix to create sharding tree for specific subtree.
        default_sharding: Default sharding to use for all arrays. Can be:
            - A Sharding object
            - A PartitionSpec (will be wrapped with NamedSharding using mesh)
            - A callable that takes array info dict and returns a Sharding
            - None (uses fully replicated sharding)

    Returns:
        Dictionary with same structure as checkpoint, where leaves are sharding
        specifications.

    Example:
        >>> # Create default replicated shardings
        >>> shard_tree = create_sharding_tree_from_index("checkpoint/")

        >>> # Create custom shardings based on array properties
        >>> def custom_shard_fn(info):
        ...     if "embedding" in info["path"]:
        ...         return NamedSharding(mesh, PartitionSpec("data", None))
        ...     return NamedSharding(mesh, PartitionSpec())
        >>> shard_tree = create_sharding_tree_from_index(
        ...     "checkpoint/", mesh=mesh, default_sharding=custom_shard_fn
        ... )
    """
    index_path = ePath(checkpoint_dir) / "tensorstore_index.json"

    if not index_path.exists():
        logger.warning(f"No tensorstore_index.json found in {checkpoint_dir}")
        return {}

    index_data = json.loads(index_path.read_text())

    if index_data.get("format") != "tensorstore":
        raise ValueError(f"Unsupported index format: {index_data.get('format')}")

    version = index_data.get("version", "1.0")

    # Get array info based on version and prefix
    if version == "2.0" and "prefixes" in index_data:
        if prefix:
            if prefix not in index_data["prefixes"]:
                available = list(index_data["prefixes"].keys())
                raise ValueError(f"Prefix '{prefix}' not found. Available: {available}")
            array_info = index_data["prefixes"][prefix]
        else:
            if "arrays" in index_data:
                array_info = index_data["arrays"]
            else:
                available = list(index_data["prefixes"].keys())
                logger.info(f"Multiple prefixes available: {available}. Using first: {available[0]}")
                prefix = available[0]
                array_info = index_data["prefixes"][prefix]
    else:
        array_info = index_data.get("arrays", [])

    # Build sharding tree
    shard_tree = {}

    for info in array_info:
        path = info["path"]

        # Remove prefix from path if present
        if prefix and path.startswith(f"{prefix}/"):
            path = path[len(prefix) + 1 :]

        # Convert path to nested dict keys
        parts = path.split("/")

        # Determine sharding for this array
        if default_sharding is None:
            # Default: fully replicated
            if mesh is None:
                sharding = None  # Will be handled during deserialization
            else:
                sharding = NamedSharding(mesh, PartitionSpec())
        elif callable(default_sharding):
            # Custom function
            sharding = default_sharding(info)
        elif isinstance(default_sharding, PartitionSpec):
            # Convert PartitionSpec to NamedSharding
            if mesh is None:
                raise ValueError("Mesh required when using PartitionSpec as default_sharding")
            sharding = NamedSharding(mesh, default_sharding)
        else:
            # Direct sharding object
            sharding = default_sharding

        # Insert into tree
        current = shard_tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = sharding

    return shard_tree


def apply_sharding_tree(
    arrays: dict,
    sharding_tree: dict | None,
    mesh: Mesh | None = None,
) -> dict:
    """Apply sharding specifications from a sharding tree to arrays.

    Args:
        arrays: Dictionary of arrays (flattened or nested).
        sharding_tree: PyTree of sharding specifications matching arrays structure.
        mesh: JAX mesh for creating shardings.

    Returns:
        Dictionary of arrays with sharding applied.
    """
    if sharding_tree is None:
        return arrays

    import jax.tree_util as jtu
    from jax.sharding import SingleDeviceSharding

    # Helper to create default sharding
    def default_sharding():
        if mesh is None:
            return SingleDeviceSharding(jax.devices()[0])
        else:
            return NamedSharding(mesh, PartitionSpec())

    # Apply shardings using tree_map
    def apply_shard(array, shard_spec):
        if shard_spec is None:
            # Use default replicated sharding
            shard_spec = default_sharding()

        if callable(shard_spec):
            # If it's a function, call it with the array
            return shard_spec(array)
        elif isinstance(shard_spec, Sharding):
            # Apply the sharding
            if hasattr(array, "shape"):
                return jax.device_put(array, shard_spec)
            else:
                return array
        else:
            # Unknown spec, return as-is
            logger.warning(f"Unknown sharding spec type: {type(shard_spec)}")
            return array

    # Use tree_map to apply shardings, handling mismatched structures gracefully
    try:
        result = jtu.tree_map(apply_shard, arrays, sharding_tree, is_leaf=lambda x: x is None or not isinstance(x, dict))
    except ValueError as e:
        # Structure mismatch - apply what we can
        logger.warning(f"Sharding tree structure mismatch: {e}")
        result = arrays

    return result


def validate_sharding_tree(sharding_tree: dict, expected_structure: dict) -> bool:
    """Validate that a sharding tree matches expected structure.

    Args:
        sharding_tree: PyTree of sharding specifications.
        expected_structure: Expected PyTree structure (e.g., from checkpoint index).

    Returns:
        True if structures match, False otherwise.
    """
    import jax.tree_util as jtu

    try:
        # Get the structure (treedef) of both trees
        _, tree_def = jtu.tree_flatten(sharding_tree)
        _, expected_def = jtu.tree_flatten(expected_structure)

        # Compare structures
        return tree_def == expected_def
    except Exception as e:
        logger.warning(f"Error validating sharding tree: {e}")
        return False


def make_itsharded(xs, mesh):
    def _procss(x):
        if isinstance(x, jax.Array) and x.is_fully_addressable:

            @partial(jax.jit, out_shardings=NamedSharding(mesh, PartitionSpec()))
            def _move(x_):
                return x_

            return _move(x)
        return x

    return jax.tree_util.tree_map(_procss, xs)
