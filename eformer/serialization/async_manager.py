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


import hashlib
import json
import os
import threading
import typing as tp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.distributed import is_initialized
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from safetensors import flax as safe_flax
from tqdm.autonotebook import tqdm

from eformer.escale import create_cpu_mesh
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.pytree import PyTree, flatten_dict, is_flatten, serialization, unflatten_dict

from .base_manager import CheckpointManager
from .serialization import tree_deserialize_leaves, tree_serialize_leaves
from .sharding_utils import make_itsharded
from .utils import derive_base_prefix_from_path, index_filename
from .utils import read_process_array as _read_process_array
from .utils import to_host as _to_host

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for checkpoints with versioning and validation.

    Stores comprehensive metadata about a checkpoint including version information,
    timestamps, checksums for validation, and custom user metadata.

    Attributes:
        version: Version string for the checkpoint format.
        timestamp: ISO format timestamp of when checkpoint was created.
        checksum: Dictionary mapping array keys to SHA256 checksums.
        array_metadata: Dictionary mapping array keys to shape/dtype info.
        framework_version: Version of the framework used to create checkpoint.
        custom_metadata: User-defined metadata dictionary.
    """

    version: str = "0.0.51"
    timestamp: str = None
    checksum: dict[str, str] = None
    array_metadata: dict[str, dict] = None
    framework_version: str = None
    custom_metadata: dict = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary format.

        Returns:
            Dictionary representation of the metadata.
        """
        return {
            "version": self.version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "checksum": self.checksum or {},
            "array_metadata": self.array_metadata or {},
            "framework_version": self.framework_version,
            "custom_metadata": self.custom_metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create CheckpointMetadata from dictionary.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            CheckpointMetadata instance.
        """
        return cls(
            version=data.get("version", "0.0.52"),
            timestamp=data.get("timestamp"),
            checksum=data.get("checksum", {}),
            array_metadata=data.get("array_metadata", {}),
            framework_version=data.get("framework_version"),
            custom_metadata=data.get("custom_metadata", {}),
        )


class AsyncCheckpointManager:
    """Checkpoint manager with concurrent operations.

    This manager provides checkpoint saving and loading with support
    for parallel operations, tensorstore backend, validation, and compression.
    Supports both TensorStore (for large-scale distributed checkpoints) and
    SafeTensors (for smaller, single-file checkpoints) formats.

    Key Features:
        - Automatic format detection (TensorStore vs SafeTensors)
        - Parallel I/O operations for faster loading/saving
        - CPU offloading to prevent OOM on accelerators
        - Checksum validation for data integrity
        - Support for sharded checkpoints across multiple files
        - Pattern-based partition rules with preserved ordering

    Attributes:
        float_dtype: Default data type for floating point arrays.
        enable: Whether checkpointing is enabled.
        verbose: Enable verbose output.
        gcs_bucket: Google Cloud Storage bucket name.
        max_workers: Maximum number of worker threads.
        enable_validation: Enable checksum validation.
        enable_compression: Enable compression for tensorstore.
        use_tensorstore: Use tensorstore backend when available.

    Example:
        >>> manager = AsyncCheckpointManager(
        ...     enable_validation=True,
        ...     max_workers=8,
        ...     use_tensorstore=True
        ... )
        >>>
        >>> manager.save(model_state, "checkpoint", mesh=mesh)
        >>>
        >>> rules = [(".*kernel", PartitionSpec("model", None))]
        >>> state, meta = manager.load("checkpoint", mesh, partition_rules=rules)
    """

    def __init__(
        self,
        enable: bool | None = None,
        float_dtype: jnp.dtype = jnp.bfloat16,
        verbose: bool = False,
        gcs_bucket: str | None = None,
        gcs_credentials_path: str | None = None,
        enable_validation: bool = False,
        enable_compression: bool = False,
        use_tensorstore: bool = True,
        max_workers: int = 8,
    ):
        if jax.process_count() > 1:
            assert is_initialized(), "you should call jax distribution init before running process."

        self.float_dtype = float_dtype
        self.enable = enable
        self.verbose = verbose
        self.gcs_bucket = gcs_bucket
        self.enable_validation = enable_validation
        self.enable_compression = enable_compression
        self.use_tensorstore = use_tensorstore
        self.max_workers = max_workers

        self.gcs_client = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._save_lock = threading.Lock()

        if gcs_bucket:
            self.gcs_client = CheckpointManager.create_gcs_client(gcs_credentials_path)

    def __del__(self):
        """Cleanup executor on deletion.

        Ensures the thread pool executor is properly shutdown when the
        manager is destroyed.
        """
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    @staticmethod
    def _estimate_nbytes(array: jax.Array) -> int:
        """Estimate the number of bytes in an array.

        Args:
            array: JAX array to estimate size for.

        Returns:
            Estimated number of bytes in the array.
        """
        if hasattr(array, "nbytes"):
            return array.nbytes
        elif hasattr(array, "shape") and hasattr(array, "dtype"):
            return np.prod(array.shape) * np.dtype(array.dtype).itemsize
        else:
            return 0

    def _calculate_optimal_chunks(self, shape: tuple, dtype: jnp.dtype) -> list[int] | None:
        """Calculate optimal chunk sizes for an array.

        Aims for chunks of ~64MB for optimal I/O performance. Balances between
        chunk size and number of chunks to optimize read/write operations.

        Args:
            shape: Shape of the array to chunk.
            dtype: Data type of the array.

        Returns:
            List of chunk sizes for each dimension, or None for small arrays
            that don't need chunking.

        Note:
            For very large dimensions (>10000), limits chunk size to 2000 elements
            to avoid overly large chunks.
        """
        if not shape:
            return None

        target_chunk_bytes = 64 * 1024 * 1024

        dtype_size = np.dtype(dtype).itemsize

        total_elements = np.prod(shape)
        total_bytes = total_elements * dtype_size

        if total_bytes < target_chunk_bytes:
            return None

        chunks = []
        remaining_bytes = target_chunk_bytes

        for dim_size in shape:
            elements_per_chunk = min(dim_size, max(1, remaining_bytes // dtype_size))

            if dim_size > 10000:
                elements_per_chunk = min(2000, elements_per_chunk)

            chunks.append(int(elements_per_chunk))
            remaining_bytes = remaining_bytes // max(1, (dim_size // elements_per_chunk))

        return chunks

    @staticmethod
    def compute_checksum(array: jax.Array) -> str:
        """Compute SHA256 checksum for validation.

        Converts array to bytes and computes SHA256 hash for data integrity
        verification.

        Args:
            array: JAX array to compute checksum for.

        Returns:
            SHA256 checksum as hexadecimal string.

        Note:
            Arrays are converted to numpy before hashing for consistency.
        """
        array_bytes = np.asarray(array).tobytes()
        return hashlib.sha256(array_bytes).hexdigest()

    def _validate_checkpoint(self, tree: dict, metadata: CheckpointMetadata) -> bool:
        """Validate checkpoint integrity using checksums.

        Compares computed checksums of loaded arrays against stored checksums
        in metadata to ensure data integrity.

        Args:
            tree: Dictionary containing checkpoint data.
            metadata: Checkpoint metadata containing checksums.

        Returns:
            True if validation passes, False otherwise.

        Note:
            Validation is skipped if enable_validation is False or no checksums
            are present in metadata.
        """
        if not self.enable_validation or not metadata.checksum:
            return True

        flat_tree = flatten_dict(tree) if not is_flatten(tree) else tree
        for key, array in flat_tree.items():
            if key in metadata.checksum:
                computed = self.compute_checksum(array)
                if computed != metadata.checksum[key]:
                    logger.error(f"Checksum mismatch for {key}")
                    return False
        return True

    def save_tree(
        self,
        tree: PyTree,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh | None = None,
        gather_fns: dict[tp.Callable] | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        metadata: dict[str, str] | None = None,
        callback: tp.Callable[[str], None] | None = None,
        prefix: str | None = None,
        do_all_gather: bool = False,
        cpu_offload: bool = False,
    ) -> str:
        """Save checkpoint with parallel shard writing.

        Saves a PyTree structure to disk using either TensorStore or SafeTensors format,
        with support for sharding large checkpoints and parallel I/O operations.

        Args:
            tree: PyTree structure to save.
            path: Path where the checkpoint will be saved.
            mesh: JAX mesh for distributed computation. If None, creates a CPU mesh
                with a warning about potential sharding issues.
            gather_fns: Dictionary of gather functions or bool for device gathering.
                If True, uses jax.device_get for all arrays.
            float_dtype: Data type for floating point arrays. Defaults to self.float_dtype.
            metadata: Additional metadata to save with checkpoint.
            callback: Optional callback function called after save completes.
            prefix: Optional prefix for saving specific tree (e.g., 'model', 'optimizer').
                Used for organizing multiple trees in same directory.
            do_all_gather: Whether to gather all arrays to host before saving. Defaults
                to True for safer and more consistent checkpoint saving.
            cpu_offload: Whether to offload arrays to CPU during gathering. Defaults to
                True to reduce memory pressure on accelerators and prevent OOM errors.

        Returns:
            Path where the checkpoint was saved.

        Note:
            - Automatically chooses between TensorStore (if available and enabled) or
              SafeTensors format based on configuration.
            - When mesh is not provided, a warning is logged and CPU mesh is used as fallback.
            - CPU offloading helps prevent out-of-memory errors on GPUs/TPUs during checkpointing.
            - Arrays are automatically flattened before saving and unflattened when loading.

        Example:
            >>> manager = AsyncCheckpointManager()
            >>> manager.save_tree(
            ...     tree=model_state,
            ...     path="checkpoint",
            ...     mesh=mesh,
            ...     prefix="model",
            ...     cpu_offload=True
            ... )
        """
        if float_dtype is None:
            float_dtype = self.float_dtype

        tree = serialization.to_state_dict(tree)
        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")
        if mesh is None:
            logger.warn("`mesh` should be provided otherwise you will face some sharding issues.")
            mesh = create_cpu_mesh()
        if gather_fns:
            tree = self._gather_fn(tree, gather_fns)
        if do_all_gather:
            tree = jax.tree_util.tree_map(
                lambda x: _to_host(x, float_dtype, mesh, cpu_offload),
                tree,
                is_leaf=lambda x: isinstance(x, jax.Array | np.generic | float | int),
            )

        if jax.process_count() > 1:
            tree = make_itsharded(tree, mesh)

        checkpoint_meta = CheckpointMetadata(timestamp=datetime.now().isoformat(), custom_metadata=metadata)

        if self.enable_validation:
            checkpoint_meta.checksum = {k: self.compute_checksum(v) for k, v in tree.items()}
            checkpoint_meta.array_metadata = {
                k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in tree.items()
            }

        path_str = str(path)

        if self.use_tensorstore:
            out = self._save_tensorstore(tree, path_str, checkpoint_meta, prefix)
        else:
            out = self._save_single(tree, path_str, checkpoint_meta.to_dict())

        if callback:
            callback(path_str)

        return out

    save = save_tree

    def _gather_fn(self, tree: dict, gather_fns: dict[tp.Callable] | bool) -> dict:
        """Gather distributed arrays.

        Performs parallel gathering of distributed arrays using provided gather
        functions or device_get.

        Args:
            tree: Dictionary of arrays to gather.
            gather_fns: Dictionary mapping keys to gather functions, or bool.
                If True, uses jax.device_get for all arrays.
                If dict, applies specific gather function for matching keys.

        Returns:
            Dictionary with gathered arrays.

        Note:
            Arrays without matching gather functions are returned unchanged.
        """
        if isinstance(gather_fns, bool):
            results = {}
            for key, value in tree.items():
                results[key] = jax.device_get(value)
            return results

        if not is_flatten(gather_fns):
            gather_fns = flatten_dict(gather_fns, sep=".")

        results = {}
        for key, value in tree.items():
            if key in gather_fns:
                results[key] = gather_fns[key](value)
            else:
                results[key] = value

        return results

    def _save_single(self, tree: dict, path: str, metadata: dict):
        """Save single checkpoint file.

        Args:
            tree: Dictionary to save.
            path: Path where the checkpoint will be saved.
            metadata: Metadata to save with the checkpoint.
        """
        if metadata:
            metadata = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()}

        safe_flax.save_file(tree, path, metadata)
        return path

    def _save_tensorstore(
        self,
        tree: dict,
        path: str,
        metadata: CheckpointMetadata,
        prefix: str | None = None,
    ) -> str:
        """Save using tensorstore via the core serialization module.

        Leverages TensorStore for efficient array serialization with support for
        zarr format and concurrent writes.

        Args:
            tree: Dictionary of arrays to save (flattened).
            path: Path where the checkpoint will be saved.
            metadata: Checkpoint metadata.
            prefix: Optional prefix for saving specific tree.

        Returns:
            Path where the checkpoint was saved.

        Note:
            Creates a unified index file (tensorstore_index.json) that supports
            multiple prefixes in v2.0 format. Also saves checkpoint metadata
            separately.
        """

        from eformer.pytree import unflatten_dict

        pytree = unflatten_dict(tree, sep=".")

        from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

        manager = GlobalAsyncCheckpointManager()

        tree_serialize_leaves(
            checkpoint_dir=path,
            pytree=pytree,
            manager=manager,
            prefix=prefix,
            commit_callback=lambda: logger.info("Committed checkpoint to Tensorstore"),
            write_index=True,
        )

        logger.info("Committed checkpoint to Tensorstore")
        meta_path = ePath(path) / "checkpoint_metadata.json"
        meta_path.write_text(json.dumps(metadata.to_dict()))

        return path

    def _load_tensorstore_sync(
        self,
        path: str,
        shardings: dict[NamedSharding] | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        mesh: Mesh | None = None,
        prefix: str | None = None,
    ) -> tuple[dict, dict]:
        """Load checkpoint saved with tensorstore using core deserialization.

        Args:
            path: Path to the tensorstore checkpoint.
            shardings: PyTree of sharding specifications or dict of functions.
            partition_rules: Pattern-based partition rules.
            mesh: JAX mesh for distributed computation.
            prefix: Optional prefix for loading specific tree.

        Returns:
            Tuple of (loaded tree dictionary, metadata dictionary).
        """
        from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

        manager = GlobalAsyncCheckpointManager()

        tree = tree_deserialize_leaves(
            checkpoint_dir=path,
            mesh=mesh,
            partition_rules=partition_rules,
            manager=manager,
            prefix=prefix,
            shardings=shardings,
        )

        meta_path = ePath(path) / "checkpoint_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {}

        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")

        return tree, metadata

    def load(
        self,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh,
        shardings: dict[NamedSharding] | None | dict[tp.Callable] = None,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        dtype: str | jnp.dtype | None = None,
        validate: bool | None = None,
        prefix_filter: str | None = None,
        prefix: str | None = None,
        use_async: bool = True,
    ) -> tuple[PyTree | dict, dict]:
        """Synchronous load method that can work with or without async.

        Automatically detects checkpoint format (TensorStore or SafeTensors) and
        loads accordingly. Can be called without async/await.

        Args:
            path: Path to the checkpoint directory or file.
            mesh: JAX mesh for distributed computation. Required for proper sharding.
            shardings: PyTree of sharding specifications matching checkpoint structure,
                or dict mapping keys to functions that process/reshard arrays after loading.
            mismatch_allowed: Whether to allow missing shard functions without error.
            callback: Optional callback to process each array after loading.
                Receives (array, key) and returns processed array.
            partition_rules: List of (regex, PartitionSpec) tuples for pattern-based
                sharding. Applied to arrays matching the regex patterns. Preserves
                order of arrays during loading.
            dtype: Data type to cast arrays to after loading.
            validate: Whether to validate checksums. If None, uses self.enable_validation.
            prefix_filter: Deprecated. Use 'prefix' instead.
            prefix: Optional prefix for loading specific tree (e.g., 'model', 'optimizer').
                Required when checkpoint contains multiple prefixes.
            use_async: Whether to use parallel loading (faster) or sequential loading.

        Returns:
            Tuple of (loaded tree, metadata dictionary).
            Tree is unflattened to nested structure.

        Raises:
            ValueError: If validation fails or prefix not found.
            FileNotFoundError: If checkpoint doesn't exist.

        Note:
            - Automatically detects TensorStore format by checking for .zarray files
              or tensorstore_index.json.
            - When using partition_rules, the order of loaded arrays is preserved
              to ensure consistent sharding application.

        Example:
            >>> manager = AsyncCheckpointManager()
            >>> rules = [(".*weight", PartitionSpec("model", None))]
            >>> tree, meta = manager.load("checkpoint", mesh, partition_rules=rules)
        """
        path_str = str(path)

        is_tensorstore = False
        path_obj = ePath(path_str)
        if path_obj.is_dir():
            if (path_obj / "tensorstore_index.json").exists():
                is_tensorstore = True
            elif any((path_obj / d / ".zarray").exists() for d in os.listdir(path_str) if (path_obj / d).is_dir()):
                is_tensorstore = True

        if is_tensorstore:
            if use_async:
                tree, metadata = self._load_tensorstore_sync(
                    path=path_str,
                    mesh=mesh,
                    partition_rules=partition_rules,
                    shardings=shardings,
                    prefix=prefix,
                )
            else:
                from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

                manager = GlobalAsyncCheckpointManager()
                tree = tree_deserialize_leaves(
                    checkpoint_dir=path_str,
                    mesh=mesh,
                    partition_rules=partition_rules,
                    manager=manager,
                    prefix=prefix,
                    shardings=shardings,
                )
                meta_path = path_obj / "checkpoint_metadata.json"
                if meta_path.exists():
                    metadata = json.loads(meta_path.read_text())
                else:
                    metadata = {}

            if not is_flatten(tree):
                tree = flatten_dict(tree, sep=".")
            tree = unflatten_dict(tree, sep=".")
            return tree, metadata
        else:
            return self.load_tree_parallel(
                path=path,
                shardings=shardings,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
                validate=validate,
                prefix_filter=prefix_filter,
            )

    def load_tree_parallel(
        self,
        path: ePathLike | str | os.PathLike,
        shardings: None | dict[tp.Callable] = None,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        validate: bool | None = None,
        prefix_filter: str | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load checkpoint with parallel shard reading.

        Args:
            path: Path to the checkpoint.
            shardings: PyTree of sharding specifications or dict of functions.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.
            validate: Whether to validate checksums.
            prefix_filter: Optional prefix to filter shards.

        Returns:
            Tuple of (loaded tree, metadata dictionary).

        Raises:
            ValueError: If checkpoint validation fails.
        """
        validate = validate if validate is not None else self.enable_validation

        path_str = str(path)
        base_prefix = derive_base_prefix_from_path(path_str)
        index_path_str = index_filename(base_prefix)

        if ePath(index_path_str).exists():
            return self._load_sharded_parallel(
                index_path_str,
                shardings,
                mismatch_allowed,
                callback,
                dtype,
                validate,
                prefix_filter,
            )

        tree, metadata = CheckpointManager.load_checkpoint(
            path,
            shardings,
            self.verbose,
            mismatch_allowed,
            callback,
            dtype,
            self.gcs_client,
        )

        if validate and metadata:
            meta = CheckpointMetadata.from_dict(metadata)
            if not self._validate_checkpoint(tree, meta):
                raise ValueError("Checkpoint validation failed")

        return tree, metadata

    def _load_sharded_parallel(
        self,
        index_path: str,
        shardings: PyTree | dict[tp.Callable] | None,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
        validate: bool,
        prefix_filter: str | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load sharded checkpoint with parallel reads.

        Args:
            index_path: Path to the index file.
            shardings: PyTree of sharding specifications or dict of functions.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.
            validate: Whether to validate checksums.
            prefix_filter: Optional prefix to filter loaded keys (deprecated).

        Returns:
            Tuple of (loaded tree, metadata dictionary).

        Raises:
            ValueError: If checkpoint validation fails.
        """
        index_data = json.loads(ePath(index_path).read_text())

        weight_map: dict[str, str] = index_data.get("weight_map", {})
        directory = str(ePath(index_path).parent)

        file_to_keys: dict[str, list[str]] = defaultdict(list)
        for k, shard_name in weight_map.items():
            file_to_keys[shard_name].append(k)

        shard_fns = None
        if shardings:
            if isinstance(shardings, dict) and not any(isinstance(v, dict) for v in shardings.values()):
                shard_fns = shardings
            else:
                from eformer.pytree import flatten_dict, is_flatten

                if not is_flatten(shardings):
                    shard_fns = flatten_dict(shardings, sep=".")
                else:
                    shard_fns = shardings

        tree = {}
        futures = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for shard_name, keys in file_to_keys.items():
                shard_path = str(ePath(directory) / shard_name)
                future = executor.submit(
                    self._load_shard_file,
                    shard_path,
                    keys,
                    shard_fns,
                    mismatch_allowed,
                    callback,
                    dtype,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading shards (parallel)",
                disable=not self.verbose,
            ):
                shard_tree = future.result()
                tree.update(shard_tree)

        tree = unflatten_dict(tree, sep=".")
        metadata = index_data.get("metadata", {})

        if validate and metadata:
            meta = CheckpointMetadata.from_dict(metadata)
            if not self._validate_checkpoint(tree, meta):
                raise ValueError("Checkpoint validation failed")

        return tree, metadata

    def _load_shard_file(
        self,
        shard_path: str,
        keys: list[str],
        shard_fns: dict | None,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
    ) -> dict:
        """Load a single shard file.

        Args:
            shard_path: Path to the shard file.
            keys: List of keys to load from the shard.
            shard_fns: Flat dictionary of functions to apply to shards.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.

        Returns:
            Dictionary with loaded tensors.
        """
        shard_tree = {}
        with safe_flax.safe_open(shard_path, framework="flax") as manager:
            process_func = partial(
                _read_process_array,
                shard_fns=shard_fns,
                mismatch_allowed=mismatch_allowed,
                manager=manager,
                callback=callback,
                dtype=dtype,
            )
            for key in keys:
                k, tensor, _ = process_func(key)
                shard_tree[k] = tensor
        return shard_tree

    @staticmethod
    def is_tensorstore(path) -> bool:
        if str(path).endswith("tensorstore_index.json"):
            return True
        return (ePath(path) / "tensorstore_index.json").exists()

    @staticmethod
    def safe_loadpath(path) -> ePathLike:
        if AsyncCheckpointManager.is_tensorstore:
            if str(path).endswith("tensorstore_index.json"):
                return ePath(str(path)[: -len("tensorstore_index.json")])
        return ePath(path)
