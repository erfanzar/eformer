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

# few updates from Levanter

"""High-level checkpoint management for eFormer.

This module provides a sophisticated checkpoint manager with time- and step-based
policies for training workflows. Key features include:

- **Flexible Checkpoint Policies**: Configure time-based and step-based saving policies
- **TensorStore Backend**: Efficient storage for large-scale distributed arrays
- **TP+FSDP Compatibility**: No all-gather required, preserves existing shardings
- **Async Operations**: Non-blocking checkpoint saves with background cleanup
- **Temporary Checkpoints**: Automatic management of temporary vs permanent checkpoints
- **Multi-host Support**: Distributed checkpoint operations across multiple hosts

The Checkpointer class is designed to be used in training loops where you want
automatic checkpoint management without manual intervention.
"""

import datetime as dt
import json
import os
import queue
import threading
import time
import typing as tp
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jax
import jax.numpy as jnp
from fsspec import AbstractFileSystem
from jax.experimental import multihost_utils as mh

from eformer.loggings import get_logger
from eformer.pytree import PyTree

from . import fsspec_utils
from .async_manager import AsyncCheckpointManager

logger = get_logger(__name__)

# Type aliases
MetadataDict = dict[str, tp.Any]
CheckpointPath = str
Sequence = tp.Sequence
Callable = tp.Callable


@dataclass(frozen=True)
class CheckpointInterval:
    """Configuration for step-based checkpoint saving policy.

    Defines when to save checkpoints based on training steps. Multiple intervals
    can be combined to create sophisticated checkpoint policies (e.g., save every
    100 steps for the first 1000 steps, then every 1000 steps thereafter).

    Attributes:
        every: Save checkpoint every N steps within this interval.
        until: Save using this policy until this step (inclusive). If None,
            this policy applies indefinitely. Only the last policy in a sequence
            can have until=None.

    Examples:
        ```python
        # Save every 100 steps
        interval = CheckpointInterval(every=100)

        # Save every 50 steps until step 1000
        interval = CheckpointInterval(every=50, until=1000)

        # Multi-stage policy: frequent saves early, less frequent later
        policies = [
            CheckpointInterval(every=100, until=1000),  # Every 100 steps up to 1000
            CheckpointInterval(every=1000),             # Every 1000 steps thereafter
        ]
        ```
    """

    every: int
    until: int | None = None


class Checkpointer:
    """High-level checkpoint manager with time- and step-based policies for eFormer.

    This class provides automatic checkpoint management for training loops with support
    for both time-based and step-based saving policies. It integrates with JAX's
    distributed training capabilities and TensorStore for efficient storage.

    Key Features:
        - Multi-policy checkpointing: Configure different save intervals at different
          training stages
        - Time-based saves: Automatically save at regular time intervals
        - Temporary checkpoints: Distinguish between temporary (time-based) and permanent
          (step-based) checkpoints
        - Async cleanup: Background deletion of old temporary checkpoints
        - Multi-host support: Coordinated saves across distributed training hosts
        - TensorStore backend: Efficient storage without all-gather operations

    The checkpointer maintains existing array shardings (TP/FSDP) during saves,
    avoiding expensive all-gather operations.

    Attributes:
        base_path: Root directory for all checkpoints.
        save_interval: Optional time interval for temporary checkpoint saves.
        step_policies: Sequence of step-based checkpoint policies.

    Examples:
        ```python
        from datetime import timedelta

        # Create checkpointer with time and step policies
        checkpointer = Checkpointer(
            base_path="/checkpoints/my_model",
            save_interval=timedelta(minutes=15),  # Temp checkpoint every 15min
            step_policies=[
                CheckpointInterval(every=500, until=5000),  # Every 500 steps until 5000
                CheckpointInterval(every=1000),             # Every 1000 steps after
            ],
        )

        # In training loop
        for step, batch in enumerate(train_loader):
            # ... training code ...
            checkpointer.on_step(training_state)

        # Wait for all saves to complete
        checkpointer.wait_until_finished()
        ```
    """

    def __init__(
        self,
        base_path: str,
        save_interval: timedelta | None,
        step_policies: Sequence[CheckpointInterval],
        *,
        manager: AsyncCheckpointManager | None = None,
        dt_now_injection: Callable[[], dt.datetime] | None = None,
        delete_old_temp_checkpoints: bool = True,
    ) -> None:
        """Initialize the Checkpointer with saving policies.

        Args:
            base_path: Root directory where all checkpoints will be saved. Can be
                a local path or a cloud storage path (e.g., "gs://bucket/path").
            save_interval: Time interval for automatic temporary checkpoint saves.
                If None, only step-based saves will occur. Temporary checkpoints
                are automatically cleaned up when newer checkpoints are saved.
            step_policies: Sequence of checkpoint interval policies. Policies must
                be sorted by 'until' value, and only the last policy can have
                until=None. Each policy defines checkpoint frequency for a range
                of training steps.
            manager: Optional AsyncCheckpointManager instance. If None, a default
                manager with TensorStore backend will be created.
            dt_now_injection: Optional function returning current datetime. Used
                primarily for testing. Defaults to datetime.now.
            delete_old_temp_checkpoints: If True, automatically discovers and
                deletes temporary checkpoints from previous runs after saving
                the first checkpoint in this session.

        Raises:
            ValueError: If step_policies are not properly sorted or if a non-final
                policy has until=None.

        Note:
            - The checkpointer validates policy monotonicity at initialization
            - Background cleanup thread is started only on process 0
            - Existing temporary checkpoints are discovered and marked for cleanup
        """
        self.base_path = str(base_path)
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self._dt_now_injection = dt_now_injection or dt.datetime.now
        self._last_save_time = self._dt_now_injection()
        self._last_save_step = 0

        # simple validation for monotonic intervals (like Levanter)
        for i in range(1, len(step_policies)):
            prev_until = step_policies[i - 1].until
            until = step_policies[i].until
            if prev_until is None:
                raise ValueError("Only the last step policy can have an 'until' of None")
            if until is None:
                continue
            if prev_until >= until:
                raise ValueError("Step policies must be sorted by 'until' value")

        self._manager = manager or AsyncCheckpointManager(use_tensorstore=True, enable_validation=False, verbose=False)

        if jax.process_index() == 0:
            self._checkpoint_cleanup_worker_queue: queue.Queue[str] = queue.Queue(maxsize=-1)
            self._checkpoint_cleanup_worker_thread = threading.Thread(
                target=self._checkpoint_cleanup_worker, daemon=True
            )
            self._checkpoint_cleanup_worker_thread.start()
            self._checkpoint_being_removed = None

        # discover latest checkpoint; if it's temporary, mark for deletion after next save
        self._last_temporary_checkpoint = None
        latest = find_latest_checkpoint(self.base_path)
        if latest is not None and delete_old_temp_checkpoints:
            try:
                metadata = _read_checkpoint_metadata(latest)
                if metadata.get("is_temporary", False):
                    logger.info(
                        f"Found prior temporary checkpoint {latest}. Will delete it after saving a new checkpoint."
                    )
                    self._last_temporary_checkpoint = latest
            except FileNotFoundError:
                pass

    def on_step(self, info: tp.Any, force: bool = False) -> None:
        """Process a training step and save checkpoint if policies dictate.

        This method should be called once per training step. It evaluates both
        time-based and step-based policies to determine whether to save a checkpoint.
        The decision is made on process 0 and broadcast to all processes to ensure
        consistency in distributed settings.

        Args:
            info: Training state object containing at least a 'step' attribute (int)
                and optionally a 'state' attribute (PyTree to save). The exact structure
                depends on your training framework.
            force: If True, force a permanent checkpoint save regardless of policies.
                Useful for saving at the end of training or before evaluation.

        Note:
            - Step 0 is skipped unless force=True (initialization step)
            - Duplicate step saves are skipped unless force=True
            - Time-based saves create temporary checkpoints (auto-cleaned)
            - Step-based saves create permanent checkpoints
            - All processes synchronize on the save decision via broadcast
            - Old temporary checkpoints are queued for async deletion

        Examples:
            ```python
            # Regular usage in training loop
            for step, batch in enumerate(dataloader):
                loss = train_step(state, batch)
                checkpointer.on_step(state)  # Automatic checkpoint management

            # Force save at end of training
            checkpointer.on_step(state, force=True)
            ```
        """
        step = int(info.step)

        if step == 0 and not force:
            self._last_save_time = self._dt_now_injection()
            return

        if step == self._last_save_step and not force:
            return

        my_should_save = bool(force)
        my_save_permanent = bool(force)

        current_every = self._get_save_interval_for_step(step)
        elapsed = self._dt_now_injection() - self._last_save_time

        if current_every is not None and step % current_every == 0:
            my_should_save = True
            my_save_permanent = True
        elif self.save_interval and elapsed >= self.save_interval:
            my_should_save = True
            my_save_permanent = False

        # Decide on host 0 and broadcast to all hosts
        flags = jnp.array([my_should_save, my_save_permanent], dtype=jnp.bool_)
        should_save, save_permanent = mh.broadcast_one_to_all(flags)
        should_save, save_permanent = bool(should_save), bool(save_permanent)

        if not should_save:
            return

        if save_permanent:
            logger.info(f"Saving checkpoint at step {step}.")
        else:
            logger.info(f"Saving temporary checkpoint at step {step}.")

        last_tmp = self._last_temporary_checkpoint
        destination = f"step-{step}"

        self._last_temporary_checkpoint = os.path.join(self.base_path, destination) if not save_permanent else None

        def callback():
            try:
                _write_checkpoint_metadata(
                    os.path.join(self.base_path, destination),
                    step=step,
                    is_temporary=not save_permanent,
                )
            except Exception as e:
                logger.warning(f"Failed to write metadata.json: {e}")
            if last_tmp is not None:
                try:
                    meta = _read_checkpoint_metadata(last_tmp)
                    if meta.get("is_temporary", False):
                        logger.info(f"Deleting old temporary checkpoint {last_tmp}")
                        self._queue_checkpoint_removal(last_tmp)
                    else:
                        logger.info(
                            f"Not deleting old temporary checkpoint {last_tmp} because it is no longer temporary."
                        )
                except FileNotFoundError:
                    logger.warning(f"Could not load metadata for last temporary checkpoint {last_tmp}.")

        self.save_checkpoint(info, destination, commit_callback=callback, is_temporary=not save_permanent)

    def save_checkpoint(
        self,
        tree: PyTree,
        destination: str,
        *,
        commit_callback: Callable[[], None] | None = None,
        is_temporary: bool = False,
        mesh: tp.Any = None,
        step: int = -1,
        shardings: tp.Any = None,
        partition_rules: tp.Any = None,
        prefix: str | None = None,
    ) -> None:
        """Save a checkpoint to the specified destination.

        This method saves a PyTree checkpoint using TensorStore backend with support
        for distributed training. It preserves existing array shardings (TP/FSDP)
        without performing all-gather operations, making it efficient for large models.

        Args:
            tree: PyTree to save. Can be any nested structure containing JAX arrays,
                NumPy arrays, or other serializable Python objects.
            destination: Subdirectory name under base_path where checkpoint will be
                saved. The full path will be base_path/destination.
            commit_callback: Optional callback function to execute after the checkpoint
                save completes. Used internally for metadata writing and cleanup.
            is_temporary: If True, marks this checkpoint as temporary in metadata.
                Temporary checkpoints are subject to automatic cleanup.
            mesh: Optional JAX mesh for distributed arrays. If None, the current
                mesh from the tree's shardings will be used.
            step: Training step number for this checkpoint. Stored in metadata.
                Defaults to -1 if not specified.
            shardings: Optional sharding specifications for arrays in the tree.
                If None, existing shardings are preserved from the tree.
            partition_rules: Optional partition rules for automatic sharding.
                Typically not needed if arrays already have shardings.
            prefix: Optional prefix for organizing multiple trees within the same
                checkpoint directory. Useful for saving multiple model states.

        Note:
            - Uses AsyncCheckpointManager for non-blocking I/O
            - Does NOT perform all-gather (preserves distributed arrays)
            - Creates destination directory if it doesn't exist
            - Updates internal tracking of last save step and time
            - All processes must call this method (distributed operation)

        Examples:
            ```python
            # Simple checkpoint save
            checkpointer.save_checkpoint(
                tree=model_state,
                destination="step-1000",
                step=1000,
            )

            # Save with mesh and callback
            def on_complete():
                print("Checkpoint saved!")

            checkpointer.save_checkpoint(
                tree=training_state,
                destination=f"step-{current_step}",
                mesh=my_mesh,
                step=current_step,
                commit_callback=on_complete,
            )
            ```
        """
        path = os.path.join(self.base_path, destination)
        fsspec_utils.mkdirs(path)

        logger.info(f"Saving checkpoint to {path}")

        # Use the AsyncCheckpointManager TS path; do_all_gather=False preserves sharding
        self._manager.save_tree(
            tree=tree,
            path=path,
            mesh=mesh,
            prefix=prefix,  # optional sub-prefix if you want multiple trees
            do_all_gather=False,
            cpu_offload=False,
            metadata={"is_temporary": is_temporary, "step": step},
            callback=commit_callback,
        )

        self._last_save_step = int(step)
        self._last_save_time = self._dt_now_injection()

    def load_checkpoint(
        self,
        mesh: tp.Any,
        *,
        path: str | None = None,
        discover_latest: bool = True,
        shardings: dict | None = None,
        partition_rules: tp.Any = None,
        dtype: tp.Any = None,
        prefix: str | None = None,
    ) -> tuple[PyTree, MetadataDict]:
        """Load a checkpoint from disk with automatic discovery.

        Loads a checkpoint directory and restores the PyTree structure with proper
        array shardings for distributed training. Can automatically discover the
        most recent checkpoint based on metadata timestamps.

        Args:
            mesh: JAX mesh for distributed array loading. Required for properly
                restoring sharded arrays across devices.
            path: Specific checkpoint directory to load. If None, uses base_path
                and discovers the latest checkpoint if discover_latest=True.
            discover_latest: If True, automatically finds and loads the most recent
                checkpoint under the specified path based on metadata timestamps.
                If False, loads from the exact path specified.
            shardings: Dictionary mapping array names to sharding specifications.
                Used to restore arrays with specific shardings. If None, attempts
                to restore original shardings from checkpoint metadata.
            partition_rules: Optional partition rules for automatic sharding inference.
                Alternative to explicit shardings dictionary.
            dtype: Optional dtype to cast loaded arrays to. If None, preserves
                original dtypes from the checkpoint.
            prefix: Optional prefix if the checkpoint contains multiple trees.
                Must match the prefix used during save_checkpoint.

        Returns:
            A tuple of (tree, metadata) where:
                - tree: Restored PyTree with the same structure as saved
                - metadata: Dictionary containing checkpoint metadata (step, timestamp, etc.)

        Raises:
            FileNotFoundError: If no checkpoint is found at the specified path
                or if discover_latest=True and no checkpoints exist.

        Note:
            - Automatically handles both local and cloud storage paths
            - Restores arrays with distributed shardings (no all-gather)
            - All processes must call this method (distributed operation)
            - Discovered checkpoints are sorted by timestamp then step number

        Examples:
            ```python
            # Load latest checkpoint automatically
            state, metadata = checkpointer.load_checkpoint(mesh=my_mesh)
            print(f"Loaded checkpoint from step {metadata['step']}")

            # Load specific checkpoint
            state, _ = checkpointer.load_checkpoint(
                mesh=my_mesh,
                path="/checkpoints/my_model/step-1000",
                discover_latest=False,
            )

            # Load with custom shardings
            state, _ = checkpointer.load_checkpoint(
                mesh=my_mesh,
                shardings=custom_shardings,
            )
            ```
        """
        root = path or self.base_path
        if discover_latest:
            discovered = find_latest_checkpoint(root)
            if discovered is None:
                raise FileNotFoundError(f"No checkpoint found under {root}")
            root = discovered

        logger.info(f"Loading checkpoint from {root}")
        tree, meta = self._manager.load(
            path=root,
            mesh=mesh,
            shardings=shardings,
            partition_rules=partition_rules,
            dtype=dtype,
            prefix=prefix,
        )
        return tree, meta

    def wait_until_finished(self) -> None:
        """Block until all checkpoint operations complete.

        Waits for both:
        1. All async checkpoint saves to finish writing to disk
        2. All background checkpoint deletion operations to complete

        This method should be called before program exit or when you need to
        ensure all checkpoint I/O has completed (e.g., before final evaluation
        or shutdown).

        Note:
            - Blocks the calling thread until completion
            - On non-primary processes, only waits for save operations
            - On primary process (0), also waits for deletion queue to empty
            - Uses polling with 0.2s sleep intervals for deletion queue

        Examples:
            ```python
            # At end of training
            for step in range(num_steps):
                train_step(state)
                checkpointer.on_step(state)

            checkpointer.wait_until_finished()
            print("All checkpoints saved and cleaned up")
            ```
        """
        self._manager.global_manager.wait_until_finished()
        if jax.process_index() == 0:
            while (
                getattr(self, "_checkpoint_being_removed", None) is not None
                or not self._checkpoint_cleanup_worker_queue.empty()
            ):
                time.sleep(0.2)

    def _queue_checkpoint_removal(self, checkpoint_dir: str) -> None:
        """Queue a checkpoint directory for asynchronous deletion.

        Args:
            checkpoint_dir: Full path to checkpoint directory to delete.

        Note:
            - Only executed on process 0 (primary process)
            - Adds directory to background deletion queue
            - Actual deletion happens asynchronously in separate thread
            - Safe to call from callback functions
        """
        if jax.process_index() == 0:
            logger.info(f"Queueing deletion of checkpoint {checkpoint_dir}")
            self._checkpoint_cleanup_worker_queue.put(checkpoint_dir)

    def _checkpoint_cleanup_worker(self) -> None:
        """Background thread worker that processes checkpoint deletion queue.

        This method runs in a daemon thread and continuously processes checkpoint
        directories from the deletion queue. It handles both local filesystem and
        cloud storage deletions using fsspec.

        The thread:
        1. Blocks waiting for checkpoint paths in the queue
        2. Deletes the checkpoint directory recursively
        3. Logs timing and any errors
        4. Updates tracking state for wait_until_finished()

        Note:
            - Runs as daemon thread (auto-terminates with main process)
            - Only started on process 0
            - Handles exceptions gracefully without crashing
            - Uses fsspec for storage abstraction (local/cloud)
            - Tracks currently deleting checkpoint for synchronization
        """
        while True:
            checkpoint = self._checkpoint_cleanup_worker_queue.get(block=True)
            self._checkpoint_being_removed = checkpoint
            try:
                fs, plain_path = _parse_storage_path(checkpoint)
                logger.info(f"Deleting old checkpoint {checkpoint} -> {plain_path}")
                t0 = time.time()
                fs.rm(plain_path, recursive=True)
                t1 = time.time()
                logger.info(f"Deleted {checkpoint} in {t1 - t0:.2f}s")
            except Exception:
                logger.exception(f"Failed to delete checkpoint {checkpoint}", exc_info=True)
            finally:
                self._checkpoint_being_removed = None

    def _get_save_interval_for_step(self, step: int) -> int | None:
        """Determine the checkpoint save interval for the given training step.

        Searches through step_policies to find which policy applies to the current
        training step and returns its save interval.

        Args:
            step: Current training step number.

        Returns:
            The checkpoint save interval (every N steps) that applies at this step,
            or None if no policy applies (shouldn't happen if policies are properly
            configured with final policy having until=None).

        Note:
            - Policies are evaluated in order
            - First policy where (until is None OR until >= step) is selected
            - Returns the 'every' value from the selected policy
        """
        current_policy = next((p for p in self.step_policies if p.until is None or p.until >= step), None)
        return None if current_policy is None else current_policy.every


def _write_checkpoint_metadata(checkpoint_path: str, step: int, is_temporary: bool) -> None:
    """Save checkpoint metadata to a JSON file.

    Creates a metadata.json file in the checkpoint directory containing step number,
    timestamp, and temporary status. Only executed on process 0 to avoid write conflicts.

    Args:
        checkpoint_path: Full path to checkpoint directory.
        step: Training step number for this checkpoint.
        is_temporary: Whether this is a temporary checkpoint (subject to cleanup).

    Note:
        - Only writes on process 0 (primary process)
        - Uses fsspec for storage abstraction (local/cloud)
        - Timestamp is in ISO 8601 format
        - Metadata is used by find_latest_checkpoint() for sorting
    """
    fs, plain_path = _parse_storage_path(checkpoint_path)
    meta = {
        "step": int(step),
        "timestamp": dt.datetime.now().isoformat(),
        "is_temporary": bool(is_temporary),
    }
    if jax.process_index() == 0:
        with fs.open(os.path.join(plain_path, "metadata.json"), "w") as out:
            json.dump(meta, out)


def _read_checkpoint_metadata(checkpoint_path: str, fs: AbstractFileSystem | None = None) -> MetadataDict:
    """Load checkpoint metadata from a JSON file.

    Reads the metadata.json file from a checkpoint directory and returns the
    parsed dictionary containing step, timestamp, and temporary status.

    Args:
        checkpoint_path: Full path to checkpoint directory.
        fs: Optional filesystem instance. If None, creates one from checkpoint_path.
            Providing an existing fs can improve performance when loading multiple
            metadata files.

    Returns:
        Dictionary containing checkpoint metadata with keys:
            - 'step': Training step number (int)
            - 'timestamp': ISO 8601 timestamp string
            - 'is_temporary': Boolean indicating if checkpoint is temporary

    Raises:
        FileNotFoundError: If metadata.json doesn't exist in the checkpoint directory.

    Note:
        - Uses fsspec for storage abstraction (local/cloud)
        - Can reuse filesystem instance for efficiency
    """
    if fs is None:
        fs, plain_path = _parse_storage_path(checkpoint_path)
    else:
        _, plain_path = _parse_storage_path(checkpoint_path, fs)
    with fs.open(os.path.join(plain_path, "metadata.json")) as f:
        return json.load(f)


def find_latest_checkpoint(base_path: str) -> str | None:
    """Find the most recent checkpoint under a directory.

    Searches for checkpoint directories containing metadata.json files and returns
    the path to the most recent one based on timestamp and step number.

    The function searches:
    1. All immediate subdirectories under base_path
    2. The base_path itself (in case it's a checkpoint directory)

    Args:
        base_path: Root directory to search for checkpoints. Can be a local path
            or cloud storage path (e.g., "gs://bucket/path").

    Returns:
        Full path to the latest checkpoint directory, or None if no valid
        checkpoints are found. If base_path has a URL scheme (e.g., "gs://"),
        the returned path preserves that scheme.

    Note:
        - Checkpoints are identified by presence of metadata.json
        - Sorting priority: timestamp first, then step number
        - Handles both local and cloud storage via fsspec
        - Logs warning if no checkpoints found
        - Handles FileNotFoundError gracefully (returns None)

    Examples:
        ```python
        # Local filesystem
        latest = find_latest_checkpoint("/checkpoints/my_model")
        # Returns: "/checkpoints/my_model/step-5000"

        # Cloud storage
        latest = find_latest_checkpoint("gs://bucket/checkpoints")
        # Returns: "gs://bucket/checkpoints/step-10000"

        # No checkpoints found
        latest = find_latest_checkpoint("/empty/dir")
        # Returns: None
        ```
    """
    fs, plain_base = _parse_storage_path(base_path)

    def is_ckpt_dir(path: str) -> bool:
        return fs.exists(os.path.join(path, "metadata.json"))

    # List immediate subdirs and also consider base_path itself
    try:
        candidates = [p for p in fs.glob(os.path.join(plain_base, "*")) if fs.isdir(p)]
    except FileNotFoundError:
        return None
    candidates.append(plain_base)

    ckpts = [p for p in candidates if is_ckpt_dir(p)]
    if not ckpts:
        logger.warning(f"No checkpoints found under {base_path}")
        return None

    def sort_key(path: str):
        meta = json.load(fs.open(os.path.join(path, "metadata.json")))
        ts = dt.datetime.fromisoformat(meta.get("timestamp"))
        step = int(meta.get("step", -1))
        return (ts, step)

    best = max(ckpts, key=sort_key)
    # If base_path used a scheme, re-attach it
    scheme = fsspec.core.split_protocol(base_path)[0] or ""
    if scheme:
        return f"{scheme}://{best}"
    return best


def _parse_storage_path(path: str, fs: AbstractFileSystem | None = None) -> tuple[AbstractFileSystem, str]:
    """Extract filesystem handler and plain path from a potentially prefixed path.

    Parses a path that may include a storage scheme (e.g., "gs://", "s3://") and
    returns the appropriate fsspec filesystem instance along with the path stripped
    of its scheme prefix.

    Args:
        path: Path string, potentially with scheme prefix (e.g., "gs://bucket/path"
            or "/local/path").
        fs: Optional existing filesystem instance. If provided, uses it for parsing
            to ensure consistency. If None, creates appropriate filesystem from path.

    Returns:
        A tuple of (filesystem, plain_path) where:
            - filesystem: fsspec AbstractFileSystem instance for the storage backend
            - plain_path: Path string without scheme prefix

    Examples:
        ```python
        # Cloud storage path
        fs, path = _parse_storage_path("gs://my-bucket/checkpoints")
        # Returns: (GCSFileSystem instance, "my-bucket/checkpoints")

        # Local path
        fs, path = _parse_storage_path("/home/user/checkpoints")
        # Returns: (LocalFileSystem instance, "/home/user/checkpoints")

        # Reuse filesystem
        fs1, path1 = _parse_storage_path("gs://bucket/path1")
        fs2, path2 = _parse_storage_path("gs://bucket/path2", fs=fs1)
        # fs1 and fs2 are the same instance
        ```

    Note:
        - Uses fsspec.core.url_to_fs for path parsing
        - Handles local filesystem, GCS, S3, and other fsspec-supported backends
        - Reusing filesystem instances can improve performance
    """
    if fs is None:
        fs, plain_path = fsspec.core.url_to_fs(str(path))
    else:
        _, plain_path = fsspec.core.url_to_fs(str(path), client=fs)  # ensure same parsing
    return fs, plain_path
