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

"""Resource pool management for distributed Ray actors.

This module provides abstractions for managing pools of Ray actors,
particularly focused on TPU/GPU slice management for distributed computing.
It includes health monitoring, automatic scaling, and resource lifecycle
management.

Key Components:
    - **ActorPoolMember**: Wrapper for actor handles with metadata
    - **ResourcePoolManager**: Abstract base for managing actor pools
    - **SlicePoolManager**: Specialized manager for TPU/GPU slices
    - **SliceActor**: Ray actor for managing individual compute slices

Example:
    Managing a multi-slice TPU configuration:

    >>> from eformer.executor.ray import SlicePoolManager
    >>>
    >>> manager = SlicePoolManager(tpu_type="v4-8")
    >>> manager.scale_multislice(num_slices=4)
    >>> actors = manager.get_all_actors_in_pool()
    >>> manager.drain_actor_pool()
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar
from uuid import uuid4

import ray
import requests
from ray.actor import ActorHandle
from ray.autoscaler.sdk import request_resources
from ray.exceptions import ActorDiedError, ActorUnavailableError, GetTimeoutError
from ray.remote_function import RemoteFunction as _RF
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from .types import SliceInfo

logger = logging.getLogger("ray")

HEALTH_CHECK_TIMEOUT_S = 60
SLICE_ACTOR_START_TIMEOUT_S = 4 * 60 * 60
ActorInfoT = TypeVar("ActorInfoT")


@dataclass(frozen=True)
class HostInfo:
    """Information about a TPU host within a slice.

    Attributes:
        host_id: Unique identifier for the host within its slice.
        slice_name: Name of the TPU slice this host belongs to.
        num_devices: Number of TPU devices available on this host.
        healthy: Whether the host is currently healthy and operational.
        failed: Whether the host has encountered a failure.
    """

    host_id: int
    slice_name: str
    num_devices: int | None
    healthy: bool
    failed: bool
    node_id: str | None = None


@dataclass(frozen=True)
class ActorPoolMember(Generic[ActorInfoT]):
    """Container for an actor handle and its associated metadata.

    Attributes:
        actor: Ray actor handle for remote execution.
        actor_info: Metadata about the actor (type depends on ActorInfoT).
    """

    actor: ActorHandle
    actor_info: ActorInfoT


class ResourcePoolManager(Generic[ActorInfoT]):
    """Abstract base class for managing pools of Ray actors.

    Provides common functionality for scaling, health monitoring, and
    lifecycle management of actor pools. Subclasses should implement
    create_actor() to define how actors are created.

    Attributes:
        _actor_pool: List of active actor pool members.
    """

    def __init__(self) -> None:
        """Initialize an empty actor pool."""
        self._actor_pool: list[ActorPoolMember[ActorInfoT]] = []

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        """Get all actor handles in the pool.

        Returns:
            List of Ray actor handles.
        """
        return [m.actor for m in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember[ActorInfoT]]:
        """Get a copy of all pool members with their metadata.

        Returns:
            List of ActorPoolMember objects containing actors and their info.
        """
        return self._actor_pool.copy()

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        return self.__class__.__name__

    def get_actor_name_from_actor_info(self, actor_info: ActorInfoT) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return str(actor_info)

    def create_actor(self) -> ActorHandle:
        """Create a new actor instance.

        Must be implemented by subclasses to define actor creation logic.

        Returns:
            Ray actor handle for the newly created actor.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        """Remove unhealthy actors from the pool.

        Performs health checks on all actors and removes those that are
        unresponsive, dead, or unhealthy. Attempts to kill removed actors.
        """
        healthy: list[ActorPoolMember[ActorInfoT]] = []
        for member in self._actor_pool:
            try:
                ray.get(member.actor.healthy.remote(), timeout=HEALTH_CHECK_TIMEOUT_S)
                healthy.append(member)
            except (ActorDiedError, ActorUnavailableError, GetTimeoutError) as e:
                name = self.get_actor_name_from_actor_info(member.actor_info)
                logger.warning(f"Removing unhealthy actor {name}: {e}")
                try:
                    ray.kill(member.actor)
                except Exception:
                    pass
        self._actor_pool = healthy

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        """Add new actors to the pool to reach desired size.

        Creates new actors asynchronously and waits for them to start.
        Actors that fail to start within the timeout are killed.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        current = len(self._actor_pool)
        if current >= desired_num_actors:
            return
        num_to_add = desired_num_actors - current
        logger.info(f"Scaling up pool {self.get_actor_pool_name()} from {current} to {desired_num_actors}")

        actors = [self.create_actor() for _ in range(num_to_add)]
        awaitables = [(actor, actor.get_info.remote()) for actor in actors]

        logger.info(f"Waiting up to {SLICE_ACTOR_START_TIMEOUT_S}s for {num_to_add} slice actors to start...")
        ray.wait([a for _, a in awaitables], num_returns=len(awaitables), timeout=SLICE_ACTOR_START_TIMEOUT_S)

        started = 0
        for actor, info_ref in awaitables:
            try:
                info = ray.get(info_ref, timeout=0)
                self._actor_pool.append(ActorPoolMember(actor, info))
                started += 1
                logger.info(f"Added actor {self.get_actor_name_from_actor_info(info)}")
            except Exception as e:
                logger.warning(f"SliceActor failed to start in time: {e}; killing actor")
                try:
                    ray.kill(actor)
                except Exception:
                    pass

        logger.info(f"Started {started}/{num_to_add} slice actors")

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        """Remove actors to reach the desired pool size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        while len(self._actor_pool) > desired_num_actors:
            member = self._actor_pool.pop()
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor)
                logger.info(f"Removed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

    def _scale_actor_pool(self, desired_num_actors: int) -> None:
        """Scale the actor pool to the desired size.

        First removes unhealthy actors, then adds or removes actors
        as needed to reach the target size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        self._remove_unhealthy_members_from_actor_pool()
        current = len(self._actor_pool)
        if current < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors)
        elif current > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)

    def drain_actor_pool(self) -> None:
        """Shut down and remove all actors from the pool.

        Attempts graceful shutdown first, then forcefully kills actors.
        Clears the actor pool after draining.
        """
        for member in self._actor_pool:
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor)
                logger.info(f"Killed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")
        self._actor_pool = []


@ray.remote
class DeviceHostActor:
    """Ray actor for managing a single TPU host within a slice.

    Handles task execution on a specific TPU host, managing TPU resources,
    environment variables, and task lifecycle. Supports cancellation and
    health monitoring.

    Attributes:
        host_id: Unique identifier for this host within its slice.
        slice_name: Name of the TPU slice this host belongs to.
        num_devices: Number of TPU devices on this host.
        _failed: Whether this host has encountered a failure.
        _awaitable: Current running task's ObjectRef.
        _node_id: Ray node ID where this actor is running.
    """

    def __init__(self, host_id: int, slice_name: str, num_devices: int | None = None):
        """Initialize a DeviceHostActor.

        Args:
            host_id: Unique identifier for this host within its slice.
            slice_name: Name of the TPU slice this host belongs to.
            num_devices: Optional number of TPU devices available on this host.
        """
        self.host_id = host_id
        self.slice_name = slice_name
        self.num_devices = num_devices or 0
        self._failed = False
        self._awaitable: ray.ObjectRef | None = None
        self._node_id = ray.get_runtime_context().get_node_id()
        logger.info(f"DeviceHostActor[{slice_name}#{host_id}] init; num_devices={num_devices}; node_id={self._node_id}")

    def healthy(self) -> bool:
        """Check if this host is healthy and operational.

        Returns:
            True if host is not failed and not being preempted.
        """
        return not self._failed and not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        """Check if this GCP instance is being preempted.

        Queries the GCP metadata server to determine if the instance
        is scheduled for preemption.

        Returns:
            True if instance is being preempted, False otherwise.
        """
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
                headers={"Metadata-Flavor": "Google"},
                timeout=1.0,
            )
            return r.status_code == 200 and r.text.strip().upper() == "TRUE"
        except requests.RequestException:
            return False

    def get_info(self) -> HostInfo:
        """Get current information about this host.

        Returns:
            HostInfo object with host metadata and status.
        """
        return HostInfo(
            host_id=self.host_id,
            slice_name=self.slice_name,
            num_devices=self.num_devices,
            healthy=self.healthy(),
            failed=self._failed,
            node_id=self._node_id,
        )

    def _merge_runtime_env(self, runtime_env: dict | None, env_vars: dict | None) -> dict:
        """Merge environment variables into a runtime environment dict.

        Args:
            runtime_env: Base runtime environment configuration.
            env_vars: Environment variables to merge in.

        Returns:
            Merged runtime environment dictionary.
        """
        re = dict(runtime_env or {})
        if env_vars:
            ev = dict(re.get("env_vars", {}))
            ev.update({str(k): str(v) for k, v in env_vars.items() if v is not None})
            re["env_vars"] = ev
        return re

    def _hacky_remove_tpu_lockfile(self):
        """Remove TPU lockfile that may prevent TPU initialization.

        Attempts to remove /tmp/libtpu_lockfile which can cause issues
        when reusing TPU resources. Falls back to sudo if needed.
        """
        try:
            if os.path.exists("/tmp/libtpu_lockfile"):
                os.unlink("/tmp/libtpu_lockfile")
        except FileNotFoundError:
            pass
        except PermissionError:
            try:
                os.system("sudo rm /tmp/libtpu_lockfile")
            except Exception:
                pass

    def _cancel_tasks_and_wait(self, tasks: list[ray.ObjectRef], timeout_s: float = 240.0) -> None:
        """Cancel Ray tasks and wait for them to complete.

        Forcefully cancels all provided tasks and waits for completion
        or timeout.

        Args:
            tasks: List of Ray ObjectRefs to cancel.
            timeout_s: Maximum time to wait for cancellation.
        """
        if not tasks:
            return
        try:
            for t in tasks:
                ray.cancel(t, force=True, recursive=True)
        except Exception as e:
            logger.warning(f"Failed to cancel some tasks: {e}")
        done, pending = ray.wait(tasks, num_returns=len(tasks), timeout=timeout_s)
        if pending:
            logger.warning(f"Cancelled {len(done)} tasks; {len(pending)} still pending after {timeout_s}s.")

    def cancel_current(self):
        """Cancel the currently running task if any.

        Cancels and waits for the current task to complete,
        then clears the awaitable reference.
        """
        if self._awaitable:
            self._cancel_tasks_and_wait([self._awaitable])
            self._awaitable = None

    def run_remote_fn(
        self,
        remote_fn,
        *,
        runtime_env: dict | None = None,
        env: dict | None = None,
        num_cpus: float = 8.0,
        memory_bytes: float = 20e9,
        extra_resources: dict | None = None,
    ) -> ray.ObjectRef:
        """
        Launch a cancelable task on this host's node, reserving TPU for the task.
        Returns the task ObjectRef (caller can wait/cancel).
        """
        if not self.healthy():
            raise RuntimeError(f"Host {self.host_id} unhealthy or preempted")

        if not isinstance(remote_fn, _RF):
            remote_fn = ray.remote(max_calls=1)(remote_fn)
        elif remote_fn._default_options.get("max_calls") is None:
            raise ValueError("remote_fn must have max_calls=1 for TPU workloads")

        if self._awaitable:
            self._cancel_tasks_and_wait([self._awaitable])

        self._hacky_remove_tpu_lockfile()
        host_env = {"TPU_HOST_ID": str(self.host_id), "TPU_SLICE_NAME": self.slice_name}
        if self.num_devices:
            host_env["TPU_NUM_DEVICES"] = str(self.num_devices)
        merged_runtime_env = self._merge_runtime_env(runtime_env, {**host_env, **(env or {})})
        resources = dict(extra_resources or {})
        if self.num_devices:
            resources["TPU"] = self.num_devices

        self._awaitable = remote_fn.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_id, soft=False),
            resources=resources,
            num_cpus=num_cpus,
            num_gpus=0,
            memory=memory_bytes,
            runtime_env=merged_runtime_env,
        ).remote()
        return self._awaitable

    def shutdown(self) -> None:
        """Gracefully shut down this host actor.

        Cancels any running task and marks the host as failed.
        """
        try:
            self.cancel_current()
        finally:
            self._failed = True
            logger.info(f"Shut down DeviceHostActor[{self.slice_name}#{self.host_id}]")


@ray.remote
class SliceActor:
    """Ray actor for managing a TPU slice with multiple hosts.

    Coordinates multiple TPU hosts within a single slice, handling
    placement groups, resource allocation, and distributed task execution.

    Attributes:
        _actor_pool: List of active actor pool members.
        _failed: Whether this slice has failed.
        _slice_info: Information about the TPU slice.
        _host_placement_group: Ray placement group for host distribution.
        _host_infos: Information about each host in the slice.
    """

    def __init__(self):
        """Initialize a slice actor.

        Creates an empty actor pool and prepares to manage TPU hosts
        within a single slice. Discovers slice information from the
        TPU environment during initialization.
        """
        self._actor_pool: list[ActorPoolMember[HostInfo]] = []
        self._failed = False
        self._slice_info: SliceInfo | None = None
        self._host_placement_group = None
        self._host_infos: list[dict] | None = None
        self._initialize_slice_info()

    @staticmethod
    @ray.remote(num_cpus=0)
    def discover_node_info():
        """Discover information about the current Ray node.

        Static remote function to gather node metadata including IP,
        node ID, pod name, and TPU count.

        Returns:
            Dictionary with node information.
        """
        import ray

        pod_name = None
        try:
            from ray.util.accelerators import tpu as ray_tpu

            pod_name = ray_tpu.get_current_pod_name()
        except Exception:
            pass
        num_devices = None
        try:
            from ray._private.accelerators import TPUAcceleratorManager

            num_devices = TPUAcceleratorManager.get_current_node_num_accelerators()
        except Exception:
            pass
        return {
            "ip": ray.util.get_node_ip_address(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "pod_name": pod_name,
            "num_devices": num_devices,
        }

    def _create_actor_for_host_id(self, host_id: int) -> ActorHandle:
        """Create a DeviceHostActor for a specific host ID.

        Creates an actor with node affinity to ensure it runs on the
        correct TPU host. The actor is created as detached to persist
        beyond the parent's lifetime.

        Args:
            host_id: Zero-based index of the host within the slice.

        Returns:
            Ray actor handle for the DeviceHostActor.

        Raises:
            RuntimeError: If slice/host info not initialized or host_id invalid.
        """
        if not self._slice_info or not self._host_infos:
            raise RuntimeError("Slice or host info not initialized")
        if host_id >= len(self._host_infos):
            raise RuntimeError(f"Missing host info for host_id={host_id}")

        info = self._host_infos[host_id]
        node_id = info["node_id"]
        num_devices_for_host = info.get("num_devices")

        return DeviceHostActor.options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            lifetime="detached",
            name=f"{self._slice_info.slice_name}-host-{host_id}-{uuid4().hex[:8]}",
        ).remote(
            host_id,
            self._slice_info.slice_name,
            num_devices_for_host,
        )

    def _initialize_slice_info(self) -> None:
        """Initialize slice information from TPU environment.

        Discovers slice name, host count, and TPU configuration.
        Sets _failed flag if initialization fails.
        """
        try:
            from ray.util.accelerators import tpu as ray_tpu

            slice_name = ray_tpu.get_current_pod_name()
            num_hosts = int(ray_tpu.get_current_pod_worker_count())
            ip_address = ray.util.get_node_ip_address()

            num_accelerators_per_host = None
            try:
                from ray._private.accelerators import TPUAcceleratorManager

                num_accelerators_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
            except Exception:
                pass

            self._slice_info = SliceInfo(
                slice_name=slice_name,
                num_hosts=num_hosts,
                ip_address=ip_address,
                num_accelerators_per_host=num_accelerators_per_host or 0,
            )
            logger.info(f"Initialized SliceActor: {self._slice_info}")
        except Exception as e:
            logger.error(f"Failed to initialize slice info: {e}")
            self._failed = True

    def healthy(self) -> bool:
        """Check if the slice is healthy and operational.

        Verifies that the slice has not failed and is not being preempted
        by the cloud provider.

        Returns:
            True if slice is healthy and available for task execution,
            False if failed or being preempted.
        """
        if self._failed:
            return False
        return not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        """Check if this GCP instance is being preempted.

        Queries GCP metadata server to determine preemption status.

        Returns:
            True if instance is being preempted, False otherwise.
        """
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
                headers={"Metadata-Flavor": "Google"},
                timeout=1.0,
            )
            return r.status_code == 200 and r.text.strip().upper() == "TRUE"
        except requests.RequestException:
            return False

    def get_info(self) -> SliceInfo:
        """Get current information about this slice.

        Returns:
            SliceInfo object containing slice configuration, host count,
            IP addresses, and TPU device information.

        Raises:
            RuntimeError: If slice information has not been initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        return self._slice_info

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        """Get all actor handles in the pool.

        Returns:
            List of Ray actor handles.
        """
        return [m.actor for m in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember[HostInfo]]:
        """Get a copy of all pool members with their metadata.

        Returns:
            List of ActorPoolMember objects containing actors and their info.
        """
        return self._actor_pool.copy()

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        if self._slice_info:
            return f"SliceActor({self._slice_info.slice_name})"
        return "SliceActor(uninitialized)"

    def get_actor_name_from_actor_info(self, actor_info: HostInfo) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return f"{actor_info.slice_name}-host-{actor_info.host_id}"

    def _ensure_host_placement_group(self) -> None:
        """Ensure placement group exists for distributing hosts.

        Creates a placement group with STRICT_SPREAD strategy to ensure
        hosts are distributed across different nodes. Also discovers
        host information for each placement group bundle.

        Raises:
            RuntimeError: If slice info is not initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        if self._host_placement_group is not None:
            return

        slice_label = self._slice_info.slice_name
        bundles = [{"CPU": 0, slice_label: 1} for _ in range(self._slice_info.num_hosts)]
        request_resources(bundles=bundles)
        self._host_placement_group = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(self._host_placement_group.ready())

        futures = [
            SliceActor.discover_node_info.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    self._host_placement_group,
                    placement_group_bundle_index=i,
                    placement_group_capture_child_tasks=True,
                )
            ).remote()
            for i in range(self._slice_info.num_hosts)
        ]
        self._host_infos = ray.get(futures)
        self._slice_info = SliceInfo(
            slice_name=self._slice_info.slice_name,
            num_hosts=self._slice_info.num_hosts,
            ip_address=self._slice_info.ip_address,
            num_accelerators_per_host=self._slice_info.num_accelerators_per_host,
            node_ids=[h.get("node_id") for h in self._host_infos],
            host_infos=self._host_infos,
        )
        logger.info(f"Prepared host placement group for slice {self._slice_info.slice_name};")

    def prepare_hosts(self) -> None:
        """Prepare host placement group for this slice.

        Ensures the placement group is created and ready for host actors.
        """
        self._ensure_host_placement_group()

    def create_actor(self) -> ActorHandle:
        """Create a new TPU host actor within this slice.

        Creates a DeviceHostActor with proper node affinity to ensure it runs
        on the correct host within the slice. Assigns TPU resources if
        available on the target node.

        Returns:
            Ray actor handle for the newly created DeviceHostActor.

        Raises:
            RuntimeError: If slice is not initialized or host info is missing.
        """
        if not self._slice_info:
            raise RuntimeError("Cannot create host actor: slice not initialized")
        self._ensure_host_placement_group()

        host_id = len(self._actor_pool)
        if not self._host_infos or host_id >= len(self._host_infos):
            raise RuntimeError(f"Missing host info for host_id={host_id}")

        info = self._host_infos[host_id]
        node_id = info["node_id"]
        num_devices_for_host = info.get("num_devices")

        return DeviceHostActor.options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(host_id, self._slice_info.slice_name, num_devices_for_host)

    def ensure_host_pool(self, desired_hosts: int | None = None) -> None:
        """Ensure the host actor pool has the desired number of hosts.

        Args:
            desired_hosts: Target number of hosts, defaults to slice's host count.

        Raises:
            RuntimeError: If slice info is not initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        self._ensure_host_placement_group()
        target = desired_hosts if desired_hosts is not None else self._slice_info.num_hosts
        self._scale_actor_pool(target)

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        """Remove unhealthy actors from the pool.

        Performs health checks on all actors and removes those that are
        unresponsive, dead, or unhealthy. Attempts to kill removed actors.
        """
        healthy: list[ActorPoolMember[HostInfo]] = []
        for member in self._actor_pool:
            try:
                ray.get(member.actor.healthy.remote(), timeout=HEALTH_CHECK_TIMEOUT_S)
                healthy.append(member)
            except (ActorDiedError, ActorUnavailableError, GetTimeoutError) as e:
                name = self.get_actor_name_from_actor_info(member.actor_info)
                logger.warning(f"Removing unhealthy actor {name}: {e}")
                try:
                    ray.kill(member.actor)
                except Exception:
                    pass
        self._actor_pool = healthy

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        """Add new host actors to reach the desired pool size.

        Creates host actors sequentially to ensure correct host_id assignment.
        Each actor is given a timeout to start, after which it's killed.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        current = len(self._actor_pool)
        if current >= desired_num_actors:
            return
        num_to_add = desired_num_actors - current
        logger.info(f"Scaling up pool {self.get_actor_pool_name()} from {current} to {desired_num_actors}")

        for host_id in range(current, current + num_to_add):
            actor = self._create_actor_for_host_id(host_id)
            try:
                info = ray.get(actor.get_info.remote(), timeout=HEALTH_CHECK_TIMEOUT_S)
                self._actor_pool.append(ActorPoolMember(actor, info))
                logger.info(f"Added actor {self.get_actor_name_from_actor_info(info)}")
            except Exception as e:
                logger.error(f"Failed to start actor for host_id={host_id}: {e}")
                try:
                    ray.kill(actor)
                except Exception:
                    pass

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        """Remove actors to reach the desired pool size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        while len(self._actor_pool) > desired_num_actors:
            member = self._actor_pool.pop()
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor)
                logger.info(f"Removed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

    def _scale_actor_pool(self, desired_num_actors: int) -> None:
        """Scale the actor pool to the desired size.

        First removes unhealthy actors, then adds or removes actors
        as needed to reach the target size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        self._remove_unhealthy_members_from_actor_pool()
        current = len(self._actor_pool)
        if current < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors)
        elif current > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)

    def drain_actor_pool(self) -> None:
        """Shut down and remove all actors from the pool.

        Attempts graceful shutdown first, then forcefully kills actors.
        Clears the actor pool after draining.
        """
        for member in self._actor_pool:
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor)
                logger.info(f"Killed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")
        self._actor_pool = []

    def run_remote_fn(self, remote_fn, runtime_env: dict | None = None, env: dict | None = None):
        """Execute a remote function on all hosts in this slice.

        Ensures all hosts are ready, then launches the function on each host
        in parallel. The function runs with TPU resources reserved.

        Args:
            remote_fn: Ray remote function or callable to execute.
            runtime_env: Optional Ray runtime environment configuration.
            env: Optional environment variables to set.

        Returns:
            List of Ray ObjectRefs for each host's execution.

        Raises:
            RuntimeError: If slice info is not initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        self.ensure_host_pool(self._slice_info.num_hosts)
        futures = [
            member.actor.run_remote_fn.remote(remote_fn, runtime_env=runtime_env, env=env) for member in self._actor_pool
        ]
        return futures

    def shutdown(self):
        """Gracefully shut down this slice actor.

        Removes the placement group, marks the slice as failed,
        and prevents any new task execution on this slice.
        """
        try:
            self.drain_actor_pool()
        except Exception:
            pass
        if self._host_placement_group:
            try:
                remove_placement_group(self._host_placement_group)
            except Exception:
                pass
            self._host_placement_group = None
        self._failed = True


class SlicePoolManager(ResourcePoolManager[SliceInfo]):
    """Manager for multiple TPU slices.

    Coordinates multiple SliceActors to manage multi-slice TPU configurations.
    Handles scaling, health monitoring, and distributed task execution across
    multiple TPU slices.

    Attributes:
        _tpu_type: Type of TPU (e.g., "v4-8").
        _last_scale_ts: Timestamp of last scaling operation.
        _last_scale_check_ts: Timestamp of last scale check.
    """

    def __init__(self, tpu_type: str | None):
        """Initialize a slice pool manager.

        Args:
            tpu_type: Type of TPU to manage (e.g., "v4-8", "v5e-16").
                     Used for resource labeling and identification.
        """
        super().__init__()
        self._tpu_type = tpu_type
        self._last_scale_ts: float | None = None
        self._last_scale_check_ts: float | None = None

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        return f"SlicePool({self._tpu_type})"

    def get_actor_name_from_actor_info(self, actor_info: SliceInfo) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return actor_info.slice_name

    def create_actor(self) -> ActorHandle:
        """Create a new SliceActor to manage a TPU slice.

        Creates a SliceActor with appropriate resource requirements
        based on the TPU type. The actor will manage all hosts within
        its assigned slice.

        Returns:
            Ray actor handle for the newly created SliceActor.
        """
        return SliceActor.options(num_cpus=0, resources={f"TPU-{self._tpu_type}-head": 1}).remote()

    def scale_multislice(self, num_slices: int | Sequence[int]) -> None:
        """Scale the pool to the desired number of slices.

        Supports flexible scaling with multiple valid sizes. Will scale
        to the largest feasible size from the provided options.

        Args:
            num_slices: Target number of slices or list of valid sizes.

        Raises:
            ValueError: If target is invalid or empty.
            RuntimeError: If requested sizes cannot be achieved.
        """
        self._last_scale_ts = time.time()

        if isinstance(num_slices, int):
            valid = [int(num_slices)]
        else:
            valid = sorted({int(x) for x in num_slices})
            if not valid:
                raise ValueError("valid sizes list is empty")

        target = valid[-1]
        if target <= 0:
            raise ValueError(f"Target slice count must be > 0, got {target}")

        head_bundles = [{"CPU": 0, f"TPU-{self._tpu_type}-head": 1} for _ in range(target)]
        request_resources(bundles=head_bundles)

        self._scale_actor_pool(target)
        current = len(self._actor_pool)
        if current not in valid:
            feasible = [v for v in valid if v <= current]
            if not feasible:
                raise RuntimeError(f"Requested one of {valid}, but only {current} slices available")
            self._scale_actor_pool(feasible[-1])

    def prepare_all_slices(self) -> None:
        """Prepare all slices by ensuring host placement groups.

        Pre-requests resources for all slices and prepares their host
        placement groups for distributed execution.
        """

        slice_infos: list[SliceInfo] = ray.get([member.actor.get_info.remote() for member in self._actor_pool])
        for info in slice_infos:
            host_bundles = [{"CPU": 0, info.slice_name: 1} for _ in range(info.num_hosts)]
            request_resources(bundles=host_bundles)

        ray.get([member.actor.prepare_hosts.remote() for member in self._actor_pool])

    def should_scale_up_multislice(self, valid_sizes: Sequence[int]) -> bool:
        """Check if pool should scale up to a larger size.

        Implements rate limiting to prevent frequent scaling operations.

        Args:
            valid_sizes: List of valid pool sizes.

        Returns:
            True if scaling up is recommended.
        """
        self._last_scale_check_ts = time.time()
        current = len(self._actor_pool)
        larger = [size for size in valid_sizes if size > current]
        if not larger:
            return False
        if self._last_scale_ts and (time.time() - self._last_scale_ts) < 60:
            return False
        return True

    def execute_on_each_slice(self, remote_fn, env: dict | None = None, runtime_env: dict | None = None):
        """Execute a function on all hosts across all slices.

        Prepares all slices and runs the function on every host in parallel.
        Returns results grouped by slice.

        Args:
            remote_fn: Ray remote function or callable to execute.
            env: Optional environment variables to set.
            runtime_env: Optional Ray runtime environment configuration.

        Returns:
            List of lists where outer list represents slices and inner lists
            contain ObjectRefs for each host in that slice.
        """
        self.prepare_all_slices()
        per_slice_futures = ray.get(
            [m.actor.run_remote_fn.remote(remote_fn, runtime_env=runtime_env, env=env) for m in self._actor_pool]
        )
        return per_slice_futures

    def execute_on_each_host_flat(self, remote_fn, env: dict | None = None, runtime_env: dict | None = None):
        """Execute a function on all hosts, returning a flat list.

        Similar to execute_on_each_slice but flattens the nested result
        structure into a single list of ObjectRefs.

        Args:
            remote_fn: Ray remote function or callable to execute.
            env: Optional environment variables to set.
            runtime_env: Optional Ray runtime environment configuration.

        Returns:
            Flat list of ObjectRefs from all hosts across all slices.
        """
        per_slice = self.execute_on_each_slice(remote_fn, env=env, runtime_env=runtime_env)
        return [f for sub in per_slice for f in sub]

    def execute_on_each_host(self, fn, *args, env: dict | None = None, **kwargs):
        """Execute a function on each host across all slices.

        Prepares all slices and runs the function on every host actor
        in parallel across all slices.

        Args:
            fn: Function to execute on each host.
            *args: Positional arguments for fn.
            env: Optional environment variables.
            **kwargs: Keyword arguments for fn.

        Returns:
            Nested list of results (outer list for slices, inner for hosts).
        """
        self.prepare_all_slices()
        ray.get([member.actor.ensure_host_pool.remote() for member in self._actor_pool])
        return ray.get(
            [
                member.actor.execute_function.remote(fn, *args, use_host_actors=True, env=env, **kwargs)
                for member in self._actor_pool
            ]
        )

    def schedule_on_each_host(self, remote_fn, env: dict | None = None, runtime_env: dict | None = None):
        """Schedule a function on all hosts without waiting for results.

        Ensures all slices and hosts are ready, then schedules the function
        execution on all hosts. Returns immediately with ObjectRefs.

        Args:
            remote_fn: Ray remote function or callable to execute.
            env: Optional environment variables to set.
            runtime_env: Optional Ray runtime environment configuration.

        Returns:
            Flat list of ObjectRefs that can be waited on later.
        """
        self.prepare_all_slices()
        ray.get([m.actor.ensure_host_pool.remote() for m in self._actor_pool])
        per_slice_futures = ray.get(
            [m.actor.run_remote_fn.remote(remote_fn, runtime_env=runtime_env, env=env) for m in self._actor_pool]
        )
        return [f for sl in per_slice_futures for f in sl]
