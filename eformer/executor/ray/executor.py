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

"""Ray-based executor for distributed machine learning workloads.

This module provides the core execution framework for running distributed
workloads on accelerators (TPUs, GPUs) using Ray. It supports single-pod,
multi-slice, and fault-tolerant execution patterns with automatic retry
mechanisms.

Key Features:
    - Single-pod and multi-slice execution on TPUs/GPUs
    - Automatic retry mechanisms for preemption and failures
    - Resource management and allocation via Ray
    - Support for both synchronous and asynchronous execution
    - Decorator-based API for easy integration
    - MegaScale coordination for multi-slice TPU workloads
    - Flexible result flattening for multi-host scenarios

Environment Variables:
    - EXECUTOR_CALL_INDEX: Set to worker index within a pod (0-based)
    - EXECUTOR_CALL_SLICE: Set to slice ID for multi-slice execution
    - COORD_PORT: Coordinator port for MegaScale (default: 8192)
    - TPU_NAME, TPU_VERSION, TPU_ZONE: TPU configuration passed to workers
    - MEGASCALE_* variables: Auto-configured for multi-slice coordination

Example:
    Basic single-pod execution:

    >>> import ray
    >>> from eformer.executor.ray import RayExecutor, TpuAcceleratorConfig
    >>>
    >>> @ray.remote
    >>> def train_model(data):
    ...     # Training logic here
    ...     return trained_model
    >>>
    >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
    >>> result = RayExecutor.execute_resumable(
    ...     train_model,
    ...     tpu_config,
    ...     max_retries_preemption=10,
    ...     max_retries_failure=3
    ... )

    Multi-slice execution with decorator:

    >>> from eformer.executor.ray import execute_multislice_resumable
    >>>
    >>> @execute_multislice_resumable(tpu_config)
    >>> @ray.remote
    >>> def distributed_train(slice_data):
    ...     # Distributed training logic
    ...     return slice_results
    >>>
    >>> results = distributed_train(training_data)

Classes:
    RayExecutor: Core executor with static methods for various execution patterns

Functions:
    execute: Decorator for single-pod execution without retry
    execute_resumable: Decorator for single-pod execution with retry
    execute_multislice: Decorator for multi-slice execution without retry
    execute_multislice_resumable: Decorator for multi-slice execution with retry
"""

import functools
import logging
import os

import ray
from ray.exceptions import RayError
from ray.remote_function import RemoteFunction

from .pool_manager import SlicePoolManager
from .resource_manager import (
    AcceleratorConfigType,
    RayResources,
    TpuAcceleratorConfig,
)
from .types import (
    JobError,
    JobFailed,
    JobInfo,
    JobPreempted,
    JobStatus,
    JobSucceeded,
    handle_ray_error,
)

ENV_CALL_INDEX = "EXECUTOR_CALL_INDEX"
ENV_CALL_SLICE = "EXECUTOR_CALL_SLICE"
MEGASCALE_DEFAULT_PORT = 8081

logger = logging.getLogger("ray")


class RayExecutor:
    """Core executor for Ray-based distributed workloads.

    Provides static methods to execute Ray remote functions on various
    accelerators (TPUs, GPUs) with support for single-pod, multi-slice,
    and fault-tolerant execution patterns.

    This class serves as the main interface for running distributed ML
    workloads with automatic resource allocation, retry mechanisms, and
    failure handling.

    Methods:
        execute: Single-pod execution without retry
        execute_multislice: Multi-slice execution without retry
        execute_resumable: Single-pod execution with automatic retry
        execute_multislice_resumable: Multi-slice execution with automatic retry

    All methods return JobStatus objects that encapsulate:
        - JobSucceeded: Successful completion with results
        - JobFailed: Failure due to exceptions
        - JobPreempted: Preemption on preemptible resources
        - JobError: Unexpected errors

    Note:
        All methods are static and can be called directly on the class.
        The class does not maintain state between executions.
    """

    @staticmethod
    def execute(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        **kwargs,
    ):
        """Execute a Ray remote function on a single pod or slice.

        Runs a Ray remote function on a single accelerator pod (TPU/GPU)
        with the specified resource configuration. For multi-slice TPU
        workloads, use execute_multislice instead.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute.
                Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources (TPU, GPU, or CPU).
            **kwargs: Additional keyword arguments passed to the remote function.

        Returns:
            ray.ObjectRef: A Ray future representing the result of execution.
                Call ray.get() on this to retrieve the actual result.

        Raises:
            AssertionError: If pod_count in accelerator_config is not 1,
                indicating that execute_multislice should be used instead.

        Example:
            >>> @ray.remote
            >>> def compute(x):
            ...     return x * 2
            >>>
            >>> config = GpuAcceleratorConfig(count=1, type="v100")
            >>> future = RayExecutor.execute(compute, config, x=10)
            >>> result = ray.get(future)  # Returns JobStatus object
        """
        assert getattr(accelerator_config, "pod_count", 1) == 1, (
            "Multi-slice workloads on TPUs should use 'execute_multislice'."
        )

        def do_run(
            remote_fn,
            accelerator_config: AcceleratorConfigType,
            kwargs,
        ) -> JobStatus:
            """Internal function to run the remote function with proper resource allocation.

            This function handles the actual execution of the remote function,
            managing multiple workers if specified in the configuration and
            capturing any errors that occur during execution.

            Args:
                remote_fn: The remote function to execute.
                accelerator_config: Accelerator configuration specifying resources.
                kwargs: Keyword arguments to pass to the remote function.

            Returns:
                JobStatus: Status object indicating:
                    - JobSucceeded: Execution completed successfully with results
                    - JobFailed: Execution failed with an error (non-preemption)
                    - JobPreempted: Execution was preempted (for preemptible instances)

            Note:
                Creates one Ray future per worker as specified in accelerator_config.worker_count.
                Each worker receives an ENV_CALL_INDEX environment variable set to its index.
            """
            info = JobInfo(accelerator_config.runtime_name, "running", accelerator_config.resource_name)
            futures = []
            for idx in range(accelerator_config.worker_count):
                _call = accelerator_config.redecorate_remote_fn_for_call(
                    remote_fn=remote_fn,
                    env_vars={ENV_CALL_INDEX: str(idx)},
                )
                futures.append(_call.remote(**kwargs))
            try:
                out = ray.get(futures)
                return JobSucceeded(info, out)
            except RayError as e:
                RayResources.cancel_all_futures(futures)
                return handle_ray_error(info, e)
            except Exception as e:
                RayResources.cancel_all_futures(futures)
                return JobFailed(info, e)

        if accelerator_config.head_name is None and not isinstance(accelerator_config, TpuAcceleratorConfig):
            do_run = ray.remote(do_run)
        else:
            default_name = f"TPU-{accelerator_config.tpu_version}-head"
            resources = {accelerator_config.head_name or default_name: accelerator_config.head_workers}
            do_run = ray.remote(resources=resources)(do_run)
        return do_run.remote(remote_fn, accelerator_config, kwargs)

    @staticmethod
    def execute_multislice(
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        flatten: bool = True,
        **kwargs,
    ) -> JobStatus:
        """Execute a Ray remote function across multiple TPU slices.

        Distributes execution of a remote function across multiple TPU slices
        for large-scale parallel processing. This method sets up the necessary
        infrastructure including slice actors, placement groups, and MegaScale
        coordination environment variables.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute
                on each slice. Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources, must include multi-slice details
                (pod_count > 1).
            **kwargs: Additional keyword arguments passed to the remote
                function on each slice.

        Returns:
            JobStatus: A single JobStatus object containing results from all slices.
                JobSucceeded contains a list of all results if successful.
                JobFailed or JobPreempted if any error occurs.

        Raises:
            RayError: If slice actor creation fails, coordinator IP cannot
                be determined, or remote function calls fail.

        Note:
            The method automatically sets up MegaScale environment variables
            for multi-slice coordination including coordinator address, slice IDs,
            and port configuration.

        Example:
            >>> @ray.remote
            >>> def train_on_slice(data, slice_id):
            ...     # Training logic for this slice
            ...     return model_weights
            >>>
            >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
            >>> job_status = RayExecutor.execute_multislice(
            ...     train_on_slice,
            ...     tpu_config,
            ...     data=training_data
            ... )
            >>> # job_status is a single JobStatus object
            >>> if isinstance(job_status, JobSucceeded):
            ...     results = job_status.result  # List of results from all slices
        """
        pool_manager = SlicePoolManager(tpu_type=accelerator_config.tpu_version)
        pool_manager.scale_multislice(accelerator_config.pod_count)

        pool_manager.prepare_all_slices()
        members = pool_manager.get_all_pool_members()
        if not members:
            raise RuntimeError("No SliceActors available after scaling.")
        ray.get([m.actor.ensure_host_pool.remote() for m in members])

        slice_infos = ray.get([m.actor.get_info.remote() for m in members])
        coord_ip = slice_infos[0].ip_address
        if not coord_ip:
            raise RuntimeError("Could not determine coordinator IP.")
        port = int(os.getenv("COORD_PORT", 8192))

        base_env = dict(
            TPU_NAME=os.getenv("TPU_NAME", "EMPTY"),
            TPU_VERSION=accelerator_config.tpu_version,
            TPU_ZONE=os.getenv("TPU_ZONE", "EMPTY"),
            TPU_POD_COUNT=str(len(members)),
        )
        if accelerator_config.execution_env:
            base_env.update({str(k): str(v) for k, v in accelerator_config.execution_env.items() if v is not None})

        per_slice_futures: list[list[ray.ObjectRef]] = []

        for slice_id, member in enumerate(members):
            if len(members) > 1:
                env_for_slice = dict(
                    **base_env,
                    MEGASCALE_COORDINATOR_ADDRESS=f"{coord_ip}:{port}",
                    MEGASCALE_NUM_SLICES=str(len(members)),
                    MEGASCALE_PORT=str(port),
                    MEGASCALE_SLICE_ID=str(slice_id),
                    EXECUTOR_CALL_INDEX="0",
                    EXECUTOR_CALL_SLICE=str(slice_id),
                    TPU_SLICE_NAME=slice_infos[slice_id].slice_name,
                )
            else:
                env_for_slice = base_env
            env_for_slice = {str(k): str(v) for k, v in env_for_slice.items()}
            host_handles: list[ray.actor.ActorHandle] = ray.get(member.actor.get_all_actors_in_pool.remote())
            host_futures = [
                h.run_remote_fn.remote(
                    remote_fn,
                    runtime_env=accelerator_config.execution_env,
                    env=env_for_slice,
                )
                for h in host_handles
            ]
            per_slice_futures.append(host_futures)

        all_futures = [f for sub in per_slice_futures for f in sub] if flatten else per_slice_futures

        info = JobInfo(accelerator_config.runtime_name, "running", accelerator_config.resource_name)

        try:
            results = ray.get(ray.get(all_futures))
            return JobSucceeded(info, results)
        except RayError as e:
            if flatten:
                RayResources.cancel_all_futures(all_futures)
            else:
                for slice_futures in all_futures:
                    RayResources.cancel_all_futures(slice_futures)
            return handle_ray_error(info, e)
        except Exception as e:
            if flatten:
                RayResources.cancel_all_futures(all_futures)
            else:
                for slice_futures in all_futures:
                    RayResources.cancel_all_futures(slice_futures)
            return JobFailed(info, e)

    @classmethod
    def execute_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """Execute a remote function with automatic retry on failures.

        Provides fault-tolerant execution of Ray remote functions with
        configurable retry policies for both preemptions and failures.
        Particularly useful for long-running jobs on preemptible resources.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute.
                Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources.
            max_retries_preemption (int): Maximum number of retries on
                preemption. Defaults to 1,000,000 (effectively unlimited).
            max_retries_failure (int): Maximum number of retries on failure.
                Defaults to 10.
            **kwargs: Additional keyword arguments passed to the remote function.

        Returns:
            Any: The result from successful execution of the remote function.
                The actual return type depends on what the remote function returns.

        Raises:
            RuntimeError: If the job is preempted more than max_retries_preemption
                times or fails more than max_retries_failure times. The error
                message indicates whether it was due to preemptions or failures.
            ray.exceptions.RayTaskError: Re-raised if it occurs and is not
                preemption-related after max retries.
            Exception: The last encountered exception if all retries are exhausted.

        Note:
            - Preemptions and failures are tracked separately
            - Each attempt logs status information for debugging
            - The method distinguishes between preemption (often recoverable)
              and failures (may indicate code issues)
            - RayTaskErrors containing "preempted" are treated as preemptions

        Example:
            >>> @ray.remote
            >>> def long_running_task(data):
            ...     # Task that might be preempted
            ...     return process(data)
            >>>
            >>> config = TpuAcceleratorConfig(type="v4-8", preemptible=True)
            >>> result = RayExecutor.execute_resumable(
            ...     long_running_task,
            ...     config,
            ...     max_retries_preemption=100,
            ...     max_retries_failure=5,
            ...     data=my_data
            ... )
        """
        num_failures = 0
        num_preemptions = 0
        attempt = 0
        problem: Exception | None = None

        while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
            logger.info(f"Running on Attempt {attempt}")
            attempt += 1
            problem = None
            try:
                out = ray.get(
                    cls.execute(
                        remote_fn=remote_fn,
                        accelerator_config=accelerator_config,
                        **kwargs,
                    )
                )
            except ray.exceptions.RayTaskError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(f"Preempted {num_preemptions} times, {e}")
                else:
                    num_failures += 1
                    logger.warning(f"Failed {num_failures} times (RayTaskError)", exc_info=e)
                continue
            except Exception as e:
                problem = e
                num_failures += 1
                if num_failures >= max_retries_failure:
                    logger.exception("Failed too many times", exc_info=e)
                    raise e
                else:
                    logger.warning(f"Failed {num_failures} times", exc_info=e)
                    continue

            if isinstance(out, JobSucceeded):
                result = out.result
                logger.info("Success")
                return result
            elif isinstance(out, JobPreempted):
                problem = out.error
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
            elif isinstance(out, JobFailed):
                problem = out.error
                num_failures += 1
                logger.warning(
                    f"JobFailed reported. Incrementing failure count to {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
            elif isinstance(out, JobError):
                problem = out.error
                num_failures += 1
                logger.warning(f"Failed {num_failures} times", exc_info=problem)
            else:
                raise RuntimeError(f"Unexpected result: {out}")

        if num_preemptions >= max_retries_preemption:
            raise RuntimeError("Preempted too many times") from problem
        elif num_failures >= max_retries_failure:
            raise RuntimeError("Failed too many times") from problem

    @classmethod
    def execute_multislice_resumable(
        cls,
        remote_fn: RemoteFunction,
        accelerator_config: AcceleratorConfigType,
        max_retries_preemption: int = int(1e6),
        max_retries_failure: int = 10,
        **kwargs,
    ):
        """Execute a multi-slice function with automatic retry on failures.

        Provides fault-tolerant execution of Ray remote functions across
        multiple TPU slices with coordinated retry mechanisms. All slices
        must succeed for the execution to be considered successful.

        Args:
            remote_fn (RemoteFunction): The Ray remote function to execute
                on each slice. Must be decorated with @ray.remote.
            accelerator_config (AcceleratorConfigType): Configuration for
                accelerator resources with multi-slice support (pod_count > 1).
            max_retries_preemption (int): Maximum number of retries when
                any slice is preempted. Defaults to 1,000,000.
            max_retries_failure (int): Maximum number of retries when any
                slice fails. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the remote
                function on each slice. The 'flatten' parameter can be used
                to control result structure.

        Returns:
            list[Any]: List of results from successful execution on all slices.
                The structure depends on the flatten parameter passed in kwargs:
                - If flatten=True (default): Flat list of all results
                - If flatten=False: List of lists, one per slice

        Raises:
            RuntimeError: If any slice is preempted more than max_retries_preemption
                times, fails more than max_retries_failure times, or if
                execute_multislice returns None or unexpected result type.
            RayError: If execute_multislice fails during setup or coordination
                (slice actor creation, placement group setup, etc.).
            ray.exceptions.RayTaskError: Re-raised if it occurs and indicates
                preemption or failure after max retries.
            Exception: The last encountered exception if retries are exhausted.

        Note:
            - Implements an all-or-nothing retry policy: if any slice fails
              or is preempted, the entire multi-slice execution is retried
            - Different error types are handled with appropriate retry logic:
              * RayTaskError/RayError with "preempted" -> preemption counter
              * Other errors -> failure counter
            - Each retry attempt creates new slice actors and placement groups
            - Detailed logging tracks retry attempts and error types

        Example:
            >>> @ray.remote
            >>> def distributed_training(data_shard):
            ...     # Training logic for each slice
            ...     return trained_weights
            >>>
            >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
            >>>
            >>> # Get flat list of results
            >>> results = RayExecutor.execute_multislice_resumable(
            ...     distributed_training,
            ...     tpu_config,
            ...     max_retries_preemption=50,
            ...     max_retries_failure=3,
            ...     data_shard=sharded_data
            ... )
            >>>
            >>> # Get nested results by slice
            >>> results_by_slice = RayExecutor.execute_multislice_resumable(
            ...     distributed_training,
            ...     tpu_config,
            ...     max_retries_preemption=50,
            ...     max_retries_failure=3,
            ...     data_shard=sharded_data,
            ...     flatten=False
            ... )
        """
        num_failures = 0
        num_preemptions = 0
        attempt = 0
        problem: Exception | None = None

        while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
            logger.info(f"Running multislice on Attempt {attempt}")
            attempt += 1
            problem = None
            job_status: JobStatus | None = None

            try:
                job_status = cls.execute_multislice(remote_fn=remote_fn, accelerator_config=accelerator_config, **kwargs)

            except ray.exceptions.RayTaskError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(
                        f"A slice was preempted (RayTaskError). Preemption count: {num_preemptions}. Error: {e}"
                    )
                else:
                    num_failures += 1
                    logger.warning(f"A slice failed (RayTaskError). Failure count: {num_failures}.", exc_info=e)
                continue
            except RayError as e:
                problem = e
                if "preempted" in str(e).lower():
                    num_preemptions += 1
                    logger.warning(
                        f"Multislice operation preempted during setup/coordination (RayError). "
                        f"Preemption count: {num_preemptions}. Error: {e}"
                    )
                else:
                    num_failures += 1
                    logger.warning(
                        f"Multislice operation failed during setup/coordination (RayError)."
                        f" Failure count: {num_failures}.",
                        exc_info=e,
                    )
                continue

            except Exception as e:
                problem = e
                num_failures += 1
                if num_failures >= max_retries_failure:
                    logger.exception(
                        "Multislice operation failed too many times (non-Ray/RayTaskError).",
                        exc_info=e,
                    )
                    raise e
                else:
                    logger.warning(
                        f"Multislice operation failed (non-Ray/RayTaskError). Failure count: {num_failures}.",
                        exc_info=e,
                    )
                    continue

            if not job_status:
                logger.warning("execute_multislice returned None. Treating as failure.")
                num_failures += 1
                problem = problem or RuntimeError("No job status from execute_multislice")
                continue

            if isinstance(job_status, JobSucceeded):
                logger.info("All slices succeeded in this attempt.")
                return job_status.result
            elif isinstance(job_status, JobPreempted):
                problem = job_status.error
                num_preemptions += 1
                logger.warning(
                    f"Multislice execution preempted. Preemption count: {num_preemptions}. Error: {problem}",
                    exc_info=problem,
                )
                continue
            elif isinstance(job_status, JobFailed):
                problem = job_status.error
                num_failures += 1
                logger.warning(
                    f"Multislice execution failed (JobFailed). Failure count: {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
                continue
            elif isinstance(job_status, JobError):
                problem = job_status.error
                num_failures += 1
                logger.warning(
                    f"Multislice execution reported JobError. Failure count: {num_failures}. Error: {problem}",
                    exc_info=problem,
                )
                continue
            else:
                err_msg = f"Unexpected result type {type(job_status)} from execute_multislice: {job_status}"
                problem = RuntimeError(err_msg)
                num_failures += 1
                logger.error(err_msg)
                continue

        if num_preemptions >= max_retries_preemption:
            logger.error(f"Multislice job preempted too many times ({num_preemptions} >= {max_retries_preemption}).")
            raise RuntimeError(f"Preempted too many times ({num_preemptions})") from problem
        elif num_failures >= max_retries_failure:
            logger.error(f"Multislice job failed too many times ({num_failures} >= {max_retries_failure}).")
            raise RuntimeError(f"Failed too many times ({num_failures})") from problem

        raise RuntimeError(
            "Exhausted retries for multislice execution without explicit success or reaching failure/preemption limits."
        ) from problem


def execute_resumable(accelerator_config: AcceleratorConfigType):
    """Decorator for fault-tolerant single-pod execution.

    Wraps a Ray remote function to automatically use RayExecutor.execute_resumable
    with the specified accelerator configuration. The decorated function will
    automatically retry on preemption or failure according to the default retry
    policies (1,000,000 retries for preemption, 10 for failures).

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources to use for execution. Should have pod_count=1 for
            single-pod execution.

    Returns:
        Callable: Decorator function that wraps the remote function and adds
            automatic retry logic.

    Note:
        To customize retry behavior, use RayExecutor.execute_resumable directly
        with max_retries_preemption and max_retries_failure parameters.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
        >>>
        >>> @execute_resumable(tpu_config)
        >>> @ray.remote
        >>> def my_task(data):
        ...     return process(data)
        >>>
        >>> result = my_task(input_data)  # Automatically retries on failure
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute_resumable(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def execute(accelerator_config: AcceleratorConfigType):
    """Decorator for single-pod execution without retry.

    Wraps a Ray remote function to automatically use RayExecutor.execute
    with the specified accelerator configuration. Results are automatically
    retrieved with ray.get(). This decorator is suitable for tasks that
    don't require fault tolerance or where failures should be handled
    by the caller.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources to use for execution. Should have pod_count=1 for
            single-pod execution.

    Returns:
        Callable: Decorator function that wraps the remote function and
            automatically retrieves results.

    Note:
        Unlike execute_resumable, this decorator does not retry on failure.
        Use this for quick tasks or when you want to handle failures yourself.

    Example:
        >>> gpu_config = GpuAcceleratorConfig(count=2, type="a100")
        >>>
        >>> @execute(gpu_config)
        >>> @ray.remote
        >>> def gpu_task(tensor):
        ...     return tensor.cuda() * 2
        >>>
        >>> result = gpu_task(my_tensor)  # Executes on GPU, no retry
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def execute_multislice(accelerator_config: AcceleratorConfigType):
    """Decorator for multi-slice execution without retry.

    Wraps a Ray remote function to automatically use RayExecutor.execute_multislice
    with the specified accelerator configuration. Results from all slices are
    automatically retrieved with ray.get(). The function will be executed
    across multiple TPU slices in parallel, with MegaScale coordination
    automatically configured.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources with multi-slice support. Must have pod_count > 1.

    Returns:
        Callable: Decorator function that wraps the remote function and returns
            a list of results, one from each slice.

    Note:
        The decorator handles slice actor creation, placement group setup,
        and MegaScale environment configuration automatically.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4)
        >>>
        >>> @execute_multislice(tpu_config)
        >>> @ray.remote
        >>> def parallel_compute(data_shard):
        ...     return compute_result(data_shard)
        >>>
        >>> results = parallel_compute(sharded_data)  # Returns list from 4 slices
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute_multislice(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator


def execute_multislice_resumable(accelerator_config: AcceleratorConfigType):
    """Decorator for fault-tolerant multi-slice execution.

    Wraps a Ray remote function to automatically use RayExecutor.execute_multislice_resumable
    with the specified accelerator configuration. Provides automatic retry on
    preemption or failure of any slice. Uses an all-or-nothing retry policy:
    if any slice fails, the entire multi-slice execution is retried.

    Args:
        accelerator_config (AcceleratorConfigType): Configuration for accelerator
            resources with multi-slice support. Must have pod_count > 1.

    Returns:
        Callable: Decorator function that wraps the remote function and adds
            automatic retry logic for all slices.

    Note:
        Default retry limits are 1,000,000 for preemptions and 10 for failures.
        To customize these limits, use RayExecutor.execute_multislice_resumable
        directly with max_retries_preemption and max_retries_failure parameters.

    Example:
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", pod_count=4, preemptible=True)
        >>>
        >>> @execute_multislice_resumable(tpu_config)
        >>> @ray.remote
        >>> def resilient_training(data_batch):
        ...     # Long-running training that might be preempted
        ...     return train_model(data_batch)
        >>>
        >>> results = resilient_training(training_data)  # Auto-retries on failure
    """

    def decorator(remote_fn: RemoteFunction):
        @functools.wraps(remote_fn)
        def wrapper(**kwargs):
            return RayExecutor.execute_multislice_resumable(
                remote_fn=remote_fn,
                accelerator_config=accelerator_config,
                **kwargs,
            )

        return wrapper

    return decorator
