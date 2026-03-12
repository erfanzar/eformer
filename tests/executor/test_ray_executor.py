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

"""Regression tests for the Ray executor helpers."""

import inspect
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray

from eformer.executor.ray.docker_executor import DockerConfig, run_docker_on_pod
from eformer.executor.ray.executor import RayExecutor, _extract_runtime_env_vars, execute, execute_resumable
from eformer.executor.ray.pool_manager import DeviceHostActor, _preserve_metadata_slice_topology
from eformer.executor.ray.resource_manager import CpuAcceleratorConfig, RayResources
from eformer.executor.ray.types import JobSucceeded


@pytest.fixture
def local_ray():
    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    try:
        yield
    finally:
        ray.shutdown()


def test_separate_process_fn_respects_disable_fork(monkeypatch):
    monkeypatch.setenv("EFORMER_DISABLE_FORK", "1")

    assert RayResources.separate_process_fn(os.getpid, (), {}) == os.getpid()


def test_extract_runtime_env_vars_only_returns_env_vars():
    runtime_env = {
        "env_vars": {"A": "1", "B": 2, "EMPTY": None},
        "working_dir": "/tmp/workdir",
        "pip": ["numpy"],
    }

    assert _extract_runtime_env_vars(runtime_env) == {"A": "1", "B": "2"}


def test_kill_vfio_holders_is_disabled_by_default(monkeypatch):
    actor_cls = DeviceHostActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)

    monkeypatch.delenv("EFORMER_KILL_VFIO", raising=False)
    with patch("shutil.which", return_value="/usr/bin/lsof"), patch("subprocess.run") as run_quiet:
        actor._kill_vfio_holders()

    assert run_quiet.call_count == 0


def test_lockfile_cleanup_uses_noninteractive_sudo():
    actor_cls = DeviceHostActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)

    with (
        patch("os.path.exists", return_value=True),
        patch("os.unlink", side_effect=PermissionError),
        patch("subprocess.run") as run_cmd,
    ):
        actor._hacky_remove_tpu_lockfile()

    assert run_cmd.call_args[0][0] == ["sudo", "-n", "rm", "-f", "/tmp/libtpu_lockfile"]


def test_preserve_metadata_slice_topology_keeps_full_host_count_for_partial_ray_registration(monkeypatch):
    monkeypatch.setenv("EFORMER_MODERATE", "1")

    with (
        patch("eformer.executor.ray.pool_manager.ray.cluster_resources", return_value={"slice-a": 1}),
        patch("eformer.executor.ray.pool_manager.logger.warning") as warn,
    ):
        assert _preserve_metadata_slice_topology("slice-a", 8, 4) == (8, 4)

    warn.assert_called_once()
    assert "1/8 hosts for slice slice-a" in warn.call_args[0][0]


def test_preserve_metadata_slice_topology_skips_ray_lookup_when_disabled(monkeypatch):
    monkeypatch.setenv("EFORMER_MODERATE", "0")

    with patch(
        "eformer.executor.ray.pool_manager.ray.cluster_resources",
        side_effect=AssertionError("cluster_resources should not be queried"),
    ):
        assert _preserve_metadata_slice_topology("slice-a", 8, 4) == (8, 4)


def test_execute_supports_plain_functions_with_positional_args(local_ray):
    @execute(CpuAcceleratorConfig(core_count=1, worker_count=1))
    def plain_add(x, y):
        return x + y

    result = plain_add(20, 22)

    assert isinstance(result, JobSucceeded)
    assert result.result == [42]


def test_execute_resumable_supports_plain_functions(local_ray):
    @execute_resumable(CpuAcceleratorConfig(core_count=1, worker_count=1))
    def plain_scale(*, x):
        return x * 2

    assert plain_scale(x=21) == [42]


def test_execute_resumable_preserves_positional_retry_arguments(local_ray):
    def plain_scale(*, x):
        return x * 2

    accelerator = CpuAcceleratorConfig(core_count=1, worker_count=1)

    assert RayExecutor.execute_resumable(plain_scale, accelerator, 1, 1, x=21) == [42]


def test_execute_multislice_preserves_positional_flatten_parameter():
    flatten = inspect.signature(RayExecutor.autoscale_execute).parameters["flatten"]

    assert flatten.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_execute_resumable_preserves_positional_retry_parameters():
    signature = inspect.signature(RayExecutor.execute_resumable)

    assert signature.parameters["max_retries_preemption"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["max_retries_failure"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_fork_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("EFORMER_DISABLE_FORK", raising=False)

    assert RayResources.fork_disabled() is True


def test_run_docker_on_pod_returns_stdout_for_single_worker(local_ray):
    config = DockerConfig(image="busybox", command=["echo", "hi"])
    accelerator = CpuAcceleratorConfig(core_count=1, worker_count=1)

    with patch(
        "eformer.executor.ray.docker_executor.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="hi\n", stderr=""),
    ):
        assert run_docker_on_pod(config, accelerator) == "hi\n"


def test_run_docker_on_pod_returns_outputs_for_multiple_workers(local_ray):
    config = DockerConfig(image="busybox", command=["echo", "hi"])
    accelerator = CpuAcceleratorConfig(core_count=1, worker_count=2)

    with patch(
        "eformer.executor.ray.docker_executor.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="hi\n", stderr=""),
    ):
        assert run_docker_on_pod(config, accelerator) == ["hi\n", "hi\n"]
