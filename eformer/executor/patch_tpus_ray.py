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

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import typing as tp
from functools import wraps
from typing import Any

import requests
import yaml

TPU_VERSION = os.getenv("TPU_VERSION", "v4")
TPU_SLICE_SIZE = int(os.getenv("TPU_SLICE_SIZE", "8"))
TPU_CORES_PER_HOST = int(os.getenv("TPU_CORES_PER_HOST", "4"))
SSH_USER = os.getenv("PATCHER_USER")
INTERNAL_IPS = ["0.0.0.0"]
EXTERNAL_IPS = ["0.0.0.0"]


COLORS: dict[str, str] = {
    "PURPLE": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "ORANGE": "\033[38;5;208m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "RESET": "\033[0m",
    "BLUE_PURPLE": "\033[38;5;99m",
}

LEVEL_COLORS: dict[str, str] = {
    "DEBUG": COLORS["ORANGE"],
    "INFO": COLORS["BLUE_PURPLE"],
    "WARNING": COLORS["YELLOW"],
    "ERROR": COLORS["RED"],
    "CRITICAL": COLORS["RED"] + COLORS["BOLD"],
    "FATAL": COLORS["RED"] + COLORS["BOLD"],
}

_LOGGING_LEVELS: dict[str, int] = {
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "WARN": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
    "critical": 50,
    "fatal": 50,
    "error": 40,
    "warning": 30,
    "warn": 30,
    "info": 20,
    "debug": 10,
    "notset": 0,
}


def _resources_json(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"))


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        color = LEVEL_COLORS.get(record.levelname, COLORS["RESET"])
        record.levelname = f"{color}{record.levelname:<8}{COLORS['RESET']}"
        current_time = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        formatted_name = f"{color}({current_time} {record.name}){COLORS['RESET']}"
        message = f"{formatted_name} {record.getMessage()}"
        record.levelname = orig_levelname
        return message


class LazyLogger:
    def __init__(self, name: str, level: int | None = None):
        self._name = name
        self._level = level or _LOGGING_LEVELS[os.getenv("LOGGING_LEVEL_ED", "INFO")]
        self._logger: logging.Logger | None = None

    def _ensure_initialized(self) -> None:
        if self._logger is not None:
            return

        logger = logging.getLogger(self._name)
        logger.propagate = False
        logger.setLevel(self._level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)
        formatter = ColorFormatter()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self._logger = logger

    def __getattr__(self, name: str) -> tp.Callable:
        if name in _LOGGING_LEVELS or name.upper() in _LOGGING_LEVELS or name in ("exception", "log"):

            @wraps(getattr(logging.Logger, name))
            def wrapped_log_method(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                self._ensure_initialized()
                return getattr(self._logger, name)(*args, **kwargs)

            return wrapped_log_method
        raise AttributeError(f"'LazyLogger' object has no attribute '{name}'")


def get_logger(name: str, level: int | None = None) -> LazyLogger:
    return LazyLogger(name, level)


class RayManager:
    """Manages Ray operations including starting, stopping, and verifying clusters."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ray_path = self._find_ray_executable()

    def _find_ray_executable(self) -> pathlib.Path:
        """Return the absolute path to 'ray' inside the current venv (or system)."""
        if ray_path := os.getenv("RAY_EXECUTABLE_PATH"):
            path = pathlib.Path(ray_path).expanduser().resolve()
            self.logger.debug(f"Using RAY_EXECUTABLE_PATH: {path}")
            return path

        bin_dir = pathlib.Path(sys.executable).parent
        ray_path = bin_dir / "ray"
        if ray_path.is_file():
            self.logger.debug(f"Found ray in bin directory: {ray_path}")
            return ray_path

        if ray_path := shutil.which("ray"):
            path = pathlib.Path(ray_path)
            self.logger.debug(f"Found ray in PATH: {path}")
            return path

        self.logger.error("Ray executable could not be located")
        raise FileNotFoundError("ray executable could not be located")

    def run_command(
        self,
        command: str | list[str],
        use_sudo: bool = False,
        check: bool = True,
        capture_output: bool = False,
        remote_ip: str | None = None,
    ) -> bool | str:
        """Run a command locally or remotely with optional output capture."""
        if isinstance(command, list):
            cmd_str = " ".join([str(c) for c in command])
        else:
            cmd_str = command

        self.logger.info(f"Running command: {cmd_str} {'on ' + remote_ip if remote_ip else 'locally'}")

        if use_sudo:
            if isinstance(command, list):
                full_command = ["sudo", *command]
            else:
                full_command = f"sudo {command}"
        else:
            full_command = command

        try:
            if remote_ip:
                if isinstance(full_command, list):
                    remote_cmd = shlex.join([str(c) for c in full_command])
                else:
                    remote_cmd = full_command
                ssh_command = [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "ConnectTimeout=5",
                    f"{SSH_USER}@{remote_ip}",
                    remote_cmd,
                ]
                result = subprocess.run(ssh_command, check=check, capture_output=capture_output, text=True)
                return result.stdout if capture_output else (result.returncode == 0)
            else:
                if isinstance(full_command, list):
                    result = subprocess.run(full_command, check=check, capture_output=capture_output, text=True)
                else:
                    result = subprocess.run(
                        full_command,
                        shell=True,
                        check=check,
                        capture_output=capture_output,
                        text=True,
                    )
                return result.stdout if capture_output else (result.returncode == 0)
        except subprocess.CalledProcessError as e:
            location = f"on {remote_ip}" if remote_ip else "locally"
            self.logger.error(f"Command failed {location}: {cmd_str}. Error: {e.stderr}")
            return False if not capture_output else ""
        except FileNotFoundError:
            self.logger.error(f"Command not found: {cmd_str}")
            return False if not capture_output else ""

    def is_ray_running(self, ip: str) -> bool:
        """Check if Ray is already running on a node."""
        local_ip = IPManager.get_local_ip()

        if local_ip == ip:
            try:
                subprocess.check_output("ps aux | grep -v grep | grep 'ray::IDLE'", shell=True)
                self.logger.debug(f"Ray is running locally on {ip}")
                return True
            except subprocess.CalledProcessError:
                self.logger.debug(f"Ray is not running locally on {ip}")
                return False
        else:
            result = self.run_command("ps aux | grep -v grep | grep 'ray::IDLE'", remote_ip=ip)
            if result:
                self.logger.debug(f"Ray is running remotely on {ip}")
            else:
                self.logger.debug(f"Ray is not running remotely on {ip}")
            return result

    def start_head_node(self, head_ip: str, resources: dict[str, Any]) -> bool:
        """Start Ray head node."""
        self.logger.info(f"Starting Ray head node on {head_ip}")
        local_ip = IPManager.get_local_ip()

        if self.is_ray_running(head_ip):
            self.logger.info(f"Ray is already running on head node {head_ip}. Stopping it first...")
            self.stop_node(head_ip)
            time.sleep(3)

        resources_str = _resources_json(resources)
        cmd = [
            self.ray_path,
            "start",
            "--head",
            "--port=6379",
            f"--resources={resources_str}",  # Fixed: Removed single quotes around resources_str
            f"--node-ip-address={head_ip}",
            "--dashboard-host=0.0.0.0",
        ]

        success = self.run_command(cmd, remote_ip=None if local_ip == head_ip else head_ip)

        if not success:
            self.logger.error("Failed to start Ray head node")
            return False

        self.logger.info("Ray head node started successfully")
        return True

    def start_worker_node(self, head_ip: str, worker_ip: str, resources: dict[str, Any]) -> bool:
        """Start Ray worker node."""
        self.logger.info(f"Starting Ray worker on {worker_ip}")
        local_ip = IPManager.get_local_ip()

        if self.is_ray_running(worker_ip):
            self.logger.info(f"Ray is already running on worker node {worker_ip}. Stopping it first...")
            self.stop_node(worker_ip)
            time.sleep(3)

        resources_str = _resources_json(resources)
        cmd = [
            str(self.ray_path),
            "start",
            f"--address={head_ip}:6379",
            f"--resources={resources_str}",
            f"--node-ip-address={worker_ip}",
        ]
        success = self.run_command(cmd, remote_ip=None if local_ip == worker_ip else worker_ip)

        if success:
            self.logger.info(f"Successfully started Ray worker on {worker_ip}")
            return True
        else:
            self.logger.error(f"Failed to start Ray worker on {worker_ip}")
            return False

    def stop_node(self, ip: str) -> bool:
        """Stop Ray on a specific node."""
        self.logger.info(f"Stopping Ray on {ip}...")
        local_ip = IPManager.get_local_ip()

        if local_ip == ip:
            return self.run_command([self.ray_path, "stop"])
        else:
            return self.run_command(f"{self.ray_path} stop", remote_ip=ip)

    def stop_cluster(self, ips: list[str]) -> None:
        """Stop Ray cluster on all nodes."""
        self.logger.info("Stopping Ray on all nodes...")
        for ip in ips:
            self.stop_node(ip)
        self.logger.info("Ray cluster stopped on all nodes.")

    def verify_cluster(self, head_ip: str, expected_tpu_count: int, allow_head_only: bool = False) -> bool:
        """Verify Ray cluster setup."""
        self.logger.info("Verifying Ray cluster setup...")
        if allow_head_only and expected_tpu_count == 0:
            self.logger.info("Head-only node detected (no TPU resources)")
            return True

        # Create verification script
        script_content = """
import ray
import time
import sys
import os

# Try to connect multiple times
for i in range(5):
    try:
        ray.init(address='auto')
        break
    except Exception as e:
        print(f"Connection attempt {i+1} failed: {e}")
        time.sleep(5)
        if i == 4:  # Last attempt failed
            print("Could not connect to Ray cluster")
            sys.exit(1)

# Print cluster resources
try:
    resources = ray.cluster_resources()
    print("\\nCLUSTER RESOURCES:\\n==================")
    for k, v in sorted(resources.items()):
        print(f"{k}: {v}")

    # Check if we have the expected TPUs
    tpu_count = resources.get('TPU', 0)
    print(f"\\nTOTAL TPU COUNT: {tpu_count}")

    # Get expected TPU count from environment variable
    expected_tpu = int(os.environ.get('EXPECTED_TPU_COUNT', 64))
    if tpu_count < expected_tpu * 0.9:  # Allow for 10% missing
        print(f"WARNING: Not all TPU cores are detected! Expected ~{expected_tpu}")
        sys.exit(2)
    else:
        print(f"SUCCESS: TPU cores detected ({tpu_count}/{expected_tpu})")
except Exception as e:
    print(f"Error getting cluster resources: {e}")
    sys.exit(3)

ray.shutdown()
"""
        script_path = "/tmp/verify_ray_tpu.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        self.logger.debug(f"Created verification script at {script_path}")

        env = os.environ.copy()
        env["EXPECTED_TPU_COUNT"] = str(expected_tpu_count)

        try:
            subprocess.run(["python", script_path], env=env, check=True)
            self.logger.info("Verification successful!")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Verification failed with status {e.returncode}")
            return False
        finally:
            try:
                os.remove(script_path)
                self.logger.debug("Removed verification script")
            except FileNotFoundError:
                self.logger.warning("Verification script not found for removal")


class IPManager:
    """Manages IP address operations and configurations."""

    @staticmethod
    def get_external_ip() -> str:
        """Get the external IP address of the machine."""
        urls = ["https://api.ipify.org", "https://ipinfo.io/ip", "https://checkip.amazonaws.com"]

        logger = logging.getLogger(__name__)

        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    ip = response.text.strip()
                    logger.debug(f"Retrieved external IP from {url}: {ip}")
                    return ip
            except Exception as e:
                logger.debug(f"Failed to get external IP from {url}: {e}")
                continue

        logger.warning("Could not determine external IP")
        return "Could not determine external IP"

    @staticmethod
    def get_local_ip() -> str:
        """Get the local IP address of the machine."""
        logger = logging.getLogger(__name__)

        try:
            hostname_output = subprocess.check_output("hostname -I", shell=True).decode().strip()
            ip = hostname_output.split()[0]
            logger.debug(f"Retrieved local IP: {ip}")
            return ip
        except Exception as e:
            logger.error(f"Error getting local IP: {e}")
            return "127.0.0.1"

    @staticmethod
    def read_ips_from_yaml(yaml_file: str) -> bool:
        """Read IPs from YAML file."""
        global INTERNAL_IPS, EXTERNAL_IPS
        logger = logging.getLogger(__name__)

        try:
            with open(yaml_file, "r") as file:
                data = yaml.safe_load(file)

            INTERNAL_IPS = data.get("internal_ips", [])
            EXTERNAL_IPS = data.get("external_ips", [])

            if not INTERNAL_IPS:
                logger.error("No internal IPs found in the YAML file")
                return False

            logger.info(f"Read {len(INTERNAL_IPS)} internal IPs and {len(EXTERNAL_IPS)} external IPs from {yaml_file}")
            return True
        except Exception as e:
            logger.error(f"Error reading YAML file: {e}")
            return False

    @staticmethod
    def test_ssh_connectivity(ips: list[str]) -> bool:
        """Test SSH connectivity to all nodes."""
        logger = logging.getLogger(__name__)
        logger.info("Testing SSH connectivity to all nodes...")
        all_good = True

        for ip in ips:
            logger.info(f"Testing connection to {ip}...")
            ssh_command = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                f"{SSH_USER}@{ip}",
                "echo OK",
            ]

            try:
                subprocess.run(
                    ssh_command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Connection to {ip}: SUCCESS")
            except subprocess.CalledProcessError:
                logger.error(f"Connection to {ip}: FAILED")
                all_good = False

        if all_good:
            logger.info("All SSH connections successful!")
            return True
        else:
            logger.error("Some SSH connections failed. Please check your SSH setup.")
            logger.error("Make sure your SSH key is added to the authorized_keys file on all nodes.")
            logger.error(f"You may need to run: ssh-copy-id {SSH_USER}@<node-ip>")
            return False


class ClusterManager:
    """Manages TPU cluster setup and configuration."""

    def __init__(self, ray_manager: RayManager):
        self.logger = logging.getLogger(__name__)
        self.ray_manager = ray_manager

    def setup_cluster(self, use_external: bool) -> bool:
        """Set up Ray cluster."""
        ips = EXTERNAL_IPS if use_external else INTERNAL_IPS
        head_ip = ips[0]
        worker_ips = ips[1:]

        self.logger.info(f"Setting up Ray cluster with {'external' if use_external else 'internal'} IPs")
        self.logger.info(f"TPU Version: {TPU_VERSION}")
        self.logger.info(f"TPU Slice Size: {TPU_SLICE_SIZE}")
        self.logger.info(f"TPU Cores per Host: {TPU_CORES_PER_HOST}")
        self.logger.info(f"SSH User: {SSH_USER}")
        self.logger.info(f"Head node: {head_ip}")
        self.logger.info(f"Worker nodes: {', '.join(worker_ips)}")

        self.ray_manager.stop_cluster(ips)
        self.logger.info("Waiting for processes to fully stop...")
        time.sleep(5)

        # Prepare head node resources
        head_resources = {
            "TPU": TPU_CORES_PER_HOST,
            f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE}-head": 1,
            f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
        }

        if not self.ray_manager.start_head_node(head_ip, head_resources):
            self.logger.error("Failed to start Ray head node. Exiting.")
            return False

        self.logger.info("Waiting for head node to initialize...")
        time.sleep(10)

        # Start worker nodes
        for i, worker_ip in enumerate(worker_ips):
            self.logger.info(f"Starting worker node at {worker_ip}")
            worker_resources = {
                "TPU": TPU_CORES_PER_HOST,
                f"TPU-{TPU_VERSION}-worker-{i}": 1,
                f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
            }
            self.ray_manager.start_worker_node(head_ip, worker_ip, worker_resources)

        self.logger.info("Ray cluster setup complete!")
        self.logger.info(f"Total expected TPU cores: {TPU_SLICE_SIZE}")
        self.logger.info("Waiting for workers to register resources...")
        time.sleep(15)

        self.ray_manager.verify_cluster(head_ip, TPU_SLICE_SIZE)

        self.logger.info("IMPORTANT: To use Ray in your applications, initialize with:")
        self.logger.info(f"ray.init(address='{head_ip}:6379')")

        return True

    def setup_self_job_node(self, args) -> int:
        """Set up a single node in self-job mode."""
        using_external = args.external or args.internal_ips is None
        local_ip = IPManager.get_external_ip() if using_external else IPManager.get_local_ip()
        external_head_ip = args.head_node_ip
        is_head_only = args.head_only

        self.logger.info(f"Running in self-job mode on {local_ip}")
        if external_head_ip:
            self.logger.info(f"Using external head node at: {external_head_ip}")
        if is_head_only:
            self.logger.info("Running as head-only node (no TPU resources)")
        self.logger.info(f"Using Ray command: {self.ray_manager.ray_path}")

        # Check if Ray is available
        try:
            result = subprocess.run([self.ray_manager.ray_path, "--version"], capture_output=True, check=True, text=True)
            self.logger.info(f"Ray version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error("Ray command not found or not working.")
            self.logger.error("Please install Ray with: pip install -U ray")
            self.logger.error("Make sure ray is in your PATH or in ~/.local/bin/")
            self.logger.error(f"Exec {e!s}")
            return 1

        # Stop any existing Ray processes
        self.logger.info("Stopping any existing Ray processes...")
        try:
            subprocess.run([self.ray_manager.ray_path, "stop"], stderr=subprocess.DEVNULL)
        except Exception:
            self.logger.warning("No Ray process was running or could not stop Ray")
        time.sleep(3)

        if args.stop:
            self.ray_manager.run_command(f"{self.ray_manager.ray_path} stop")
            return 0

        # Determine head IP
        is_external_head = external_head_ip and local_ip == external_head_ip
        if external_head_ip:
            head_ip = external_head_ip
        else:
            ips_to_use = EXTERNAL_IPS if using_external else INTERNAL_IPS
            if not ips_to_use:
                self.logger.error("No IPs provided")
                return 1
            head_ip = ips_to_use[0]

        # Check if current machine is the head
        is_head_node = is_external_head or (local_ip == head_ip and not external_head_ip)

        if is_head_node:
            self.logger.info(f"This machine ({local_ip}) is the HEAD node")

            if is_head_only:
                resources = {"head-node": 1, "control-plane": 1, "ray-cluster-head": 1}
                self.logger.info("Starting as head-only node (no TPU resources)")
            else:
                self.logger.info("Starting as head node with TPU resources")
                resources = {
                    "TPU": TPU_CORES_PER_HOST,
                    f"TPU-{TPU_VERSION}": TPU_CORES_PER_HOST,
                    f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE}-head": 1,
                    f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE * args.num_slices}-global-head": 1,
                    f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
                    "head-node": 1,
                    "ray-cluster-head": 1,
                }

            if not self.ray_manager.start_head_node(local_ip, resources):
                self.logger.error("Failed to start Ray head")
                return 1

            self.logger.info("Ray head started successfully!")
            self.logger.info(f"Dashboard available at: http://{local_ip}:8265")
            self.logger.info(f"Ray cluster address: {local_ip}:6379")

            if is_head_only:
                self.logger.info("Waiting for TPU workers to connect...")
                self.logger.info(f"Run the script on TPU nodes with --head-node-ip {local_ip}")
        else:
            self.logger.info(f"This machine ({local_ip}) is a WORKER node")
            self.logger.info(f"Will connect to head node at: {head_ip}")

            resources = {
                "TPU": TPU_CORES_PER_HOST,
                f"TPU-{TPU_VERSION}": TPU_CORES_PER_HOST,
                f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
            }

            if not self.ray_manager.start_worker_node(head_ip, local_ip, resources):
                self.logger.error("Failed to start Ray worker")
                return 1

            self.logger.info("Ray worker started successfully!")
            self.logger.info(f"Connected to head at: {head_ip}:6379")
            self.logger.info(f"Worker resources: {resources}")

        if is_head_node and args.verify:
            self.logger.info("Waiting for cluster to stabilize...")
            time.sleep(10)
            expected_tpu = 0 if is_head_only else TPU_SLICE_SIZE * args.num_slices
            if self.ray_manager.verify_cluster(head_ip, expected_tpu):
                self.logger.info("Cluster verification successful!")
            else:
                self.logger.error("Cluster verification failed!")
                return 1

        return 0


class ArgumentParser:
    """Handles command-line argument parsing."""

    @staticmethod
    def parse_arguments():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Ray TPU Cluster Setup for TPU Slices",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # General options
        general_group = parser.add_argument_group("General Options")
        general_group.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Set the logging level",
        )
        general_group.add_argument("--log-file", help="Path to log file (if not specified, logs to console only)")

        # Cluster configuration
        cluster_group = parser.add_argument_group("Cluster Configuration")
        cluster_group.add_argument("--external", action="store_true", help="Use external IPs instead of internal IPs")
        cluster_group.add_argument(
            "--tpu-version",
            default=TPU_VERSION,
            help=f"Set TPU version (default: {TPU_VERSION})",
        )
        cluster_group.add_argument(
            "--tpu-slice",
            type=int,
            default=TPU_SLICE_SIZE,
            help=f"Set TPU slice size (default: {TPU_SLICE_SIZE})",
        )
        cluster_group.add_argument(
            "--num-slices",
            type=int,
            default=1,
            help="Number of TPU slices to combine (default: 1)",
        )
        cluster_group.add_argument("--ssh-user", default=SSH_USER, help=f"SSH username to use (default: {SSH_USER})")

        # IP configuration
        ip_group = parser.add_argument_group("IP Configuration")
        ip_group.add_argument("--config", help="Path to YAML config file with IP addresses")
        ip_group.add_argument("--internal-ips", help="Comma-separated list of internal IPs for all slices")
        ip_group.add_argument("--external-ips", help="Comma-separated list of external IPs for all slices")
        ip_group.add_argument("--slice-config", help="Path to YAML config file with slice configurations")
        ip_group.add_argument("--head-node-ip", help="IP address of external head node (if not using first IP in list)")

        # Operation modes
        operation_group = parser.add_argument_group("Operation Modes")
        operation_group.add_argument("--stop", action="store_true", help="Stop the Ray cluster")
        operation_group.add_argument("--verify", action="store_true", help="Verify the Ray cluster setup")
        operation_group.add_argument("--test-ssh", action="store_true", help="Test SSH connectivity to all nodes")
        operation_group.add_argument("--self-job", action="store_true", help="Run only on the current machine")
        operation_group.add_argument(
            "--head-only",
            action="store_true",
            help="Run this node as head only (no TPU resources)",
        )

        return parser.parse_args()


def main():
    """Main entry point."""
    args = ArgumentParser.parse_arguments()

    logger = get_logger("RayPatcher")

    logger.debug(f"Command line arguments: {vars(args)}")

    # Update global variables from arguments
    global TPU_VERSION, TPU_SLICE_SIZE, SSH_USER, INTERNAL_IPS, EXTERNAL_IPS
    TPU_VERSION = args.tpu_version
    TPU_SLICE_SIZE = args.tpu_slice
    SSH_USER = args.ssh_user

    if args.internal_ips:
        INTERNAL_IPS = args.internal_ips.split(",")
        logger.info(f"Using internal IPs from command line: {INTERNAL_IPS}")

    if args.external_ips:
        EXTERNAL_IPS = args.external_ips.split(",")
        logger.info(f"Using external IPs from command line: {EXTERNAL_IPS}")

    if args.config:
        if not IPManager.read_ips_from_yaml(args.config):
            logger.error("Failed to read configuration from YAML file.")
            return 1

    # Initialize managers
    ray_manager = RayManager()
    cluster_manager = ClusterManager(ray_manager)

    # Handle different operation modes
    if args.self_job:
        return cluster_manager.setup_self_job_node(args)
    else:
        using_external = args.external or args.internal_ips is None
        ips = EXTERNAL_IPS if using_external else INTERNAL_IPS
        total_tpu_cores = TPU_SLICE_SIZE * args.num_slices

        if args.test_ssh:
            return 0 if IPManager.test_ssh_connectivity(ips) else 1

        if args.stop:
            ray_manager.stop_cluster(ips)
            return 0

        if args.verify:
            head_ip = ips[0] if ips else "0.0.0.0"
            return 0 if ray_manager.verify_cluster(head_ip, total_tpu_cores) else 1

        return 0 if cluster_manager.setup_cluster(using_external) else 1


if __name__ == "__main__":
    sys.exit(main())
