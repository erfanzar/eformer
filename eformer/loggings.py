# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import datetime
import logging
import os
import sys
import threading
import time
import typing as tp
from functools import wraps

import jax

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

        try:
            if jax.process_index() > 0:
                self._level = logging.WARNING
        except RuntimeError:
            pass

        logger = logging.getLogger(self._name)
        logger.propagate = False

        # Set the logging level
        logger.setLevel(self._level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)

        # Use our custom color formatter
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
    """
    Function to create a lazy logger that only initializes when first used.

    Args:
        name (str): The name of the logger.
        level (Optional[int]): The logging level. Defaults to environment variable LOGGING_LEVEL_ED or "INFO".

    Returns:
        LazyLogger: A lazy logger instance that initializes on first use.
    """
    return LazyLogger(name, level)


class ProgressLogger:
    """A progress logger that displays updating progress bars and messages.

    This class provides a clean way to show progress for long-running operations
    with support for progress bars, ETAs, and streaming updates that overwrite
    the same line in the terminal.

    Attributes:
        name: Logger name to use for fallback logging
        use_tty: Whether to use TTY features (auto-detected)
        start_time: Start time of the progress operation
        _logger: Underlying logger for fallback

    Example:
        >>> progress = ProgressLogger("Training")
        >>> for i in range(100):
        ...     progress.update(i, 100, f"Processing batch {i}")
        ...     # Do work
        >>> progress.complete("Training finished!")
    """

    def __init__(self, name: str = "Progress", logger_instance: LazyLogger | None = None):
        """Initialize the progress logger.

        Args:
            name: Name to display in progress messages
            logger_instance: Optional logger instance to use for fallback
        """
        self.name = name
        self.use_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self.start_time = time.time()
        self._logger = logger_instance or get_logger(name)
        self._last_message_length = 0

    def update(
        self,
        current: int,
        total: int,
        message: str = "",
        bar_width: int = 20,
        show_eta: bool = True,
        extra_info: str = "",
    ) -> None:
        """Update the progress display.

        Args:
            current: Current progress value (0-based)
            total: Total number of items
            message: Message to display after the progress bar
            bar_width: Width of the progress bar in characters
            show_eta: Whether to show estimated time remaining
            extra_info: Additional info to append at the end
        """
        if total <= 0:
            return

        # Calculate progress
        progress = min(current / total, 1.0)
        progress_pct = progress * 100

        # Create progress bar
        filled = int(bar_width * progress)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        # Calculate ETA
        eta_str = ""
        if show_eta and current > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / current
            remaining = (total - current) * avg_time
            if remaining > 0:
                if remaining < 60:
                    eta_str = f" ETA: {remaining:.1f}s"
                elif remaining < 3600:
                    eta_str = f" ETA: {remaining / 60:.1f}m"
                else:
                    eta_str = f" ETA: {remaining / 3600:.1f}h"

        # Format the complete message
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"({timestamp} {self.name}) [{bar}] {progress_pct:5.1f}% {message}{eta_str}"
        if extra_info:
            full_message += f" {extra_info}"

        if self.use_tty:
            # Clear previous line and write new progress
            sys.stdout.write("\r" + " " * self._last_message_length + "\r")
            sys.stdout.write(full_message)
            sys.stdout.flush()
            self._last_message_length = len(full_message)
        else:
            # Fall back to regular logging
            self._logger.info(f"{progress_pct:.1f}% - {message}")

    def update_simple(self, message: str) -> None:
        """Update with a simple message without progress bar.

        Args:
            message: Message to display
        """
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"({timestamp} {self.name}) {message}"

        if self.use_tty:
            sys.stdout.write("\r" + " " * self._last_message_length + "\r")
            sys.stdout.write(full_message)
            sys.stdout.flush()
            self._last_message_length = len(full_message)
        else:
            self._logger.info(message)

    def complete(self, message: str | None = None, show_time: bool = True) -> None:
        """Complete the progress and show final message.

        Args:
            message: Optional completion message
            show_time: Whether to show total elapsed time
        """
        if message is None:
            message = "Completed"

        total_time = time.time() - self.start_time
        timestamp = time.strftime("%H:%M:%S")

        if show_time:
            time_str = ""
            if total_time < 60:
                time_str = f" in {total_time:.1f}s"
            elif total_time < 3600:
                time_str = f" in {total_time / 60:.1f}m"
            else:
                time_str = f" in {total_time / 3600:.1f}h"
            full_message = f"({timestamp} {self.name}) {message}{time_str}"
        else:
            full_message = f"({timestamp} {self.name}) {message}"

        if self.use_tty:
            sys.stdout.write("\r" + " " * self._last_message_length + "\r")
            sys.stdout.write(full_message + "\n")
            sys.stdout.flush()
        else:
            self._logger.info(full_message)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - complete the progress."""
        if exc_type is None:
            self.complete()
        return False


logger = get_logger("eformerLoggings")


def create_step_profiler(
    profile_path: str,
    start_step: int,
    duration_steps: int,
    enable_perfetto: bool,
) -> tp.Callable[[int], None]:
    """
    Creates a step-aware profiler that activates during a specific training window.

    Args:
        profile_path: Directory to store profiling results
        start_step: Step number to begin profiling (inclusive)
        duration_steps: How many steps to profile
        enable_perfetto: Whether to generate Perfetto UI links

    Returns:
        A callback function for training step profiling
    """
    from eformer.escale import barrier_sync

    class ProfilerState:
        def __init__(self):
            self.active = False
            self.completed = False

    state = ProfilerState()

    def profile_step(step) -> None:
        """Handles profiling lifecycle based on current step."""
        if state.completed:
            return

        if step == start_step - 1 and not state.active:
            logger.info(f"Activating profiler for steps {start_step}-{start_step + duration_steps - 1}")
            ignite_profiler(profile_path, enable_perfetto)
            state.active = True

        elif step == start_step + duration_steps - 1 and state.active:
            logger.info("Deactivating profiler")
            extinguish_profiler(enable_perfetto)
            barrier_sync()
            state.completed = True

    return profile_step


def ignite_profiler(profile_path: str, enable_perfetto: bool = False) -> None:
    """
    Ignites the JAX profiler with optional Perfetto integration.

    Args:
        profile_path: Directory to store profiling results
        enable_perfetto: Whether to generate Perfetto UI links (only on primary process)
    """
    should_enable_perfetto = enable_perfetto and jax.process_index() == 0
    jax.profiler.start_trace(profile_path, create_perfetto_link=should_enable_perfetto, create_perfetto_trace=True)


def extinguish_profiler(enable_perfetto: bool) -> None:
    """
    Safely stops the profiler and handles Perfetto link generation.

    Args:
        enable_perfetto: Whether Perfetto links were enabled
    """
    completion_signal = threading.Event()
    if enable_perfetto and jax.process_index() == 0:
        _pulse_output_during_wait(completion_signal)

    jax.profiler.stop_trace()

    if enable_perfetto and jax.process_index() == 0:
        completion_signal.set()


def _pulse_output_during_wait(completion_signal: threading.Event) -> None:
    """
    Keeps output streams alive during blocking profiler shutdown.

    Args:
        completion_signal: Event signaling when to stop pulsing
    """

    def pulse_output() -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)

        while not completion_signal.is_set():
            print("Profiler finalizing...", flush=True)
            print("\n", file=sys.stderr, flush=True)
            time.sleep(5)

    thread = threading.Thread(target=pulse_output, daemon=True)
    thread.start()
