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

import glob
import os

import braceexpand
import fsspec
from fsspec.asyn import AsyncFileSystem


def exists(url, **kwargs) -> bool:
    """Check if a file or directory exists at the given URL.

    Args:
        url: URL or path to check. Supports various protocols (file://, gs://, s3://, etc.).
        **kwargs: Additional arguments passed to fsspec.core.url_to_fs.

    Returns:
        True if the path exists, False otherwise.
    """
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    return fs.exists(path)


def mkdirs(path):
    """Create a directory and all necessary parent directories.

    Args:
        path: Path or URL of the directory to create.

    Note:
        Uses exist_ok=True, so no error is raised if the directory already exists.
    """
    fs, path = fsspec.core.url_to_fs(path)
    fs.makedirs(path, exist_ok=True)


def expand_glob(url):
    """Expand glob patterns and brace expressions in URLs.

    Supports both brace expansion (e.g., file_{1..3}.txt) and glob patterns
    (e.g., *.txt). Works with various filesystem protocols.

    Args:
        url: URL or path with potential glob patterns and/or brace expressions.

    Yields:
        Expanded URLs/paths matching the pattern.

    Example:
        >>> list(expand_glob("data/*.json"))
        ['data/file1.json', 'data/file2.json']
        >>> list(expand_glob("file_{1..3}.txt"))
        ['file_1.txt', 'file_2.txt', 'file_3.txt']
    """
    for candidate in braceexpand.braceexpand(url):
        fs, path = fsspec.core.url_to_fs(candidate)

        if glob.has_magic(path):
            proto = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
            for p in fs.glob(path):
                yield f"{proto}://{p}" if proto else p
        else:
            yield candidate


def remove(url, *, recursive=False, **kwargs):
    """Remove a file or directory.

    Args:
        url: URL or path of the file/directory to remove.
        recursive: If True, remove directories and their contents recursively.
        **kwargs: Additional arguments passed to fsspec.core.url_to_fs.
    """
    fs, path = fsspec.core.url_to_fs(url, **kwargs)

    fs.rm(path, recursive=recursive)


async def async_remove(url, *, recursive=False, **kwargs):
    """Asynchronously remove a file or directory.

    Uses async operations when the filesystem supports it, otherwise falls back
    to synchronous removal.

    Args:
        url: URL or path of the file/directory to remove.
        recursive: If True, remove directories and their contents recursively.
        **kwargs: Additional arguments passed to fsspec.core.url_to_fs.

    Returns:
        None if successful, or result from async filesystem operation.
    """
    fs, path = fsspec.core.url_to_fs(url, **kwargs)

    if isinstance(fs, AsyncFileSystem):
        return await fs._rm(path, recursive=recursive)
    else:
        fs.rm(path, recursive=recursive)


def join_path(lhs, rhs):
    """Join two path components intelligently, handling protocols.

    Similar to os.path.join but handles URLs with protocols (gs://, s3://, etc.).
    If rhs has a protocol, it's returned as-is (absolute path behavior).

    Args:
        lhs: Left-hand side path or URL.
        rhs: Right-hand side path or URL.

    Returns:
        Joined path, preserving protocols appropriately.

    Raises:
        ValueError: If both paths have different protocols.

    Example:
        >>> join_path("gs://bucket/dir", "file.txt")
        'gs://bucket/dir/file.txt'
        >>> join_path("/local/dir", "gs://bucket/file.txt")
        'gs://bucket/file.txt'
    """
    lhs_protocol, lhs_rest = fsspec.core.split_protocol(lhs)
    rhs_protocol, rhs_rest = fsspec.core.split_protocol(rhs)

    if rhs_protocol is not None and lhs_protocol is not None and lhs_protocol != rhs_protocol:
        raise ValueError(f"Cannot join paths with different protocols: {lhs} and {rhs}")

    if rhs_protocol is not None:
        return rhs
    else:
        return os.path.join(lhs, rhs)
