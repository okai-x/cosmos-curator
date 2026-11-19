# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content identities and copies for immutable inputs and replay-safe payloads."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SOURCE_TREES = ("engine", "factory", "tools")
RUNTIME_PROBE = """
import hashlib, json, subprocess, sys
import numpy, cv2, PIL, jinja2, psutil
versions = {m.__name__: m.__version__ for m in (numpy, cv2, PIL, jinja2, psutil)}
versions['python'] = sys.version
versions['ffmpeg'] = subprocess.run(
    ['ffmpeg', '-version'], check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
).stdout
print(hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest())
"""


def tree_digest(root: Path) -> str:
    """Hash relative paths and streamed file contents; refuse links and special files."""
    if not root.is_dir() or root.is_symlink():
        msg = f"Expected an immutable directory: {root}"
        raise ValueError(msg)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            msg = f"Links and special files are not supported: {path}"
            raise ValueError(msg)
        if path.is_file():
            with path.open("rb") as stream:
                content = hashlib.file_digest(stream, "sha256").hexdigest()
            digest.update(json.dumps([relative.as_posix(), content], separators=(",", ":")).encode())
    return digest.hexdigest()


def source_digest(root: Path) -> str:
    """Fingerprint only the factory's executable source/config/template trees."""
    content = {name: tree_digest(root / name) for name in SOURCE_TREES}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    """Copy verified trees without caches; preserve links for the subsequent rejecting validation."""
    shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the source digest for build.source_digest.")
    parser.add_argument("source_root", type=Path, nargs="?")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    if args.runtime:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-c", RUNTIME_PROBE], check=True, stdin=subprocess.DEVNULL)  # noqa: S603 -- fixed runtime probe
    elif args.source_root is not None:
        print(source_digest(args.source_root))  # noqa: T201 -- this CLI prints the requested digest
    else:
        parser.error("provide SOURCE_ROOT or --runtime")
