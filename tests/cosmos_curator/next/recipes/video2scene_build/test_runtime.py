# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime admission checks before any expensive factory work."""

import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cosmos_curator.next.recipes.video2scene_build.config import (
    ResolvedVideo2SceneBuildConfig,
    config_template,
    resolve_config_data,
)
from cosmos_curator.next.recipes.video2scene_build.discovery import discover_items
from cosmos_curator.next.recipes.video2scene_build.processing import build_sample, check_runtime
from cosmos_curator.next.recipes.video2scene_build.snapshot import RUNTIME_PROBE


@pytest.mark.parametrize("mismatch", ["blender", "python", "none"])
def test_runtime_requires_the_pinned_tools(mismatch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject mismatched Blender/worker runtime before invoking the factory without checking Docker."""
    build = resolve_config_data(config_template()).build
    replies = [
        subprocess.CompletedProcess([], 0, stdout="Blender 0.0.0" if mismatch == "blender" else "Blender 5.1.2\n"),
        subprocess.CompletedProcess([], 0, stdout="wrong" if mismatch == "python" else build.python_runtime_digest),
    ]
    run = Mock(side_effect=replies)
    monkeypatch.setattr(subprocess, "run", run)
    if mismatch == "none":
        check_runtime(build, {"BLENDER": "/opt/blender/blender"})
        assert run.call_count == 2
    else:
        with pytest.raises(RuntimeError, match=r"Blender does not match|Python runtime does not match"):
            check_runtime(build, {"BLENDER": "/opt/blender/blender"})


def test_native_build_runs_with_docker_disabled(
    config: ResolvedVideo2SceneBuildConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use real preflight and a real factory subprocess with a Docker executable that fails every call."""
    tools = tmp_path / "bin"
    tools.mkdir()
    marker = tmp_path / "docker-called"
    docker = tools / "docker"
    docker.write_text(f"#!/bin/sh\nprintf called > '{marker}'\nexit 97\n")
    docker.chmod(0o755)
    blender = tools / "blender"
    blender.write_text("#!/bin/sh\nprintf 'Blender 5.1.2\\n'\n")
    blender.chmod(0o755)
    ffmpeg = tools / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nprintf 'FFmpeg test fixture\\n'\n")
    ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    runtime = subprocess.run(  # noqa: S603 -- trusted test interpreter and fixed runtime probe
        [config.build.python_executable, "-c", RUNTIME_PROBE],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    data = config.model_dump(mode="json")
    data["build"].update(blender_executable=str(blender), python_runtime_digest=runtime)
    config = resolve_config_data(data)
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "released")
    assert build_sample(item, config)["tasks"][0]["state"] == "released"
    assert not marker.exists()


def test_runtime_identity_includes_ffmpeg(
    config: ResolvedVideo2SceneBuildConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native encoding changes must invalidate the build identity, even when Python packages stay fixed."""
    ffmpeg = tmp_path / "ffmpeg"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    digests = []
    for version in ("worker-ffmpeg-A", "worker-ffmpeg-B"):
        ffmpeg.write_text(f"#!/bin/sh\nprintf '{version}\\n'\n")
        ffmpeg.chmod(0o755)
        digests.append(
            subprocess.run(  # noqa: S603 -- trusted test interpreter and fixed runtime probe
                [config.build.python_executable, "-c", RUNTIME_PROBE],
                capture_output=True,
                text=True,
                check=True,
                stdin=subprocess.DEVNULL,
            ).stdout.strip()
        )
    assert digests[0] != digests[1]
