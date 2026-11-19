# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Ray Data, real factory subprocesses and Lance recovery across cluster restarts."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest
import ray

from cosmos_curator.next.recipes.video2scene_build import pipeline, processing
from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig
from cosmos_curator.next.recipes.video2scene_build.discovery import discover_items
from tests.cosmos_curator.next.append import fresh_cluster
from tests.cosmos_curator.next.fresh_driver import run_in_fresh_driver
from tests.cosmos_curator.next.recipes.video2scene_build.conftest import RECIPE


def _build_without_external_tools(batch: pa.Table, *, config: ResolvedVideo2SceneBuildConfig) -> pa.Table:
    """Replace native tool preflight only; execute the test factory as a subprocess."""
    log = str(Path(config.output.root).parent / "build-calls.txt")
    guarded_path = str(Path(config.output.root).parent / "guard-bin") + os.pathsep + os.environ["PATH"]
    with (
        patch.object(processing, "check_runtime"),
        patch.dict("os.environ", {"TEST_BUILD_CALLS": log, "PATH": guarded_path}),
    ):
        return processing.build_batch(batch, config=config)


def _start_cluster(num_cpus: int) -> None:
    ray.init(
        address="local",
        num_cpus=num_cpus,
        num_gpus=0,
        resources={pipeline.BUILD_SLOT: 1},
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
        _memory=24 * 1024**3,
    )


def run_recovery(config_json: str) -> None:
    """Run the shared fresh-cluster scenario through this recipe's complete publication path."""
    config = ResolvedVideo2SceneBuildConfig.model_validate_json(config_json)
    items = {item["sample"]: item["build_id"] for item in discover_items(config).to_pylist()}
    calls = Path(config.output.root).parent / "build-calls.txt"
    marker = calls.with_name("docker-called")
    guard = calls.parent / "guard-bin"
    guard.mkdir()
    docker = guard / "docker"
    docker.write_text(f"#!/bin/sh\nprintf called > '{marker}'\nexit 97\n")
    docker.chmod(0o755)

    def work_logs() -> dict[str, list[str]]:
        return {"factory": [items[name] for name in calls.read_text().splitlines()]}

    with (
        patch.object(pipeline, "build_batch", _build_without_external_tools),
        patch.object(fresh_cluster, "start_local_cluster", _start_cluster),
    ):
        summary = fresh_cluster.assert_fresh_cluster_resume(
            RECIPE,
            config,
            process=pipeline.build_bundles,
            work_logs=work_logs,
            num_cpus=8,
        )
    table = lance.dataset(config.output.results_lance_uri).to_table()
    assert {row["status"] for row in table.to_pylist()} == {"completed", "returned"}
    assert summary["errors"] == 1
    assert not marker.exists()
    for row in table.to_pylist():
        manifest = json.loads(Path(row["manifest_uri"]).read_text())
        assert manifest["result"]["build_id"] == row["build_id"]
        for task in row["tasks"]:
            assert Path(task["task_uri"], "task.toml").is_file()


@pytest.mark.integration
def test_fresh_cluster_resumes_without_rebuilding_checkpointed_bundles(config: ResolvedVideo2SceneBuildConfig) -> None:
    """Kill the first cluster after a durable item, then finish and publish exactly once."""
    run_in_fresh_driver(f"{__name__}:run_recovery", config.model_dump_json(), timeout=300)
