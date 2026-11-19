# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle-parallel Ray Data builds followed by shared atomic Lance append."""

from functools import partial
from typing import Any

from cosmos_curator.next.append.contracts import AppendBindings
from cosmos_curator.next.append.execution import ProcessingScope, run_append
from cosmos_curator.next.core.ray_runtime import (
    configure_ray_data_progress,
    configure_ray_data_stability,
    ensure_ray_initialized,
)
from cosmos_curator.next.recipes.video2scene_build.bindings import resolve_bindings
from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig
from cosmos_curator.next.recipes.video2scene_build.discovery import discover_items
from cosmos_curator.next.recipes.video2scene_build.processing import build_batch

BUILD_SLOT = "video2scene_build_slot"


def start_ray(bindings: AppendBindings, config: ResolvedVideo2SceneBuildConfig) -> None:
    """Local execution gets one build slot; managed workers advertise one each."""
    ensure_ray_initialized(local_resources={BUILD_SLOT: 1}, runtime_env_vars=bindings.storage_env_vars())
    configure_ray_data_progress(progress=config.execution.progress)
    configure_ray_data_stability(no_progress_timeout_s=-1)


def build_bundles(processing: ProcessingScope, config: ResolvedVideo2SceneBuildConfig) -> None:
    """Preserve a bundle's internal sequence while Ray parallelizes bundles across workers."""
    results = processing.read().map_batches(
        partial(build_batch, config=config),
        batch_format="pyarrow",
        batch_size=1,
        num_cpus=config.build.cpus,
        memory=int(config.execution.memory_gib * 1024**3),
        resources={BUILD_SLOT: 1},
    )
    processing.write(results)


def run_config(config: ResolvedVideo2SceneBuildConfig) -> dict[str, Any]:
    """Resume a frozen selection before discovering new bundles, then publish all outcomes."""
    outcome = run_append(
        resolve_bindings(config),
        partial(discover_items, config),
        start_ray=partial(start_ray, config=config),
        process=partial(build_bundles, config=config),
    )
    return {
        **outcome.summary(),
        "builds_lance_uri": config.output.results_lance_uri,
        "builds_lance_version": int(outcome.dataset.version),
    }
