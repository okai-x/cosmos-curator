# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline configuration and result identity for environment builds."""

import pytest
from pydantic import ValidationError

from cosmos_curator.next.recipes.video2scene_build.config import config_template, resolve_config_data


def test_template_is_resolved_without_runtime_or_input_access() -> None:
    """Config tools need neither a checkout, Docker, nor an input bundle."""
    config = resolve_config_data(config_template())
    assert config.kind == "video2scene-build"
    assert config.output.results_lance_uri.endswith("/builds.lance")
    assert config.execution.rows_per_checkpoint == 1
    assert config.build.judge is False
    assert "scorer_image" not in config.build.model_dump()
    assert config.build.result_contract["execution_backend"] == "local"
    assert config.build.result_contract["adapter_version"] == 2
    assert config.build.result_contract["memory_metric"] == "sampled_process_tree_rss"
    assert config.build.result_contract["memory_sample_interval_s"] == 0.05
    tuned = resolve_config_data(config_template(), overrides=["execution.rows_per_checkpoint=2"])
    assert config.build.result_contract == tuned.build.result_contract


@pytest.mark.parametrize(
    "override",
    [
        "build.scorer_image=v2s-scorer:latest",
        "build.source_digest=unversioned",
        "build.cpus=1",
        "input.bundles=[]",
        "output.root=s3://bucket/builds",
        "execution.typo=true",
        "kind=video2scene_build",
    ],
)
def test_invalid_settings_fail_before_work(override: str) -> None:
    """Obsolete image configuration, malformed identities and unsupported storage are rejected."""
    with pytest.raises(ValidationError):
        resolve_config_data(config_template(), overrides=[override])
