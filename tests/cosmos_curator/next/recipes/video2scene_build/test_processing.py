# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real subprocess builds, payload replay, and failure classification."""

import json
from pathlib import Path

import pytest

from cosmos_curator.next.recipes.video2scene_build import processing
from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig, resolve_config_data
from cosmos_curator.next.recipes.video2scene_build.discovery import discover_items


@pytest.fixture(autouse=True)
def tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only external-tool prerequisites are replaced; factory execution is a real subprocess."""
    monkeypatch.setattr(processing, "check_runtime", lambda *_args: None)
    monkeypatch.setenv("V2S_TRUTH_SAMPLE", "/ambient/wrong-sample.json")


def test_complete_payload_is_reused_and_tampering_is_rejected(
    config: ResolvedVideo2SceneBuildConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between payload publication and checkpointing must not repeat a completed build."""
    calls = tmp_path / "calls"
    monkeypatch.setenv("TEST_BUILD_CALLS", str(calls))
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "released")
    first = processing.build_sample(item, config)
    assert first["tasks"][0]["state"] == "released"
    task = Path(first["tasks"][0]["task_uri"])
    assert (task / "task.toml").is_file()
    assert processing.build_sample(item, config) == first
    assert calls.read_text().splitlines() == ["released"]
    (task / "task.toml").write_text("corrupt")
    with pytest.raises(ValueError, match="payload changed"):
        processing.build_sample(item, config)


def test_environment_failure_stays_pending_and_retries_in_a_fresh_directory(
    config: ResolvedVideo2SceneBuildConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a zero-exit factory must not checkpoint a scorer infrastructure failure."""
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "held")
    monkeypatch.setenv("TEST_BUILD_FAILURE", "1")
    with pytest.raises(RuntimeError, match="Scoring did not complete"):
        processing.build_sample(item, config)
    final = Path(config.output.payload_root) / item["build_id"]
    assert not final.exists()
    monkeypatch.delenv("TEST_BUILD_FAILURE")
    result = processing.build_sample(item, config)
    assert result["status"] == "completed"
    assert result["tasks"][0]["state"] == "held"


def test_intake_return_is_a_durable_outcome(config: ResolvedVideo2SceneBuildConfig) -> None:
    """Explicit input rejection is data, preserving its report without inventing a Harbor task."""
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "returned")
    result = processing.build_sample(item, config)
    assert (result["status"], result["error_stage"], result["tasks"]) == ("returned", "intake", [])
    assert Path(result["manifest_uri"]).is_file()


def test_full_profile_absence_is_recorded(
    config: ResolvedVideo2SceneBuildConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed rules build may explicitly omit the full profile."""
    config = resolve_config_data(config.model_dump(mode="json"), overrides=["build.judge=true"])
    monkeypatch.setenv("V2S_VLM_KEY", "test-only-not-a-credential")
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "released")
    result = processing.build_sample(item, config)
    assert result["tasks"][1]["state"] == "not_generated"
    assert result["tasks"][1]["task_uri"] is None


def test_changed_input_is_not_processed_under_old_identity(config: ResolvedVideo2SceneBuildConfig) -> None:
    """A frozen selection cannot silently build replacement input bytes."""
    item = discover_items(config).to_pylist()[0]
    (Path(item["bundle_uri"]) / "bundle.json").write_text(json.dumps({"sample": "changed"}))
    with pytest.raises(ValueError, match="content changed"):
        processing.build_sample(item, config)


@pytest.mark.parametrize("failure", ["TEST_JUDGE_FAILURE", "TEST_TIMING_FAILURE"])
def test_missing_rewards_are_retryable_even_without_error_json(
    config: ResolvedVideo2SceneBuildConfig, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A verifier failing to start must not be published as an ordinary quality rejection."""
    config = resolve_config_data(config.model_dump(mode="json"), overrides=["build.judge=true"])
    monkeypatch.setenv("V2S_VLM_KEY", "test-only-not-a-credential")
    monkeypatch.setenv(failure, "1")
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "released")
    with pytest.raises(RuntimeError, match=r"Scoring did not complete|Rules timing failed"):
        processing.build_sample(item, config)
    assert not (Path(config.output.payload_root) / item["build_id"]).exists()


def test_model_zero_in_full_timing_is_a_business_outcome(
    config: ResolvedVideo2SceneBuildConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate-refused reference can return zero before reaching the intentional missing-key stop."""
    config = resolve_config_data(config.model_dump(mode="json"), overrides=["build.judge=true"])
    monkeypatch.setenv("V2S_VLM_KEY", "test-only-not-a-credential")
    monkeypatch.setenv("TEST_FULL_MODEL_ZERO", "1")
    item = next(row for row in discover_items(config).to_pylist() if row["sample"] == "held")
    result = processing.build_sample(item, config)
    assert result["status"] == "completed"
    assert result["tasks"][0]["state"] == "held"
