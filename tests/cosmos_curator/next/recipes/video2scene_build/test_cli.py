# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic CLI config tooling and runtime dispatch for the new recipe."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cosmos_curator.next.recipes.video2scene_build import pipeline
from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig, config_template


def test_cli_template_schema_and_runtime_dispatch(
    config: ResolvedVideo2SceneBuildConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose the recipe through the same CLI and runtime entrypoint as other Next recipes."""
    from cosmos_curator.client.cli import cosmos_curator  # noqa: PLC0415 -- dev-only deployment CLI
    from cosmos_curator.pipelines import pipeline_runtime  # noqa: PLC0415

    runner = CliRunner()
    template = runner.invoke(cosmos_curator, ["pipeline", "template", "video2scene-build"])
    assert template.exit_code == 0, template.output
    assert yaml.safe_load(template.stdout) == config_template()
    schema = runner.invoke(cosmos_curator, ["pipeline", "schema", "video2scene-build", "--json"])
    assert schema.exit_code == 0, schema.output
    assert json.loads(schema.stdout)["properties"]["kind"]["const"] == "video2scene-build"
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    kind = pipeline_runtime.BUILTIN_PIPELINE_KINDS.get("video2scene-build")
    assert kind.validate(path, ()) == {"ok": True}
    assert json.loads(kind.render(path, ())) == config.model_dump(mode="json")
    seen = []

    def run(resolved: ResolvedVideo2SceneBuildConfig) -> dict[str, int]:
        seen.append(resolved)
        return {"selected_items": 3, "appended_rows": 3, "remaining_items": 0, "errors": 1}

    monkeypatch.setattr(pipeline, "run_config", run)
    pipeline_runtime.main(path)
    captured = capsys.readouterr()
    assert "video2scene-build: published 3 item(s) as 3 row(s) with 1 terminal error(s)" in captured.out
    assert not captured.err
    assert seen == [config]
