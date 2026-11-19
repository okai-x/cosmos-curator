# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tiny external factory executable using the real subprocess artifact protocol."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video2scene_build import pipeline
from cosmos_curator.next.recipes.video2scene_build.bindings import resolve_bindings
from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig, resolve_config_data
from cosmos_curator.next.recipes.video2scene_build.contracts import RESULT_SCHEMA
from cosmos_curator.next.recipes.video2scene_build.discovery import discover_items
from cosmos_curator.next.recipes.video2scene_build.snapshot import source_digest
from tests.cosmos_curator.next.append.conformance import RecipeUnderTest

FACTORY = """
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('bundle')
p.add_argument('--out')
p.add_argument('--cpus')
p.add_argument('--parallel')
p.add_argument('--backend', choices=['local'], required=True)
p.add_argument('--judge', action='store_true')
p.add_argument('--max-usd')
a = p.parse_args()
assert a.parallel == '1'
assert os.environ['BLENDER'] == os.environ['V2S_BLENDER']
assert os.environ['PYTHONHASHSEED'] == '0'
assert 'V2S_TRUTH_SAMPLE' not in os.environ
out = Path(a.out)
assert not out.exists(), 'An attempt reused partial factory state'
out.mkdir(parents=True)
meta = json.loads((Path(a.bundle) / 'bundle.json').read_text())
steps = {'task': {'status': 'done'}}
if meta['sample'] == 'returned':
    steps = {'intake': {'status': 'failed'}}
    (out / 'intake.json').write_text(json.dumps({'count': {'fail': 1}, 'verdict': 'rejected'}))
else:
    task = out / 'task_rules'
    task.mkdir()
    (task / 'scorecard.json').write_text(json.dumps({'state': meta.get('state', 'released')}))
    (task / 'task.toml').write_text('version = "1.0"')
    acceptance = out / 'acceptance'
    acceptance.mkdir()
    row = {'result': 'environment_error' if os.environ.get('TEST_BUILD_FAILURE') else 'scored'}
    (acceptance / 'acceptance.json').write_text(json.dumps({'fixtures': {'reference': row}}))
    timing = acceptance / 'timing' / 'rules'
    timing.mkdir(parents=True)
    timed = {'exit': 0, 'reward': {'reward': 1}, 'error': None}
    if os.environ.get('TEST_TIMING_FAILURE'):
        timed = {'exit': 125, 'reward': None, 'error': None}
    (timing / 'timing.json').write_text(json.dumps(timed))
    if a.judge:
        full_timing = acceptance / 'timing' / 'full_without_judge'
        full_timing.mkdir()
        expected_error = 'no key for the judge: set V2S_VLM_KEY_FILE (or V2S_VLM_KEY)'
        if os.environ.get('TEST_FULL_MODEL_ZERO'):
            (full_timing / 'timing.json').write_text(json.dumps({'exit': 0, 'reward': {'G': 0}, 'error': None}))
        else:
            (full_timing / 'error.json').write_text(json.dumps({'error': expected_error}))
            (full_timing / 'timing.json').write_text(json.dumps({'exit': 1, 'error': expected_error}))
        judge_row = {'result': 'no reward' if os.environ.get('TEST_JUDGE_FAILURE') else 'scored'}
        (acceptance / 'judge.json').write_text(json.dumps({'runs': {'reference': judge_row}}))
if os.environ.get('TEST_BUILD_CALLS'):
    with open(os.environ['TEST_BUILD_CALLS'], 'a') as log:
        log.write(meta['sample'] + '\\n')
(out / 'build.json').write_text(json.dumps({'steps': steps}))
raise SystemExit(1 if meta['sample'] == 'returned' else 0)
"""


@pytest.fixture
def config(tmp_path: Path) -> ResolvedVideo2SceneBuildConfig:
    """Use real files and a factory that can complete, hold, or return a bundle."""
    source = tmp_path / "source"
    for name in ("factory", "engine", "tools"):
        (source / name).mkdir(parents=True)
    (source / "factory" / "build.py").write_text(FACTORY)
    bundles = []
    for sample, state in (("released", "released"), ("held", "held"), ("returned", "returned")):
        bundle = tmp_path / "bundles" / sample
        bundle.mkdir(parents=True)
        (bundle / "bundle.json").write_text(json.dumps({"sample": sample, "state": state}))
        bundles.append(str(bundle))
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video2scene-build",
            "input": {"bundles": bundles},
            "build": {
                "source_root": str(source),
                "source_digest": source_digest(source),
                "python_runtime_digest": "0" * 64,
                "python_executable": sys.executable,
                "blender_executable": sys.executable,
                "measurement_class": "test-worker",
            },
            "output": {"root": str(tmp_path / "output")},
            "execution": {"rows_per_checkpoint": 1},
        }
    )


def expected_result(item: dict, config: ResolvedVideo2SceneBuildConfig) -> dict:
    """Canonical synthetic outcomes for publication-only conformance tests."""
    payload = Path(config.output.payload_root) / item["build_id"]
    returned = item["sample"] == "returned"
    return {
        **item,
        "status": "returned" if returned else "completed",
        "manifest_uri": str(payload / "manifest.json"),
        "error_stage": "intake" if returned else None,
        "error_message": "rejected" if returned else None,
        "tasks": []
        if returned
        else [
            {
                "profile": "rules",
                "state": item["sample"],
                "task_uri": str(payload / "workspace/work_factory/build/task_rules"),
                "scorecard_json": json.dumps({"state": item["sample"]}),
                "reason": "null",
            }
        ],
    }


@pytest.fixture
def results(config: ResolvedVideo2SceneBuildConfig) -> pa.Table:
    """Include successful, held, and explicit terminal-error results."""
    return pa.Table.from_pylist(
        [expected_result(item, config) for item in discover_items(config).to_pylist()], schema=RESULT_SCHEMA
    )


def retune(config: ResolvedVideo2SceneBuildConfig) -> ResolvedVideo2SceneBuildConfig:
    """Change checkpoint packing without changing measured outputs."""
    return resolve_config_data(config.model_dump(mode="json"), overrides=["execution.rows_per_fragment=2"])


def recontract(config: ResolvedVideo2SceneBuildConfig) -> ResolvedVideo2SceneBuildConfig:
    """Change measurement class to require a new output table."""
    return resolve_config_data(config.model_dump(mode="json"), overrides=["build.measurement_class=other-worker"])


RECIPE = RecipeUnderTest(
    pipeline=pipeline, resolve_bindings=resolve_bindings, discover=discover_items, retune=retune, recontract=recontract
)


@pytest.fixture
def recipe() -> RecipeUnderTest:
    """Exercise shared publication through this recipe's real bindings."""
    return RECIPE
