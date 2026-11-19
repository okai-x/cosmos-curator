# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-safe subprocess adapter; the factory remains the owner of scoring semantics."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa

from cosmos_curator.next.recipes.video2scene_build.config import BuildConfig, ResolvedVideo2SceneBuildConfig
from cosmos_curator.next.recipes.video2scene_build.contracts import RESULT_SCHEMA, validate_results
from cosmos_curator.next.recipes.video2scene_build.snapshot import (
    RUNTIME_PROBE,
    SOURCE_TREES,
    copy_tree,
    source_digest,
    tree_digest,
)


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Expected a JSON object: {path}"
        raise TypeError(msg)
    return data


def build_environment(build: BuildConfig) -> dict[str, str]:
    """Remove ambient factory overrides and pin both Blender discovery paths."""
    allowed = {"V2S_VLM_KEY", "V2S_VLM_KEY_FILE"} if build.judge else set()
    env = {key: value for key, value in os.environ.items() if not key.startswith("V2S_") or key in allowed}
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    blender = shutil.which(build.blender_executable)
    if blender is None:
        msg = f"Blender executable is unavailable: {build.blender_executable}"
        raise FileNotFoundError(msg)
    env.update(
        V2S_BLENDER=blender,
        BLENDER=blender,
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS=str(build.cpus),
    )
    if build.judge and not any(env.get(key) for key in allowed):
        msg = "Judge builds require V2S_VLM_KEY or V2S_VLM_KEY_FILE on the worker"
        raise RuntimeError(msg)
    return env


def check_runtime(build: BuildConfig, env: dict[str, str]) -> None:
    """Fail before processing when the selected tools do not match the result contract."""
    version = subprocess.run(  # noqa: S603 -- configured trusted executable, argument vector without a shell
        [env["BLENDER"], "--version"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=60,
    ).stdout
    if re.search(r"^Blender " + re.escape(build.blender_version) + r"(?:\s|$)", version, re.MULTILINE) is None:
        msg = f"Blender does not match configured release {build.blender_version}"
        raise RuntimeError(msg)
    runtime = subprocess.run(  # noqa: S603 -- configured interpreter, fixed probe
        [build.python_executable, "-c", RUNTIME_PROBE],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=60,
    ).stdout.strip()
    if runtime != build.python_runtime_digest:
        msg = "Python runtime does not match build.python_runtime_digest"
        raise RuntimeError(msg)


def _check_scoring(out: Path, *, judge: bool) -> None:
    """Separate failed measurements from genuine quality failures, including AC-14's intentional stop."""
    acceptance = _read(out / "acceptance" / "acceptance.json")
    rows = [*acceptance.get("fixtures", {}).values(), *acceptance.get("second_scoring", {}).values()]
    if judge:
        judged = _read(out / "acceptance" / "judge.json")
        rows.extend(judged["runs"].values())
    if (
        not rows
        or acceptance.get("made_not_scored")
        or any(row.get("result") in {"environment_error", "no reward"} for row in rows)
    ):
        msg = f"Scoring did not complete; retry with a fresh workspace: {out}"
        raise RuntimeError(msg)
    rules_timing = _read(out / "acceptance" / "timing" / "rules" / "timing.json")
    if rules_timing.get("exit") != 0 or rules_timing.get("reward") is None or rules_timing.get("error"):
        msg = f"Rules timing failed; inspect {out}"
        raise RuntimeError(msg)
    intentional = out / "acceptance" / "timing" / "full_without_judge" / "error.json"
    if judge:
        full_timing = _read(intentional.with_name("timing.json"))
        scored = full_timing.get("exit") == 0 and full_timing.get("reward") is not None and not full_timing.get("error")
        reached_judge = full_timing.get("exit") == 1 and "no key for the judge:" in (full_timing.get("error") or "")
        if not (scored or reached_judge):
            msg = f"Full timing did not reach the intentional missing-key stop; inspect {out}"
            raise RuntimeError(msg)
    errors = [path for path in out.rglob("error.json") if path != intentional]
    if errors:
        msg = f"Verifier environment error; inspect {errors[0]}"
        raise RuntimeError(msg)


def _outcome(item: dict[str, Any], root: Path, final: Path, *, judge: bool, code: int) -> dict[str, Any]:
    out = root / "work_factory" / "build"
    state = _read(out / "build.json")
    steps = state["steps"]
    failed = [name for name, step in steps.items() if step.get("status") == "failed"]
    result = {
        **item,
        "status": "completed",
        "manifest_uri": str(final / "manifest.json"),
        "tasks": [],
        "error_stage": None,
        "error_message": None,
    }
    if code and failed == ["intake"]:
        intake = _read(out / "intake.json")
        if intake.get("count", {}).get("fail", 0) > 0:
            return result | {"status": "returned", "error_stage": "intake", "error_message": str(intake["verdict"])}
    if code or failed or steps.get("task", {}).get("status") != "done":
        msg = f"Build failed (exit {code}, steps {failed}); inspect {root.parent / 'build.log'}"
        raise RuntimeError(msg)
    _check_scoring(out, judge=judge)
    tasks = []
    for profile in ("rules", "full") if judge else ("rules",):
        task = out / f"task_{profile}"
        if not task.is_dir() and profile == "full":
            tasks.append(
                {
                    "profile": profile,
                    "state": "not_generated",
                    "task_uri": None,
                    "scorecard_json": None,
                    "reason": "Factory did not produce full profile; see build.json and logs/task.log",
                }
            )
            continue
        card = _read(task / "scorecard.json")
        if not (task / "task.toml").is_file():
            msg = f"Incomplete Harbor task: {task}"
            raise RuntimeError(msg)
        tasks.append(
            {
                "profile": profile,
                "state": card["state"],
                "task_uri": str(final / "workspace" / task.relative_to(root)),
                "scorecard_json": json.dumps(card, sort_keys=True),
                "reason": json.dumps(card.get("state_why"), sort_keys=True),
            }
        )
    return result | {"tasks": tasks}


def read_completed(final: Path, item: dict[str, Any], build: BuildConfig) -> dict[str, Any]:
    """Verify the durable payload before reusing an attempt completed before its checkpoint."""
    manifest = _read(final / "manifest.json")
    if manifest["contract"] != build.result_contract or manifest["item"] != item:
        msg = f"Incompatible completed payload: {final}"
        raise ValueError(msg)
    if tree_digest(final / "workspace") != manifest["workspace_digest"]:
        msg = f"Completed payload changed or is incomplete: {final}"
        raise ValueError(msg)
    result = manifest["result"]
    if not isinstance(result, dict):
        msg = f"Invalid completed result: {final}"
        raise TypeError(msg)
    validate_results(pa.RecordBatch.from_pylist([result], schema=RESULT_SCHEMA))
    return result


def build_sample(item: dict[str, Any], config: ResolvedVideo2SceneBuildConfig) -> dict[str, Any]:
    """Build one bundle in a fresh ROOT and publish immutable payloads before returning a row.

    Attempts never reuse the factory's step/error caches. A completed payload can be
    reused after worker/driver loss; concurrent attempts publish via directory rename
    and accept only a fully validated winner. Failed attempts remain for diagnosis.
    """
    payloads = Path(config.output.payload_root)
    final = payloads / item["build_id"]
    if final.exists():
        return read_completed(final, item, config.build)
    source, bundle = Path(config.build.source_root), Path(item["bundle_uri"])
    if source_digest(source) != config.build.source_digest or tree_digest(bundle) != item["bundle_digest"]:
        msg = "Source or bundle content changed since the build identity was selected"
        raise ValueError(msg)
    env = build_environment(config.build)
    check_runtime(config.build, env)
    attempts = payloads / ".attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=item["build_id"] + "-", dir=attempts))
    root = attempt / "workspace"
    root.mkdir()
    for name in SOURCE_TREES:
        copy_tree(source / name, root / name)
    copied_bundle = root / "bundles" / item["sample"]
    copied_bundle.parent.mkdir()
    copy_tree(bundle, copied_bundle)
    if source_digest(root) != config.build.source_digest or tree_digest(copied_bundle) != item["bundle_digest"]:
        msg = "Source or bundle changed while copying the build workspace"
        raise ValueError(msg)
    out = root / "work_factory" / "build"
    command = [
        config.build.python_executable,
        "-B",
        str(root / "factory" / "build.py"),
        str(copied_bundle),
        "--out",
        str(out),
        "--cpus",
        str(config.build.cpus),
        "--parallel",
        "1",
        "--backend",
        "local",
    ]
    if config.build.judge:
        command.extend(["--judge", "--max-usd", str(config.build.judge_max_usd)])
    with (attempt / "build.log").open("w") as log:
        process = subprocess.run(  # noqa: S603 -- trusted source snapshot, argument vector without a shell
            command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    result = _outcome(item, root, final, judge=config.build.judge, code=process.returncode)
    validate_results(pa.RecordBatch.from_pylist([result], schema=RESULT_SCHEMA))
    manifest = {
        "item": item,
        "contract": config.build.result_contract,
        "result": result,
        "workspace_digest": tree_digest(root),
    }
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        attempt.rename(final)
    except OSError:
        if not final.exists():
            raise
        # Never overwrite a concurrent successful attempt or mutate a published payload.
        return read_completed(final, item, config.build)
    return read_completed(final, item, config.build)


def build_batch(batch: pa.Table, *, config: ResolvedVideo2SceneBuildConfig) -> pa.Table:
    """Ray task boundary: only terminal build results reach shared checkpointing."""
    return pa.Table.from_pylist([build_sample(row, config) for row in batch.to_pylist()], schema=RESULT_SCHEMA)
