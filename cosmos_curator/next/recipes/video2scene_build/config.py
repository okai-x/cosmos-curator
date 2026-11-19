# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, versioned configuration for shared-filesystem environment builds."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cosmos_curator.next.append.config import AppendExecutionConfig
from cosmos_curator.next.core.config import apply_dotted_overrides


def local_path(value: str) -> str:
    """Require explicit local/shared paths, without accessing their contents."""
    if not value.strip() or value != value.strip() or "://" in value:
        msg = "Use a local or shared-filesystem path without surrounding whitespace"
        raise ValueError(msg)
    return str(Path(value).expanduser().resolve())


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class InputConfig(_Config):
    """Explicit immutable bundle directories, accessible to driver and workers."""

    bundles: list[str] = Field(min_length=1)

    @field_validator("bundles")
    @classmethod
    def _paths(cls, values: list[str]) -> list[str]:
        return sorted({local_path(value) for value in values})


class BuildConfig(_Config):
    """Result-defining source, tools, and controlled measurement conditions."""

    source_root: str
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_runtime_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_executable: str = "python3"
    blender_executable: str = "blender"
    blender_version: str = Field(default="5.1.2", pattern=r"^\d+\.\d+\.\d+$")
    measurement_class: str = Field(min_length=1, description="Operator-defined homogeneous, exclusive worker class.")
    cpus: int = Field(default=8, ge=8, description="Worker CPUs reserved for the factory and its native subprocesses.")
    judge: bool = False
    judge_max_usd: float = Field(default=5.0, gt=0, allow_inf_nan=False)

    _source_path = field_validator("source_root")(local_path)

    @property
    def result_contract(self) -> dict[str, Any]:
        """Pin code, native runtime, Blender and measurement semantics; exclude deployment paths."""
        return {
            "adapter_version": 2,
            "execution_backend": "local",
            "memory_metric": "sampled_process_tree_rss",
            "memory_sample_interval_s": 0.05,
            "source_digest": self.source_digest,
            "python_runtime_digest": self.python_runtime_digest,
            "blender_version": self.blender_version,
            "measurement_class": self.measurement_class,
            "cpus": self.cpus,
            "fixture_parallelism": 1,
            "judge": self.judge,
            "judge_max_usd": self.judge_max_usd if self.judge else None,
        }


class OutputConfig(_Config):
    """One durable shared root, separate from source and input trees."""

    root: str

    _root_path = field_validator("root")(local_path)

    @property
    def results_lance_uri(self) -> str:
        """Canonical build outcomes, including non-released tasks."""
        return str(Path(self.root) / "builds.lance")

    @property
    def staging_root(self) -> str:
        """Shared append recovery workspace."""
        return str(Path(self.root) / "staging")

    @property
    def payload_root(self) -> str:
        """Durable build directories referenced by canonical results."""
        return str(Path(self.root) / "payloads")


class ExecutionConfig(AppendExecutionConfig):
    """Scheduling and checkpoint tuning, independent of scoring semantics."""

    rows_per_checkpoint: int = Field(default=1, ge=1)
    memory_gib: float = Field(default=20.0, ge=16, allow_inf_nan=False)


class ResolvedVideo2SceneBuildConfig(_Config):
    """Canonical configuration for a bundle-to-environment recipe."""

    schema_version: Literal[1]
    kind: Literal["video2scene-build"]
    input: InputConfig
    build: BuildConfig
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def _separate(self) -> Self:
        output = Path(self.output.root)
        for value in (self.build.source_root, *self.input.bundles):
            source = Path(value)
            if output.is_relative_to(source) or source.is_relative_to(output):
                msg = "Output, source checkout and input bundle trees must not overlap"
                raise ValueError(msg)
        return self


def resolve_config_data(data: Mapping[str, Any], *, overrides: Sequence[str] = ()) -> ResolvedVideo2SceneBuildConfig:
    """Resolve overrides and strict settings without starting runtime tools."""
    resolved = copy.deepcopy(dict(data))
    apply_dotted_overrides(resolved, overrides)
    return ResolvedVideo2SceneBuildConfig.model_validate(resolved)


def resolve_config(path: str | Path, *, overrides: Sequence[str] = ()) -> ResolvedVideo2SceneBuildConfig:
    """Read a YAML or JSON recipe configuration."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "video2scene-build config must be a mapping"
        raise TypeError(msg)
    return resolve_config_data(data, overrides=overrides)


def resolved_config_to_json(config: ResolvedVideo2SceneBuildConfig) -> str:
    """Render canonical settings without credentials."""
    return config.model_dump_json(indent=2) + "\n"


def config_schema_json() -> str:
    """Render the strict public configuration schema."""
    return json.dumps(ResolvedVideo2SceneBuildConfig.model_json_schema(), indent=2) + "\n"


def config_template() -> dict[str, Any]:
    """Provide an editable rules-only configuration with explicit identity placeholders."""
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "video2scene-build",
            "input": {"bundles": ["/shared/bundles/example"]},
            "build": {
                "source_root": "/opt/video2sceneRL-ver",
                "source_digest": "0" * 64,
                "python_runtime_digest": "0" * 64,
                "measurement_class": "exclusive-cpu-worker",
            },
            "output": {"root": "/shared/video2scene-output"},
        }
    ).model_dump(mode="json", exclude_none=True)


def config_template_yaml() -> str:
    """Explain immutable identities and worker requirements alongside defaults."""
    return (
        "# Replace source and worker runtime digests before running.\n"
        "# python -m cosmos_curator.next.recipes.video2scene_build.snapshot SOURCE_ROOT\n"
        "# Use --runtime with the selected Python and worker PATH (includes FFmpeg).\n"
        "# Managed workers need video2scene_build_slot: 1 and the same shared paths.\n"
        "# Use dedicated homogeneous workers: AC-14 timing affects task release.\n"
        "# judge: true makes paid VLM calls; retries may repeat charges.\n"
    ) + yaml.safe_dump(config_template(), sort_keys=False)


def config_template_payload() -> dict[str, Any]:
    """Describe the template for generic CLI consumers."""
    return {
        "kind": "video2scene-build",
        "description": "Build Harbor environments from immutable bundles.",
        "required_fields": [
            {"path": key}
            for key in (
                "input.bundles",
                "build.source_root",
                "build.source_digest",
                "build.python_runtime_digest",
                "build.measurement_class",
                "output.root",
            )
        ],
        "config": config_template(),
    }
