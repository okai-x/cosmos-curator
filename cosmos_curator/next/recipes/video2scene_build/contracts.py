# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One canonical build outcome per immutable bundle and build contract."""

from collections.abc import Iterable

import pyarrow as pa

INPUT_SCHEMA = pa.schema(
    [
        pa.field("build_id", pa.string(), nullable=False),
        pa.field("sample", pa.string(), nullable=False),
        pa.field("bundle_uri", pa.string(), nullable=False),
        pa.field("bundle_digest", pa.string(), nullable=False),
    ]
)
TASK_TYPE = pa.struct(
    [
        pa.field("profile", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("task_uri", pa.string()),
        pa.field("scorecard_json", pa.string()),
        pa.field("reason", pa.string()),
    ]
)
RESULT_SCHEMA = pa.schema(
    [
        *INPUT_SCHEMA,
        pa.field("status", pa.string(), nullable=False),
        pa.field("manifest_uri", pa.string(), nullable=False),
        pa.field("tasks", pa.list_(TASK_TYPE), nullable=False),
        pa.field("error_stage", pa.string()),
        pa.field("error_message", pa.string()),
    ]
)


def validate_results(batch: pa.RecordBatch) -> None:
    """Only completed builds or explicit intake returns may be checkpointed."""
    for row in batch.to_pylist():
        valid = bool(row["manifest_uri"] and row["build_id"] and row["bundle_digest"])
        if row["status"] == "completed":
            tasks = row["tasks"]
            valid &= row["error_stage"] is None and bool(tasks)
            valid &= len({task["profile"] for task in tasks}) == len(tasks)
            valid &= any(task["profile"] == "rules" and task["task_uri"] for task in tasks)
            for task in tasks:
                valid &= task["profile"] in {"rules", "full"}
                if task["state"] == "not_generated":
                    valid &= task["profile"] == "full" and task["task_uri"] is None and bool(task["reason"])
                else:
                    valid &= task["state"] in {"released", "returned", "held", "quarantined", "draft"}
                    valid &= bool(task["task_uri"] and task["scorecard_json"])
        else:
            valid &= row["status"] == "returned" and row["error_stage"] == "intake"
            valid &= not row["tasks"] and bool(row["error_message"])
        if not valid:
            msg = f"Nonterminal or inconsistent build result: {row['build_id']}"
            raise ValueError(msg)


def expand_results(results: pa.Table) -> Iterable[pa.Table]:
    """Keep all outcomes; profiles remain together under their build identity."""
    yield results
