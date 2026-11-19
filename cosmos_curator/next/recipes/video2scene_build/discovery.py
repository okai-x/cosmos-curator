# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze explicit bundles by content before their Ray work is selected."""

import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa

from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig
from cosmos_curator.next.recipes.video2scene_build.contracts import INPUT_SCHEMA
from cosmos_curator.next.recipes.video2scene_build.snapshot import tree_digest
from cosmos_curator.next.utils.identity import canonical_digest


def discover_items(config: ResolvedVideo2SceneBuildConfig) -> pa.Table:
    """Save bundle identity and content; a resumed selection does not rediscover."""
    items: dict[str, dict[str, Any]] = {}
    for location in config.input.bundles:
        root = Path(location)
        digest = tree_digest(root)
        meta = json.loads((root / "bundle.json").read_text())
        sample = meta.get("sample") or meta.get("name") or root.name
        if not isinstance(sample, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", sample):
            msg = "Bundle sample/name must be a safe, nonempty path component"
            raise ValueError(msg)
        build_id = canonical_digest({"sample": sample, "bundle": digest, "contract": config.build.result_contract})
        items.setdefault(
            build_id, {"build_id": build_id, "sample": sample, "bundle_uri": location, "bundle_digest": digest}
        )
    return pa.Table.from_pylist(list(items.values()), schema=INPUT_SCHEMA)
