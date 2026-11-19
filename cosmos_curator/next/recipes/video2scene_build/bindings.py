# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the recipe's canonical build outcomes to shared append publication."""

from cosmos_curator.next.append.contracts import AppendBindings, AppendSpec
from cosmos_curator.next.recipes.video2scene_build.config import ResolvedVideo2SceneBuildConfig
from cosmos_curator.next.recipes.video2scene_build.contracts import (
    INPUT_SCHEMA,
    RESULT_SCHEMA,
    expand_results,
    validate_results,
)


def resolve_bindings(config: ResolvedVideo2SceneBuildConfig) -> AppendBindings:
    """Resolve publication without accessing the source checkout or runtime tools."""
    return AppendBindings(
        uri=config.output.results_lance_uri,
        staging_root=config.output.staging_root,
        execution=config.execution,
        spec=AppendSpec(
            owner=config.kind,
            id_column="build_id",
            input_fields=INPUT_SCHEMA,
            result_fields=RESULT_SCHEMA,
            output_fields=RESULT_SCHEMA,
            result_contract=config.build.result_contract,
            validate_batch=validate_results,
            expand=expand_results,
            error_column="error_stage",
        ),
    )
