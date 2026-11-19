# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared append publication and recovery for build outcomes."""

from tests.cosmos_curator.next.append.conformance import AppendConformance


class TestBuildConformance(AppendConformance):
    """Apply the common recovery contract, including failed and held builds."""
