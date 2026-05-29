"""Milo composite reward module — spec §4.4, §5.4, §6.8.

Public API:

    >>> from milo.reward import CompositeRewardAggregator, compute_delta, compute_tir
    >>> agg = CompositeRewardAggregator("composite")
    >>> decomp = agg.aggregate(verifier_report, rubric_report, shaping_rewards, tir_rewards)
    >>> decomp.to_json()

See spec §4.4 (formula), §6.8 (decomposition schema), and §7 (the v0.7-hardened
invariant override that zeroes `R_terminal` when any invariant fails).
"""

from __future__ import annotations

from milo.reward.composite import (
    CompositeRewardAggregator,
    RubricReportLike,
    StubRubricReport,
    StubVerifierReport,
    VerifierReportLike,
)
from milo.reward.decomposition import RewardDecomposition, RewardPreset
from milo.reward.shaping import TestResult, compute_delta
from milo.reward.tir import MILO_TOOL_SCHEMAS, compute_tir

__all__ = [
    "CompositeRewardAggregator",
    "MILO_TOOL_SCHEMAS",
    "RewardDecomposition",
    "RewardPreset",
    "RubricReportLike",
    "StubRubricReport",
    "StubVerifierReport",
    "TestResult",
    "VerifierReportLike",
    "compute_delta",
    "compute_tir",
]
