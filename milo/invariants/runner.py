"""Invariant runner — orchestrates I-1..I-8 in order and returns an InvariantsResult.

Implements `IMPLEMENTATION_PLAN.md` v0.4 Phase 6 / `RL_GYM_SPEC.md` v0.7 §7.
Per spec: a trace that violates ANY invariant terminates with R_total = 0,
regardless of test outcome. The reward aggregator (`milo/reward/composite.py`)
reads `InvariantsResult.passed` and forces R_terminal to 0 when False.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from milo.invariants.checks import (
    InvariantsResult,
    InvariantViolation,
    check_i_1,
    check_i_2,
    check_i_3,
    check_i_4,
    check_i_5,
    check_i_6,
    check_i_7,
    check_i_8,
)

if TYPE_CHECKING:
    from milo.verifier.report import VerifierReport


# Order matters: cheap deterministic checks first (I-7 git-apply is the cheapest
# and most common rejection); judge-influenced I-6 last because it may need a
# rubric report that arrives after the verifier report.
_INVARIANT_PIPELINE = (
    ("I-7", check_i_7),   # malformed diff
    ("I-1", check_i_1),   # test-path edits
    ("I-4", check_i_4),   # cwd escape
    ("I-5", check_i_5),   # write to forbidden paths
    ("I-6", check_i_6),   # test-fixture tampering (structural part)
    ("I-2", check_i_2),   # F2P+P2P count nonzero (needs verifier_report)
    ("I-3", check_i_3),   # P2P regression (needs verifier_report)
    ("I-8", check_i_8),   # runtime-cost bound (needs verifier_report)
)


def run_all_invariants(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: "VerifierReport | None" = None,
) -> InvariantsResult:
    """Run I-1..I-8 in pipeline order and collect every violation.

    We do NOT short-circuit on the first violation — auditors want the full list
    so they can see which combinations of hacks the policy attempted. The
    reward aggregator only cares about the boolean `passed`.

    `verifier_report` may be None during pre-rollout dry-run validation (e.g.
    smoke tests of just I-1/I-4/I-5/I-6/I-7 against a candidate diff). Checks
    that need it (I-2, I-3, I-8) skip cleanly when it's missing — they return
    None when the report is absent.
    """
    violations: list[InvariantViolation] = []
    for code, check in _INVARIANT_PIPELINE:
        try:
            v = check(candidate_patch, instance, verifier_report)
        except Exception as exc:  # invariant code must not crash the pipeline
            v = InvariantViolation(
                code=code,
                message=f"invariant check raised {type(exc).__name__}: {exc}",
                details={"exception_type": type(exc).__name__},
            )
        if v is not None:
            violations.append(v)
    return InvariantsResult(passed=not violations, violations=violations)
