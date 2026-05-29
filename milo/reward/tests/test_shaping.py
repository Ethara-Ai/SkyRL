"""Per-step test-delta shaping reward tests — spec §4.4.2.

Covers Δ_pass / Δ_regress edge cases, the λ=2 asymmetry, the prev=None
first-step case, and that f2p/p2p set membership is required to count.
"""

from __future__ import annotations

import pytest

from milo.reward.shaping import TestResult, compute_delta


def _result(passing: list[str], failing: list[str],
            f2p: list[str], p2p: list[str]) -> TestResult:
    return TestResult(
        passing_tests=set(passing),
        failing_tests=set(failing),
        f2p_tests=set(f2p),
        p2p_tests=set(p2p),
    )


def test_first_step_with_no_progress_is_zero() -> None:
    """prev=None; current has nothing newly passing in f2p → R_delta = 0."""
    curr = _result(passing=[], failing=["test_x"], f2p=["test_x"], p2p=[])
    assert compute_delta(None, curr) == 0.0


def test_first_step_picks_up_pass_to_pass_as_zero() -> None:
    """A test passing before the first run doesn't count as Δ_pass.

    With prev=None we treat *previous* as empty; the test is "newly passing"
    relative to nothing, BUT it must be in f2p for the delta to fire. p2p
    tests passing on the first run do not contribute to Δ_pass.
    """
    curr = _result(passing=["t_p2p"], failing=[], f2p=["t_f2p"], p2p=["t_p2p"])
    assert compute_delta(None, curr) == 0.0


def test_single_f2p_flip_yields_plus_one() -> None:
    prev = _result(passing=[], failing=["t_f2p"], f2p=["t_f2p"], p2p=[])
    curr = _result(passing=["t_f2p"], failing=[], f2p=["t_f2p"], p2p=[])
    assert compute_delta(prev, curr) == 1.0


def test_single_p2p_regression_yields_minus_lambda() -> None:
    """Default λ=2: one p2p regression → -2."""
    prev = _result(passing=["t_p2p"], failing=[], f2p=[], p2p=["t_p2p"])
    curr = _result(passing=[], failing=["t_p2p"], f2p=[], p2p=["t_p2p"])
    assert compute_delta(prev, curr) == -2.0


def test_lambda_is_configurable() -> None:
    prev = _result(passing=["t_p2p"], failing=[], f2p=[], p2p=["t_p2p"])
    curr = _result(passing=[], failing=["t_p2p"], f2p=[], p2p=["t_p2p"])
    assert compute_delta(prev, curr, lambda_=5.0) == -5.0
    assert compute_delta(prev, curr, lambda_=0.0) == 0.0


def test_asymmetric_one_fix_plus_one_regression() -> None:
    """Mixed step: +1 f2p, -1 p2p regression → 1 - 2*1 = -1 (asymmetry)."""
    prev = _result(
        passing=["t_p2p"], failing=["t_f2p"],
        f2p=["t_f2p"], p2p=["t_p2p"],
    )
    curr = _result(
        passing=["t_f2p"], failing=["t_p2p"],
        f2p=["t_f2p"], p2p=["t_p2p"],
    )
    assert compute_delta(prev, curr) == -1.0


def test_symmetric_one_fix_no_regression() -> None:
    """Pure-progress step: +1 f2p, 0 regression → 1.0 with any λ."""
    prev = _result(
        passing=["t_p2p"], failing=["t_f2p"],
        f2p=["t_f2p"], p2p=["t_p2p"],
    )
    curr = _result(
        passing=["t_p2p", "t_f2p"], failing=[],
        f2p=["t_f2p"], p2p=["t_p2p"],
    )
    for lam in (0.0, 1.0, 2.0, 5.0):
        assert compute_delta(prev, curr, lambda_=lam) == 1.0


def test_test_outside_f2p_or_p2p_does_not_count() -> None:
    """A test that flips PASS but is in neither f2p nor p2p does nothing."""
    prev = _result(passing=[], failing=["t_other"], f2p=[], p2p=[])
    curr = _result(passing=["t_other"], failing=[], f2p=[], p2p=[])
    assert compute_delta(prev, curr) == 0.0


def test_test_outside_p2p_failing_does_not_regress() -> None:
    """A failing test not in p2p doesn't count as Δ_regress."""
    prev = _result(passing=["t_x"], failing=[], f2p=[], p2p=[])
    curr = _result(passing=[], failing=["t_x"], f2p=[], p2p=[])
    assert compute_delta(prev, curr) == 0.0


def test_multiple_fixes_and_regressions_sum() -> None:
    """Δ_pass and Δ_regress both count cardinality. 3 fixes, 2 regress → 3 - 4 = -1."""
    prev = _result(
        passing=["p1", "p2"],
        failing=["f1", "f2", "f3"],
        f2p=["f1", "f2", "f3"],
        p2p=["p1", "p2"],
    )
    curr = _result(
        passing=["f1", "f2", "f3"],
        failing=["p1", "p2"],
        f2p=["f1", "f2", "f3"],
        p2p=["p1", "p2"],
    )
    # +3 fixes, -2 regressions × λ=2 → 3 - 4 = -1.
    assert compute_delta(prev, curr) == -1.0


def test_no_change_step_is_zero() -> None:
    prev = _result(
        passing=["p1"], failing=["f1"], f2p=["f1"], p2p=["p1"],
    )
    curr = _result(
        passing=["p1"], failing=["f1"], f2p=["f1"], p2p=["p1"],
    )
    assert compute_delta(prev, curr) == 0.0


def test_negative_lambda_raises() -> None:
    prev = _result(passing=[], failing=[], f2p=[], p2p=[])
    curr = _result(passing=[], failing=[], f2p=[], p2p=[])
    with pytest.raises(ValueError):
        compute_delta(prev, curr, lambda_=-1.0)


def test_test_result_accepts_lists_and_normalizes_to_sets() -> None:
    """`TestResult` __post_init__ should coerce list inputs to set."""
    r = TestResult(
        passing_tests=["a", "b"],     # type: ignore[arg-type]
        failing_tests=["c"],          # type: ignore[arg-type]
        f2p_tests=["c"],              # type: ignore[arg-type]
        p2p_tests=["a", "b"],         # type: ignore[arg-type]
    )
    assert r.passing_tests == {"a", "b"}
    assert r.failing_tests == {"c"}
    assert r.f2p_tests == {"c"}
    assert r.p2p_tests == {"a", "b"}
