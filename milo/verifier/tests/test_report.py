"""Tests for milo.verifier.report.

Covers:
    - TestResult round-trip via to_dict / from_dict.
    - VerifierReport round-trip.
    - R_terminal formula (spec §4.4.1) — all four conjuncts.
"""

from __future__ import annotations

import json

import pytest

from milo.verifier.report import TestResult, VerifierReport


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------
class TestTestResult:
    def test_empty_constructor(self) -> None:
        r = TestResult()
        assert r.passed == []
        assert r.failed == []
        assert r.skipped == []
        assert r.passed_count == 0
        assert r.total_count == 0

    def test_counts_are_derived(self) -> None:
        r = TestResult(passed=["a", "b"], failed=["c"], skipped=["d", "e", "f"])
        assert r.passed_count == 2
        assert r.failed_count == 1
        assert r.skipped_count == 3
        assert r.total_count == 6

    def test_to_dict_includes_derived_counts(self) -> None:
        r = TestResult(passed=["a"], failed=["b"], skipped=["c"], elapsed_s=1.5, exit_code=1)
        d = r.to_dict()
        assert d["passed"] == ["a"]
        assert d["failed_count"] == 1
        assert d["skipped_count"] == 1
        assert d["elapsed_s"] == 1.5
        assert d["exit_code"] == 1
        # raw_log_path None → None in JSON.
        assert d["raw_log_path"] is None

    def test_round_trip(self) -> None:
        original = TestResult(
            passed=["t1::a", "t2::b"],
            failed=["t3::c"],
            skipped=[],
            elapsed_s=42.0,
            exit_code=1,
        )
        rebuilt = TestResult.from_dict(original.to_dict())
        assert rebuilt.passed == original.passed
        assert rebuilt.failed == original.failed
        assert rebuilt.skipped == original.skipped
        assert rebuilt.elapsed_s == original.elapsed_s
        assert rebuilt.exit_code == original.exit_code

    def test_from_dict_defensive_on_string_counts(self) -> None:
        """The on-disk milo schema stringifies counts; from_dict tolerates."""
        # We don't read counts (they're derived from lists), but make sure
        # the loader doesn't choke on partial dicts.
        rebuilt = TestResult.from_dict({"passed": [], "elapsed_s": "0.5"})
        assert rebuilt.elapsed_s == 0.5

    def test_json_round_trip(self) -> None:
        original = TestResult(passed=["a"], failed=["b"], elapsed_s=2.5)
        blob = json.dumps(original.to_dict())
        rebuilt = TestResult.from_dict(json.loads(blob))
        assert rebuilt.passed == ["a"]
        assert rebuilt.failed == ["b"]


# ---------------------------------------------------------------------------
# VerifierReport.r_terminal — spec §4.4.1 truth table
# ---------------------------------------------------------------------------
def _make_report(
    f2p_passed: list[str] | None = None,
    f2p_failed: list[str] | None = None,
    p2p_regressed: list[str] | None = None,
    test_count_nonzero: bool = True,
    passes_invariant_check: bool = True,
    invariant_violations: list[str] | None = None,
) -> VerifierReport:
    """Helper: synthesize a VerifierReport with whatever invariants we want."""
    fp_passed = list(f2p_passed or [])
    fp_failed_set = list(f2p_failed or []) + list(p2p_regressed or [])
    return VerifierReport(
        instance_id="testinst",
        baseline_result=TestResult(
            passed=list(p2p_regressed or []),  # baseline must have them passing
        ),
        test_patch_result=TestResult(),
        fix_patch_result=TestResult(
            passed=fp_passed,
            failed=fp_failed_set,
            skipped=["dummy_skip"] if test_count_nonzero else [],
        ),
        f2p_passed=fp_passed,
        f2p_failed=list(f2p_failed or []),
        p2p_regressed=list(p2p_regressed or []),
        test_count_nonzero=test_count_nonzero,
        passes_invariant_check=passes_invariant_check,
        invariant_violations=list(invariant_violations or []),
    )


class TestVerifierReportRTerminal:
    def test_all_clear_returns_1(self) -> None:
        r = _make_report(f2p_passed=["t1"], f2p_failed=[], p2p_regressed=[])
        assert r.r_terminal() == 1

    def test_f2p_failed_returns_0(self) -> None:
        r = _make_report(f2p_passed=["t1"], f2p_failed=["t2"])
        assert r.r_terminal() == 0

    def test_p2p_regressed_returns_0(self) -> None:
        r = _make_report(f2p_passed=["t1"], p2p_regressed=["t3"])
        assert r.r_terminal() == 0

    def test_zero_test_count_returns_0(self) -> None:
        r = _make_report(f2p_passed=["t1"], test_count_nonzero=False)
        assert r.r_terminal() == 0

    def test_invariant_violation_returns_0(self) -> None:
        r = _make_report(
            f2p_passed=["t1"],
            passes_invariant_check=False,
            invariant_violations=["I-1: test path edited"],
        )
        assert r.r_terminal() == 0

    def test_invariant_violation_overrides_all_else(self) -> None:
        """Spec §7 — any violation → R_terminal=0 regardless of test outcome."""
        r = _make_report(
            f2p_passed=["t1", "t2"],
            f2p_failed=[],
            p2p_regressed=[],
            test_count_nonzero=True,
            passes_invariant_check=False,
            invariant_violations=["I-7: git apply --check failed"],
        )
        assert r.r_terminal() == 0


class TestVerifierReportSerialization:
    def test_to_dict_includes_r_terminal(self) -> None:
        r = _make_report(f2p_passed=["t1"])
        d = r.to_dict()
        assert d["r_terminal"] == 1
        assert d["instance_id"] == "testinst"
        assert d["passes_invariant_check"] is True

    def test_round_trip(self) -> None:
        original = _make_report(
            f2p_passed=["pass1"],
            f2p_failed=["fail1"],
            p2p_regressed=["reg1"],
            passes_invariant_check=False,
            invariant_violations=["I-3: p2p regressed"],
        )
        rebuilt = VerifierReport.from_dict(original.to_dict())
        assert rebuilt.instance_id == original.instance_id
        assert rebuilt.f2p_passed == original.f2p_passed
        assert rebuilt.f2p_failed == original.f2p_failed
        assert rebuilt.p2p_regressed == original.p2p_regressed
        assert rebuilt.passes_invariant_check is False
        assert rebuilt.invariant_violations == ["I-3: p2p regressed"]
        assert rebuilt.r_terminal() == 0

    def test_json_round_trip(self) -> None:
        original = _make_report(f2p_passed=["x"])
        blob = json.dumps(original.to_dict())
        rebuilt = VerifierReport.from_dict(json.loads(blob))
        assert rebuilt.r_terminal() == 1


# ---------------------------------------------------------------------------
# VerifierReport constructor requires instance_id
# ---------------------------------------------------------------------------
def test_verifier_report_requires_instance_id() -> None:
    """The dataclass requires instance_id (positional)."""
    with pytest.raises(TypeError):
        VerifierReport(  # type: ignore[call-arg]
            baseline_result=TestResult(),
            test_patch_result=TestResult(),
            fix_patch_result=TestResult(),
        )
