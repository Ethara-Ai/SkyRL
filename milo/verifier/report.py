"""TestResult + VerifierReport dataclasses.

Implements:
    - RL_GYM_SPEC.md v0.7 §4.4.1 (R_terminal formula).
    - RL_GYM_SPEC.md v0.7 §6.6 (verifier report shape).
    - IMPLEMENTATION_PLAN.md v0.4 Phase 2.3 (VerifierReport schema + invariant
      integration).

The verifier emits one TestResult per Docker run (baseline / test_patch /
fix_patch) and one VerifierReport per rollout (composition + invariant flags).

The Phase 4 reward aggregator reads `VerifierReport.r_terminal()` plus the
invariant flags and composes the final scalar reward.

GOTCHA: this module does NOT compute composite reward (alpha/beta/lambda/gamma).
It computes only `R_terminal ∈ {0, 1}` per spec §4.4.1. The reward layer is
responsible for the composite. See milo/reward/aggregator.py (Phase 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=False)
class TestResult:
    """The output of one Docker run.

    Fields are kept as `list[str]` (not counts) deliberately — the on-disk
    milo-bench schema stores counts as strings (`"37"`), which is a footgun on
    the consumer side. Counts are derived properties, not stored fields.

    Attributes:
        passed:        stable test-IDs of passing tests.
        failed:        stable test-IDs of failing tests.
        skipped:       stable test-IDs of skipped tests.
        elapsed_s:     wall-clock elapsed of the test run (seconds).
        exit_code:     exit code of the underlying test command.
        raw_log_path:  absolute path to the raw stdout/stderr capture on disk.

    Stable test-ID conventions vary by language. We adopt the conventions used
    by the corresponding test framework's machine-readable output:
        * Python pytest:  `path/to/test_file.py::ClassName::test_name`
        * Jest (JS/TS):   `<describe>... > <it>`
        * JUnit (Java):   `package.Class#method`
        * Go:             `<package>.Test<Name>` or `Test<Name>` per package
        * Cargo (Rust):   `<crate>::module::test_name`
        * CTest (C/C++):  `<testname>` as registered with `add_test(...)`
    """

    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    exit_code: int = 0
    raw_log_path: Path | None = None

    # ------------------------------------------------------------------
    # Derived properties (NOT stored)
    # ------------------------------------------------------------------
    @property
    def passed_count(self) -> int:
        return len(self.passed)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def total_count(self) -> int:
        return self.passed_count + self.failed_count + self.skipped_count

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-safe dict suitable for caching to disk."""
        return {
            "passed": list(self.passed),
            "failed": list(self.failed),
            "skipped": list(self.skipped),
            "elapsed_s": float(self.elapsed_s),
            "exit_code": int(self.exit_code),
            "raw_log_path": str(self.raw_log_path) if self.raw_log_path else None,
            # Derived counts: written for convenience but never read by
            # `from_dict` (lengths are the source of truth).
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        """Inverse of to_dict. Defensive on str-vs-int counts (on-disk milo
        schema quirk; see milo/audit/audit_dataset.py)."""
        raw_log = data.get("raw_log_path")
        return cls(
            passed=list(data.get("passed", []) or []),
            failed=list(data.get("failed", []) or []),
            skipped=list(data.get("skipped", []) or []),
            elapsed_s=float(data.get("elapsed_s", 0.0) or 0.0),
            exit_code=int(data.get("exit_code", 0) or 0),
            raw_log_path=Path(raw_log) if raw_log else None,
        )

    @classmethod
    def empty(cls) -> TestResult:
        """An empty result used as a sentinel before a run completes."""
        return cls()


@dataclass(frozen=False)
class VerifierReport:
    """The structured artefact emitted at terminal time.

    Composes three TestResult instances (baseline / test_patch / fix_patch) plus
    the F2P/P2P classification and the §7 invariant flags. Consumers (Phase 4
    reward aggregator, Phase 5 logging overlay, Phase 17 nightly audit) read
    this dataclass.

    Schema mirrors RL_GYM_SPEC.md v0.7 §6.6 (and Aurora §6.5 Report).

    Attributes:
        instance_id:            the LHT instance identifier.
        baseline_result:        TestResult of the no-patches run.
        test_patch_result:      TestResult of the test_patch-only run.
        fix_patch_result:       TestResult of the fix_patch+test_patch run.
        f2p_passed:             test-IDs from the expected F2P set that actually
                                pass under fix_patch.
        f2p_failed:             test-IDs from the expected F2P set that did NOT
                                pass under fix_patch.
        p2p_regressed:          test-IDs from the expected P2P set that
                                regressed (passed in baseline, failed under
                                fix_patch).
        test_count_nonzero:     True iff fix_patch_result has any total tests.
        passes_invariant_check: True iff ALL of I-1..I-8 pass.
        invariant_violations:   human-readable list, e.g.
                                ["I-2: f2p_passed empty",
                                 "I-7: git apply --check failed"].
        elapsed_total_s:        wall-clock for the three runs (sum, not max).
    """

    instance_id: str
    baseline_result: TestResult
    test_patch_result: TestResult
    fix_patch_result: TestResult
    f2p_passed: list[str] = field(default_factory=list)
    f2p_failed: list[str] = field(default_factory=list)
    p2p_regressed: list[str] = field(default_factory=list)
    test_count_nonzero: bool = False
    passes_invariant_check: bool = True
    invariant_violations: list[str] = field(default_factory=list)
    elapsed_total_s: float = 0.0

    # ------------------------------------------------------------------
    # Spec §4.4.1 terminal-reward computation
    # ------------------------------------------------------------------
    def r_terminal(self) -> int:
        """Compute R_terminal ∈ {0, 1} per RL_GYM_SPEC.md v0.7 §4.4.1.

            R_terminal = 1 iff
                (every f2p test passes in fix_patch_run) AND
                (no p2p test regressed) AND
                (fix_patch_run.test_count > 0) AND
                (no §7 invariant violated)
              = 0 otherwise

        Pure function over the report's own fields. The reward aggregator
        (Phase 4) reads this and applies alpha/beta/lambda/gamma.
        """
        if not self.passes_invariant_check:
            return 0
        if not self.test_count_nonzero:
            return 0
        if self.f2p_failed:
            return 0
        if self.p2p_regressed:
            return 0
        # All F2P pass, no P2P regress, the runner ran tests, and invariants OK.
        # NOTE: an empty f2p_passed list is acceptable only if the f2p set
        # itself is empty (which I-2 v0.7-hardened rejects). The check here is
        # purely "no failures." I-2 handles the empty-F2P case via the
        # invariant-violation path, not via r_terminal().
        return 1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; mirrors RL_GYM_SPEC.md v0.7 §6.6."""
        return {
            "instance_id": self.instance_id,
            "baseline_result": self.baseline_result.to_dict(),
            "test_patch_result": self.test_patch_result.to_dict(),
            "fix_patch_result": self.fix_patch_result.to_dict(),
            "f2p_passed": list(self.f2p_passed),
            "f2p_failed": list(self.f2p_failed),
            "p2p_regressed": list(self.p2p_regressed),
            "test_count_nonzero": bool(self.test_count_nonzero),
            "passes_invariant_check": bool(self.passes_invariant_check),
            "invariant_violations": list(self.invariant_violations),
            "elapsed_total_s": float(self.elapsed_total_s),
            "r_terminal": self.r_terminal(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifierReport:
        return cls(
            instance_id=data["instance_id"],
            baseline_result=TestResult.from_dict(data["baseline_result"]),
            test_patch_result=TestResult.from_dict(data["test_patch_result"]),
            fix_patch_result=TestResult.from_dict(data["fix_patch_result"]),
            f2p_passed=list(data.get("f2p_passed", []) or []),
            f2p_failed=list(data.get("f2p_failed", []) or []),
            p2p_regressed=list(data.get("p2p_regressed", []) or []),
            test_count_nonzero=bool(data.get("test_count_nonzero", False)),
            passes_invariant_check=bool(data.get("passes_invariant_check", True)),
            invariant_violations=list(data.get("invariant_violations", []) or []),
            elapsed_total_s=float(data.get("elapsed_total_s", 0.0) or 0.0),
        )

    # asdict is provided for parity with the dataclasses ecosystem (e.g.
    # serializers that don't know about to_dict). Prefer to_dict() in new
    # code — it does the field shaping the rest of the pipeline expects.
    def asdict(self) -> dict[str, Any]:
        return asdict(self)
