"""milo.verifier — three-Docker-run verifier + per-language test runners.

Implements IMPLEMENTATION_PLAN.md v0.4 Phase 2 and RL_GYM_SPEC.md v0.7 §5.2.

Modules:
    report          — TestResult, VerifierReport dataclasses; r_terminal() formula
                      per spec §4.4.1.
    three_run       — ThreeRunVerifier: caches baseline + test_patch per instance,
                      runs fix_patch per rollout, composes the report and delegates
                      to the §7 invariant checks.
    runners.base    — PerLanguageTestRunner Protocol + RawTestRun dataclass (the
                      contract for plugging in a new language).
    runners.<lang>  — Concrete runners for python, javascript, typescript, java,
                      go, rust, c, cpp. See runners/registry.py for the lookup.

Design notes:
    * The verifier is execution-only. It does NOT compute composite reward (that
      lives in milo/reward/aggregator.py per Phase 4). It DOES delegate to the
      I-1..I-8 invariant checks (Phase 6 / milo/invariants/checks.py) and reports
      pass/fail flags. The reward layer reads those flags and forces R_terminal=0
      on any violation per spec §7.
    * The on-disk milo-bench schema has two known traps (see milo/audit):
        (a) `run_result.passed_count` etc. may be string-encoded ints
        (b) `f2p_tests[name]` may be `str(json.dumps({"run","test","fix"}))`
      The verifier's runners produce TestResult with proper list[str], avoiding
      the trap on the emit side. Consumers reading from disk must defensively
      decode (see milo.audit.audit_dataset.get_test_count for the helper).

See also:
    IMPLEMENTATION_NOTES.md in this directory — what's complete vs. stubbed.
"""

from __future__ import annotations

from milo.verifier.report import (
    TestResult,
    VerifierReport,
)
from milo.verifier.three_run import ThreeRunVerifier

__all__ = [
    "TestResult",
    "VerifierReport",
    "ThreeRunVerifier",
]
