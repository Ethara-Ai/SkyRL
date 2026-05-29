"""Phase 11.8 verifier construction subroutine.

Synthesizes F2P / P2P test lists for Cohort B milo-bench instances (the
~174 tasks with empty `f2p_tests` and `p2p_tests` but non-empty `fix_patch`
+ `test_patch`). Implements `IMPLEMENTATION_PLAN.md` v0.4 §11.8 algorithm:

    1. Apply `test_patch` to `tag_start`; run runner with test-discovery;
       capture every test found.
    2. Diff vs baseline: newly-existing or status-changed tests = candidate F2P;
       unchanged passing tests = candidate P2P.
    3. Apply `fix_patch`; re-run. Tests flipping FAIL→PASS confirm F2P;
       tests staying PASS confirm P2P.
    4. Emit enriched instance with `f2p_tests`/`p2p_tests` populated in the
       on-disk milo schema shape (`{test_id: str(json.dumps({"run","test","fix"}))}`)
       and a `verifier_synthesized: true` provenance flag.

Returns `None` when reconstruction yields zero F2P candidates — the task is
then demoted to Cohort C (drop) per §11.9.

Per-task wall-clock: 3 × `test_command_timeout` (default 600 s each = ~30 min).
Compute: trivially parallelizable across instances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from milo.verifier.runners.base import PerLanguageTestRunner


# ---------------------------------------------------------------------------
# Sandbox contract (mirrors milo.lht_adapter.docker_runtime.Sandbox to avoid a
# hard import dependency from this offline pipeline onto the runtime).
# ---------------------------------------------------------------------------


class SandboxProto(Protocol):
    def exec(self, cmd: str, timeout: int = 180, cwd: str | None = None) -> dict[str, Any]: ...
    def apply_patch(self, diff: str) -> dict[str, Any]: ...
    def reset_to(self, sha: str) -> None: ...


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclass
class VerifierConstructionResult:
    """Outcome of running `construct_verifier` on one instance."""

    instance_id: str
    enriched_instance: dict[str, Any] | None
    success: bool
    reason: str = ""
    candidate_f2p_count: int = 0
    candidate_p2p_count: int = 0
    confirmed_f2p_count: int = 0
    confirmed_p2p_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "success": self.success,
            "reason": self.reason,
            "candidate_f2p_count": self.candidate_f2p_count,
            "candidate_p2p_count": self.candidate_p2p_count,
            "confirmed_f2p_count": self.confirmed_f2p_count,
            "confirmed_p2p_count": self.confirmed_p2p_count,
            "timings": self.timings,
        }


# ---------------------------------------------------------------------------
# The algorithm.
# ---------------------------------------------------------------------------


def _milo_inner_test_record(run: str, test: str, fix: str) -> str:
    """Encode a per-test status triple in the on-disk milo schema shape.

    On disk, each test's value is `str(json.dumps({"run","test","fix"}))` —
    the outer str() is what creates the double-encoded JSON quirk Phase 0.6
    documented. We reproduce it exactly so the synthesized record is
    indistinguishable from a hand-written one downstream.
    """
    return json.dumps({"run": run, "test": test, "fix": fix})


def construct_verifier(
    instance: dict[str, Any],
    sandbox: SandboxProto,
    runner: PerLanguageTestRunner,
    test_command_timeout: int = 600,
) -> VerifierConstructionResult:
    """Run the 4-step Phase 11.8 algorithm against one Cohort B instance.

    `sandbox` is a fresh sandbox already pinned to `base.sha`. The caller
    owns sandbox lifecycle (open before, close after).
    """
    iid = instance["instance_id"]
    base_sha = (instance.get("base") or {}).get("sha", "")
    test_patch = instance.get("test_patch", "") or ""
    fix_patch = instance.get("fix_patch", "") or ""

    if not test_patch.strip() or not fix_patch.strip():
        return VerifierConstructionResult(
            instance_id=iid,
            enriched_instance=None,
            success=False,
            reason="missing test_patch or fix_patch — not eligible for Cohort B",
        )

    timings: dict[str, float] = {}

    # ---- Step 0: baseline run (no patches) ----
    sandbox.reset_to(base_sha)
    raw_baseline = runner.run_tests(sandbox, test_filter=None, timeout_s=test_command_timeout)
    timings["baseline_s"] = raw_baseline.elapsed_s
    baseline = runner.parse(raw_baseline)
    baseline_status: dict[str, str] = {}
    for t in baseline.passed:
        baseline_status[t] = "PASS"
    for t in baseline.failed:
        baseline_status[t] = "FAIL"
    for t in baseline.skipped:
        baseline_status[t] = "SKIP"

    # ---- Step 1: apply test_patch, re-run, capture all tests ----
    apply_res = sandbox.apply_patch(test_patch)
    if apply_res.get("returncode", 0) != 0:
        return VerifierConstructionResult(
            instance_id=iid,
            enriched_instance=None,
            success=False,
            reason=f"test_patch apply failed: {apply_res.get('output', '')[:500]}",
            timings=timings,
        )
    raw_after_test_patch = runner.run_tests(sandbox, test_filter=None, timeout_s=test_command_timeout)
    timings["test_patch_s"] = raw_after_test_patch.elapsed_s
    after_test = runner.parse(raw_after_test_patch)
    after_test_status: dict[str, str] = {}
    for t in after_test.passed:
        after_test_status[t] = "PASS"
    for t in after_test.failed:
        after_test_status[t] = "FAIL"
    for t in after_test.skipped:
        after_test_status[t] = "SKIP"

    # ---- Step 2: classify candidates ----
    candidate_f2p: list[str] = []
    candidate_p2p: list[str] = []

    for t, status in after_test_status.items():
        prev = baseline_status.get(t, "NONE")
        if prev == "NONE" or (prev == "PASS" and status == "FAIL"):
            # newly existing OR was-PASS-now-FAIL → expected to flip back under fix_patch
            candidate_f2p.append(t)
        elif prev == "PASS" and status == "PASS":
            candidate_p2p.append(t)

    # ---- Step 3: apply fix_patch on top, re-run ----
    apply_res = sandbox.apply_patch(fix_patch)
    if apply_res.get("returncode", 0) != 0:
        return VerifierConstructionResult(
            instance_id=iid,
            enriched_instance=None,
            success=False,
            reason=f"fix_patch apply failed: {apply_res.get('output', '')[:500]}",
            candidate_f2p_count=len(candidate_f2p),
            candidate_p2p_count=len(candidate_p2p),
            timings=timings,
        )
    raw_after_fix = runner.run_tests(sandbox, test_filter=None, timeout_s=test_command_timeout)
    timings["fix_patch_s"] = raw_after_fix.elapsed_s
    after_fix = runner.parse(raw_after_fix)
    after_fix_status: dict[str, str] = {}
    for t in after_fix.passed:
        after_fix_status[t] = "PASS"
    for t in after_fix.failed:
        after_fix_status[t] = "FAIL"
    for t in after_fix.skipped:
        after_fix_status[t] = "SKIP"

    # ---- Step 4: confirm and emit ----
    confirmed_f2p: list[str] = [t for t in candidate_f2p if after_fix_status.get(t) == "PASS"]
    confirmed_p2p: list[str] = [t for t in candidate_p2p if after_fix_status.get(t) == "PASS"]

    if not confirmed_f2p:
        return VerifierConstructionResult(
            instance_id=iid,
            enriched_instance=None,
            success=False,
            reason="zero F2P candidates passed under fix_patch — task is broken or fix_patch incomplete",
            candidate_f2p_count=len(candidate_f2p),
            candidate_p2p_count=len(candidate_p2p),
            timings=timings,
        )

    f2p_dict = {
        t: _milo_inner_test_record(
            run=baseline_status.get(t, "NONE"),
            test=after_test_status.get(t, "NONE"),
            fix=after_fix_status.get(t, "NONE"),
        )
        for t in confirmed_f2p
    }
    p2p_dict = {
        t: _milo_inner_test_record(
            run=baseline_status.get(t, "NONE"),
            test=after_test_status.get(t, "NONE"),
            fix=after_fix_status.get(t, "NONE"),
        )
        for t in confirmed_p2p
    }

    enriched = dict(instance)
    enriched["f2p_tests"] = f2p_dict
    enriched["p2p_tests"] = p2p_dict
    enriched["verifier_synthesized"] = True

    return VerifierConstructionResult(
        instance_id=iid,
        enriched_instance=enriched,
        success=True,
        reason="ok",
        candidate_f2p_count=len(candidate_f2p),
        candidate_p2p_count=len(candidate_p2p),
        confirmed_f2p_count=len(confirmed_f2p),
        confirmed_p2p_count=len(confirmed_p2p),
        timings=timings,
    )
