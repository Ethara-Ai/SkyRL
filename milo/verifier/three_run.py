"""ThreeRunVerifier — the 3-Docker-run pattern.

Implements:
    - IMPLEMENTATION_PLAN.md v0.4 §2.1 (the 3-Docker-run pattern).
    - RL_GYM_SPEC.md v0.7 §5.2 (verifier subsystem).
    - RL_GYM_SPEC.md v0.7 §4.4.1 (R_terminal formula — delegated to
      VerifierReport.r_terminal()).
    - RL_GYM_SPEC.md v0.7 §7 (I-1..I-8 invariants — delegated to
      milo.invariants.checks.run_all_invariants).

Three runs:
    (1) baseline    — no patches. CACHED per instance (deterministic given
                      the Docker image + tag_start).
    (2) test_patch  — held-out test patch only. CACHED per instance.
    (3) fix_patch   — agent's candidate patch + held-out test patch. Per
                      rollout (changes per agent attempt).

Caching is via JSON files at:
    <cache_dir>/<instance_id>_baseline.json
    <cache_dir>/<instance_id>_test_patch.json

The fix_patch result is NEVER cached (different per rollout).

Sandbox factory:
    The verifier doesn't own sandbox lifecycle. It accepts a `sandbox_factory`
    callable (no-args; returns a Sandbox context manager) so it can spin up a
    fresh sandbox per Docker run without coupling to the Phase 1 docker_runtime
    module directly. Production uses `lambda: milo_sandbox(instance, ...)`;
    tests use a fake.

The Phase 6 invariants module may not be present yet (Phase 6 ships after
Phase 2 per the parallel-track plan). We import via try/except with a
permissive fallback so this file is import-safe in isolation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from milo.verifier.report import TestResult, VerifierReport
from milo.verifier.runners.base import PerLanguageTestRunner

_log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Phase 6 invariants module — optional, with permissive fallback.
# ----------------------------------------------------------------------
try:
    from milo.invariants.checks import (  # type: ignore[import-not-found]
        InvariantResult,
        run_all_invariants,
    )
except ImportError:  # pragma: no cover — exercised in real env once Phase 6 lands
    @dataclass(frozen=False)
    class InvariantResult:  # type: ignore[no-redef]
        """Permissive fallback when milo.invariants.checks is not yet importable.

        Returns "all good" so the rest of the verifier still functions in
        isolation. Phase 6 must replace this with the real implementation
        before any reward signal is trusted.
        """

        passed: bool = True
        violations: list[str] = field(default_factory=list)

    def run_all_invariants(*args: Any, **kwargs: Any) -> InvariantResult:  # type: ignore[no-redef]
        _log.warning(
            "ThreeRunVerifier: milo.invariants.checks not available; "
            "running with permissive fallback (no invariant enforcement). "
            "Phase 6 must be present before reward signal is trusted."
        )
        return InvariantResult(passed=True, violations=[])


# ----------------------------------------------------------------------
# Decorator: cache the per-instance TestResult to a JSON file on first call.
# ----------------------------------------------------------------------
def _cached_run(cache_filename_suffix: str) -> Callable:
    """Decorator factory: cache a method's TestResult to JSON.

    Cache key is `<cache_dir>/<instance_id>_<suffix>.json`. The decorated
    method must take only `self`.
    """

    def decorator(method: Callable[..., TestResult]) -> Callable[..., TestResult]:
        def wrapped(self: ThreeRunVerifier) -> TestResult:
            cache_path = self._cache_path(cache_filename_suffix)
            if cache_path.exists():
                try:
                    with cache_path.open("r") as f:
                        data = json.load(f)
                    cached = TestResult.from_dict(data)
                    _log.debug("cache HIT: %s", cache_path)
                    return cached
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    _log.warning(
                        "cache READ failure (%s): %s; recomputing", cache_path, e
                    )
            _log.debug("cache MISS: %s", cache_path)
            result = method(self)
            # Atomic write via temp + rename.
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with tmp.open("w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                tmp.replace(cache_path)
            except OSError as e:
                _log.warning("cache WRITE failure (%s): %s", cache_path, e)
            return result

        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        return wrapped

    return decorator


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class ThreeRunVerifier:
    """3-Docker-run executor per spec §5.2.

    Lifecycle (typical):
        v = ThreeRunVerifier(instance, runner, cache_dir, sandbox_factory)
        # Cheap if cached, expensive otherwise:
        baseline = v.baseline()
        test_patch = v.test_patch()
        # Per-rollout (always expensive):
        report = v.report(candidate_patch)
        r_term = report.r_terminal()

    Args:
        instance:         the milo-bench instance dict (see milo/audit for the
                          on-disk schema). At minimum we read `instance_id`,
                          `fix_patch`, `test_patch`, `f2p_tests`, `p2p_tests`.
        runner:           the language-specific test runner per Phase 2.2.
                          Use `milo.verifier.runners.registry.get_runner(lang)`.
        cache_dir:        where to persist baseline/test_patch results between
                          invocations. Atomic JSON writes.
        sandbox_factory:  zero-arg callable returning a Sandbox context
                          manager (per Phase 1's milo/lht_adapter/docker_runtime.py
                          `milo_sandbox(instance)` contract). Each Docker run
                          gets a fresh sandbox via `with sandbox_factory() as s:`.
        test_timeout_s:   per-run timeout. Default 1800s, matching the spec
                          §4.5 max_episode_seconds.
    """

    DEFAULT_TIMEOUT_S = 1800

    def __init__(
        self,
        instance: dict[str, Any],
        runner: PerLanguageTestRunner,
        cache_dir: Path,
        sandbox_factory: Callable[[], Any],
        test_timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        if "instance_id" not in instance:
            raise ValueError("instance must have 'instance_id'")
        self._instance: dict[str, Any] = instance
        self._runner: PerLanguageTestRunner = runner
        self._cache_dir: Path = Path(cache_dir)
        self._sandbox_factory: Callable[[], Any] = sandbox_factory
        self._test_timeout_s: int = int(test_timeout_s)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def instance_id(self) -> str:
        return str(self._instance["instance_id"])

    @property
    def fix_patch_str(self) -> str:
        return str(self._instance.get("fix_patch", "") or "")

    @property
    def test_patch_str(self) -> str:
        return str(self._instance.get("test_patch", "") or "")

    def _cache_path(self, suffix: str) -> Path:
        return self._cache_dir / f"{self.instance_id}_{suffix}.json"

    # ------------------------------------------------------------------
    # The three runs.
    # ------------------------------------------------------------------
    @_cached_run("baseline")
    def baseline(self) -> TestResult:
        """Run #1: no patches. Deterministic per instance; cached."""
        with self._open_sandbox() as sandbox:
            self._runner.install_deps(sandbox)
            raw = self._runner.run_tests(
                sandbox, test_filter=None, timeout_s=self._test_timeout_s
            )
            return self._runner.parse(raw)

    @_cached_run("test_patch")
    def test_patch(self) -> TestResult:
        """Run #2: test_patch only. Cached per instance.

        Applies the held-out test patch first, then runs the test suite.
        The test patch should introduce tests that fail without the fix —
        in baseline they don't exist; in test_patch they exist and fail; in
        fix_patch (run #3) they exist and pass.
        """
        with self._open_sandbox() as sandbox:
            self._runner.install_deps(sandbox)
            self._apply_patch(sandbox, self.test_patch_str, label="test_patch")
            raw = self._runner.run_tests(
                sandbox, test_filter=None, timeout_s=self._test_timeout_s
            )
            return self._runner.parse(raw)

    def fix_patch(self, candidate_patch: str) -> TestResult:
        """Run #3: candidate fix + test_patch. Per-rollout; never cached.

        Args:
            candidate_patch: the agent's git diff. Applied AFTER test_patch
                so the test patch's new tests are present during the run.
        """
        with self._open_sandbox() as sandbox:
            self._runner.install_deps(sandbox)
            # Apply test_patch first (the verifier's mandated test set) then
            # the candidate fix on top. If the candidate touches test files
            # (forbidden — caught later by I-1), it may conflict; we surface
            # any apply failure as a TestResult with no passes/failures.
            try:
                self._apply_patch(sandbox, self.test_patch_str, label="test_patch")
                self._apply_patch(sandbox, candidate_patch, label="fix_patch")
            except _PatchApplyError as e:
                _log.info("fix_patch apply failed: %s", e)
                return TestResult(
                    passed=[],
                    failed=[],
                    skipped=[],
                    elapsed_s=0.0,
                    exit_code=1,
                    raw_log_path=None,
                )
            raw = self._runner.run_tests(
                sandbox, test_filter=None, timeout_s=self._test_timeout_s
            )
            return self._runner.parse(raw)

    # ------------------------------------------------------------------
    # The composed report (the artefact consumers actually want).
    # ------------------------------------------------------------------
    def report(self, candidate_patch: str) -> VerifierReport:
        """Compose all three runs into a VerifierReport, including the
        §7 invariant checks.

        Order of operations:
            1. baseline()       — cheap if cached.
            2. test_patch()     — cheap if cached.
            3. fix_patch(...)   — always expensive (one Docker run).
            4. Compute F2P pass/fail and P2P regression sets from the
               expected f2p/p2p test lists in `instance`.
            5. Run all I-1..I-8 invariants (Phase 6).
            6. Materialize VerifierReport with `passes_invariant_check`
               and the violation list.

        The VerifierReport itself computes R_terminal via its `r_terminal()`
        method per spec §4.4.1. Reward composition (alpha/beta/lambda/gamma)
        is the Phase 4 reward aggregator's job, not ours.
        """
        t0 = time.perf_counter()

        baseline = self.baseline()
        tp = self.test_patch()
        fp = self.fix_patch(candidate_patch)

        # ------------------------------------------------------------------
        # F2P / P2P classification
        # ------------------------------------------------------------------
        expected_f2p = _extract_expected_test_ids(self._instance.get("f2p_tests"))
        expected_p2p = _extract_expected_test_ids(self._instance.get("p2p_tests"))

        fp_passed_set = set(fp.passed)
        fp_failed_set = set(fp.failed)
        baseline_passed_set = set(baseline.passed)

        f2p_passed = sorted(fp_passed_set & expected_f2p)
        f2p_failed = sorted(expected_f2p - fp_passed_set)
        # A P2P regression = was passing in baseline AND now failing in
        # fix_patch. (Per spec §7 I-3.)
        p2p_regressed = sorted(
            (expected_p2p & baseline_passed_set) & fp_failed_set
        )

        test_count_nonzero = fp.total_count > 0

        # ------------------------------------------------------------------
        # Invariant checks
        # ------------------------------------------------------------------
        # Phase 6's run_all_invariants takes the instance, the candidate
        # patch, and the three TestResults; returns aggregate pass/fail.
        # We pass everything by keyword so future signature evolution
        # doesn't break us.
        inv = run_all_invariants(
            instance=self._instance,
            candidate_patch=candidate_patch,
            baseline_result=baseline,
            test_patch_result=tp,
            fix_patch_result=fp,
            expected_f2p=expected_f2p,
            expected_p2p=expected_p2p,
        )
        passes_invariant_check = bool(getattr(inv, "passed", True))
        invariant_violations: list[str] = list(getattr(inv, "violations", []) or [])

        elapsed_total_s = (
            baseline.elapsed_s + tp.elapsed_s + fp.elapsed_s
        )
        # Also include the wall-clock spent in this method (cache hits are
        # very fast; this just adds the orchestration time).
        elapsed_total_s = max(elapsed_total_s, time.perf_counter() - t0)

        report = VerifierReport(
            instance_id=self.instance_id,
            baseline_result=baseline,
            test_patch_result=tp,
            fix_patch_result=fp,
            f2p_passed=f2p_passed,
            f2p_failed=f2p_failed,
            p2p_regressed=p2p_regressed,
            test_count_nonzero=test_count_nonzero,
            passes_invariant_check=passes_invariant_check,
            invariant_violations=invariant_violations,
            elapsed_total_s=elapsed_total_s,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _open_sandbox(self):
        """Open a fresh sandbox via the configured factory. The factory may
        return either a context manager or a bare object — we accept both."""
        cm = self._sandbox_factory()
        if hasattr(cm, "__enter__") and hasattr(cm, "__exit__"):
            with cm as sandbox:
                yield sandbox
        else:
            # Bare object — caller didn't wrap in a CM. Yield directly.
            try:
                yield cm
            finally:
                close = getattr(cm, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        _log.exception("sandbox.close() raised; continuing")

    def _apply_patch(self, sandbox: Any, patch_str: str, label: str) -> None:
        """Apply a git-format patch inside the sandbox via heredoc.

        We mirror the pattern from
        examples/train/mini_swe_agent/mini_swe_utils.py: `evaluate_trajectory()`
        — heredoc with a UUID-marker delimiter to avoid collisions with patch
        content.

        Raises:
            _PatchApplyError: if `git apply` returns non-zero.
        """
        if not patch_str.strip():
            return  # Empty patch is a no-op.
        import uuid

        delimiter = f"PATCH_{uuid.uuid4().hex}"
        cmd = f"git apply <<'{delimiter}'\n{patch_str}\n{delimiter}"
        result = sandbox.exec(cmd, timeout=300, cwd="/workspace")
        if isinstance(result, dict):
            rc = result.get("exit_code")
            if rc is None:
                rc = result.get("returncode", 0)
            stderr = result.get("stderr") or result.get("output") or ""
        else:
            rc = 0
            stderr = ""
        if int(rc or 0) != 0:
            raise _PatchApplyError(
                f"git apply ({label}) failed with exit={rc}: {stderr[:500]}"
            )


# ----------------------------------------------------------------------
# Module-private exception + helpers
# ----------------------------------------------------------------------
class _PatchApplyError(RuntimeError):
    """Raised when `git apply` returns non-zero inside the sandbox."""


def _extract_expected_test_ids(value: Any) -> set[str]:
    """Pull the canonical test IDs from a milo-bench `f2p_tests` / `p2p_tests`
    field.

    On-disk shape (see milo/audit/audit_dataset.py SCHEMA_MAPPING notes):

        Field is a dict whose KEYS are test IDs and whose VALUES are either
        a dict `{"run": <status>, "test": <status>, "fix": <status>}` OR a
        string containing the JSON of that dict (double-encoded).

        We only care about the KEYS — the expected test IDs.

    Fallback shapes we also accept:
        * list[str] — already-canonical IDs.
        * None / empty → empty set.
    """
    if not value:
        return set()
    if isinstance(value, dict):
        return {str(k) for k in value}
    if isinstance(value, list):
        return {str(t) for t in value}
    return set()
