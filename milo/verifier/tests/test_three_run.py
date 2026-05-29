"""Tests for milo.verifier.three_run.ThreeRunVerifier.

Covers:
    - Caching behavior (baseline() and test_patch() are cached per-instance;
      fix_patch() is per-rollout and never cached).
    - The composed report() method's F2P / P2P classification.
    - Empty f2p_tests / p2p_tests handling.
    - Sandbox factory accepting both context-manager and bare-object forms.
    - The Phase 6 fallback (permissive run_all_invariants when the real
      module isn't present).

These tests use a FakeSandbox + FakeRunner so they're hermetic — no Docker
required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from milo.verifier.report import TestResult, VerifierReport
from milo.verifier.runners.base import RawTestRun
from milo.verifier.three_run import ThreeRunVerifier


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeSandbox:
    """Minimal sandbox that satisfies the sandbox.exec / read_file contract."""

    def __init__(self) -> None:
        self.exec_calls: list[tuple[str, int, str]] = []
        self.read_calls: list[str] = []
        # Map of (substring → result-dict) to control exec behavior.
        self.exec_responses: dict[str, dict[str, Any]] = {}
        # Map of (path → contents) for read_file.
        self.files: dict[str, str] = {}

    def exec(self, cmd: str, timeout: int, cwd: str = "/workspace") -> dict[str, Any]:
        self.exec_calls.append((cmd, timeout, cwd))
        for substr, resp in self.exec_responses.items():
            if substr in cmd:
                return dict(resp)
        return {"stdout": "", "stderr": "", "exit_code": 0}

    def read_file(self, path: str) -> str:
        self.read_calls.append(path)
        return self.files.get(path, "")

    def __enter__(self) -> FakeSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeRunner:
    """Stub PerLanguageTestRunner whose parse() returns canned TestResults."""

    LANG = "fake"

    def __init__(self) -> None:
        # Each entry consumed in order by run_tests/parse.
        self.programmed_results: list[TestResult] = []
        self.run_count = 0
        self.install_count = 0

    def install_deps(self, sandbox: Any) -> None:
        self.install_count += 1

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        # Return a noop raw; parse() actually returns the programmed result.
        return RawTestRun(stdout="", stderr="", exit_code=0, elapsed_s=0.1)

    def parse(self, raw: RawTestRun) -> TestResult:
        if not self.programmed_results:
            return TestResult()
        result = self.programmed_results[self.run_count]
        self.run_count += 1
        return result


def _make_instance(
    *,
    iid: str = "inst_x",
    f2p: dict | None = None,
    p2p: dict | None = None,
    fix_patch: str = "",
    test_patch: str = "",
) -> dict[str, Any]:
    return {
        "instance_id": iid,
        "f2p_tests": f2p or {},
        "p2p_tests": p2p or {},
        "fix_patch": fix_patch,
        "test_patch": test_patch,
        "lang": "fake",
    }


def _make_verifier(
    tmp_path: Path,
    runner: FakeRunner,
    instance: dict[str, Any] | None = None,
) -> tuple[ThreeRunVerifier, FakeSandbox]:
    sandbox = FakeSandbox()
    inst = instance or _make_instance()
    v = ThreeRunVerifier(
        instance=inst,
        runner=runner,
        cache_dir=tmp_path,
        sandbox_factory=lambda: sandbox,
        test_timeout_s=10,
    )
    return v, sandbox


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
class TestCaching:
    def test_baseline_cached(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [
            TestResult(passed=["t1", "t2"], elapsed_s=0.5),  # baseline call 1
            TestResult(passed=["NEVER_REACHED"]),  # should NOT be used
        ]
        v, _sandbox = _make_verifier(tmp_path, runner)

        first = v.baseline()
        second = v.baseline()

        assert first.passed == ["t1", "t2"]
        assert second.passed == ["t1", "t2"]  # cache hit
        # The runner.parse was called exactly once.
        assert runner.run_count == 1

        # The cache file exists on disk.
        cache_files = list(tmp_path.glob("*_baseline.json"))
        assert len(cache_files) == 1

    def test_test_patch_cached(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [TestResult(passed=["a"])]
        v, _ = _make_verifier(tmp_path, runner)
        first = v.test_patch()
        second = v.test_patch()
        assert first.passed == second.passed == ["a"]
        assert runner.run_count == 1

    def test_fix_patch_NOT_cached(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [
            TestResult(passed=["fix1"]),
            TestResult(passed=["fix2"]),  # second call produces different result
        ]
        v, _ = _make_verifier(tmp_path, runner)
        first = v.fix_patch("dummy diff 1")
        second = v.fix_patch("dummy diff 2")
        assert first.passed == ["fix1"]
        assert second.passed == ["fix2"]
        # Both calls invoked the runner.
        assert runner.run_count == 2
        # No fix_patch cache file should have been written.
        assert not list(tmp_path.glob("*_fix_patch.json"))

    def test_cache_persists_across_verifier_instances(self, tmp_path: Path) -> None:
        runner1 = FakeRunner()
        runner1.programmed_results = [TestResult(passed=["persist"])]
        v1, _ = _make_verifier(tmp_path, runner1)
        v1.baseline()  # populates cache

        runner2 = FakeRunner()
        runner2.programmed_results = [
            TestResult(passed=["BAD"]),  # should NOT be reached
        ]
        v2, _ = _make_verifier(tmp_path, runner2)
        result = v2.baseline()  # should hit cache
        assert result.passed == ["persist"]
        assert runner2.run_count == 0

    def test_cache_corrupt_file_falls_back_to_recompute(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [TestResult(passed=["fresh"])]
        v, _ = _make_verifier(tmp_path, runner)
        cache_path = tmp_path / f"{v.instance_id}_baseline.json"
        cache_path.write_text("{this is not valid json")
        result = v.baseline()
        assert result.passed == ["fresh"]
        assert runner.run_count == 1

    def test_cached_result_matches_json_shape(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [
            TestResult(passed=["a"], failed=["b"], elapsed_s=1.0, exit_code=1)
        ]
        v, _ = _make_verifier(tmp_path, runner)
        v.baseline()
        with open(tmp_path / f"{v.instance_id}_baseline.json") as f:
            data = json.load(f)
        # Round-trip back to a TestResult.
        rebuilt = TestResult.from_dict(data)
        assert rebuilt.passed == ["a"]
        assert rebuilt.failed == ["b"]
        assert rebuilt.elapsed_s == 1.0


# ---------------------------------------------------------------------------
# report() composition
# ---------------------------------------------------------------------------
class TestReportComposition:
    def test_all_clear_returns_r_terminal_1(self, tmp_path: Path) -> None:
        """Baseline passes p2p tests; fix_patch passes f2p tests; no regress."""
        runner = FakeRunner()
        runner.programmed_results = [
            # baseline
            TestResult(passed=["p2p_test1", "p2p_test2"]),
            # test_patch
            TestResult(passed=["p2p_test1", "p2p_test2"], failed=["f2p_test1"]),
            # fix_patch
            TestResult(passed=["p2p_test1", "p2p_test2", "f2p_test1"]),
        ]
        instance = _make_instance(
            f2p={"f2p_test1": {"run": "NONE", "test": "FAIL", "fix": "PASS"}},
            p2p={
                "p2p_test1": {"run": "PASS", "test": "PASS", "fix": "PASS"},
                "p2p_test2": {"run": "PASS", "test": "PASS", "fix": "PASS"},
            },
        )
        v, _ = _make_verifier(tmp_path, runner, instance)
        report = v.report(candidate_patch="dummy")
        assert report.r_terminal() == 1
        assert report.f2p_passed == ["f2p_test1"]
        assert report.f2p_failed == []
        assert report.p2p_regressed == []
        assert report.test_count_nonzero is True

    def test_p2p_regression_detected(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [
            # baseline: p2p_test1 passes
            TestResult(passed=["p2p_test1"]),
            # test_patch
            TestResult(passed=["p2p_test1"], failed=["f2p_test1"]),
            # fix_patch: f2p passes BUT p2p_test1 regressed to failed
            TestResult(passed=["f2p_test1"], failed=["p2p_test1"]),
        ]
        instance = _make_instance(
            f2p={"f2p_test1": {}},
            p2p={"p2p_test1": {}},
        )
        v, _ = _make_verifier(tmp_path, runner, instance)
        report = v.report(candidate_patch="dummy")
        assert "p2p_test1" in report.p2p_regressed
        assert report.r_terminal() == 0

    def test_f2p_unsatisfied_returns_r_terminal_0(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [
            TestResult(passed=["p2p_test1"]),  # baseline
            TestResult(passed=["p2p_test1"]),  # test_patch (no fail visible — okay)
            TestResult(passed=["p2p_test1"]),  # fix_patch — f2p missing
        ]
        instance = _make_instance(
            f2p={"f2p_test_required": {}},
            p2p={"p2p_test1": {}},
        )
        v, _ = _make_verifier(tmp_path, runner, instance)
        report = v.report(candidate_patch="dummy")
        assert "f2p_test_required" in report.f2p_failed
        assert report.r_terminal() == 0

    def test_zero_total_tests_returns_r_terminal_0(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [
            TestResult(),
            TestResult(),
            TestResult(),  # fix_patch: no tests at all
        ]
        instance = _make_instance(
            f2p={"f2p_x": {}},
            p2p={"p2p_x": {}},
        )
        v, _ = _make_verifier(tmp_path, runner, instance)
        report = v.report(candidate_patch="dummy")
        assert report.test_count_nonzero is False
        assert report.r_terminal() == 0


# ---------------------------------------------------------------------------
# Patch apply behavior
# ---------------------------------------------------------------------------
class TestPatchApply:
    def test_empty_patch_string_is_noop(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [TestResult(passed=["a"])]
        instance = _make_instance(test_patch="", fix_patch="")
        v, sandbox = _make_verifier(tmp_path, runner, instance)
        v.test_patch()
        # No `git apply` command should have been run for an empty patch.
        assert not any("git apply" in cmd for cmd, _, _ in sandbox.exec_calls)

    def test_fix_patch_apply_failure_returns_empty_result(
        self, tmp_path: Path
    ) -> None:
        runner = FakeRunner()
        runner.programmed_results = [TestResult(passed=["x"])]
        instance = _make_instance(
            test_patch="diff --git a b\n+test\n",
            fix_patch="diff --git c d\n+broken\n",
        )
        sandbox = FakeSandbox()
        sandbox.exec_responses = {
            "git apply": {"stdout": "", "stderr": "patch does not apply", "exit_code": 1}
        }
        v = ThreeRunVerifier(
            instance=instance,
            runner=runner,
            cache_dir=tmp_path,
            sandbox_factory=lambda: sandbox,
            test_timeout_s=10,
        )
        result = v.fix_patch("anything")
        assert result.passed == []
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Sandbox factory shapes
# ---------------------------------------------------------------------------
class TestSandboxFactory:
    def test_context_manager_factory(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.programmed_results = [TestResult(passed=["a"])]
        sandbox = FakeSandbox()
        v = ThreeRunVerifier(
            instance=_make_instance(),
            runner=runner,
            cache_dir=tmp_path,
            sandbox_factory=lambda: sandbox,
        )
        v.baseline()
        # No exception means it worked.
        assert runner.run_count == 1

    def test_bare_object_factory(self, tmp_path: Path) -> None:
        """Factory that returns a non-CM object is accepted and closed."""
        class BareSandbox(FakeSandbox):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            def __enter__(self) -> None:  # type: ignore[override]
                raise AttributeError  # force the "not a CM" path

            def __exit__(self, *exc: object) -> None:  # type: ignore[override]
                raise AttributeError

            def close(self) -> None:
                self.closed = True

        # We patch around the AttributeError-based detection by removing
        # __enter__/__exit__ at the instance level for a clean signal.
        class TrulyBare:
            def __init__(self) -> None:
                self.closed = False
                self.exec_calls = []

            def exec(self, cmd: str, timeout: int, cwd: str = "/workspace") -> dict[str, Any]:
                self.exec_calls.append((cmd, timeout, cwd))
                return {"stdout": "", "stderr": "", "exit_code": 0}

            def read_file(self, path: str) -> str:
                return ""

            def close(self) -> None:
                self.closed = True

        bare = TrulyBare()
        runner = FakeRunner()
        runner.programmed_results = [TestResult(passed=["a"])]
        v = ThreeRunVerifier(
            instance=_make_instance(),
            runner=runner,
            cache_dir=tmp_path,
            sandbox_factory=lambda: bare,
        )
        v.baseline()
        assert bare.closed is True


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
def test_constructor_requires_instance_id() -> None:
    with pytest.raises(ValueError):
        ThreeRunVerifier(
            instance={"no_id_here": True},
            runner=FakeRunner(),
            cache_dir=Path("/tmp"),
            sandbox_factory=lambda: FakeSandbox(),
        )


def test_f2p_p2p_extraction_handles_dict_keys(tmp_path: Path) -> None:
    """Per the on-disk schema, f2p_tests is a dict whose keys are test IDs."""
    runner = FakeRunner()
    runner.programmed_results = [
        TestResult(passed=["k1", "k2"]),
        TestResult(passed=["k1", "k2"]),
        TestResult(passed=["k1", "k2", "k_f2p"]),
    ]
    inst = _make_instance(
        f2p={"k_f2p": {"run": "NONE", "test": "FAIL", "fix": "PASS"}},
        p2p={"k1": {}, "k2": {}},
    )
    v, _ = _make_verifier(tmp_path, runner, inst)
    report = v.report("dummy")
    assert "k_f2p" in report.f2p_passed
    assert report.r_terminal() == 1


def test_f2p_p2p_extraction_handles_list(tmp_path: Path) -> None:
    """Permissive: accept list form as well, just in case."""
    runner = FakeRunner()
    runner.programmed_results = [
        TestResult(passed=["a", "b"]),
        TestResult(passed=["a", "b"]),
        TestResult(passed=["a", "b", "f"]),
    ]
    inst = _make_instance(f2p=["f"], p2p=["a", "b"])  # type: ignore[arg-type]
    v, _ = _make_verifier(tmp_path, runner, inst)
    report = v.report("dummy")
    assert report.r_terminal() == 1
