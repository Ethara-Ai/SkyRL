"""Phase 11.8 verifier-construction unit tests with a mocked sandbox + runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from milo.lht_pipeline.verifier_construction import construct_verifier
from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun


# ---- Fakes ----


@dataclass
class _FakeSandbox:
    """Records the call sequence and returns scripted responses."""

    apply_calls: list[str] = field(default_factory=list)
    reset_calls: list[str] = field(default_factory=list)
    apply_returncodes: list[int] = field(default_factory=lambda: [0, 0])

    def exec(self, cmd: str, timeout: int = 180, cwd: str | None = None) -> dict[str, Any]:
        return {"returncode": 0, "output": ""}

    def apply_patch(self, diff: str) -> dict[str, Any]:
        self.apply_calls.append(diff)
        idx = len(self.apply_calls) - 1
        rc = self.apply_returncodes[idx] if idx < len(self.apply_returncodes) else 0
        return {"returncode": rc, "output": "applied" if rc == 0 else "patch failed"}

    def reset_to(self, sha: str) -> None:
        self.reset_calls.append(sha)


@dataclass
class _ScriptedRunner:
    """Returns a queue of pre-canned TestResults across 3 sequential calls."""

    LANG: str = "python"
    scripted: list[TestResult] = field(default_factory=list)
    raw_index: int = 0

    def install_deps(self, sandbox: Any) -> None:
        pass

    def run_tests(self, sandbox: Any, test_filter: Any, timeout_s: int) -> RawTestRun:
        return RawTestRun(stdout="", stderr="", exit_code=0, elapsed_s=1.0, extras={})

    def parse(self, raw: RawTestRun) -> TestResult:
        result = self.scripted[self.raw_index]
        self.raw_index += 1
        return result


def _tr(passed: list[str] = [], failed: list[str] = [], skipped: list[str] = []) -> TestResult:
    return TestResult(
        passed=list(passed),
        failed=list(failed),
        skipped=list(skipped),
        elapsed_s=1.0,
        exit_code=0,
    )


# ---- Tests ----


def test_construct_success_emits_enriched_instance() -> None:
    instance = {
        "instance_id": "stub__stub-1",
        "lang": "python",
        "base": {"sha": "abc"},
        "test_patch": "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        "fix_patch": "diff --git a/y b/y\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-a\n+b\n",
    }
    sandbox = _FakeSandbox()
    runner = _ScriptedRunner(
        scripted=[
            _tr(passed=["existing_p2p"]),                                          # baseline
            _tr(passed=["existing_p2p"], failed=["new_f2p"]),                      # +test_patch
            _tr(passed=["existing_p2p", "new_f2p"]),                               # +fix_patch
        ]
    )
    res = construct_verifier(instance, sandbox, runner)
    assert res.success is True
    assert res.confirmed_f2p_count == 1
    assert res.confirmed_p2p_count == 1
    enriched = res.enriched_instance
    assert enriched is not None
    assert "new_f2p" in enriched["f2p_tests"]
    assert "existing_p2p" in enriched["p2p_tests"]
    assert enriched["verifier_synthesized"] is True
    # Inner-JSON shape preserved
    inner = json.loads(enriched["f2p_tests"]["new_f2p"])
    assert inner == {"run": "NONE", "test": "FAIL", "fix": "PASS"}


def test_construct_no_f2p_candidates_returns_none() -> None:
    """fix_patch doesn't flip any FAIL → PASS → no F2P → reconstruction fails."""
    instance = {
        "instance_id": "stub__stub-2",
        "lang": "python",
        "base": {"sha": "abc"},
        "test_patch": "diff\n",
        "fix_patch": "diff\n",
    }
    sandbox = _FakeSandbox()
    runner = _ScriptedRunner(
        scripted=[_tr(passed=["x"]), _tr(passed=["x"]), _tr(passed=["x"])]
    )
    res = construct_verifier(instance, sandbox, runner)
    assert res.success is False
    assert res.enriched_instance is None
    assert "zero F2P candidates" in res.reason


def test_construct_test_patch_apply_failure() -> None:
    instance = {
        "instance_id": "stub__stub-3",
        "lang": "python",
        "base": {"sha": "abc"},
        "test_patch": "broken\n",
        "fix_patch": "diff\n",
    }
    sandbox = _FakeSandbox(apply_returncodes=[1, 0])
    runner = _ScriptedRunner(scripted=[_tr(), _tr(), _tr()])
    res = construct_verifier(instance, sandbox, runner)
    assert res.success is False
    assert "test_patch apply failed" in res.reason


def test_construct_skips_when_patches_missing() -> None:
    instance = {
        "instance_id": "stub__stub-4",
        "lang": "python",
        "base": {"sha": "abc"},
        "test_patch": "",
        "fix_patch": "diff\n",
    }
    sandbox = _FakeSandbox()
    runner = _ScriptedRunner(scripted=[])
    res = construct_verifier(instance, sandbox, runner)
    assert res.success is False
    assert "missing test_patch" in res.reason
