"""Phase 6 acceptance tests — invariants I-1..I-8 fire as documented.

Each test exercises one of the 8 spec §7 invariants against a stub
`VerifierReport` / `instance` shape that matches the agent-written
checks.py signatures. See `RL_GYM_SPEC.md` v0.7 §7 for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from milo.invariants.checks import (
    check_i_1,
    check_i_2,
    check_i_3,
    check_i_4,
    check_i_5,
    check_i_6,
    check_i_7,
    check_i_8,
)
from milo.invariants.runner import run_all_invariants


# ---- Stubs that match the agent's check_i_* expected shapes ----


@dataclass
class _StubTestRun:
    """Mirrors verifier 'fix_patch_run' / 'test_patch_run' attribute shape."""
    elapsed_s: float = 1.0


@dataclass
class _StubVerifierReport:
    """Mirrors what milo/invariants/checks.py:check_i_2/3/8 read off the report."""
    # I-2
    f2p_passed: list[str] = field(default_factory=list)
    p2p_tests: list[str] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    # I-3 (the agent's check_i_3 reads `p2p_failed` OR derives from p2p_tests - passing_tests)
    p2p_failed: list[str] = field(default_factory=list)
    passing_tests: list[str] = field(default_factory=list)
    # I-8
    fix_patch_run: _StubTestRun = field(default_factory=_StubTestRun)
    test_patch_run: _StubTestRun = field(default_factory=_StubTestRun)


_PY_INSTANCE: dict[str, Any] = {
    "instance_id": "stub__stub-1",
    "lang": "python",
    "base": {"sha": "deadbeef"},
    "f2p_tests": {"tests/test_x.py::test_a": "{}"},
    "p2p_tests": {"tests/test_y.py::test_b": "{}"},
}


# ---------- I-1: edit a test file ----------
def test_i_1_test_path_edit_rejected() -> None:
    patch = (
        "diff --git a/tests/test_runners.py b/tests/test_runners.py\n"
        "--- a/tests/test_runners.py\n+++ b/tests/test_runners.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-def test_foo():\n+def test_foo_HAHA_DELETED():\n     pass\n"
    )
    v = check_i_1(patch, _PY_INSTANCE, None)
    assert v is not None
    assert v.code == "I-1"


# ---------- I-2: F2P passed + P2P present + runner ran ----------
def test_i_2_no_report_rejected() -> None:
    v = check_i_2("", _PY_INSTANCE, None)
    assert v is not None
    assert v.code == "I-2"


def test_i_2_all_three_clauses_pass() -> None:
    rep = _StubVerifierReport(
        f2p_passed=["tests/test_x.py::test_a"],
        p2p_tests=["tests/test_y.py::test_b"],
        passed_count=2,
    )
    assert check_i_2("", _PY_INSTANCE, rep) is None


def test_i_2_empty_f2p_passed_rejected() -> None:
    rep = _StubVerifierReport(
        f2p_passed=[],
        p2p_tests=["tests/test_y.py::test_b"],
        passed_count=1,
    )
    v = check_i_2("", _PY_INSTANCE, rep)
    assert v is not None
    assert v.code == "I-2"


def test_i_2_zero_runner_count_rejected() -> None:
    rep = _StubVerifierReport(
        f2p_passed=["tests/test_x.py::test_a"],
        p2p_tests=["tests/test_y.py::test_b"],
        passed_count=0, failed_count=0, skipped_count=0,
    )
    v = check_i_2("", _PY_INSTANCE, rep)
    assert v is not None
    assert v.code == "I-2"


# ---------- I-3: P2P regression ----------
def test_i_3_p2p_regression_rejected() -> None:
    rep = _StubVerifierReport(p2p_failed=["tests/test_y.py::test_b"])
    v = check_i_3("", _PY_INSTANCE, rep)
    assert v is not None
    assert v.code == "I-3"


# ---------- I-4: run_command cwd escape ----------
def test_i_4_cwd_escape_via_tool_calls() -> None:
    inst = {**_PY_INSTANCE, "tool_calls": [
        {"name": "run_command", "parameters": {"cmd": "ls", "cwd": "/etc"}},
    ]}
    v = check_i_4("", inst, None)
    assert v is not None
    assert v.code == "I-4"


def test_i_4_no_run_commands_returns_none() -> None:
    assert check_i_4("", _PY_INSTANCE, None) is None


# ---------- I-5: forbidden write paths ----------
def test_i_5_diff_writes_to_verifier_substring_rejected() -> None:
    """Path containing 'verifier' substring tripps I-5 even relative."""
    patch = (
        "diff --git a/src/verifier_hack.py b/src/verifier_hack.py\n"
        "--- a/src/verifier_hack.py\n+++ b/src/verifier_hack.py\n"
        "@@ -1,1 +1,1 @@\n-a\n+b\n"
    )
    v = check_i_5(patch, _PY_INSTANCE, None)
    assert v is not None
    assert v.code == "I-5"


def test_i_5_run_command_redirect_to_forbidden_rejected() -> None:
    inst = {**_PY_INSTANCE, "tool_calls": [
        {"name": "run_command", "parameters": {"cmd": "echo x > /etc/foo"}},
    ]}
    v = check_i_5("", inst, None)
    assert v is not None
    assert v.code == "I-5"


# ---------- I-6: test-fixture tampering (structural) ----------
def test_i_6_conftest_edit_rejected() -> None:
    patch = (
        "diff --git a/tests/conftest.py b/tests/conftest.py\n"
        "--- a/tests/conftest.py\n+++ b/tests/conftest.py\n"
        "@@ -1,1 +1,1 @@\n-x = 1\n+x = 999\n"
    )
    v = check_i_6(patch, _PY_INSTANCE, None)
    assert v is not None
    assert v.code == "I-6"


def test_i_6_fixture_dir_rejected() -> None:
    patch = (
        "diff --git a/tests/fixtures/golden.json b/tests/fixtures/golden.json\n"
        "--- a/tests/fixtures/golden.json\n+++ b/tests/fixtures/golden.json\n"
        "@@ -1,1 +1,1 @@\n-real\n+fake\n"
    )
    v = check_i_6(patch, _PY_INSTANCE, None)
    assert v is not None
    assert v.code == "I-6"


# ---------- I-7: git apply --check ----------
def test_i_7_without_base_dir_returns_none() -> None:
    """Per the agent's check_i_7: returns None when no base_dir is provided
    so the trainer-side caller doesn't spuriously block (the gym-side
    verifier runs `git apply --check` inside the sandbox instead)."""
    assert check_i_7("diff --git a/x b/x\n", _PY_INSTANCE, None) is None


def test_i_7_empty_patch_passes() -> None:
    assert check_i_7("", _PY_INSTANCE, None) is None


# ---------- I-8: runtime-cost bound ----------
def test_i_8_slow_fix_run_rejected() -> None:
    rep = _StubVerifierReport(
        test_patch_run=_StubTestRun(elapsed_s=10.0),
        fix_patch_run=_StubTestRun(elapsed_s=1000.0),
    )
    v = check_i_8("", _PY_INSTANCE, rep)
    assert v is not None
    assert v.code == "I-8"


def test_i_8_normal_fix_run_passes() -> None:
    rep = _StubVerifierReport(
        test_patch_run=_StubTestRun(elapsed_s=10.0),
        fix_patch_run=_StubTestRun(elapsed_s=20.0),
    )
    assert check_i_8("", _PY_INSTANCE, rep) is None


# ---------- Runner orchestration ----------
def test_runner_clean_patch_passes() -> None:
    """Patch that touches no test files + verifier report ok across I-2/I-3/I-8."""
    patch = (
        "diff --git a/src/main.py b/src/main.py\n--- a/src/main.py\n+++ b/src/main.py\n"
        "@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
    )
    rep = _StubVerifierReport(
        f2p_passed=["tests/test_x.py::test_a"],
        p2p_tests=["tests/test_y.py::test_b"],
        p2p_failed=[],
        passing_tests=["tests/test_y.py::test_b"],   # so derived p2p_failed is empty
        passed_count=2,
        test_patch_run=_StubTestRun(elapsed_s=5.0),
        fix_patch_run=_StubTestRun(elapsed_s=6.0),
    )
    res = run_all_invariants(patch, _PY_INSTANCE, rep)
    assert res.passed is True, f"violations: {[(v.code, v.message) for v in res.violations]}"


def test_runner_collects_multiple_violations() -> None:
    """A patch touching a 'verifier' substring path (I-5) AND a conftest (I-6)."""
    patch = (
        "diff --git a/src/verifier_hack.py b/src/verifier_hack.py\n"
        "--- a/src/verifier_hack.py\n+++ b/src/verifier_hack.py\n"
        "@@ -1,1 +1,1 @@\n-a\n+b\n"
        "diff --git a/tests/conftest.py b/tests/conftest.py\n"
        "--- a/tests/conftest.py\n+++ b/tests/conftest.py\n"
        "@@ -1,1 +1,1 @@\n-a\n+b\n"
    )
    res = run_all_invariants(patch, _PY_INSTANCE, None)
    assert res.passed is False
    codes = {v.code for v in res.violations}
    assert "I-5" in codes
    assert "I-6" in codes
