"""Rust cargo test runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (rust row).

Two-mode runner:

    (1) NIGHTLY (preferred) — `cargo test --no-fail-fast -- --format json
        -Z unstable-options`. Each test outputs one line of JSON:
            {"type":"test","event":"started","name":"crate::module::test_name"}
            {"type":"test","event":"ok","name":"crate::module::test_name"}
            {"type":"test","event":"failed","name":"crate::module::test_name"}
            {"type":"test","event":"ignored","name":"crate::module::test_name"}

    (2) STABLE (fallback) — `cargo test --no-fail-fast` and parse the text
        output. Format:
            test crate::module::test_name ... ok
            test crate::module::test_name ... FAILED
            test crate::module::test_name ... ignored

Test-ID format: `<crate>::<module path>::<test_name>` (matches the
output verbatim).

v0.4 — TODO("real parser in Phase 2.2.next"): the nightly path is a
prefer-on-detect not an assert. If the sandbox lacks nightly toolchain,
we fall back to the stable parser. A future revision should detect this
once at startup and pin the chosen mode.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun, time_exec


class RustCargoRunner:
    """`cargo test`-based runner for Rust milo-bench instances."""

    LANG: ClassVar[str] = "rust"

    def __init__(
        self,
        cargo_bin: str = "cargo",
        prefer_nightly: bool = False,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self._cargo_bin = cargo_bin
        self._prefer_nightly = prefer_nightly
        self._extra_args = tuple(extra_args)

    def install_deps(self, sandbox: Any) -> None:
        """Multi-SWE-bench Rust images pre-fetch with `cargo fetch`. No-op
        by default."""
        return None

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        # Build the cargo invocation.
        cmd_parts: list[str] = [self._cargo_bin, "test", "--no-fail-fast"]
        cmd_parts.extend(self._extra_args)

        # Test filter (positional) goes BEFORE the `--` separator.
        if test_filter:
            for t in test_filter:
                cmd_parts.append(t)

        # Args after `--` go to the test binary itself.
        cmd_parts.append("--")
        if self._prefer_nightly:
            cmd_parts.extend(["--format", "json", "-Z", "unstable-options"])

        cmd = " ".join(cmd_parts)
        raw = time_exec(sandbox, cmd, timeout_s=timeout_s)
        raw.extras["mode"] = "nightly" if self._prefer_nightly else "stable"
        return raw

    def parse(self, raw: RawTestRun) -> TestResult:
        mode = raw.extras.get("mode", "stable")
        if mode == "nightly":
            try:
                return _parse_nightly_json(raw)
            except Exception:
                # Nightly parsing failed — fall through to stable parser.
                pass
        return _parse_stable_text(raw)


# ----------------------------------------------------------------------
# Nightly --format=json parser
# ----------------------------------------------------------------------
def _parse_nightly_json(raw: RawTestRun) -> TestResult:
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for line in (raw.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "test":
            continue
        name = obj.get("name", "") or ""
        event = (obj.get("event") or "").lower()
        if event == "ok":
            passed.append(name)
        elif event == "failed":
            failed.append(name)
        elif event in {"ignored", "skipped"}:
            skipped.append(name)
        # "started" events are ignored — they don't yet have an outcome.
    return TestResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        elapsed_s=raw.elapsed_s,
        exit_code=raw.exit_code,
    )


# ----------------------------------------------------------------------
# Stable text parser
# ----------------------------------------------------------------------
# Matches lines like:
#   test crate::module::name ... ok
#   test crate::module::name ... FAILED
#   test crate::module::name ... ignored
_TEXT_RE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)\s*$")


def _parse_stable_text(raw: RawTestRun) -> TestResult:
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for line in (raw.stdout or "").splitlines():
        m = _TEXT_RE.match(line.strip())
        if not m:
            continue
        name, outcome = m.group(1), m.group(2)
        if outcome == "ok":
            passed.append(name)
        elif outcome == "FAILED":
            failed.append(name)
        elif outcome == "ignored":
            skipped.append(name)
    return TestResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        elapsed_s=raw.elapsed_s,
        exit_code=raw.exit_code,
    )
