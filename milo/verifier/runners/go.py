"""Go test runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (go row).

Canonical command:
    go test -v ./...

Go's `-v` mode emits one of the following lines per test outcome:
    === RUN   TestFoo
    --- PASS: TestFoo (0.00s)
    --- FAIL: TestFoo (0.00s)
    --- SKIP: TestFoo (0.00s)

Subtests via `t.Run("name", ...)` produce:
    --- PASS: TestFoo/subtest_name (0.00s)

Stable test-ID format: `<package>.TestName[/subtest]`.

The package prefix is recovered from the per-package output marker:
    === RUN   TestFoo
    --- PASS: TestFoo (0.00s)
    PASS
    ok      package/path    0.012s

`go test -json` is more robust but adds a parsing layer and is gated on
Go 1.10+. For v0.4 we use the line-scanner; switch to `-json` if it bites.
TODO("real parser in Phase 2.2.next") — flip to -json.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun, time_exec

# Regex for `--- PASS: <name> (<dur>s)` etc.
_OUTCOME_RE = re.compile(r"^---\s+(PASS|FAIL|SKIP):\s+(\S+)")
# `ok <package> <dur>` / `FAIL <package> <dur>` per-package summary line.
_PKG_OK_RE = re.compile(r"^(ok|FAIL|---\s+FAIL:)\s+(\S+)")


class GoTestRunner:
    """`go test`-based runner for Go milo-bench instances."""

    LANG: ClassVar[str] = "go"

    def __init__(
        self,
        go_bin: str = "go",
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self._go_bin = go_bin
        self._extra_args = tuple(extra_args)

    def install_deps(self, sandbox: Any) -> None:
        """Multi-SWE-bench Go images run `go mod download` at image build
        time. No-op by default."""
        return None

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        cmd_parts: list[str] = [self._go_bin, "test", "-v", *self._extra_args]
        if test_filter:
            # go test -run is a regex; OR-join with alternation.
            joined = "|".join(re.escape(t) for t in test_filter)
            cmd_parts.append(f"-run={joined}")
        cmd_parts.append("./...")

        cmd = " ".join(cmd_parts)
        return time_exec(sandbox, cmd, timeout_s=timeout_s)

    def parse(self, raw: RawTestRun) -> TestResult:
        passed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        current_pkg = ""
        # The package marker (`ok package`, `FAIL package`) appears AFTER its
        # test outcomes. We buffer test outcomes per slab and apply the
        # package when we see the marker.
        slab: list[tuple[str, str]] = []  # [(outcome, test_name), ...]

        for raw_line in (raw.stdout or "").splitlines():
            line = raw_line.rstrip("\r")
            m = _OUTCOME_RE.match(line)
            if m:
                outcome, name = m.group(1), m.group(2)
                slab.append((outcome, name))
                continue
            m_pkg = _PKG_OK_RE.match(line)
            if m_pkg:
                current_pkg = m_pkg.group(2)
                # Flush the slab using the now-known package.
                for outcome, name in slab:
                    node_id = f"{current_pkg}.{name}" if current_pkg else name
                    if outcome == "PASS":
                        passed.append(node_id)
                    elif outcome == "FAIL":
                        failed.append(node_id)
                    elif outcome == "SKIP":
                        skipped.append(node_id)
                slab = []
                current_pkg = ""
                continue

        # Any unflushed slab (e.g. compile failure with no package marker)
        # gets emitted without a package prefix.
        for outcome, name in slab:
            node_id = name
            if outcome == "PASS":
                passed.append(node_id)
            elif outcome == "FAIL":
                failed.append(node_id)
            elif outcome == "SKIP":
                skipped.append(node_id)

        return TestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            elapsed_s=raw.elapsed_s,
            exit_code=raw.exit_code,
        )
