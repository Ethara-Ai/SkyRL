"""C CTest runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (c row).

Canonical command:
    ctest --output-on-failure

CTest emits per-test lines of the form:
    Test #1: my_test_name ............................   Passed    0.01 sec
    Test #2: failing_test ............................***Failed    0.02 sec
    Test #3: skipped_test ............................   Skipped   0.00 sec

And a final summary:
    100% tests passed, 0 tests failed out of 5

Stable test-ID format: the literal test name as registered with `add_test(...)`
in the project's CMakeLists.

v0.4 — TODO("real parser in Phase 2.2.next"): this regex parser handles
the common ctest output, but not:
    * `--output-junit` XML output (more reliable on newer ctest)
    * Subtest groupings
    * Multi-line failure annotations (the parser currently treats them as
      noise, which is correct, but the IDs of failed tests with embedded
      whitespace in their names are not robust)
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun, time_exec

# Captures the test number, name, and outcome word.
# Examples that should match:
#     "    Test #1: my_test ........   Passed    0.01 sec"
#     "    Test #2: bad_test ......***Failed    0.02 sec"
#     "    Test #3: meh_test .......  Skipped   0.00 sec"
_CTEST_LINE_RE = re.compile(
    r"^\s*Test\s+#\d+:\s+(\S+)\s+\.+\s*\*{0,3}\s*(Passed|Failed|Skipped|Not Run|Timeout)",
    re.IGNORECASE,
)


class CCtestRunner:
    """CTest-based runner for C milo-bench instances."""

    LANG: ClassVar[str] = "c"

    def __init__(
        self,
        ctest_bin: str = "ctest",
        extra_args: tuple[str, ...] = ("--output-on-failure",),
        build_dir: str = "build",
    ) -> None:
        self._ctest_bin = ctest_bin
        self._extra_args = tuple(extra_args)
        self._build_dir = build_dir

    def install_deps(self, sandbox: Any) -> None:
        """Multi-SWE-bench C images build with CMake at image build time.
        No-op by default."""
        return None

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        cmd_parts: list[str] = [self._ctest_bin, *self._extra_args]
        if test_filter:
            # `-R <regex>` to filter; OR-join with alternation.
            joined = "|".join(re.escape(t) for t in test_filter)
            cmd_parts.extend(["-R", joined])

        cmd = " ".join(cmd_parts)
        # Run in the build dir (CMake convention).
        return time_exec(sandbox, cmd, timeout_s=timeout_s, cwd=f"/workspace/{self._build_dir}")

    def parse(self, raw: RawTestRun) -> TestResult:
        passed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        for line in (raw.stdout or "").splitlines():
            m = _CTEST_LINE_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            outcome = m.group(2).lower()
            if outcome == "passed":
                passed.append(name)
            elif outcome in {"failed", "timeout"}:
                failed.append(name)
            elif outcome in {"skipped", "not run"}:
                skipped.append(name)

        return TestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            elapsed_s=raw.elapsed_s,
            exit_code=raw.exit_code,
        )
