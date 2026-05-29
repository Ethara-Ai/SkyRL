"""PerLanguageTestRunner Protocol + RawTestRun dataclass.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (per-language test runner
mini-spec).

The Protocol is the *only* contract between the ThreeRunVerifier and a
language-specific runner. Adding a 9th language is a matter of:

    class MyLangRunner:
        LANG = "mylang"
        def install_deps(self, sandbox): ...
        def run_tests(self, sandbox, test_filter, timeout_s) -> RawTestRun: ...
        def parse(self, raw) -> TestResult: ...

    # and updating milo/verifier/runners/registry.py to point "mylang" at it.

The Sandbox contract used here is the one defined in Phase 1's
`milo/lht_adapter/docker_runtime.py` (`Sandbox.exec(cmd, timeout, cwd)`,
`Sandbox.read_file(path)`, etc.). We don't import that module here to avoid a
hard dependency from Phase 2 to Phase 1's runtime — tests use a fake Sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from milo.verifier.report import TestResult


@dataclass(frozen=False)
class RawTestRun:
    """The raw output of executing one test-runner invocation in a sandbox.

    Attributes:
        stdout:     full stdout of the test command.
        stderr:     full stderr of the test command.
        exit_code:  process exit code (0 = success for the runner; not
                    necessarily 0 if individual tests failed — pytest e.g.
                    returns 1 on test failures).
        elapsed_s:  wall-clock seconds of the test command.
        extras:     framework-specific artefacts retrieved from the sandbox
                    (e.g. {"junit_xml": "<...>"} for pytest, {"json": {...}}
                    for jest). Parsers consume this if present.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    elapsed_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PerLanguageTestRunner(Protocol):
    """Contract that every language-specific runner implements.

    Lifecycle (per invocation, called by ThreeRunVerifier):
        1. `install_deps(sandbox)` — idempotent, may be a no-op if the image
           is already prebuilt with all deps (the Multi-SWE-bench convention).
        2. `run_tests(sandbox, test_filter, timeout_s)` — execute the canonical
           test command for this language. Returns RawTestRun.
        3. `parse(raw)` — convert RawTestRun → TestResult.

    `LANG` is the canonical lowercase language string (matches the milo-bench
    on-disk `lang` field: python/javascript/typescript/java/go/rust/c/cpp).

    `test_filter` semantics:
        * None → run all tests the framework discovers.
        * list[str] → run only the named tests (framework-specific syntax;
          e.g. pytest accepts node-IDs, jest accepts a regex, go accepts
          `-run` regex). Concrete runners document their conventions.
    """

    LANG: ClassVar[str]

    def install_deps(self, sandbox: Any) -> None:
        """Install any per-language test dependencies. Idempotent. Usually
        a no-op for Multi-SWE-bench images where deps are pre-installed."""
        ...

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        """Execute the canonical test command. Returns the raw output for
        downstream parsing."""
        ...

    def parse(self, raw: RawTestRun) -> TestResult:
        """Convert raw output → TestResult. Pure function over `raw`."""
        ...


# ----------------------------------------------------------------------
# Convenience helpers shared by concrete runners
# ----------------------------------------------------------------------
def time_exec(sandbox: Any, cmd: str, timeout_s: int, cwd: str = "/workspace") -> RawTestRun:
    """Execute `cmd` in `sandbox` and wrap the output in a RawTestRun with
    wall-clock timing.

    The Sandbox contract (see Phase 1's docker_runtime.py) returns a dict
    with keys at minimum {stdout, stderr, exit_code} or {output, returncode}
    for minisweagent compatibility. We defensively normalize both shapes.

    Tests pass a fake sandbox whose `exec` returns either shape; this helper
    is the only place that has to know.
    """
    import time

    t0 = time.perf_counter()
    result = sandbox.exec(cmd, timeout=timeout_s, cwd=cwd)
    elapsed = time.perf_counter() - t0

    if not isinstance(result, dict):
        # Some sandboxes return bytes / str; treat as stdout-only success.
        return RawTestRun(
            stdout=str(result),
            stderr="",
            exit_code=0,
            elapsed_s=elapsed,
        )

    stdout = result.get("stdout") or result.get("output") or ""
    stderr = result.get("stderr") or ""
    exit_code = result.get("exit_code")
    if exit_code is None:
        exit_code = result.get("returncode", 0)
    return RawTestRun(
        stdout=str(stdout),
        stderr=str(stderr),
        exit_code=int(exit_code or 0),
        elapsed_s=elapsed,
    )
