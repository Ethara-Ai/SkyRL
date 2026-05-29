"""Python pytest runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (Python row of the per-language
runner table) and is the *reference* runner — the others mirror its shape.

Canonical command:
    pytest --tb=no -q --no-header --junit-xml=/tmp/junit.xml [<filters>...]

We pin `--junit-xml` because the pytest text output is not stable across
versions (counts on the last line, dot vs verbose mode, ANSI codes from
plugins, etc.). The JUnit XML schema is stable and unambiguous, which is
what spec §2.2 means by "stable test IDs".

The stable test-ID format is the pytest node-ID:
    path/to/test_file.py::ClassName::test_name
or for non-class tests:
    path/to/test_file.py::test_name
or for parametrized tests:
    path/to/test_file.py::test_name[param-id]

This is the same ID milo-bench uses on disk (see e.g.
`locust/test/test_runners.py::TestMasterWorkerRunners::test_distributed_shape_stop_and_restart`
in the audit fixture). Good — interop is automatic.
"""

from __future__ import annotations

import shlex
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun, time_exec

# Path the runner writes the JUnit XML to inside the sandbox.
JUNIT_PATH = "/tmp/junit.xml"


class PythonPytestRunner:
    """pytest-based test runner for Python milo-bench instances."""

    LANG: ClassVar[str] = "python"

    def __init__(
        self,
        pytest_bin: str = "pytest",
        extra_args: tuple[str, ...] = (),
        junit_path: str = JUNIT_PATH,
    ) -> None:
        """
        Args:
            pytest_bin: how to invoke pytest. Override to `python -m pytest`
                or `uv run pytest` if the sandbox needs it.
            extra_args: passed verbatim to pytest. E.g. ("-p", "no:cacheprovider").
            junit_path: where pytest writes the JUnit XML inside the sandbox.
        """
        self._pytest_bin = pytest_bin
        self._extra_args = tuple(extra_args)
        self._junit_path = junit_path

    # ------------------------------------------------------------------
    # PerLanguageTestRunner contract
    # ------------------------------------------------------------------
    def install_deps(self, sandbox: Any) -> None:
        """Multi-SWE-bench images pre-install Python deps. No-op by default.

        Subclass and override if the instance needs an extra ad-hoc install
        (e.g. some Locust forks need `pip install -e .[dev]` to register
        the entry points used by tests).
        """
        return None

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        cmd_parts = [
            self._pytest_bin,
            "--tb=no",
            "-q",
            "--no-header",
            f"--junit-xml={self._junit_path}",
        ]
        cmd_parts.extend(self._extra_args)
        if test_filter:
            # pytest takes node-IDs verbatim. Quote each one defensively.
            cmd_parts.extend(shlex.quote(t) for t in test_filter)

        cmd = " ".join(cmd_parts)
        raw = time_exec(sandbox, cmd, timeout_s=timeout_s)

        # Pull the JUnit XML out of the sandbox; parse() consumes it.
        try:
            junit_xml = sandbox.read_file(self._junit_path)
            if isinstance(junit_xml, bytes):
                junit_xml = junit_xml.decode("utf-8", errors="replace")
            raw.extras["junit_xml"] = str(junit_xml)
        except Exception as e:
            # If the XML wasn't produced (e.g. pytest crashed before the
            # plugin wrote it), fall back to stdout parsing in parse().
            raw.extras["junit_xml"] = ""
            raw.extras["junit_read_error"] = repr(e)

        return raw

    def parse(self, raw: RawTestRun) -> TestResult:
        """Parse pytest's JUnit XML (preferred) or fall back to stdout."""
        junit_xml = raw.extras.get("junit_xml", "") or ""
        if junit_xml.strip():
            try:
                return _parse_junit_xml(junit_xml, raw)
            except ET.ParseError:
                # Corrupt XML — fall through to stdout parser.
                pass
        return _parse_pytest_stdout(raw)


# ----------------------------------------------------------------------
# JUnit XML parser (preferred path)
# ----------------------------------------------------------------------
def _parse_junit_xml(xml_text: str, raw: RawTestRun) -> TestResult:
    """Parse pytest's --junit-xml output into a TestResult.

    The pytest JUnit format is:
        <testsuite ...>
          <testcase classname="path.to.test_file.TestClass" name="test_name">
            <failure .../>         <-- present if failed
            <skipped .../>         <-- present if skipped
          </testcase>
          ...
        </testsuite>

    We reconstruct the pytest node-ID from `(classname, name)` per the
    plugin's convention:
        classname = "path/to/test_file" → "path/to/test_file.py::test_name"
        classname = "path/to/test_file.TestClass" → "path/to/test_file.py::TestClass::test_name"

    Pytest writes the path with dots, not slashes; we have to undo this.
    """
    root = ET.fromstring(xml_text)
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    # The root may be <testsuites> wrapping <testsuite> elements, or a single
    # <testsuite>. Handle both.
    suites = root.findall("testsuite") or [root]

    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "") or ""
            name = case.get("name", "") or ""
            node_id = _reconstruct_pytest_node_id(classname, name)

            if case.find("failure") is not None or case.find("error") is not None:
                failed.append(node_id)
            elif case.find("skipped") is not None:
                skipped.append(node_id)
            else:
                passed.append(node_id)

    return TestResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        elapsed_s=raw.elapsed_s,
        exit_code=raw.exit_code,
        raw_log_path=None,
    )


def _reconstruct_pytest_node_id(classname: str, name: str) -> str:
    """Best-effort reconstruction of a pytest node-ID from JUnit (classname, name).

    pytest's `_pytest/junitxml.py` writes:
        classname = "<file_path_with_dots_for_slashes>.<ClassName>" or
                    "<file_path_with_dots_for_slashes>" if no class
    We can't perfectly distinguish "file path component" from "class name" —
    the convention is that the rightmost component is a class iff it's a
    PascalCase identifier and there's at least one dot. Heuristic:
        * Split on dots.
        * If the last segment starts with an uppercase letter, treat it
          as a class name.
        * Join everything before with "/" and append ".py".
    """
    if not classname:
        return name

    parts = classname.split(".")
    if parts and parts[-1] and parts[-1][0].isupper():
        # Last part is the class.
        class_name = parts[-1]
        file_parts = parts[:-1]
        file_path = "/".join(file_parts) + ".py" if file_parts else ""
        if file_path:
            return f"{file_path}::{class_name}::{name}"
        return f"{class_name}::{name}"

    # No class — classname is the full file path (with dots for slashes).
    file_path = "/".join(parts) + ".py"
    return f"{file_path}::{name}"


# ----------------------------------------------------------------------
# Stdout fallback parser
# ----------------------------------------------------------------------
def _parse_pytest_stdout(raw: RawTestRun) -> TestResult:
    """Fallback parser for when --junit-xml didn't produce a file.

    This is INTENTIONALLY shallow. Pytest's terse `-q` mode emits one dot/F/s
    per test, then a summary line. We can't recover test-IDs from dots, so
    this parser returns a TestResult with counts surfaced via the summary
    line if present, but `passed`/`failed`/`skipped` LISTS are empty.

    Returning empty lists is safe — downstream code (ThreeRunVerifier,
    invariant I-2 v0.7-hardened) checks `len(...) > 0` and will mark the
    invariant as failed if it can't see the test IDs. That's the correct
    behavior: a fix_patch that crashed pytest before the plugin wrote XML
    should NOT score R_terminal=1.
    """
    return TestResult(
        passed=[],
        failed=[],
        skipped=[],
        elapsed_s=raw.elapsed_s,
        exit_code=raw.exit_code,
        raw_log_path=None,
    )
