"""Java Maven Surefire runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (java row).

Canonical command:
    mvn test -B -q

Surefire writes one XML file per test class to `target/surefire-reports/`,
shape:
    <testsuite name="<package.Class>" tests="N" failures="F" errors="E" skipped="S">
      <testcase classname="<package.Class>" name="testMethod" time="0.123">
        [<failure ...>] [<error ...>] [<skipped ...>]
      </testcase>
      ...
    </testsuite>

We collect all surefire XMLs the sandbox can list, parse each, and union the
results. Test-ID format:
    <package.Class>#<method>

v0.4 NOTE — this runner uses a shallow `find` + per-file read strategy. It
does NOT handle:
    * gradle (`./gradlew test` and the gradle XML format under
      `build/test-results/`)
    * sbt (Scala but on the JVM)
    * Spotless-formatted XML quirks
TODO("real parser in Phase 2.2.next") — fixture-driven hardening when we hit
Java instances in Cohort A/B.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun, time_exec

SUREFIRE_REPORTS_DIR = "target/surefire-reports"


class JavaMavenRunner:
    """Maven + Surefire runner for Java milo-bench instances."""

    LANG: ClassVar[str] = "java"

    def __init__(
        self,
        mvn_bin: str = "mvn",
        extra_args: tuple[str, ...] = ("-B", "-q"),
        reports_dir: str = SUREFIRE_REPORTS_DIR,
    ) -> None:
        self._mvn_bin = mvn_bin
        self._extra_args = tuple(extra_args)
        self._reports_dir = reports_dir

    def install_deps(self, sandbox: Any) -> None:
        """Multi-SWE-bench Java images run `mvn dependency:go-offline` at
        image build time. No-op by default."""
        return None

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        cmd_parts: list[str] = [self._mvn_bin, "test", *self._extra_args]
        if test_filter:
            # `-Dtest=Class#method,Class2#method2` is Surefire's selection syntax.
            cmd_parts.append("-Dtest=" + ",".join(test_filter))

        cmd = " ".join(cmd_parts)
        raw = time_exec(sandbox, cmd, timeout_s=timeout_s)

        # Grab the surefire XML files. We list the directory via `ls -1`
        # (portable; no `find -print0` needed for this shallow case).
        ls_cmd = f"ls -1 {self._reports_dir}/TEST-*.xml 2>/dev/null || true"
        ls_raw = sandbox.exec(ls_cmd, timeout=30, cwd="/workspace")
        ls_stdout = ""
        if isinstance(ls_raw, dict):
            ls_stdout = ls_raw.get("stdout") or ls_raw.get("output") or ""
        elif ls_raw is not None:
            ls_stdout = str(ls_raw)

        xml_files = [line.strip() for line in str(ls_stdout).splitlines() if line.strip()]
        xmls: list[str] = []
        for path in xml_files:
            try:
                contents = sandbox.read_file(path)
                if isinstance(contents, bytes):
                    contents = contents.decode("utf-8", errors="replace")
                xmls.append(str(contents))
            except Exception:
                # Skip unreadable files; we'll fall back to stdout if every
                # one fails.
                continue
        raw.extras["surefire_xmls"] = xmls

        return raw

    def parse(self, raw: RawTestRun) -> TestResult:
        passed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        xmls = raw.extras.get("surefire_xmls", []) or []
        if xmls:
            for xml_text in xmls:
                try:
                    p, f, s = _parse_surefire_xml(xml_text)
                except ET.ParseError:
                    continue
                passed.extend(p)
                failed.extend(f)
                skipped.extend(s)
        else:
            # Fallback: shallow regex over Maven's `-q` summary line.
            # Returns empty lists (no test-IDs available) but preserves
            # counts via the elapsed/exit_code so caller can detect the
            # zero-coverage state.
            # v0.4 — TODO("real parser in Phase 2.2.next"): teach the
            # parser to dig per-class counts out of Maven's text summary
            # if the XMLs aren't reachable.
            _ = _parse_mvn_summary(raw.stdout)

        return TestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            elapsed_s=raw.elapsed_s,
            exit_code=raw.exit_code,
        )


def _parse_surefire_xml(xml_text: str) -> tuple[list[str], list[str], list[str]]:
    """Parse one TEST-<class>.xml file. Returns (passed, failed, skipped)."""
    root = ET.fromstring(xml_text)
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    # `root` may be <testsuite> directly (Surefire) or <testsuites> (newer
    # Surefire versions wrap multiple). Handle both.
    suites = root.findall("testsuite") or [root]

    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "") or suite.get("name", "") or ""
            name = case.get("name", "") or ""
            node_id = f"{classname}#{name}" if classname else name

            if case.find("failure") is not None or case.find("error") is not None:
                failed.append(node_id)
            elif case.find("skipped") is not None:
                skipped.append(node_id)
            else:
                passed.append(node_id)
    return passed, failed, skipped


_MVN_SUMMARY_RE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
)


def _parse_mvn_summary(stdout: str) -> dict[str, int]:
    """Best-effort extraction of the Maven Surefire summary line.

    Returns {} if no match. This is only used to surface counts when XMLs
    are unreachable; downstream code prefers the XML path.
    """
    m = _MVN_SUMMARY_RE.search(stdout or "")
    if not m:
        return {}
    total = int(m.group(1))
    failures = int(m.group(2))
    errors = int(m.group(3))
    skipped = int(m.group(4))
    passed = total - failures - errors - skipped
    return {
        "total": total,
        "passed": passed,
        "failed": failures + errors,
        "skipped": skipped,
    }
