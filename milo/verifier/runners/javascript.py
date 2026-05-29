"""JavaScript Jest runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (javascript row).

Canonical command:
    npx jest --json --useStderr [<filters>...]

Jest's `--json` reporter emits a single JSON object on stdout with the full
test tree. We pin it because the text reporter is unstable across versions.

Stable test-ID format used here:
    <test_file_relative_path> > <describe_path> > <it_name>

The leading file path makes the ID grep-able; the `>` separator follows
Jest's own convention for test paths.

If the repo uses Mocha or Vitest instead of Jest, see `TsJestRunner` (which
uses ts-jest preset) or subclass and override `_jest_cmd`. A separate
JsMochaRunner is *not* shipped; per Phase 2.2 the v0.4 default is Jest for
both JS and TS to keep the parser surface area small.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, ClassVar

from milo.verifier.report import TestResult
from milo.verifier.runners.base import RawTestRun, time_exec


class JsJestRunner:
    """Jest-based test runner for JavaScript milo-bench instances."""

    LANG: ClassVar[str] = "javascript"

    def __init__(
        self,
        jest_bin: str = "npx jest",
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self._jest_bin = jest_bin
        self._extra_args = tuple(extra_args)

    # ------------------------------------------------------------------
    # PerLanguageTestRunner contract
    # ------------------------------------------------------------------
    def install_deps(self, sandbox: Any) -> None:
        """Multi-SWE-bench JS images pre-install via `npm ci` at build time.
        No-op by default. Override for repos that need extra peer-deps."""
        return None

    def run_tests(
        self,
        sandbox: Any,
        test_filter: list[str] | None,
        timeout_s: int,
    ) -> RawTestRun:
        cmd_parts: list[str] = [
            self._jest_bin,
            "--json",
            # Use stderr for Jest's progress output so stdout stays pure JSON.
            "--useStderr",
        ]
        cmd_parts.extend(self._extra_args)
        if test_filter:
            # Jest's positional args are regex over file paths. Quote each.
            cmd_parts.extend(shlex.quote(t) for t in test_filter)

        cmd = " ".join(cmd_parts)
        raw = time_exec(sandbox, cmd, timeout_s=timeout_s)
        return raw

    def parse(self, raw: RawTestRun) -> TestResult:
        """Parse Jest's --json output.

        Jest emits:
            {
              "testResults": [
                {
                  "name": "/absolute/path/to/test_file.js",
                  "testResults": [
                    {
                      "ancestorTitles": ["describe block", "nested describe"],
                      "title": "it block",
                      "status": "passed" | "failed" | "skipped" | "pending" | "todo"
                    },
                    ...
                  ]
                },
                ...
              ]
            }

        We normalize to:
            <basename(name)> > <ancestorTitles joined by " > "> > <title>
        """
        stdout = raw.stdout or ""
        try:
            data = _extract_first_json_object(stdout)
        except ValueError:
            # Jest didn't produce JSON (likely a build failure before tests
            # ran). Empty TestResult — I-2 will catch the empty case.
            return TestResult(
                passed=[],
                failed=[],
                skipped=[],
                elapsed_s=raw.elapsed_s,
                exit_code=raw.exit_code,
            )

        passed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        for suite in data.get("testResults", []) or []:
            file_path = suite.get("name", "") or ""
            basename = file_path.rsplit("/", 1)[-1] if file_path else ""

            for tc in suite.get("testResults", []) or []:
                ancestors = tc.get("ancestorTitles", []) or []
                title = tc.get("title", "") or ""
                parts = [p for p in [basename, *ancestors, title] if p]
                node_id = " > ".join(parts)
                status = (tc.get("status") or "").lower()

                if status == "passed":
                    passed.append(node_id)
                elif status == "failed":
                    failed.append(node_id)
                elif status in {"skipped", "pending", "todo", "disabled"}:
                    skipped.append(node_id)
                else:
                    # Unknown status — treat as failed to be safe.
                    failed.append(node_id)

        return TestResult(
            passed=passed,
            failed=failed,
            skipped=skipped,
            elapsed_s=raw.elapsed_s,
            exit_code=raw.exit_code,
        )


def _extract_first_json_object(text: str) -> dict:
    """Extract the first balanced top-level JSON object from `text`.

    Jest may print non-JSON noise (warnings, deprecation notices) before the
    JSON object. We scan for the first `{` and use a depth counter to find
    the matching `}`.

    Raises:
        ValueError: if no balanced JSON object is found.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("no '{' in jest output")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                return json.loads(blob)

    raise ValueError("unbalanced JSON in jest output")
