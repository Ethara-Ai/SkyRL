"""Per-language test runners for the milo verifier (Phase 2.2).

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 and the registry/dispatch helper
at §11 of the spec's `CAPACITY.md` table.

Eight languages are supported (one runner module each, c and cpp share, ts
and js share via the `--preset ts-jest` variant):

    +-------------+---------------------+---------+
    | Language    | Runner class        | Module  |
    +-------------+---------------------+---------+
    | python      | PythonPytestRunner  | python  |
    | javascript  | JsJestRunner        | js      |
    | typescript  | TsJestRunner        | ts      |
    | java        | JavaMavenRunner     | java    |
    | go          | GoTestRunner        | go      |
    | rust        | RustCargoRunner     | rust    |
    | c           | CCtestRunner        | c       |
    | cpp         | CppCtestRunner      | cpp     |
    +-------------+---------------------+---------+

Use `get_runner(lang: str)` from `milo.verifier.runners.registry` to dispatch.

The Java/Rust/C runners are v0.4 — they implement the contract and produce
correct results for the canonical test commands, but the parsers are
intentionally shallow regex / line-scanners. Fixture-driven hardening is
tracked under Phase 2.2.next; the contract and call sites are stable.
"""

from __future__ import annotations

from milo.verifier.runners.base import (
    PerLanguageTestRunner,
    RawTestRun,
)
from milo.verifier.runners.c import CCtestRunner
from milo.verifier.runners.cpp import CppCtestRunner
from milo.verifier.runners.go import GoTestRunner
from milo.verifier.runners.java import JavaMavenRunner
from milo.verifier.runners.javascript import JsJestRunner
from milo.verifier.runners.python import PythonPytestRunner
from milo.verifier.runners.registry import (
    DEFAULT_RUNNERS,
    get_runner,
    register_runner,
)
from milo.verifier.runners.rust import RustCargoRunner
from milo.verifier.runners.typescript import TsJestRunner

__all__ = [
    "PerLanguageTestRunner",
    "RawTestRun",
    "PythonPytestRunner",
    "JsJestRunner",
    "TsJestRunner",
    "JavaMavenRunner",
    "GoTestRunner",
    "RustCargoRunner",
    "CCtestRunner",
    "CppCtestRunner",
    "get_runner",
    "register_runner",
    "DEFAULT_RUNNERS",
]
