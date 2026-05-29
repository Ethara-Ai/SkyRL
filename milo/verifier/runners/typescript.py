"""TypeScript Jest runner — variant of the JavaScript runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (typescript row).

Per the plan: "alias to JsJestRunner (`TsJestRunner = JsJestRunner`) plus a
`--preset ts-jest` variant."

We don't literally alias — we subclass so that `LANG = "typescript"` and the
default invocation passes `--preset ts-jest`. Subclassing keeps the registry
dispatch obvious and lets language-specific instrumentation evolve
independently from the JS path later.
"""

from __future__ import annotations

from typing import ClassVar

from milo.verifier.runners.javascript import JsJestRunner


class TsJestRunner(JsJestRunner):
    """Jest + ts-jest preset for TypeScript milo-bench instances.

    Defaults to `npx jest --preset ts-jest --json --useStderr`. The base
    class's parser handles the JSON output identically — TypeScript test
    results are shape-compatible with the JS reporter.
    """

    LANG: ClassVar[str] = "typescript"

    def __init__(
        self,
        jest_bin: str = "npx jest",
        extra_args: tuple[str, ...] = ("--preset", "ts-jest"),
    ) -> None:
        super().__init__(jest_bin=jest_bin, extra_args=extra_args)
