"""C++ CTest runner — variant of the C runner.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (cpp row): "alias to CCtestRunner."

We subclass rather than alias literally so that:
    * `LANG = "cpp"` for registry dispatch.
    * Future C++-specific instrumentation (gtest XML, catch2 XML reporters)
      can land here without touching the C path.

For v0.4 the behavior is identical to CCtestRunner.
"""

from __future__ import annotations

from typing import ClassVar

from milo.verifier.runners.c import CCtestRunner


class CppCtestRunner(CCtestRunner):
    """CTest-based runner for C++ milo-bench instances."""

    LANG: ClassVar[str] = "cpp"
