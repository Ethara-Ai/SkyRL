"""Per-language runner registry.

Implements IMPLEMENTATION_PLAN.md v0.4 §2.2 (the language → runner lookup
table).

The on-disk milo-bench `lang` field is one of:
    python | javascript | typescript | java | go | rust | c | cpp

`get_runner(lang)` returns a *fresh instance* of the appropriate runner per
call (runners are cheap to construct; a fresh instance avoids any
accidental cross-call state from cached compilations etc.).

Integrators can register additional runners (e.g. a 9th language) via
`register_runner("kotlin", KotlinRunner)`. Overriding an existing language
is allowed but logged at WARNING level — the test budget is set per-lang.
"""

from __future__ import annotations

import logging
from typing import Callable

from milo.verifier.runners.base import PerLanguageTestRunner
from milo.verifier.runners.c import CCtestRunner
from milo.verifier.runners.cpp import CppCtestRunner
from milo.verifier.runners.go import GoTestRunner
from milo.verifier.runners.java import JavaMavenRunner
from milo.verifier.runners.javascript import JsJestRunner
from milo.verifier.runners.python import PythonPytestRunner
from milo.verifier.runners.rust import RustCargoRunner
from milo.verifier.runners.typescript import TsJestRunner

_log = logging.getLogger(__name__)

# Factory functions so each `get_runner` call returns a fresh instance.
_RunnerFactory = Callable[[], PerLanguageTestRunner]


DEFAULT_RUNNERS: dict[str, _RunnerFactory] = {
    "python": PythonPytestRunner,
    "javascript": JsJestRunner,
    "typescript": TsJestRunner,
    "java": JavaMavenRunner,
    "go": GoTestRunner,
    "rust": RustCargoRunner,
    "c": CCtestRunner,
    "cpp": CppCtestRunner,
}

# Aliases users may pass in (case-insensitive match handled in get_runner).
_LANG_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
    "rs": "rust",
    "c++": "cpp",
    "cxx": "cpp",
}

# The registry is mutable so integrators can plug in custom runners. We
# keep DEFAULT_RUNNERS frozen as the authoritative ship state and copy it
# into _registry on first import.
_registry: dict[str, _RunnerFactory] = dict(DEFAULT_RUNNERS)


def get_runner(lang: str) -> PerLanguageTestRunner:
    """Return a fresh runner instance for the given language string.

    The `lang` argument is case-insensitive; common aliases (py/js/ts/c++)
    are normalized.

    Raises:
        KeyError: if `lang` is not recognized. Caller should treat this as
            an unsupported-language error (the milo-bench task should be
            moved to Cohort C with the appropriate audit flag).
    """
    norm = (lang or "").strip().lower()
    norm = _LANG_ALIASES.get(norm, norm)
    if norm not in _registry:
        raise KeyError(
            f"no runner registered for language {lang!r} "
            f"(known: {sorted(_registry)})"
        )
    return _registry[norm]()


def register_runner(lang: str, factory: _RunnerFactory) -> None:
    """Register a new runner factory under `lang` (or override an existing one).

    `factory` is a zero-arg callable that returns a fresh runner. The
    runner instance must satisfy the `PerLanguageTestRunner` Protocol.
    """
    norm = (lang or "").strip().lower()
    if not norm:
        raise ValueError("lang must be a non-empty string")
    if norm in _registry:
        _log.warning("registry: overriding runner for lang=%s", norm)
    _registry[norm] = factory


def reset_registry() -> None:
    """Restore the registry to its default state. Useful in tests."""
    _registry.clear()
    _registry.update(DEFAULT_RUNNERS)
