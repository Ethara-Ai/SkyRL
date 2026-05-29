"""Loader + SHA-256 digest for the frozen judge system prompt.

Implements RL_GYM_SPEC.md v0.7 §5.3 (frozen rubric judge prompt) and
IMPLEMENTATION_PLAN.md v0.4 Phase 3.2 (prompt loader). The prompt text and
its SHA-256 are cached at module import time so every judge call uses the
exact same bytes; bumping the prompt MUST be a deliberate version bump and
forces a fresh cache (the SHA-256 is part of the cache key, see service.py).
"""

# sha256: e20ef68291f1d0a9fb9049e146e52c383d3428d3352bb75c9d196fd6f4b429a4

from __future__ import annotations

import hashlib
from pathlib import Path

# Module-level cache. Loaded exactly once on first call.
_CACHE: tuple[str, str] | None = None

# Absolute path to the frozen prompt file, sibling of this module.
_PROMPT_PATH: Path = Path(__file__).resolve().parent / "SYSTEM_PROMPT.md"


def load_prompt() -> tuple[str, str]:
    """Return `(prompt_text, prompt_sha256)`.

    The prompt is read from `SYSTEM_PROMPT.md` next to this module. The
    result is cached at module level — repeated calls return the same tuple
    without re-reading the file. The SHA-256 digest is computed over the
    raw bytes (no normalisation), which means any whitespace change to the
    prompt file invalidates downstream caches automatically.
    """

    global _CACHE
    if _CACHE is None:
        raw = _PROMPT_PATH.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        # Decode after hashing so the digest matches the on-disk bytes
        # regardless of platform line-ending normalisation in the editor.
        text = raw.decode("utf-8")
        _CACHE = (text, sha)
    return _CACHE


def _reset_cache_for_tests() -> None:
    """Test hook: drop the module-level cache so a re-read of the file
    after monkey-patching `_PROMPT_PATH` actually re-reads.

    Production code MUST NOT call this. The name is underscore-prefixed
    and explicitly tagged `for_tests` to make grep-based misuse obvious.
    """

    global _CACHE
    _CACHE = None


# Eagerly compute the prompt SHA so other modules can `from milo.judge.prompt
# import PROMPT_SHA256` without paying the syscall on every reference.
PROMPT_TEXT, PROMPT_SHA256 = load_prompt()


__all__ = [
    "PROMPT_SHA256",
    "PROMPT_TEXT",
    "load_prompt",
]
