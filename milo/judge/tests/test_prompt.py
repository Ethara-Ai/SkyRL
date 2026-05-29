"""Tests for `milo.judge.prompt` — loader determinism + SHA stability.

Asserts:
  * `load_prompt()` returns the same tuple on repeated calls (cached).
  * The SHA-256 matches a fresh hash of the on-disk bytes (no drift
    between the module-level `PROMPT_SHA256` constant and what's on disk).
  * The prompt content contains the expected version marker / sections so
    `# version: 1.0` is not silently lost in a refactor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from milo.judge import prompt as prompt_module
from milo.judge.prompt import PROMPT_SHA256, PROMPT_TEXT, load_prompt


def test_load_prompt_returns_cached_tuple() -> None:
    """Two calls must return the exact same tuple object (cache hit)."""

    a = load_prompt()
    b = load_prompt()
    # Identity check: the cache is module-level, so we get the same tuple.
    assert a is b
    text, sha = a
    assert isinstance(text, str) and text
    assert isinstance(sha, str) and len(sha) == 64


def test_sha_matches_on_disk_bytes() -> None:
    """Module constant must match a fresh hash of the file bytes."""

    path = Path(prompt_module.__file__).resolve().parent / "SYSTEM_PROMPT.md"
    raw = path.read_bytes()
    fresh = hashlib.sha256(raw).hexdigest()
    assert PROMPT_SHA256 == fresh
    assert PROMPT_TEXT == raw.decode("utf-8")


def test_prompt_contains_required_sections() -> None:
    """The frozen prompt must include the spec-mandated sections.

    These are load-bearing — the service prompt-renders rely on the
    presence of explicit `0 | 0.5 | 1` scoring guidance and the
    tampering-detection contract. Silent prompt drift would break
    calibration.
    """

    text = PROMPT_TEXT
    assert "# version: 1.0" in text, "version marker must be in the prompt"
    assert "Rubric Judge System Prompt" in text
    assert "0 | 0.5 | 1" in text
    assert "tampering_detected" in text
    assert "test-fixture-tampering" in text


def test_sha_stable_across_resets() -> None:
    """After resetting and reloading, the SHA must match."""

    sha_before = PROMPT_SHA256
    prompt_module._reset_cache_for_tests()
    _, sha_after = load_prompt()
    assert sha_before == sha_after
