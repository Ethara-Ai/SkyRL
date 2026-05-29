"""Tests for `milo.judge` — rubric judge service.

Covers prompt loader determinism, service-level caching, tampering
hard-floor, and stub-backend round-tripping. The Bedrock and Anthropic
backends themselves are exercised only through their `JudgeBackend`
Protocol; no live API calls run in CI.
"""

from __future__ import annotations
