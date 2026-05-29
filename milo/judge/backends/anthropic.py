"""Direct Anthropic API backend for the rubric judge.

Fallback to the official `anthropic` Python SDK when Bedrock is unavailable
(e.g. local development outside AWS, or judge cross-checks against the
non-Bedrock Anthropic surface). Per RL_GYM_SPEC.md v0.7 §5.3 the judge is
configurable; per §29 (customization swap points) every adapter is a
one-line config change. Reads its API key from `${MILO_ANTHROPIC_API_KEY}`.
"""

from __future__ import annotations

import os
from typing import Any

from milo.judge.backends.base import JudgeBackend, JudgeBackendError

# Anthropic's modern Claude judge: today's GA is claude-opus-4-6.
_DEFAULT_MODEL = "claude-opus-4-6"


def _default_model_id() -> str:
    """Resolve the default judge model from the env. Same convention as
    the Bedrock backend (`MILO_JUDGE_MODEL`) so both share one knob.
    """

    return os.environ.get("MILO_JUDGE_MODEL", _DEFAULT_MODEL)


class AnthropicJudgeBackend:
    """`JudgeBackend` implementation backed by the Anthropic Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key. Defaults to `${MILO_ANTHROPIC_API_KEY}` and then
        `${ANTHROPIC_API_KEY}` for compatibility with the SDK's own
        convention.
    client:
        Optional pre-built `anthropic.Anthropic` (or any duck-typed object
        exposing `messages.create(...)`). Injectable for unit tests.
    max_tokens:
        Cap on completion length. Judge replies are short JSON; 4096 is ample.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("MILO_ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._client = client
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # SDK client lifecycle
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """Lazily build the Anthropic client. Import is deferred so the
        module loads cleanly in environments without the SDK installed
        (handy for unit tests with a stub backend).
        """

        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise JudgeBackendError(
                "anthropic SDK is required for AnthropicJudgeBackend; install via `pip install anthropic`"
            ) from exc
        if not self._api_key:
            raise JudgeBackendError(
                "AnthropicJudgeBackend requires MILO_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY) to be set"
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # JudgeBackend protocol
    # ------------------------------------------------------------------
    def call(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
        """Round-trip a single judge call via the Anthropic Messages API."""

        client = self._get_client()
        try:
            response = client.messages.create(
                model=model or _default_model_id(),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=self._max_tokens,
                temperature=float(temperature),
            )
        except Exception as exc:
            raise JudgeBackendError(f"Anthropic messages.create() failed: {exc}") from exc

        return _extract_text_from_messages_response(response)


def _extract_text_from_messages_response(response: Any) -> str:
    """Pull the assistant text out of an Anthropic Messages response.

    The response `.content` is a list of typed blocks; we concatenate
    every block whose `.type == "text"`, skipping tool_use / image blocks
    that the judge prompt never solicits.
    """

    # The SDK returns objects with attributes, but a dict is also possible
    # if the caller passed a hand-rolled stub. Handle both gracefully.
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if not content:
        raise JudgeBackendError(f"Anthropic response missing content: {response!r}")

    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text_value = getattr(block, "text", None)
        if text_value is None and isinstance(block, dict):
            text_value = block.get("text")
        if text_value:
            parts.append(text_value)

    if not parts:
        raise JudgeBackendError(f"Anthropic response had no text blocks: {content!r}")
    return "".join(parts)


# Protocol conformance sanity check at import time.
_: JudgeBackend = AnthropicJudgeBackend()  # type: ignore[assignment]


__all__ = ["AnthropicJudgeBackend"]
