"""`JudgeBackend` Protocol — uniform LLM-inference contract for the judge.

Implements the swap point called out in RL_GYM_SPEC.md v0.7 §5.3 ("Swapping
the judge to GPT-5, Gemini Pro, or an ensemble is a one-line config
change"). The Protocol is intentionally minimal: one `call(prompt, model,
temperature) -> str` method. Backend-specific message threading lives in
each implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class JudgeBackendError(RuntimeError):
    """Raised by a `JudgeBackend` when the upstream inference call fails.

    The judge service treats this as a recoverable error (caller can retry
    or fall back to a different backend) — distinct from a malformed
    response which the service handles itself via its JSON parser.
    """


@runtime_checkable
class JudgeBackend(Protocol):
    """One-shot, stateless LLM caller for the rubric judge.

    Each invocation is independent: there is no conversation memory and
    no streaming. The judge prompt is fully self-contained per call.

    Implementations must:
      * be thread-safe (the service may dispatch concurrent calls).
      * respect `temperature` — for the judge we always pass 0.
      * raise `JudgeBackendError` on transport / API failure rather than
        leaking provider-specific exceptions to the caller.
    """

    def call(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
        """Send `system_prompt` + `user_prompt` to `model`, return the raw
        assistant text. No JSON parsing; the service handles that.

        Implementations should:
          * separate system vs. user content using the provider's native
            conventions (Bedrock Converse `system=`, Anthropic Messages
            `system=`, etc.).
          * not retry internally — retries are the service's job.
        """
        ...


__all__ = ["JudgeBackend", "JudgeBackendError"]
