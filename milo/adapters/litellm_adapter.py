"""LiteLLM-backed policy adapter — Phase 7 / spec §15.3.

Covers OpenAI / Anthropic-direct / Gemini / Mistral / vLLM-OpenAI-compat
via a single class. `litellm` already normalises the wire format across all
of them, so we don't have to maintain per-provider code.

Per `IMPLEMENTATION_PLAN.md` v0.4: 4 of the 5 shipped policy adapters come
through this class (only Bedrock is direct-boto3 per `milo.adapters.bedrock`).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from milo.adapters.base import PolicyAdapter, PolicyResult

logger = logging.getLogger("milo.adapters.litellm")


DEFAULT_LITELLM_MODEL_ENV = "MILO_LITELLM_POLICY_MODEL"
DEFAULT_LITELLM_MODEL = "openai/gpt-5"


class LiteLLMAdapter:
    """`PolicyAdapter` impl wrapping litellm.

    Construction takes a `litellm_model_name` string of the form
    `provider/model` (e.g. `openai/gpt-5`, `anthropic/claude-opus-4-6`,
    `gemini/gemini-2.5-pro`, `hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct`).

    Like `BedrockPolicyAdapter`, this class does not own the rollout loop —
    it exposes the PolicyAdapter contract and a `complete()` helper that
    the rollout driver calls per turn.
    """

    def __init__(
        self,
        litellm_model_name: str | None = None,
        rollout_fn: Any | None = None,
        completion_fn: Any | None = None,
    ) -> None:
        self._policy_id = (
            litellm_model_name
            if litellm_model_name is not None
            else os.environ.get(DEFAULT_LITELLM_MODEL_ENV, DEFAULT_LITELLM_MODEL)
        )
        self._rollout_fn = rollout_fn
        self._completion_fn = completion_fn

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def rollout(self, instance_id: str, seed: int) -> PolicyResult:
        if self._rollout_fn is None:
            raise RuntimeError(
                "LiteLLMAdapter.rollout() requires a `rollout_fn` injected at "
                "construction. The real wiring lives in milo.lht_adapter.generator; "
                "tests must inject a stub."
            )
        result = self._rollout_fn(self, instance_id, seed)
        if not isinstance(result, PolicyResult):
            raise TypeError(
                f"rollout_fn must return PolicyResult, got {type(result).__name__}"
            )
        return result

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """One litellm completion call.

        Returns the OpenAI-compatible response dict. Tool-call parsing is the
        caller's job (use `OpenAIFunctionsParser`).
        """
        if self._completion_fn is not None:
            return self._completion_fn(
                model=self._policy_id, messages=messages, tools=tools,
                max_tokens=max_tokens, temperature=temperature, **kwargs,
            )
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "litellm not installed — install milo with the miniswe extra"
            ) from exc
        return litellm.completion(
            model=self._policy_id, messages=messages, tools=tools,
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        )


# Convenience subclasses — each pins a sensible default model id for the four
# providers we ship as canonical policy options per spec §15.3. AGIF or any
# other integrator can subclass further or just instantiate `LiteLLMAdapter`
# with their own model string.


class AnthropicLiteLLMAdapter(LiteLLMAdapter):
    """Anthropic via litellm. Default `${MILO_ANTHROPIC_MODEL:-anthropic/claude-opus-4-6}`."""

    def __init__(self, **kwargs: Any) -> None:
        default_model = os.environ.get(
            "MILO_ANTHROPIC_MODEL", "anthropic/claude-opus-4-6"
        )
        super().__init__(
            litellm_model_name=kwargs.pop("litellm_model_name", default_model),
            **kwargs,
        )


class OpenAILiteLLMAdapter(LiteLLMAdapter):
    """OpenAI via litellm. Default `${MILO_OPENAI_MODEL:-openai/gpt-5}`."""

    def __init__(self, **kwargs: Any) -> None:
        default_model = os.environ.get("MILO_OPENAI_MODEL", "openai/gpt-5")
        super().__init__(
            litellm_model_name=kwargs.pop("litellm_model_name", default_model),
            **kwargs,
        )


class GeminiLiteLLMAdapter(LiteLLMAdapter):
    """Gemini via litellm. Default `${MILO_GEMINI_MODEL:-gemini/gemini-2.5-pro}`."""

    def __init__(self, **kwargs: Any) -> None:
        default_model = os.environ.get(
            "MILO_GEMINI_MODEL", "gemini/gemini-2.5-pro"
        )
        super().__init__(
            litellm_model_name=kwargs.pop("litellm_model_name", default_model),
            **kwargs,
        )


class VLLMLiteLLMAdapter(LiteLLMAdapter):
    """vLLM OpenAI-compatible endpoint via litellm.

    Default model: `${MILO_VLLM_POLICY_MODEL:-hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct}`.
    The endpoint URL must be passed via `api_base=...` (litellm convention).
    """

    def __init__(self, api_base: str | None = None, **kwargs: Any) -> None:
        default_model = os.environ.get(
            "MILO_VLLM_POLICY_MODEL",
            "hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct",
        )
        super().__init__(
            litellm_model_name=kwargs.pop("litellm_model_name", default_model),
            **kwargs,
        )
        self._api_base = api_base or os.environ.get(
            "MILO_VLLM_API_BASE", "http://localhost:8000/v1"
        )

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("api_base", self._api_base)
        return super().complete(messages=messages, **kwargs)
