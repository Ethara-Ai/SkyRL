"""Bedrock-native policy adapter — Phase 7 / spec §15.3.

The only Ethara-authored adapter that does not flow through litellm. We go
direct via boto3 because AGIF (the proposal's target customer) requires
Bedrock-native API access in `ap-south-1`, and because the rest of the
pipeline (judge model in `milo/judge/backends/bedrock.py`, calibration in
Phase 8) is already wired through boto3 — keeping the policy adapter on
the same path simplifies credential management and IAM.

Model id is env-driven via `${MILO_POLICY_MODEL:-anthropic.claude-opus-4-6}`
per `IMPLEMENTATION_PLAN.md` v0.4 conventions (no forward-dated hardcoded
identifiers — today's GA defaults only).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from milo.adapters.base import PolicyAdapter, PolicyResult

logger = logging.getLogger("milo.adapters.bedrock")


DEFAULT_MODEL_ENV = "MILO_POLICY_MODEL"
DEFAULT_MODEL = "anthropic.claude-opus-4-6"
DEFAULT_REGION_ENV = "MILO_BEDROCK_REGION"
DEFAULT_REGION = "ap-south-1"


@dataclass
class BedrockPolicyAdapter:
    """`PolicyAdapter` impl that calls Bedrock's `converse` / `converse_stream`.

    Tool-call parsing follows Bedrock's `toolUse` block schema; the
    rollout-loop driver in `milo/lht_adapter/generator.py` consumes these
    via the `ToolCall` shape from `milo.adapters.tool_call_parsers`.

    The class does not own the rollout loop — it only exposes the
    PolicyAdapter contract (`policy_id`, `rollout(...)`) so the calibration
    runner (Phase 8) and eval harness (Phase 16) can invoke it uniformly.
    """

    _policy_id: str = ""
    _region: str = ""
    _client: Any | None = None       # boto3 client; injectable for tests
    _rollout_fn: Any | None = None    # injectable for tests (sync callable)

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        client: Any | None = None,
        rollout_fn: Any | None = None,
    ) -> None:
        self._policy_id = (
            model
            if model is not None
            else os.environ.get(DEFAULT_MODEL_ENV, DEFAULT_MODEL)
        )
        self._region = (
            region
            if region is not None
            else os.environ.get(DEFAULT_REGION_ENV, DEFAULT_REGION)
        )
        self._client = client
        self._rollout_fn = rollout_fn

    # ------------------------------------------------------------------ Protocol

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def rollout(self, instance_id: str, seed: int) -> PolicyResult:
        """Invoke the configured rollout driver, return its PolicyResult.

        For unit tests, callers pass `rollout_fn` directly so we don't hit
        Bedrock. For production rollouts, the driver is set by
        `milo.lht_adapter.generator.MiloLHTGenerator` which constructs the
        adapter with `rollout_fn` bound to its own per-rollout coroutine.
        """
        if self._rollout_fn is None:
            raise RuntimeError(
                "BedrockPolicyAdapter.rollout() requires a `rollout_fn` injected at "
                "construction. The real wiring lives in milo.lht_adapter.generator; "
                "tests must inject a stub."
            )
        result = self._rollout_fn(self, instance_id, seed)
        if not isinstance(result, PolicyResult):
            raise TypeError(
                f"rollout_fn must return PolicyResult, got {type(result).__name__}"
            )
        return result

    # ---------------------------------------------------------------- internals

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 not installed — install milo with the boto3 extra "
                "or inject a `client` for tests."
            ) from exc
        self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def call_converse(
        self,
        messages: list[dict[str, Any]],
        system: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """One round-trip to Bedrock's `converse` API.

        Returns the raw response dict. Tool-call parsing is the caller's job
        (use `milo.adapters.tool_call_parsers.OpenAIFunctionsParser` or the
        Bedrock-specific helper here if you prefer the native shape).
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "modelId": self._policy_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = {"tools": tools}
        return client.converse(**kwargs)
