"""Smoke tests for Bedrock + LiteLLM adapters using injected stubs (no network)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from milo.adapters.base import PolicyResult
from milo.adapters.bedrock import BedrockPolicyAdapter
from milo.adapters.litellm_adapter import (
    AnthropicLiteLLMAdapter,
    GeminiLiteLLMAdapter,
    LiteLLMAdapter,
    OpenAILiteLLMAdapter,
    VLLMLiteLLMAdapter,
)


def _make_stub_result(passed: bool) -> PolicyResult:
    return PolicyResult(
        instance_id="x",
        seed=0,
        passed=passed,
        reward_decomposition={"r_total": 1.0 if passed else 0.0},
    )


# ---------------- Bedrock ----------------


def test_bedrock_policy_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILO_POLICY_MODEL", "anthropic.claude-opus-4-6")
    a = BedrockPolicyAdapter()
    assert a.policy_id == "anthropic.claude-opus-4-6"


def test_bedrock_rollout_uses_injected_fn() -> None:
    captured: list[tuple[str, int]] = []

    def fake(adapter: Any, instance_id: str, seed: int) -> PolicyResult:
        captured.append((instance_id, seed))
        return _make_stub_result(True)

    a = BedrockPolicyAdapter(model="m", rollout_fn=fake)
    res = a.rollout("inst-1", 7)
    assert res.passed is True
    assert captured == [("inst-1", 7)]


def test_bedrock_rollout_without_fn_raises() -> None:
    a = BedrockPolicyAdapter(model="m")
    with pytest.raises(RuntimeError):
        a.rollout("inst", 0)


def test_bedrock_converse_uses_injected_client() -> None:
    @dataclass
    class FakeClient:
        last: dict[str, Any] | None = None

        def converse(self, **kwargs: Any) -> dict[str, Any]:
            self.last = kwargs
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    fc = FakeClient()
    a = BedrockPolicyAdapter(model="m", region="us-east-1", client=fc)
    response = a.call_converse(
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        max_tokens=512,
        temperature=0.0,
    )
    assert response["output"]["message"]["content"][0]["text"] == "ok"
    assert fc.last is not None
    assert fc.last["modelId"] == "m"
    assert fc.last["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.0}


# ---------------- LiteLLM family ----------------


def test_litellm_default_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILO_LITELLM_POLICY_MODEL", "openai/gpt-5")
    a = LiteLLMAdapter()
    assert a.policy_id == "openai/gpt-5"


def test_anthropic_subclass_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MILO_ANTHROPIC_MODEL", raising=False)
    a = AnthropicLiteLLMAdapter()
    assert a.policy_id == "anthropic/claude-opus-4-6"


def test_openai_subclass_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILO_OPENAI_MODEL", "openai/gpt-5-mini")
    a = OpenAILiteLLMAdapter()
    assert a.policy_id == "openai/gpt-5-mini"


def test_gemini_subclass_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MILO_GEMINI_MODEL", raising=False)
    a = GeminiLiteLLMAdapter()
    assert a.policy_id == "gemini/gemini-2.5-pro"


def test_vllm_subclass_uses_api_base() -> None:
    seen: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    a = VLLMLiteLLMAdapter(api_base="http://host:8000/v1", completion_fn=fake_completion)
    a.complete(messages=[{"role": "user", "content": "hi"}])
    assert seen["api_base"] == "http://host:8000/v1"


def test_litellm_rollout_uses_injected_fn() -> None:
    def fake(adapter: Any, iid: str, seed: int) -> PolicyResult:
        return _make_stub_result(False)
    a = LiteLLMAdapter(litellm_model_name="x", rollout_fn=fake)
    assert a.rollout("i", 1).passed is False
