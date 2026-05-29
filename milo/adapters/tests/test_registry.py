"""Tests for milo.adapters.registry."""

from __future__ import annotations

import pytest

from milo.adapters import bedrock, litellm_adapter, registry


def test_default_registrations_present() -> None:
    names = registry.list_adapters()
    assert "bedrock" in names
    assert "litellm" in names
    assert "anthropic" in names
    assert "openai" in names
    assert "gemini" in names
    assert "vllm" in names


def test_get_returns_class() -> None:
    assert registry.get_adapter("bedrock") is bedrock.BedrockPolicyAdapter
    assert registry.get_adapter("openai") is litellm_adapter.OpenAILiteLLMAdapter


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        registry.get_adapter("nonexistent")


def test_register_then_get() -> None:
    class _Dummy:
        @property
        def policy_id(self) -> str: return "dummy"
        def rollout(self, instance_id: str, seed: int) -> object:  # type: ignore[override]
            return None
    registry.register_adapter("dummy_one", _Dummy)
    assert registry.get_adapter("dummy_one") is _Dummy


def test_double_register_raises_without_override() -> None:
    class _Dummy:
        @property
        def policy_id(self) -> str: return "dummy"
        def rollout(self, instance_id: str, seed: int) -> object:  # type: ignore[override]
            return None
    registry.register_adapter("dummy_two", _Dummy)
    with pytest.raises(ValueError):
        registry.register_adapter("dummy_two", _Dummy)
    # but override=True works
    registry.register_adapter("dummy_two", _Dummy, override=True)
