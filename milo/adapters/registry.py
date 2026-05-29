"""Policy-adapter registry — Phase 7 / plan §15.

`register_adapter(name, cls)` lets integrators plug their own
`PolicyAdapter` implementations in without touching milo's source. The
five shipped adapters (Bedrock + 4 via litellm) auto-register at module
import; AGIF (or any other integrator) adds theirs via:

    from milo.adapters.registry import register_adapter
    from my_company.adapters import MyCustomAdapter
    register_adapter("my_custom", MyCustomAdapter)

Then anywhere in the pipeline:

    from milo.adapters.registry import get_adapter
    cls = get_adapter("my_custom")
    adapter = cls(...)

The dispatch is intentionally string-keyed (not class-keyed) so the
adapter name can live in a config file (`milo/config/extensions.py` /
CLI override) without requiring a Python import on the config side.
"""

from __future__ import annotations

from typing import Any, Callable, Type

from milo.adapters.base import PolicyAdapter
from milo.adapters.bedrock import BedrockPolicyAdapter
from milo.adapters.litellm_adapter import (
    AnthropicLiteLLMAdapter,
    GeminiLiteLLMAdapter,
    LiteLLMAdapter,
    OpenAILiteLLMAdapter,
    VLLMLiteLLMAdapter,
)


_REGISTRY: dict[str, Type[Any]] = {}


def register_adapter(name: str, cls: Type[Any], *, override: bool = False) -> None:
    """Register a PolicyAdapter subclass under `name`.

    Raises `ValueError` if `name` is already registered unless
    `override=True`.
    """
    name = name.lower()
    if name in _REGISTRY and not override:
        raise ValueError(
            f"adapter {name!r} already registered ({_REGISTRY[name].__name__}); "
            f"pass override=True to replace"
        )
    _REGISTRY[name] = cls


def get_adapter(name: str) -> Type[Any]:
    """Look up an adapter class by name. Raises KeyError if absent."""
    name = name.lower()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_adapters() -> list[str]:
    """Names of all registered adapters."""
    return sorted(_REGISTRY)


# ---------- Default registrations ----------

# Direct-boto3 Bedrock adapter (the only non-litellm shipped option).
register_adapter("bedrock", BedrockPolicyAdapter)

# 4 litellm-backed adapters. `litellm` is the generic catch-all; the four
# named ones pin sensible defaults per provider.
register_adapter("litellm", LiteLLMAdapter)
register_adapter("anthropic", AnthropicLiteLLMAdapter)
register_adapter("openai", OpenAILiteLLMAdapter)
register_adapter("gemini", GeminiLiteLLMAdapter)
register_adapter("vllm", VLLMLiteLLMAdapter)
