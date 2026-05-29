"""milo.adapters — policy adapter Protocol stubs.

This module exists as a minimal contract surface so the calibration runner
(Phase 8), the replay tool (Phase 9), the eval harness (Phase 16), and the
nightly audit (Phase 17) can be written and unit-tested before the real
adapter implementations land in Phase 7 (per IMPLEMENTATION_PLAN.md §7 —
policy adapters via litellm + Bedrock-native).

The full adapter set per plan §7 / §15.3:

    | Adapter | Backend |
    |---|---|
    | AnthropicAdapter   | litellm     |
    | OpenAIAdapter      | litellm     |
    | GeminiAdapter      | litellm     |
    | BedrockAdapter     | boto3       |
    | VLLMAdapter        | OpenAI-compat |

For now we only declare the Protocol + a deterministic in-memory stub
implementation that the unit tests for Phases 8/9/16/17 can use without
needing GPU, network, or container access. The real adapters land in Phase 7.
"""

from __future__ import annotations

from milo.adapters.base import (
    PolicyAdapter,
    PolicyResult,
    StubPolicyAdapter,
    make_stub_adapter,
)

__all__ = [
    "PolicyAdapter",
    "PolicyResult",
    "StubPolicyAdapter",
    "make_stub_adapter",
]
