"""Judge backend implementations — pluggable inference providers.

The judge service speaks to its underlying LLM through the `JudgeBackend`
Protocol (`base.py`). Two reference backends ship: `BedrockJudgeBackend`
(production, AWS Bedrock Converse) and `AnthropicJudgeBackend` (direct
Anthropic API fallback). Both honour the `${MILO_JUDGE_MODEL}` env override
per RL_GYM_SPEC.md v0.7 §5.3.
"""

from __future__ import annotations

from milo.judge.backends.base import JudgeBackend, JudgeBackendError

__all__ = ["JudgeBackend", "JudgeBackendError"]
