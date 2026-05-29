"""AWS Bedrock Converse-API backend for the rubric judge.

Default production judge backend per RL_GYM_SPEC.md v0.7 §5.3. Talks to
Bedrock via `boto3.client('bedrock-runtime').converse(...)`. The model id
is env-driven (`${MILO_JUDGE_MODEL:-claude-opus-4-6}`) so swapping to a
different judge is one env-var. The boto3 client is lazy-imported so unit
tests that stub the backend never need AWS credentials.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from milo.judge.backends.base import JudgeBackend, JudgeBackendError

if TYPE_CHECKING:  # pragma: no cover - import-time guard for type checkers
    pass


# Reasonable default for Bedrock: ap-south-1 matches the rest of the
# Ethara/Milo deployment (per spec §5.1 / multiswebench ECR registry).
_DEFAULT_REGION = "ap-south-1"


def _default_model_id() -> str:
    """Resolve the default judge model from the env, falling back to today's
    GA Claude Opus on Bedrock. Spec §5.3 mandates env-driven model ids.
    """

    return os.environ.get("MILO_JUDGE_MODEL", "claude-opus-4-6")


class BedrockJudgeBackend:
    """`JudgeBackend` implementation backed by AWS Bedrock Converse.

    Parameters
    ----------
    region_name:
        AWS region. Defaults to `${AWS_REGION:-ap-south-1}`.
    client:
        Optional pre-built boto3 client (or any duck-typed object exposing
        `.converse(**kwargs)`). Injectable for unit tests.
    max_tokens:
        Cap on completion length. Judge replies are short JSON — 4096 is
        ample.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._region_name = region_name or os.environ.get("AWS_REGION", _DEFAULT_REGION)
        self._client = client
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # boto3 client lifecycle
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """Lazily build the boto3 client. We avoid the import at module
        load time so callers without AWS credentials (CI on a Mac without
        boto3 installed) can still import the module.
        """

        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise JudgeBackendError(
                "boto3 is required for BedrockJudgeBackend; install via `pip install boto3`"
            ) from exc
        self._client = boto3.client("bedrock-runtime", region_name=self._region_name)
        return self._client

    # ------------------------------------------------------------------
    # JudgeBackend protocol
    # ------------------------------------------------------------------
    def call(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
        """Round-trip a single judge call via Bedrock Converse."""

        client = self._get_client()
        try:
            response = client.converse(
                modelId=model or _default_model_id(),
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                inferenceConfig={
                    "temperature": float(temperature),
                    "maxTokens": self._max_tokens,
                },
            )
        except Exception as exc:  # narrow once boto3 is in scope
            raise JudgeBackendError(f"Bedrock converse() failed: {exc}") from exc

        return _extract_text_from_converse_response(response)


def _extract_text_from_converse_response(response: dict[str, Any]) -> str:
    """Pull the assistant text out of a Bedrock Converse response.

    The Bedrock `converse` response shape is:
        {"output": {"message": {"role": "assistant",
                                "content": [{"text": "..."}, ...]}}, ...}
    We concatenate every text block, skipping non-text blocks (tool_use,
    image, etc.) which the judge prompt never solicits.
    """

    try:
        message = response["output"]["message"]
        content_blocks = message["content"]
    except (KeyError, TypeError) as exc:
        raise JudgeBackendError(
            f"Unexpected Bedrock converse() response shape: {response!r}"
        ) from exc

    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    if not parts:
        raise JudgeBackendError(
            f"Bedrock converse() returned no text blocks: {content_blocks!r}"
        )
    return "".join(parts)


# Sanity-check at import: the concrete class implements the Protocol.
_: JudgeBackend = BedrockJudgeBackend()  # type: ignore[assignment]


__all__ = ["BedrockJudgeBackend"]
