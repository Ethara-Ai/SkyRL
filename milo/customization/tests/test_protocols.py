"""Tests asserting the shipped default impls satisfy the spec §29 Protocols."""

from __future__ import annotations

from milo.adapters.tool_call_parsers import (
    LlamaToolCallParser,
    OpenAIFunctionsParser,
    QwenToolCallParser,
)
from milo.customization.protocols import (
    ObservabilityBackend,
    ServingAdapter,
    ToolCallParser,
    TrainerAlgo,
    TrainerStack,
)


def test_tool_call_parsers_conform_to_protocol() -> None:
    assert isinstance(OpenAIFunctionsParser(), ToolCallParser)
    assert isinstance(QwenToolCallParser(), ToolCallParser)
    assert isinstance(LlamaToolCallParser(), ToolCallParser)


def test_protocols_importable_from_package_init() -> None:
    """`from milo.customization import TrainerStack` must work for integrators."""
    from milo.customization import (
        ObservabilityBackend as OB,
        ServingAdapter as SA,
        ToolCallParser as TCP,
        TrainerAlgo as TA,
        TrainerStack as TS,
    )
    assert OB is ObservabilityBackend
    assert SA is ServingAdapter
    assert TCP is ToolCallParser
    assert TA is TrainerAlgo
    assert TS is TrainerStack
