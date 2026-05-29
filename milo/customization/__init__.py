"""Phase 15 — customization Protocols + cookbook.

The five swap-point Protocols from spec §29 — TrainerStack, TrainerAlgo,
ServingAdapter, ToolCallParser, ObservabilityBackend. Re-importable here for
integrator convenience (`from milo.customization import TrainerStack`).
"""

from milo.customization.protocols import (
    ObservabilityBackend,
    ServingAdapter,
    ToolCallParser,
    TrainerAlgo,
    TrainerStack,
)

__all__ = [
    "TrainerStack",
    "TrainerAlgo",
    "ServingAdapter",
    "ToolCallParser",
    "ObservabilityBackend",
]
