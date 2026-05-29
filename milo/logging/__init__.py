"""Logging subsystem — Phase 5 of the Milo-bench RL plan.

Implements the OpenHands rollout-JSON overlay (`milo_extension`), the local
per-rollout recorder, the async S3 syncer, and the one-shot legacy-trajectory
converter. See RL_GYM_SPEC.md v0.7 §5.5 (logging hierarchy) and §6.5 (trace
schema = OpenHands + `milo_extension`) and IMPLEMENTATION_PLAN.md v0.4
Phase 5 for the full contract.
"""

from __future__ import annotations

from milo.logging.recorder import RolloutRecorder
from milo.logging.s3_syncer import S3Syncer
from milo.logging.trace_format import (
    MILO_TRACE_SCHEMA_VERSION,
    MiloExtension,
    StepShaping,
    TerminalSummary,
    attach_overlay,
)

__all__ = [
    "MILO_TRACE_SCHEMA_VERSION",
    "MiloExtension",
    "RolloutRecorder",
    "S3Syncer",
    "StepShaping",
    "TerminalSummary",
    "attach_overlay",
]
