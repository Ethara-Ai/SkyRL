"""milo.replay — deterministic trace replay + golden-trace tooling (Phase 9).

Implements ``RL_GYM_SPEC.md`` v0.7 §10.3 ("Deterministic replay tool")
and ``IMPLEMENTATION_PLAN.md`` v0.4 Phase 9 (``gym/replay/replay.py``).

Three modes per spec §10.3 + plan §9.1:

* **Default replay** — re-executes the stored tool-call sequence inside a
  fresh container with the same seed and asserts that the resulting
  ``reward_decomposition`` matches the recorded one to within
  :data:`REWARD_TOLERANCE`. Required for reproducibility audits (spec
  §23.4) and CI (spec §10.4).
* **``--record-golden``** — drops into human-in-the-loop mode: the SME
  drives the rollout via the gym REPL, and the trace is persisted as a
  golden trace (spec §6.5 + §10.1.3).
* **``--golden``** — replays a stored golden trace and asserts
  ``R_terminal == 1`` and ``R_rubric == 1.0`` per spec §10.1.3 golden-trace
  certification.

Public entry points:

    ``milo replay`` CLI       — ``python -m milo.replay.cli``
    ``replay_trace()``        — programmatic replay used by nightly audit
    ``SmokePack``             — 5-task smoke pack runner (plan §9.2)
"""

from __future__ import annotations

from milo.replay.cli import (
    REWARD_TOLERANCE,
    ReplayMismatch,
    ReplayResult,
    replay_trace,
)
from milo.replay.smoke_pack import SmokePack, SmokeTask

__all__ = [
    "REWARD_TOLERANCE",
    "ReplayMismatch",
    "ReplayResult",
    "replay_trace",
    "SmokePack",
    "SmokeTask",
]
