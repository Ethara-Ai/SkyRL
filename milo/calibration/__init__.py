"""milo.calibration — difficulty calibration runner (Phase 8).

Implements ``RL_GYM_SPEC.md`` v0.7 §8 and ``IMPLEMENTATION_PLAN.md`` v0.4
Phase 8 (``gym/calibration/runner.py`` and ``gym/calibration/recalibrate.py``).

Per spec §8: every task is calibrated by running pass@8 against two
frontier reference models (default Claude Opus + Gemini Pro, env-driven via
``${MILO_JUDGE_MODEL:-claude-opus-4-6}`` and
``${MILO_CALIBRATION_MODEL_2:-gemini-2.5-pro}``). The mean per-model pass
rate is binned into ``trivial | medium | hard | unsolvable``; tasks whose
two models disagree by > 30 pp are rejected as noisy.

Public entry points:

    CalibrationRunner       — programmatic API (Phase 8.1).
    TaskCalibration         — per-task result dataclass.
    assign_tier             — pure binning function (used by tests + CLI).
    recalibrate (CLI)       — ``python -m milo.calibration.recalibrate``.

The recalibration CLI is the operational lever for spec §8's
"Re-calibration policy": when a new frontier model lands (e.g. when
Anthropic ships Opus 4.7), ops swap the env vars + re-run, no code edit
required.
"""

from __future__ import annotations

from milo.calibration.runner import (
    CalibrationRunner,
    TaskCalibration,
    assign_tier,
)

__all__ = [
    "CalibrationRunner",
    "TaskCalibration",
    "assign_tier",
]
