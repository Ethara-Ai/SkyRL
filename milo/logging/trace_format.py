"""Canonical trace-format dataclasses for the `milo_extension` overlay.

Implements RL_GYM_SPEC.md v0.7 §6.5 (trace schema = OpenHands rollout JSON
with `milo_extension` overlay) and IMPLEMENTATION_PLAN.md v0.4 §5.1. The
overlay is a single top-level key on the OpenHands rollout JSON, carrying
per-step shaping rewards and a terminal summary. The schema-version constant
`milo-trace/2.0` distinguishes this overlay-on-OpenHands layout from the
v1 standalone trace JSONL that the v0.6 spec described.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Schema version pinned for handoff to AGIF. Bumped to 2.0 when v0.7 reconciled
# the trace format to OpenHands+overlay; v1.0 was the old standalone JSONL.
MILO_TRACE_SCHEMA_VERSION: str = "milo-trace/2.0"


@dataclass(slots=True)
class StepShaping:
    """Per-step overlay data — one entry per gym step that we want to attribute
    a shaping reward to or stamp a working-tree patch on.

    Attributes
    ----------
    step_index:
        Zero-based index into the OpenHands `history` array of the corresponding
        ActionEvent. Aligns to OpenHands' step ordering so downstream consumers
        can join the overlay back to the rollout without a separate map.
    shaping_reward:
        Per-step `R_delta(t)` value (spec §4.4.2). Zero on steps where the
        test runner didn't execute.
    working_tree_patch:
        `git diff HEAD` at the end of step `step_index`. May be empty when the
        step made no filesystem mutation. Logged verbatim for offline replay.
    """

    step_index: int
    shaping_reward: float = 0.0
    working_tree_patch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "shaping_reward": float(self.shaping_reward),
            "working_tree_patch": self.working_tree_patch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepShaping:
        return cls(
            step_index=int(data.get("step_index", 0)),
            shaping_reward=float(data.get("shaping_reward", 0.0)),
            working_tree_patch=str(data.get("working_tree_patch", "")),
        )


@dataclass(slots=True)
class TerminalSummary:
    """Terminal-step overlay data — one record per rollout.

    Mirrors the `terminal_summary` block in spec §6.5 verbatim. Sub-records
    (verifier_report, rubric_report, reward_decomposition) are stored as
    dicts here rather than typed dataclasses so the overlay round-trips
    cleanly even if the upstream schemas drift between weekly merges.

    Attributes
    ----------
    termination_reason:
        One of `submit | timeout | tool_budget | container_error |
        cost_guardrail` (spec §4.5).
    verifier_report:
        Spec §6.6 — parsed verifier report (`milo.verifier.VerifierReport`)
        as a dict.
    rubric_report:
        Spec §6.7 — rubric report from `milo.judge.RubricReport.to_dict()`.
    reward_decomposition:
        Spec §6.8 — from `milo.reward.RewardDecomposition.to_dict()`.
    r_total:
        The composite reward scalar (spec §4.4). Redundant with
        `reward_decomposition["r_total"]` but stamped here for fast scans.
    cost_usd:
        Dollar cost of this rollout from the policy adapter's accounting.
    tokens:
        Dict with `prompt | completion | cache_read | cache_write` counts.
    """

    termination_reason: str
    verifier_report: dict[str, Any] = field(default_factory=dict)
    rubric_report: dict[str, Any] = field(default_factory=dict)
    reward_decomposition: dict[str, Any] = field(default_factory=dict)
    r_total: float = 0.0
    cost_usd: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "termination_reason": self.termination_reason,
            "verifier_report": dict(self.verifier_report),
            "rubric_report": dict(self.rubric_report),
            "reward_decomposition": dict(self.reward_decomposition),
            "r_total": float(self.r_total),
            "cost_usd": float(self.cost_usd),
            "tokens": dict(self.tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminalSummary:
        return cls(
            termination_reason=str(data.get("termination_reason", "")),
            verifier_report=dict(data.get("verifier_report") or {}),
            rubric_report=dict(data.get("rubric_report") or {}),
            reward_decomposition=dict(data.get("reward_decomposition") or {}),
            r_total=float(data.get("r_total", 0.0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            tokens=dict(data.get("tokens") or {}),
        )


@dataclass(slots=True)
class MiloExtension:
    """The full `milo_extension` overlay block.

    Attached as a single top-level key on the OpenHands rollout JSON (one
    line per rollout in the `.jsonl` file). Spec §6.5 lists the canonical
    field order; we preserve it in `to_dict()` for byte-stable serialisation.

    Attributes
    ----------
    schema_version:
        Pinned to `MILO_TRACE_SCHEMA_VERSION` ("milo-trace/2.0"). Consumers
        gate parsing on this string.
    gym_version:
        Semver of the gym build that produced the rollout. Stamped at
        recorder-construction time.
    reward_config:
        The `(alpha, beta, lambda, gamma, preset)` configuration in effect
        at rollout time. Captured so re-scoring is reproducible.
    per_step_shaping:
        List of `StepShaping` records, one per indexable step.
    terminal_summary:
        Single `TerminalSummary` record. `None` only on a rollout that
        was force-killed before any terminal summary could be computed —
        in that case the overlay is incomplete and downstream consumers
        should treat the rollout as corrupted.
    """

    schema_version: str = MILO_TRACE_SCHEMA_VERSION
    gym_version: str = ""
    reward_config: dict[str, Any] = field(default_factory=dict)
    per_step_shaping: list[StepShaping] = field(default_factory=list)
    terminal_summary: TerminalSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gym_version": self.gym_version,
            "reward_config": dict(self.reward_config),
            "per_step_shaping": [s.to_dict() for s in self.per_step_shaping],
            "terminal_summary": (
                self.terminal_summary.to_dict() if self.terminal_summary is not None else None
            ),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MiloExtension:
        per_step = [StepShaping.from_dict(s) for s in (data.get("per_step_shaping") or [])]
        terminal_raw = data.get("terminal_summary")
        terminal = TerminalSummary.from_dict(terminal_raw) if terminal_raw else None
        return cls(
            schema_version=str(data.get("schema_version", MILO_TRACE_SCHEMA_VERSION)),
            gym_version=str(data.get("gym_version", "")),
            reward_config=dict(data.get("reward_config") or {}),
            per_step_shaping=per_step,
            terminal_summary=terminal,
        )


def attach_overlay(
    openhands_rollout_dict: dict[str, Any],
    overlay: MiloExtension,
) -> dict[str, Any]:
    """Return a new dict equal to `openhands_rollout_dict` plus
    `"milo_extension": overlay.to_dict()` as a top-level key.

    Defensive copy: we don't mutate the caller's dict. Returns the new
    dict so the caller can write it back to disk in one shot.
    """

    if not isinstance(openhands_rollout_dict, dict):
        raise TypeError(
            f"openhands_rollout_dict must be a dict; got {type(openhands_rollout_dict).__name__}"
        )
    enriched = dict(openhands_rollout_dict)
    enriched["milo_extension"] = overlay.to_dict()
    return enriched


def extract_overlay(openhands_rollout_dict: dict[str, Any]) -> MiloExtension | None:
    """Inverse of `attach_overlay`. Returns `None` when no overlay is present.

    Useful for the SME / replay / calibration pipelines that need to read
    the overlay back without round-tripping through the recorder.
    """

    raw = openhands_rollout_dict.get("milo_extension")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"milo_extension is not a dict: {type(raw).__name__}"
        )
    return MiloExtension.from_dict(raw)


__all__ = [
    "MILO_TRACE_SCHEMA_VERSION",
    "MiloExtension",
    "StepShaping",
    "TerminalSummary",
    "attach_overlay",
    "extract_overlay",
]
