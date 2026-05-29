"""Reward decomposition record — spec §6.8.

Implements the `RewardDecomposition` dataclass that the composite reward
aggregator emits per episode. The JSON schema returned by `to_json()` is
byte-for-byte the §6.8 contract that RL engineers consume for debugging,
and the `components` dict pre-computes each weighted contribution so a reader
can see the channel-by-channel breakdown of `r_total` at a glance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Spec §4.4 ships two presets.
RewardPreset = Literal["composite", "pure_rlvr"]


@dataclass(slots=True)
class RewardDecomposition:
    """One reward record per episode — the full §6.8 schema.

    Fields are kept in the *exact* order of the spec §6.8 JSON block so
    `to_json()` round-trips with predictable key ordering. The redundancy
    between the raw per-knob fields (`r_delta_sum`, `r_rubric_mean`, ...) and
    the pre-multiplied `components` map is intentional — readers should not
    have to re-do the alpha/beta/gamma arithmetic to read the artifact.

    Attributes
    ----------
    preset:
        Which named preset produced this record. Matches the YAML filename
        under `milo/reward/presets/` minus the `.yaml`.
    r_terminal:
        Verifier terminal reward, 0 or 1 (spec §4.4.1). Forced to 0 when any
        invariant in §7 fails — the aggregator does the forcing, not us.
    r_delta_steps:
        Per-step `R_delta(t)` values, one per step where the test runner
        actually executed. Empty on episodes that never ran tests.
    r_delta_sum:
        `sum(r_delta_steps)`. Cached so consumers don't recompute it.
    r_rubric_per_item:
        Per-item rubric scores in `{0, 0.5, 1}` (spec §4.4.3 / §6.7).
    r_rubric_mean:
        `mean(r_rubric_per_item)` or 0.0 when the rubric was disabled / empty.
    r_tir_steps:
        Per-step `R_tir(t)` values in `{-1, 0}` (spec §4.4.7).
    r_tir_sum:
        `sum(r_tir_steps)`. Always `<= 0` by the TIR definition.
    alpha, beta, lambda_, gamma:
        Reward weights / shaping asymmetry — exactly the knobs in the preset
        YAML. `lambda_` is named with the trailing underscore to avoid
        clashing with the Python builtin while still serializing as `lambda`
        in the JSON output (handled in `to_json()`).
    r_total:
        The full composite score (spec §4.4 formula). Not clipped.
    components:
        Pre-computed weighted breakdown — `{terminal, shaping, rubric, tir}` —
        such that `r_total == sum(components.values())` to within floating
        point. The aggregator fills this in.
    """

    preset: RewardPreset
    r_terminal: int
    r_delta_steps: list[float]
    r_delta_sum: float
    r_rubric_per_item: list[float]
    r_rubric_mean: float
    r_tir_steps: list[int]
    r_tir_sum: float
    alpha: float
    beta: float
    lambda_: float
    gamma: float
    r_total: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the §6.8 JSON-compatible dict.

        Renames the dataclass field `lambda_` back to the spec-mandated key
        `lambda`. Field ordering matches the spec block exactly.
        """
        return {
            "preset": self.preset,
            "r_terminal": int(self.r_terminal),
            "r_delta_steps": list(self.r_delta_steps),
            "r_delta_sum": float(self.r_delta_sum),
            "r_rubric_per_item": list(self.r_rubric_per_item),
            "r_rubric_mean": float(self.r_rubric_mean),
            "r_tir_steps": list(self.r_tir_steps),
            "r_tir_sum": float(self.r_tir_sum),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "lambda": float(self.lambda_),
            "gamma": float(self.gamma),
            "r_total": float(self.r_total),
            "components": {
                "terminal": float(self.components.get("terminal", 0.0)),
                "shaping": float(self.components.get("shaping", 0.0)),
                "rubric": float(self.components.get("rubric", 0.0)),
                "tir": float(self.components.get("tir", 0.0)),
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string in the exact §6.8 key order."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardDecomposition:
        """Inverse of `to_dict()`. Handles the `lambda` → `lambda_` rename."""
        components = data.get("components") or {}
        return cls(
            preset=data["preset"],
            r_terminal=int(data["r_terminal"]),
            r_delta_steps=list(data.get("r_delta_steps", [])),
            r_delta_sum=float(data.get("r_delta_sum", 0.0)),
            r_rubric_per_item=list(data.get("r_rubric_per_item", [])),
            r_rubric_mean=float(data.get("r_rubric_mean", 0.0)),
            r_tir_steps=list(data.get("r_tir_steps", [])),
            r_tir_sum=float(data.get("r_tir_sum", 0.0)),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            lambda_=float(data.get("lambda", data.get("lambda_", 0.0))),
            gamma=float(data["gamma"]),
            r_total=float(data["r_total"]),
            components={
                "terminal": float(components.get("terminal", 0.0)),
                "shaping": float(components.get("shaping", 0.0)),
                "rubric": float(components.get("rubric", 0.0)),
                "tir": float(components.get("tir", 0.0)),
            },
        )

    # Convenience: dataclass asdict produces the field names verbatim, which
    # we want for debugging but NOT for spec-compliant serialization.
    def asdict_raw(self) -> dict[str, Any]:
        """Raw dataclass dict — uses `lambda_` (not `lambda`). For debugging."""
        return asdict(self)
