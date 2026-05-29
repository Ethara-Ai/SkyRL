"""CalibrationRunner — pass@k difficulty binning per spec §8.

Implements ``RL_GYM_SPEC.md`` v0.7 §8 and ``IMPLEMENTATION_PLAN.md`` v0.4
§8.1. The procedure is fixed by the spec:

1. Run pass@8 against two frontier reference models (default
   ``${MILO_JUDGE_MODEL:-claude-opus-4-6}`` and
   ``${MILO_CALIBRATION_MODEL_2:-gemini-2.5-pro}``).
2. Compute per-model pass rate.
3. Mean across the two models = the canonical task pass-rate.
4. Bin by mean:
       ≥ 0.60 → trivial
       0.20–0.60 → medium
       0.05–0.20 → hard
       < 0.05 → unsolvable
5. Reject tasks whose two-model pass rates disagree by > 0.30 — those
   are noisy and not informative.

The runner is intentionally tiny: it delegates rollouts to a
:class:`~milo.adapters.PolicyAdapter` (real ones land in Phase 7; the
Phase 8 tests use :class:`milo.adapters.StubPolicyAdapter`) so the
calibration logic itself is pure and unit-testable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from milo.adapters import PolicyAdapter

# Spec §8 binning bounds. Centralized as module constants so the CLI, the
# Phase 16 evaluator (which reports pass@8 per tier), and the dashboards
# all use one source of truth.
TIER_BOUNDS: dict[str, tuple[float, float]] = {
    # tier: (lower_inclusive, upper_exclusive). Upper bound is +inf for
    # 'trivial' since spec §8 only specifies the lower edge (≥ 0.60).
    "trivial": (0.60, float("inf")),
    "medium": (0.20, 0.60),
    "hard": (0.05, 0.20),
    "unsolvable": (0.0, 0.05),
}

# Spec §8 step 4: max allowed disagreement between the two reference models
# in absolute pass-rate (e.g. one model passes 80%, the other 40% → 0.40
# disagreement → rejected).
DISAGREEMENT_THRESHOLD = 0.30

# Default model handles per spec §8 and plan §8.1. Both are env-driven so
# operators can swap without code edits (spec §8 re-calibration policy).
DEFAULT_MODEL_1_ENV = "MILO_JUDGE_MODEL"
DEFAULT_MODEL_1_FALLBACK = "claude-opus-4-6"
DEFAULT_MODEL_2_ENV = "MILO_CALIBRATION_MODEL_2"
DEFAULT_MODEL_2_FALLBACK = "gemini-2.5-pro"

# Spec §8: pass@8 — k is the standard SWE-bench convention. Plan §8.1
# constructor signature pins k=8 as the default but allows override.
DEFAULT_K = 8


DifficultyTier = Literal["trivial", "medium", "hard", "unsolvable"]
RejectedReason = Literal["disagreement", ""]


def default_model_handles() -> tuple[str, str]:
    """Resolve the env-driven default two-model pair.

    Centralized so the CLI, runner, and recalibrate script all agree. The
    two env vars match spec §8 / plan §8.1 verbatim.
    """
    return (
        os.environ.get(DEFAULT_MODEL_1_ENV, DEFAULT_MODEL_1_FALLBACK),
        os.environ.get(DEFAULT_MODEL_2_ENV, DEFAULT_MODEL_2_FALLBACK),
    )


def _now_iso() -> str:
    """ISO 8601 UTC stamp; centralized so all artifacts use the same format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class TaskCalibration:
    """Per-task calibration result — spec §8 + plan §8.1 dataclass shape.

    Mirrors plan §8.1 ``TaskCalibration`` exactly with two operational
    additions:

    * ``rejected_reason`` — empty string when accepted, ``"disagreement"``
      when the two reference models' pass rates differed by more than
      :data:`DISAGREEMENT_THRESHOLD`. The on-disk JSON keeps this string
      because the W&B dashboard groups bars by reason and ``None`` doesn't
      group cleanly.
    * ``k`` — the pass@k k. Always 8 in production per spec §8 ("Why
      pass@8") but exposed because the CLI accepts an override for
      cost-constrained re-calibration runs.

    The ``model_1_pass_rate`` / ``model_2_pass_rate`` naming matches the
    prompt-mandated output schema; the dict ``per_model_pass_at_k`` mirrors
    plan §8.1 verbatim so callers can iterate models without knowing
    which slot is "1" and which is "2".
    """

    instance_id: str
    per_model_pass_at_k: dict[str, float]
    mean_pass_at_k: float
    tier: DifficultyTier | Literal["REJECT_DISAGREEMENT"]
    disagreement: float
    rejected_for_disagreement: bool
    rejected_reason: RejectedReason
    calibration_date: str
    k: int
    model_handles: tuple[str, str] = field(default_factory=tuple)  # type: ignore[arg-type]

    @property
    def model_1_pass_rate(self) -> float:
        """First-listed model's pass rate. Spec §8 names this as Claude Opus."""
        if not self.model_handles:
            # Fall back to dict iteration order (Python ≥ 3.7 guarantees
            # insertion order). Defensive — model_handles is always set
            # by the runner.
            return next(iter(self.per_model_pass_at_k.values()))
        return self.per_model_pass_at_k[self.model_handles[0]]

    @property
    def model_2_pass_rate(self) -> float:
        """Second-listed model's pass rate. Spec §8 names this as Gemini Pro."""
        if not self.model_handles:
            vals = list(self.per_model_pass_at_k.values())
            return vals[1] if len(vals) > 1 else vals[0]
        return self.per_model_pass_at_k[self.model_handles[1]]

    def to_results_entry(self) -> dict[str, Any]:
        """Render the per-task entry for ``calibration_results.json``.

        Schema matches the prompt's mandated output:

            {tier, model_1_pass_rate, model_2_pass_rate, disagreement,
             rejected_reason}

        plus the auxiliary fields needed downstream (calibration_date,
        mean_pass_at_k, k, model_handles) so the Phase 16 evaluator can
        consume the file without a second lookup.
        """
        return {
            "tier": self.tier,
            "model_1_pass_rate": self.model_1_pass_rate,
            "model_2_pass_rate": self.model_2_pass_rate,
            "disagreement": self.disagreement,
            "rejected_reason": self.rejected_reason,
            "mean_pass_at_k": self.mean_pass_at_k,
            "rejected_for_disagreement": self.rejected_for_disagreement,
            "calibration_date": self.calibration_date,
            "k": self.k,
            "model_handles": list(self.model_handles),
            "per_model_pass_at_k": dict(self.per_model_pass_at_k),
        }


def assign_tier(
    per_model_pass_rates: dict[str, float],
    *,
    disagreement_threshold: float = DISAGREEMENT_THRESHOLD,
) -> tuple[DifficultyTier | Literal["REJECT_DISAGREEMENT"], float, RejectedReason]:
    """Pure binning per spec §8 — returns ``(tier, disagreement, reason)``.

    Pulled out as a standalone function so the unit tests (plan §8.1
    test (a)–(c) "easy / disagreement / boundary") can exercise it without
    spinning up a runner. The boundary semantics are pinned by the spec:

    * 0.60 → trivial (≥ 0.60)
    * 0.20 → medium (≥ 0.20 and < 0.60)
    * 0.05 → hard (≥ 0.05 and < 0.20)
    * < 0.05 → unsolvable

    so a task at mean 0.60 → trivial (not medium), and 0.05 → hard
    (not unsolvable). These boundary cases are tested.

    The disagreement check fires *before* binning per spec §8 step 4: a
    rejected task does not get a tier label at all (the spec says "reject
    rather than mis-bin").
    """
    if not per_model_pass_rates:
        raise ValueError("per_model_pass_rates must contain at least one model")
    rates = list(per_model_pass_rates.values())
    for handle, rate in per_model_pass_rates.items():
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"pass rate for {handle!r} out of range: {rate!r} (expected [0, 1])"
            )

    disagreement = max(rates) - min(rates) if len(rates) > 1 else 0.0

    if disagreement > disagreement_threshold:
        # Spec §8 step 4 — reject before binning. The mean is still recorded
        # for forensic comparison but the tier label is REJECT_DISAGREEMENT.
        return "REJECT_DISAGREEMENT", disagreement, "disagreement"

    mean = sum(rates) / len(rates)
    for tier_name, (lower, upper) in TIER_BOUNDS.items():
        if lower <= mean < upper:
            return tier_name, disagreement, ""  # type: ignore[return-value]
    # Defensive: TIER_BOUNDS covers [0, +inf); if we ever fall through,
    # mean is < 0 which assign_tier's range check would already have
    # rejected. Raise loudly rather than silently returning unsolvable.
    raise AssertionError(
        f"mean pass rate {mean!r} did not match any tier bound (bug in TIER_BOUNDS)"
    )


def _pass_at_k(rollouts: Sequence[bool]) -> float:
    """pass@k = ``mean(passed)`` over k independent rollouts.

    This is the SWE-bench convention used throughout spec §8 / §19. We use
    the simple per-task pass-rate (sometimes called pass@k with n=k); the
    unbiased pass@k estimator (Chen et al. 2021) collapses to the same
    number when k == n_rollouts, which is always the case here.
    """
    if not rollouts:
        raise ValueError("cannot compute pass@k over empty rollouts")
    return sum(1 for r in rollouts if r) / len(rollouts)


class CalibrationRunner:
    """Drives pass@k against two reference models per spec §8.

    Constructor takes a list of :class:`~milo.adapters.PolicyAdapter`. The
    Phase 8 tests pass two :class:`~milo.adapters.StubPolicyAdapter`
    instances; production uses the real Phase 7 adapters
    (``AnthropicAdapter`` for Claude, ``GeminiAdapter`` for Gemini Pro).

    The runner is sync. Real callers will want concurrency across the
    300 × 2 × 8 = 4800 rollouts — the spec budgets ~$80K and a few days
    of wall clock at 64-way concurrency (spec §9.4) for the full
    calibration pass. The Phase 7 adapters expose async APIs; the
    production CLI can spin them up with ``asyncio.gather`` without
    touching this class.
    """

    def __init__(
        self,
        adapters: Sequence[PolicyAdapter],
        k: int = DEFAULT_K,
        *,
        seed_base: int = 0,
        disagreement_threshold: float = DISAGREEMENT_THRESHOLD,
    ) -> None:
        if len(adapters) < 2:
            raise ValueError(
                "spec §8 mandates two reference models; got "
                f"{len(adapters)} adapter(s)"
            )
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        # Spec §8 step 4 — the disagreement check is across the two models.
        # We support >2 models for future-proofing (e.g. tripling up when
        # GPT-5.3 Codex ships) but the default workflow is two.
        self.adapters: list[PolicyAdapter] = list(adapters)
        self.k = k
        self.seed_base = seed_base
        self.disagreement_threshold = disagreement_threshold

    @property
    def model_handles(self) -> tuple[str, ...]:
        return tuple(a.policy_id for a in self.adapters)

    def _rollouts_for_model(
        self,
        adapter: PolicyAdapter,
        instance_id: str,
    ) -> list[bool]:
        """Run ``k`` deterministic rollouts and return the per-rollout pass list.

        Seeds are ``seed_base, seed_base+1, ..., seed_base + k - 1``. Two
        adapters get the *same* seed sequence — this is intentional so the
        per-task pass-rate noise is structurally similar across providers
        (different policies, same seed schedule).
        """
        results: list[bool] = []
        for i in range(self.k):
            r = adapter.rollout(instance_id, seed=self.seed_base + i)
            results.append(bool(r.passed))
        return results

    def calibrate_task(self, instance_id: str) -> TaskCalibration:
        """Run the spec §8 procedure for one task.

        Returns the :class:`TaskCalibration` whether the task was accepted
        or rejected. Callers (``calibrate_all`` + the recalibrate CLI)
        inspect ``rejected_for_disagreement`` to bucket the output.
        """
        per_model: dict[str, float] = {}
        for adapter in self.adapters:
            passes = self._rollouts_for_model(adapter, instance_id)
            per_model[adapter.policy_id] = _pass_at_k(passes)

        tier, disagreement, reason = assign_tier(
            per_model, disagreement_threshold=self.disagreement_threshold
        )
        rejected = reason == "disagreement"
        mean = sum(per_model.values()) / len(per_model)
        return TaskCalibration(
            instance_id=instance_id,
            per_model_pass_at_k=per_model,
            mean_pass_at_k=mean,
            tier=tier,
            disagreement=disagreement,
            rejected_for_disagreement=rejected,
            rejected_reason=reason,
            calibration_date=_now_iso(),
            k=self.k,
            model_handles=self.model_handles[:2],  # first two only — see __init__
        )

    def calibrate_all(
        self,
        instance_ids: Iterable[str],
    ) -> dict[str, TaskCalibration]:
        """Convenience wrapper — calibrate a batch sequentially.

        Production code will want concurrency (see class docstring). The
        sequential path is what the unit tests + the smoke pack use.
        """
        out: dict[str, TaskCalibration] = {}
        for instance_id in instance_ids:
            out[instance_id] = self.calibrate_task(instance_id)
        return out

    def write_results(
        self,
        results: dict[str, TaskCalibration],
        output_path: Path,
    ) -> None:
        """Serialize a batch of calibrations to ``calibration_results.json``.

        Format matches the prompt:

            {
              "calibration_date": "...",
              "model_handles": ["claude-opus-4-6", "gemini-2.5-pro"],
              "k": 8,
              "disagreement_threshold": 0.30,
              "results": {
                "<instance_id>": {
                  "tier": "...",
                  "model_1_pass_rate": ...,
                  "model_2_pass_rate": ...,
                  "disagreement": ...,
                  "rejected_reason": "..."
                  ...
                }
              }
            }

        The top-level metadata block makes the file self-describing for
        the spec §8 re-calibration audit.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "milo-calibration/1.0",
            "calibration_date": _now_iso(),
            "model_handles": list(self.model_handles),
            "k": self.k,
            "disagreement_threshold": self.disagreement_threshold,
            "results": {
                iid: cal.to_results_entry() for iid, cal in results.items()
            },
            "summary": _summarize(results),
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )


def _summarize(results: dict[str, TaskCalibration]) -> dict[str, Any]:
    """Tier-distribution summary so the CLI can print a one-line report.

    Mirrors the AGIF distribution mandated by spec §8: 10/40/40/10. The
    summary is informational only — no caller takes action on it; the
    Phase 0.6 audit + the Phase 11 acceptance gate are the places where
    the distribution actually has to match.
    """
    counts: dict[str, int] = {
        "trivial": 0, "medium": 0, "hard": 0, "unsolvable": 0,
        "REJECT_DISAGREEMENT": 0,
    }
    for cal in results.values():
        counts[cal.tier] = counts.get(cal.tier, 0) + 1
    total = max(len(results), 1)
    return {
        "total_tasks": len(results),
        "tier_counts": counts,
        "tier_pct": {k: round(v / total, 4) for k, v in counts.items()},
        "rejected_count": counts.get("REJECT_DISAGREEMENT", 0),
    }
