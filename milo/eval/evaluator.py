"""Eval harness — Phase 16 / spec §19.

Computes pass@k overall + per-tier + per-language + full reward
decomposition mean. Runs paired bootstrap CI (`milo.eval.bootstrap_ci`)
on holdout pass@k delta vs the SFT baseline (`milo/data/baselines/sft.json`).
Release gate per `IMPLEMENTATION_PLAN.md` v0.4 §19.2: `point ≥ +5pp AND
bootstrap_lower > 0`.

Consumes any `PolicyAdapter` per `milo/adapters/base.py`. Stub adapter
(`StubPolicyAdapter`) lets unit tests exercise the harness without GPU.
"""

from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from milo.adapters.base import PolicyAdapter, PolicyResult
from milo.eval.baseline_format import BaselineReport
from milo.eval.bootstrap_ci import paired_bootstrap_ci


# ----------------------------------------------------------------- result type


@dataclass
class EvalReport:
    """Output of `MiloEvaluator.evaluate(...)`."""

    policy_id: str
    split: str                                # "holdout" | "train" | "smoke"
    k: int                                    # pass@k
    n_instances: int
    pass_at_k_overall: float
    pass_at_k_by_tier: dict[str, float] = field(default_factory=dict)
    pass_at_k_by_lang: dict[str, float] = field(default_factory=dict)
    mean_r_total: float = 0.0
    mean_r_terminal: float = 0.0
    mean_r_rubric: float = 0.0
    mean_r_delta_sum: float = 0.0
    mean_r_tir_sum: float = 0.0
    bootstrap_point: float | None = None      # vs baseline (None if no baseline)
    bootstrap_lower: float | None = None
    bootstrap_upper: float | None = None
    release_gate_passed: bool | None = None
    evaluation_date: str = ""
    evaluation_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# ------------------------------------------------------------------ evaluator


def _pass_at_k(rollouts: list[PolicyResult], k: int) -> float:
    """1 if any of the k rollouts passed; else 0. Aggregated across instances
    by the caller (mean across instances)."""
    if not rollouts:
        return 0.0
    return float(any(r.passed for r in rollouts[:k]))


class MiloEvaluator:
    """Eval-loop driver. Stateless per-call."""

    def __init__(self, split: str, k: int = 8) -> None:
        self.split = split
        self.k = k

    def evaluate(
        self,
        adapter: PolicyAdapter,
        instances: Iterable[dict[str, Any]],
        baselines: dict[str, Path] | None = None,
        bootstrap_baseline_key: str = "sft",
    ) -> EvalReport:
        """Run k rollouts per instance, score, return aggregated report.

        `instances` is an iterable of dicts with at least:
            {instance_id, difficulty_tier, lang}

        `baselines` maps baseline-name → JSON path written by `BaselineReport`.
        When `baselines[bootstrap_baseline_key]` is present, the report
        includes the paired-bootstrap CI on the (this_run - baseline) per-tier
        pass@k delta and the release gate verdict.
        """
        instances = list(instances)
        by_instance: list[list[PolicyResult]] = []
        for inst in instances:
            iid = inst["instance_id"]
            rollouts = [adapter.rollout(iid, seed=s) for s in range(self.k)]
            by_instance.append(rollouts)

        n = len(by_instance)
        per_instance_pass = [_pass_at_k(rs, self.k) for rs in by_instance]
        overall = statistics.fmean(per_instance_pass) if per_instance_pass else 0.0

        # Per-tier and per-lang breakdowns.
        by_tier: dict[str, list[float]] = {}
        by_lang: dict[str, list[float]] = {}
        for inst, p in zip(instances, per_instance_pass):
            tier = str(inst.get("difficulty_tier") or "unknown")
            lang = str(inst.get("lang") or "unknown")
            by_tier.setdefault(tier, []).append(p)
            by_lang.setdefault(lang, []).append(p)
        tier_means = {t: statistics.fmean(v) for t, v in by_tier.items() if v}
        lang_means = {ll: statistics.fmean(v) for ll, v in by_lang.items() if v}

        # Reward decomposition means (sampled from rollout[0] of each task).
        first = [rs[0] for rs in by_instance if rs]
        def _mean_field(key: str) -> float:
            vals = [
                float((r.reward_decomposition or {}).get(key, 0.0)) for r in first
            ]
            return statistics.fmean(vals) if vals else 0.0

        report = EvalReport(
            policy_id=adapter.policy_id,
            split=self.split,
            k=self.k,
            n_instances=n,
            pass_at_k_overall=overall,
            pass_at_k_by_tier=tier_means,
            pass_at_k_by_lang=lang_means,
            mean_r_total=_mean_field("r_total"),
            mean_r_terminal=_mean_field("r_terminal"),
            mean_r_rubric=_mean_field("r_rubric_mean"),
            mean_r_delta_sum=_mean_field("r_delta_sum"),
            mean_r_tir_sum=_mean_field("r_tir_sum"),
            evaluation_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            evaluation_run_id=uuid.uuid4().hex,
        )

        if baselines and bootstrap_baseline_key in baselines:
            baseline = BaselineReport.read_json(baselines[bootstrap_baseline_key])
            # Per-tier deltas (this run minus baseline) feed the paired bootstrap.
            tiers = sorted(set(tier_means) | set(baseline.pass_at_k_by_tier))
            deltas = [
                tier_means.get(t, 0.0) - baseline.pass_at_k_by_tier.get(t, 0.0)
                for t in tiers
            ]
            point, lo, hi = paired_bootstrap_ci(deltas, n_resamples=10000, alpha=0.05)
            report.bootstrap_point = point
            report.bootstrap_lower = lo
            report.bootstrap_upper = hi
            # Release gate per IMPLEMENTATION_PLAN.md v0.4 §19.2.
            report.release_gate_passed = bool(point >= 0.05 and lo > 0.0)

        return report
