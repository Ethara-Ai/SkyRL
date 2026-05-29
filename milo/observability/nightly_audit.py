"""Phase 17.5 — nightly invariant audit.

Per spec §23.4 / plan §17.5: replay 20 random training rollouts via
`milo replay` (Phase 9). For each, assert the reward decomposition matches
the stored one. Catches gym/trainer schema drift early and is the canary
for the v0.7 §29 contract (the "explicitly NOT swappable" surface).

Designed to be invoked from `milo/slurm/nightly_audit.slurm` (Phase 18).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("milo.observability.nightly_audit")


@dataclass
class AuditFailure:
    rollout_path: Path
    reason: str
    stored: dict[str, Any] = field(default_factory=dict)
    replayed: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    started_iso: str
    finished_iso: str
    n_sampled: int
    n_ok: int
    failures: list[AuditFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_iso": self.started_iso,
            "finished_iso": self.finished_iso,
            "n_sampled": self.n_sampled,
            "n_ok": self.n_ok,
            "passed": self.passed,
            "failures": [
                {
                    "rollout_path": str(f.rollout_path),
                    "reason": f.reason,
                    "stored": f.stored,
                    "replayed": f.replayed,
                }
                for f in self.failures
            ],
        }


class NightlyAudit:
    """Picks `n_sample` random rollouts under `traces_root`, replays each
    deterministically, and asserts the replay's reward_decomposition matches
    the stored one (modulo floating-point tolerance).
    """

    def __init__(
        self,
        traces_root: Path,
        n_sample: int = 20,
        replay_fn: Callable[[Path], dict[str, Any]] | None = None,
        rng_seed: int = 0,
        tolerance: float = 1e-6,
    ) -> None:
        self.traces_root = Path(traces_root)
        self.n_sample = n_sample
        self._replay_fn = replay_fn
        self._rng_seed = rng_seed
        self._tolerance = tolerance

    def _all_traces(self) -> list[Path]:
        return list(self.traces_root.rglob("*.jsonl"))

    def _read_stored(self, path: Path) -> dict[str, Any]:
        text = path.read_text().strip().split("\n", 1)[0]
        obj = json.loads(text) if text else {}
        ext = obj.get("milo_extension") or {}
        return ext.get("terminal_summary", {}).get("reward_decomposition", {}) or {}

    def _decomps_match(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        keys = set(a) | set(b)
        # Skip volatile keys (run_id, timestamps).
        keys.discard("run_id")
        keys.discard("timestamp")
        for k in keys:
            va, vb = a.get(k), b.get(k)
            if isinstance(va, float) or isinstance(vb, float):
                try:
                    if abs(float(va or 0.0) - float(vb or 0.0)) > self._tolerance:
                        return False
                    continue
                except (TypeError, ValueError):
                    return False
            if va != vb:
                return False
        return True

    def run(self) -> AuditReport:
        traces = self._all_traces()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not traces:
            return AuditReport(
                started_iso=started,
                finished_iso=started,
                n_sampled=0,
                n_ok=0,
            )

        rng = random.Random(self._rng_seed)
        sample = rng.sample(traces, k=min(self.n_sample, len(traces)))
        failures: list[AuditFailure] = []
        n_ok = 0

        for path in sample:
            stored = self._read_stored(path)
            try:
                if self._replay_fn is None:
                    # No replay backend provided — degrade to "schema check":
                    # assert the stored decomposition has the required keys.
                    required = {"preset", "r_terminal", "r_total"}
                    if not required.issubset(stored):
                        failures.append(AuditFailure(
                            rollout_path=path,
                            reason=f"stored decomp missing keys: {required - set(stored)}",
                            stored=stored,
                        ))
                    else:
                        n_ok += 1
                    continue
                replayed = self._replay_fn(path)
                if not self._decomps_match(stored, replayed):
                    failures.append(AuditFailure(
                        rollout_path=path,
                        reason="reward decomposition mismatch on replay",
                        stored=stored,
                        replayed=replayed,
                    ))
                else:
                    n_ok += 1
            except Exception as exc:
                failures.append(AuditFailure(
                    rollout_path=path,
                    reason=f"replay raised {type(exc).__name__}: {exc}",
                    stored=stored,
                ))

        return AuditReport(
            started_iso=started,
            finished_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_sampled=len(sample),
            n_ok=n_ok,
            failures=failures,
        )
