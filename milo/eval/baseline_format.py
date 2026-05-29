"""Baseline-report dataclass + JSON I/O — Phase 21.0.

Per `IMPLEMENTATION_PLAN.md` v0.4 §21.0: every eval during training and
every ablation compares against frozen baseline JSONs:

    milo/data/baselines/{vanilla,sft,claude_opus,gemini}.json

The schema is fixed once and never changed; integrators read these to
reproduce numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BaselineReport:
    model_name: str
    model_revision: str
    k: int
    pass_at_k_overall: float
    pass_at_k_by_tier: dict[str, float] = field(default_factory=dict)
    pass_at_k_by_lang: dict[str, float] = field(default_factory=dict)
    evaluation_date: str = ""        # ISO 8601
    evaluation_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read_json(cls, path: Path) -> "BaselineReport":
        obj = json.loads(Path(path).read_text())
        return cls(**obj)
