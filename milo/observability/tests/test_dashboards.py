"""Tests asserting each shipped W&B dashboard JSON parses + has the right shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DASHBOARDS = ["health", "training", "rollout", "cost", "invariants"]
_ROOT = Path(__file__).parents[1] / "dashboards"


@pytest.mark.parametrize("name", _DASHBOARDS)
def test_dashboard_json_parses(name: str) -> None:
    obj = json.loads((_ROOT / f"{name}.json").read_text())
    assert obj["_dashboard_id"].startswith("milo_")
    assert obj["title"].startswith("Milo —")
    assert isinstance(obj["sections"], list)
    assert obj["sections"], f"{name} has no sections"
    for sec in obj["sections"]:
        assert "name" in sec
        assert "panels" in sec
        for p in sec["panels"]:
            assert "type" in p


def test_health_includes_4_headline_charts() -> None:
    """Plan §17.4 — on-call dashboard must surface 4 headline charts."""
    obj = json.loads((_ROOT / "health.json").read_text())
    headline = next(s for s in obj["sections"] if s["name"].startswith("Headline"))
    assert len(headline["panels"]) == 4


def test_invariants_dashboard_covers_all_8() -> None:
    """Per spec §7 / v0.7: I-1..I-8 each get a line panel."""
    obj = json.loads((_ROOT / "invariants.json").read_text())
    detail = next(s for s in obj["sections"] if s["name"] == "Per-invariant detail")
    metrics = {p["metric"] for p in detail["panels"]}
    for i in range(1, 9):
        assert f"invariant/i_{i}_rate" in metrics
