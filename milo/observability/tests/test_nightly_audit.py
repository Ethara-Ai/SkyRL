"""Tests for the nightly invariant audit."""

from __future__ import annotations

import json
from pathlib import Path

from milo.observability.nightly_audit import NightlyAudit


def _write_trace(p: Path, decomp: dict[str, object]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    rollout = {
        "instance_id": p.stem,
        "milo_extension": {
            "terminal_summary": {
                "reward_decomposition": decomp,
                "r_total": decomp.get("r_total", 0.0),
            }
        },
    }
    p.write_text(json.dumps(rollout) + "\n")


def test_audit_no_replay_fn_passes_when_schema_complete(tmp_path: Path) -> None:
    _write_trace(tmp_path / "r1.jsonl", {"preset": "composite", "r_terminal": 1, "r_total": 1.0})
    _write_trace(tmp_path / "r2.jsonl", {"preset": "composite", "r_terminal": 0, "r_total": 0.0})
    rep = NightlyAudit(tmp_path, n_sample=2).run()
    assert rep.passed
    assert rep.n_sampled == 2
    assert rep.n_ok == 2


def test_audit_no_replay_fn_fails_on_schema_drift(tmp_path: Path) -> None:
    _write_trace(tmp_path / "r1.jsonl", {"r_terminal": 1})   # missing preset, r_total
    rep = NightlyAudit(tmp_path, n_sample=1).run()
    assert not rep.passed
    assert "missing keys" in rep.failures[0].reason


def test_audit_with_replay_fn_detects_mismatch(tmp_path: Path) -> None:
    _write_trace(tmp_path / "r1.jsonl", {"preset": "composite", "r_terminal": 1, "r_total": 1.0})
    def fake_replay(_: Path) -> dict[str, object]:
        return {"preset": "composite", "r_terminal": 0, "r_total": 0.0}
    rep = NightlyAudit(tmp_path, n_sample=1, replay_fn=fake_replay).run()
    assert not rep.passed
    assert "mismatch" in rep.failures[0].reason


def test_audit_empty_dir_returns_zero_sampled(tmp_path: Path) -> None:
    rep = NightlyAudit(tmp_path, n_sample=20).run()
    assert rep.n_sampled == 0
    assert rep.passed
