"""Tests for the Milo eval harness."""

from __future__ import annotations

from pathlib import Path

from milo.adapters.base import PolicyResult, StubPolicyAdapter
from milo.eval.baseline_format import BaselineReport
from milo.eval.evaluator import MiloEvaluator


def _instances() -> list[dict[str, str]]:
    return [
        {"instance_id": "p1", "difficulty_tier": "trivial", "lang": "python"},
        {"instance_id": "p2", "difficulty_tier": "medium",  "lang": "python"},
        {"instance_id": "p3", "difficulty_tier": "medium",  "lang": "go"},
        {"instance_id": "p4", "difficulty_tier": "hard",    "lang": "rust"},
    ]


def test_evaluate_basic_shape() -> None:
    adapter = StubPolicyAdapter(policy_id="m", default_pass_rate=1.0)
    rep = MiloEvaluator(split="smoke", k=4).evaluate(adapter, _instances())
    assert rep.n_instances == 4
    assert rep.policy_id == "m"
    assert rep.split == "smoke"
    assert rep.k == 4
    assert rep.pass_at_k_overall == 1.0
    assert set(rep.pass_at_k_by_tier) == {"trivial", "medium", "hard"}
    assert set(rep.pass_at_k_by_lang) == {"python", "go", "rust"}


def test_evaluate_zero_pass_rate() -> None:
    adapter = StubPolicyAdapter(policy_id="m", default_pass_rate=0.0)
    rep = MiloEvaluator(split="smoke", k=4).evaluate(adapter, _instances())
    assert rep.pass_at_k_overall == 0.0
    assert rep.release_gate_passed is None    # no baseline → no gate


def test_evaluate_with_baseline_release_gate(tmp_path: Path) -> None:
    baseline = BaselineReport(
        model_name="qwen2.5-coder-32b-milo-sft",
        model_revision="sft-v1",
        k=4,
        pass_at_k_overall=0.30,
        pass_at_k_by_tier={"trivial": 0.5, "medium": 0.3, "hard": 0.1},
    )
    baseline_path = tmp_path / "sft.json"
    baseline.write_json(baseline_path)

    # Stub adapter that significantly beats the baseline on every tier
    overrides = {
        ("p1", s): PolicyResult("p1", s, True, {"r_total": 1.0}) for s in range(4)
    }
    overrides.update(
        {("p2", s): PolicyResult("p2", s, True, {"r_total": 1.0}) for s in range(4)}
    )
    overrides.update(
        {("p3", s): PolicyResult("p3", s, True, {"r_total": 1.0}) for s in range(4)}
    )
    overrides.update(
        {("p4", s): PolicyResult("p4", s, True, {"r_total": 1.0}) for s in range(4)}
    )
    adapter = StubPolicyAdapter("rl-run", outcomes=overrides)
    rep = MiloEvaluator(split="holdout", k=4).evaluate(
        adapter, _instances(), baselines={"sft": baseline_path}
    )
    assert rep.bootstrap_point is not None
    assert rep.bootstrap_point > 0.05
    # Release gate must be either True (likely) or False (CI too noisy) — never None.
    assert rep.release_gate_passed in (True, False)


def test_evaluate_writes_json(tmp_path: Path) -> None:
    adapter = StubPolicyAdapter("m", default_pass_rate=0.5)
    rep = MiloEvaluator(split="smoke", k=2).evaluate(adapter, _instances())
    rep.write_json(tmp_path / "report.json")
    assert (tmp_path / "report.json").is_file()
