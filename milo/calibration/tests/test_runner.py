"""Tests for ``milo.calibration.runner`` — plan §8.1 acceptance criteria.

Plan §8.1 specifies three test cases verbatim:

    (a) easy task agreed-upon → tier "trivial"
    (b) disagreement >0.30 → rejected_for_disagreement=True
    (c) split-tier task at boundary → assert deterministic boundary behaviour

Plus a few extras covering the spec §8 disagreement-equal-to-threshold edge
case and the JSON serialization shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from milo.adapters import PolicyResult, StubPolicyAdapter, make_stub_adapter
from milo.calibration.runner import (
    DISAGREEMENT_THRESHOLD,
    CalibrationRunner,
    TaskCalibration,
    assign_tier,
    default_model_handles,
)


# ---------------------------------------------------------------------------
# assign_tier — pure binning
# ---------------------------------------------------------------------------

def test_assign_tier_trivial() -> None:
    """Plan §8.1 (a): agreed-upon easy task → 'trivial'."""
    tier, disagreement, reason = assign_tier({"opus": 0.875, "gemini": 0.875})
    assert tier == "trivial"
    assert disagreement == 0.0
    assert reason == ""


def test_assign_tier_disagreement_rejected() -> None:
    """Plan §8.1 (b): disagreement > 0.30 → REJECT."""
    tier, disagreement, reason = assign_tier({"opus": 0.875, "gemini": 0.500})
    # 0.875 - 0.500 = 0.375 > 0.30
    assert tier == "REJECT_DISAGREEMENT"
    assert disagreement == pytest.approx(0.375)
    assert reason == "disagreement"


def test_assign_tier_boundary_at_threshold_accepted() -> None:
    """Spec §8 step 4 says reject if disagreement > 0.30 — exactly 0.30 is accepted."""
    tier, disagreement, reason = assign_tier({"opus": 0.700, "gemini": 0.400})
    assert tier != "REJECT_DISAGREEMENT"
    assert disagreement == pytest.approx(0.300)
    assert reason == ""
    # mean = 0.55 → medium
    assert tier == "medium"


def test_assign_tier_boundary_trivial_inclusive_at_0_60() -> None:
    """Plan §8.1 (c): boundary case — spec §8 says '≥ 0.60' is trivial."""
    tier, _, _ = assign_tier({"opus": 0.60, "gemini": 0.60})
    assert tier == "trivial"


def test_assign_tier_boundary_medium_inclusive_at_0_20() -> None:
    """Plan §8.1 (c): boundary — '0.20-0.60' includes 0.20."""
    tier, _, _ = assign_tier({"opus": 0.20, "gemini": 0.20})
    assert tier == "medium"


def test_assign_tier_boundary_hard_inclusive_at_0_05() -> None:
    """Plan §8.1 (c): boundary — '0.05-0.20' includes 0.05."""
    tier, _, _ = assign_tier({"opus": 0.05, "gemini": 0.05})
    assert tier == "hard"


def test_assign_tier_unsolvable_below_0_05() -> None:
    """Spec §8: '< 0.05 → unsolvable'."""
    tier, _, _ = assign_tier({"opus": 0.0, "gemini": 0.0})
    assert tier == "unsolvable"
    tier, _, _ = assign_tier({"opus": 0.04, "gemini": 0.04})
    assert tier == "unsolvable"


def test_assign_tier_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        assign_tier({})
    with pytest.raises(ValueError):
        assign_tier({"opus": 1.5, "gemini": 0.5})
    with pytest.raises(ValueError):
        assign_tier({"opus": -0.1, "gemini": 0.5})


# ---------------------------------------------------------------------------
# CalibrationRunner — end-to-end with stub adapters
# ---------------------------------------------------------------------------

def _all_pass_overrides(instance_id: str, k: int = 8, passed: bool = True) -> dict:
    """Build a pinned outcome map for one task across all k seeds."""
    return {
        (instance_id, seed): PolicyResult(
            instance_id=instance_id,
            seed=seed,
            passed=passed,
            reward_decomposition={},
        )
        for seed in range(k)
    }


def test_runner_pins_trivial_task() -> None:
    """All-pass on both models → trivial tier, 0 disagreement, not rejected."""
    iid = "fake/repo-trivial"
    a = StubPolicyAdapter("opus", outcomes=_all_pass_overrides(iid))
    b = StubPolicyAdapter("gemini", outcomes=_all_pass_overrides(iid))
    runner = CalibrationRunner([a, b], k=8)
    cal = runner.calibrate_task(iid)
    assert cal.tier == "trivial"
    assert cal.disagreement == 0.0
    assert cal.mean_pass_at_k == 1.0
    assert cal.rejected_for_disagreement is False
    assert cal.model_handles == ("opus", "gemini")


def test_runner_pins_disagreement_rejected() -> None:
    """Model 1 all-pass, model 2 all-fail → 1.0 disagreement → reject."""
    iid = "fake/repo-disagree"
    a = StubPolicyAdapter("opus", outcomes=_all_pass_overrides(iid, passed=True))
    b = StubPolicyAdapter("gemini", outcomes=_all_pass_overrides(iid, passed=False))
    runner = CalibrationRunner([a, b], k=8)
    cal = runner.calibrate_task(iid)
    assert cal.rejected_for_disagreement is True
    assert cal.tier == "REJECT_DISAGREEMENT"
    assert cal.rejected_reason == "disagreement"
    assert cal.disagreement == pytest.approx(1.0)


def test_runner_pins_hard_tier() -> None:
    """1/8 pass on both models = mean 0.125 → hard."""
    iid = "fake/repo-hard"
    overrides_one_pass = {
        (iid, 0): PolicyResult(iid, 0, True, {}),
        **{(iid, s): PolicyResult(iid, s, False, {}) for s in range(1, 8)},
    }
    a = StubPolicyAdapter("opus", outcomes=overrides_one_pass)
    b = StubPolicyAdapter("gemini", outcomes=overrides_one_pass)
    runner = CalibrationRunner([a, b], k=8)
    cal = runner.calibrate_task(iid)
    assert cal.tier == "hard"
    assert cal.disagreement == 0.0


def test_runner_requires_two_adapters() -> None:
    a = StubPolicyAdapter("only")
    with pytest.raises(ValueError, match="two reference models"):
        CalibrationRunner([a], k=8)


def test_runner_invalid_k() -> None:
    a = StubPolicyAdapter("a")
    b = StubPolicyAdapter("b")
    with pytest.raises(ValueError):
        CalibrationRunner([a, b], k=0)


def test_runner_calibrate_all_writes_json(tmp_path: Path) -> None:
    iids = ["fake/repo-1", "fake/repo-2"]
    a = make_stub_adapter("opus", pass_rate=0.0)  # all fail
    b = make_stub_adapter("gemini", pass_rate=0.0)
    runner = CalibrationRunner([a, b], k=4)
    results = runner.calibrate_all(iids)
    out = tmp_path / "calibration_results.json"
    runner.write_results(results, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "milo-calibration/1.0"
    assert payload["k"] == 4
    assert payload["model_handles"] == ["opus", "gemini"]
    assert set(payload["results"].keys()) == set(iids)
    # all-fail → unsolvable on every task
    for entry in payload["results"].values():
        assert entry["tier"] == "unsolvable"
        assert entry["model_1_pass_rate"] == 0.0
        assert entry["model_2_pass_rate"] == 0.0
        assert entry["disagreement"] == 0.0
        assert entry["rejected_reason"] == ""


def test_runner_results_entry_schema() -> None:
    """The per-task dict must contain the exact keys required by the prompt."""
    iid = "fake/repo-schema"
    a = make_stub_adapter("opus", pass_rate=0.5)
    b = make_stub_adapter("gemini", pass_rate=0.5)
    runner = CalibrationRunner([a, b], k=2)
    cal = runner.calibrate_task(iid)
    entry = cal.to_results_entry()
    required = {"tier", "model_1_pass_rate", "model_2_pass_rate",
                "disagreement", "rejected_reason"}
    assert required.issubset(entry.keys())


def test_default_model_handles_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILO_JUDGE_MODEL", "claude-opus-future")
    monkeypatch.setenv("MILO_CALIBRATION_MODEL_2", "gemini-future")
    assert default_model_handles() == ("claude-opus-future", "gemini-future")


def test_default_model_handles_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MILO_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("MILO_CALIBRATION_MODEL_2", raising=False)
    handles = default_model_handles()
    assert handles == ("claude-opus-4-6", "gemini-2.5-pro")


def test_disagreement_threshold_constant_matches_spec() -> None:
    # Spec §8 step 4 — pinned at 0.30. Guard against accidental drift.
    assert DISAGREEMENT_THRESHOLD == 0.30


# ---------------------------------------------------------------------------
# Recalibrate CLI — argparse + JSONL parsing only (no API calls)
# ---------------------------------------------------------------------------

def test_recalibrate_cli_runs_with_stub(tmp_path: Path) -> None:
    from milo.calibration.recalibrate import run as recalibrate_run

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"instance_id": "a/repo-1"}) + "\n"
        + json.dumps({"instance_id": "a/repo-2"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "results.json"
    code = recalibrate_run([
        "--tasks", str(tasks),
        "--output", str(out),
        "--use-stub",
        "--k", "2",
        "--stub-pass-rates", "0.0", "0.0",
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    assert len(payload["results"]) == 2


def test_recalibrate_cli_plain_text_tasks(tmp_path: Path) -> None:
    from milo.calibration.recalibrate import run as recalibrate_run

    tasks = tmp_path / "tasks.txt"
    tasks.write_text("a/repo-1\na/repo-2\n", encoding="utf-8")
    out = tmp_path / "results.json"
    code = recalibrate_run([
        "--tasks", str(tasks),
        "--output", str(out),
        "--use-stub", "--k", "2",
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    assert set(payload["results"].keys()) == {"a/repo-1", "a/repo-2"}


def test_recalibrate_cli_empty_returns_exit_2(tmp_path: Path) -> None:
    from milo.calibration.recalibrate import run as recalibrate_run

    tasks = tmp_path / "empty.jsonl"
    tasks.write_text("", encoding="utf-8")
    out = tmp_path / "results.json"
    code = recalibrate_run([
        "--tasks", str(tasks),
        "--output", str(out),
        "--use-stub",
    ])
    assert code == 2
