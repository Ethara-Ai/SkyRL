"""Tests for milo.logging.converter — round-trips a synthetic on-disk rollout."""

from __future__ import annotations

import json
from pathlib import Path

from milo.logging.converter import convert_legacy_trajectory


def _write_synthetic_rollout(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rollout = {
        "instance_id": "fake__fake-1",
        "attempt": 1,
        "test_result": {"resolved": True, "git_patch": "diff..."},
        "instruction": "<task/>",
        "metadata": {"agent_class": "ReactAgent"},
        "history": [
            {"id": 1, "kind": "SystemPromptEvent", "source": "agent"},
            {
                "id": 2,
                "kind": "ActionEvent",
                "source": "agent",
                "action": {"command": "ls"},
                "tool_call": {"name": "bash", "arguments": {"cmd": "ls"}},
            },
            {
                "id": 3,
                "kind": "ObservationEvent",
                "source": "environment",
                "observation": {"content": "file.py"},
            },
            {
                "id": 4,
                "kind": "ConversationStateUpdateEvent",
                "source": "agent",
                "state": "finished",
            },
        ],
        "metrics": {
            "accumulated_cost": 0.42,
            "accumulated_token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 0,
            },
        },
        "error": None,
    }
    (d / "output.jsonl").write_text(json.dumps(rollout) + "\n")
    (d / "output.report.json").write_text(json.dumps({"resolved": True}))
    (d / "cost_report.jsonl").write_text(
        json.dumps({"summary": {"total_cost_usd": 0.42}}) + "\n"
    )


def test_round_trip_includes_overlay(tmp_path: Path) -> None:
    _write_synthetic_rollout(tmp_path)
    out = tmp_path / "out.jsonl"
    merged = convert_legacy_trajectory(tmp_path / "output.jsonl", out)
    assert "milo_extension" in merged
    ext = merged["milo_extension"]
    assert ext["schema_version"].startswith("milo-trace/")
    assert ext["terminal_summary"]["r_total"] == 1.0
    assert ext["terminal_summary"]["termination_reason"] == "finished"
    # Cost / tokens picked up from the cost + metrics files.
    assert ext["terminal_summary"]["cost_usd"] == 0.42
    assert ext["terminal_summary"]["tokens"]["prompt"] == 1000


def test_unresolved_rollout_gets_zero_reward(tmp_path: Path) -> None:
    _write_synthetic_rollout(tmp_path)
    (tmp_path / "output.report.json").write_text(json.dumps({"resolved": False}))
    out = tmp_path / "out.jsonl"
    merged = convert_legacy_trajectory(tmp_path / "output.jsonl", out)
    assert merged["milo_extension"]["terminal_summary"]["r_total"] == 0.0
