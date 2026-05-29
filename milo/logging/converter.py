"""Legacy-trajectory → OpenHands+overlay converter — Phase 5.4.

Implements `IMPLEMENTATION_PLAN.md` v0.4 §5.4. The 50 existing milo-bench
trajectories on disk under `<freya>/milo-bench/trajectories/<task>/<model>/
run_*/` are *already* in OpenHands SDK rollout format
(`{instance_id, attempt, test_result, instruction, metadata, history,
metrics, error, instance, runtime_runs}`) but lack the v0.7 `milo_extension`
overlay block (per spec §6.5). This module reads the on-disk artefacts
(`output.jsonl`, `output.report.json`, `cost_report.jsonl`) and emits a
single overlay-enriched jsonl line that downstream Milo tooling
(replay, calibration, SME-golden recording) reads uniformly.

One-shot script; runs once during Phase 11 enrichment.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from milo.logging.trace_format import (
    MILO_TRACE_SCHEMA_VERSION,
    MiloExtension,
    StepShaping,
    TerminalSummary,
    attach_overlay,
)


def _read_jsonl_first_line(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return json.loads(text.split("\n", 1)[0])


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _read_cost_report(path: Path) -> dict[str, Any] | None:
    """cost_report.jsonl is technically jsonl but only ever has one record."""
    return _read_jsonl_first_line(path)


def _derive_termination_reason(rollout: dict[str, Any]) -> str:
    """Best-effort: read the OpenHands history for a final agent state."""
    history = rollout.get("history") or []
    # Look for the last ConversationStateUpdateEvent with finished state.
    for ev in reversed(history):
        if not isinstance(ev, dict):
            continue
        if ev.get("kind") == "ConversationStateUpdateEvent":
            state = ev.get("state") or ev.get("agent_state")
            if state:
                return str(state)
    if rollout.get("error"):
        return "error"
    return "submit"


def _derive_terminal_summary(
    rollout: dict[str, Any],
    report: dict[str, Any] | None,
    cost: dict[str, Any] | None,
) -> TerminalSummary:
    """Pull together the spec §6.8-shaped terminal summary.

    Source-of-truth precedence for `resolved`:
        1. report.resolved (post-eval-harness verdict) if present
        2. rollout.test_result.resolved (agent's self-report) otherwise
    """
    if report is not None and "resolved" in report:
        resolved = bool(report["resolved"])
    else:
        resolved = bool(rollout.get("test_result", {}).get("resolved"))
    # Legacy traces had no rubric, no shaping reward — only the verifier verdict.
    verifier_report = {
        "resolved": resolved,
        "raw_report": report or {},
    }
    rubric_report = {"items": [], "mean_score": 0.0, "tampering_detected": False}
    reward_decomposition = {
        "preset": "legacy",
        "r_terminal": 1.0 if resolved else 0.0,
        "r_delta_sum": 0.0,
        "r_rubric_mean": 0.0,
        "r_tir_sum": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "lambda_": 0.0,
        "gamma": 0.0,
        "r_total": 1.0 if resolved else 0.0,
    }
    cost_summary = (cost or {}).get("summary") or (cost or {}).get("main_output") or {}
    cost_usd = float(
        cost_summary.get("total_cost_usd")
        or cost_summary.get("total_cost")
        or (rollout.get("metrics") or {}).get("accumulated_cost")
        or 0.0
    )
    metrics = rollout.get("metrics") or {}
    token_usage = metrics.get("accumulated_token_usage") or metrics.get("usage") or {}
    tokens = {
        "prompt": int(token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0),
        "completion": int(token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0),
        "cache_read": int(token_usage.get("cache_read_input_tokens", 0) or 0),
        "cache_write": int(token_usage.get("cache_creation_input_tokens", 0) or 0),
    }
    return TerminalSummary(
        termination_reason=_derive_termination_reason(rollout),
        verifier_report=verifier_report,
        rubric_report=rubric_report,
        reward_decomposition=reward_decomposition,
        r_total=reward_decomposition["r_total"],
        cost_usd=cost_usd,
        tokens=tokens,
    )


def convert_legacy_trajectory(
    input_path: Path,
    output_path: Path,
    derive_overlay_from: dict[str, Any] | None = None,
    report_path: Path | None = None,
    cost_path: Path | None = None,
    gym_version: str = "milo-gym/0.1.0",
) -> dict[str, Any]:
    """Read an on-disk milo-bench OpenHands rollout and write the v0.7 overlay form.

    Returns the merged rollout dict (also written to `output_path` as one
    jsonl line).
    """
    rollout = _read_jsonl_first_line(input_path)
    if rollout is None:
        raise FileNotFoundError(f"empty or missing rollout at {input_path}")

    if report_path is None:
        report_path = input_path.with_name("output.report.json")
    if cost_path is None:
        cost_path = input_path.with_name("cost_report.jsonl")

    report = _read_json(report_path)
    cost = _read_cost_report(cost_path)

    terminal = _derive_terminal_summary(rollout, report, cost)
    overlay = MiloExtension(
        schema_version=MILO_TRACE_SCHEMA_VERSION,
        gym_version=gym_version,
        reward_config={
            "preset": "legacy",
            "alpha": 0.0,
            "beta": 0.0,
            "lambda_": 0.0,
            "gamma": 0.0,
        },
        per_step_shaping=[],   # legacy traces have no per-step shaping signal
        terminal_summary=terminal,
    )

    merged = attach_overlay(rollout, overlay)
    if derive_overlay_from:
        merged.setdefault("milo_extension", {}).update(derive_overlay_from)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged) + "\n")
    return merged


def convert_directory(
    trajectories_root: Path,
    output_root: Path,
    gym_version: str = "milo-gym/0.1.0",
) -> list[Path]:
    """Walk `<root>/<task>/<model>/run_*/output.jsonl` and convert each."""
    written: list[Path] = []
    for output_jsonl in trajectories_root.glob("*/*/run_*/output.jsonl"):
        rel = output_jsonl.relative_to(trajectories_root)
        out = output_root / rel
        try:
            convert_legacy_trajectory(output_jsonl, out, gym_version=gym_version)
            written.append(out)
        except Exception as exc:  # pragma: no cover - exercised in CI on real fixtures
            print(f"[converter] failed on {output_jsonl}: {exc}")
    return written
