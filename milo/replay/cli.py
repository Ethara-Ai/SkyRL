"""milo replay CLI — spec §10.3 deterministic replay tool.

Usage::

    # 1. Default — replay a stored rollout against the gym; assert
    #    reward_decomposition matches stored (within numerical tolerance).
    python -m milo.replay.cli --instance locustio__locust-1541

    # 2. Record-golden — human-in-the-loop SME drives the rollout, which is
    #    persisted to <golden_root>/<instance_id>/trace.jsonl per spec §10.1.3.
    python -m milo.replay.cli --instance locustio__locust-1541 --record-golden

    # 3. Golden — replay a stored golden trace, asserts R_terminal == 1 and
    #    R_rubric == 1.0 deterministically per spec §10.1.3.
    python -m milo.replay.cli --instance locustio__locust-1541 --golden

The CLI is intentionally a thin orchestration shell over :func:`replay_trace`;
the real work (container restart, tool-call replay, verifier re-run) lives
in Phase 1 (``milo.lht_adapter.generator``) and Phase 2 (``milo.verifier``).
This file knows how to walk the on-disk trace JSONL, surface mismatches,
and exit non-zero on failure — which is what CI and the nightly audit
consume.

On-disk trace format follows spec §6.5: one JSONL file per rollout, with
a ``trace_header`` first record and a ``terminal_summary`` last record;
all middle records are ``step_record``s. The replay tool only needs the
header (for ``seed`` + ``policy_id``) and the terminal summary (for the
recorded ``reward_decomposition``) — the per-step records are consumed by
the rollout driver when re-executing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from milo.adapters import PolicyAdapter, make_stub_adapter


# Spec §10.4: numerical tolerance for the property test
# ``R_total = R_terminal + α·sum(R_delta) + β·R_rubric`` within 1e-9.
# The replay tool uses the same tolerance for reward_decomposition matching.
REWARD_TOLERANCE = 1e-9

# Golden-trace certification thresholds per spec §10.1.3:
# "the SME-authored golden trace, when replayed deterministically, must
#  achieve R_terminal = 1 and all rubric items at 1."
GOLDEN_R_TERMINAL = 1
GOLDEN_R_RUBRIC_MIN = 1.0

# Default on-disk roots — match the production layout from spec §5.5
# (per-rollout directory layout) and plan §9 ("golden trace path" lives
# in the LHT instance dict at ``instance['golden_trace_path']``).
DEFAULT_TRACE_ROOT = Path(os.environ.get("MILO_TRACE_ROOT", "milo/data/traces"))
DEFAULT_GOLDEN_ROOT = Path(os.environ.get("MILO_GOLDEN_ROOT", "milo/data/golden_traces"))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReplayMismatch:
    """Single key-level mismatch between stored and replayed decomposition.

    ``key`` is a dotted path into the spec §6.8 reward_decomposition dict
    (e.g. ``"components.terminal"``); ``stored`` and ``replayed`` are the
    raw values from each side; ``delta`` is the absolute numeric
    difference when both sides are floats (otherwise None).
    """

    key: str
    stored: Any
    replayed: Any
    delta: float | None


@dataclass(slots=True)
class ReplayResult:
    """Outcome of a ``replay_trace`` call.

    Attributes
    ----------
    instance_id, mode:
        Echo of the inputs for downstream reporting.
    matched:
        ``True`` iff replay reproduced the stored reward_decomposition
        within :data:`REWARD_TOLERANCE` (and, when ``mode == "golden"``,
        the §10.1.3 golden-trace assertions all held).
    mismatches:
        List of key-level mismatches; empty when ``matched``.
    stored_decomposition, replayed_decomposition:
        Full §6.8 dicts on each side. Useful for diff output in CI logs.
    golden_assertions_passed:
        Set on ``mode == "golden"``: maps assertion name to pass/fail.
        Empty on default replay.
    seed, policy_id:
        Echoed from the trace header. Pinned to detect accidental
        cross-rollout replay misuse.
    replay_iso:
        Timestamp this replay started — used by the nightly audit report.
    """

    instance_id: str
    mode: str
    matched: bool
    mismatches: list[ReplayMismatch] = field(default_factory=list)
    stored_decomposition: dict[str, Any] = field(default_factory=dict)
    replayed_decomposition: dict[str, Any] = field(default_factory=dict)
    golden_assertions_passed: dict[str, bool] = field(default_factory=dict)
    seed: int | None = None
    policy_id: str | None = None
    replay_iso: str = ""


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def _load_trace_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Parse a §6.5 trace.jsonl into (header, steps, terminal_summary).

    Raises ValueError on a malformed trace (missing header, missing
    terminal summary, etc.) — these are integrator-facing errors and the
    CLI surfaces them with the offending path in the message.
    """
    if not path.exists():
        raise FileNotFoundError(f"trace file not found: {path}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(
            f"trace at {path} has < 2 records — needs at least a header and a "
            "terminal_summary per spec §6.5"
        )
    parsed = [json.loads(ln) for ln in lines]
    header = parsed[0]
    if "schema_version" not in header or not header["schema_version"].startswith("milo-trace/"):
        raise ValueError(
            f"trace at {path} missing milo-trace schema_version header "
            "(spec §6.5)"
        )
    terminal = parsed[-1]
    if "reward_decomposition" not in terminal:
        raise ValueError(
            f"trace at {path} terminal record missing 'reward_decomposition' — "
            "not a valid §6.5 terminal_summary"
        )
    steps = parsed[1:-1]
    return header, steps, terminal


def _resolve_trace_path(
    instance_id: str,
    *,
    trace_root: Path,
    golden: bool,
    golden_root: Path,
    explicit_path: Path | None,
) -> Path:
    """Locate the trace JSONL for an instance + mode.

    Resolution order:
      1. ``--trace-path`` explicit override
      2. ``<golden_root>/<instance_id>/trace.jsonl``  (when ``--golden``)
      3. ``<trace_root>/<instance_id>/run_1/output.jsonl`` (OpenHands layout,
         matches the on-disk milo-bench trajectory format)
      4. ``<trace_root>/<instance_id>/trace.jsonl`` (spec §6.5 native layout)
    """
    if explicit_path is not None:
        return explicit_path
    if golden:
        return golden_root / instance_id / "trace.jsonl"
    # OpenHands-layout fallback first because that's what the on-disk
    # ``milo-bench/trajectories/<iid>/<model>/run_<n>/output.jsonl`` files
    # use (see ``milo-bench/trajectories/locustio__locust-1541/...``).
    openhands_layout = trace_root / instance_id / "run_1" / "output.jsonl"
    if openhands_layout.exists():
        return openhands_layout
    return trace_root / instance_id / "trace.jsonl"


# ---------------------------------------------------------------------------
# Decomposition comparison
# ---------------------------------------------------------------------------

def _iter_flat(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    """Yield (dotted_key, leaf_value) pairs, recursing into dicts only.

    Lists are leaves — we compare them element-wise by value, not by
    further recursion, because spec §6.8 lists (``r_delta_steps``,
    ``r_rubric_per_item``, ``r_tir_steps``) are positionally meaningful.
    """
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _iter_flat(f"{prefix}.{k}" if prefix else k, v)
    else:
        yield prefix, value


def _compare_values(stored: Any, replayed: Any, tol: float) -> tuple[bool, float | None]:
    """Return (match, abs_delta) for one leaf-vs-leaf comparison.

    Numeric leaves use absolute tolerance ``tol`` (spec §10.4: 1e-9).
    Lists of numerics get element-wise comparison; lists of non-numerics
    use exact equality. Strings / bools / ints use exact equality.
    """
    if isinstance(stored, list) and isinstance(replayed, list):
        if len(stored) != len(replayed):
            return False, None
        worst = 0.0
        for s, r in zip(stored, replayed):
            ok, d = _compare_values(s, r, tol)
            if not ok:
                return False, d
            if d is not None and d > worst:
                worst = d
        return True, worst
    if isinstance(stored, (int, float)) and isinstance(replayed, (int, float)):
        # Exclude bools (which subclass int) from float-tolerance: bools
        # are matched exactly so ``r_terminal == 1`` never silently
        # accepts r_terminal == 0.999999999.
        if isinstance(stored, bool) or isinstance(replayed, bool):
            return stored == replayed, None
        if math.isnan(stored) and math.isnan(replayed):
            return True, 0.0
        if math.isnan(stored) or math.isnan(replayed):
            return False, None
        delta = abs(float(stored) - float(replayed))
        return delta <= tol, delta
    return stored == replayed, None


def _diff_decompositions(
    stored: dict[str, Any],
    replayed: dict[str, Any],
    tol: float = REWARD_TOLERANCE,
) -> list[ReplayMismatch]:
    """Yield :class:`ReplayMismatch` for every key that fails to match.

    Performs a deep comparison over both decompositions: keys present in
    only one side count as mismatches. We compare a *union* of keys so
    schema drift in either direction (gym added a new component vs.
    trainer dropped one) surfaces.
    """
    stored_flat = dict(_iter_flat("", stored))
    replayed_flat = dict(_iter_flat("", replayed))
    all_keys = sorted(set(stored_flat) | set(replayed_flat))
    out: list[ReplayMismatch] = []
    for k in all_keys:
        if k not in stored_flat:
            out.append(ReplayMismatch(k, None, replayed_flat[k], None))
            continue
        if k not in replayed_flat:
            out.append(ReplayMismatch(k, stored_flat[k], None, None))
            continue
        ok, delta = _compare_values(stored_flat[k], replayed_flat[k], tol)
        if not ok:
            out.append(ReplayMismatch(k, stored_flat[k], replayed_flat[k], delta))
    return out


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------

def _execute_replay(
    instance_id: str,
    seed: int,
    policy_id: str,
    *,
    adapter: PolicyAdapter | None,
) -> dict[str, Any]:
    """Re-run a rollout deterministically and return the §6.8 dict.

    The production path will invoke ``MiloLHTGenerator`` (Phase 1) which
    spins up a fresh container, replays the recorded tool calls, and
    re-runs the verifier. Until that lands, this function delegates to
    the provided ``adapter`` (defaulting to a deterministic stub) — which
    is sufficient for the Phase 9 unit tests + the nightly audit's
    "did the recorded decomposition round-trip" check.
    """
    if adapter is None:
        # Deterministic stub keyed off policy_id so replay of the same
        # (iid, seed, policy_id) triple always returns the same answer.
        adapter = make_stub_adapter(policy_id, pass_rate=0.0)
    result = adapter.rollout(instance_id, seed=seed)
    return dict(result.reward_decomposition)


def replay_trace(
    instance_id: str,
    *,
    trace_path: Path | None = None,
    mode: str = "default",
    adapter: PolicyAdapter | None = None,
    trace_root: Path | None = None,
    golden_root: Path | None = None,
    tolerance: float = REWARD_TOLERANCE,
) -> ReplayResult:
    """Replay a stored trace and return a :class:`ReplayResult`.

    ``mode`` ∈ ``{"default", "golden", "record-golden"}``:

    * ``"default"`` — load the trace from ``trace_root``, replay, assert
      decomposition matches.
    * ``"golden"`` — load from ``golden_root``, replay, assert spec §10.1.3
      thresholds (``R_terminal == 1`` and ``R_rubric == 1.0``).
    * ``"record-golden"`` — caller is responsible for driving the rollout;
      this function only writes the resulting trace to the golden root.
      For programmatic API use, prefer :func:`record_golden_trace`.

    The function never raises on mismatch — callers (the CLI, the nightly
    audit) inspect ``ReplayResult.matched`` and decide the exit code.
    """
    if mode not in ("default", "golden", "record-golden"):
        raise ValueError(
            f"unsupported replay mode {mode!r}; "
            "expected 'default', 'golden', or 'record-golden'"
        )
    if mode == "record-golden":
        raise ValueError(
            "record-golden mode is interactive; use record_golden_trace() "
            "or the CLI's --record-golden flag instead of replay_trace()"
        )
    troot = trace_root if trace_root is not None else DEFAULT_TRACE_ROOT
    groot = golden_root if golden_root is not None else DEFAULT_GOLDEN_ROOT
    path = _resolve_trace_path(
        instance_id,
        trace_root=troot,
        golden=(mode == "golden"),
        golden_root=groot,
        explicit_path=trace_path,
    )
    header, _steps, terminal = _load_trace_records(path)
    seed = int(header.get("seed", 0))
    policy_id = str(header.get("policy_id", "unknown"))
    stored = terminal["reward_decomposition"]

    replayed = _execute_replay(
        instance_id,
        seed=seed,
        policy_id=policy_id,
        adapter=adapter,
    )

    mismatches = _diff_decompositions(stored, replayed, tol=tolerance)
    matched = len(mismatches) == 0

    golden_asserts: dict[str, bool] = {}
    if mode == "golden":
        golden_asserts = _check_golden_assertions(replayed)
        if not all(golden_asserts.values()):
            matched = False

    return ReplayResult(
        instance_id=instance_id,
        mode=mode,
        matched=matched,
        mismatches=mismatches,
        stored_decomposition=dict(stored),
        replayed_decomposition=replayed,
        golden_assertions_passed=golden_asserts,
        seed=seed,
        policy_id=policy_id,
        replay_iso=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _check_golden_assertions(decomposition: Mapping[str, Any]) -> dict[str, bool]:
    """Apply spec §10.1.3 golden-trace certification thresholds."""
    return {
        "r_terminal_eq_1": int(decomposition.get("r_terminal", 0)) == GOLDEN_R_TERMINAL,
        "r_rubric_eq_1.0": float(decomposition.get("r_rubric_mean", 0.0))
        >= GOLDEN_R_RUBRIC_MIN - 1e-9,
    }


def record_golden_trace(
    instance_id: str,
    *,
    sme_id: str,
    rollout_decomposition: dict[str, Any],
    rollout_steps: Iterable[dict[str, Any]] | None = None,
    golden_root: Path | None = None,
    seed: int = 0,
    gym_version: str = "0.7.0",
) -> Path:
    """Persist a SME-driven rollout as a golden trace per spec §10.1.3.

    Writes a §6.5-compliant ``trace.jsonl`` under
    ``<golden_root>/<instance_id>/`` and returns the path. The ``policy_id``
    is forced to ``"sme:<sme_id>"`` per spec §6.5 ("Golden trace is the
    same schema with `policy_id = "sme:<reviewer_id>"`").

    This function does *not* re-run the verifier or assert the §10.1.3
    thresholds — the canonical workflow is:

        1. SME drives the rollout via the gym (this function persists it).
        2. Operator runs ``milo replay --instance <id> --golden`` (this
           module's CLI), which loads the persisted trace and asserts
           R_terminal == 1 and R_rubric == 1.0.

    Separating step 1 from step 2 mirrors the spec's QC pipeline (§10.1
    layers 3 and 4: golden-trace replay then SME human review).
    """
    groot = golden_root if golden_root is not None else DEFAULT_GOLDEN_ROOT
    out_dir = groot / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trace.jsonl"

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = {
        "schema_version": "milo-trace/1.0",
        "instance_id": instance_id,
        "seed": int(seed),
        "policy_id": f"sme:{sme_id}",
        "started_iso": started,
        "gym_version": gym_version,
        "config": {
            "alpha": 0.05, "beta": 0.20, "lambda": 2.0, "gamma": 0.0,
            "max_episode_seconds": 1800, "max_tool_calls": 1000,
            "preset": "composite",
            "judge_model": os.environ.get("MILO_JUDGE_MODEL", "claude-opus-4-6"),
            "calibration_models": [
                os.environ.get("MILO_JUDGE_MODEL", "claude-opus-4-6"),
                os.environ.get("MILO_CALIBRATION_MODEL_2", "gemini-2.5-pro"),
            ],
        },
    }
    terminal = {
        "ended_iso": started,
        "termination_reason": "submit",
        "verifier_report": {},
        "rubric_report": {},
        "reward_decomposition": rollout_decomposition,
        "r_total": float(rollout_decomposition.get("r_total", 1.0)),
        "cost_usd": 0.0,
        "tokens": {"prompt": 0, "completion": 0, "cache_read": 0, "cache_write": 0},
    }
    records: list[dict[str, Any]] = [header]
    if rollout_steps is not None:
        records.extend(rollout_steps)
    records.append(terminal)

    out_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_mismatches(mismatches: Iterable[ReplayMismatch]) -> str:
    """Pretty-print mismatches for a CI log line."""
    lines: list[str] = []
    for m in mismatches:
        delta_str = f", delta={m.delta:.3e}" if m.delta is not None else ""
        lines.append(
            f"  - {m.key}: stored={m.stored!r} replayed={m.replayed!r}{delta_str}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="milo.replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--instance",
        required=True,
        help="instance_id (e.g. locustio__locust-1541) to replay.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--record-golden",
        action="store_true",
        help="Drop into SME human-in-the-loop mode and persist a golden trace.",
    )
    group.add_argument(
        "--golden",
        action="store_true",
        help="Replay a stored golden trace; assert R_terminal == 1 and "
             "R_rubric == 1.0 per spec §10.1.3.",
    )
    p.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Override the trace.jsonl path. By default, locates the trace "
             "under MILO_TRACE_ROOT (or MILO_GOLDEN_ROOT with --golden).",
    )
    p.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=f"Override the trace root. Default: {DEFAULT_TRACE_ROOT}.",
    )
    p.add_argument(
        "--golden-root",
        type=Path,
        default=DEFAULT_GOLDEN_ROOT,
        help=f"Override the golden trace root. Default: {DEFAULT_GOLDEN_ROOT}.",
    )
    p.add_argument(
        "--sme-id",
        default=os.environ.get("USER", "sme"),
        help="SME reviewer id, used in policy_id 'sme:<id>' when "
             "--record-golden. Defaults to $USER.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=REWARD_TOLERANCE,
        help=f"Float-comparison tolerance. Default: spec §10.4 = {REWARD_TOLERANCE}.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the ReplayResult as JSON on stdout (for the nightly audit).",
    )
    return p


def run(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    if args.record_golden:
        # Interactive SME mode — outside of CI/unit-test scope, this
        # would attach to a gym REPL and stream the SME's tool calls.
        # For now we print a one-line instruction and exit non-zero so
        # the operator can't mistake an empty placeholder for a real
        # golden trace. The programmatic API is record_golden_trace().
        print(
            "[replay] --record-golden requires the interactive SME REPL "
            "(milo.lht_adapter.repl, Phase 1.4). Use the programmatic "
            "milo.replay.cli.record_golden_trace() in tests + scripts.",
            file=sys.stderr,
        )
        return 3

    mode = "golden" if args.golden else "default"
    try:
        result = replay_trace(
            args.instance,
            trace_path=args.trace_path,
            mode=mode,
            trace_root=args.trace_root,
            golden_root=args.golden_root,
            tolerance=args.tolerance,
        )
    except FileNotFoundError as e:
        print(f"[replay] {e}", file=sys.stderr)
        return 4
    except ValueError as e:
        print(f"[replay] malformed trace: {e}", file=sys.stderr)
        return 5

    if args.json:
        json.dump(
            {
                "instance_id": result.instance_id,
                "mode": result.mode,
                "matched": result.matched,
                "n_mismatches": len(result.mismatches),
                "mismatches": [
                    {"key": m.key, "stored": m.stored,
                     "replayed": m.replayed, "delta": m.delta}
                    for m in result.mismatches
                ],
                "golden_assertions_passed": result.golden_assertions_passed,
                "seed": result.seed,
                "policy_id": result.policy_id,
                "replay_iso": result.replay_iso,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        status = "OK" if result.matched else "MISMATCH"
        print(
            f"[replay] {result.instance_id} (mode={result.mode}, "
            f"seed={result.seed}, policy_id={result.policy_id}) -> {status}",
            file=sys.stderr,
        )
        if result.mismatches:
            print("[replay] mismatches:", file=sys.stderr)
            print(_format_mismatches(result.mismatches), file=sys.stderr)
        if mode == "golden" and result.golden_assertions_passed:
            for name, ok in result.golden_assertions_passed.items():
                print(
                    f"[replay] golden_assert: {name} = {'PASS' if ok else 'FAIL'}",
                    file=sys.stderr,
                )

    return 0 if result.matched else 1


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
