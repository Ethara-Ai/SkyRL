"""Recalibration CLI — spec §8 "Re-calibration policy" + plan §8.2.

Usage::

    python -m milo.calibration.recalibrate \\
        --tasks  /path/to/instances.jsonl \\
        --output /path/to/calibration_results.json
        [--models  claude-opus-4-6  gemini-2.5-pro]    # today's GA defaults
        [--k 8]

When ``--models`` is omitted, model IDs are resolved from the env vars per
spec §8:

    Model 1: ``${MILO_JUDGE_MODEL:-claude-opus-4-6}``
    Model 2: ``${MILO_CALIBRATION_MODEL_2:-gemini-2.5-pro}``

so that when a new frontier model lands (Anthropic ships Opus 4.7,
Google ships Gemini 2.6), the operator sets the env var and re-runs;
no code change.

The CLI is intentionally a thin shell over :class:`CalibrationRunner` and
the real Phase 7 adapters. Until Phase 7 lands, ``--use-stub`` produces a
deterministic dry-run useful for smoke testing the orchestration without
incurring API spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from milo.adapters import PolicyAdapter, make_stub_adapter
from milo.calibration.runner import (
    DEFAULT_K,
    CalibrationRunner,
    default_model_handles,
)


def _read_instance_ids(tasks_path: Path) -> list[str]:
    """Pull ``instance_id`` from each line of a JSONL task manifest.

    Accepts either a JSONL file of instance dicts (the milo-bench on-disk
    format — see ``milo/audit/audit_dataset.py``) or a plain text file
    with one ``instance_id`` per line. Auto-detected by the first
    non-blank line.
    """
    text = tasks_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    first = lines[0].lstrip()
    if first.startswith("{"):
        out: list[str] = []
        for ln in lines:
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"failed to parse JSONL line in {tasks_path}: {e}"
                ) from e
            if "instance_id" not in obj:
                raise ValueError(
                    f"JSONL row in {tasks_path} missing 'instance_id': {obj!r}"
                )
            out.append(str(obj["instance_id"]))
        return out
    # plain text mode
    return lines


def _build_adapters(
    model_handles: tuple[str, str],
    *,
    use_stub: bool,
    stub_pass_rates: tuple[float, float] = (0.35, 0.40),
) -> list[PolicyAdapter]:
    """Construct the two-adapter pair.

    Production path raises ``NotImplementedError`` until Phase 7's real
    adapters land; the stub path is used by ``--use-stub`` and by the
    Phase 8 unit tests for orchestration smoke.
    """
    if use_stub:
        return [
            make_stub_adapter(model_handles[0], pass_rate=stub_pass_rates[0]),
            make_stub_adapter(model_handles[1], pass_rate=stub_pass_rates[1]),
        ]
    # Real adapter path — to be wired up once Phase 7 ships
    # milo/adapters/{anthropic,gemini,bedrock,openai,vllm}.py.
    raise NotImplementedError(
        "Real adapters land in Phase 7 (plan §7). "
        "Use --use-stub for orchestration smoke tests, or wait for "
        "milo/adapters/anthropic.py + gemini.py to land."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Standalone for tests — see ``tests/test_runner.py``."""
    p = argparse.ArgumentParser(
        prog="milo.calibration.recalibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tasks",
        required=True,
        type=Path,
        help="Path to a JSONL of instance dicts (or plain text "
             "with one instance_id per line).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write calibration_results.json.",
    )
    p.add_argument(
        "--models",
        nargs=2,
        metavar=("MODEL_1", "MODEL_2"),
        default=None,
        help="Override the two-model pair. Defaults resolve from "
             "${MILO_JUDGE_MODEL:-claude-opus-4-6} and "
             "${MILO_CALIBRATION_MODEL_2:-gemini-2.5-pro}.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"pass@k k. Spec §8 fixes this at {DEFAULT_K}; override only "
             "for cost-constrained recalibration runs.",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Starting rollout seed (spec §9.2). Each rollout uses "
             "seed_base + i for i in [0, k).",
    )
    p.add_argument(
        "--use-stub",
        action="store_true",
        help="Use deterministic stub adapters (no API spend). For "
             "orchestration smoke + Phase 8 tests only.",
    )
    p.add_argument(
        "--stub-pass-rates",
        nargs=2,
        type=float,
        default=(0.35, 0.40),
        metavar=("RATE_1", "RATE_2"),
        help="Stub fallback pass rates (only meaningful with --use-stub).",
    )
    return p


def run(argv: Iterable[str] | None = None) -> int:
    """Run the recalibration CLI. Returns process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    handles = tuple(args.models) if args.models else default_model_handles()
    assert len(handles) == 2  # argparse nargs=2 guarantees this
    adapters = _build_adapters(
        handles,
        use_stub=args.use_stub,
        stub_pass_rates=tuple(args.stub_pass_rates),  # type: ignore[arg-type]
    )

    instance_ids = _read_instance_ids(args.tasks)
    if not instance_ids:
        print(f"[recalibrate] no tasks found in {args.tasks}", file=sys.stderr)
        return 2

    runner = CalibrationRunner(
        adapters=adapters,
        k=args.k,
        seed_base=args.seed_base,
    )
    print(
        f"[recalibrate] calibrating {len(instance_ids)} tasks with "
        f"models {handles[0]!r} + {handles[1]!r}, k={args.k}",
        file=sys.stderr,
    )
    results = runner.calibrate_all(instance_ids)
    runner.write_results(results, args.output)

    rejected = sum(1 for r in results.values() if r.rejected_for_disagreement)
    tier_counts: dict[str, int] = {}
    for r in results.values():
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
    print(
        f"[recalibrate] wrote {args.output} — "
        f"{len(results)} tasks, {rejected} rejected (disagreement), "
        f"tier counts: {tier_counts}",
        file=sys.stderr,
    )
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
