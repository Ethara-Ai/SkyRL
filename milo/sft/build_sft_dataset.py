"""Build a Qwen-format SFT JSONL dataset from OpenHands+overlay golden traces.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §12.1 and ``RL_GYM_SPEC.md`` v0.7 §16.2.

The function :func:`build_sft_dataset` is the only public surface. It is the
plan-side deliverable named *trainer/sft/build_sft_dataset.py* (in milo we keep
everything under ``milo/`` per the v0.4 namespace convention).

Output schema per JSONL line (matches HuggingFace ``chat`` SFT, with an explicit
``loss_mask`` array so downstream collators do not have to re-derive
assistant-only masking from chat-template rendering — important because the
Qwen2.5-Coder template emits a ``<tool_call>`` block we want masked-in while
``tool``/``system``/``user`` blocks stay masked-out)::

    {
      "messages": [
        {"role": "system",    "content": "...",                "loss_mask": false},
        {"role": "user",      "content": "...",                "loss_mask": false},
        {"role": "assistant", "content": "...",                "loss_mask": true,
         "tool_calls": [{"id": "...", "type": "function",
                          "function": {"name": "...", "arguments": "..."}}]},
        {"role": "tool",      "content": "...", "tool_call_id": "...",
                                                              "loss_mask": false},
        ...
        {"role": "assistant", "content": "final submit reasoning",
                                                              "loss_mask": true,
         "tool_calls": [{"id": "...", "function": {"name": "submit", ...}}]}
      ],
      "instance_id": "...",
      "source_rollout_id": "..."
    }

The golden-trace on-disk format is the OpenHands+overlay JSONL written by
``milo/logging/`` (Phase 5). We read one ``trace.jsonl`` per ``<rollout_id>/``
directory under ``golden_traces_dir`` and emit one line per converted trace.

Spec §16.2 details that *the assistant `content` field includes the model's
reasoning text up to the tool call*, while the tool-call JSON lives in the
dedicated ``tool_calls`` field. We honour that here: the assistant message
content is the pre-tool-call free-text reasoning the recorder captured under
``"reasoning_text"`` in the OpenHands step record (Phase 5.1 schema).

The function is **deliberately tolerant** of missing fields, because the
golden-trace schema is still evolving alongside Phase 11.11. Anything we cannot
parse goes into the per-trace ``parse_errors`` list returned in the per-line
``_warnings`` field — never silently dropped. Hard failures (no system prompt,
no user observation, no assistant turns) raise ``ValueError`` with the rollout
id so SME review can locate the offending source trace.

The function is **pure I/O + dict-shuffling**: no torch, no transformers, no
network — so it can run in CI for laptop-scale unit tests against the golden
trace corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SFTBuildReport",
    "SFTBuildError",
    "build_sft_dataset",
    "main",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


class SFTBuildError(ValueError):
    """Raised when a golden trace cannot be converted to SFT format.

    The ``rollout_id`` attribute names the offending rollout for SME triage.
    """

    def __init__(self, rollout_id: str, reason: str) -> None:
        super().__init__(f"[{rollout_id}] {reason}")
        self.rollout_id = rollout_id
        self.reason = reason


@dataclass(slots=True)
class SFTBuildReport:
    """Summary returned from :func:`build_sft_dataset`.

    Mirrored on disk as ``<output_path>.build_report.json`` so the Phase 19.0
    pre-flight checklist can verify the SFT data was built cleanly.
    """

    output_path: Path
    examples_written: int
    examples_skipped: int
    train_ids_requested: int
    train_ids_missing: list[str]
    rollouts_seen: int
    rollouts_with_warnings: int
    sha256_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "examples_written": self.examples_written,
            "examples_skipped": self.examples_skipped,
            "train_ids_requested": self.train_ids_requested,
            "train_ids_missing": list(self.train_ids_missing),
            "rollouts_seen": self.rollouts_seen,
            "rollouts_with_warnings": self.rollouts_with_warnings,
            "sha256_output": self.sha256_output,
        }


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------


def _read_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Read a single ``trace.jsonl`` file. Returns its step records.

    Tolerant of trailing blank lines and lines that fail to parse (those are
    logged at WARNING and skipped, mirroring ``milo/logging/recorder.py``'s
    write-then-recover semantics).
    """
    steps: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed line %s:%d (%s)", trace_path, lineno, e)
    return steps


def _extract_system_prompt(steps: Sequence[dict[str, Any]]) -> str | None:
    """The first step's ``meta.system_prompt`` is the rendered template.

    Falls back to the top-level ``"system_prompt"`` key, then to ``None``.
    """
    if not steps:
        return None
    first = steps[0]
    meta = first.get("meta") or {}
    sp = meta.get("system_prompt") or first.get("system_prompt")
    return sp if isinstance(sp, str) else None


def _extract_user_observation(steps: Sequence[dict[str, Any]]) -> str | None:
    """The ``reset`` observation. Looked up under ``meta.user_message`` then
    ``observation_text`` then None. See spec §4.1 for the canonical shape.
    """
    if not steps:
        return None
    first = steps[0]
    meta = first.get("meta") or {}
    for k in ("user_message", "user_prompt", "task_framing"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v
    obs = first.get("observation_text") or first.get("observation")
    if isinstance(obs, str) and obs.strip():
        return obs
    return None


def _step_assistant_payload(step: dict[str, Any]) -> dict[str, Any] | None:
    """Build the assistant chat message for one tool-call step.

    Per spec §16.2, the assistant ``content`` is the model's reasoning text
    *up to the tool call*. The tool call itself is in ``tool_calls`` so the
    Qwen chat template renders it as ``<tool_call>...</tool_call>``.
    """
    action = step.get("action")
    if not isinstance(action, dict):
        return None
    tool_name = action.get("tool") or action.get("name")
    if not tool_name:
        return None
    args = action.get("args") or action.get("arguments") or {}
    if isinstance(args, dict):
        arguments = json.dumps(args, sort_keys=True, separators=(",", ":"))
    elif isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps({"raw": args})
    reasoning = step.get("reasoning_text") or step.get("assistant_text") or ""
    call_id = step.get("tool_call_id") or f"call_{step.get('step', 0)}"
    return {
        "role": "assistant",
        "content": reasoning,
        "loss_mask": True,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": arguments},
            }
        ],
    }


def _step_tool_response(step: dict[str, Any]) -> dict[str, Any] | None:
    """Tool response message for one step. Pulled from ``step_outputs`` or
    inline ``tool_result_text``. Spec §16.2 requires the *full untruncated*
    tool I/O.
    """
    action = step.get("action") or {}
    call_id = step.get("tool_call_id") or f"call_{step.get('step', 0)}"
    # Prefer the inline already-decoded text — recorder writes both.
    text = step.get("tool_result_text")
    if not isinstance(text, str):
        outputs = step.get("step_outputs") or {}
        if isinstance(outputs, dict):
            stdout = outputs.get("stdout") or ""
            stderr = outputs.get("stderr") or ""
            text = (stdout + ("\n" + stderr if stderr else "")).rstrip()
        else:
            text = ""
    if not text:
        # Some tools (e.g. submit) have no observable output.
        return None
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": text,
        "loss_mask": False,
    }


def _convert_trace(
    rollout_id: str,
    trace_path: Path,
    *,
    strict: bool,
) -> dict[str, Any] | None:
    """Convert one ``trace.jsonl`` into a single SFT example dict.

    Returns ``None`` (and logs a warning) if the trace cannot be converted and
    ``strict=False``; raises :class:`SFTBuildError` if ``strict=True``.
    """
    steps = _read_trace(trace_path)
    if not steps:
        msg = f"empty trace at {trace_path}"
        if strict:
            raise SFTBuildError(rollout_id, msg)
        logger.warning(msg)
        return None

    sys_prompt = _extract_system_prompt(steps)
    user_obs = _extract_user_observation(steps)
    if not sys_prompt or not user_obs:
        msg = "missing system_prompt or user observation in step 0"
        if strict:
            raise SFTBuildError(rollout_id, msg)
        logger.warning("%s for %s", msg, rollout_id)
        return None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt, "loss_mask": False},
        {"role": "user", "content": user_obs, "loss_mask": False},
    ]

    warnings: list[str] = []
    assistant_turns_seen = 0
    instance_id = (steps[0].get("meta") or {}).get("instance_id") or steps[0].get("instance_id") or rollout_id
    for step in steps:
        assistant_msg = _step_assistant_payload(step)
        if assistant_msg is None:
            warnings.append(f"step {step.get('step')} has no parseable action")
            continue
        messages.append(assistant_msg)
        assistant_turns_seen += 1
        tool_msg = _step_tool_response(step)
        if tool_msg is not None:
            messages.append(tool_msg)

    if assistant_turns_seen == 0:
        msg = "trace contains no assistant turns"
        if strict:
            raise SFTBuildError(rollout_id, msg)
        logger.warning("%s for %s", msg, rollout_id)
        return None

    return {
        "instance_id": instance_id,
        "source_rollout_id": rollout_id,
        "messages": messages,
        "_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Top-level build function
# ---------------------------------------------------------------------------


def _resolve_trace_paths(
    golden_traces_dir: Path, train_ids: Sequence[str]
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Find ``trace.jsonl`` for each requested id under ``golden_traces_dir``.

    Layout precedence (matches Phase 5 + plan §11.11):
        <dir>/<id>/trace.jsonl
        <dir>/<id>.jsonl
        <dir>/<id>/golden/trace.jsonl

    Returns ``(found, missing)``.
    """
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    for tid in train_ids:
        candidates = [
            golden_traces_dir / tid / "trace.jsonl",
            golden_traces_dir / f"{tid}.jsonl",
            golden_traces_dir / tid / "golden" / "trace.jsonl",
        ]
        for c in candidates:
            if c.is_file():
                found.append((tid, c))
                break
        else:
            missing.append(tid)
    return found, missing


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sft_dataset(
    golden_traces_dir: Path | str,
    train_ids: Sequence[str],
    output_path: Path | str,
    *,
    strict: bool = False,
    report_path: Path | str | None = None,
) -> SFTBuildReport:
    """Convert OpenHands+overlay golden traces into Qwen-format SFT JSONL.

    Parameters
    ----------
    golden_traces_dir:
        Root directory containing one subdirectory per rollout, each with a
        ``trace.jsonl`` (the OpenHands+overlay format from Phase 5).
    train_ids:
        Rollout (instance) ids to include — typically the 240 train-split ids
        from ``trainer/data/splits.json``.
    output_path:
        Where to write the JSONL. Parent directory is created if absent.
    strict:
        If ``True``, the first :class:`SFTBuildError` aborts the build. If
        ``False`` (default), unparseable traces are logged at WARNING and the
        ``examples_skipped`` counter is incremented.
    report_path:
        Optional explicit path for the JSON report. Defaults to
        ``<output_path>.build_report.json``.

    Returns
    -------
    SFTBuildReport
        Summary statistics plus a SHA-256 of the output file (also recorded in
        the reproducibility manifest — see Phase 14.7 / spec §21.2).
    """
    src = Path(golden_traces_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"golden_traces_dir does not exist: {src}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    found, missing = _resolve_trace_paths(src, train_ids)
    examples_written = 0
    examples_skipped = 0
    rollouts_with_warnings = 0

    with out.open("w", encoding="utf-8") as fh:
        for rollout_id, trace_path in found:
            try:
                example = _convert_trace(rollout_id, trace_path, strict=strict)
            except SFTBuildError:
                if strict:
                    raise
                examples_skipped += 1
                continue
            if example is None:
                examples_skipped += 1
                continue
            if example.pop("_warnings", None):
                rollouts_with_warnings += 1
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            examples_written += 1

    sha = _sha256_file(out) if out.is_file() else ""
    report = SFTBuildReport(
        output_path=out,
        examples_written=examples_written,
        examples_skipped=examples_skipped,
        train_ids_requested=len(train_ids),
        train_ids_missing=missing,
        rollouts_seen=len(found),
        rollouts_with_warnings=rollouts_with_warnings,
        sha256_output=sha,
    )
    report_target = Path(report_path) if report_path else out.with_suffix(out.suffix + ".build_report.json")
    report_target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    logger.info(
        "wrote %d SFT examples (skipped %d, missing %d) to %s",
        examples_written,
        examples_skipped,
        len(missing),
        out,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_train_ids(train_ids: str | None, train_ids_file: str | None) -> list[str]:
    if train_ids:
        return [tid.strip() for tid in train_ids.split(",") if tid.strip()]
    if train_ids_file:
        p = Path(train_ids_file)
        if p.suffix == ".json":
            return list(json.loads(p.read_text()))
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    raise SystemExit("Pass --train-ids or --train-ids-file")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Qwen-format SFT dataset (plan §12.1)")
    parser.add_argument("--traces", required=True, type=Path, help="Golden traces root dir")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--train-ids", type=str, default=None, help="Comma-separated rollout ids")
    parser.add_argument("--train-ids-file", type=str, default=None, help="File with one rollout id per line")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on first unconvertable trace (default: warn-and-skip)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("MILO_LOG_LEVEL", "INFO"),
        help="Python log level (default: INFO, override via MILO_LOG_LEVEL)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s %(message)s")
    ids = _parse_train_ids(args.train_ids, args.train_ids_file)
    report = build_sft_dataset(args.traces, ids, args.output, strict=args.strict)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    # Exit non-zero if every input failed.
    return 0 if report.examples_written > 0 else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
