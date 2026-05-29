"""SFT-warmstart validation per ``RL_GYM_SPEC.md`` v0.7 §16.4.

The three acceptance criteria the SFT step must pass before being promoted to
the model registry (Phase 18.5) as ``qwen2.5-coder-32b-milo-sft-v<N>``:

1. **pass@8 ≥ vanilla Qwen2.5-Coder-32B-Instruct pass@8** on the 60-task
   validation split.
2. **Held-out CE loss ≤ training CE loss × 1.10** (no severe overfit).
3. **At least one f2p test passes on at least 30 % of validation tasks**
   (sanity check that the model can produce *any* correct patch).

This module owns the *gating logic* — it does **not** itself run the rollouts
or the held-out loss pass. Both are computed elsewhere (the eval harness in
:mod:`milo.eval` and the SFT runner ``main_sft.py`` respectively) and handed
to :func:`validate_sft_checkpoint` as artifact JSON paths. We keep this thin
on purpose so the gate runs in seconds on a laptop in CI.

Inputs (``val_split`` directory layout)::

    <val_split>/
        pass_at_8.json              # current checkpoint pass@8 numbers
        vanilla_pass_at_8.json      # the baseline shipped with the registry
        loss_report.json            # {"train_ce_loss": ..., "val_ce_loss": ...}
        per_task_f2p_pass.json      # {"task_id": true | false} (any f2p passed?)

The ``validate_sft_checkpoint`` function deliberately accepts a *directory*
(not the raw artifact files) so the SFT runner can prepare all four artifacts
together and the caller doesn't have to thread four paths. If only some
artifacts exist, the validation falls back to ``status="incomplete"``.

Exit codes follow the plan §12.3 contract: ``0`` on pass; non-zero with a
diagnostic on fail. See the ``main`` function for the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "SFTValidationReport",
    "validate_sft_checkpoint",
    "main",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SFTValidationReport:
    """Outcome of one §16.4 acceptance pass.

    ``passed`` is the AND of all three checks. ``status`` collapses individual
    criteria into one of:

    * ``"passed"``  — all three criteria met.
    * ``"failed"``  — one or more criteria not met.
    * ``"incomplete"`` — missing artifact(s); the caller should rebuild and
      retry before treating this as a failure.
    """

    checkpoint_path: Path
    val_split: Path
    passed: bool
    status: Literal["passed", "failed", "incomplete"]

    # Criterion 1: pass@8 vs vanilla.
    pass_at_8: float | None = None
    vanilla_pass_at_8: float | None = None
    pass_at_8_delta: float | None = None
    criterion_1_pass: bool | None = None

    # Criterion 2: held-out CE loss ratio.
    train_ce_loss: float | None = None
    val_ce_loss: float | None = None
    val_train_ratio: float | None = None
    criterion_2_pass: bool | None = None

    # Criterion 3: any-f2p-pass fraction.
    any_f2p_fraction: float | None = None
    criterion_3_pass: bool | None = None

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "checkpoint_path": str(self.checkpoint_path),
            "val_split": str(self.val_split),
            "passed": self.passed,
            "status": self.status,
            "criteria": {
                "1_pass_at_8_vs_vanilla": {
                    "pass_at_8": self.pass_at_8,
                    "vanilla_pass_at_8": self.vanilla_pass_at_8,
                    "delta": self.pass_at_8_delta,
                    "passed": self.criterion_1_pass,
                },
                "2_held_out_loss_ratio": {
                    "train_ce_loss": self.train_ce_loss,
                    "val_ce_loss": self.val_ce_loss,
                    "ratio": self.val_train_ratio,
                    "passed": self.criterion_2_pass,
                },
                "3_any_f2p_fraction": {
                    "fraction": self.any_f2p_fraction,
                    "threshold": 0.30,
                    "passed": self.criterion_3_pass,
                },
            },
            "notes": list(self.notes),
        }
        return d


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not read %s: %s", path, e)
        return None


def _coerce_pass_at_8(blob: Any) -> float | None:
    """Either a bare float or a dict like ``{"pass_at_8_overall": ...}``."""
    if isinstance(blob, (int, float)):
        return float(blob)
    if isinstance(blob, dict):
        for key in ("pass_at_8_overall", "pass_at_8", "overall"):
            v = blob.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def validate_sft_checkpoint(
    checkpoint_path: Path | str,
    val_split: Path | str,
    *,
    ratio_max: float = 1.10,
    any_f2p_min_fraction: float = 0.30,
) -> SFTValidationReport:
    """Run the §16.4 acceptance criteria and return a structured report.

    Parameters
    ----------
    checkpoint_path:
        The SFT checkpoint directory (passed through for provenance, not
        loaded — this function never touches torch).
    val_split:
        Directory containing ``pass_at_8.json``, ``vanilla_pass_at_8.json``,
        ``loss_report.json``, ``per_task_f2p_pass.json``. See module docstring.
    ratio_max:
        Maximum allowed ``val_ce / train_ce``. Spec §16.4 says ``1.10``.
    any_f2p_min_fraction:
        Minimum fraction of validation tasks that must have at least one f2p
        test pass. Spec §16.4 says ``0.30``.
    """
    ckpt = Path(checkpoint_path)
    split = Path(val_split)
    report = SFTValidationReport(
        checkpoint_path=ckpt, val_split=split, passed=False, status="incomplete"
    )

    if not split.is_dir():
        report.notes.append(f"val_split is not a directory: {split}")
        return report

    # ---- criterion 1: pass@8 vs vanilla ----------------------------------
    cur_blob = _read_json_if_exists(split / "pass_at_8.json")
    van_blob = _read_json_if_exists(split / "vanilla_pass_at_8.json")
    report.pass_at_8 = _coerce_pass_at_8(cur_blob)
    report.vanilla_pass_at_8 = _coerce_pass_at_8(van_blob)
    if report.pass_at_8 is None or report.vanilla_pass_at_8 is None:
        report.notes.append("missing pass_at_8.json or vanilla_pass_at_8.json")
    else:
        report.pass_at_8_delta = report.pass_at_8 - report.vanilla_pass_at_8
        report.criterion_1_pass = report.pass_at_8 >= report.vanilla_pass_at_8

    # ---- criterion 2: loss ratio -----------------------------------------
    loss_blob = _read_json_if_exists(split / "loss_report.json")
    if isinstance(loss_blob, dict):
        tr = loss_blob.get("train_ce_loss")
        vl = loss_blob.get("val_ce_loss")
        if isinstance(tr, (int, float)) and isinstance(vl, (int, float)) and tr > 0:
            report.train_ce_loss = float(tr)
            report.val_ce_loss = float(vl)
            report.val_train_ratio = float(vl) / float(tr)
            report.criterion_2_pass = report.val_train_ratio <= ratio_max
        else:
            report.notes.append("loss_report.json missing or malformed (need train_ce_loss + val_ce_loss)")
    else:
        report.notes.append("missing loss_report.json")

    # ---- criterion 3: any-f2p-pass fraction ------------------------------
    per_task = _read_json_if_exists(split / "per_task_f2p_pass.json")
    if isinstance(per_task, dict) and per_task:
        passed = sum(1 for v in per_task.values() if bool(v))
        report.any_f2p_fraction = passed / len(per_task)
        report.criterion_3_pass = report.any_f2p_fraction >= any_f2p_min_fraction
    else:
        report.notes.append("missing per_task_f2p_pass.json")

    # ---- collapse to pass / fail / incomplete ----------------------------
    if any(
        c is None
        for c in (report.criterion_1_pass, report.criterion_2_pass, report.criterion_3_pass)
    ):
        report.status = "incomplete"
        report.passed = False
    elif all((report.criterion_1_pass, report.criterion_2_pass, report.criterion_3_pass)):
        report.status = "passed"
        report.passed = True
    else:
        report.status = "failed"
        report.passed = False

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an SFT warmstart checkpoint (spec §16.4)")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--val-split", required=True, type=Path)
    parser.add_argument("--ratio-max", type=float, default=1.10)
    parser.add_argument("--any-f2p-min-fraction", type=float, default=0.30)
    parser.add_argument("--report-out", type=Path, default=None, help="Where to write the JSON report (default: stdout-only)")
    args = parser.parse_args(argv)

    report = validate_sft_checkpoint(
        args.checkpoint,
        args.val_split,
        ratio_max=args.ratio_max,
        any_f2p_min_fraction=args.any_f2p_min_fraction,
    )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.report_out:
        args.report_out.write_text(payload)
    sys.stdout.write(payload + "\n")

    if report.status == "passed":
        return 0
    if report.status == "incomplete":
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
