"""`RolloutRecorder` — per-rollout atomic JSONL writer.

Implements RL_GYM_SPEC.md v0.7 §5.5 (per-rollout log directory layout +
atomic completion via `.completed` sentinel) and IMPLEMENTATION_PLAN.md
v0.4 §5.2. Buffer per-step shaping records in memory; on `finalize()`,
merge the OpenHands rollout dict with a `milo_extension` overlay and
atomically write `<rollout_id>.jsonl` + `<rollout_id>.completed`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from milo.logging.trace_format import (
    MiloExtension,
    StepShaping,
    TerminalSummary,
    attach_overlay,
)

logger = logging.getLogger(__name__)


class RolloutRecorder:
    """Stateful recorder for one rollout's worth of trace data.

    Use pattern:

        rec = RolloutRecorder(log_root=Path("/var/log/milo/run42"), rollout_id="abc")
        rec.record_step(0, shaping_reward=0.0, working_tree_patch="")
        rec.record_step(1, shaping_reward=1.5, working_tree_patch="diff ...")
        rec.finalize(openhands_rollout_dict, terminal_summary)

    Parameters
    ----------
    log_root:
        Per-run directory. Created if missing.
    rollout_id:
        Unique identifier. Drives the output filenames `<rollout_id>.jsonl`
        and `<rollout_id>.completed`.
    gym_version:
        Stamped on the overlay. Pass the running gym build's semver.
    reward_config:
        The `{alpha, beta, lambda, gamma, preset}` knob set used for this
        rollout — captured so the trace is self-describing.
    """

    def __init__(
        self,
        log_root: Path,
        rollout_id: str,
        *,
        gym_version: str = "",
        reward_config: dict | None = None,
    ) -> None:
        if not rollout_id:
            raise ValueError("rollout_id must be non-empty")
        # Forbid path separators so the rollout_id can't escape log_root.
        if os.sep in rollout_id or "/" in rollout_id or "\\" in rollout_id:
            raise ValueError(f"rollout_id must not contain path separators: {rollout_id!r}")

        self._log_root = Path(log_root)
        self._rollout_id = rollout_id
        self._gym_version = gym_version
        self._reward_config = dict(reward_config or {})

        self._log_root.mkdir(parents=True, exist_ok=True)
        self._steps: list[StepShaping] = []
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # File-path helpers
    # ------------------------------------------------------------------
    @property
    def output_path(self) -> Path:
        """Final `.jsonl` location after `finalize()` runs."""

        return self._log_root / f"{self._rollout_id}.jsonl"

    @property
    def completed_sentinel_path(self) -> Path:
        """0-byte sentinel that signals durability. Spec §5.5."""

        return self._log_root / f"{self._rollout_id}.completed"

    # ------------------------------------------------------------------
    # Public recording API
    # ------------------------------------------------------------------
    def record_step(
        self,
        step_index: int,
        shaping_reward: float,
        working_tree_patch: str,
    ) -> None:
        """Append a per-step overlay record.

        Buffered in memory until `finalize()` — at the per-rollout scale
        (≤ 1000 tool calls per spec §4.5) this is well under 1 MB of state.
        """

        if self._finalized:
            raise RuntimeError("RolloutRecorder.record_step called after finalize")
        self._steps.append(
            StepShaping(
                step_index=int(step_index),
                shaping_reward=float(shaping_reward),
                working_tree_patch=str(working_tree_patch),
            )
        )

    def finalize(
        self,
        openhands_rollout: dict,
        terminal: TerminalSummary,
    ) -> None:
        """Merge overlay → atomic-write rollout → drop the `.completed` sentinel.

        Sequence:
          1. Build `MiloExtension` from in-memory step buffer + the
             passed-in `TerminalSummary`.
          2. `attach_overlay()` onto the OpenHands rollout dict (defensive
             copy; we don't mutate the input).
          3. Serialise to JSON on one line (OpenHands `.jsonl` convention).
          4. Write `<output>.tmp` → fsync → `os.rename` to `<output>`.
          5. Create the 0-byte `.completed` sentinel and fsync the dir.

        After step 4 the rollout JSONL exists at its final path. The
        sentinel is the contract for "everything is on disk." Consumers
        (S3 syncer, orchestrator) must wait for the sentinel before
        treating the rollout as durable.
        """

        if self._finalized:
            raise RuntimeError("RolloutRecorder.finalize called twice")
        if not isinstance(openhands_rollout, dict):
            raise TypeError("openhands_rollout must be a dict")
        if not isinstance(terminal, TerminalSummary):
            raise TypeError("terminal must be a TerminalSummary")

        overlay = MiloExtension(
            gym_version=self._gym_version,
            reward_config=self._reward_config,
            per_step_shaping=list(self._steps),
            terminal_summary=terminal,
        )
        enriched = attach_overlay(openhands_rollout, overlay)

        # One-line JSON per OpenHands convention; the rollout JSONL is
        # typically one rollout per file in our setup, but we keep the
        # JSONL extension for compatibility with multi-rollout files.
        serialised = json.dumps(enriched, ensure_ascii=False) + "\n"

        # Atomic write: tmp file in the same dir → fsync → rename.
        # `tempfile.NamedTemporaryFile(delete=False)` gives us a uniquely
        # named tmp that we manually rename. Use `dir=self._log_root` so
        # the rename is on the same filesystem (atomic guarantee).
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._log_root),
            prefix=f"{self._rollout_id}.",
            suffix=".jsonl.tmp",
            delete=False,
        )
        try:
            tmp_handle.write(serialised)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
            tmp_path = Path(tmp_handle.name)
        finally:
            tmp_handle.close()

        os.replace(tmp_path, self.output_path)

        # Sentinel last — its presence promises the data file is durable.
        with open(self.completed_sentinel_path, "wb") as sentinel:
            sentinel.flush()
            os.fsync(sentinel.fileno())

        # fsync the directory so the rename + sentinel hit disk together.
        try:
            dir_fd = os.open(str(self._log_root), os.O_RDONLY)
        except OSError:
            # Not all filesystems support directory fsync (notably some
            # remote mounts). The data is still durable via fsync above.
            dir_fd = -1
        if dir_fd >= 0:
            try:
                os.fsync(dir_fd)
            except OSError:  # pragma: no cover - platform-dependent
                logger.debug("Directory fsync not supported for %s", self._log_root)
            finally:
                os.close(dir_fd)

        self._finalized = True


__all__ = ["RolloutRecorder"]
