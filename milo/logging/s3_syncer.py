"""Background S3 syncer for completed rollout traces — Phase 5.

Implements `IMPLEMENTATION_PLAN.md` v0.4 §5.3 and spec §5.5.

Polls `<log_root>` for `*.completed` 0-byte sentinels (written atomically by
`milo/logging/recorder.py:RolloutRecorder.finalize()`) and uploads each
rollout's trace file to `s3://{MILO_LOG_BUCKET}/{run_id}/{rollout_id}/`
with KMS encryption. Local-only mode (no S3) when `MILO_LOG_BUCKET` is unset
— intended for development and the spike.

Retry semantics: 3 attempts with exponential backoff (1s, 2s, 4s). On
permanent failure, raises an event that `milo/observability/alarms.py`
picks up (Phase 17 alarm 1).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("milo.logging.s3_syncer")


@dataclass
class SyncResult:
    rollout_id: str
    local_path: Path
    s3_uri: str | None      # None when running in local-only mode
    attempts: int
    success: bool
    error: str = ""


class S3Syncer:
    """Polls a log root for completed rollouts and uploads them to S3.

    Designed to be run as a long-lived background process (e.g. via the
    Slurm script at `milo/slurm/serve_policy.slurm`) or invoked
    synchronously by the test suite. The two modes share the same
    `sync_once()` entry point.
    """

    SENTINEL_SUFFIX = ".completed"
    DEFAULT_KMS_KEY_ENV = "MILO_S3_KMS_KEY_ID"

    def __init__(
        self,
        log_root: Path,
        run_id: str,
        bucket: str | None = None,
        kms_key_id: str | None = None,
        boto3_client: Any | None = None,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        alarm_hook: Callable[[SyncResult], None] | None = None,
    ) -> None:
        self.log_root = Path(log_root)
        self.run_id = run_id
        self.bucket = bucket if bucket is not None else os.environ.get("MILO_LOG_BUCKET", "")
        self.kms_key_id = (
            kms_key_id
            if kms_key_id is not None
            else os.environ.get(self.DEFAULT_KMS_KEY_ENV, "")
        )
        self._client = boto3_client
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._alarm_hook = alarm_hook
        # Track which sentinels we've already processed in this process lifetime
        # so a long-running syncer doesn't double-upload the same rollout.
        self._seen: set[Path] = set()

    # ------------------------------------------------------------------ public

    def is_local_only(self) -> bool:
        return not self.bucket

    def sync_once(self) -> list[SyncResult]:
        """Scan the log root, upload any new completed rollouts, return results."""
        results: list[SyncResult] = []
        for sentinel in sorted(self.log_root.glob(f"*{self.SENTINEL_SUFFIX}")):
            if sentinel in self._seen:
                continue
            rollout_id = sentinel.name[: -len(self.SENTINEL_SUFFIX)]
            trace_path = self.log_root / f"{rollout_id}.jsonl"
            if not trace_path.is_file():
                logger.warning(
                    "sentinel %s present but trace %s missing — skipping",
                    sentinel, trace_path,
                )
                self._seen.add(sentinel)
                continue
            res = self._sync_one(rollout_id, trace_path)
            results.append(res)
            if res.success:
                self._seen.add(sentinel)
            elif self._alarm_hook is not None:
                self._alarm_hook(res)
        return results

    def run_forever(self, poll_interval_s: float = 5.0) -> None:
        """Long-running loop. Caller owns SIGTERM handling."""
        while True:
            self.sync_once()
            time.sleep(poll_interval_s)

    # ----------------------------------------------------------------- private

    def _sync_one(self, rollout_id: str, trace_path: Path) -> SyncResult:
        if self.is_local_only():
            logger.debug("local-only mode — not uploading %s", trace_path)
            return SyncResult(
                rollout_id=rollout_id,
                local_path=trace_path,
                s3_uri=None,
                attempts=0,
                success=True,
            )
        client = self._get_client()
        key = f"{self.run_id}/{rollout_id}/trace.jsonl"
        extra_args: dict[str, Any] = {}
        if self.kms_key_id:
            extra_args.update(
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=self.kms_key_id,
            )
        else:
            extra_args["ServerSideEncryption"] = "AES256"

        last_err = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                client.upload_file(
                    Filename=str(trace_path),
                    Bucket=self.bucket,
                    Key=key,
                    ExtraArgs=extra_args,
                )
                return SyncResult(
                    rollout_id=rollout_id,
                    local_path=trace_path,
                    s3_uri=f"s3://{self.bucket}/{key}",
                    attempts=attempt,
                    success=True,
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_attempts:
                    time.sleep(self._backoff_base * (2 ** (attempt - 1)))
        return SyncResult(
            rollout_id=rollout_id,
            local_path=trace_path,
            s3_uri=None,
            attempts=self._max_attempts,
            success=False,
            error=last_err,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "boto3 not installed — install via `--extra milo` or set MILO_LOG_BUCKET=''"
            ) from exc
        self._client = boto3.client("s3")
        return self._client
