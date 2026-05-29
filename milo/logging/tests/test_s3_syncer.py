"""Tests for milo.logging.s3_syncer — uses a fake boto3 client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from milo.logging.s3_syncer import S3Syncer, SyncResult


@dataclass
class _FakeS3Client:
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_first_n: int = 0
    _seen: int = 0

    def upload_file(
        self, Filename: str, Bucket: str, Key: str, ExtraArgs: dict[str, Any]
    ) -> None:
        self._seen += 1
        self.calls.append({"Filename": Filename, "Bucket": Bucket, "Key": Key, "ExtraArgs": ExtraArgs})
        if self._seen <= self.fail_first_n:
            raise RuntimeError("simulated upload failure")


def _write_completed_rollout(log_root: Path, rollout_id: str) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    trace = log_root / f"{rollout_id}.jsonl"
    trace.write_text(json.dumps({"instance_id": rollout_id, "history": []}) + "\n")
    sentinel = log_root / f"{rollout_id}.completed"
    sentinel.write_text("")
    return trace


def test_local_only_mode_no_uploads(tmp_path: Path) -> None:
    _write_completed_rollout(tmp_path, "r1")
    syncer = S3Syncer(log_root=tmp_path, run_id="run-1", bucket="", boto3_client=None)
    assert syncer.is_local_only()
    results = syncer.sync_once()
    assert len(results) == 1
    assert results[0].success
    assert results[0].s3_uri is None
    assert results[0].attempts == 0


def test_successful_upload(tmp_path: Path) -> None:
    _write_completed_rollout(tmp_path, "r1")
    fake = _FakeS3Client()
    syncer = S3Syncer(
        log_root=tmp_path, run_id="run-1", bucket="b", kms_key_id="kms-key", boto3_client=fake,
    )
    results = syncer.sync_once()
    assert len(results) == 1
    assert results[0].success
    assert results[0].s3_uri == "s3://b/run-1/r1/trace.jsonl"
    assert len(fake.calls) == 1
    assert fake.calls[0]["ExtraArgs"]["SSEKMSKeyId"] == "kms-key"


def test_retry_then_succeed(tmp_path: Path) -> None:
    _write_completed_rollout(tmp_path, "r1")
    fake = _FakeS3Client(fail_first_n=2)
    syncer = S3Syncer(
        log_root=tmp_path, run_id="run-1", bucket="b", boto3_client=fake,
        max_attempts=3, backoff_base=0.0,
    )
    results = syncer.sync_once()
    assert results[0].success
    assert results[0].attempts == 3


def test_retry_exhausted_alarms(tmp_path: Path) -> None:
    _write_completed_rollout(tmp_path, "r1")
    fake = _FakeS3Client(fail_first_n=99)
    alarmed: list[SyncResult] = []
    syncer = S3Syncer(
        log_root=tmp_path, run_id="run-1", bucket="b", boto3_client=fake,
        max_attempts=2, backoff_base=0.0, alarm_hook=alarmed.append,
    )
    results = syncer.sync_once()
    assert not results[0].success
    assert results[0].attempts == 2
    assert len(alarmed) == 1
    assert alarmed[0].rollout_id == "r1"


def test_sentinel_without_trace_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "r1.completed").write_text("")
    syncer = S3Syncer(log_root=tmp_path, run_id="run-1", bucket="b", boto3_client=_FakeS3Client())
    results = syncer.sync_once()
    assert results == []


def test_no_double_upload(tmp_path: Path) -> None:
    _write_completed_rollout(tmp_path, "r1")
    fake = _FakeS3Client()
    syncer = S3Syncer(log_root=tmp_path, run_id="run-1", bucket="b", boto3_client=fake)
    syncer.sync_once()
    syncer.sync_once()
    syncer.sync_once()
    assert len(fake.calls) == 1
