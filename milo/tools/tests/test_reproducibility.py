"""Tests for milo.tools.reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from milo.tools.reproducibility import ManifestIncomplete, write_manifest


def test_write_complete_manifest(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"; train.write_bytes(b"train bytes")
    hold = tmp_path / "hold.parquet";   hold.write_bytes(b"hold bytes")
    cfg = {"trainer": {"strategy": "fsdp"}, "generator": {"max_turns": 30}}
    m = write_manifest(
        run_dir=tmp_path / "run",
        cfg=cfg,
        base_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        base_model_revision="abc123",
        dataset_version="milo-lht-v1",
        train_split_path=train,
        holdout_split_path=hold,
    )
    written = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert written["base_model"] == m.base_model
    assert written["hyperparameters"]["trainer"]["strategy"] == "fsdp"
    assert written["train_split_sha256"] == hashlib.sha256(b"train bytes").hexdigest()


def test_missing_required_field_raises(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"; train.write_bytes(b"x")
    hold = tmp_path / "hold.parquet";   hold.write_bytes(b"y")
    with pytest.raises(ManifestIncomplete):
        write_manifest(
            run_dir=tmp_path / "run",
            cfg={"a": 1},
            base_model="",          # missing!
            base_model_revision="r",
            dataset_version="milo-lht-v1",
            train_split_path=train,
            holdout_split_path=hold,
        )
