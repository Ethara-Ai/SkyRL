"""Tests for milo.tools.checkpoint_verify."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from milo.tools.checkpoint_verify import verify_checkpoint


def _make_ckpt(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (p / "tokenizer_config.json").write_text("{}")
    (p / "model.safetensors").write_bytes(b"binary weights here")


def test_verify_missing_dir(tmp_path: Path) -> None:
    rep = verify_checkpoint(tmp_path / "nope")
    assert not rep.ok
    assert not rep.exists


def test_verify_complete_checkpoint(tmp_path: Path) -> None:
    _make_ckpt(tmp_path)
    rep = verify_checkpoint(tmp_path)
    assert rep.ok, rep.to_dict()


def test_verify_missing_weights(tmp_path: Path) -> None:
    _make_ckpt(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    rep = verify_checkpoint(tmp_path)
    assert not rep.ok
    assert not rep.weights_present


def test_verify_sha256_matches(tmp_path: Path) -> None:
    _make_ckpt(tmp_path)
    sha = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    rep = verify_checkpoint(tmp_path, expected_sha256={"config.json": sha})
    assert rep.ok


def test_verify_sha256_mismatch(tmp_path: Path) -> None:
    _make_ckpt(tmp_path)
    rep = verify_checkpoint(tmp_path, expected_sha256={"config.json": "deadbeef"})
    assert not rep.ok
    assert rep.sha256_mismatches
