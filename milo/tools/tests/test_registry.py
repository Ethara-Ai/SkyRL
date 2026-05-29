"""Tests for milo.tools.registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from milo.tools.registry import ModelRegistry


def test_register_and_get(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register("m1", "/tmp/foo", manifest={"k": 1})
    got = reg.get("m1")
    assert got.path == "/tmp/foo"
    assert got.manifest == {"k": 1}


def test_double_register_same_path_is_idempotent(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register("m", "/tmp/foo")
    reg.register("m", "/tmp/foo")
    assert reg.list_names() == ["m"]


def test_double_register_different_path_raises(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register("m", "/tmp/foo")
    with pytest.raises(ValueError):
        reg.register("m", "/tmp/bar")


def test_override_updates_path(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register("m", "/tmp/foo")
    reg.register("m", "/tmp/bar", override=True)
    assert reg.get("m").path == "/tmp/bar"


def test_persistence(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    a = ModelRegistry(path)
    a.register("m", "/tmp/foo")
    b = ModelRegistry(path)
    assert "m" in b.list_names()


def test_unregister(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register("m", "/tmp/foo")
    reg.unregister("m")
    assert "m" not in reg.list_names()
