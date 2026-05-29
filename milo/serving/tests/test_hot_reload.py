"""Unit tests for :class:`milo.serving.hot_reload.HotReloadWatcher`.

We never hit a real vLLM endpoint — the ``http_post`` constructor hook lets us
inject a stub.

Covered:

* Sentinel + weights-file detection → POST is called with the right body.
* Sentinel is deleted on success.
* Sentinel is NOT deleted on failure (retry path).
* Multiple weights files → highest-numbered ``<step>.safetensors`` wins.
* Sentinel payload (``{"weights_path": ...}``) overrides directory scan.
* Empty / missing dir → ``poll_once`` returns ``None`` (no exception).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from milo.serving.hot_reload import (
    DEFAULT_RELOAD_ENDPOINT,
    DEFAULT_SENTINEL_NAME,
    HotReloadWatcher,
    ReloadEvent,
)


class FakeHTTP:
    """In-memory stand-in for the HTTP POST hook."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, payload: dict[str, Any]) -> int:
        self.calls.append((url, payload))
        return self.status


def _make_watcher(tmp_path: Path, http: FakeHTTP) -> HotReloadWatcher:
    return HotReloadWatcher(
        scratch_dir=tmp_path,
        server_url="http://test:9000",
        http_post=http,
        poll_interval_s=0.01,
    )


def _write_weights_and_sentinel(tmp_path: Path, step: int, sentinel_payload: dict | None = None) -> Path:
    weights = tmp_path / f"{step}.safetensors"
    weights.write_bytes(b"\x00")
    sentinel = tmp_path / DEFAULT_SENTINEL_NAME
    if sentinel_payload is None:
        sentinel.write_text("")
    else:
        sentinel.write_text(json.dumps(sentinel_payload))
    return weights


def test_no_sentinel_no_event(tmp_path: Path) -> None:
    http = FakeHTTP()
    w = _make_watcher(tmp_path, http)
    assert w.poll_once() is None
    assert http.calls == []


def test_missing_dir_does_not_crash(tmp_path: Path) -> None:
    http = FakeHTTP()
    w = HotReloadWatcher(scratch_dir=tmp_path / "does_not_exist", server_url="http://x", http_post=http)
    assert w.poll_once() is None


def test_success_path_deletes_sentinel(tmp_path: Path) -> None:
    http = FakeHTTP(status=200)
    w = _make_watcher(tmp_path, http)
    weights = _write_weights_and_sentinel(tmp_path, step=7)
    sentinel = tmp_path / DEFAULT_SENTINEL_NAME

    event = w.poll_once()
    assert isinstance(event, ReloadEvent)
    assert event.succeeded is True
    assert event.error is None
    assert event.step == 7
    assert event.weights_path == weights

    # Sentinel removed
    assert not sentinel.exists()
    # Endpoint received the expected payload
    assert len(http.calls) == 1
    url, payload = http.calls[0]
    assert url == "http://test:9000" + DEFAULT_RELOAD_ENDPOINT
    assert payload["weights_path"] == str(weights)
    assert payload["step"] == 7


def test_failure_keeps_sentinel(tmp_path: Path) -> None:
    http = FakeHTTP(status=500)
    w = _make_watcher(tmp_path, http)
    _write_weights_and_sentinel(tmp_path, step=3)
    sentinel = tmp_path / DEFAULT_SENTINEL_NAME

    event = w.poll_once()
    assert event is not None
    assert event.succeeded is False
    # Sentinel preserved for retry on the next poll
    assert sentinel.is_file()


def test_no_weights_logs_error_event(tmp_path: Path) -> None:
    http = FakeHTTP(status=200)
    w = _make_watcher(tmp_path, http)
    # Sentinel only, no weights
    (tmp_path / DEFAULT_SENTINEL_NAME).write_text("")
    event = w.poll_once()
    assert event is not None
    assert event.succeeded is False
    assert event.weights_path is None
    assert "no weights" in (event.error or "")
    # HTTP not called
    assert http.calls == []


def test_picks_highest_numbered_weights(tmp_path: Path) -> None:
    http = FakeHTTP(status=200)
    w = _make_watcher(tmp_path, http)
    for step in (10, 12, 11, 9):
        (tmp_path / f"{step}.safetensors").write_bytes(b"\x00")
    (tmp_path / DEFAULT_SENTINEL_NAME).write_text("")

    event = w.poll_once()
    assert event is not None
    assert event.succeeded is True
    assert event.step == 12


def test_sentinel_payload_overrides_directory_scan(tmp_path: Path) -> None:
    http = FakeHTTP(status=200)
    w = _make_watcher(tmp_path, http)
    other = tmp_path / "elsewhere.safetensors"
    other.write_bytes(b"\x00")
    # Add a misleading 9.safetensors that should be ignored
    (tmp_path / "9.safetensors").write_bytes(b"\x00")
    (tmp_path / DEFAULT_SENTINEL_NAME).write_text(
        json.dumps({"weights_path": str(other), "step": 42})
    )

    event = w.poll_once()
    assert event is not None
    assert event.succeeded is True
    assert event.weights_path == other
    assert event.step == 42


def test_history_is_bounded(tmp_path: Path) -> None:
    http = FakeHTTP(status=200)
    w = HotReloadWatcher(
        scratch_dir=tmp_path,
        server_url="http://test:9000",
        http_post=http,
        max_history=3,
    )
    for i in range(5):
        (tmp_path / f"{i}.safetensors").write_bytes(b"\x00")
        (tmp_path / DEFAULT_SENTINEL_NAME).write_text("")
        w.poll_once()
    assert len(w.history) == 3
