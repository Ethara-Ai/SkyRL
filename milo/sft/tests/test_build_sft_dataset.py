"""Unit tests for :mod:`milo.sft.build_sft_dataset`.

Covers the spec §16.2 contract:

* Synthetic golden trace → one JSONL line with system/user/assistant/tool
  messages in the correct order and loss-mask flags.
* Missing trace files are recorded in ``train_ids_missing``.
* Unparseable trace lines do not crash the build (``strict=False`` default).
* SHA-256 of the output is computed and matches the file on disk.
* Empty trace → ``examples_skipped`` increments; nothing written.
* The build report sidecar file is written next to the output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from milo.sft.build_sft_dataset import (
    SFTBuildError,
    SFTBuildReport,
    _resolve_trace_paths,
    build_sft_dataset,
)


def _write_trace(dir_path: Path, steps: list[dict]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "trace.jsonl"
    with path.open("w") as fh:
        for s in steps:
            fh.write(json.dumps(s) + "\n")
    return path


def _golden_steps() -> list[dict]:
    """A minimal but realistic golden trace: 2 tool calls + a final submit."""
    return [
        {
            "step": 0,
            "meta": {
                "system_prompt": "You are a SWE agent. <tools omitted>",
                "user_message": "Fix the failing test in pkg/foo.py.",
                "instance_id": "demo/repo__123",
            },
            "reasoning_text": "I will read the failing file first.",
            "action": {"tool": "read_file", "args": {"path": "pkg/foo.py"}},
            "tool_call_id": "call_0",
            "tool_result_text": "def foo():\n    return 1\n",
        },
        {
            "step": 1,
            "reasoning_text": "Now patch it.",
            "action": {
                "tool": "apply_patch",
                "args": {"patch": "--- a/pkg/foo.py\n+++ b/pkg/foo.py\n..."},
            },
            "tool_call_id": "call_1",
            "step_outputs": {"stdout": "patch applied", "stderr": ""},
        },
        {
            "step": 2,
            "reasoning_text": "All tests pass. Submit.",
            "action": {"tool": "submit", "args": {"summary": "Fixed foo."}},
            "tool_call_id": "call_2",
            # no stdout/stderr — submit has no observable I/O
        },
    ]


def test_happy_path(tmp_path: Path) -> None:
    src = tmp_path / "golden"
    out = tmp_path / "sft_train.jsonl"
    _write_trace(src / "demo__repo__123", _golden_steps())

    report = build_sft_dataset(src, ["demo__repo__123"], out)

    assert isinstance(report, SFTBuildReport)
    assert report.examples_written == 1
    assert report.examples_skipped == 0
    assert report.train_ids_missing == []
    assert out.is_file()

    line = json.loads(out.read_text().strip())
    assert line["instance_id"] == "demo/repo__123"
    assert line["source_rollout_id"] == "demo__repo__123"

    roles = [m["role"] for m in line["messages"]]
    # system, user, then assistant+tool pairs ending with submit (no tool reply)
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[2] == "assistant"
    assert roles[3] == "tool"
    assert roles[4] == "assistant"
    assert roles[5] == "tool"
    # Final submit has no tool reply
    assert roles[-1] == "assistant"

    # Loss masking: only assistant turns contribute to loss.
    for m in line["messages"]:
        if m["role"] == "assistant":
            assert m["loss_mask"] is True
            assert "tool_calls" in m
            assert isinstance(m["tool_calls"], list) and len(m["tool_calls"]) == 1
        else:
            assert m["loss_mask"] is False

    # Report sidecar exists and matches
    sidecar = out.with_suffix(out.suffix + ".build_report.json")
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text())
    assert data["examples_written"] == 1

    # SHA-256 round trip
    expected_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert report.sha256_output == expected_sha


def test_missing_train_id_recorded(tmp_path: Path) -> None:
    src = tmp_path / "golden"
    out = tmp_path / "out.jsonl"
    _write_trace(src / "present__id", _golden_steps())
    report = build_sft_dataset(src, ["present__id", "absent__id"], out)
    assert report.examples_written == 1
    assert "absent__id" in report.train_ids_missing


def test_empty_trace_skipped(tmp_path: Path) -> None:
    src = tmp_path / "golden"
    out = tmp_path / "out.jsonl"
    # Empty trace.jsonl
    d = src / "empty__id"
    d.mkdir(parents=True)
    (d / "trace.jsonl").write_text("")
    report = build_sft_dataset(src, ["empty__id"], out)
    assert report.examples_written == 0
    assert report.examples_skipped == 1


def test_resolver_layout_variants(tmp_path: Path) -> None:
    src = tmp_path / "golden"
    src.mkdir()
    # layout 1: <id>/trace.jsonl
    (src / "id1").mkdir()
    (src / "id1" / "trace.jsonl").write_text("{}")
    # layout 2: <id>.jsonl
    (src / "id2.jsonl").write_text("{}")
    # layout 3: <id>/golden/trace.jsonl
    (src / "id3" / "golden").mkdir(parents=True)
    (src / "id3" / "golden" / "trace.jsonl").write_text("{}")
    found, missing = _resolve_trace_paths(src, ["id1", "id2", "id3", "id4"])
    found_ids = [tid for tid, _ in found]
    assert sorted(found_ids) == ["id1", "id2", "id3"]
    assert missing == ["id4"]


def test_strict_mode_raises_on_no_system_prompt(tmp_path: Path) -> None:
    src = tmp_path / "golden"
    out = tmp_path / "out.jsonl"
    # No system_prompt — must fail in strict mode
    _write_trace(
        src / "id1",
        [{"step": 0, "action": {"tool": "submit", "args": {}}, "tool_call_id": "c"}],
    )
    with pytest.raises(SFTBuildError):
        build_sft_dataset(src, ["id1"], out, strict=True)


def test_malformed_jsonl_line_does_not_crash(tmp_path: Path) -> None:
    src = tmp_path / "golden"
    out = tmp_path / "out.jsonl"
    d = src / "id1"
    d.mkdir(parents=True)
    # First line OK, second line garbage, third line OK
    with (d / "trace.jsonl").open("w") as fh:
        fh.write(json.dumps(_golden_steps()[0]) + "\n")
        fh.write("{this is not json\n")
        fh.write(json.dumps(_golden_steps()[1]) + "\n")
    report = build_sft_dataset(src, ["id1"], out)
    # The good lines were enough to produce one example.
    assert report.examples_written == 1
