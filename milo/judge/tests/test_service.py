"""Tests for `RubricJudgeService` with a stub backend.

We never make a real Bedrock / Anthropic call in CI — every test injects a
`StubJudgeBackend` that returns a hand-crafted JSON string. This isolates
the service's parsing, scoring, tampering-floor, and report-shape logic
from any network dependency. The cache tests in `test_cache.py` cover
persistence; this file covers the per-call semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from milo.judge.backends.base import JudgeBackend
from milo.judge.service import (
    PerItemScore,
    RubricJudgeService,
    RubricReport,
    _coerce_to_valid_score,
)


# ---------------------------------------------------------------------
# Stub backend
# ---------------------------------------------------------------------
@dataclass
class StubJudgeBackend:
    """Hand-rolled `JudgeBackend` for tests.

    Returns whatever `response` is set to (default: a perfect-score reply
    against any rubric). Records every call into `calls` for assertion.
    """

    response: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    raise_on_call: bool = False

    def call(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:  # noqa: D401
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": model,
                "temperature": temperature,
            }
        )
        if self.raise_on_call:
            from milo.judge.backends.base import JudgeBackendError
            raise JudgeBackendError("stub error")
        return self.response


# Protocol conformance check — caught at runtime to ensure the stub
# satisfies `JudgeBackend` without inheritance.
def test_stub_satisfies_protocol() -> None:
    stub = StubJudgeBackend()
    # `runtime_checkable` Protocol — isinstance is structural.
    assert isinstance(stub, JudgeBackend)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def rubric_items() -> list[dict[str, Any]]:
    return [
        {"item_id": "r1", "natural_language_assertion": "The patch fixes the off-by-one in foo()."},
        {"item_id": "r2", "natural_language_assertion": "The patch adds a regression test."},
        {"item_id": "r3", "natural_language_assertion": "The patch does not break the public API."},
    ]


def _perfect_response(item_ids: list[str], *, tampering: bool = False) -> str:
    """Helper: build a JSON judge reply scoring every item 1."""

    payload = {
        "items": [
            {"item_id": iid, "score": 1, "justification": f"line 42 satisfies {iid}"}
            for iid in item_ids
        ],
        "tampering_detected": tampering,
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_perfect_score(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """A judge reply giving every item 1 → mean_score == 1.0."""

    stub = StubJudgeBackend(response=_perfect_response(["r1", "r2", "r3"]))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="--- a\n+++ b\n", submit_summary=None)
    assert isinstance(report, RubricReport)
    assert report.mean_score == 1.0
    assert report.tampering_detected is False
    assert [p.score for p in report.per_item] == [1.0, 1.0, 1.0]
    assert all(p.justification.startswith("line 42") for p in report.per_item)


def test_mixed_scores(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Mixed `{0, 0.5, 1}` scores produce the correct mean."""

    payload = {
        "items": [
            {"item_id": "r1", "score": 1, "justification": "foo"},
            {"item_id": "r2", "score": 0.5, "justification": "bar"},
            {"item_id": "r3", "score": 0, "justification": "baz"},
        ],
        "tampering_detected": False,
    }
    stub = StubJudgeBackend(response=json.dumps(payload))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="diff", submit_summary="summary")
    assert report.mean_score == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_tampering_hard_floor(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """When `tampering_detected=True`, every item score → 0 and mean → 0.

    Spec §4.4.3 / §I-6: the rubric reward is hard-floored to zero when
    the judge flags test-fixture tampering, regardless of the per-item
    scores the judge tried to emit.
    """

    payload = {
        "items": [
            {"item_id": "r1", "score": 1, "justification": "(would be 1)"},
            {"item_id": "r2", "score": 1, "justification": "(would be 1)"},
            {"item_id": "r3", "score": 0.5, "justification": "(would be 0.5)"},
        ],
        "tampering_detected": True,
    }
    stub = StubJudgeBackend(response=json.dumps(payload))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="diff")
    assert report.tampering_detected is True
    assert report.mean_score == 0.0
    assert all(p.score == 0.0 for p in report.per_item)
    assert all(p.justification == "test-fixture-tampering" for p in report.per_item)


def test_judge_omitted_item_gets_zero(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Items the judge forgot to score appear as 0 with placeholder text."""

    payload = {
        "items": [
            {"item_id": "r1", "score": 1, "justification": "good"},
            # r2 omitted!
            {"item_id": "r3", "score": 0.5, "justification": "meh"},
        ],
        "tampering_detected": False,
    }
    stub = StubJudgeBackend(response=json.dumps(payload))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="diff")
    by_id = {p.item_id: p for p in report.per_item}
    assert by_id["r2"].score == 0.0
    assert by_id["r2"].justification == "judge-omitted-item"


def test_empty_rubric_short_circuits(tmp_path: Path) -> None:
    """An empty rubric must NOT call the backend and must return mean=0."""

    stub = StubJudgeBackend(response="should-never-be-returned")
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge([], candidate_diff="diff")
    assert report.mean_score == 0.0
    assert report.per_item == []
    assert stub.calls == []  # no backend invocation


def test_fenced_json_response_parsed(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Models sometimes wrap JSON in ```json fences. The parser strips them."""

    fenced = "```json\n" + _perfect_response(["r1", "r2", "r3"]) + "\n```"
    stub = StubJudgeBackend(response=fenced)
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="diff")
    assert report.mean_score == 1.0


def test_out_of_range_score_clamped(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Score values not in `{0, 0.5, 1}` get nearest-clamped per the
    defensive coercion in `_coerce_to_valid_score`."""

    payload = {
        "items": [
            {"item_id": "r1", "score": 0.7, "justification": "x"},
            {"item_id": "r2", "score": 1.4, "justification": "y"},
            {"item_id": "r3", "score": -0.2, "justification": "z"},
        ],
        "tampering_detected": False,
    }
    stub = StubJudgeBackend(response=json.dumps(payload))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="diff")
    scores = {p.item_id: p.score for p in report.per_item}
    # 0.7 → 0.5; 1.4 → 1.0; -0.2 → 0.0.
    assert scores == {"r1": 0.5, "r2": 1.0, "r3": 0.0}


def test_coerce_to_valid_score_boundaries() -> None:
    """Direct test of the score-clamp helper."""

    assert _coerce_to_valid_score(0.0) == 0.0
    assert _coerce_to_valid_score(0.25) == 0.0
    assert _coerce_to_valid_score(0.26) == 0.5
    assert _coerce_to_valid_score(0.5) == 0.5
    assert _coerce_to_valid_score(0.74) == 0.5
    assert _coerce_to_valid_score(0.75) == 1.0
    assert _coerce_to_valid_score(1.0) == 1.0
    assert _coerce_to_valid_score(5.0) == 1.0
    assert _coerce_to_valid_score(-1.0) == 0.0


def test_per_item_score_to_dict_roundtrip() -> None:
    p = PerItemScore(item_id="x", score=0.5, justification="z")
    assert p.to_dict() == {"item_id": "x", "score": 0.5, "justification": "z"}


def test_report_to_dict_roundtrip(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """`RubricReport.to_dict()` is the exact shape the cache stores."""

    stub = StubJudgeBackend(response=_perfect_response(["r1", "r2", "r3"]))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    report = svc.judge(rubric_items, candidate_diff="diff")
    d = report.to_dict()
    assert set(d.keys()) == {"per_item", "mean_score", "tampering_detected", "rubric_sha", "judge_model"}
    assert len(d["per_item"]) == 3
    assert d["mean_score"] == 1.0
    assert d["judge_model"]  # not empty


def test_judge_model_override(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Explicit `judge_model` arg overrides the env-driven default."""

    stub = StubJudgeBackend(response=_perfect_response(["r1", "r2", "r3"]))
    svc = RubricJudgeService(
        backend=stub, cache_dir=tmp_path / "cache", judge_model="my-special-judge"
    )
    svc.judge(rubric_items, candidate_diff="diff")
    assert stub.calls[0]["model"] == "my-special-judge"


def test_temperature_default_is_zero(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Spec §5.3: judge runs at temperature 0."""

    stub = StubJudgeBackend(response=_perfect_response(["r1", "r2", "r3"]))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    svc.judge(rubric_items, candidate_diff="diff")
    assert stub.calls[0]["temperature"] == 0.0


def test_user_prompt_includes_all_inputs(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """The rendered user prompt must include the rubric items, diff, and summary."""

    stub = StubJudgeBackend(response=_perfect_response(["r1", "r2", "r3"]))
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    svc.judge(rubric_items, candidate_diff="MY_DIFF_TOKEN", submit_summary="MY_SUMMARY_TOKEN")
    user = stub.calls[0]["user_prompt"]
    assert "MY_DIFF_TOKEN" in user
    assert "MY_SUMMARY_TOKEN" in user
    assert "r1" in user and "r2" in user and "r3" in user
    assert "natural_language_assertion" in user


def test_non_json_response_raises(tmp_path: Path, rubric_items: list[dict[str, Any]]) -> None:
    """Non-JSON judge reply → `JudgeBackendError`."""

    from milo.judge.backends.base import JudgeBackendError

    stub = StubJudgeBackend(response="this is not json at all")
    svc = RubricJudgeService(backend=stub, cache_dir=tmp_path / "cache")
    with pytest.raises(JudgeBackendError):
        svc.judge(rubric_items, candidate_diff="diff")
