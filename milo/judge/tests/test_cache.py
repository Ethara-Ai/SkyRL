"""Tests for `RubricJudgeService` caching.

Asserts:
  * identical inputs → cache hit (zero additional backend calls).
  * changed rubric items → cache miss (one more backend call).
  * changed diff → cache miss.
  * changed summary → cache miss.
  * changed judge model → cache miss.
  * different `submit_summary=None` vs `""` produce distinct keys.
  * cached entries survive `RubricJudgeService.close()` and a new instance.

The tests use the auto-fallback SQLite cache (LMDB is optional) so they
run without the `lmdb` wheel installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from milo.judge.service import RubricJudgeService


@dataclass
class CountingStubBackend:
    """Stub that returns a perfect-score JSON and counts every call."""

    item_ids: list[str] = field(default_factory=list)
    n_calls: int = 0

    def call(self, *, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
        self.n_calls += 1
        payload = {
            "items": [
                {"item_id": iid, "score": 1, "justification": f"j-{iid}"}
                for iid in self.item_ids
            ],
            "tampering_detected": False,
        }
        return json.dumps(payload)


@pytest.fixture
def rubric() -> list[dict[str, Any]]:
    return [
        {"item_id": "a", "natural_language_assertion": "alpha"},
        {"item_id": "b", "natural_language_assertion": "beta"},
    ]


def test_identical_input_is_cache_hit(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    """Second judge() with identical args must not call the backend."""

    backend = CountingStubBackend(item_ids=["a", "b"])
    svc = RubricJudgeService(backend=backend, cache_dir=tmp_path / "cache")
    r1 = svc.judge(rubric, candidate_diff="DIFF", submit_summary="SUM")
    r2 = svc.judge(rubric, candidate_diff="DIFF", submit_summary="SUM")
    assert backend.n_calls == 1
    assert r1.to_dict() == r2.to_dict()


def test_changed_rubric_is_cache_miss(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    """Mutating any rubric item changes `rubric_sha` → miss."""

    backend = CountingStubBackend(item_ids=["a", "b"])
    svc = RubricJudgeService(backend=backend, cache_dir=tmp_path / "cache")
    svc.judge(rubric, candidate_diff="DIFF")
    # Same diff, different assertion text:
    rubric2 = [
        {"item_id": "a", "natural_language_assertion": "ALPHA (modified)"},
        {"item_id": "b", "natural_language_assertion": "beta"},
    ]
    svc.judge(rubric2, candidate_diff="DIFF")
    assert backend.n_calls == 2


def test_changed_diff_is_cache_miss(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    backend = CountingStubBackend(item_ids=["a", "b"])
    svc = RubricJudgeService(backend=backend, cache_dir=tmp_path / "cache")
    svc.judge(rubric, candidate_diff="DIFF_1")
    svc.judge(rubric, candidate_diff="DIFF_2")
    assert backend.n_calls == 2


def test_changed_summary_is_cache_miss(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    backend = CountingStubBackend(item_ids=["a", "b"])
    svc = RubricJudgeService(backend=backend, cache_dir=tmp_path / "cache")
    svc.judge(rubric, candidate_diff="DIFF", submit_summary="s1")
    svc.judge(rubric, candidate_diff="DIFF", submit_summary="s2")
    assert backend.n_calls == 2


def test_none_summary_distinct_from_empty(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    """`None` and `""` are conceptually different inputs; cache keys differ."""

    backend = CountingStubBackend(item_ids=["a", "b"])
    svc = RubricJudgeService(backend=backend, cache_dir=tmp_path / "cache")
    svc.judge(rubric, candidate_diff="DIFF", submit_summary=None)
    svc.judge(rubric, candidate_diff="DIFF", submit_summary="")
    assert backend.n_calls == 2


def test_changed_model_is_cache_miss(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    backend = CountingStubBackend(item_ids=["a", "b"])
    svc1 = RubricJudgeService(
        backend=backend, cache_dir=tmp_path / "cache", judge_model="model-A"
    )
    svc1.judge(rubric, candidate_diff="DIFF")
    svc2 = RubricJudgeService(
        backend=backend, cache_dir=tmp_path / "cache", judge_model="model-B"
    )
    svc2.judge(rubric, candidate_diff="DIFF")
    assert backend.n_calls == 2


def test_cache_persists_across_instances(tmp_path: Path, rubric: list[dict[str, Any]]) -> None:
    """A fresh `RubricJudgeService` on the same cache_dir reuses entries."""

    backend = CountingStubBackend(item_ids=["a", "b"])
    cache_dir = tmp_path / "cache"
    svc1 = RubricJudgeService(backend=backend, cache_dir=cache_dir, judge_model="m")
    svc1.judge(rubric, candidate_diff="DIFF")
    svc1.close()

    svc2 = RubricJudgeService(backend=backend, cache_dir=cache_dir, judge_model="m")
    svc2.judge(rubric, candidate_diff="DIFF")
    svc2.close()
    # Backend should still only have been called once across both services.
    assert backend.n_calls == 1


def test_rubric_item_reorder_is_cache_miss(tmp_path: Path) -> None:
    """Reordering items changes the rubric hash (order-sensitive)."""

    backend = CountingStubBackend(item_ids=["a", "b"])
    svc = RubricJudgeService(backend=backend, cache_dir=tmp_path / "cache")
    r1 = [
        {"item_id": "a", "natural_language_assertion": "alpha"},
        {"item_id": "b", "natural_language_assertion": "beta"},
    ]
    r2 = [
        {"item_id": "b", "natural_language_assertion": "beta"},
        {"item_id": "a", "natural_language_assertion": "alpha"},
    ]
    svc.judge(r1, candidate_diff="DIFF")
    svc.judge(r2, candidate_diff="DIFF")
    assert backend.n_calls == 2
