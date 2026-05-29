"""Tests for milo.algos.reference_cache."""

from __future__ import annotations

from pathlib import Path

from milo.algos.reference_cache import (
    CacheEntry,
    ReferenceLogprobsCache,
    _hash_prompt,
    build_cache,
)


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = ReferenceLogprobsCache(tmp_path / "ref.jsonl")
    entry = CacheEntry(
        prompt_hash=_hash_prompt("hello"),
        tokens=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        model_revision="abc123",
        tokenizer_sha="def456",
    )
    cache.put(entry)
    got = cache.get("hello")
    assert got is not None
    assert got.tokens == [1, 2, 3]
    assert got.logprobs == [-0.1, -0.2, -0.3]


def test_persistent_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "ref.jsonl"
    c1 = ReferenceLogprobsCache(path)
    c1.put(CacheEntry(_hash_prompt("p1"), [1], [-1.0], "rev", "tok"))
    c2 = ReferenceLogprobsCache(path)
    got = c2.get("p1")
    assert got is not None
    assert got.tokens == [1]


def test_build_cache_writes_each_unique_prompt(tmp_path: Path) -> None:
    def fake_compute(_: Path, text: str) -> tuple[list[int], list[float]]:
        return ([len(text)], [-float(len(text))])

    n = build_cache(
        sft_checkpoint=Path("/dev/null"),
        prompts=["a", "ab", "a"],            # duplicate "a" should dedup
        out_cache_path=tmp_path / "cache.jsonl",
        compute_logprobs_fn=fake_compute,
        model_revision="rev1",
        tokenizer_sha="tok1",
    )
    assert n == 2
    c = ReferenceLogprobsCache(tmp_path / "cache.jsonl")
    assert len(c) == 2
    assert c.get("a").tokens == [1]
    assert c.get("ab").tokens == [2]
