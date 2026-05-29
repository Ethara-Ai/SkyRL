"""Offline reference-logprobs cache — spec v0.7 §18 / plan v0.4 §14 + §19.1.5.

v0.7 retired the 4-H100 live reference vLLM server (it was idle ~70% of the
time, costing ~$20K per RL run for no benefit). Reference logprobs are now
computed *once* after SFT and cached to S3 (or local disk for the spike).
The trainer reads from the cache for prompt-token sequences hit during RL,
and falls back to an on-demand `compute_logprobs` call against the policy
server in `logprobs_only` mode (with SFT weights momentarily loaded) for
novel tokens.

This module owns:
    * `build_cache(...)` — one-shot post-SFT batch job.
    * `ReferenceLogprobsCache` — lookup interface used by the GRPO loss
      wrapper in `milo/algos/grpo_wrapper.py`.

Cache file format (JSON line per prompt, gzip-compressed when large):
    {
      "prompt_hash": "<sha256 of the canonical prompt+token-prefix>",
      "tokens": [int, ...],
      "logprobs": [float, ...],
      "model_revision": "<HF revision sha>",
      "tokenizer_sha": "<sha256 of tokenizer.json>",
      "schema": "milo-ref-cache/1.0"
    }
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger("milo.algos.reference_cache")


CACHE_SCHEMA = "milo-ref-cache/1.0"


@dataclass
class CacheEntry:
    prompt_hash: str
    tokens: list[int]
    logprobs: list[float]
    model_revision: str
    tokenizer_sha: str


def _hash_prompt(prompt_text: str, token_prefix: Sequence[int] | None = None) -> str:
    h = hashlib.sha256()
    h.update(prompt_text.encode("utf-8"))
    if token_prefix:
        h.update(b"|")
        h.update(json.dumps(list(token_prefix), separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


class ReferenceLogprobsCache:
    """Append-only read-mostly cache of (prompt, tokens) → logprobs.

    Backed by a single JSONL file (gzip when `.gz` suffix). In-process: an
    LRU dict keyed by `prompt_hash` so reads are O(1) after first warm-up.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._memory: dict[str, CacheEntry] = {}
        if self.path.exists():
            self._load()

    # ---------------------------------------------------------------- reads

    def get(
        self, prompt_text: str, token_prefix: Sequence[int] | None = None
    ) -> CacheEntry | None:
        key = _hash_prompt(prompt_text, token_prefix)
        return self._memory.get(key)

    def __contains__(self, prompt_hash: str) -> bool:
        return prompt_hash in self._memory

    # ---------------------------------------------------------------- writes

    def put(self, entry: CacheEntry) -> None:
        self._memory[entry.prompt_hash] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        opener: Callable[..., Any] = gzip.open if self.path.suffix == ".gz" else open
        mode = "at"
        with opener(self.path, mode, encoding="utf-8") as f:
            f.write(json.dumps({
                "schema": CACHE_SCHEMA,
                "prompt_hash": entry.prompt_hash,
                "tokens": entry.tokens,
                "logprobs": entry.logprobs,
                "model_revision": entry.model_revision,
                "tokenizer_sha": entry.tokenizer_sha,
            }) + "\n")

    # ---------------------------------------------------------------- internals

    def _load(self) -> None:
        opener: Callable[..., Any] = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("schema") != CACHE_SCHEMA:
                    logger.warning("skipping cache line with schema=%r", obj.get("schema"))
                    continue
                entry = CacheEntry(
                    prompt_hash=obj["prompt_hash"],
                    tokens=obj["tokens"],
                    logprobs=obj["logprobs"],
                    model_revision=obj["model_revision"],
                    tokenizer_sha=obj["tokenizer_sha"],
                )
                self._memory[entry.prompt_hash] = entry

    def __len__(self) -> int:
        return len(self._memory)


# ---------------------------------------------------------------------- builder


def build_cache(
    sft_checkpoint: Path,
    prompts: Iterable[str],
    out_cache_path: Path,
    compute_logprobs_fn: Callable[[Path, str], tuple[list[int], list[float]]],
    model_revision: str = "",
    tokenizer_sha: str = "",
) -> int:
    """One-shot batch job: compute logprobs for every prompt against the SFT
    checkpoint and write the cache.

    `compute_logprobs_fn(sft_checkpoint, prompt_text) -> (tokens, logprobs)`
    is injected so we don't hard-depend on a particular inference backend
    here (production uses vLLM in `logprobs_only` mode; tests inject a stub).
    Returns the number of entries written.
    """
    cache = ReferenceLogprobsCache(out_cache_path)
    count = 0
    for prompt_text in prompts:
        key = _hash_prompt(prompt_text)
        if key in cache:
            continue
        tokens, logprobs = compute_logprobs_fn(sft_checkpoint, prompt_text)
        cache.put(CacheEntry(
            prompt_hash=key,
            tokens=list(tokens),
            logprobs=list(logprobs),
            model_revision=model_revision,
            tokenizer_sha=tokenizer_sha,
        ))
        count += 1
    return count
