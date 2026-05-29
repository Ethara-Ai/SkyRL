"""`RubricJudgeService` — the stateful wrapper around a `JudgeBackend`.

Implements RL_GYM_SPEC.md v0.7 §4.4.3 (rubric reward), §5.3 (stateless HTTP
service shape), §6.7 (rubric schema), §I-6 (test-fixture-tampering
hard-floor to 0), and IMPLEMENTATION_PLAN.md v0.4 Phase 3.3. Caches calls
in LMDB keyed by sha256(rubric_sha + diff + summary + model + prompt_sha)
so identical patches across rollouts don't pay the judge cost twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from milo.judge.backends.base import JudgeBackend, JudgeBackendError
from milo.judge.prompt import PROMPT_SHA256, load_prompt

logger = logging.getLogger(__name__)

# Valid per-item scores per spec §4.4.3 — `{0, 0.5, 1}` exactly.
_VALID_SCORES: frozenset[float] = frozenset({0.0, 0.5, 1.0})


def _default_judge_model() -> str:
    """Resolve the default judge model id from the env (spec §5.3)."""

    return os.environ.get("MILO_JUDGE_MODEL", "claude-opus-4-6")


def _default_judge_temperature() -> float:
    """Spec §5.3 mandates temperature 0 for reproducibility."""

    return 0.0


# =====================================================================
# Result dataclasses
# =====================================================================
@dataclass(slots=True)
class PerItemScore:
    """One judged rubric item."""

    item_id: str
    score: float  # one of {0, 0.5, 1}
    justification: str

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "score": float(self.score), "justification": self.justification}


@dataclass(slots=True)
class RubricReport:
    """Aggregate judge output. Fed into the reward aggregator (§4.4)."""

    per_item: list[PerItemScore] = field(default_factory=list)
    mean_score: float = 0.0
    tampering_detected: bool = False
    rubric_sha: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_item": [p.to_dict() for p in self.per_item],
            "mean_score": float(self.mean_score),
            "tampering_detected": bool(self.tampering_detected),
            "rubric_sha": self.rubric_sha,
            "judge_model": self.judge_model,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# =====================================================================
# KV cache abstractions
# =====================================================================
class _KVCache:
    """Minimal `get/put` interface — `LMDBCache` or `SQLiteCache` implement it."""

    def get(self, key: str) -> bytes | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def put(self, key: str, value: bytes) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class _LMDBCache(_KVCache):
    """LMDB-backed cache. Lazy-imports `lmdb` so the dependency is only
    required when the cache is actually instantiated.
    """

    def __init__(self, path: Path, *, map_size_gb: int = 8) -> None:
        try:
            import lmdb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "lmdb is required for LMDBCache; install via `pip install lmdb`"
            ) from exc
        path.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(path),
            map_size=map_size_gb * 1024 * 1024 * 1024,
            subdir=True,
            create=True,
            readonly=False,
            lock=True,
            max_dbs=1,
        )

    def get(self, key: str) -> bytes | None:
        with self._env.begin(write=False) as txn:
            return txn.get(key.encode("utf-8"))

    def put(self, key: str, value: bytes) -> None:
        with self._env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), value)

    def close(self) -> None:
        self._env.close()


class _SQLiteCache(_KVCache):
    """Pure-stdlib fallback cache (no lmdb installed).

    The cache contract is "byte-in, byte-out, durable across process
    restarts" — both LMDB and SQLite satisfy that. We auto-fall-back to
    SQLite when lmdb is not importable so unit tests run cleanly on a
    Mac without the binary wheel.
    """

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        db_path = path / "judge_cache.sqlite3"
        # `check_same_thread=False`: we serialise writes with a lock below.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT v FROM kv WHERE k = ?", (key,)
            ).fetchone()
        return None if row is None else bytes(row[0])

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _open_cache(cache_dir: Path) -> _KVCache:
    """Open LMDB if available, else fall back to SQLite. The on-disk
    layout differs between the two, but both honour the same `get/put`
    contract so the service is agnostic.
    """

    try:
        import lmdb  # noqa: F401 - probing for availability
    except ImportError:
        logger.info("lmdb not installed; falling back to SQLite-backed judge cache at %s", cache_dir)
        return _SQLiteCache(cache_dir)
    return _LMDBCache(cache_dir)


# =====================================================================
# Service
# =====================================================================
class RubricJudgeService:
    """Stateless-call-wise rubric judge with persistent KV cache.

    Parameters
    ----------
    backend:
        A `JudgeBackend` implementation (Bedrock / Anthropic / stub).
    cache_dir:
        Directory for the LMDB env. Created on first use. Cache key:
        `sha256(rubric_sha + candidate_diff + (submit_summary or "") + judge_model + prompt_sha)`.
    judge_model:
        Override the env-driven default; useful for ablation runs that
        want to pin a specific model id without setting an env var.
    temperature:
        Overrideable for ablations. Default 0.0 (spec §5.3).
    """

    def __init__(
        self,
        backend: JudgeBackend,
        cache_dir: Path,
        *,
        judge_model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._backend = backend
        self._judge_model = judge_model or _default_judge_model()
        self._temperature = (
            _default_judge_temperature() if temperature is None else float(temperature)
        )
        _, self._prompt_sha = load_prompt()
        # Sanity check the prompt-loader cache. Defensive: prevents a
        # stale module import shadowing PROMPT_SHA256.
        assert self._prompt_sha == PROMPT_SHA256, "PROMPT_SHA256 module constant out of sync with loaded prompt"
        self._cache = _open_cache(cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def judge(
        self,
        rubric_items: list[dict[str, Any]],
        candidate_diff: str,
        submit_summary: str | None = None,
    ) -> RubricReport:
        """Score `rubric_items` against `candidate_diff`.

        Steps:
          1. Compute rubric_sha and the composite cache key.
          2. Cache lookup; on hit, decode and return.
          3. On miss, render prompt, call backend, parse strict JSON.
          4. Apply tampering hard-floor per spec §4.4.3 / §I-6.
          5. Persist to cache; return.
        """

        if not rubric_items:
            # Mean of an empty rubric is defined as 0.0 (cf. reward decomposition).
            # We do NOT charge the judge for an empty list.
            return RubricReport(
                per_item=[],
                mean_score=0.0,
                tampering_detected=False,
                rubric_sha=self._compute_rubric_sha([]),
                judge_model=self._judge_model,
            )

        rubric_sha = self._compute_rubric_sha(rubric_items)
        cache_key = self._compute_cache_key(
            rubric_sha=rubric_sha,
            candidate_diff=candidate_diff,
            submit_summary=submit_summary,
        )

        cached = self._cache.get(cache_key)
        if cached is not None:
            try:
                payload = json.loads(cached.decode("utf-8"))
                return self._report_from_dict(payload, rubric_sha=rubric_sha)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Corrupted cache entry — fall through to a fresh call
                # and let the put() overwrite the broken bytes.
                logger.warning("Corrupted judge cache entry for key %s; recomputing", cache_key)

        prompt_system, _ = load_prompt()
        prompt_user = self._render_user_prompt(rubric_items, candidate_diff, submit_summary)
        try:
            raw_text = self._backend.call(
                system_prompt=prompt_system,
                user_prompt=prompt_user,
                model=self._judge_model,
                temperature=self._temperature,
            )
        except JudgeBackendError:
            raise

        parsed = self._parse_judge_json(raw_text)
        report = self._build_report(parsed, rubric_items=rubric_items, rubric_sha=rubric_sha)

        # Store the *canonical* dict, not the raw model output, so cache
        # hits return the post-processed (tampering-hard-floored) report.
        self._cache.put(cache_key, json.dumps(report.to_dict(), sort_keys=False).encode("utf-8"))
        return report

    def close(self) -> None:
        """Release the cache handles. Idempotent."""

        try:
            self._cache.close()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Error closing judge cache", exc_info=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_rubric_sha(self, rubric_items: list[dict[str, Any]]) -> str:
        """Stable, order-sensitive SHA-256 over the rubric items.

        We canonicalise to JSON with sorted keys per item so a reordering
        of optional fields doesn't blow the cache. The *order of items*
        is part of the input, however, because the judge prompt is
        order-sensitive (item N citing the same hunk as item N-1).
        """

        canonical = json.dumps(rubric_items, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _compute_cache_key(
        self,
        *,
        rubric_sha: str,
        candidate_diff: str,
        submit_summary: str | None,
    ) -> str:
        """Composite cache key per spec §5.3.

        Includes:
          * rubric_sha (item set + order)
          * candidate_diff (exact bytes)
          * submit_summary (treat None and empty string as distinct)
          * judge_model_id (so swapping models doesn't read stale entries)
          * prompt_sha (so bumping the system prompt invalidates everything)
        """

        h = hashlib.sha256()
        h.update(b"v1\n")  # cache namespace; bump if key format changes.
        h.update(rubric_sha.encode("utf-8"))
        h.update(b"\n")
        h.update(candidate_diff.encode("utf-8"))
        h.update(b"\n")
        # Distinguish None from "":
        if submit_summary is None:
            h.update(b"<<NONE>>")
        else:
            h.update(submit_summary.encode("utf-8"))
        h.update(b"\n")
        h.update(self._judge_model.encode("utf-8"))
        h.update(b"\n")
        h.update(self._prompt_sha.encode("utf-8"))
        return h.hexdigest()

    def _render_user_prompt(
        self,
        rubric_items: list[dict[str, Any]],
        candidate_diff: str,
        submit_summary: str | None,
    ) -> str:
        """Pack the per-call inputs into a single user-turn string.

        Strict JSON for the rubric items (the model must echo `item_id`
        back to us); plain text for the diff and summary because diffs
        are not valid JSON strings without escape gymnastics and the
        judge prompt explicitly asks the model to cite specific lines.
        """

        # Project to the minimal subset the prompt actually needs.
        minimal_items = [
            {
                "item_id": str(it.get("item_id", "")),
                "natural_language_assertion": str(it.get("natural_language_assertion", "")),
            }
            for it in rubric_items
        ]
        items_json = json.dumps(minimal_items, indent=2, ensure_ascii=False)

        summary_block = (
            f"<submit_summary>\n{submit_summary}\n</submit_summary>\n\n"
            if submit_summary
            else "<submit_summary>(none provided)</submit_summary>\n\n"
        )

        return (
            "Score the following candidate patch against the rubric items. "
            "Respond with strict JSON per the format in the system prompt.\n\n"
            f"<rubric_items>\n{items_json}\n</rubric_items>\n\n"
            f"{summary_block}"
            f"<candidate_diff>\n{candidate_diff}\n</candidate_diff>\n"
        )

    def _parse_judge_json(self, raw_text: str) -> dict[str, Any]:
        """Lenient JSON parser for the judge's reply.

        The system prompt instructs the model to return strict JSON with
        no fences. In practice, models sometimes wrap the JSON in ```json
        ... ``` fences anyway. We strip code fences before parsing.
        """

        stripped = raw_text.strip()
        # Remove ```json ... ``` or ``` ... ``` fences if present.
        fence_re = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
        m = fence_re.match(stripped)
        if m:
            stripped = m.group(1).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise JudgeBackendError(
                f"Judge returned non-JSON: {raw_text[:500]!r} ({exc})"
            ) from exc

    def _build_report(
        self,
        parsed: dict[str, Any],
        *,
        rubric_items: list[dict[str, Any]],
        rubric_sha: str,
    ) -> RubricReport:
        """Validate + assemble the `RubricReport` from a parsed judge reply.

        Enforces:
          * `score in {0, 0.5, 1}` — coerce 0/1 ints to floats; reject
            anything else.
          * Every rubric input item gets a score (model omissions become 0
            with a placeholder justification — defensive, so the reward
            aggregator never sees a partial rubric).
          * Tampering hard-floor: if `tampering_detected`, every item
            score becomes 0 and justification becomes
            `"test-fixture-tampering"`. Per spec §4.4.3 / §I-6.
        """

        tampering_detected = bool(parsed.get("tampering_detected", False))
        items_in = parsed.get("items", [])
        if not isinstance(items_in, list):
            raise JudgeBackendError(f"Judge `items` field is not a list: {items_in!r}")

        # Build a lookup so we can fill in missing items.
        by_id: dict[str, dict[str, Any]] = {}
        for it in items_in:
            if not isinstance(it, dict):
                continue
            iid = str(it.get("item_id", ""))
            if iid:
                by_id[iid] = it

        per_item: list[PerItemScore] = []
        for input_item in rubric_items:
            iid = str(input_item.get("item_id", ""))
            judged = by_id.get(iid)
            if judged is None:
                # Model didn't score this item — treat as 0 with a flag.
                per_item.append(
                    PerItemScore(
                        item_id=iid,
                        score=0.0,
                        justification="judge-omitted-item",
                    )
                )
                continue

            raw_score = judged.get("score", 0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            if score not in _VALID_SCORES:
                # Clamp into the valid set without raising — the prompt
                # explicitly forbids fractional / out-of-range scores
                # but we don't want to take down a rollout because the
                # judge model briefly misbehaved.
                score = _coerce_to_valid_score(score)

            justification = str(judged.get("justification", "")).strip()
            per_item.append(PerItemScore(item_id=iid, score=score, justification=justification))

        if tampering_detected:
            # Hard-floor to zero per spec §4.4.3 / §I-6.
            per_item = [
                PerItemScore(item_id=p.item_id, score=0.0, justification="test-fixture-tampering")
                for p in per_item
            ]
            mean_score = 0.0
        else:
            mean_score = (
                sum(p.score for p in per_item) / len(per_item) if per_item else 0.0
            )

        return RubricReport(
            per_item=per_item,
            mean_score=mean_score,
            tampering_detected=tampering_detected,
            rubric_sha=rubric_sha,
            judge_model=self._judge_model,
        )

    def _report_from_dict(self, data: dict[str, Any], *, rubric_sha: str) -> RubricReport:
        """Inverse of `RubricReport.to_dict()`, used for cache hydration.

        We trust cached entries (we wrote them ourselves) and rebuild
        directly without re-validating individual scores.
        """

        per_item = [
            PerItemScore(
                item_id=str(it.get("item_id", "")),
                score=float(it.get("score", 0.0)),
                justification=str(it.get("justification", "")),
            )
            for it in data.get("per_item", [])
        ]
        return RubricReport(
            per_item=per_item,
            mean_score=float(data.get("mean_score", 0.0)),
            tampering_detected=bool(data.get("tampering_detected", False)),
            rubric_sha=data.get("rubric_sha") or rubric_sha,
            judge_model=str(data.get("judge_model", self._judge_model)),
        )


def _coerce_to_valid_score(raw: float) -> float:
    """Clamp a judge-emitted score into `{0, 0.5, 1}` using nearest-value.

    Bias toward the safer (lower) score on the boundary so a model that
    emits 0.74 lands at 0.5 rather than 1.0. Per spec §4.4.3 the valid
    set is exactly three values; this is a defensive clamp, not an
    interpretation of "the model meant something between".
    """

    if raw <= 0.25:
        return 0.0
    if raw < 0.75:
        return 0.5
    if raw <= 1.0:
        return 1.0
    # Strictly greater than 1 — out of range, clamp to 1.
    return 1.0


# Convenience for callers who don't want to thread a backend object
# through their config. Exposes both raw classes and a default factory.
def build_default_service(cache_dir: Path) -> RubricJudgeService:
    """Build a service with the Bedrock backend at default settings.

    Used by the gym runtime at boot; callers that want something else
    (Anthropic direct, ensemble, stub) construct `RubricJudgeService`
    directly. Importing inside the function avoids the boto3 import at
    module-load time.
    """

    from milo.judge.backends.bedrock import BedrockJudgeBackend  # local import
    return RubricJudgeService(backend=BedrockJudgeBackend(), cache_dir=cache_dir)


__all__ = [
    "PerItemScore",
    "RubricJudgeService",
    "RubricReport",
    "build_default_service",
]
