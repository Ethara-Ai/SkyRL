"""5-task smoke pack — plan §9.2 + spec §10.2.

Spec §10.2 mandates a 5-task smoke pack across Python / TypeScript / Go
(the three simplest container builds), runnable in < 10 minutes,
exercising install + container build + verifier + judge + at least one
policy adapter end-to-end. Plan §9.2 fixes the specific composition:

    | Task                          | Lang       | Tier        |
    |-------------------------------|------------|-------------|
    | python_print_fix              | Python     | trivial     |
    | python_off_by_one             | Python     | medium      |
    | ts_null_check                 | TypeScript | medium      |
    | go_error_handling             | Go         | hard        |
    | unresolvable_lorem            | Python     | unsolvable  |

The composition matches the prompt mandate (1 trivial / 2 medium / 1 hard
/ 1 unsolvable across 3 languages) and exercises the lower bound of the
spec §8 difficulty bins.

The manifest itself lives at ``milo/smoke/manifest.yaml`` (per the prompt
constraint that the manifest is under ``milo/smoke/``, not ``milo/replay/``).
This module loads and runs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

# Path to the manifest. We resolve relative to this file so the package
# works whether installed or run in-place.
MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "smoke" / "manifest.yaml"
)

# Spec §10.2 wall-clock SLA — informational, surfaced in the report.
SMOKE_PACK_WALL_CLOCK_BUDGET_SECONDS = 10 * 60

SmokeTier = Literal["trivial", "medium", "hard", "unsolvable"]
SmokeLang = Literal["python", "typescript", "go"]


@dataclass(slots=True)
class SmokeTask:
    """One row of the smoke pack manifest.

    Fields mirror the YAML structure 1:1 — see ``milo/smoke/manifest.yaml``.
    """

    instance_id: str
    lang: SmokeLang
    expected_tier: SmokeTier
    description: str = ""


@dataclass(slots=True)
class SmokeResult:
    """Per-task result from a smoke-pack run."""

    instance_id: str
    expected_tier: SmokeTier
    passed: bool
    actual_decomposition: dict[str, Any] | None
    duration_seconds: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Manifest loading — pure-Python YAML so we don't take a hard PyYAML dep
# ---------------------------------------------------------------------------

def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser — enough for our smoke manifest.

    The smoke manifest only uses:
      * top-level scalar keys (string values)
      * top-level lists of dicts (the ``tasks:`` block)
      * inside each list item, scalar key: value pairs (string + bool/int)

    We *could* depend on PyYAML, but ``milo`` aims for a minimal dep
    surface in the smoke-pack runner so it can boot inside the trainer
    container without ``pip install pyyaml``. The full feature surface
    is unnecessary — this parser handles the ~12 lines of our manifest
    and raises on anything it can't.

    When PyYAML *is* present (e.g. on a dev box with the extra), we
    prefer it. This keeps round-trip fidelity with hand-edited YAML.
    """
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("smoke manifest top level must be a mapping")
        return loaded
    except ImportError:
        pass

    out: dict[str, Any] = {}
    current_list_key: str | None = None
    current_item: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        # Strip trailing whitespace + skip comments / blank.
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            # top-level key
            if current_list_key is not None and current_item is not None:
                out.setdefault(current_list_key, []).append(current_item)
                current_item = None
            current_list_key = None
            if ":" not in line:
                raise ValueError(f"unrecognized line: {raw_line!r}")
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                # opens a list block; we infer list vs nested-dict at first
                # indented line.
                current_list_key = key
                out.setdefault(key, [])
            else:
                out[key] = _coerce(value)
            continue
        # indented line
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"list item without parent key: {raw_line!r}")
            if current_item is not None:
                out.setdefault(current_list_key, []).append(current_item)
            current_item = {}
            after = stripped[2:].strip()
            if after:
                if ":" not in after:
                    raise ValueError(f"unrecognized inline item: {raw_line!r}")
                k, _, v = after.partition(":")
                current_item[k.strip()] = _coerce(v.strip())
        else:
            if current_item is None or ":" not in stripped:
                raise ValueError(f"unrecognized indented line: {raw_line!r}")
            k, _, v = stripped.partition(":")
            current_item[k.strip()] = _coerce(v.strip())
    if current_list_key is not None and current_item is not None:
        out.setdefault(current_list_key, []).append(current_item)
    return out


def _coerce(value: str) -> Any:
    """Map a YAML scalar literal to a Python scalar."""
    if value.startswith(('"', "'")) and value.endswith(('"', "'")) and len(value) >= 2:
        return value[1:-1]
    lower = value.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    if lower in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_manifest(path: Path | None = None) -> list[SmokeTask]:
    """Load the smoke-pack manifest from disk and return its task rows.

    Raises ``FileNotFoundError`` if the manifest is missing (the file is
    checked into the repo at ``milo/smoke/manifest.yaml``; missing it
    means the install is broken).
    """
    p = path or MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"smoke pack manifest not found at {p}. "
            "Expected at milo/smoke/manifest.yaml per plan §9.2."
        )
    parsed = _parse_simple_yaml(p.read_text(encoding="utf-8"))
    tasks_raw = parsed.get("tasks", [])
    if not isinstance(tasks_raw, list):
        raise ValueError("smoke manifest 'tasks' must be a list")
    if len(tasks_raw) != 5:
        raise ValueError(
            f"smoke manifest must list exactly 5 tasks per spec §10.2; "
            f"got {len(tasks_raw)}"
        )
    out: list[SmokeTask] = []
    for row in tasks_raw:
        if not isinstance(row, dict):
            raise ValueError(f"smoke manifest task must be a mapping; got {row!r}")
        out.append(
            SmokeTask(
                instance_id=str(row["instance_id"]),
                lang=row["lang"],
                expected_tier=row["expected_tier"],
                description=str(row.get("description", "")),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SmokePack:
    """5-task smoke pack driver per plan §9.2.

    The default ``run()`` re-uses :func:`milo.replay.cli.replay_trace`:
    for every task in the manifest, we look up the on-disk trace and
    confirm the stored reward_decomposition round-trips. This catches:

    * Missing on-disk traces (install broken, S3 sync incomplete).
    * Trace-schema drift between gym and trainer (spec §10.4 property).
    * Reward-aggregator regressions (spec §10.4 CI requirement).

    The "real" smoke pack also runs the full end-to-end rollout against a
    live gym container and a live policy adapter — that path is owned by
    Phase 10 (``milo/scripts/run_smoke.sh``) and is exercised by
    ``make gym-smoke``. This Python class is the cheap-to-test driver
    that CI runs on every PR.
    """

    def __init__(
        self,
        manifest_path: Path | None = None,
        *,
        trace_root: Path | None = None,
        wall_clock_budget_seconds: int = SMOKE_PACK_WALL_CLOCK_BUDGET_SECONDS,
    ) -> None:
        self.manifest_path = manifest_path or MANIFEST_PATH
        self.trace_root = trace_root
        self.wall_clock_budget_seconds = wall_clock_budget_seconds
        self._tasks: list[SmokeTask] | None = None

    @property
    def tasks(self) -> list[SmokeTask]:
        if self._tasks is None:
            self._tasks = load_manifest(self.manifest_path)
        return self._tasks

    def expected_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "trivial": 0, "medium": 0, "hard": 0, "unsolvable": 0,
        }
        for t in self.tasks:
            counts[t.expected_tier] = counts.get(t.expected_tier, 0) + 1
        return counts

    def expected_language_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tasks:
            counts[t.lang] = counts.get(t.lang, 0) + 1
        return counts

    def validate_composition(self) -> list[str]:
        """Return a list of human-readable warnings about the composition.

        Empty list = composition matches the prompt's mandate (1 trivial,
        2 medium, 1 hard, 1 unsolvable across 3 languages). We return
        warnings rather than raising so CI surfaces the smoke-pack issue
        independently from a "smoke pack failed to run" failure.
        """
        warnings: list[str] = []
        d = self.expected_distribution()
        if d.get("trivial", 0) != 1:
            warnings.append(
                f"expected 1 trivial task, got {d.get('trivial', 0)}"
            )
        if d.get("medium", 0) != 2:
            warnings.append(
                f"expected 2 medium tasks, got {d.get('medium', 0)}"
            )
        if d.get("hard", 0) != 1:
            warnings.append(
                f"expected 1 hard task, got {d.get('hard', 0)}"
            )
        if d.get("unsolvable", 0) != 1:
            warnings.append(
                f"expected 1 unsolvable task, got {d.get('unsolvable', 0)}"
            )
        langs = self.expected_language_distribution()
        if len(langs) != 3:
            warnings.append(
                f"expected exactly 3 languages, got {len(langs)}: "
                f"{sorted(langs.keys())}"
            )
        return warnings

    def run(
        self,
        *,
        skip_missing_traces: bool = True,
    ) -> list[SmokeResult]:
        """Replay every task in the manifest and collect outcomes.

        ``skip_missing_traces``: when True (default), tasks whose on-disk
        trace doesn't exist yet produce a SmokeResult with
        ``passed=False`` and ``error='trace_missing'`` rather than raising.
        This lets CI surface coverage gaps without blocking unrelated PRs.
        """
        from milo.replay.cli import replay_trace  # local to dodge import cycle

        out: list[SmokeResult] = []
        for task in self.tasks:
            t0 = datetime.now(timezone.utc).timestamp()
            try:
                result = replay_trace(
                    task.instance_id,
                    trace_root=self.trace_root,
                    mode="default",
                )
                dt = datetime.now(timezone.utc).timestamp() - t0
                out.append(SmokeResult(
                    instance_id=task.instance_id,
                    expected_tier=task.expected_tier,
                    passed=result.matched,
                    actual_decomposition=result.replayed_decomposition,
                    duration_seconds=dt,
                    error=None if result.matched else "decomposition_mismatch",
                ))
            except FileNotFoundError:
                dt = datetime.now(timezone.utc).timestamp() - t0
                if skip_missing_traces:
                    out.append(SmokeResult(
                        instance_id=task.instance_id,
                        expected_tier=task.expected_tier,
                        passed=False,
                        actual_decomposition=None,
                        duration_seconds=dt,
                        error="trace_missing",
                    ))
                else:
                    raise
            except Exception as e:  # noqa: BLE001 — smoke pack reports, never raises
                dt = datetime.now(timezone.utc).timestamp() - t0
                out.append(SmokeResult(
                    instance_id=task.instance_id,
                    expected_tier=task.expected_tier,
                    passed=False,
                    actual_decomposition=None,
                    duration_seconds=dt,
                    error=f"{type(e).__name__}: {e}",
                ))
        return out

    def write_report(
        self,
        results: Iterable[SmokeResult],
        output_path: Path,
    ) -> None:
        """Persist a ``smoke_report.json`` consumable by the dashboards."""
        results_list = list(results)
        payload = {
            "schema_version": "milo-smoke/1.0",
            "ran_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_tasks": len(results_list),
            "n_passed": sum(1 for r in results_list if r.passed),
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "expected_distribution": self.expected_distribution(),
            "expected_language_distribution": self.expected_language_distribution(),
            "composition_warnings": self.validate_composition(),
            "results": [
                {
                    "instance_id": r.instance_id,
                    "expected_tier": r.expected_tier,
                    "passed": r.passed,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in results_list
            ],
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
