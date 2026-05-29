"""Milo jsonl -> SkyRL parquet preprocessor.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §1.1: generalizes the per-instance
``milo/spike/preprocess_one_milo.py`` to a *directory* of jsonl files, with
cohort filtering driven by the Phase 0.6 dataset audit. The output schema
matches what ``skyrl.train.dataset.PromptDataset`` reads (one parquet row per
training example) and is consumed by :class:`milo.lht_adapter.env.MiloLHTEnv`
via the ``extras["instance"]`` thread-through. See ``RL_GYM_SPEC.md`` v0.7 §6.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from milo.lht_adapter.image_naming import get_image_name


__all__ = [
    "build_parquet",
    "load_cohort",
    "DEFAULT_DATA_SOURCE",
    "DEFAULT_ENV_CLASS",
    "iter_milo_jsonls",
    "milo_to_row",
    "MiloRow",
    "DatasetStats",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — keep aligned with milo/spike/preprocess_one_milo.py
# ---------------------------------------------------------------------------

DEFAULT_DATA_SOURCE: str = "milo-bench"
DEFAULT_ENV_CLASS: str = "milo_lht"

# Default test command template — pytest for Python; the generator will swap
# this on Phase 11 per-language test runners (verifier owns that).
DEFAULT_TEST_CMD_TEMPLATE: str = "pytest -x --tb=short {f2p_tests}"

# Where the milo-bench repo image expects the checked-out source. Matches
# multiswebench images (see freya/benchmarks/multiswebench/build_images.py).
DEFAULT_WORKDIR: str = "/testbed"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MiloRow:
    """A single parquet row in the SkyRL ``PromptDataset`` shape.

    Mirrors the dict ``mini_swe_agent/preprocess_swegym.py`` writes plus an
    ``extra_info`` slot for the ``instance`` payload (the field the trainer
    threads into ``extras`` for the env constructor).
    """

    data_source: str
    prompt: List[Dict[str, str]]
    env_class: str
    reward_spec: Dict[str, Any]
    extra_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_source": self.data_source,
            "prompt": self.prompt,
            "env_class": self.env_class,
            "reward_spec": self.reward_spec,
            "extra_info": self.extra_info,
        }


@dataclass
class DatasetStats:
    """Per-build stats — surfaced to caller for auditing / logging."""

    seen_files: int = 0
    parsed_instances: int = 0
    accepted_rows: int = 0
    rejected_no_f2p: int = 0
    rejected_no_p2p: int = 0
    rejected_by_filter: int = 0
    rejected_parse_error: int = 0
    languages: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.languages is None:
            self.languages = {}

    def summary(self) -> str:
        return (
            f"Dataset build summary: "
            f"seen_files={self.seen_files} parsed={self.parsed_instances} "
            f"accepted={self.accepted_rows} "
            f"(no_f2p={self.rejected_no_f2p}, no_p2p={self.rejected_no_p2p}, "
            f"filter={self.rejected_by_filter}, parse_err={self.rejected_parse_error}) "
            f"langs={dict(sorted(self.languages.items()))}"
        )


# ---------------------------------------------------------------------------
# Cohort loader
# ---------------------------------------------------------------------------


def load_cohort(cohort_assignments_path: Path, allowed: List[str]) -> Set[str]:
    """Read the Phase 0.6 cohort assignment file and return the set of
    instance_ids in any of the ``allowed`` cohorts.

    The expected on-disk format (per ``IMPLEMENTATION_PLAN.md`` v0.4 §0.6.4)
    is a JSON object shaped:

        {
            "instance_id_1": {"cohort": "A", ...},
            "instance_id_2": {"cohort": "B", ...},
            "instance_id_3": {"cohort": "C", "drop_reason": "..."}
        }

    We also accept the simpler shape::

        {"A": ["id1", "id2"], "B": ["id3"], "C": ["id4"]}

    for forward compatibility with how the audit script might emit results.

    Args:
        cohort_assignments_path: path to ``dataset/cohort_assignments.json``.
        allowed: list of cohort labels to include (e.g. ``["A"]`` or
            ``["A", "B"]``).

    Returns:
        Set of instance_ids in any of the allowed cohorts. Empty set if the
        file doesn't exist (so callers can default-pass-all gracefully via the
        ``cohort_filter`` callable).
    """
    if not cohort_assignments_path.is_file():
        logger.warning(
            "load_cohort: %s not found; returning empty set (caller decides whether to fail)",
            cohort_assignments_path,
        )
        return set()

    with cohort_assignments_path.open() as f:
        raw = json.load(f)

    allowed_set = set(allowed)
    out: Set[str] = set()

    # Shape detection.
    if all(isinstance(v, dict) for v in raw.values()):
        # {instance_id: {cohort: ..., ...}}
        for iid, info in raw.items():
            cohort = info.get("cohort")
            if cohort in allowed_set:
                out.add(iid)
    elif all(isinstance(v, list) for v in raw.values()):
        # {cohort_label: [instance_id, ...]}
        for label, ids in raw.items():
            if label in allowed_set:
                out.update(ids)
    else:
        raise ValueError(
            f"Unrecognized cohort file shape at {cohort_assignments_path}: "
            f"expected dict-of-dicts or dict-of-lists, got mixed values."
        )

    logger.info(
        "load_cohort: loaded %d ids from %s (cohorts=%s)",
        len(out),
        cohort_assignments_path,
        sorted(allowed_set),
    )
    return out


# ---------------------------------------------------------------------------
# Row construction (generalization of preprocess_one_milo.to_minisweagent_row)
# ---------------------------------------------------------------------------


def _synthesize_problem_statement(milo: Dict[str, Any]) -> str:
    """Same prose as ``milo/spike/preprocess_one_milo.synthesize_problem_statement``
    — duplicated here to keep the dataset module dependency-free of the spike.
    """
    title = (milo.get("title") or "").strip()
    body = (milo.get("body") or "").strip()
    bundle = milo.get("prs_in_bundle") or []
    ps = f"# {title}\n\n{body}" if title else body
    if len(bundle) > 1:
        ps += (
            f"\n\n---\n\n*Note: this task corresponds to a bundle of "
            f"{len(bundle)} related PRs ({', '.join('#' + str(n) for n in bundle)}). "
            f"The fix may span multiple commits.*"
        )
    return ps


def _extract_f2p_test_ids(milo: Dict[str, Any]) -> List[str]:
    f2p = milo.get("f2p_tests") or {}
    if isinstance(f2p, dict):
        return sorted(f2p.keys())
    if isinstance(f2p, list):
        return sorted(f2p)
    return []


def _extract_p2p_test_ids(milo: Dict[str, Any]) -> List[str]:
    p2p = milo.get("p2p_tests") or {}
    if isinstance(p2p, dict):
        return sorted(p2p.keys())
    if isinstance(p2p, list):
        return sorted(p2p)
    return []


def _synthesize_eval_script(
    milo: Dict[str, Any],
    test_cmd_template: str,
    workdir: str,
) -> str:
    """Build the bash eval script consumed by the generator/verifier path.

    For Cohort A we expect ``f2p_tests`` to be populated (Phase 0.6 audit
    classified Cohort A as exactly this). For Cohort B (no F2P, needs Phase
    11.X verifier construction) callers must override the test command on a
    per-instance basis — the script we emit here is a placeholder that fails
    loudly so it's obvious downstream.
    """
    f2p_tests = _extract_f2p_test_ids(milo)
    if not f2p_tests:
        # Cohort B — emit a script that fails so the verifier returns
        # R_terminal=0 unambiguously rather than silently passing.
        return (
            "set -e\n"
            f"cd {workdir}\n"
            "echo 'milo: no F2P tests on this instance (Cohort B); "
            "verifier construction needed (Phase 11.X).' >&2\n"
            "exit 1\n"
        )

    f2p_arg = " ".join(repr(t) for t in f2p_tests)
    test_patch = milo.get("test_patch", "") or ""

    return (
        "set -e\n"
        f"cd '{workdir}'\n"
        "\n"
        "# Apply the canonical test_patch first (sets up the new failing test infrastructure).\n"
        "cat <<'MILO_TEST_PATCH_EOF' > /tmp/milo_test.patch\n"
        f"{test_patch}\n"
        "MILO_TEST_PATCH_EOF\n"
        "patch --batch --fuzz=5 -p1 -i /tmp/milo_test.patch "
        "|| git apply --reject /tmp/milo_test.patch || true\n"
        "\n"
        "# The model patch was already applied by the generator before this script.\n"
        "\n"
        "# F2P: every test below MUST pass.\n"
        f"{test_cmd_template.format(f2p_tests=f2p_arg)}\n"
    )


def milo_to_row(
    milo: Dict[str, Any],
    image_name: Optional[str] = None,
    test_cmd_template: str = DEFAULT_TEST_CMD_TEMPLATE,
    workdir: str = DEFAULT_WORKDIR,
    data_source: str = DEFAULT_DATA_SOURCE,
    env_class: str = DEFAULT_ENV_CLASS,
) -> MiloRow:
    """Convert a single milo-bench instance dict into a :class:`MiloRow`.

    ``image_name`` may be supplied explicitly (handy for tests / offline runs);
    if omitted, it's derived via :func:`milo.lht_adapter.image_naming.get_image_name`.
    """
    iid = milo.get("instance_id")
    if not iid:
        raise ValueError(f"milo instance missing instance_id: keys={list(milo)[:10]}")

    if image_name is None:
        image_name = get_image_name(milo)

    problem = _synthesize_problem_statement(milo)
    eval_script = _synthesize_eval_script(milo, test_cmd_template, workdir)

    f2p_ids = _extract_f2p_test_ids(milo)
    p2p_ids = _extract_p2p_test_ids(milo)

    # The ``instance`` payload the env / generator gets via extras.
    instance_payload: Dict[str, Any] = {
        # Fields mini_swe_agent + milo runtime consume directly:
        "instance_id": iid,
        "image_name": image_name,
        "problem_statement": problem,
        "eval_script": eval_script,
        # Milo-specific provenance (for invariants + logging):
        "milo_org": milo.get("org"),
        "milo_repo": milo.get("repo"),
        "milo_number": milo.get("number"),
        "milo_base_sha": (milo.get("base") or {}).get("sha"),
        "milo_lang": milo.get("lang"),
        "milo_prs_in_bundle": milo.get("prs_in_bundle"),
        "milo_f2p_test_ids": f2p_ids,
        "milo_p2p_test_ids": p2p_ids,
        # Verifier-facing payloads — kept here so the generator can hand them
        # straight to the Phase 2 verifier without re-reading the jsonl.
        "test_patch": milo.get("test_patch", ""),
        "fix_patch": milo.get("fix_patch", ""),
        "tag_start": milo.get("tag_start"),
        "tag_end": milo.get("tag_end"),
    }

    return MiloRow(
        data_source=data_source,
        prompt=[{"role": "user", "content": problem}],
        env_class=env_class,
        reward_spec={"method": "verifier", "ground_truth": None},
        extra_info={"instance": instance_payload},
    )


# ---------------------------------------------------------------------------
# Directory iteration
# ---------------------------------------------------------------------------


def iter_milo_jsonls(src: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Iterate over (path, parsed-dict) tuples for every milo jsonl under ``src``.

    Accepts:
      - a single ``.jsonl`` file (yields each line).
      - a directory; recursively finds every ``*.jsonl`` and yields each line.

    Lines that fail JSON parsing are logged and skipped (not raised — one
    corrupt line should not kill a 300-task build).
    """
    if src.is_file():
        files: List[Path] = [src]
    elif src.is_dir():
        files = sorted(src.rglob("*.jsonl"))
    else:
        raise FileNotFoundError(src)

    for p in files:
        try:
            text = p.read_text()
        except Exception as e:
            logger.warning("iter_milo_jsonls: cannot read %s: %s", p, e)
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("iter_milo_jsonls: %s:%d bad json: %s", p, lineno, e)
                continue
            yield p, obj


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def _default_filter(instance: Dict[str, Any]) -> bool:
    """The minimal Cohort A predicate: needs F2P+P2P populated (the v0.7
    hardened I-2 prerequisite — see RL_GYM_SPEC §7 row I-2).
    """
    return bool(_extract_f2p_test_ids(instance)) and bool(_extract_p2p_test_ids(instance))


def build_parquet(
    src_jsonl_dir: Path,
    cohort_filter: Callable[[Dict[str, Any]], bool],
    out_path: Path,
    *,
    test_cmd_template: str = DEFAULT_TEST_CMD_TEMPLATE,
    workdir: str = DEFAULT_WORKDIR,
    data_source: str = DEFAULT_DATA_SOURCE,
    env_class: str = DEFAULT_ENV_CLASS,
    image_name_override: Optional[str] = None,
    require_f2p: bool = True,
    require_p2p: bool = True,
) -> DatasetStats:
    """Produce the parquet file the SkyRL trainer expects.

    Generalizes ``milo/spike/preprocess_one_milo.py`` from one file to a whole
    directory; honors the cohort filter the caller passes (typically built
    from :func:`load_cohort`).

    Args:
        src_jsonl_dir: directory of milo-bench ``*.jsonl`` files (or a single
            jsonl). Lines that fail JSON parse are skipped with a log line.
        cohort_filter: predicate ``(instance_dict) -> bool``. Common values:

              ``lambda i: i["instance_id"] in cohort_a_ids``  # explicit set
              ``lambda _: True``                              # accept all
              :func:`_default_filter`                         # F2P+P2P populated

        out_path: destination ``.parquet`` path. Parent dirs auto-created.
        test_cmd_template: bash template; ``{f2p_tests}`` is substituted.
        workdir: in-container working dir written into ``eval_script``.
        data_source: ``data_source`` column value (default ``"milo-bench"``).
        env_class: ``env_class`` column value (default ``"milo_lht"`` — matches
            :class:`milo.lht_adapter.env.MiloLHTEnv` registration).
        image_name_override: skip canonical image-name derivation and use this
            literal value for every row. For offline tests.
        require_f2p / require_p2p: extra gate beyond ``cohort_filter`` —
            reject any instance missing those buckets. ON by default per the
            v0.7-hardened I-2 invariant.

    Returns:
        :class:`DatasetStats` summarizing the build.
    """
    src_jsonl_dir = Path(src_jsonl_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = DatasetStats()
    rows: List[Dict[str, Any]] = []

    for path, instance in iter_milo_jsonls(src_jsonl_dir):
        stats.seen_files = len({p for p, _ in [(path, instance)]} | {Path(p.parent) for p in [path]})  # rough
        # Track unique source files more cleanly:
        # (we just want a counter; not worth maintaining a set)
        stats.parsed_instances += 1

        lang = instance.get("lang", "unknown")
        stats.languages[lang] = stats.languages.get(lang, 0) + 1

        if require_f2p and not _extract_f2p_test_ids(instance):
            stats.rejected_no_f2p += 1
            continue
        if require_p2p and not _extract_p2p_test_ids(instance):
            stats.rejected_no_p2p += 1
            continue
        try:
            if not cohort_filter(instance):
                stats.rejected_by_filter += 1
                continue
        except Exception as e:
            logger.warning("cohort_filter raised on %s: %s", instance.get("instance_id"), e)
            stats.rejected_by_filter += 1
            continue

        try:
            row = milo_to_row(
                instance,
                image_name=image_name_override,
                test_cmd_template=test_cmd_template,
                workdir=workdir,
                data_source=data_source,
                env_class=env_class,
            )
        except Exception as e:
            logger.warning("milo_to_row failed on %s: %s", instance.get("instance_id"), e)
            stats.rejected_parse_error += 1
            continue

        rows.append(row.to_dict())
        stats.accepted_rows += 1

    # Fix the seen_files counter properly post-loop.
    if src_jsonl_dir.is_file():
        stats.seen_files = 1
    else:
        stats.seen_files = sum(1 for _ in src_jsonl_dir.rglob("*.jsonl"))

    if not rows:
        raise RuntimeError(
            f"build_parquet produced 0 rows from {src_jsonl_dir}. {stats.summary()}"
        )

    _write_parquet(rows, out_path)
    logger.info("build_parquet: wrote %d rows to %s. %s", len(rows), out_path, stats.summary())
    return stats


# ---------------------------------------------------------------------------
# Parquet writer — pandas preferred, fall back to pyarrow only.
# ---------------------------------------------------------------------------


def _write_parquet(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write rows to parquet. Uses pandas (a SkyRL transitive dep) with a
    pyarrow fallback so the function works in lighter test envs that pin
    only pyarrow.
    """
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        return
    except ImportError:
        pass

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:  # pragma: no cover - covered by pandas path in CI
        raise RuntimeError(
            "Neither pandas nor pyarrow is installed; cannot write parquet."
        ) from e

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_path))


# ---------------------------------------------------------------------------
# CLI for ad-hoc use (mirrors preprocess_one_milo.py)
# ---------------------------------------------------------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    """``python -m milo.lht_adapter.dataset --src ... --out ...``."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="dir or .jsonl file")
    p.add_argument("--out", type=Path, required=True, help="output .parquet path")
    p.add_argument(
        "--cohort-assignments",
        type=Path,
        default=None,
        help="optional Phase 0.6 cohort assignments JSON",
    )
    p.add_argument(
        "--cohorts",
        nargs="+",
        default=["A"],
        help="cohort labels to include when --cohort-assignments is set",
    )
    p.add_argument("--image-override", default=None)
    p.add_argument("--test-cmd", default=DEFAULT_TEST_CMD_TEMPLATE)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    args = p.parse_args(argv)

    if args.cohort_assignments:
        ids = load_cohort(args.cohort_assignments, args.cohorts)
        cohort_filter: Callable[[Dict[str, Any]], bool] = lambda inst: inst.get("instance_id") in ids
    else:
        cohort_filter = _default_filter

    stats = build_parquet(
        src_jsonl_dir=args.src,
        cohort_filter=cohort_filter,
        out_path=args.out,
        test_cmd_template=args.test_cmd,
        workdir=args.workdir,
        image_name_override=args.image_override,
    )
    print(stats.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_cli(sys.argv[1:]))
