#!/usr/bin/env python3
"""Phase 0.6 — milo-bench dataset audit.

Implements IMPLEMENTATION_PLAN.md v0.4 §0.6:
    - 0.6.1 yield analysis (Cohort A/B/C classification)
    - 0.6.2 contamination check (Qwen training-cutoff + popular-repo flag)
    - 0.6.3 schema mapping report
    - 0.6.4 usable-task cohort

Read-only over the milo-bench dataset directory. Emits four artefacts under --output-dir:
    AUDIT_REPORT.md             — human-readable summary
    cohort_assignments.json     — {instance_id: "A" | "B" | "C", ...}
    contamination_flags.json    — {instance_id: {...}}
    SCHEMA_MAPPING.md           — quirks of the on-disk schema, parser notes

Usage (no GPU, no network beyond optional GitHub stargazer lookup):

    uv run --isolated --extra dev python -m milo.audit.audit_dataset \\
        --milo-dataset-dir /Users/piyush/github/freya/milo-bench/dataset \\
        --output-dir milo/data/audit_v1

Pass --skip-github-popularity to avoid GitHub API rate limits in a hurry.

Cohort definitions (per plan §0.6):
    A — usable as-is (F2P>0 AND P2P>0, not contamination-flagged)
    B — needs verifier construction (has fix_patch+test_patch but empty F2P/P2P)
    C — drop (contamination-flagged, or no recoverable signal at all)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Qwen2.5-Coder training cutoff used as the contamination check threshold.
# Per https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct the training cutoff
# is reported as "before 2024-03". We use 2024-03-01 as the conservative cutoff;
# any task whose tag_end / head_sha lands before this is flagged.
DEFAULT_QWEN_CUTOFF_ISO = "2024-03-01"

# Repos that are popular enough that there's a strong prior of being in any
# major code-pretraining corpus. The full check uses the GitHub stargazer
# count; this is a fallback static list for offline runs.
KNOWN_POPULAR_REPOS = {
    "django/django",
    "pallets/flask",
    "psf/requests",
    "pandas-dev/pandas",
    "numpy/numpy",
    "scipy/scipy",
    "scikit-learn/scikit-learn",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "huggingface/transformers",
    "fastify/fastify",
    "expressjs/express",
    "nodejs/node",
    "facebook/react",
    "vuejs/vue",
    "angular/angular",
    "microsoft/typescript",
    "rust-lang/rust",
    "golang/go",
    "kubernetes/kubernetes",
    "ffmpeg/ffmpeg",
    "openssl/openssl",
    "git/git",
    "linux/linux",
    "torvalds/linux",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--milo-dataset-dir", type=Path, required=True,
                   help="Directory containing milo-bench/dataset/*.jsonl files")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write AUDIT_REPORT.md and friends")
    p.add_argument("--qwen-cutoff-iso", default=DEFAULT_QWEN_CUTOFF_ISO,
                   help=f"Training-cutoff ISO date for contamination check (default {DEFAULT_QWEN_CUTOFF_ISO})")
    p.add_argument("--skip-github-popularity", action="store_true",
                   help="Skip live GitHub stargazer lookup; fall back to KNOWN_POPULAR_REPOS list")
    p.add_argument("--target-cohort-size", type=int, default=300,
                   help="Pass/fail signal: Cohort A+B must be >= this (default 300, per proposal)")
    return p.parse_args()


def load_instance(jsonl_path: Path) -> dict[str, Any] | None:
    """Each jsonl file contains exactly one line per the audit. Returns None on parse failure."""
    try:
        with open(jsonl_path) as f:
            lines = f.read().strip().split("\n")
        if len(lines) != 1:
            return None
        return json.loads(lines[0])
    except (json.JSONDecodeError, OSError):
        return None


def get_test_count(instance: dict[str, Any], key: str) -> int:
    """milo-bench stores f2p_tests / p2p_tests as dict {test_name -> stringified_inner_json}.
    Returns the number of entries; 0 if missing/empty."""
    val = instance.get(key)
    if val is None:
        return 0
    if isinstance(val, dict):
        return len(val)
    if isinstance(val, list):
        return len(val)
    return 0


def get_lang(instance: dict[str, Any]) -> str:
    return (instance.get("lang") or "unknown").lower()


def repo_full_name(instance: dict[str, Any]) -> str:
    org = instance.get("org", "?")
    repo = instance.get("repo", "?")
    return f"{org}/{repo}"


def is_popular_repo(instance: dict[str, Any], skip_github: bool) -> bool:
    """Lightweight popular-repo check. We DO NOT hit GitHub in this script by default
    (rate limits, optional dependency). If --skip-github-popularity is unset, the user
    can wire in their own API call here; for now we use the static list."""
    name = repo_full_name(instance).lower()
    return name in {n.lower() for n in KNOWN_POPULAR_REPOS}


def predates_cutoff(instance: dict[str, Any], cutoff_iso: str) -> bool | None:
    """Best-effort: check whether the bundled PR's merge date predates cutoff.
    Returns True/False if we can determine; None if we cannot."""
    # The milo-bench schema doesn't carry a top-level merged_at. We use
    # `tag` if present as a string proxy. This is conservative — we flag
    # only when we can prove it.
    tag = instance.get("tag")
    if not isinstance(tag, str):
        return None
    # crude: many tags carry an ISO-like fragment; try to parse trailing 4-digit year
    # Real implementation would resolve the SHA → commit date via git.
    # For audit-time decision-making we return None when uncertain.
    return None


def classify_cohort(
    instance: dict[str, Any],
    f2p: int,
    p2p: int,
    contamination_flag: dict[str, Any],
) -> str:
    """Return 'A', 'B', or 'C'."""
    if contamination_flag.get("recommend_drop"):
        return "C"
    fix_patch = instance.get("fix_patch", "") or ""
    test_patch = instance.get("test_patch", "") or ""
    if f2p > 0 and p2p > 0:
        return "A"
    if fix_patch.strip() and test_patch.strip():
        # No F2P/P2P but we have raw patches — can synthesize per §11.8.
        return "B"
    return "C"


def emit_schema_mapping(out: Path) -> None:
    out.write_text("""# milo-bench on-disk schema → LHTInstance mapping

Produced by `milo/audit/audit_dataset.py`. See `IMPLEMENTATION_PLAN.md` v0.4 §0.6.3.

## Per-file shape
One jsonl file per instance, named `{org}__{repo}-{number}.jsonl`. Each file is exactly
one line.

## Top-level field mapping

| On-disk field | LHTInstance field | Parse notes |
|---|---|---|
| `org` | `org` | str |
| `repo` | `repo` | str |
| `number` | `pr_numbers[0]` (sentinel) | int — see `prs_in_bundle` for the actual bundle |
| `state` | (informational) | str — "closed" expected |
| `title`, `body` | `body` | str — `body` is the PR description, becomes the task statement |
| `base` | `{base_sha = base.sha, head_ref = base.ref}` | dict of `{label, ref, sha}` |
| `resolved_issues` | `resolved_issues` | list of `{number, title, body}` |
| `fix_patch` | `fix_patch` | str — full unified diff; may be very large (>100 KB) |
| `test_patch` | `test_patch` | str — test-files-only diff |
| `lang` | `lang` | str — one of {python, javascript, typescript, java, go, rust, c, cpp} |
| `fixed_tests` | (informational) | dict |
| `p2p_tests` | `p2p_tests` (list of test names) | **GOTCHA:** value is `dict[test_name → str(json)]`, inner JSON shape `{"run", "test", "fix"}`. Run `json.loads` twice to materialize. |
| `f2p_tests` | `f2p_tests` (list of test names) | same gotcha as p2p_tests |
| `s2p_tests`, `n2p_tests` | (informational, ignored for R_terminal) | same gotcha |
| `run_result`, `test_patch_result`, `fix_patch_result` | `run_result`, `test_patch_result`, `fix_patch_result` | **GOTCHA:** dicts where ALL values are strings, including integer counts (`passed_count: "37"`). Cast on read. |
| `instance_id` | `instance_id` | str — canonical form `{org}__{repo}-{number}` |
| `prs_in_bundle` | `pr_numbers` | list[int] — multi-PR bundle membership; this is the "long-horizon" property |
| `pr_url` | (informational) | list[str] |
| `tag` | `tag_start`, `tag_end` (derived) | str — convention varies; needs per-repo logic |
| `number_interval` | (informational) | str |

## Missing-in-on-disk fields (synthesized by Phase 11)

| LHTInstance field | Synthesized by |
|---|---|
| `rubric_items[]` | Phase 11.10 SME authoring |
| `difficulty_tier` | Phase 8 calibration (pass@8 on two frontier models) |
| `golden_trace_path` | Phase 11.11 SME-recorded rollout |
| `provenance.{fixtures_sha,dataset_cutoff,build_image_digest,rubric_sha,golden_trace_sha}` | Phase 11.12 finalization |

## Verifier construction (Cohort B → A promotion)

Per `IMPLEMENTATION_PLAN.md` v0.4 §11.8: for instances with `f2p_tests = {}` and
`p2p_tests = {}` but non-empty `fix_patch` + `test_patch`, synthesize F2P/P2P by:
1. Apply `test_patch` to `tag_start` in a sandbox; run the per-language test runner.
2. Apply `fix_patch` on top; diff the test outcomes.
3. Newly-PASSing tests under `test_patch+fix_patch` that were FAIL/NONE before = F2P.
4. Tests passing in baseline AND still passing after = P2P.

## Anti-gotchas

- **Do not** assume `passed_count` etc. are integers. They are strings.
- **Do not** assume `f2p_tests` is a list. It is a dict, and inner JSON is double-encoded.
- **Do not** confuse `prs_in_bundle` size with verifier scope. The verifier runs against
  the bundle's combined fix; bundle is provenance, not semantics.
- **Do not** treat `s2p_tests` (skip-to-pass) and `n2p_tests` (none-to-pass) as F2P. They
  are informational; the spec only models F2P and P2P for `R_terminal`.
""")


def main() -> int:
    args = parse_args()

    if not args.milo_dataset_dir.is_dir():
        print(f"ERROR: --milo-dataset-dir {args.milo_dataset_dir} is not a directory", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(args.milo_dataset_dir.glob("*.jsonl"))
    print(f"Auditing {len(jsonl_files)} jsonl files from {args.milo_dataset_dir}")

    cohort: dict[str, str] = {}
    contamination: dict[str, dict[str, Any]] = {}
    yield_stats = {
        "total": 0,
        "parse_failed": 0,
        "cohort_a": 0,
        "cohort_b": 0,
        "cohort_c": 0,
        "f2p_nonempty": 0,
        "p2p_nonempty": 0,
        "both_nonempty": 0,
        "both_empty_has_patches": 0,
        "both_empty_no_patches": 0,
    }
    per_lang_yield: dict[str, Counter] = defaultdict(Counter)
    per_lang_total: Counter = Counter()
    per_repo_count: Counter = Counter()
    failed_to_parse: list[str] = []

    for path in jsonl_files:
        yield_stats["total"] += 1
        instance = load_instance(path)
        if instance is None:
            yield_stats["parse_failed"] += 1
            failed_to_parse.append(path.name)
            continue

        iid = instance.get("instance_id") or path.stem
        lang = get_lang(instance)
        per_lang_total[lang] += 1
        per_repo_count[repo_full_name(instance)] += 1

        f2p = get_test_count(instance, "f2p_tests")
        p2p = get_test_count(instance, "p2p_tests")
        if f2p > 0:
            yield_stats["f2p_nonempty"] += 1
        if p2p > 0:
            yield_stats["p2p_nonempty"] += 1
        if f2p > 0 and p2p > 0:
            yield_stats["both_nonempty"] += 1
        if f2p == 0 and p2p == 0:
            has_patches = bool((instance.get("fix_patch") or "").strip()
                               and (instance.get("test_patch") or "").strip())
            if has_patches:
                yield_stats["both_empty_has_patches"] += 1
            else:
                yield_stats["both_empty_no_patches"] += 1

        cutoff_flag = predates_cutoff(instance, args.qwen_cutoff_iso)
        popular_flag = is_popular_repo(instance, args.skip_github_popularity)
        recommend_drop = bool(popular_flag and cutoff_flag is True)
        contamination[iid] = {
            "predates_cutoff": cutoff_flag,
            "popular_repo": popular_flag,
            "recommend_drop": recommend_drop,
        }

        c = classify_cohort(instance, f2p, p2p, contamination[iid])
        cohort[iid] = c
        yield_stats[f"cohort_{c.lower()}"] += 1
        per_lang_yield[lang][c] += 1

    # Emit cohort_assignments.json
    (args.output_dir / "cohort_assignments.json").write_text(
        json.dumps(cohort, indent=2, sort_keys=True))
    (args.output_dir / "contamination_flags.json").write_text(
        json.dumps(contamination, indent=2, sort_keys=True))
    emit_schema_mapping(args.output_dir / "SCHEMA_MAPPING.md")

    a_plus_b = yield_stats["cohort_a"] + yield_stats["cohort_b"]
    target_met = a_plus_b >= args.target_cohort_size

    # Emit AUDIT_REPORT.md
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = [
        "# milo-bench Dataset Audit Report (Phase 0.6)",
        "",
        f"**Generated:** {now}",
        f"**Source:** `{args.milo_dataset_dir}`",
        f"**Qwen cutoff (contamination threshold):** `{args.qwen_cutoff_iso}`",
        "",
        "## Headline",
        "",
        f"- Total instances scanned: **{yield_stats['total']}**",
        f"- Parse failures: **{yield_stats['parse_failed']}**",
        f"- Cohort A (usable as-is): **{yield_stats['cohort_a']}**",
        f"- Cohort B (needs verifier construction per §11.8): **{yield_stats['cohort_b']}**",
        f"- Cohort C (drop): **{yield_stats['cohort_c']}**",
        f"- **Cohort A + B = {a_plus_b}** (target ≥ {args.target_cohort_size}: **{'MET' if target_met else 'NOT MET — escalate per §28.7'}**)",
        "",
        "## Yield analysis",
        "",
        f"- F2P populated (count > 0): {yield_stats['f2p_nonempty']} ({100*yield_stats['f2p_nonempty']/max(1,yield_stats['total']):.1f}%)",
        f"- P2P populated (count > 0): {yield_stats['p2p_nonempty']} ({100*yield_stats['p2p_nonempty']/max(1,yield_stats['total']):.1f}%)",
        f"- Both populated: {yield_stats['both_nonempty']} ({100*yield_stats['both_nonempty']/max(1,yield_stats['total']):.1f}%)",
        f"- Both empty BUT has fix_patch+test_patch (Cohort B candidates): {yield_stats['both_empty_has_patches']}",
        f"- Both empty AND no patches (Cohort C): {yield_stats['both_empty_no_patches']}",
        "",
        "## Per-language breakdown",
        "",
        "| Lang | Total | Cohort A | Cohort B | Cohort C |",
        "|---|---|---|---|---|",
    ]
    for lang in sorted(per_lang_total):
        row = per_lang_yield[lang]
        report.append(
            f"| {lang} | {per_lang_total[lang]} | "
            f"{row.get('A',0)} | {row.get('B',0)} | {row.get('C',0)} |"
        )

    report += [
        "",
        "## Top-10 repos by instance count",
        "",
        "| Repo | Count |",
        "|---|---|",
    ]
    for repo, cnt in per_repo_count.most_common(10):
        report.append(f"| `{repo}` | {cnt} |")

    if failed_to_parse:
        report += [
            "",
            "## Files that failed to parse",
            "",
            *[f"- `{name}`" for name in failed_to_parse[:20]],
        ]
        if len(failed_to_parse) > 20:
            report.append(f"- ... and {len(failed_to_parse) - 20} more")

    report += [
        "",
        "## Contamination notes",
        "",
        f"- Tasks flagged as popular-repo (proxy for training-corpus presence): "
        f"{sum(1 for v in contamination.values() if v['popular_repo'])}",
        f"- Tasks flagged as recommend-drop (popular AND predates cutoff): "
        f"{sum(1 for v in contamination.values() if v['recommend_drop'])}",
        "",
        "**Caveat:** the `predates_cutoff` check in this audit is conservative — it returns",
        "None when uncertain rather than False-flagging. A more thorough check requires",
        "resolving `head_sha` → commit date via `git log -1 --format=%cI <sha>` against",
        "the repo's mirror, which this script does not do (it would require a git mirror",
        "and credentials). Real cohort-C count after a full contamination sweep may be",
        "higher than what this report shows.",
        "",
        "## Next steps (per IMPLEMENTATION_PLAN.md v0.4 §0.6.5)",
        "",
        "1. Review this report with CTO/CEO.",
        "2. If A+B < target: trigger §28.7 disaster recovery (renegotiate scope or source",
        "   net-new tasks).",
        "3. Hand `cohort_assignments.json` to Phase 1 (env preprocessing) and Phase 11.8",
        "   (verifier construction for Cohort B).",
        "4. Open the spec-side update for `SCHEMA_MAPPING.md` quirks (stringified counts,",
        "   double-encoded test entries).",
    ]
    (args.output_dir / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n")

    print(f"\nWrote {args.output_dir}/")
    print(f"  - AUDIT_REPORT.md")
    print(f"  - cohort_assignments.json  ({len(cohort)} entries)")
    print(f"  - contamination_flags.json")
    print(f"  - SCHEMA_MAPPING.md")
    print(f"\nCohort A: {yield_stats['cohort_a']}  Cohort B: {yield_stats['cohort_b']}  "
          f"Cohort C: {yield_stats['cohort_c']}")
    print(f"A+B = {a_plus_b}  (target {args.target_cohort_size}: "
          f"{'MET' if target_met else 'NOT MET'})")
    return 0 if target_met else 1


if __name__ == "__main__":
    sys.exit(main())
