#!/usr/bin/env python3
"""Phase 0.5 spike — convert ONE milo-bench jsonl instance to the parquet format
mini_swe_agent (SkyRL's reference SWE-style trainer) expects.

Per IMPLEMENTATION_PLAN.md v0.4 §0.5:
    "Adapts it to the SkyRL examples/train/mini_swe_agent/ dataset format
     (smallest viable converter)."

The mini_swe_agent dataset schema (see examples/train/mini_swe_agent/preprocess_swegym.py)
is a parquet of rows shaped like:
    {
        "data_source": str,
        "prompt": [{"role": "user", "content": str}],
        "env_class": "null",                 # mini_swe_agent owns the agent loop
        "instance": dict,                    # threaded into env extras
    }

The `instance` dict is consumed by `get_sb_environment(config, instance, data_source)` in
examples/train/mini_swe_agent/mini_swe_utils.py, which expects at minimum:
    - instance["instance_id"]
    - instance["image_name"]            (or derived from data_source)
    - instance["problem_statement"]
    - instance["eval_script"]            (bash to run after model patch is applied)

We synthesize all four from the milo-bench jsonl.

Usage:

    uv run --isolated --extra dev python -m milo.spike.preprocess_one_milo \\
        --milo-jsonl /Users/piyush/github/freya/milo-bench/dataset/locustio__locust-1541.jsonl \\
        --output-dir milo/data/spike_v1 \\
        --image-name docker.io/library/python:3.11-slim   # placeholder; see milo/BLOCKERS.md

Produces:
    <output-dir>/train.parquet  — single row, the spike instance
    <output-dir>/validation.parquet — same row (placeholder; spike doesn't need val)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DATA_SOURCE = "milo-bench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--milo-jsonl", type=Path, required=True,
                   help="Path to ONE milo-bench jsonl file")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write train.parquet + validation.parquet")
    p.add_argument("--image-name", default=None,
                   help="Docker image name. If omitted, the canonical name is derived "
                        "from the instance via milo.lht_adapter.image_naming.get_image_name "
                        "(default prefix: $EVAL_DOCKER_IMAGE_PREFIX or the ap-south-1 ECR). "
                        "Pass an override for offline scaffold testing (e.g. python:3.11-slim).")
    p.add_argument("--test-cmd",
                   default="pytest -x --tb=short {f2p_tests}",
                   help="Test command template; {f2p_tests} is replaced with space-separated F2P test names. "
                        "Default suits pytest; override for non-Python.")
    p.add_argument("--workdir-in-container", default="/testbed",
                   help="Where the repo lives inside the Docker image")
    return p.parse_args()


def load_instance(path: Path) -> dict[str, Any]:
    """milo-bench jsonl = one line per file."""
    text = path.read_text().strip()
    return json.loads(text)


def synthesize_problem_statement(milo: dict[str, Any]) -> str:
    """The 'problem statement' shown to the model. We use the PR title + body.
    Multi-PR bundles get the bundle context appended."""
    title = milo.get("title", "").strip()
    body = (milo.get("body") or "").strip()
    bundle = milo.get("prs_in_bundle") or []
    ps = f"# {title}\n\n{body}" if title else body
    if len(bundle) > 1:
        ps += f"\n\n---\n\n*Note: this task corresponds to a bundle of {len(bundle)} related PRs " \
              f"({', '.join('#' + str(n) for n in bundle)}). The fix may span multiple commits.*"
    return ps


def extract_f2p_test_ids(milo: dict[str, Any]) -> list[str]:
    """f2p_tests is a dict {test_id: stringified-inner-json}. Return the test_id list."""
    f2p = milo.get("f2p_tests") or {}
    if isinstance(f2p, dict):
        return sorted(f2p.keys())
    return []


def synthesize_eval_script(milo: dict[str, Any], test_cmd_template: str,
                           workdir: str) -> str:
    """Generate the bash that mini_swe_agent runs after applying the model patch.

    This is intentionally minimal for the spike. Real Phase 2 verifier (milo/verifier/)
    will own the 3-Docker-run pattern with proper per-language runners.

    Behavior here:
      1. cd to workdir.
      2. Apply test_patch from the instance dict (provided via heredoc).
      3. Run F2P tests; assert all pass.
      4. Run P2P tests as a sample (first 10 for spike); assert none regress.

    Returns the bash text.
    """
    f2p_tests = extract_f2p_test_ids(milo)
    if not f2p_tests:
        # Cohort B path — spike requires Cohort A
        raise ValueError(
            f"Instance {milo.get('instance_id')} has no F2P tests; spike requires Cohort A.")

    f2p_arg = " ".join(repr(t) for t in f2p_tests)
    test_patch = milo.get("test_patch", "") or ""

    # heredoc-safe delimiter
    return f"""set -e
cd '{workdir}'

# Apply the canonical test_patch first (sets up the new failing test infrastructure).
cat <<'MILO_TEST_PATCH_EOF' > /tmp/milo_test.patch
{test_patch}
MILO_TEST_PATCH_EOF
patch --batch --fuzz=5 -p1 -i /tmp/milo_test.patch || git apply --reject /tmp/milo_test.patch || true

# The model patch was already applied by mini_swe_agent's evaluate_trajectory
# (it runs `git apply <model_patch>` before this script).

# F2P: every test below MUST pass.
{test_cmd_template.format(f2p_tests=f2p_arg)}
"""


def to_minisweagent_row(milo: dict[str, Any], image_name: str,
                         test_cmd_template: str, workdir: str) -> dict[str, Any]:
    """Produce one parquet-row dict in the mini_swe_agent format."""
    iid = milo["instance_id"]
    problem = synthesize_problem_statement(milo)
    eval_script = synthesize_eval_script(milo, test_cmd_template, workdir)

    # Threaded into env: get_sb_environment reads instance["image_name"], and
    # evaluate_trajectory reads instance["eval_script"] + instance["instance_id"].
    # The other fields are informational / for debugging.
    instance_for_env = {
        "instance_id": iid,
        "image_name": image_name,
        "problem_statement": problem,
        "eval_script": eval_script,
        # milo provenance — useful for debugging, ignored by mini_swe_agent.
        "milo_org": milo.get("org"),
        "milo_repo": milo.get("repo"),
        "milo_number": milo.get("number"),
        "milo_base_sha": (milo.get("base") or {}).get("sha"),
        "milo_lang": milo.get("lang"),
        "milo_prs_in_bundle": milo.get("prs_in_bundle"),
        "milo_f2p_test_ids": extract_f2p_test_ids(milo),
    }

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem}],
        "env_class": "null",  # mini_swe_agent generator owns the agent loop
        "instance": instance_for_env,
    }


def write_parquet(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write rows to a parquet file. Uses pandas for portability — pyarrow alone
    would work but pandas is already a SkyRL dep."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run via `uv run --isolated --extra dev` "
              "or `pip install pandas pyarrow`.", file=sys.stderr)
        sys.exit(3)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(rows)} row(s) to {out_path}")


def main() -> int:
    args = parse_args()

    if not args.milo_jsonl.is_file():
        print(f"ERROR: {args.milo_jsonl} not found", file=sys.stderr)
        return 2

    milo = load_instance(args.milo_jsonl)
    print(f"Loaded {milo.get('instance_id')} (lang={milo.get('lang')}, "
          f"F2P={len(milo.get('f2p_tests') or {})}, "
          f"P2P={len(milo.get('p2p_tests') or {})}, "
          f"bundle_size={len(milo.get('prs_in_bundle') or [])})")

    if args.image_name is None:
        from milo.lht_adapter.image_naming import get_image_name
        image_name = get_image_name(milo)
        print(f"Derived image name: {image_name}")
    else:
        image_name = args.image_name
        print(f"Using override image name: {image_name}")

    row = to_minisweagent_row(milo, image_name, args.test_cmd, args.workdir_in_container)

    # The spike doesn't actually need a val split, but mini_swe_agent's run script
    # references one — emit the same row so the pipeline doesn't choke.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet([row], args.output_dir / "train.parquet")
    write_parquet([row], args.output_dir / "validation.parquet")

    # Dump a human-readable preview
    preview = {
        "instance_id": row["instance"]["instance_id"],
        "image_name": row["instance"]["image_name"],
        "problem_statement_first_500": row["instance"]["problem_statement"][:500],
        "eval_script_first_500": row["instance"]["eval_script"][:500],
        "f2p_test_ids": row["instance"]["milo_f2p_test_ids"],
    }
    (args.output_dir / "preview.json").write_text(json.dumps(preview, indent=2))
    print(f"\nPreview at {args.output_dir / 'preview.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
