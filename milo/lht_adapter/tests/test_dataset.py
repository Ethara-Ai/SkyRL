"""Tests for milo.lht_adapter.dataset.

Validates the milo jsonl -> SkyRL parquet preprocessor against synthetic
fixtures. Per ``IMPLEMENTATION_PLAN.md`` v0.4 §1.1: round-trip one instance
through the parquet path and back, assert all fields preserved including the
stringified-inner-JSON parsing from §0.6.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from milo.lht_adapter.dataset import (
    DEFAULT_DATA_SOURCE,
    DEFAULT_ENV_CLASS,
    DatasetStats,
    MiloRow,
    _default_filter,
    build_parquet,
    iter_milo_jsonls,
    load_cohort,
    milo_to_row,
)


# ---------------------------------------------------------------------------
# Synthetic milo-bench instances
# ---------------------------------------------------------------------------


def _synthetic_milo(instance_id: str, lang: str = "python", with_f2p: bool = True,
                    with_p2p: bool = True, bundle_size: int = 1) -> Dict[str, Any]:
    return {
        "instance_id": instance_id,
        "org": instance_id.split("__")[0],
        "repo": instance_id.split("__")[1].split("-")[0],
        "number": int(instance_id.rsplit("-", 1)[-1]),
        "lang": lang,
        "title": f"Fix bug in {instance_id}",
        "body": "Body text describing the issue.\nLine 2.",
        "prs_in_bundle": list(range(1, bundle_size + 1)),
        "f2p_tests": (
            {"tests/test_foo.py::test_a": "{}", "tests/test_foo.py::test_b": "{}"}
            if with_f2p else {}
        ),
        "p2p_tests": (
            {"tests/test_existing.py::test_c": "{}"} if with_p2p else {}
        ),
        "test_patch": "diff --git a/tests/test_foo.py b/tests/test_foo.py\n+def test_a(): pass\n",
        "fix_patch": "diff --git a/src/foo.py b/src/foo.py\n+x = 1\n",
        "base": {"sha": "deadbeef"},
        "tag_start": "v1.0.0",
        "tag_end": "v1.0.1",
    }


@pytest.fixture
def synth_jsonl_dir(tmp_path: Path) -> Path:
    """Lay down three synthetic milo jsonl files in a temp dir."""
    d = tmp_path / "milo_jsonls"
    d.mkdir()
    instances = [
        _synthetic_milo("locustio__locust-1541"),
        _synthetic_milo("foo__bar-42", lang="go", bundle_size=3),
        _synthetic_milo("baz__qux-7", lang="rust", with_f2p=False),  # Cohort B
    ]
    for inst in instances:
        (d / f"{inst['instance_id']}.jsonl").write_text(json.dumps(inst) + "\n")
    return d


@pytest.fixture
def cohort_file(tmp_path: Path) -> Path:
    """Phase 0.6 cohort_assignments.json fixture."""
    path = tmp_path / "cohort_assignments.json"
    payload = {
        "locustio__locust-1541": {"cohort": "A"},
        "foo__bar-42": {"cohort": "A"},
        "baz__qux-7": {"cohort": "B"},
        "drop__me-99": {"cohort": "C", "drop_reason": "contamination"},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


@pytest.fixture
def cohort_file_listshape(tmp_path: Path) -> Path:
    """Alternate cohort file shape (lists per cohort)."""
    path = tmp_path / "cohort_lists.json"
    path.write_text(json.dumps({"A": ["x__y-1"], "B": ["x__y-2"], "C": []}))
    return path


# ---------------------------------------------------------------------------
# Unit: iter_milo_jsonls
# ---------------------------------------------------------------------------


class TestIterMiloJsonls:
    def test_iter_dir(self, synth_jsonl_dir):
        items = list(iter_milo_jsonls(synth_jsonl_dir))
        assert len(items) == 3
        ids = sorted([obj["instance_id"] for _, obj in items])
        assert ids == ["baz__qux-7", "foo__bar-42", "locustio__locust-1541"]

    def test_iter_single_file(self, synth_jsonl_dir):
        f = next(synth_jsonl_dir.iterdir())
        items = list(iter_milo_jsonls(f))
        assert len(items) == 1

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(iter_milo_jsonls(tmp_path / "does-not-exist"))

    def test_bad_json_line_is_skipped(self, tmp_path):
        f = tmp_path / "broken.jsonl"
        f.write_text('{"instance_id": "ok__1-1"}\nthis is not json\n{"instance_id": "ok__2-2"}\n')
        items = list(iter_milo_jsonls(f))
        ids = sorted([o["instance_id"] for _, o in items])
        assert ids == ["ok__1-1", "ok__2-2"]


# ---------------------------------------------------------------------------
# Unit: load_cohort
# ---------------------------------------------------------------------------


class TestLoadCohort:
    def test_dict_of_dicts_shape(self, cohort_file):
        a = load_cohort(cohort_file, ["A"])
        assert a == {"locustio__locust-1541", "foo__bar-42"}

    def test_multi_cohort(self, cohort_file):
        ab = load_cohort(cohort_file, ["A", "B"])
        assert ab == {"locustio__locust-1541", "foo__bar-42", "baz__qux-7"}

    def test_drop_cohort_isolated(self, cohort_file):
        c = load_cohort(cohort_file, ["C"])
        assert c == {"drop__me-99"}

    def test_dict_of_lists_shape(self, cohort_file_listshape):
        a = load_cohort(cohort_file_listshape, ["A"])
        assert a == {"x__y-1"}

    def test_missing_file_returns_empty_set(self, tmp_path):
        result = load_cohort(tmp_path / "nope.json", ["A"])
        assert result == set()


# ---------------------------------------------------------------------------
# Unit: milo_to_row
# ---------------------------------------------------------------------------


class TestMiloToRow:
    def test_basic_row_shape(self):
        inst = _synthetic_milo("foo__bar-1")
        row = milo_to_row(inst, image_name="python:3.11-slim")
        assert isinstance(row, MiloRow)
        d = row.to_dict()
        assert d["data_source"] == DEFAULT_DATA_SOURCE
        assert d["env_class"] == DEFAULT_ENV_CLASS
        assert d["reward_spec"] == {"method": "verifier", "ground_truth": None}
        assert isinstance(d["prompt"], list)
        assert d["prompt"][0]["role"] == "user"
        assert "instance" in d["extra_info"]

    def test_instance_payload_complete(self):
        inst = _synthetic_milo("foo__bar-1")
        row = milo_to_row(inst, image_name="img:tag")
        payload = row.extra_info["instance"]
        for key in (
            "instance_id",
            "image_name",
            "problem_statement",
            "eval_script",
            "milo_org",
            "milo_repo",
            "milo_f2p_test_ids",
            "milo_p2p_test_ids",
            "test_patch",
            "fix_patch",
            "tag_start",
            "tag_end",
        ):
            assert key in payload, f"missing field {key}"
        assert payload["image_name"] == "img:tag"
        assert payload["milo_f2p_test_ids"] == [
            "tests/test_foo.py::test_a",
            "tests/test_foo.py::test_b",
        ]

    def test_bundle_note_appended(self):
        inst = _synthetic_milo("foo__bar-1", bundle_size=4)
        row = milo_to_row(inst, image_name="img:tag")
        ps = row.extra_info["instance"]["problem_statement"]
        assert "bundle of 4 related PRs" in ps

    def test_image_name_derivation(self, monkeypatch):
        monkeypatch.delenv("EVAL_DOCKER_IMAGE_PREFIX", raising=False)
        inst = _synthetic_milo("locustio__locust-1541")
        row = milo_to_row(inst)  # no override
        assert row.extra_info["instance"]["image_name"].endswith("locustio_m_locust:pr-1541")

    def test_missing_instance_id_raises(self):
        with pytest.raises(ValueError):
            milo_to_row({"org": "x"}, image_name="img:tag")

    def test_no_f2p_emits_failing_eval_script(self):
        inst = _synthetic_milo("foo__bar-1", with_f2p=False)
        row = milo_to_row(inst, image_name="img:tag")
        es = row.extra_info["instance"]["eval_script"]
        assert "exit 1" in es
        assert "Cohort B" in es


# ---------------------------------------------------------------------------
# Integration: build_parquet round-trip
# ---------------------------------------------------------------------------


class TestBuildParquet:
    def test_default_filter_excludes_no_f2p(self, synth_jsonl_dir, tmp_path):
        out = tmp_path / "train.parquet"
        stats = build_parquet(
            src_jsonl_dir=synth_jsonl_dir,
            cohort_filter=_default_filter,
            out_path=out,
            image_name_override="img:tag",
        )
        assert isinstance(stats, DatasetStats)
        # 3 input instances; 1 lacks F2P (Cohort B), so 2 accepted.
        assert stats.accepted_rows == 2
        assert stats.rejected_no_f2p == 1
        assert out.is_file()

    def test_round_trip_parquet(self, synth_jsonl_dir, tmp_path):
        out = tmp_path / "round.parquet"
        build_parquet(
            src_jsonl_dir=synth_jsonl_dir,
            cohort_filter=lambda inst: inst["instance_id"] == "locustio__locust-1541",
            out_path=out,
            image_name_override="my-img:tag",
        )
        # Read back via pandas if available, else pyarrow.
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(out)
            assert len(df) == 1
            row = df.iloc[0].to_dict()
        except ImportError:
            import pyarrow.parquet as pq  # type: ignore

            tbl = pq.read_table(out)
            rows = tbl.to_pylist()
            assert len(rows) == 1
            row = rows[0]
        assert row["data_source"] == DEFAULT_DATA_SOURCE
        assert row["env_class"] == DEFAULT_ENV_CLASS
        assert row["extra_info"]["instance"]["instance_id"] == "locustio__locust-1541"
        assert row["extra_info"]["instance"]["image_name"] == "my-img:tag"
        assert row["extra_info"]["instance"]["milo_f2p_test_ids"] == [
            "tests/test_foo.py::test_a",
            "tests/test_foo.py::test_b",
        ]

    def test_cohort_filter_drops_b_and_c(self, synth_jsonl_dir, cohort_file, tmp_path):
        out = tmp_path / "out.parquet"
        ids_a = load_cohort(cohort_file, ["A"])
        stats = build_parquet(
            src_jsonl_dir=synth_jsonl_dir,
            cohort_filter=lambda inst: inst["instance_id"] in ids_a,
            out_path=out,
            image_name_override="img:tag",
        )
        # foo__bar-42 has F2P+P2P AND in cohort A; locustio__locust-1541 same.
        # baz__qux-7 lacks F2P, gets rejected.
        assert stats.accepted_rows == 2

    def test_zero_rows_raises(self, synth_jsonl_dir, tmp_path):
        out = tmp_path / "out.parquet"
        with pytest.raises(RuntimeError, match="0 rows"):
            build_parquet(
                src_jsonl_dir=synth_jsonl_dir,
                cohort_filter=lambda inst: False,
                out_path=out,
                image_name_override="img:tag",
            )

    def test_lang_stats_collected(self, synth_jsonl_dir, tmp_path):
        out = tmp_path / "out.parquet"
        stats = build_parquet(
            src_jsonl_dir=synth_jsonl_dir,
            cohort_filter=lambda inst: True,
            out_path=out,
            image_name_override="img:tag",
            require_f2p=False,
            require_p2p=False,
        )
        assert "python" in stats.languages
        assert "go" in stats.languages
        assert "rust" in stats.languages


class TestDatasetStatsSummary:
    def test_summary_is_string(self):
        s = DatasetStats()
        out = s.summary()
        assert isinstance(out, str)
        assert "accepted=" in out
