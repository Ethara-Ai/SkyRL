"""Tests for the paired-bootstrap CI."""

from __future__ import annotations

import statistics

from milo.eval.bootstrap_ci import paired_bootstrap_ci


def test_point_matches_mean() -> None:
    deltas = [0.1, 0.05, 0.2, -0.05, 0.15]
    point, _, _ = paired_bootstrap_ci(deltas, n_resamples=1000)
    assert abs(point - statistics.fmean(deltas)) < 1e-9


def test_all_zero_deltas() -> None:
    point, lo, hi = paired_bootstrap_ci([0.0] * 20, n_resamples=2000)
    assert point == 0.0
    assert abs(lo) < 1e-9
    assert abs(hi) < 1e-9


def test_release_gate_clearly_positive() -> None:
    """Deltas all ≥ +10pp → bootstrap_lower > 0 → release gate passes."""
    deltas = [0.10, 0.12, 0.11, 0.09, 0.13] * 12
    point, lo, hi = paired_bootstrap_ci(deltas, n_resamples=2000)
    assert point > 0.05
    assert lo > 0.0


def test_empty_input_returns_zeros() -> None:
    point, lo, hi = paired_bootstrap_ci([])
    assert point == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_seeded_reproducibility() -> None:
    deltas = [0.05, 0.10, -0.02, 0.08]
    a = paired_bootstrap_ci(deltas, n_resamples=500, seed=42)
    b = paired_bootstrap_ci(deltas, n_resamples=500, seed=42)
    assert a == b
