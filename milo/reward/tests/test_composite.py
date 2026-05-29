"""Composite-reward aggregator tests — spec §4.4 / §10.4 / Phase 4.

Covers:
  * Property-based test over 10K random inputs (spec §10.4 CI gate).
  * Invariant override: `passes_invariant_check=False` zeroes R_terminal.
  * Both `composite` and `pure_rlvr` presets load with the documented
    spec-default weights.
  * Override mechanism (Hydra-like dict overrides).
"""

from __future__ import annotations

import json
import random

import pytest

from milo.reward import (
    CompositeRewardAggregator,
    RewardDecomposition,
    StubRubricReport,
    StubVerifierReport,
)


# ---------------------------------------------------------------------------
# Property-based test — spec §10.4 requires 10K random inputs.
# ---------------------------------------------------------------------------


def test_decomposition_arithmetic_10k_random() -> None:
    """`r_total == r_terminal + alpha*sum(delta) + beta*rubric + gamma*sum(tir)`.

    Spec §10.4: 10K random inputs, tolerance 1e-9. We sweep across both
    presets and a range of override weights so the property holds for the
    full configurable space.
    """
    rng = random.Random(0xC0FFEE)
    presets = ("composite", "pure_rlvr")
    for trial in range(10_000):
        preset = presets[trial % len(presets)]

        # Random overrides — including edge cases like 0 and large alphas.
        overrides = {
            "alpha": rng.choice([0.0, 0.01, 0.05, 0.5, 1.0, 5.0]),
            "beta": rng.choice([0.0, 0.05, 0.2, 1.0, 3.0]),
            "lambda": rng.choice([0.0, 1.0, 2.0, 5.0]),
            "gamma": rng.choice([0.0, 0.001, 0.01, 0.1]),
        }
        agg = CompositeRewardAggregator(preset=preset, overrides=overrides)

        # Random shaping (any real value in a wide range).
        n_shaping = rng.randint(0, 64)
        shaping = [rng.uniform(-20.0, 20.0) for _ in range(n_shaping)]

        # Random TIR steps (each -1 or 0).
        n_tir = rng.randint(0, 128)
        tir = [rng.choice([-1, 0]) for _ in range(n_tir)]

        # Random rubric scores in {0, 0.5, 1}.
        n_rubric = rng.randint(0, 8)
        rubric_scores = [rng.choice([0.0, 0.5, 1.0]) for _ in range(n_rubric)]
        rubric = StubRubricReport(per_item_scores=rubric_scores)

        # Random verifier resolution AND independent random invariant pass.
        resolved = bool(rng.getrandbits(1))
        passes_inv = bool(rng.getrandbits(1))
        verifier = StubVerifierReport(
            resolved=resolved, passes_invariant_check=passes_inv
        )

        decomp = agg.aggregate(verifier, rubric, shaping, tir)

        # Expected r_total per spec §4.4.
        expected_terminal = 1 if (resolved and passes_inv) else 0
        expected_rubric_mean = (
            sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0
        )
        expected_total = (
            expected_terminal
            + overrides["alpha"] * sum(shaping)
            + overrides["beta"] * expected_rubric_mean
            + overrides["gamma"] * sum(tir)
        )
        assert abs(decomp.r_total - expected_total) < 1e-9, (
            f"trial {trial}: got {decomp.r_total}, expected {expected_total}, "
            f"diff={decomp.r_total - expected_total}"
        )

        # Components must sum to r_total to within fp tolerance.
        assert abs(sum(decomp.components.values()) - decomp.r_total) < 1e-9


def test_decomposition_arithmetic_pure_rlvr_zeros_shaping() -> None:
    """`pure_rlvr` preset literally cannot give shaping/rubric/TIR signal."""
    agg = CompositeRewardAggregator(preset="pure_rlvr")
    verifier = StubVerifierReport(resolved=True, passes_invariant_check=True)
    rubric = StubRubricReport(per_item_scores=[1.0, 1.0, 1.0])
    # Lots of shaping + lots of TIR penalty — none should leak into r_total.
    decomp = agg.aggregate(verifier, rubric, [10.0] * 100, [-1] * 100)
    assert decomp.r_total == 1.0
    assert decomp.components["shaping"] == 0.0
    assert decomp.components["rubric"] == 0.0
    assert decomp.components["tir"] == 0.0


# ---------------------------------------------------------------------------
# Invariant override (spec §7 / v0.7-hardened I-2).
# ---------------------------------------------------------------------------


def test_invariant_failure_forces_terminal_zero() -> None:
    """Spec §7: invariant violation → R_terminal := 0 regardless of test result."""
    agg = CompositeRewardAggregator(preset="composite")
    verifier = StubVerifierReport(resolved=True, passes_invariant_check=False)
    rubric = StubRubricReport(per_item_scores=[1.0])
    decomp = agg.aggregate(verifier, rubric, [], [])
    assert decomp.r_terminal == 0
    # Only the rubric channel can contribute; shaping/TIR were empty.
    assert decomp.components["terminal"] == 0.0
    # With beta=0.2 and rubric mean=1.0, expected total is 0.2.
    assert abs(decomp.r_total - 0.2) < 1e-9


def test_invariant_check_independent_of_resolved() -> None:
    """All four (resolved, passes) combinations behave correctly."""
    agg = CompositeRewardAggregator(preset="pure_rlvr")
    cases = [
        (True, True, 1),
        (True, False, 0),     # hardened: tests pass but invariant fails -> 0
        (False, True, 0),
        (False, False, 0),
    ]
    for resolved, passes_inv, expected_terminal in cases:
        verifier = StubVerifierReport(
            resolved=resolved, passes_invariant_check=passes_inv
        )
        decomp = agg.aggregate(verifier, None, [], [])
        assert decomp.r_terminal == expected_terminal, (
            f"resolved={resolved}, passes_inv={passes_inv}"
        )


def test_no_verifier_report_means_zero_terminal() -> None:
    """Missing verifier report → terminal 0 (defensive default)."""
    agg = CompositeRewardAggregator(preset="composite")
    decomp = agg.aggregate(None, None, [], [])
    assert decomp.r_terminal == 0
    assert decomp.r_total == 0.0


# ---------------------------------------------------------------------------
# Preset loading.
# ---------------------------------------------------------------------------


def test_composite_preset_loads_spec_defaults() -> None:
    """Spec-§4.4 defaults: alpha=0.05, beta=0.20, lambda=2.0, gamma=0.0."""
    agg = CompositeRewardAggregator(preset="composite")
    assert agg.preset == "composite"
    assert agg.alpha == pytest.approx(0.05)
    assert agg.beta == pytest.approx(0.20)
    assert agg.lambda_ == pytest.approx(2.0)
    assert agg.gamma == pytest.approx(0.0)


def test_pure_rlvr_preset_zeros_alpha_beta_gamma() -> None:
    """Spec-§4.4 pure_rlvr: alpha=beta=gamma=0."""
    agg = CompositeRewardAggregator(preset="pure_rlvr")
    assert agg.preset == "pure_rlvr"
    assert agg.alpha == 0.0
    assert agg.beta == 0.0
    assert agg.gamma == 0.0


def test_overrides_take_effect() -> None:
    """Overrides dict trumps preset YAML."""
    agg = CompositeRewardAggregator(
        preset="composite",
        overrides={"alpha": 1.0, "beta": 0.5, "gamma": 0.01, "lambda": 3.0},
    )
    assert agg.alpha == 1.0
    assert agg.beta == 0.5
    assert agg.gamma == 0.01
    assert agg.lambda_ == 3.0


def test_unknown_preset_raises() -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        CompositeRewardAggregator(preset="does_not_exist")


def test_negative_lambda_raises() -> None:
    with pytest.raises(ValueError):
        CompositeRewardAggregator(
            preset="composite", overrides={"lambda": -1.0}
        )


# ---------------------------------------------------------------------------
# Decomposition schema (spec §6.8).
# ---------------------------------------------------------------------------


def test_decomposition_to_json_matches_spec_6_8_keys() -> None:
    """Spec §6.8 declares exactly these keys; check we emit them all."""
    agg = CompositeRewardAggregator(preset="composite")
    verifier = StubVerifierReport(resolved=True, passes_invariant_check=True)
    rubric = StubRubricReport(per_item_scores=[1.0, 0.5, 0.0])
    decomp = agg.aggregate(verifier, rubric, [1.0, -2.0], [0, -1, 0])
    payload = json.loads(decomp.to_json())
    expected_keys = {
        "preset",
        "r_terminal",
        "r_delta_steps",
        "r_delta_sum",
        "r_rubric_per_item",
        "r_rubric_mean",
        "r_tir_steps",
        "r_tir_sum",
        "alpha",
        "beta",
        "lambda",
        "gamma",
        "r_total",
        "components",
    }
    assert set(payload.keys()) == expected_keys
    assert set(payload["components"].keys()) == {
        "terminal",
        "shaping",
        "rubric",
        "tir",
    }
    # lambda (not lambda_) in the serialized form.
    assert "lambda" in payload
    assert "lambda_" not in payload


def test_decomposition_round_trip_preserves_fields() -> None:
    """`from_dict(to_dict())` returns an equal RewardDecomposition."""
    agg = CompositeRewardAggregator(
        preset="composite", overrides={"gamma": 0.01}
    )
    verifier = StubVerifierReport(resolved=True, passes_invariant_check=True)
    rubric = StubRubricReport(per_item_scores=[0.5])
    decomp = agg.aggregate(verifier, rubric, [3.0], [0, -1])
    recovered = RewardDecomposition.from_dict(decomp.to_dict())
    assert recovered.to_dict() == decomp.to_dict()


def test_r_total_not_clipped() -> None:
    """Spec §4.4.4: R_total is NOT clipped — large positive values pass through."""
    agg = CompositeRewardAggregator(
        preset="composite", overrides={"alpha": 1.0, "beta": 1.0}
    )
    verifier = StubVerifierReport(resolved=True, passes_invariant_check=True)
    rubric = StubRubricReport(per_item_scores=[1.0])
    decomp = agg.aggregate(verifier, rubric, [100.0] * 10, [])
    # Should be 1 (terminal) + 1.0 * 1000 (shaping) + 1.0 * 1.0 (rubric).
    assert decomp.r_total == pytest.approx(1002.0)
