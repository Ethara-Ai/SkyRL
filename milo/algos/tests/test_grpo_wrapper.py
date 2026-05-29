"""Tests for the milo_grpo loss wrapper and β_KL annealing."""

from __future__ import annotations

import math

from milo.algos.grpo_wrapper import (
    DEFAULT_BETA_KL_ANNEAL_STEPS,
    DEFAULT_BETA_KL_FINAL,
    DEFAULT_BETA_KL_INITIAL,
    GRPOLossComponents,
    anneal_beta_kl,
    compute_grpo_loss_components,
)


def test_anneal_beta_kl_boundaries() -> None:
    assert anneal_beta_kl(0) == DEFAULT_BETA_KL_INITIAL
    assert anneal_beta_kl(-100) == DEFAULT_BETA_KL_INITIAL
    assert anneal_beta_kl(DEFAULT_BETA_KL_ANNEAL_STEPS) == DEFAULT_BETA_KL_FINAL
    assert anneal_beta_kl(DEFAULT_BETA_KL_ANNEAL_STEPS * 2) == DEFAULT_BETA_KL_FINAL


def test_anneal_beta_kl_linear_midpoint() -> None:
    """Halfway through anneal_steps should give halfway between initial and final."""
    mid = DEFAULT_BETA_KL_ANNEAL_STEPS // 2
    expected = (DEFAULT_BETA_KL_INITIAL + DEFAULT_BETA_KL_FINAL) / 2
    got = anneal_beta_kl(mid)
    assert math.isclose(got, expected, abs_tol=1e-6)


def test_v04_defaults_match_spec() -> None:
    """v0.4 spec §14.5: β_KL: 0.01 → 0.005 over 4800 steps."""
    assert DEFAULT_BETA_KL_INITIAL == 0.01
    assert DEFAULT_BETA_KL_FINAL == 0.005
    assert DEFAULT_BETA_KL_ANNEAL_STEPS == 4800


def test_compute_grpo_loss_zero_advantage_zero_pg() -> None:
    """All-zero advantages → PG loss ≈ 0."""
    import numpy as np  # type: ignore[import-not-found]

    n = 16
    p = np.zeros(n)
    olp = np.zeros(n)
    rl = np.zeros(n)
    adv = np.zeros(n)
    components = compute_grpo_loss_components(p, olp, rl, adv, step=0)
    assert abs(components.pg) < 1e-9


def test_compute_grpo_loss_identity_zero_kl() -> None:
    """policy == reference → KL ≈ 0."""
    import numpy as np  # type: ignore[import-not-found]

    n = 16
    p = np.array([-1.0] * n)
    olp = np.array([-1.0] * n)
    rl = np.array([-1.0] * n)
    adv = np.array([0.5] * n)
    components = compute_grpo_loss_components(p, olp, rl, adv, step=0)
    assert abs(components.kl) < 1e-9


def test_compute_grpo_loss_log_dict_keys() -> None:
    """Log dict must contain the spec §20.2 metric names."""
    import numpy as np  # type: ignore[import-not-found]

    p = np.array([-1.0, -2.0])
    olp = np.array([-1.0, -2.0])
    rl = np.array([-1.5, -2.5])
    adv = np.array([0.5, 0.5])
    components = compute_grpo_loss_components(p, olp, rl, adv, step=2400)
    log = components.to_log_dict()
    for key in (
        "train/loss_total", "train/loss_pg", "train/loss_kl",
        "train/advantage_mean", "train/advantage_std",
        "train/ratio_mean", "train/ratio_max",
        "train/kl_mean", "train/beta_kl",
        "train/grad_norm_pre_clip", "train/grad_norm_post_clip",
    ):
        assert key in log


def test_register_milo_grpo_no_op_without_skyrl() -> None:
    """register_milo_grpo should not crash when SkyRL trainer extras absent."""
    from milo.algos.grpo_wrapper import register_milo_grpo
    register_milo_grpo()
