"""Tests for the Schulman k3 KL estimator."""

from __future__ import annotations

import math

import pytest

from milo.algos.kl_k3 import per_token_kl_k3


def test_identity_is_zero_python_fallback() -> None:
    p = [-1.0, -2.0, -3.0]
    r = [-1.0, -2.0, -3.0]
    out = per_token_kl_k3(p, r)
    for v in out:
        assert abs(v) < 1e-12


def test_non_negative_for_random_inputs_python_fallback() -> None:
    import random

    random.seed(0)
    p = [random.uniform(-5, 0) for _ in range(200)]
    r = [random.uniform(-5, 0) for _ in range(200)]
    out = per_token_kl_k3(p, r)
    for v in out:
        assert v >= -1e-9   # exact zero for identity; otherwise strictly > 0


def test_monotone_in_distance() -> None:
    """k3 grows as |ref - policy| grows."""
    p = [-1.0]
    base = per_token_kl_k3(p, [-1.0])[0]
    a = per_token_kl_k3(p, [-1.5])[0]
    b = per_token_kl_k3(p, [-2.0])[0]
    assert base < a < b


def test_known_value() -> None:
    """ref=0, policy=0 → kl=0. ref=1, policy=0 → e^1 - 1 - 1 = ~0.718."""
    out = per_token_kl_k3([0.0], [1.0])
    assert math.isclose(out[0], math.e - 2.0, abs_tol=1e-9)


def test_nested_list_shape_preserved() -> None:
    """Schulman k3 should preserve [batch, seq] shape (list or ndarray)."""
    p = [[-1.0, -2.0], [-3.0, -4.0]]
    r = [[-1.0, -2.0], [-3.0, -4.0]]
    out = per_token_kl_k3(p, r)
    # Accept either numpy ndarray or list — depends on which backend was importable.
    try:
        import numpy as np  # type: ignore[import-not-found]
        if isinstance(out, np.ndarray):
            assert out.shape == (2, 2)
            assert (out == 0).all()
            return
    except ImportError:
        pass
    assert isinstance(out, list)
    assert len(out) == 2
    assert len(out[0]) == 2  # type: ignore[arg-type]


def test_shape_mismatch_raises_in_python_fallback() -> None:
    """The pure-Python fallback raises on shape mismatch; numpy broadcasts.
    We force the fallback by passing object dtype to make numpy raise too."""
    from milo.algos.kl_k3 import _per_token_kl_k3_python
    with pytest.raises(ValueError):
        _per_token_kl_k3_python([1.0, 2.0], [1.0])
