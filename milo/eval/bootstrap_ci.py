"""Paired bootstrap CI for pass@k deltas — Phase 16 / spec §19.

Implementation of the release-gate statistical test from `IMPLEMENTATION_PLAN.md`
v0.4 §19.2: `point_estimate >= +5pp AND bootstrap_lower > 0` against the SFT
baseline. 10,000 resamples per spec §19.

Pure function, no torch/numpy hard dependency at import time (uses random +
statistics from stdlib for the fallback).
"""

from __future__ import annotations

import random
import statistics
from typing import Sequence


def paired_bootstrap_ci(
    deltas: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Paired-bootstrap (1-alpha)% CI on the mean of `deltas`.

    Returns `(point_estimate, lower, upper)` where:
        point_estimate = mean(deltas)
        lower, upper   = the (alpha/2, 1-alpha/2) percentiles of the
                         bootstrap distribution of resampled means.

    `deltas` is the list of per-task (model_pass_rate - baseline_pass_rate)
    values; the bootstrap resamples with replacement, recomputes the mean,
    and percentile-clips.
    """
    if len(deltas) == 0:
        return 0.0, 0.0, 0.0

    deltas = list(deltas)
    point = float(statistics.fmean(deltas))

    # Try numpy/scipy for speed; fall back to pure-Python.
    try:
        import numpy as np  # type: ignore[import-not-found]
        rng = np.random.default_rng(seed)
        arr = np.asarray(deltas, dtype=np.float64)
        n = len(arr)
        idx = rng.integers(0, n, size=(n_resamples, n))
        means = arr[idx].mean(axis=1)
        lower = float(np.percentile(means, 100 * (alpha / 2)))
        upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return point, lower, upper
    except ImportError:  # pragma: no cover
        pass

    r = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(n_resamples):
        resample = [deltas[r.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    means.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    return point, means[max(lo_idx, 0)], means[max(min(hi_idx, n_resamples - 1), 0)]
