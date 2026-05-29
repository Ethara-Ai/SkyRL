"""Schulman k3 per-token KL estimator — spec §14.5 / plan §14.3.

    KL_k3(token) = exp(log_ratio) - log_ratio - 1
                   where log_ratio = ref_logprob - policy_logprob

Properties (vs the k1 estimator `ref - policy`):
    * unbiased
    * always non-negative
    * lower variance

Used by the GRPO loss in `milo/algos/grpo_wrapper.py` and tested in
`milo/algos/tests/test_kl_k3.py` (identity → zero, monotonicity,
non-negativity, asymmetry).

Pure function — no torch/numpy hard dependency at import time. Operates
on whichever array library the caller supplies (works with numpy ndarrays,
torch Tensors, and Python lists via the numpy fallback). When neither
numpy nor torch is importable, falls back to pure-Python for short lists.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def per_token_kl_k3(
    policy_logprobs: Any,
    reference_logprobs: Any,
    sampled_tokens: Any | None = None,
) -> Any:
    """Compute Schulman k3 per-token KL.

    `policy_logprobs` and `reference_logprobs` must have the same shape
    (typically `[batch, seq]`). `sampled_tokens` is unused by k3 itself but
    accepted for API symmetry with k1 / sample-aware variants; the trainer
    passes it through so call sites can switch estimators without rewiring.
    """
    # Try torch first (the trainer's native dtype).
    try:
        import torch  # type: ignore[import-not-found]
        if isinstance(policy_logprobs, torch.Tensor) or isinstance(reference_logprobs, torch.Tensor):
            log_ratio = reference_logprobs - policy_logprobs
            return torch.exp(log_ratio) - log_ratio - 1.0
    except ImportError:  # pragma: no cover
        pass
    # Then numpy.
    try:
        import numpy as np  # type: ignore[import-not-found]
        p = np.asarray(policy_logprobs, dtype=np.float64)
        r = np.asarray(reference_logprobs, dtype=np.float64)
        log_ratio = r - p
        return np.exp(log_ratio) - log_ratio - 1.0
    except ImportError:  # pragma: no cover
        pass
    # Python fallback (used by sandboxed tests that don't have torch/numpy).
    return _per_token_kl_k3_python(policy_logprobs, reference_logprobs)


def _per_token_kl_k3_python(
    policy_logprobs: Sequence[Any], reference_logprobs: Sequence[Any]
) -> list[Any]:
    if hasattr(policy_logprobs, "__len__") and hasattr(reference_logprobs, "__len__"):
        if len(policy_logprobs) != len(reference_logprobs):  # type: ignore[arg-type]
            raise ValueError(
                f"shape mismatch: {len(policy_logprobs)} vs {len(reference_logprobs)}"
            )
    out: list[Any] = []
    for p, r in zip(policy_logprobs, reference_logprobs):
        if isinstance(p, (list, tuple)) or isinstance(r, (list, tuple)):
            out.append(_per_token_kl_k3_python(p, r))
        else:
            log_ratio = float(r) - float(p)
            out.append(math.exp(log_ratio) - log_ratio - 1.0)
    return out
