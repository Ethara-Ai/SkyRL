"""milo.sft — SFT warmstart pipeline (Phase 12).

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §12 / ``RL_GYM_SPEC.md`` v0.7 §16.

Public surface:

* :func:`milo.sft.build_sft_dataset.build_sft_dataset` — convert OpenHands+overlay
  golden traces into Qwen-format conversational SFT JSONL with assistant-only
  loss masking (spec §16.2).
* :func:`milo.sft.validate.validate_sft_checkpoint` — run the spec §16.4
  acceptance criteria over a warmstart checkpoint, returning a
  :class:`SFTValidationReport` the registry (Phase 18.5) consumes.
* :mod:`milo.sft.sft_v1` — dataclass config snapshot of the reference SFT recipe
  (1 epoch, LR 5e-6, AdamW (0.9, 0.95), wd=0.01, seq_len 32K, 8 GPUs, full FT,
  cosine LR with 3% warmup, assistant-only loss masking).

Notes
-----
The actual HuggingFace ``Trainer`` driver (``main_sft.py``) is intentionally not
in this milestone — Phase 12.2 in the plan calls for a thin SkyRL-side runner
that consumes :class:`SFTConfig` and routes the data through ``transformers``
with DeepSpeed Zero-2. The interfaces here cover the build + validate halves
of the pipeline that have no GPU dependency and can be unit-tested on a laptop.
"""

from __future__ import annotations

__all__ = [
    "build_sft_dataset",
    "validate_sft_checkpoint",
    "SFTValidationReport",
    "SFT_V1_CONFIG",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Lazy imports keep `import milo.sft` cheap (avoids pulling transformers).
    if name == "build_sft_dataset":
        from milo.sft.build_sft_dataset import build_sft_dataset

        return build_sft_dataset
    if name in ("validate_sft_checkpoint", "SFTValidationReport"):
        from milo.sft.validate import SFTValidationReport, validate_sft_checkpoint

        return {"validate_sft_checkpoint": validate_sft_checkpoint, "SFTValidationReport": SFTValidationReport}[name]
    if name == "SFT_V1_CONFIG":
        from milo.sft.sft_v1 import SFT_V1_CONFIG

        return SFT_V1_CONFIG
    raise AttributeError(name)
