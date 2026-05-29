"""Reference SFT recipe — `qwen2.5-coder-32b-milo-sft-v1`.

Implements the hyperparameter table in ``RL_GYM_SPEC.md`` v0.7 §16.3 and the
plan ``IMPLEMENTATION_PLAN.md`` v0.4 §12.2 — the locked SFT configuration that
produces the first warmstart checkpoint.

The values here are the **shipped defaults** that AGIF receives. Each can be
overridden via ``MiloSFTConfig.from_cli_overrides`` (the same dataclass-CLI
pattern SkyRL uses; see :mod:`skyrl.train.config.config`).

The dataclass deliberately mirrors SkyRL's typed config style (typed fields,
explicit defaults, ``__post_init__`` validation) so the SFT runner can be a
thin wrapper that loads :class:`SFT_V1_CONFIG`, applies CLI overrides, and
hands the result to HuggingFace ``Trainer`` + DeepSpeed Zero-2.

The actual model identifier is **env-driven** per IMPLEMENTATION_PLAN v0.4 §22
("Engineering rules of the road" — no hard-coded forward-dated IDs); the
default below resolves through ``MILO_SFT_BASE_MODEL`` and only falls back to
the spec §15 default if unset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


DEFAULT_BASE_MODEL = os.environ.get(
    "MILO_SFT_BASE_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
)


@dataclass(slots=True)
class MiloOptimizerSFTConfig:
    """AdamW (`β=(0.9, 0.95)`, `wd=0.01`) per spec §16.3.

    The (`0.9`, `0.95`) betas differ from the SkyRL default (`0.9`, `0.999`) —
    spec §16.3 explicitly calls for the SFT-style `β₂=0.95`; the RL run swaps
    back to `0.999` per §14.2.
    """

    lr: float = 5.0e-6
    """Conservative SFT LR — preserves base capabilities (spec §16.3)."""

    adam_betas: tuple[float, float] = (0.9, 0.95)
    """`β₁=0.9`, `β₂=0.95`. SFT-style; RL uses 0.999."""

    eps: float = 1.0e-8

    weight_decay: float = 0.01

    max_grad_norm: float = 1.0
    """Gradient clipping per spec §14.3."""

    scheduler: Literal["cosine", "constant", "constant_with_warmup"] = "cosine"
    warmup_pct: float = 0.03
    """3 % cosine warmup per spec §16.3."""


@dataclass(slots=True)
class MiloLossSFTConfig:
    """Loss config for assistant-only masked CE (spec §16.3)."""

    loss_fn: Literal["cross_entropy"] = "cross_entropy"
    mask_non_assistant: bool = True
    """Mask system/user/tool tokens; only assistant tokens contribute to loss."""

    label_smoothing: float = 0.0


@dataclass(slots=True)
class MiloSFTDataConfig:
    """Dataset paths + sequence-length config."""

    train_jsonl: str = ""
    """Built by :mod:`milo.sft.build_sft_dataset` — typically
    ``milo/data/sft_train.jsonl`` (240 train traces)."""

    val_jsonl: str = ""
    """Held-out validation traces — typically ``milo/data/sft_val.jsonl``
    (60 val traces). See spec §19.1 for the train/val split."""

    sequence_length: int = 32_768
    """Spec §16.3. Longer traces are truncated **from the head** to preserve
    the recent context the final assistant turn depends on."""

    pack_sequences: bool = True
    """Pack short sequences to fill the 32K window — common for long-context
    SFT. Disable for byte-equality SFT debugging."""


@dataclass(slots=True)
class MiloSFTDistributedConfig:
    """8 H100s, full FT, no LoRA — see spec §16.3."""

    num_gpus: int = 8
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    @property
    def effective_batch_size(self) -> int:
        """4 × 8 × 8 = 256 effective examples per update."""
        return self.num_gpus * self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass(slots=True)
class MiloSFTConfig:
    """Top-level SFT config snapshot — see module docstring.

    The actual SkyRL-style ``SkyRLSFTConfig`` (with ``from_cli_overrides``) is
    not in this module — when the SFT runner lands, it will subclass SkyRL's
    config machinery and embed this dataclass as the milo-specific layer.
    """

    base_model: str = DEFAULT_BASE_MODEL
    """`MILO_SFT_BASE_MODEL` overrides; defaults per spec §15."""

    base_model_revision: str | None = None
    """HF revision pin. Required at runtime for reproducibility; unset by
    default so a developer can iterate locally without a revision lookup."""

    adapter: Literal["none", "lora"] = "none"
    """Spec §16.3: full FT, no adapter. The LoRA-vs-FT bake-off in plan
    §19.1.6 is for the RL phase, not SFT."""

    epochs: int = 1
    """One epoch on 240 traces — over-training hurts (spec §16.3)."""

    seed: int = 42
    """Reproducibility seed; recorded in the manifest (spec §21.2)."""

    optimizer: MiloOptimizerSFTConfig = field(default_factory=MiloOptimizerSFTConfig)
    loss: MiloLossSFTConfig = field(default_factory=MiloLossSFTConfig)
    data: MiloSFTDataConfig = field(default_factory=MiloSFTDataConfig)
    dist: MiloSFTDistributedConfig = field(default_factory=MiloSFTDistributedConfig)

    output_dir: str = "milo/data/sft_runs/sft_v1"
    """Where the runner writes checkpoints + tokenizer + manifest."""

    save_every_steps: int | None = None
    """If set, save intermediate checkpoints every N steps. ``None`` means
    only save the final epoch checkpoint (matches the 1-epoch SFT plan)."""

    log_every_steps: int = 10
    """W&B / stdout cadence."""


SFT_V1_CONFIG = MiloSFTConfig()
"""The shipped reference SFT recipe instance (spec §16.3, plan §12.2)."""

__all__ = [
    "DEFAULT_BASE_MODEL",
    "MiloOptimizerSFTConfig",
    "MiloLossSFTConfig",
    "MiloSFTDataConfig",
    "MiloSFTDistributedConfig",
    "MiloSFTConfig",
    "SFT_V1_CONFIG",
]
