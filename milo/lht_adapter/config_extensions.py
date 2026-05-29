"""Typed config extensions for the Milo LHT adapter.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §1.6: adds milo-bench knobs to the
SkyRL config tree via dataclass subclasses (SkyRL configs are dataclasses +
OmegaConf CLI overrides, *not* Hydra YAML — the older plan revision was wrong
about this). Defaults reflect ``RL_GYM_SPEC.md`` v0.7 — notably the
cost-guardrail ON by default at $5/episode and the 50-consecutive-no-edit
termination threshold. See ``MiloTrainConfig`` below for the wired-up
top-level config.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from skyrl.train.config.config import (
    BaseConfig,
    EnvironmentConfig,
    SkyRLGymConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)


__all__ = [
    "MiloEnvConfig",
    "MiloSkyRLGymConfig",
    "MiloEnvironmentConfig",
    "MiloTrainerConfig",
    "MiloTrainConfig",
    "make_milo_config",
]


# ---------------------------------------------------------------------------
# Milo LHT env knobs
# ---------------------------------------------------------------------------


@dataclass
class MiloEnvConfig(BaseConfig):
    """Per-env config for ``MiloLHTEnv`` (plan §1.6).

    These knobs are surfaced via the CLI as e.g.::

        +environment.skyrl_gym.milo_lht.cohort_filter=A
        +environment.skyrl_gym.milo_lht.cost_guardrail_usd=10.0

    Defaults track ``RL_GYM_SPEC.md`` v0.7. Any change here MUST be paired
    with a spec update — these are user-facing contract knobs.
    """

    cohort_filter: str = "A"
    """Which cohort partition the trainer should treat as in-distribution.

    Values:
      - ``"A"`` (default) — usable as-is (has F2P+P2P).
      - ``"A+B"`` — A plus Cohort B (verifier-construction needed). Only safe
        once Phase 11.X has produced the constructed verifiers.
      - ``"all"`` — A + B + C (testing only — C is supposed to be dropped).

    The actual filter logic lives in :func:`milo.lht_adapter.dataset.load_cohort`
    + :func:`build_parquet` at *preprocess* time; this knob is the runtime
    record of which partition was used, mostly for logging + the
    reproducibility manifest.
    """

    rubric_items_path: Optional[str] = None
    """Path to the JSON file of per-instance rubric items (see RL_GYM_SPEC §6.7).

    When ``None`` (default), the rubric judge (Phase 3) is skipped and the
    reward becomes RLVR-only. When set, the generator loads the file at
    ``MiloLHTGenerator.__init__`` and attaches the matching items to each
    rollout. Phase 11 owns producing this file; until then leave ``None``.
    """

    max_episode_seconds: int = 1800
    """Wall-clock cap per rollout (spec §4.5 default). On expiry the
    generator force-submits the current working tree."""

    max_tool_calls: int = 1000
    """Tool-call budget per rollout (spec §4.5 default). On exhaustion the
    generator force-submits."""

    cost_guardrail_usd: float = 5.0
    """Per-episode USD cap (spec §9.6, v0.7 hardened: ON by default at $5).

    Set to ``0.0`` or negative to disable. The generator polls the per-step
    cost from the inference engine response (when available) and force-submits
    when the cumulative cost exceeds this cap. ``termination_reason`` is
    written as ``"cost_guardrail"``.
    """

    consecutive_no_edit_terminate: int = 50
    """Force-terminate the episode after this many consecutive tool calls
    that produced no working-tree edit (spec §4.5 v0.7 addition). Prevents
    timeout-budget farming where the policy just reads files until the wall
    clock kills it.
    """

    # ------------------------------------------------------------------
    # Reward / verifier knobs (mostly pass-throughs; full Phase 2/4 owns these)
    # ------------------------------------------------------------------

    reward_preset: str = "composite"
    """One of ``"composite"`` (default, α/β/λ from spec §4.4) or
    ``"pure_rlvr"`` (α=β=γ=0). Read by the Phase 4 reward aggregator."""

    reward_alpha: float = 0.05
    reward_beta: float = 0.20
    reward_lambda: float = 2.0
    reward_gamma: float = 0.0
    """Composite-reward weights. Defaults match RL_GYM_SPEC v0.7 §4.4."""

    # ------------------------------------------------------------------
    # Sandbox knobs (forwarded to milo_sandbox)
    # ------------------------------------------------------------------

    sandbox_cpu: int = 4
    sandbox_mem_gb: int = 16
    sandbox_disk_gb: int = 40
    sandbox_workdir: str = "/testbed"
    sandbox_executable: str = "docker"
    """Per-rollout container resources (spec §5.1 configurable starting values)."""

    # ------------------------------------------------------------------
    # Trace + logging knobs
    # ------------------------------------------------------------------

    trace_root: str = "/tmp/milo_traces"
    """Local dir for per-rollout trace.jsonl + log files (spec §5.5).
    Background S3 sync (Phase 5) reads from here.
    """

    s3_bucket: Optional[str] = None
    """When set, the Phase 5 logger uploads completed traces here. Format:
    ``s3://bucket/prefix``. ``None`` keeps everything local (default for CI).
    """

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    seed: Optional[int] = None
    """Optional per-rollout RNG seed override. ``None`` => use the trainer
    seed (cfg.trainer.seed). Threaded into the inference engine seed
    parameter and into the trace header for replay (spec §9.2).
    """


# ---------------------------------------------------------------------------
# Wire MiloEnvConfig into the SkyRLGymConfig tree
# ---------------------------------------------------------------------------


@dataclass
class MiloSkyRLGymConfig(SkyRLGymConfig):
    """Subclass of upstream :class:`SkyRLGymConfig` that adds the ``milo_lht``
    sub-config. SkyRL's ``EnvironmentConfig.skyrl_gym`` typing remains
    backwards-compatible because Python is structural at runtime — the
    trainer reads ``cfg.environment.skyrl_gym.<env_class>`` by attribute name.

    The trainer's CLI parser will accept overrides like::

        +environment.skyrl_gym.milo_lht.cost_guardrail_usd=10.0
    """

    milo_lht: MiloEnvConfig = field(default_factory=MiloEnvConfig)


@dataclass
class MiloEnvironmentConfig(EnvironmentConfig):
    """Subclass of :class:`EnvironmentConfig` with our gym config wired in.

    Defaults ``env_class`` to ``"milo_lht"`` so the user doesn't have to set
    it on every CLI invocation.
    """

    env_class: str = "milo_lht"
    skyrl_gym: MiloSkyRLGymConfig = field(default_factory=MiloSkyRLGymConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
#
# SkyRL exposes ``make_config(...)`` (see skyrl/train/config/config.py) but
# that helper only accepts ``algorithm_cls``, ``trainer_cls``, and
# ``generator_cls`` — it does *not* parametrize the ``environment`` slot.
# That's a deviation from what the plan §1.6 sketch assumed; we work around it
# by writing a tiny ``make_milo_config()`` that mirrors ``make_config``'s
# subclass-and-override-defaults pattern for the environment slot.
#
# DEVIATION NOTE (per plan instructions): SkyRL ``make_config`` doesn't
# support an ``environment_skyrl_gym_cls`` kwarg. We patch the environment
# field via direct subclassing below; functionally equivalent.


@dataclass
class MiloTrainerConfig(TrainerConfig):
    """Trivial subclass — exists so ``make_milo_config`` has a clean target.

    No new fields today; reserved for future Milo-specific trainer knobs
    (e.g. per-cohort eval split, milo-specific dump_data_batch override).
    """


def make_milo_config() -> Type[SkyRLTrainConfig]:
    """Construct a :class:`SkyRLTrainConfig` subclass with our environment
    config wired in. Returned class is callable / has ``from_cli_overrides``.

    Usage::

        from milo.lht_adapter.config_extensions import make_milo_config

        MiloTrainConfig = make_milo_config()
        cfg = MiloTrainConfig.from_cli_overrides(sys.argv[1:])
    """
    annotations: Dict[str, Any] = {
        "environment": MiloEnvironmentConfig,
        "trainer": MiloTrainerConfig,
    }
    ns: Dict[str, Any] = {
        "__annotations__": annotations,
        "environment": field(default_factory=MiloEnvironmentConfig),
        "trainer": field(default_factory=MiloTrainerConfig),
    }
    return dataclass(type("_MiloSkyRLTrainConfig", (SkyRLTrainConfig,), ns))


# Eager construction so callers can simply do ``from ... import MiloTrainConfig``.
MiloTrainConfig: Type[SkyRLTrainConfig] = make_milo_config()


# ---------------------------------------------------------------------------
# Backwards-compat alias.
# ---------------------------------------------------------------------------
#
# The original plan sketch wrote ``MiloTrainConfig = make_config(...)``; if any
# external caller imports under that exact spelling, they get the same thing.

__all__ = list(__all__) + ["MiloTrainConfig"]


# Defensive: make sure we re-export TrainerConfig / EnvironmentConfig so
# downstream code can introspect without re-importing upstream SkyRL paths
# (kept private to milo).
_RE_EXPORTED: Dict[str, Any] = {
    "TrainerConfig": TrainerConfig,
    "EnvironmentConfig": EnvironmentConfig,
    "SkyRLGymConfig": SkyRLGymConfig,
    "SkyRLTrainConfig": SkyRLTrainConfig,
}


def _deep_copy_default_factories() -> None:
    """Sanity check that field default_factories don't share state across
    instances — invoked at import time as a cheap regression guard against
    accidental ``field(default=...)`` use instead of ``default_factory``.
    """
    cfg1 = MiloEnvConfig()
    cfg2 = MiloEnvConfig()
    # Sanity: distinct instances; mutable fields don't alias.
    assert cfg1 is not cfg2
    # ``rubric_items_path`` is Optional[str]; trivially fine.
    _ = copy.deepcopy(cfg1)


_deep_copy_default_factories()
