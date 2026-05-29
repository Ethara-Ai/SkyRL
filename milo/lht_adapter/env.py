"""MiloLHTEnv — minimal ``BaseTextEnv`` registration for milo-bench instances.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §1.2 and ``RL_GYM_SPEC.md`` v0.7 §1
"system architecture / extension surface" decision (b): the agent loop lives in
``MiloLHTGenerator`` (a ``SkyRLGymGenerator`` subclass) rather than in this env.
This file exists so dataset rows can carry ``env_class="milo_lht"`` and so the
trainer-side ``skyrl_gym.make("milo_lht", ...)`` registry resolution succeeds.
See the module docstring of ``generator.py`` for the actual rollout driver.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.registration import register


__all__ = ["MiloLHTEnv"]


class MiloLHTEnv(BaseTextEnv):
    """Stub env for milo-bench LHT instances.

    The contract here is intentionally tiny because the heavy lifting (docker
    sandbox, multi-turn OpenHands-style agent loop, verifier, reward) lives in
    :class:`milo.lht_adapter.generator.MiloLHTGenerator` per the Phase 1 design
    decision in ``IMPLEMENTATION_PLAN.md`` v0.4 §1 (extension surface = b).

    The constructor matches the SkyRL convention used by every other env in
    ``skyrl_gym.envs.*`` (e.g. ``GSM8kEnv``): ``env_config`` for the dataclass
    knob bag (here the ``MiloEnvConfig`` from :mod:`milo.lht_adapter.config_extensions`)
    and ``extras`` for the per-row task payload threaded through
    ``preprocess_swegym.py``-style dataset preprocessing.

    ``extras["instance"]`` is the canonical milo-bench instance dict (matches
    what :mod:`milo.spike.preprocess_one_milo` and :mod:`milo.lht_adapter.dataset`
    write into the parquet ``instance`` column).
    """

    # Class metadata used by the generator + tests; matches the registry id.
    env_id: str = "milo_lht"

    def __init__(self, env_config: Any = None, extras: Optional[Dict[str, Any]] = None):
        super().__init__()
        # Mirror skyrl_gym convention: ``extras`` defaults to {} but we don't
        # use a mutable default arg (PEP 8 / linter).
        extras = extras or {}

        # ``instance`` is mandatory — the generator can't open a sandbox or run
        # the verifier without it. ``preprocess_swegym.py`` and
        # ``milo.spike.preprocess_one_milo`` both place it on
        # ``row["extra_info"]["instance"]`` which the trainer flattens into
        # ``extras["instance"]`` before reaching us.
        if "instance" not in extras:
            raise ValueError(
                "MiloLHTEnv requires extras['instance'] (the milo-bench task dict). "
                "Check your dataset preprocessor — see milo.lht_adapter.dataset.build_parquet."
            )
        self.instance: Dict[str, Any] = extras["instance"]

        # Stash the env_config (a MiloEnvConfig) for future hooks; for the v0.4
        # stub we only use ``cohort_filter`` informationally.
        self.env_config = env_config

        # max_turns is consulted by SkyRLGymGenerator.agent_loop. Even though
        # MiloLHTGenerator owns the loop and ignores this, set it to a sane
        # large value so any fallback to the base loop doesn't terminate after
        # one step.
        self.max_turns = getattr(env_config, "max_tool_calls", 1000) if env_config else 1000

        # Episode-level counters surfaced via ``get_metrics()``.
        self._step_count = 0
        self._submitted = False

    # ------------------------------------------------------------------
    # Env lifecycle (stubbed)
    # ------------------------------------------------------------------

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """No-op fall-through.

        Returns immediately with ``done=True``, an empty observation, and zero
        reward. The trainer's generator (``MiloLHTGenerator.generate``) never
        actually calls this — it drives the rollout loop directly against the
        inference engine and the docker sandbox. If something *does* call it
        (e.g. a smoke test, a misconfigured run), we want a graceful no-op
        rather than an exception. The ``metadata`` payload includes a marker so
        consumers can detect this stub was invoked.
        """
        self._step_count += 1
        self._submitted = True
        return BaseTextEnvStepOutput(
            observations=[],
            reward=0.0,
            done=True,
            metadata={
                "milo_lht_stub_step": True,
                "instance_id": self.instance.get("instance_id"),
                "step_count": self._step_count,
            },
        )

    def close(self) -> None:
        """Nothing to release at the env level — sandbox lifetime belongs to
        the generator's per-rollout ``milo_sandbox`` context manager.
        """
        # Reset flags so re-use of the same env instance is safe.
        self._step_count = 0
        self._submitted = False

    # ------------------------------------------------------------------
    # Metrics surfaced to the trainer
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return env-level metrics for the episode.

        Real per-rollout metrics (reward decomposition, verifier report,
        invariant violations, cost, tokens) are computed inside the generator
        and surfaced via ``GeneratorOutput.rollout_metrics`` /
        ``env_metrics``. This stub returns an empty dict because the
        :class:`BaseTextEnv` does no real work here — kept explicit so the
        contract is documented.
        """
        return {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
#
# Registering at import time matches the pattern used by skyrl_gym's built-in
# envs (see ``skyrl_gym/envs/__init__.py``: ``register(id="gsm8k", ...)``). We
# can't add a row to ``skyrl_gym/envs/__init__.py`` itself (we never modify
# upstream files per the milo plan), so the registration happens here and we
# rely on this module being imported before ``skyrl_gym.make("milo_lht", ...)``
# is called. The generator imports it transitively; tests import it
# explicitly. If ``register`` is called twice (e.g. in test re-imports) it
# raises ``RegistrationError`` — catch and ignore to keep tests idempotent.

try:
    register(
        id="milo_lht",
        entry_point="milo.lht_adapter.env:MiloLHTEnv",
    )
except Exception:  # pragma: no cover - idempotent re-registration
    # Already registered (e.g. another import path) — fine.
    pass
