"""milo.lht_adapter — Phase 1 of the Milo-bench RL plan.

Wraps SkyRL's ``BaseTextEnv`` + ``SkyRLGymGenerator`` for milo-bench LHT
(version-interval) instances. See ``IMPLEMENTATION_PLAN.md`` v0.4 §1 and
``RL_GYM_SPEC.md`` v0.7 for the design contract. Module map:

  - ``env``                — MiloLHTEnv (BaseTextEnv stub, registry id "milo_lht").
  - ``generator``          — MiloLHTGenerator (rollout driver).
  - ``docker_runtime``     — milo_sandbox context manager + Sandbox facade.
  - ``dataset``            — milo jsonl directory -> SkyRL parquet preprocessor.
  - ``main_milo``          — MiloPPOExp / main() entrypoint.
  - ``config_extensions``  — MiloEnvConfig + MiloTrainConfig dataclasses.
  - ``image_naming``       — canonical ECR image-name helper (pre-existing).
"""

from __future__ import annotations

# Eager side-effect import so ``skyrl_gym.make("milo_lht", ...)`` works as
# soon as anything in this package is imported.
from milo.lht_adapter import env as _env  # noqa: F401
from milo.lht_adapter.image_naming import get_image_name  # re-export for convenience

__all__ = ["get_image_name"]
