"""MiloPPOExp — training entrypoint for milo-bench LHT rollouts.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §1.5: subclass of
:class:`skyrl.train.entrypoints.main_base.BasePPOExp` that overrides
``get_generator`` to return a :class:`MiloLHTGenerator`. The ``main()`` driver
uses :class:`MiloTrainConfig` from :mod:`milo.lht_adapter.config_extensions`
so the user can override our milo-specific knobs via the standard SkyRL
``+environment.skyrl_gym.milo_lht.*=value`` CLI surface.
"""

from __future__ import annotations

import sys

import ray
from loguru import logger

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from milo.lht_adapter.config_extensions import MiloTrainConfig
from milo.lht_adapter.generator import MiloLHTGenerator

# Ensure the env is registered (side-effect import) before any
# skyrl_gym.make("milo_lht") happens downstream.
from milo.lht_adapter import env as _milo_env  # noqa: F401


__all__ = ["MiloPPOExp", "main"]


class MiloPPOExp(BasePPOExp):
    """Milo-flavored PPO experiment.

    Single override: :meth:`get_generator` returns a :class:`MiloLHTGenerator`
    instead of the generic :class:`SkyRLGymGenerator`. Everything else
    (dataset loading, inference engine wiring, trainer setup, eval) is
    inherited from upstream :class:`BasePPOExp` unchanged.
    """

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        logger.info(
            "MiloPPOExp: constructing MiloLHTGenerator (model_path=%s, cohort=%s)",
            self.cfg.trainer.policy.model.path,
            getattr(cfg.environment.skyrl_gym, "milo_lht", None) and cfg.environment.skyrl_gym.milo_lht.cohort_filter,
        )
        return MiloLHTGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )


@ray.remote(num_cpus=1)
def _milo_skyrl_entrypoint(cfg):
    """Mirrors ``BasePPOExp.skyrl_entrypoint`` but uses our subclass."""
    exp = MiloPPOExp(cfg)
    exp.run()


def main() -> None:
    """CLI entrypoint.

    Usage::

        uv run --isolated --extra fsdp \\
          -m milo.lht_adapter.main_milo \\
          data.train_data="['/path/to/milo_train.parquet']" \\
          trainer.policy.model.path=Qwen/Qwen2.5-Coder-32B-Instruct \\
          +environment.skyrl_gym.milo_lht.cohort_filter=A \\
          +environment.skyrl_gym.milo_lht.cost_guardrail_usd=5.0
    """
    cfg = MiloTrainConfig.from_cli_overrides(sys.argv[1:])

    # validate_cfg expects the upstream type; MiloTrainConfig is a subclass
    # so this is fine.
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(_milo_skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":  # pragma: no cover
    main()
