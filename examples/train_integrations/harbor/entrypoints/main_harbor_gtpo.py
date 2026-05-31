"""
Main entrypoint for training on Harbor tasks with GTPO.

GTPO (Group Turn Policy Optimization) extends GRPO with:
1. Turn-level reward assignment
2. Return-based advantage with temporal discounting (gamma=0.9)
3. Self-supervised reward shaping for failed trajectories

Paper: https://arxiv.org/abs/2511.14846

Usage:
    uv run --isolated --extra fsdp -m examples.train_integrations.harbor.entrypoints.main_harbor_gtpo \
        trainer.algorithm.adv_estimator=gtpo \
        trainer.algorithm.gamma=0.9 \
        generator.gtpo_turn_rewards=true \
        generator.gtpo_format_penalty=-0.1 \
        ...
"""

import sys

import ray
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.config import SkyRLTrainConfig, GeneratorConfig, get_config_as_yaml_str
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry
from ..harbor_generator import HarborGenerator
from ..dataset import HarborTaskDataset

HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class HarborGTPOGeneratorConfig(GeneratorConfig):
    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    gtpo_turn_rewards: bool = True
    gtpo_format_penalty: float = -0.1
    step_wise_trajectories: bool = True
    merge_stepwise_output: bool = True


@dataclass
class HarborGTPOConfig(SkyRLTrainConfig):
    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGTPOGeneratorConfig = field(default_factory=HarborGTPOGeneratorConfig)


class HarborGTPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        prompts_dataset = HarborTaskDataset(data_files=self.cfg.data.train_data)
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return HarborTaskDataset(data_files=self.cfg.data.val_data)
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = HarborGTPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborGTPOConfig.from_cli_overrides(sys.argv[1:])

    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    # Force adv_estimator=gtpo — this is the GTPO entrypoint.
    from loguru import logger
    if cfg.trainer.algorithm.adv_estimator != "gtpo":
        logger.info(f"GTPO entrypoint: setting adv_estimator=gtpo (was '{cfg.trainer.algorithm.adv_estimator}')")
        cfg.trainer.algorithm.adv_estimator = "gtpo"

    # Set gamma=0.9 only if user didn't explicitly override via CLI.
    # The AlgorithmConfig default is 1.0. We check if it's still the default.
    # If user explicitly passed gamma=1.0, they get a warning but we respect it.
    cli_args = set(sys.argv[1:])
    gamma_explicitly_set = any("gamma=" in arg for arg in cli_args)
    if not gamma_explicitly_set and cfg.trainer.algorithm.gamma == 1.0:
        logger.info("GTPO entrypoint: setting gamma=0.9 (paper optimal, override with trainer.algorithm.gamma=X)")
        cfg.trainer.algorithm.gamma = 0.9
    elif gamma_explicitly_set and cfg.trainer.algorithm.gamma == 1.0:
        logger.warning("GTPO: gamma=1.0 disables temporal discounting (equivalent to no turn-level advantage)")

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training."
        )

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
