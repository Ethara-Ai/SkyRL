"""Five customization Protocols — Phase 15 / spec §29.

Every Protocol is `runtime_checkable` so integrators can assert their
implementation conforms without inheriting from a base class. The shipped
default implementations live in:

    TrainerStack         → SkyRL's main_base (no Ethara override needed)
    TrainerAlgo          → milo/algos/grpo_wrapper.py (register_milo_grpo)
    ServingAdapter       → milo/serving/ (vllm_config.py + hot_reload.py)
    ToolCallParser       → milo/adapters/tool_call_parsers.py
    ObservabilityBackend → milo/observability/ (alarms + W&B dashboards)

Each Protocol surface is the *contract* — the actual code lives in the
named modules. Integrators implement these Protocols against their own
backends and either subclass our defaults or wire in via the relevant
registry (see `milo/adapters/registry.py` for the pattern).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# ----------------------------------------------------------------- TrainerStack


@runtime_checkable
class TrainerStack(Protocol):
    """Owns the rollout loop, loss computation, distributed training, and
    checkpointing.

    SkyRL's `BasePPOExp` already satisfies this Protocol implicitly through
    its `_setup_trainer()` method; the Protocol here is for AGIF-side custom
    stacks (e.g. an integrator who wants to plug verl or OpenRLHF in
    instead of SkyRL's native trainer).
    """

    def setup(self, config: Any) -> None:
        """Initialise model loading, FSDP/Megatron sharding, optimizer, scheduler."""
        ...

    def rollout_batch(self, tasks: list[str]) -> list[Any]:
        """Drive one batch of rollouts; return a list of Trajectory dicts."""
        ...

    def compute_advantages(self, trajs: list[Any]) -> Any:
        """Return advantage tensors for the trainer to consume."""
        ...

    def policy_update(self, trajs: list[Any], advs: Any) -> dict[str, float]:
        """Run one optimizer step. Return per-batch metrics for logging."""
        ...

    def checkpoint_save(self, path: str) -> None: ...
    def checkpoint_load(self, path: str) -> None: ...


# ------------------------------------------------------------------ TrainerAlgo


@runtime_checkable
class TrainerAlgo(Protocol):
    """The advantage estimator + policy-loss combo.

    Default: `milo_grpo` (registered in `milo/algos/grpo_wrapper.py:register_milo_grpo`).
    Integrators ship their own by registering with
    `skyrl.backends.skyrl_train.utils.ppo_utils.register_advantage_estimator`
    and `register_policy_loss`.
    """

    def compute_advantages(self, trajectories: list[Any]) -> Any: ...
    def compute_loss(
        self, trajectories: list[Any], advantages: Any, ref_logprobs: Any
    ) -> Any: ...


# --------------------------------------------------------------- ServingAdapter


@runtime_checkable
class ServingAdapter(Protocol):
    """Wraps an inference server.

    v0.7: only the policy server runs live during RL; the reference is an
    offline S3 logprobs cache (see `milo/algos/reference_cache.py`). This
    Protocol still describes the 'live reference' case for integrators who
    deliberately want one.
    """

    def start(
        self,
        model_path: str,
        port: int,
        tensor_parallel: int,
        max_model_length: int,
        **backend_specific_kwargs: Any,
    ) -> None: ...

    def stop(self) -> None: ...
    def health(self) -> bool: ...

    def reload_weights(self, new_weights_path: str) -> None:
        """Hot-reload pattern from spec §18.3. Policy server only — reference
        adapter instances (if any) raise NotImplementedError. v0.4 default
        reference path uses no live server."""
        ...

    @property
    def openai_compatible_url(self) -> str:
        """The base URL the policy adapter points at. Always OpenAI-schema-compatible."""
        ...


# ---------------------------------------------------------------- ToolCallParser


@runtime_checkable
class ToolCallParser(Protocol):
    """Stateless parser. Default impls in `milo/adapters/tool_call_parsers.py`:
    OpenAIFunctionsParser, QwenToolCallParser, LlamaToolCallParser.
    """

    def parse(self, model_output: str) -> list[Any]: ...


# ----------------------------------------------------------- ObservabilityBackend


@runtime_checkable
class ObservabilityBackend(Protocol):
    """Records metrics + fires alarms. Default impl uses W&B + milo's
    alarm classes (`milo/observability/alarms.py`).

    Integrators wire in their own metrics backend by implementing this
    Protocol; the milo trainer calls `record_metrics()` per step and
    `fire_alarm()` when an alarm condition matches.
    """

    def record_metrics(self, step: int, metrics: dict[str, Any]) -> None: ...
    def fire_alarm(self, alarm_name: str, payload: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...
