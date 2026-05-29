"""MiloLHTGenerator — agent-loop driver for milo-bench LHT rollouts.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §1.4 (the rollout driver) and the
trajectory contract from ``RL_GYM_SPEC.md`` v0.7 §4–§6 (observation / action /
reward / step record / trace). Subclasses :class:`SkyRLGymGenerator` and
overrides ``generate()`` to: (1) open a per-rollout docker sandbox via
:func:`milo.lht_adapter.docker_runtime.milo_sandbox`, (2) drive an OpenHands-
style ``ActionEvent``/``ObservationEvent`` history against the inference
engine HTTP endpoint (matches the on-disk milo-bench trajectory schema), (3)
enforce termination conditions, (4) hand off to the verifier + reward
subsystems (stubbed via try/except ImportError per the v0.4 plan).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    BatchMetadata,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.utils import (
    get_response_ids_and_loss_mask_from_messages,
    get_rollout_metrics,
)

# Ensure the env is registered before any skyrl_gym.make("milo_lht") call.
from milo.lht_adapter import env as _milo_env  # noqa: F401 -- side-effect: register

from milo.lht_adapter.docker_runtime import Sandbox, SandboxError, milo_sandbox


__all__ = [
    "MiloLHTGenerator",
    "RolloutResult",
    "ToolCall",
    "Observation",
    "Budgets",
    "TerminationReason",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional integrations (Phase 2/4/5/6 stubs)
# ---------------------------------------------------------------------------
#
# These subsystems aren't implemented yet (per ``IMPLEMENTATION_PLAN.md`` v0.4
# the responsible Phases haven't shipped). We import behind try/except so the
# generator is runnable in stub mode: reward defaults to 0, trace is still
# written. When the modules land, the imports start succeeding and the
# generator picks them up with no code change here.

# TODO: integrate once Phase 2 (verifier) is done — milo.verifier.three_run.ThreeRunVerifier
try:
    from milo.verifier import three_run as _milo_verifier  # type: ignore
except ImportError:
    _milo_verifier = None  # type: ignore

# TODO: integrate once Phase 4 (reward aggregator) is done — milo.reward.composite.aggregate
try:
    from milo.reward import composite as _milo_reward  # type: ignore
except ImportError:
    _milo_reward = None  # type: ignore

# TODO: integrate once Phase 5 (logging) is done — milo.logging.trace_writer.TraceWriter
try:
    from milo.logging import trace_writer as _milo_trace  # type: ignore
except ImportError:
    _milo_trace = None  # type: ignore

# TODO: integrate once Phase 6 (invariants) is done — milo.invariants.checks.run_all
try:
    from milo.invariants import checks as _milo_invariants  # type: ignore
except ImportError:
    _milo_invariants = None  # type: ignore


# ---------------------------------------------------------------------------
# Internal dataclasses (mirror RL_GYM_SPEC v0.7 §6)
# ---------------------------------------------------------------------------


# Tool name -> handler closure shape; the generator uses a tiny set of
# canonical names mirrored on the spec's §4.2 six-tool surface.
TOOL_NAMES = {
    "read_file",
    "list_files",
    "search_grep",
    "apply_patch",
    "run_command",
    "submit",
}


@dataclass
class ToolCall:
    """One tool call (spec §6.3)."""

    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str = ""  # the original model output for the trace

    def to_event(self, source: str = "agent") -> Dict[str, Any]:
        """Render as an OpenHands-style ``ActionEvent`` dict."""
        return {
            "id": self.tool_call_id,
            "timestamp": time.time(),
            "source": source,
            "kind": "ActionEvent",
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_call": {"name": self.tool_name, "arguments": self.arguments},
            "action": self.raw_text,
            "thought": "",
            "reasoning_content": "",
            "thinking_blocks": [],
            "responses_reasoning_item": None,
            "llm_response_id": "",
            "security_risk": None,
            "critic_result": None,
            "summary": "",
        }


@dataclass
class Observation:
    """Result of executing one tool call (spec §6.2)."""

    step: int
    tool_call_id: str
    tool_name: str
    stdout: str
    stderr: str
    exit_code: Optional[int]
    truncated: bool = False
    truncation_marker: Optional[str] = None
    full_output_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    remaining_budget_seconds: float = 0.0
    remaining_tool_calls: int = 0
    shaping_reward_this_step: float = 0.0

    def to_event(self, action_id: str) -> Dict[str, Any]:
        """Render as an OpenHands-style ``ObservationEvent`` dict."""
        body = self.stdout if self.stdout else ""
        if self.stderr:
            body = body + ("\n--- stderr ---\n" + self.stderr if body else self.stderr)
        return {
            "id": f"obs_{self.tool_call_id}",
            "timestamp": time.time(),
            "source": "environment",
            "kind": "ObservationEvent",
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "observation": body,
            "action_id": action_id,
        }


@dataclass
class Budgets:
    """Per-rollout budgets (spec §4.5)."""

    max_episode_seconds: int
    max_tool_calls: int
    started_at: float = field(default_factory=time.monotonic)
    tool_calls_used: int = 0
    cost_usd_used: float = 0.0
    cost_guardrail_usd: float = 0.0  # 0 disables
    consecutive_no_edit: int = 0
    consecutive_no_edit_terminate: int = 50

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_episode_seconds - self.elapsed_seconds)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls_used)


class TerminationReason:
    """Termination-reason string constants (spec §6.5 terminal_summary).
    Not an Enum so it round-trips through JSON without serialisation tricks.
    """

    SUBMIT = "submit"
    TIMEOUT = "timeout"
    TOOL_BUDGET = "tool_budget"
    CONTAINER_ERROR = "container_error"
    COST_GUARDRAIL = "cost_guardrail"
    NO_EDIT_LOOP = "consecutive_no_edit"
    PARSE_FAIL_LOOP = "parse_fail_loop"


@dataclass
class RolloutResult:
    """Aggregate per-rollout output passed back to ``generate()``."""

    trajectory_id: TrajectoryID
    instance_id: str
    messages: List[Dict[str, Any]]  # for token-id derivation
    history: List[Dict[str, Any]]  # OpenHands-style event list (trace)
    reward: float
    reward_decomposition: Dict[str, Any]
    termination_reason: str
    cost_usd: float
    tokens_prompt: int
    tokens_completion: int
    error: Optional[str] = None
    trace_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool-call parser
# ---------------------------------------------------------------------------


def _parse_tool_call(model_output: str) -> Optional[ToolCall]:
    """Extract a tool call from a model response.

    The reference implementation supports two surface formats so the
    generator can drive both:

      1. **OpenAI-style JSON tool_calls** — the model wraps its tool call in
         a fenced ``json`` block matching the spec §6.3 schema::

             ```json
             {"tool_name": "read_file", "arguments": {"path": "/workspace/foo.py"}}
             ```

      2. **mini_swe_agent-style bash block** — a fenced ``bash`` block; we
         lift it as a ``run_command`` call (matches the on-disk milo-bench
         claude_opus traces which use bash one-liners).

    Returns ``None`` if the output contains no parseable tool call (the
    caller treats this as a malformed step and bumps ``parse_fail`` counter).
    """
    import re

    # First: JSON block.
    m = re.search(r"```json\s*\n(.*?)\n```", model_output, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            tool = obj.get("tool_name") or obj.get("name")
            args = obj.get("arguments") or obj.get("args") or {}
            if tool in TOOL_NAMES:
                return ToolCall(
                    tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
                    tool_name=tool,
                    arguments=args,
                    raw_text=model_output,
                )
        except json.JSONDecodeError:
            pass

    # Second: bash block (mini_swe_agent convention).
    m = re.search(r"```bash\s*\n(.*?)\n```", model_output, re.DOTALL)
    if m:
        cmd = m.group(1).strip()
        if not cmd:
            return None
        # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is the mini_swe submit sentinel.
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
            return ToolCall(
                tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
                tool_name="submit",
                arguments={"summary": ""},
                raw_text=model_output,
            )
        return ToolCall(
            tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
            tool_name="run_command",
            arguments={"cmd": cmd, "timeout": 300},
            raw_text=model_output,
        )

    return None


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


def _execute_tool(sandbox: Sandbox, tool: ToolCall, step: int, budgets: Budgets) -> Observation:
    """Run a single tool call against the sandbox.

    Each branch maps to one of the spec §4.2 tools. ``submit`` is a sentinel
    — we don't actually execute anything; the rollout loop drains the
    candidate patch via ``sandbox.git_diff()`` after seeing this call.
    """
    t0 = time.monotonic()
    args = tool.arguments

    if tool.tool_name == "submit":
        return Observation(
            step=step,
            tool_call_id=tool.tool_call_id,
            tool_name=tool.tool_name,
            stdout="<submitted>",
            stderr="",
            exit_code=0,
            elapsed_seconds=time.monotonic() - t0,
            remaining_budget_seconds=budgets.remaining_seconds,
            remaining_tool_calls=budgets.remaining_tool_calls,
        )

    if tool.tool_name == "run_command":
        cmd = args.get("cmd", "")
        timeout = int(args.get("timeout") or 300)
        cwd = args.get("cwd") or None
        r = sandbox.exec(cmd, timeout=timeout, cwd=cwd)
        return Observation(
            step=step,
            tool_call_id=tool.tool_call_id,
            tool_name=tool.tool_name,
            stdout=_truncate(r.output, 32 * 1024),
            stderr="",
            exit_code=r.returncode,
            truncated=len(r.output) > 32 * 1024,
            elapsed_seconds=r.elapsed_s,
            remaining_budget_seconds=budgets.remaining_seconds,
            remaining_tool_calls=budgets.remaining_tool_calls,
        )

    if tool.tool_name == "read_file":
        path = args.get("path", "")
        try:
            content = sandbox.read_file(path)
            return Observation(
                step=step,
                tool_call_id=tool.tool_call_id,
                tool_name=tool.tool_name,
                stdout=_truncate(content, 32 * 1024),
                stderr="",
                exit_code=0,
                truncated=len(content) > 32 * 1024,
                elapsed_seconds=time.monotonic() - t0,
                remaining_budget_seconds=budgets.remaining_seconds,
                remaining_tool_calls=budgets.remaining_tool_calls,
            )
        except FileNotFoundError:
            return Observation(
                step=step,
                tool_call_id=tool.tool_call_id,
                tool_name=tool.tool_name,
                stdout="",
                stderr=f"file not found: {path}",
                exit_code=2,
                elapsed_seconds=time.monotonic() - t0,
                remaining_budget_seconds=budgets.remaining_seconds,
                remaining_tool_calls=budgets.remaining_tool_calls,
            )

    if tool.tool_name == "list_files":
        path = args.get("path", "")
        recursive = bool(args.get("recursive", False))
        max_depth = int(args.get("max_depth", 3))
        flag = "-R" if recursive else ""
        cmd = f"find {path} -maxdepth {max_depth} {flag} 2>/dev/null | head -n 5000"
        r = sandbox.exec(cmd, timeout=60)
        return _exec_result_to_obs(r, tool, step, budgets)

    if tool.tool_name == "search_grep":
        pattern = args.get("pattern", "")
        scope = args.get("scope", sandbox.workdir)
        file_glob = args.get("file_glob", "")
        max_results = int(args.get("max_results", 200))
        rg_glob = f"--glob '{file_glob}'" if file_glob else ""
        cmd = (
            f"rg --line-number {rg_glob} --max-count {max_results} "
            f"-- {json.dumps(pattern)} {scope} 2>/dev/null | head -n {max_results}"
        )
        r = sandbox.exec(cmd, timeout=60)
        return _exec_result_to_obs(r, tool, step, budgets)

    if tool.tool_name == "apply_patch":
        diff = args.get("diff", "")
        r = sandbox.apply_patch(diff)
        return _exec_result_to_obs(r, tool, step, budgets)

    # Unknown tool — treat as parse failure.
    return Observation(
        step=step,
        tool_call_id=tool.tool_call_id,
        tool_name=tool.tool_name,
        stdout="",
        stderr=f"unknown tool: {tool.tool_name}",
        exit_code=-1,
        elapsed_seconds=time.monotonic() - t0,
        remaining_budget_seconds=budgets.remaining_seconds,
        remaining_tool_calls=budgets.remaining_tool_calls,
    )


def _exec_result_to_obs(r, tool: ToolCall, step: int, budgets: Budgets) -> Observation:
    return Observation(
        step=step,
        tool_call_id=tool.tool_call_id,
        tool_name=tool.tool_name,
        stdout=_truncate(r.output, 32 * 1024),
        stderr="",
        exit_code=r.returncode,
        truncated=len(r.output) > 32 * 1024,
        elapsed_seconds=r.elapsed_s,
        remaining_budget_seconds=budgets.remaining_seconds,
        remaining_tool_calls=budgets.remaining_tool_calls,
    )


def _truncate(s: str, max_len: int) -> str:
    """Head+tail truncation (spec §4.1)."""
    if len(s) <= max_len:
        return s
    keep = max_len // 2
    elided = len(s) - 2 * keep
    return s[:keep] + f"\n<... truncated {elided} bytes ...>\n" + s[-keep:]


# ---------------------------------------------------------------------------
# Termination check
# ---------------------------------------------------------------------------


def _check_termination(
    history: List[Dict[str, Any]],
    budgets: Budgets,
    last_tool_was_submit: bool,
    parse_fail_streak: int,
) -> Optional[str]:
    """Return a TerminationReason string if we should stop, else None."""
    if last_tool_was_submit:
        return TerminationReason.SUBMIT
    if budgets.remaining_seconds <= 0:
        return TerminationReason.TIMEOUT
    if budgets.remaining_tool_calls <= 0:
        return TerminationReason.TOOL_BUDGET
    if (
        budgets.cost_guardrail_usd > 0
        and budgets.cost_usd_used >= budgets.cost_guardrail_usd
    ):
        return TerminationReason.COST_GUARDRAIL
    if budgets.consecutive_no_edit >= budgets.consecutive_no_edit_terminate:
        return TerminationReason.NO_EDIT_LOOP
    if parse_fail_streak >= 10:
        return TerminationReason.PARSE_FAIL_LOOP
    return None


# ---------------------------------------------------------------------------
# Stub reward + verifier (used when Phase 2/4 modules absent)
# ---------------------------------------------------------------------------


def _stub_verifier_report(instance: Dict[str, Any], candidate_patch: str) -> Dict[str, Any]:
    """Synthesize an empty verifier report so the reward aggregator has a
    well-formed input even in stub mode. Always reports failure (0/0) so
    R_terminal=0 — accurate, since we didn't actually run any tests.
    """
    f2p = instance.get("milo_f2p_test_ids") or []
    p2p = instance.get("milo_p2p_test_ids") or []
    return {
        "instance_id": instance.get("instance_id"),
        "baseline_result": {"passed": [], "failed": [], "skipped": []},
        "test_patch_result": {"passed": [], "failed": [], "skipped": []},
        "fix_patch_result": {"passed": [], "failed": [], "skipped": []},
        "f2p_passed": [],
        "f2p_failed": list(f2p),
        "p2p_regressed": [],
        "test_count_nonzero": False,
        "passes_invariant_check": True,
        "invariant_violations": [],
        "elapsed_total_s": 0.0,
        "stub": True,
    }


def _stub_reward(
    verifier_report: Dict[str, Any],
    rubric_report: Optional[Dict[str, Any]],
    shaping_rewards: List[float],
    cfg_alpha: float,
    cfg_beta: float,
    cfg_lambda: float,
    cfg_gamma: float,
    tir_rewards: List[float],
) -> Tuple[float, Dict[str, Any]]:
    """Trivial reward aggregator used while Phase 4 is unbuilt.

    Returns 0.0 when running in stub mode (verifier didn't actually run);
    decomposition mirrors the spec §6.8 shape so downstream consumers see a
    well-formed record either way.
    """
    r_terminal = 0
    r_delta_sum = sum(shaping_rewards)
    r_rubric_mean = 0.0
    if rubric_report and rubric_report.get("per_item"):
        items = rubric_report["per_item"]
        r_rubric_mean = sum(items) / max(1, len(items))
    r_tir_sum = sum(tir_rewards)

    r_total = (
        float(r_terminal)
        + cfg_alpha * r_delta_sum
        + cfg_beta * r_rubric_mean
        + cfg_gamma * r_tir_sum
    )
    decomp = {
        "preset": "composite",
        "r_terminal": r_terminal,
        "r_delta_steps": shaping_rewards,
        "r_delta_sum": r_delta_sum,
        "r_rubric_per_item": rubric_report.get("per_item", []) if rubric_report else [],
        "r_rubric_mean": r_rubric_mean,
        "r_tir_steps": tir_rewards,
        "r_tir_sum": r_tir_sum,
        "alpha": cfg_alpha,
        "beta": cfg_beta,
        "lambda": cfg_lambda,
        "gamma": cfg_gamma,
        "r_total": r_total,
        "components": {
            "terminal": float(r_terminal),
            "shaping": cfg_alpha * r_delta_sum,
            "rubric": cfg_beta * r_rubric_mean,
            "tir": cfg_gamma * r_tir_sum,
        },
        "stub": True,
    }
    return r_total, decomp


def _write_trace_stub(
    trace_root: Path,
    trajectory_id: TrajectoryID,
    history: List[Dict[str, Any]],
    terminal_summary: Dict[str, Any],
) -> Path:
    """Minimal local trace writer used until Phase 5 ships ``milo.logging``.

    Lays down the per-rollout directory layout from RL_GYM_SPEC §5.5 but only
    populates ``trace.jsonl`` + the ``trace.completed`` sentinel. Phase 5
    will replace this with the full layout (verifier/, judge/, container/, ...).
    """
    rollout_dir = trace_root / trajectory_id.to_string()
    rollout_dir.mkdir(parents=True, exist_ok=True)

    trace_path = rollout_dir / "trace.jsonl"
    with trace_path.open("w") as f:
        # Trace header.
        f.write(
            json.dumps(
                {
                    "schema_version": "milo-trace/0.1-stub",
                    "trajectory_id": trajectory_id.to_string(),
                    "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )
        for event in history:
            f.write(json.dumps(event, default=str) + "\n")
        f.write(json.dumps(terminal_summary, default=str) + "\n")
    # Atomic-completion sentinel.
    (rollout_dir / "trace.completed").write_text("")
    return trace_path


# ---------------------------------------------------------------------------
# The Generator class
# ---------------------------------------------------------------------------


class MiloLHTGenerator(SkyRLGymGenerator):
    """Drop-in replacement for :class:`SkyRLGymGenerator` that owns the
    milo-bench LHT rollout loop.

    Wiring matches ``examples/train/mini_swe_agent/main_mini_swe.py`` — the
    trainer constructs an instance via :class:`milo.lht_adapter.main_milo.MiloPPOExp.get_generator`,
    which forwards the generator + skyrl_gym config + the inference engine
    client. The HTTP endpoint of the inference engine is the seam through
    which we talk to the policy (matches ``MiniSweAgentGenerator``).
    """

    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client,
        tokenizer,
        model_name: str,
    ) -> None:
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)
        self.generator_cfg = generator_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + model_name

        # The milo env config lives at skyrl_gym_cfg.milo_lht (per
        # MiloSkyRLGymConfig); fall back to defaults if not present (unit tests
        # may construct with bare SkyRLGymConfig).
        self.milo_env_cfg = getattr(skyrl_gym_cfg, "milo_lht", None)
        if self.milo_env_cfg is None:
            from milo.lht_adapter.config_extensions import MiloEnvConfig

            logger.warning(
                "MiloLHTGenerator: skyrl_gym_cfg has no .milo_lht; using MiloEnvConfig defaults"
            )
            self.milo_env_cfg = MiloEnvConfig()

        # Inference engine HTTP endpoint (mirrors MiniSweAgentGenerator).
        self.http_host = generator_cfg.inference_engine.http_endpoint_host
        self.http_port = generator_cfg.inference_engine.http_endpoint_port
        self.base_url = f"http://{self.http_host}:{self.http_port}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Drive ``len(prompts)`` rollouts in parallel.

        Mirrors ``MiniSweAgentGenerator.generate`` structure: ``asyncio.gather``
        over per-rollout tasks, post-process into the trainer's expected dicts.
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"] or [{}] * len(prompts)
        trajectory_ids = input_batch.get("trajectory_ids") or [
            TrajectoryID(instance_id=str(i), repetition_id=0) for i in range(len(prompts))
        ]
        batch_metadata = input_batch.get("batch_metadata") or BatchMetadata(
            global_step=0, training_phase="train"
        )

        tasks = [
            self._rollout_one(
                prompt=prompts[i],
                env_extra=env_extras[i],
                trajectory_id=trajectory_ids[i],
                batch_metadata=batch_metadata,
            )
            for i in range(len(prompts))
        ]
        results: List[RolloutResult] = await asyncio.gather(*tasks, return_exceptions=False)

        # Convert into GeneratorOutput. Mirrors MiniSweAgentGenerator's
        # tokenization path.
        prompt_token_ids: List[List[int]] = []
        response_ids: List[List[int]] = []
        rewards: List[float] = []
        loss_masks: List[List[int]] = []
        stop_reasons: List[str] = []
        env_metrics: List[Dict[str, Any]] = []
        trajectory_ids_out: List[TrajectoryID] = []

        for r in results:
            if not r.messages:
                # Failed rollout — skip rather than crash (matches
                # MiniSweAgentGenerator behavior).
                continue
            # Tokenize. Use the first two as the prompt (system + user) per
            # the same hardcoding mini_swe_agent applies.
            if len(r.messages) < 2:
                continue
            initial = self.tokenizer.apply_chat_template(
                r.messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True
            )
            resp_msgs = r.messages[2:]
            resp_ids, mask, _ = get_response_ids_and_loss_mask_from_messages(
                resp_msgs, self.tokenizer, assistant_logprobs=None
            )
            prompt_token_ids.append(initial)
            response_ids.append(resp_ids)
            rewards.append(r.reward)
            loss_masks.append(mask)
            stop_reasons.append(r.termination_reason)
            env_metrics.append(
                {
                    "instance_id": r.instance_id,
                    "termination_reason": r.termination_reason,
                    "cost_usd": r.cost_usd,
                    "tokens_prompt": r.tokens_prompt,
                    "tokens_completion": r.tokens_completion,
                    "reward_decomposition": r.reward_decomposition,
                    "trace_path": r.trace_path,
                    "error": r.error,
                }
            )
            trajectory_ids_out.append(r.trajectory_id)

        if not response_ids:
            raise ValueError(
                "MiloLHTGenerator: no successful rollouts in batch — "
                "check sandbox bring-up + inference engine HTTP endpoint."
            )

        rollout_metrics = get_rollout_metrics(response_ids, rewards)

        output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            "trajectory_ids": trajectory_ids_out,
            "rollout_expert_indices": None,
            "is_last_step": None,
            "env_metrics": env_metrics,
            "pixel_values": None,
            "image_grid_thw": None,
        }
        return output

    # ------------------------------------------------------------------
    # Per-rollout driver
    # ------------------------------------------------------------------

    async def _rollout_one(
        self,
        prompt: List[Dict[str, str]],
        env_extra: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> RolloutResult:
        """Run one rollout end-to-end. Async wrapper around a sync core
        (the docker exec path is sync-only; we offload via the default
        executor to keep generate() truly concurrent).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._rollout_one_sync,
            prompt,
            env_extra,
            trajectory_id,
            batch_metadata,
        )

    def _rollout_one_sync(
        self,
        prompt: List[Dict[str, str]],
        env_extra: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> RolloutResult:
        instance = env_extra.get("instance")
        if not instance:
            return self._failed_result(
                trajectory_id, "", "missing extras['instance']", prompt
            )

        instance_id = instance.get("instance_id", "unknown")
        cfg = self.milo_env_cfg
        budgets = Budgets(
            max_episode_seconds=cfg.max_episode_seconds,
            max_tool_calls=cfg.max_tool_calls,
            cost_guardrail_usd=cfg.cost_guardrail_usd,
            consecutive_no_edit_terminate=cfg.consecutive_no_edit_terminate,
        )

        # OpenHands-style history (matches the on-disk trajectory schema we
        # saw in milo-bench/trajectories/.../output.jsonl).
        history: List[Dict[str, Any]] = []
        messages: List[Dict[str, str]] = list(prompt)
        shaping_rewards: List[float] = []
        tir_rewards: List[float] = []
        parse_fail_streak = 0
        last_tool_was_submit = False
        termination_reason: Optional[str] = None
        error: Optional[str] = None
        candidate_patch: str = ""

        # Try to open a sandbox; if that fails we early-out with a structured
        # failure result (this is the "container_error" path per spec §4.5).
        sandbox_cm = self._open_sandbox(instance)
        try:
            with sandbox_cm as sandbox:
                # Seed history with the system + user messages (OpenHands
                # SystemPromptEvent + MessageEvent equivalents).
                history.append(self._system_event(instance, cfg))
                for msg in prompt:
                    history.append(self._message_event(msg, sender="user"))

                while termination_reason is None:
                    # 1. Ask the policy for the next action.
                    try:
                        response_text = self._invoke_policy(messages, instance_id)
                    except Exception as e:
                        logger.warning(
                            "rollout %s: policy call failed: %s", trajectory_id.to_string(), e
                        )
                        error = f"policy call failed: {e}"
                        termination_reason = TerminationReason.CONTAINER_ERROR
                        break

                    messages.append({"role": "assistant", "content": response_text})

                    # 2. Parse a tool call.
                    tool = _parse_tool_call(response_text)
                    if tool is None:
                        parse_fail_streak += 1
                        tir_rewards.append(-1.0)
                        budgets.tool_calls_used += 1
                        # Add a synthetic user-side error nudge so the model
                        # can recover next turn.
                        nudge = (
                            "Your last response could not be parsed as a tool call. "
                            "Please return either a single ```json``` block with "
                            "{tool_name, arguments} or a single ```bash``` block."
                        )
                        messages.append({"role": "user", "content": nudge})
                        termination_reason = _check_termination(
                            history, budgets, False, parse_fail_streak
                        )
                        continue

                    parse_fail_streak = 0
                    tir_rewards.append(0.0)
                    history.append(tool.to_event(source="agent"))

                    # 3. Execute the tool.
                    pre_diff = sandbox.git_diff() if tool.tool_name != "submit" else ""
                    obs = _execute_tool(sandbox, tool, step=len(history), budgets=budgets)
                    history.append(obs.to_event(action_id=tool.tool_call_id))
                    messages.append(
                        {"role": "user", "content": self._render_obs(obs)}
                    )
                    budgets.tool_calls_used += 1
                    shaping_rewards.append(0.0)  # Phase 4 will compute real R_delta

                    # 4. Edit-tracking for the consecutive-no-edit guard.
                    if tool.tool_name == "submit":
                        last_tool_was_submit = True
                        candidate_patch = sandbox.git_diff()
                    else:
                        post_diff = sandbox.git_diff()
                        if post_diff == pre_diff:
                            budgets.consecutive_no_edit += 1
                        else:
                            budgets.consecutive_no_edit = 0

                    # 5. Termination check.
                    termination_reason = _check_termination(
                        history, budgets, last_tool_was_submit, parse_fail_streak
                    )

                # If we exited the loop without a submit, capture the
                # current diff as the candidate patch (force-submit path).
                if not candidate_patch:
                    try:
                        candidate_patch = sandbox.git_diff()
                    except Exception:
                        candidate_patch = ""
        except SandboxError as e:
            logger.warning("rollout %s: sandbox error: %s", trajectory_id.to_string(), e)
            error = str(e)
            termination_reason = TerminationReason.CONTAINER_ERROR
        except Exception as e:
            logger.exception("rollout %s: unexpected error: %s", trajectory_id.to_string(), e)
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            termination_reason = TerminationReason.CONTAINER_ERROR

        # 6. Verifier — stubbed.
        if _milo_verifier is not None and termination_reason != TerminationReason.CONTAINER_ERROR:
            # TODO: integrate once Phase 2 (verifier) is done.
            try:
                report = _milo_verifier.run(instance, candidate_patch)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("verifier failed: %s; falling back to stub", e)
                report = _stub_verifier_report(instance, candidate_patch)
        else:
            report = _stub_verifier_report(instance, candidate_patch)

        # 7. Reward aggregation — stubbed.
        rubric_report = None  # TODO: integrate once Phase 3 (judge) is done.
        if _milo_reward is not None:
            try:
                reward, decomp = _milo_reward.aggregate(  # type: ignore[attr-defined]
                    verifier_report=report,
                    rubric_report=rubric_report,
                    shaping_rewards=shaping_rewards,
                    tir_rewards=tir_rewards,
                    alpha=cfg.reward_alpha,
                    beta=cfg.reward_beta,
                    lambda_=cfg.reward_lambda,
                    gamma=cfg.reward_gamma,
                )
            except Exception as e:
                logger.warning("reward aggregation failed: %s; falling back to stub", e)
                reward, decomp = _stub_reward(
                    report,
                    rubric_report,
                    shaping_rewards,
                    cfg.reward_alpha,
                    cfg.reward_beta,
                    cfg.reward_lambda,
                    cfg.reward_gamma,
                    tir_rewards,
                )
        else:
            reward, decomp = _stub_reward(
                report,
                rubric_report,
                shaping_rewards,
                cfg.reward_alpha,
                cfg.reward_beta,
                cfg.reward_lambda,
                cfg.reward_gamma,
                tir_rewards,
            )

        # 8. Trace.
        terminal_summary = {
            "kind": "terminal_summary",
            "ended_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "termination_reason": termination_reason or TerminationReason.SUBMIT,
            "verifier_report": report,
            "rubric_report": rubric_report,
            "reward_decomposition": decomp,
            "r_total": reward,
            "candidate_patch": candidate_patch,
            "cost_usd": budgets.cost_usd_used,
            "tokens": {"prompt": 0, "completion": 0},
            "error": error,
        }
        trace_path: Optional[str] = None
        try:
            trace_root = Path(cfg.trace_root)
            if _milo_trace is not None:
                # TODO: integrate once Phase 5 (logging) is done.
                trace_path = str(
                    _milo_trace.write(  # type: ignore[attr-defined]
                        trace_root=trace_root,
                        trajectory_id=trajectory_id,
                        history=history,
                        terminal_summary=terminal_summary,
                    )
                )
            else:
                trace_path = str(
                    _write_trace_stub(trace_root, trajectory_id, history, terminal_summary)
                )
        except Exception as e:
            logger.warning(
                "trace write failed for %s: %s", trajectory_id.to_string(), e
            )

        return RolloutResult(
            trajectory_id=trajectory_id,
            instance_id=instance_id,
            messages=messages,
            history=history,
            reward=reward,
            reward_decomposition=decomp,
            termination_reason=termination_reason or TerminationReason.SUBMIT,
            cost_usd=budgets.cost_usd_used,
            tokens_prompt=0,
            tokens_completion=0,
            error=error,
            trace_path=trace_path,
        )

    # ------------------------------------------------------------------
    # Hooks (subclass-friendly)
    # ------------------------------------------------------------------

    def _open_sandbox(self, instance: Dict[str, Any]):
        """Return a context-manager yielding a :class:`Sandbox`. Overridable
        in tests via subclass to inject :func:`milo.lht_adapter.docker_runtime.null_sandbox`.
        """
        cfg = self.milo_env_cfg
        return milo_sandbox(
            instance=instance,
            cpu=cfg.sandbox_cpu,
            mem_gb=cfg.sandbox_mem_gb,
            disk_gb=cfg.sandbox_disk_gb,
            workdir=cfg.sandbox_workdir,
            timeout_s=300,
            executable=cfg.sandbox_executable,
        )

    def _invoke_policy(self, messages: List[Dict[str, str]], instance_id: str) -> str:
        """Call the inference engine HTTP endpoint with the current message
        history and return the assistant content as a string.

        Uses ``requests`` synchronously inside the rollout's thread (we're
        already off the event loop via ``run_in_executor``).
        """
        try:
            import requests  # type: ignore
        except ImportError:
            # Without an HTTP client we can't reach the inference engine —
            # synthesize a trivial "give up" response so smoke tests still
            # complete a rollout instead of crashing the whole batch.
            logger.warning("requests not installed; emitting stub submit action.")
            return "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff\n```"

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.generator_cfg.sampling_params.temperature,
            "max_tokens": self.generator_cfg.sampling_params.max_generate_length,
            "stream": False,
        }
        url = f"{self.base_url}/v1/chat/completions"
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Event-emission helpers
    # ------------------------------------------------------------------

    def _system_event(self, instance: Dict[str, Any], cfg) -> Dict[str, Any]:
        return {
            "id": f"sys_{uuid.uuid4().hex[:12]}",
            "timestamp": time.time(),
            "source": "agent",
            "kind": "SystemPromptEvent",
            "system_prompt": "You are a milo-bench LHT agent.",
            "tools": sorted(TOOL_NAMES),
            "dynamic_context": {"instance_id": instance.get("instance_id"), "lang": instance.get("milo_lang")},
        }

    def _message_event(self, msg: Dict[str, str], sender: str) -> Dict[str, Any]:
        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "timestamp": time.time(),
            "source": sender,
            "kind": "MessageEvent",
            "llm_message": msg,
            "llm_response_id": "",
            "activated_skills": [],
            "extended_content": [],
            "sender": sender,
            "critic_result": None,
        }

    def _render_obs(self, obs: Observation) -> str:
        """Render an observation as the user-message text the policy sees."""
        body = f"<returncode>{obs.exit_code}</returncode>\n"
        if obs.truncated:
            body += "<warning>output was truncated</warning>\n"
        body += f"<output>\n{obs.stdout}\n</output>\n"
        if obs.stderr:
            body += f"<stderr>\n{obs.stderr}\n</stderr>\n"
        return body

    # ------------------------------------------------------------------
    # Failure-path helper
    # ------------------------------------------------------------------

    def _failed_result(
        self,
        trajectory_id: TrajectoryID,
        candidate_patch: str,
        error: str,
        prompt: List[Dict[str, str]],
    ) -> RolloutResult:
        decomp = _stub_reward({}, None, [], 0.05, 0.20, 2.0, 0.0, [])[1]
        return RolloutResult(
            trajectory_id=trajectory_id,
            instance_id="",
            messages=list(prompt),
            history=[],
            reward=0.0,
            reward_decomposition=decomp,
            termination_reason=TerminationReason.CONTAINER_ERROR,
            cost_usd=0.0,
            tokens_prompt=0,
            tokens_completion=0,
            error=error,
            trace_path=None,
        )
