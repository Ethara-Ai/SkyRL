"""Policy adapter Protocol + deterministic stub.

Implements the minimum contract surface required by:

* ``milo.calibration.runner`` — Phase 8 (pass@k per task per model).
* ``milo.replay.cli`` — Phase 9 (deterministic trace replay).
* ``milo.eval.evaluator`` — Phase 16 (holdout pass@k against baselines).
* ``milo.observability.nightly_audit`` — Phase 17 (replay 20 random rollouts).

The real adapters (litellm + boto3 Bedrock) land in Phase 7 per
``IMPLEMENTATION_PLAN.md`` v0.4 §7. Until then, every caller in this
module can be unit-tested against :class:`StubPolicyAdapter`, which
returns deterministic per-(instance, seed) results keyed off a tiny
in-memory dict.

The Protocol is intentionally small — the calibration runner and eval
harness only need ``policy_id`` + ``rollout`` — because every other shape
detail (tool-call format, vLLM vs Bedrock JSON quirks) is the real
adapter's problem, not the consumer's. See plan §7 for the per-adapter
break-down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(slots=True)
class PolicyResult:
    """Single-rollout outcome from a :class:`PolicyAdapter`.

    Only the fields the consumers above actually read are modeled. The full
    on-disk trace shape (spec §6.5) is the rollout driver's responsibility —
    the adapter just returns "did it pass, what was the reward, how did we
    end" so the calibration / eval / replay loops can score it.

    Attributes
    ----------
    instance_id:
        The task that was rolled out.
    seed:
        The rollout seed (spec §9.2). Two rollouts with the same
        ``(instance_id, seed, policy_id)`` triple must produce the same
        ``passed`` and ``reward_decomposition`` (modulo provider-level
        nondeterminism — see plan §7 for the per-provider seed contract).
    passed:
        ``True`` iff ``r_terminal == 1`` and no §7 invariant was violated.
        This is what pass@k counts.
    reward_decomposition:
        The full §6.8 dict. Calibration only needs ``passed``; the replay
        tool needs the full dict to assert byte-equality against the stored
        trace.
    cost_usd:
        Per-rollout cost (sums of prompt + completion + cache tokens times
        the per-model pricing). Logged but not used in pass@k.
    termination_reason:
        One of ``submit | timeout | tool_budget | container_error |
        cost_guardrail`` per spec §6.5 terminal_summary.
    extras:
        Free-form for future per-adapter metadata. Consumers should not
        rely on any particular key.
    """

    instance_id: str
    seed: int
    passed: bool
    reward_decomposition: dict[str, Any]
    cost_usd: float = 0.0
    termination_reason: str = "submit"
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PolicyAdapter(Protocol):
    """Minimal Protocol every policy backend implements.

    The real Phase 7 implementations live in ``milo/adapters/{anthropic,
    openai, gemini, bedrock, vllm}.py``. The Protocol is intentionally
    sync — the Phase 7 implementations will be async; this stub keeps the
    interface usable from synchronous unit tests without the
    ``async`` plumbing leaking into the test surface.
    """

    @property
    def policy_id(self) -> str:
        """A short stable identifier — e.g. ``"claude-opus-4-6"``.

        Used as the dict key in :class:`milo.calibration.runner.CalibrationRunner`
        results and as the W&B group tag.
        """
        ...

    def rollout(self, instance_id: str, seed: int) -> PolicyResult:
        """Run one rollout end-to-end and return the result.

        Must be deterministic when called repeatedly with the same
        ``(instance_id, seed)`` — replay (spec §10.3) depends on this.
        """
        ...


class StubPolicyAdapter:
    """In-memory deterministic adapter used by every Phase 8/9/16/17 test.

    Construction takes a flat ``outcomes`` map:

        {(instance_id, seed): PolicyResult}

    plus a fallback ``default_pass_rate``: any (instance, seed) not in the
    map gets a hash-derived deterministic pass/fail at the configured rate.
    This is the only stochasticity in the stub, and it's seeded off
    ``hash((policy_id, instance_id, seed))`` so two adapters with different
    ``policy_id`` strings produce different pass profiles — which is exactly
    what the §8 calibration disagreement check needs to exercise.
    """

    def __init__(
        self,
        policy_id: str,
        outcomes: Mapping[tuple[str, int], PolicyResult] | None = None,
        default_pass_rate: float = 0.0,
    ) -> None:
        self._policy_id = policy_id
        self._outcomes: dict[tuple[str, int], PolicyResult] = dict(outcomes or {})
        if not 0.0 <= default_pass_rate <= 1.0:
            raise ValueError(
                f"default_pass_rate must be in [0, 1], got {default_pass_rate!r}"
            )
        self._default_pass_rate = default_pass_rate

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def rollout(self, instance_id: str, seed: int) -> PolicyResult:
        key = (instance_id, seed)
        if key in self._outcomes:
            # Caller-provided outcome — return as-is, preserving determinism.
            return self._outcomes[key]
        # Fallback: hash-derived deterministic pass/fail at the configured rate.
        # We use SHA-256 over the explicit tuple (not Python's salted hash())
        # so the result is reproducible across processes and Python versions.
        import hashlib

        digest = hashlib.sha256(
            f"{self._policy_id}|{instance_id}|{seed}".encode("utf-8")
        ).digest()
        # Map first 8 bytes to a [0, 1) float.
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        passed = bucket < self._default_pass_rate
        return PolicyResult(
            instance_id=instance_id,
            seed=seed,
            passed=passed,
            reward_decomposition={
                "preset": "composite",
                "r_terminal": 1 if passed else 0,
                "r_delta_steps": [],
                "r_delta_sum": 0.0,
                "r_rubric_per_item": [],
                "r_rubric_mean": 1.0 if passed else 0.0,
                "r_tir_steps": [],
                "r_tir_sum": 0.0,
                "alpha": 0.05,
                "beta": 0.20,
                "lambda": 2.0,
                "gamma": 0.0,
                "r_total": 1.0 if passed else 0.0,
                "components": {
                    "terminal": 1.0 if passed else 0.0,
                    "shaping": 0.0,
                    "rubric": 0.20 if passed else 0.0,
                    "tir": 0.0,
                },
            },
            cost_usd=0.0,
            termination_reason="submit",
            extras={"stub": True},
        )


def make_stub_adapter(
    policy_id: str,
    pass_rate: float = 0.0,
    overrides: Mapping[tuple[str, int], PolicyResult] | None = None,
) -> StubPolicyAdapter:
    """Convenience factory for tests.

    ``pass_rate`` sets the deterministic fallback pass-rate; ``overrides``
    pins specific ``(instance_id, seed)`` outcomes. Typical use:

        adapter_a = make_stub_adapter("claude-opus-4-6", pass_rate=0.8)
        adapter_b = make_stub_adapter("gemini-2.5-pro",   pass_rate=0.7)
    """
    return StubPolicyAdapter(
        policy_id=policy_id,
        outcomes=overrides,
        default_pass_rate=pass_rate,
    )
