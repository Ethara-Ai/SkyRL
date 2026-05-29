"""Phase 17.2 — three production alarms.

Per spec §20.4 / plan §17.2:

    1. ImagePullFailureAlarm     — rolling 100-rollout image_pull_failure_rate > 5%
                                     → triggers Phase 28.1 disaster recovery
                                       (quarantine + image rebuild)
    2. RewardHackingAlarm        — rolling 100-rollout invariant_violation_rate > 1%
                                     → pause + audit traces
    3. BedrockJudgeRateLimitAlarm — 3 consecutive judge calls return 429
                                     → fall back to direct Anthropic API

Each alarm exposes `check(metrics: dict) -> AlarmEvent | None`. The
`milo.observability.NightlyAudit` and the training driver both call these
on a tick — when an alarm fires, the wired ObservabilityBackend
(`milo.customization.ObservabilityBackend`) takes the recovery action.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque


@dataclass
class AlarmEvent:
    name: str
    severity: str          # "warning" | "critical"
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "payload": self.payload,
        }


# ----------------------------------------------------------- Image-pull failure


class ImagePullFailureAlarm:
    """Fires when the rolling rate of `image_pull_failure` events exceeds threshold."""

    NAME = "image_pull_failure_rate"

    def __init__(self, window: int = 100, threshold: float = 0.05) -> None:
        self.window = window
        self.threshold = threshold
        self._events: Deque[bool] = deque(maxlen=window)

    def observe(self, failure: bool) -> None:
        """Call once per rollout — True if Docker pull failed."""
        self._events.append(failure)

    def check(self, metrics: dict[str, Any] | None = None) -> AlarmEvent | None:
        if len(self._events) < self.window:
            return None
        rate = sum(self._events) / len(self._events)
        if rate > self.threshold:
            return AlarmEvent(
                name=self.NAME,
                severity="critical",
                message=f"image-pull failure rate {rate:.1%} > {self.threshold:.1%}",
                payload={"rate": rate, "window": self.window},
            )
        return None


# ---------------------------------------------------------- Reward-hacking alarm


class RewardHackingAlarm:
    """Fires when the rolling rate of invariant violations exceeds threshold."""

    NAME = "invariant_violation_rate"

    def __init__(self, window: int = 100, threshold: float = 0.01) -> None:
        self.window = window
        self.threshold = threshold
        self._events: Deque[bool] = deque(maxlen=window)

    def observe(self, violated: bool) -> None:
        self._events.append(violated)

    def check(self, metrics: dict[str, Any] | None = None) -> AlarmEvent | None:
        if len(self._events) < self.window:
            return None
        rate = sum(self._events) / len(self._events)
        if rate > self.threshold:
            return AlarmEvent(
                name=self.NAME,
                severity="critical",
                message=f"invariant violation rate {rate:.1%} > {self.threshold:.1%}",
                payload={"rate": rate, "window": self.window},
            )
        return None


# ----------------------------------------------------------- Bedrock-rate-limit


class BedrockJudgeRateLimitAlarm:
    """Fires when 3 consecutive judge calls return HTTP 429."""

    NAME = "bedrock_judge_rate_limit"

    def __init__(self, consecutive_threshold: int = 3) -> None:
        self.consecutive_threshold = consecutive_threshold
        self._consecutive_429 = 0

    def observe(self, status_code: int) -> None:
        if status_code == 429:
            self._consecutive_429 += 1
        else:
            self._consecutive_429 = 0

    def check(self, metrics: dict[str, Any] | None = None) -> AlarmEvent | None:
        if self._consecutive_429 >= self.consecutive_threshold:
            ev = AlarmEvent(
                name=self.NAME,
                severity="warning",
                message=(
                    f"{self._consecutive_429} consecutive 429s from Bedrock judge — "
                    f"fall back to direct Anthropic (plan §28)"
                ),
                payload={"consecutive_429": self._consecutive_429},
            )
            self._consecutive_429 = 0   # reset after firing
            return ev
        return None
