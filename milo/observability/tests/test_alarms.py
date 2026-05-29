"""Tests for the three production alarms."""

from __future__ import annotations

from milo.observability.alarms import (
    BedrockJudgeRateLimitAlarm,
    ImagePullFailureAlarm,
    RewardHackingAlarm,
)


def test_image_pull_alarm_below_threshold() -> None:
    a = ImagePullFailureAlarm(window=10, threshold=0.05)
    for _ in range(10):
        a.observe(False)
    assert a.check() is None


def test_image_pull_alarm_above_threshold() -> None:
    a = ImagePullFailureAlarm(window=10, threshold=0.05)
    for _ in range(8):
        a.observe(False)
    a.observe(True)
    a.observe(True)
    ev = a.check()
    assert ev is not None
    assert ev.severity == "critical"


def test_image_pull_alarm_warms_up_silently() -> None:
    """Until the window fills, no alarm even with 100% failure rate."""
    a = ImagePullFailureAlarm(window=20, threshold=0.05)
    for _ in range(10):
        a.observe(True)
    assert a.check() is None


def test_reward_hacking_alarm() -> None:
    a = RewardHackingAlarm(window=100, threshold=0.01)
    for _ in range(99):
        a.observe(False)
    a.observe(False)            # 0% — under
    assert a.check() is None
    # 2 violations in window → 2% > 1% → fires
    for _ in range(98):
        a.observe(False)
    a.observe(True)
    a.observe(True)
    ev = a.check()
    assert ev is not None
    assert ev.severity == "critical"


def test_bedrock_rate_limit_alarm_fires_then_resets() -> None:
    a = BedrockJudgeRateLimitAlarm(consecutive_threshold=3)
    a.observe(429); a.observe(429)
    assert a.check() is None        # only 2 consecutive
    a.observe(429)
    ev = a.check()
    assert ev is not None
    assert ev.severity == "warning"
    # After firing, consecutive resets so we don't spam.
    assert a.check() is None


def test_bedrock_rate_limit_alarm_non_429_resets_counter() -> None:
    a = BedrockJudgeRateLimitAlarm(consecutive_threshold=3)
    a.observe(429); a.observe(429); a.observe(200)
    a.observe(429); a.observe(429)
    assert a.check() is None
