"""Rubric judge service — Phase 3 of the Milo-bench RL plan.

Implements the LLM-based rubric judge that scores a candidate `git diff`
against a list of natural-language assertions at `submit()` time. See
RL_GYM_SPEC.md v0.7 §4.4.3 (rubric reward), §5.3 (judge service), §6.7
(rubric schema), and IMPLEMENTATION_PLAN.md v0.4 Phase 3 for the full
contract. The judge model is configurable via `${MILO_JUDGE_MODEL}`.
"""

from __future__ import annotations

from milo.judge.prompt import PROMPT_SHA256, load_prompt
from milo.judge.service import (
    PerItemScore,
    RubricReport,
    RubricJudgeService,
)

__all__ = [
    "PROMPT_SHA256",
    "PerItemScore",
    "RubricJudgeService",
    "RubricReport",
    "load_prompt",
]
