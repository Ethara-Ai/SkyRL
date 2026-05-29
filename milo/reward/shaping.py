"""Per-step test-delta shaping reward — spec §4.4.2.

Implements `compute_delta`, the pure function that scores one test-runner
step in the form `Δ_pass − λ · Δ_regress`. Asymmetric by design (default
`λ = 2.0`): regressing a previously-passing p2p test hurts twice as much as
flipping a fail-to-pass f2p test helps, which discourages the locally-greedy
"burn it all down and rebuild" strategy (spec §4.4.2 + Skalse et al. on
reward-hacking).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TestResult:
    """Minimal per-step test-result summary needed for shaping.

    The full verifier `TestResult` (defined in Phase 2 / `milo/verifier`) is
    richer; this slim shape is the *only* surface the shaping function needs,
    which makes shaping unit-testable without standing up the verifier.

    Attributes
    ----------
    passing_tests:
        Names of tests that ended this step in the PASS state.
    failing_tests:
        Names of tests that ended this step in the FAIL (or ERROR) state.
    f2p_tests:
        The instance's `f2p_tests` set (from the task manifest). Static across
        the episode, but stamped on every TestResult so the shaping function
        has everything it needs in one argument.
    p2p_tests:
        The instance's `p2p_tests` set.
    """

    # `__test__ = False` tells pytest this is not a test class despite the
    # `Test*` name (we keep the spec-aligned name).
    __test__ = False

    passing_tests: set[str] = field(default_factory=set)
    failing_tests: set[str] = field(default_factory=set)
    f2p_tests: set[str] = field(default_factory=set)
    p2p_tests: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Defensive copy — callers often hand in lists.
        if not isinstance(self.passing_tests, set):
            self.passing_tests = set(self.passing_tests)
        if not isinstance(self.failing_tests, set):
            self.failing_tests = set(self.failing_tests)
        if not isinstance(self.f2p_tests, set):
            self.f2p_tests = set(self.f2p_tests)
        if not isinstance(self.p2p_tests, set):
            self.p2p_tests = set(self.p2p_tests)


def compute_delta(
    prev_test_result: TestResult | None,
    curr_test_result: TestResult,
    lambda_: float = 2.0,
) -> float:
    """Compute `R_delta(t) = Δ_pass(t) − λ · Δ_regress(t)`. Spec §4.4.2.

    Parameters
    ----------
    prev_test_result:
        Test result from the previous test-runner step, or ``None`` for the
        first run of the episode. When ``None``, we treat *all* currently
        passing f2p tests as newly-passing (i.e., baseline is "nothing has
        run yet"). This matches the plan's `compute_step_delta` contract.
    curr_test_result:
        Test result from the current step. Must carry `f2p_tests` and
        `p2p_tests` so the function can compute the diffs without external
        state.
    lambda_:
        Asymmetry factor for regressions. Default 2.0 (spec §4.4.2 default).
        Configurable via the reward preset YAML; the regression-asymmetry
        rationale is in spec §4.4.2 "Why the λ = 2 asymmetry".

    Returns
    -------
    R_delta(t) as a float. May be positive (net progress), negative (net
    regression), or zero (no change against the f2p/p2p test sets).

    Notes
    -----
    * `Δ_pass(t)` counts only tests that *newly* pass and are in `f2p_tests`.
      A test passing-since-before-this-step does not contribute (it would be
      double-counted across episode steps).
    * `Δ_regress(t)` counts only tests that *newly* fail and were in
      `p2p_tests` (i.e., the regression set). f2p tests that flip back to
      failing are not counted as "regressions" — they just lose their Δ_pass
      credit on a subsequent step where they re-pass.
    * The function is intentionally symmetric in test-set type but
      *asymmetric* in lambda: f2p gains weight 1, p2p losses weight `λ`.
    * Uses `curr_test_result`'s f2p/p2p sets as the source of truth — these
      are static per episode but stamped on every TestResult for caller
      convenience.
    """
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative, got {lambda_}")

    f2p_tests = curr_test_result.f2p_tests
    p2p_tests = curr_test_result.p2p_tests

    if prev_test_result is None:
        prev_passing: set[str] = set()
        prev_failing: set[str] = set()
    else:
        prev_passing = prev_test_result.passing_tests
        prev_failing = prev_test_result.failing_tests

    curr_passing = curr_test_result.passing_tests
    curr_failing = curr_test_result.failing_tests

    # Δ_pass(t): tests that are now passing AND are in f2p AND were not
    # passing on the previous step. "Was not passing" includes both
    # previously failing and previously unobserved (some test runners
    # discover new tests as code lands).
    newly_passing = curr_passing - prev_passing
    delta_pass = len(newly_passing & f2p_tests)

    # Δ_regress(t): tests that are now failing AND are in p2p AND were not
    # failing on the previous step. p2p_tests are by definition tests that
    # were passing at baseline; the per-step diff catches when one drops.
    newly_failing = curr_failing - prev_failing
    delta_regress = len(newly_failing & p2p_tests)

    return float(delta_pass) - lambda_ * float(delta_regress)
