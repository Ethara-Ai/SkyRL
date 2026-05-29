"""Composite reward aggregator — spec §4.4 / §5.4 / plan Phase 4.

Implements `CompositeRewardAggregator`, the pure-function class that takes a
verifier report, rubric report, the per-step shaping rewards (already
computed via `milo.reward.shaping.compute_delta`), and per-step TIR events
(already computed via `milo.reward.tir.compute_tir`), and produces the
`RewardDecomposition` (§6.8). Honors the v0.7-hardened invariant override:
if `verifier_report.passes_invariant_check` is False, `R_terminal` is
forced to 0 regardless of test outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from milo.reward.decomposition import RewardDecomposition, RewardPreset

# ---------------------------------------------------------------------------
# Lightweight Protocols for the verifier/rubric reports. The actual concrete
# classes live in `milo/verifier/` (Phase 2) and `milo/judge/` (Phase 3); we
# bind to the structural attributes the aggregator needs, which keeps Phase 4
# decoupled from those modules' eventual concrete shapes.
# ---------------------------------------------------------------------------


@runtime_checkable
class VerifierReportLike(Protocol):
    """Structural protocol — see Phase 2 `milo.verifier`.

    Only the fields the aggregator reads are listed. A concrete report may
    carry far more (timings, per-test status records, log paths).
    """

    resolved: bool                       # §4.4.1 terminal "R_terminal = 1?"
    passes_invariant_check: bool         # §7 — if False, R_terminal := 0


@runtime_checkable
class RubricReportLike(Protocol):
    """Structural protocol — see Phase 3 `milo.judge`.

    `per_item_scores` is a list of floats in `{0, 0.5, 1}` (spec §4.4.3).
    `R_rubric` is the mean (computed by the aggregator, not the judge).
    """

    per_item_scores: Sequence[float]


# Fallback dataclasses for callers (tests, prototypes) that don't have a real
# verifier or judge wired up yet. These satisfy the protocols above.


@dataclass(slots=True)
class StubVerifierReport:
    """Minimal verifier-report stand-in. Mirrors `VerifierReportLike`."""

    resolved: bool = False
    passes_invariant_check: bool = True
    f2p_passed: list[str] = field(default_factory=list)
    p2p_tests: list[str] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0


@dataclass(slots=True)
class StubRubricReport:
    """Minimal rubric-report stand-in. Mirrors `RubricReportLike`."""

    per_item_scores: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preset loader. We avoid hard-depending on PyYAML — try `yaml`, then fall
# back to a one-purpose loader that handles the simple key:value format
# our preset files use.
# ---------------------------------------------------------------------------


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the minimal YAML subset our presets use.

    Supports lines of the form `key: value` (comments after `#` ignored,
    blank lines ignored). Values are coerced to int/float/bool/string in
    that order. This exists so we don't add a runtime PyYAML dep just to
    read four scalars; the trainer brings PyYAML transitively in any real
    install, in which case we use the real parser.
    """
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not value:
            continue
        # bool first (Python int subclass of bool would otherwise grab them)
        low = value.lower()
        if low in {"true", "false"}:
            out[key] = low == "true"
            continue
        try:
            out[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(value)
            continue
        except ValueError:
            pass
        out[key] = value
    return out


def _load_preset(preset_name: str) -> dict[str, Any]:
    """Load `milo/reward/presets/{preset_name}.yaml`.

    Prefers PyYAML when available (it understands the full grammar); falls
    back to the inlined `_parse_simple_yaml` for environments without it.
    """
    presets_dir = Path(__file__).resolve().parent / "presets"
    preset_path = presets_dir / f"{preset_name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(
            f"Reward preset '{preset_name}' not found at {preset_path}. "
            f"Known presets: {sorted(p.stem for p in presets_dir.glob('*.yaml'))}."
        )
    text = preset_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ValueError(f"Preset {preset_path} did not parse to a dict.")
        return parsed
    except ImportError:
        return _parse_simple_yaml(text)


# ---------------------------------------------------------------------------
# The aggregator.
# ---------------------------------------------------------------------------


class CompositeRewardAggregator:
    """Pure function-as-class implementing spec §4.4.

    `aggregate(...)` returns a `RewardDecomposition`. The class form is for
    config bookkeeping — one aggregator instance per training run, holding
    the preset name + weight knobs — but `aggregate` itself takes everything
    it needs as arguments and has zero side effects. This is what lets CI's
    property-based test (spec §10.4) hammer it.

    Composite formula (spec §4.4):

        R_total = R_terminal
                + alpha · Σ R_delta(t)
                + beta  · R_rubric
                + gamma · Σ R_tir(t)

    Notes
    -----
    * `R_total` is NOT clipped per spec §4.4.5 — the GRPO trainer normalizes
      downstream (plan §25.5 row 4 / spec §14.4).
    * `R_terminal` is forced to 0 whenever
      `verifier_report.passes_invariant_check` is False, per spec §7 and
      v0.7-hardened I-2.
    * The aggregator does *not* recompute `R_rubric` per item — it consumes
      the rubric report's `per_item_scores` and means them. Per-item logic
      (deterministic vs. judge, scoring rubric) is the judge's job.
    """

    def __init__(
        self,
        preset: str = "composite",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        config = _load_preset(preset)
        if overrides:
            config = {**config, **overrides}

        # Validate / coerce.
        self.preset: RewardPreset = config.get("preset", preset)  # type: ignore[assignment]
        if self.preset not in ("composite", "pure_rlvr"):
            raise ValueError(
                f"Unknown preset '{self.preset}'. Expected 'composite' or 'pure_rlvr'."
            )
        self.alpha: float = float(config.get("alpha", 0.0))
        self.beta: float = float(config.get("beta", 0.0))
        self.lambda_: float = float(config.get("lambda", config.get("lambda_", 2.0)))
        self.gamma: float = float(config.get("gamma", 0.0))

        # Bounds — these are not contract-fixed but flag clearly busted YAML.
        if not (0.0 <= self.alpha <= 100.0):
            raise ValueError(f"alpha out of sane range: {self.alpha}")
        if not (0.0 <= self.beta <= 100.0):
            raise ValueError(f"beta out of sane range: {self.beta}")
        if not (0.0 <= self.gamma <= 100.0):
            raise ValueError(f"gamma out of sane range: {self.gamma}")
        if self.lambda_ < 0:
            raise ValueError(f"lambda must be non-negative, got {self.lambda_}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        verifier_report: VerifierReportLike | None,
        rubric_report: RubricReportLike | None,
        shaping_rewards: Iterable[float],
        tir_rewards: Iterable[int],
    ) -> RewardDecomposition:
        """Apply the §4.4 composite formula. Returns a `RewardDecomposition`.

        Parameters
        ----------
        verifier_report:
            Anything satisfying `VerifierReportLike`. May be ``None`` (e.g.,
            episode terminated before verifier ran) — treated as
            unresolved + invariant-failed → `R_terminal = 0`.
        rubric_report:
            Anything satisfying `RubricReportLike`. May be ``None`` or
            empty — `R_rubric = 0` in that case, and `r_rubric_per_item` is
            an empty list.
        shaping_rewards:
            Per-step `R_delta(t)` values (already computed). One entry per
            *test-runner step*, NOT per total step. May be empty.
        tir_rewards:
            Per-step `R_tir(t)` values (already computed). One entry per
            tool call. Each must be -1 or 0; non-conforming values are
            preserved verbatim (the aggregator does not enforce TIR's
            domain — the contract is on the caller).

        Returns
        -------
        RewardDecomposition matching spec §6.8.
        """
        # --- R_terminal ----------------------------------------------------
        if verifier_report is None:
            r_terminal = 0
        else:
            resolved = bool(getattr(verifier_report, "resolved", False))
            passes = bool(getattr(verifier_report, "passes_invariant_check", True))
            # v0.7-hardened: invariant failure ALWAYS zeros the terminal
            # reward, even if all tests pass. Spec §7 top of section.
            r_terminal = 1 if (resolved and passes) else 0

        # --- R_delta -------------------------------------------------------
        r_delta_steps = [float(x) for x in shaping_rewards]
        r_delta_sum = sum(r_delta_steps)

        # --- R_rubric ------------------------------------------------------
        if rubric_report is None:
            r_rubric_per_item: list[float] = []
        else:
            r_rubric_per_item = [float(x) for x in getattr(rubric_report, "per_item_scores", [])]
        r_rubric_mean = (
            sum(r_rubric_per_item) / len(r_rubric_per_item)
            if r_rubric_per_item
            else 0.0
        )

        # --- R_tir ---------------------------------------------------------
        r_tir_steps = [int(x) for x in tir_rewards]
        r_tir_sum = float(sum(r_tir_steps))

        # --- Weighted components + total ----------------------------------
        c_terminal = float(r_terminal)
        c_shaping = self.alpha * r_delta_sum
        c_rubric = self.beta * r_rubric_mean
        c_tir = self.gamma * r_tir_sum
        r_total = c_terminal + c_shaping + c_rubric + c_tir

        return RewardDecomposition(
            preset=self.preset,
            r_terminal=r_terminal,
            r_delta_steps=r_delta_steps,
            r_delta_sum=float(r_delta_sum),
            r_rubric_per_item=r_rubric_per_item,
            r_rubric_mean=float(r_rubric_mean),
            r_tir_steps=r_tir_steps,
            r_tir_sum=r_tir_sum,
            alpha=self.alpha,
            beta=self.beta,
            lambda_=self.lambda_,
            gamma=self.gamma,
            r_total=float(r_total),
            components={
                "terminal": c_terminal,
                "shaping": float(c_shaping),
                "rubric": float(c_rubric),
                "tir": float(c_tir),
            },
        )

    # Convenience: allow callers to treat the aggregator as a function.
    __call__ = aggregate

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return the aggregator's configured weights — for trace headers."""
        return {
            "preset": self.preset,
            "alpha": self.alpha,
            "beta": self.beta,
            "lambda": self.lambda_,
            "gamma": self.gamma,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CompositeRewardAggregator(preset={self.preset!r}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"lambda={self.lambda_}, gamma={self.gamma})"
        )
