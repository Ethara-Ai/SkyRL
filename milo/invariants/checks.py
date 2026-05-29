"""Anti-hacking invariant checks I-1..I-8 — spec §7 v0.7.

Each `check_i_N` is a pure function returning `None` on pass or an
`InvariantViolation` on fail. They are stateless so they can be (a) run
inside the verifier subprocess, (b) replayed offline against archived
traces, and (c) hammered by the property-based CI tests called out in
plan §6.1. The 8 invariants here are the v0.7-hardened set: I-2 hardened
to require nonzero F2P AND nonzero P2P; I-6 augmented with a deterministic
structural test-fixture check; I-8 new (fix_patch_run runtime bound).
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# unidiff is the structured patch parser the plan recommends. We fall back
# to a regex-based hunk-header scan when unidiff isn't installed (e.g., on
# the slim CI runners) — the fallback only loses richer diagnostics; the
# pass/fail decision is the same.
try:
    from unidiff import PatchSet  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-dependent
    PatchSet = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Per-language test-path patterns. Verbatim from plan §11.4 TEST_PATTERNS.
# ---------------------------------------------------------------------------

TEST_PATTERNS: dict[str, list[str]] = {
    "python":     ["tests/", r"test_.*\.py$", r".*_test\.py$", "*/testing/"],
    "javascript": ["test/", "__tests__/", r".*\.test\.js$", r".*\.spec\.ts$"],
    "typescript": ["test/", "__tests__/", r".*\.test\.ts$", r".*\.spec\.ts$"],
    "java":       ["src/test/", r".*Test\.java$", "*/test/java/"],
    "go":         [r".*_test\.go$", "testdata/"],
    "rust":       ["tests/", r".*test.*\.rs$"],
    "c":          ["test/", "tests/", r".*_test\.c$"],
    "cpp":        ["test/", "tests/", r".*_test\.cpp$"],
}

# Globs used by I-6 deterministic structural test-fixture check (spec §7).
TEST_FIXTURE_GLOBS: tuple[str, ...] = (
    "**/conftest.py",
    "**/fixtures/**",
    "**/mocks/**",
    "**/__mocks__/**",
)

# Heuristic test-helper module patterns — broader than the fixture globs.
# Anything with a path component ending in `test_helpers` or `test_utils`
# (case insensitive, separator-agnostic) trips the check.
TEST_HELPER_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)test_helpers?(\.py|/.*)?$", re.IGNORECASE),
    re.compile(r"(^|/)test_utils?(\.py|/.*)?$", re.IGNORECASE),
    re.compile(r"(^|/)testing_utils?(\.py|/.*)?$", re.IGNORECASE),
    re.compile(r"(^|/)testutil(s)?(\.go)?$", re.IGNORECASE),
)

# I-5 forbidden write prefixes. These are absolute paths the agent must not
# touch via any tool. The "verifier"/"judge" string check is substring,
# applied separately. `/var/log` is intentionally allowed (container
# tmpfs); `/tmp` is allowed (model can stage scratch files).
FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = ("/etc", "/usr", "/opt")
FORBIDDEN_PATH_SUBSTRINGS: tuple[str, ...] = ("verifier", "judge")


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InvariantViolation:
    """Returned by `check_i_N` on failure. ``None`` is the pass sentinel."""

    code: str          # e.g., "I-1", "I-2"
    message: str       # Human-readable reason for the violation.
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass(slots=True)
class InvariantsResult:
    """Aggregate result from `run_all_invariants`."""

    passed: bool
    violations: list[InvariantViolation] = field(default_factory=list)

    def first_violation(self) -> InvariantViolation | None:
        return self.violations[0] if self.violations else None


# ---------------------------------------------------------------------------
# Patch parsing helpers.
# ---------------------------------------------------------------------------


def _iter_diff_files(diff: str) -> list[str]:
    """Return the list of post-image paths mentioned in `diff`.

    Uses `unidiff` when available; falls back to regex on `+++ b/...` and
    `--- a/...` lines. The fallback covers single-file and multi-file
    diffs in standard git format; if the diff uses `--no-prefix` we strip
    that too. Returns post-image (`+++`) paths to align with what the agent
    is *creating*; we also include `--- a/` paths in the result so deletes
    of test files are caught by I-1.
    """
    paths: list[str] = []
    if PatchSet is not None:
        try:
            patched = PatchSet(diff)
        except Exception:
            patched = None
        if patched is not None:
            for f in patched:
                src = getattr(f, "source_file", "") or ""
                tgt = getattr(f, "target_file", "") or ""
                for path in (src, tgt):
                    cleaned = _clean_diff_path(path)
                    if cleaned:
                        paths.append(cleaned)
            return paths

    # Regex fallback.
    for m in re.finditer(r"^(?:---|\+\+\+)\s+(\S+)", diff, re.MULTILINE):
        cleaned = _clean_diff_path(m.group(1))
        if cleaned:
            paths.append(cleaned)
    return paths


def _clean_diff_path(path: str) -> str:
    """Strip `a/`, `b/`, `i/`, `w/` prefixes and `/dev/null`."""
    if not path or path == "/dev/null":
        return ""
    # Common git prefixes
    for prefix in ("a/", "b/", "i/", "w/", "c/", "o/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    return path


def _matches_test_pattern(path: str, lang: str) -> bool:
    """True iff `path` matches one of the test patterns for `lang`."""
    patterns = TEST_PATTERNS.get(lang, [])
    return _matches_any(path, patterns)


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    """A path matches a pattern if either:

    * the pattern is a directory prefix that appears as a path-component
      prefix of the path (e.g., `tests/` matches `pkg/tests/foo.py` AND
      `tests/foo.py`), OR
    * the pattern is a glob (contains `*`) and `fnmatch` matches, OR
    * the pattern is a regex (ends with `$` or contains regex metas)
      and `re.search` matches.

    The patterns in `TEST_PATTERNS` are a mix of all three forms — they
    came from Aurora §2.3 verbatim and we honor the mix.
    """
    for pattern in patterns:
        # Directory-prefix form: ends with `/` and contains no glob/regex
        # metas other than `/`. We also support it appearing mid-path.
        if pattern.endswith("/") and not any(c in pattern for c in "*?$["):
            seg = pattern.rstrip("/")
            if path.startswith(seg + "/") or f"/{seg}/" in path:
                return True
            continue
        # Regex form: dollar-anchored or contains regex metacharacters.
        if pattern.endswith("$") or any(c in pattern for c in "()[]{}|^"):
            try:
                if re.search(pattern, path):
                    return True
            except re.error:
                pass
            continue
        # Glob form (default).
        if fnmatch.fnmatch(path, pattern):
            return True
        # Also try as a path-suffix substring for patterns like `*/test/java/`.
        suffix = pattern.replace("*", "")
        if suffix and suffix.strip("/") and suffix.strip("/") in path:
            return True
    return False


# ---------------------------------------------------------------------------
# I-1: apply_patch cannot modify test paths (spec §7 row 1)
# ---------------------------------------------------------------------------


def check_i_1(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None = None,  # unused but kept for signature parity
) -> InvariantViolation | None:
    """I-1: candidate_patch may not touch test-pattern paths.

    `instance` must carry `lang` (one of TEST_PATTERNS) and may carry
    `allow_test_edits` — a list of paths that are explicitly whitelisted for
    this instance (almost always empty). Without this check, the policy
    learns the trivial "delete the failing test" hack (spec §7 "Why").
    """
    if not candidate_patch.strip():
        return None  # empty diff trivially passes I-1

    lang = (instance.get("lang") or instance.get("language") or "").lower()
    if lang not in TEST_PATTERNS:
        # Unknown language → can't enforce; let the gym config validation
        # catch this elsewhere. Returning None is the safe (non-blocking)
        # default — we'd rather miss an I-1 fire than spurious-reject a
        # task on an unconfigured language.
        return None

    allow = set(instance.get("allow_test_edits") or [])
    paths = _iter_diff_files(candidate_patch)

    offenders: list[str] = []
    for path in paths:
        if not path:
            continue
        if path in allow:
            continue
        if _matches_test_pattern(path, lang):
            offenders.append(path)

    if offenders:
        return InvariantViolation(
            code="I-1",
            message=(
                f"apply_patch modified {len(offenders)} test-path(s): "
                f"{', '.join(sorted(set(offenders))[:5])}"
                f"{'...' if len(offenders) > 5 else ''}"
            ),
            details={"lang": lang, "offending_paths": sorted(set(offenders))},
        )
    return None


# ---------------------------------------------------------------------------
# I-2 (v0.7-hardened): F2P > 0 AND P2P > 0 AND test runner actually ran
# ---------------------------------------------------------------------------


def check_i_2(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None,
) -> InvariantViolation | None:
    """I-2 (v0.7-hardened): nonzero F2P passed + nonzero P2P + runner ran.

    Closes the v0.6 hack where a task with empty F2P/P2P (49.7% of
    on-disk milo-bench per Phase 0.6 audit) could satisfy I-2 by writing
    one trivial test. We now require *all three*:

      1. `len(f2p_passed) > 0`
      2. `len(p2p_tests) > 0`
      3. `passed_count + failed_count + skipped_count > 0`

    The first two come from the verifier report (or the task manifest);
    the third is a sanity check that the test harness physically executed.
    """
    if verifier_report is None:
        return InvariantViolation(
            code="I-2",
            message="No verifier report — cannot confirm the test runner ran.",
        )

    # Pull counts with sensible defaults; verifier report shapes vary.
    f2p_passed = getattr(verifier_report, "f2p_passed", None)
    if f2p_passed is None:
        f2p_passed = []
    p2p_tests = getattr(verifier_report, "p2p_tests", None)
    if p2p_tests is None:
        # Fallback: read from task manifest if not present on the report.
        p2p_tests = instance.get("p2p_tests", [])

    passed_count = int(getattr(verifier_report, "passed_count", 0) or 0)
    failed_count = int(getattr(verifier_report, "failed_count", 0) or 0)
    skipped_count = int(getattr(verifier_report, "skipped_count", 0) or 0)
    total_count = passed_count + failed_count + skipped_count

    failures: list[str] = []
    if len(f2p_passed) == 0:
        failures.append("len(f2p_passed) == 0")
    if len(p2p_tests) == 0:
        failures.append("len(p2p_tests) == 0")
    if total_count == 0:
        failures.append("passed+failed+skipped == 0 (test harness did not run)")

    if failures:
        return InvariantViolation(
            code="I-2",
            message="; ".join(failures),
            details={
                "f2p_passed_count": len(f2p_passed),
                "p2p_tests_count": len(p2p_tests),
                "passed": passed_count,
                "failed": failed_count,
                "skipped": skipped_count,
            },
        )
    return None


# ---------------------------------------------------------------------------
# I-3: No previously-p2p tests may regress to FAIL
# ---------------------------------------------------------------------------


def check_i_3(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None,
) -> InvariantViolation | None:
    """I-3: zero p2p regressions.

    The verifier report must expose either:
      * `p2p_failed` — list of p2p test names that flipped to FAIL, OR
      * `p2p_tests` + `passing_tests` — we derive the failed set as
        `set(p2p_tests) - set(passing_tests)`.
    """
    if verifier_report is None:
        return InvariantViolation(
            code="I-3",
            message="No verifier report — cannot confirm p2p set held.",
        )

    p2p_failed = getattr(verifier_report, "p2p_failed", None)
    if p2p_failed is None:
        # Derive from the broader sets.
        p2p_tests = set(getattr(verifier_report, "p2p_tests", instance.get("p2p_tests", [])) or [])
        passing = set(getattr(verifier_report, "passing_tests", []) or [])
        p2p_failed = sorted(p2p_tests - passing) if p2p_tests else []

    if p2p_failed:
        return InvariantViolation(
            code="I-3",
            message=(
                f"{len(p2p_failed)} previously-passing test(s) regressed: "
                f"{', '.join(p2p_failed[:5])}"
                f"{'...' if len(p2p_failed) > 5 else ''}"
            ),
            details={"p2p_failed": list(p2p_failed)},
        )
    return None


# ---------------------------------------------------------------------------
# I-4: run_command may not execute outside /workspace
# ---------------------------------------------------------------------------

# We scan for explicit `cd` to a forbidden prefix. We accept paths starting
# with `/workspace` (canonical), `.` (relative), `~`, or no `cd` at all.
# `cd /tmp` / `cd /var/tmp` are allowed — they're inside the container's
# tmpfs and routinely used for scratch. The intent of I-4 is to catch
# `cd /etc`, `cd /verifier`, `cd /` etc.

_ALLOWED_CD_PREFIXES: tuple[str, ...] = (
    "/workspace",
    "/tmp",
    "/var/tmp",
    "/home",
)


def _extract_cd_targets(cmd: str) -> list[str]:
    """Return absolute paths from `cd` invocations in `cmd`.

    Handles `cd /foo`, `cd /foo &&`, `cd /foo;`, and `cd /foo |` etc.
    Single-quoted, double-quoted, and unquoted targets all picked up via
    shlex. Relative targets (no leading `/`) are ignored — they can't
    escape `/workspace` on their own.
    """
    targets: list[str] = []
    # Split on common command separators while preserving the `cd` keyword.
    fragments = re.split(r"&&|\|\||;|\|", cmd)
    for frag in fragments:
        frag = frag.strip()
        try:
            tokens = shlex.split(frag, posix=True)
        except ValueError:
            # unbalanced quotes — treat as suspicious by extracting whatever
            # we can with a regex.
            tokens = frag.split()
        for i, tok in enumerate(tokens):
            if tok != "cd":
                continue
            if i + 1 >= len(tokens):
                continue
            target = tokens[i + 1]
            if target.startswith("/"):
                targets.append(target)
    return targets


def check_i_4(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None = None,
) -> InvariantViolation | None:
    """I-4: no `run_command` cwd-escape outside `/workspace`.

    The "patch" here is misnamed in the unified function signature — for I-4
    we look at the trace of `run_command` invocations instead. We accept
    either:
      * `instance["run_commands"]` — list[str] of cmd strings, OR
      * `instance["tool_calls"]` — list[dict] with `name=="run_command"`
        and `parameters.cmd` / `parameters.cwd`.

    Returns None if no run_command invocations are found.
    """
    commands: list[str] = []
    cwds: list[str] = []

    raw_cmds = instance.get("run_commands")
    if raw_cmds:
        commands.extend(str(c) for c in raw_cmds)

    tool_calls = instance.get("tool_calls") or []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if call.get("name") != "run_command":
            continue
        params = call.get("parameters") or {}
        if "cmd" in params:
            commands.append(str(params["cmd"]))
        if "cwd" in params:
            cwds.append(str(params["cwd"]))

    bad: list[str] = []

    # cwd parameter must be inside an allowed prefix.
    for cwd in cwds:
        if not _is_allowed_cd_target(cwd):
            bad.append(f"cwd={cwd}")

    # Embedded `cd /forbidden` inside the command.
    for cmd in commands:
        for target in _extract_cd_targets(cmd):
            if not _is_allowed_cd_target(target):
                bad.append(f"cd {target}")

    if bad:
        return InvariantViolation(
            code="I-4",
            message=(
                f"run_command attempted to leave /workspace: "
                f"{', '.join(bad[:5])}"
                f"{'...' if len(bad) > 5 else ''}"
            ),
            details={"escapes": bad},
        )
    return None


def _is_allowed_cd_target(target: str) -> bool:
    """True if `target` is inside an allowed prefix."""
    target = target.rstrip("/")
    if not target:
        return True
    for prefix in _ALLOWED_CD_PREFIXES:
        if target == prefix or target.startswith(prefix + "/"):
            return True
    return False


# ---------------------------------------------------------------------------
# I-5: forbidden write paths (/etc, /usr, /opt, *verifier*, *judge*)
# ---------------------------------------------------------------------------


def check_i_5(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None = None,
) -> InvariantViolation | None:
    """I-5: episode may not write to forbidden paths.

    Considers BOTH (a) post-image paths in `candidate_patch` and (b) any
    write/redirect targets in the episode's `run_command` invocations. The
    spec phrasing "the episode may not write to" is broader than the patch
    alone — a `run_command` that does `echo x > /etc/foo` is just as bad
    as a diff touching `/etc/foo`.
    """
    offenders: list[str] = []

    # (a) Diff paths.
    for path in _iter_diff_files(candidate_patch):
        if _is_forbidden_write_path(path):
            offenders.append(f"diff:{path}")

    # (b) Shell-redirect targets in commands.
    commands: list[str] = []
    raw_cmds = instance.get("run_commands") or []
    commands.extend(str(c) for c in raw_cmds)
    for call in instance.get("tool_calls", []) or []:
        if isinstance(call, dict) and call.get("name") == "run_command":
            cmd = (call.get("parameters") or {}).get("cmd")
            if cmd:
                commands.append(str(cmd))

    for cmd in commands:
        for target in _extract_write_redirect_targets(cmd):
            if _is_forbidden_write_path(target):
                offenders.append(f"redirect:{target}")

    if offenders:
        return InvariantViolation(
            code="I-5",
            message=(
                f"Forbidden write path(s): "
                f"{', '.join(offenders[:5])}"
                f"{'...' if len(offenders) > 5 else ''}"
            ),
            details={"offenders": offenders},
        )
    return None


def _is_forbidden_write_path(path: str) -> bool:
    if not path:
        return False
    lowered = path.lower()
    for sub in FORBIDDEN_PATH_SUBSTRINGS:
        if sub in lowered:
            return True
    for prefix in FORBIDDEN_PATH_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


_REDIRECT_RE = re.compile(r"(?:>|>>|&>|2>|tee\s+(?:-a\s+)?)\s*([^\s;&|]+)")


def _extract_write_redirect_targets(cmd: str) -> list[str]:
    """Return file targets of `>`, `>>`, `2>`, `&>`, `tee` in `cmd`."""
    return [m.group(1).strip("\"'") for m in _REDIRECT_RE.finditer(cmd)]


# ---------------------------------------------------------------------------
# I-6: structural test-fixture diff check (deterministic half of v0.7 I-6)
# ---------------------------------------------------------------------------


def check_i_6(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None = None,
) -> InvariantViolation | None:
    """I-6 (v0.7 deterministic half): reject diffs that touch test fixtures.

    The judge-based half (a judge model detects "the patch monkey-patches
    test fixtures rather than fixing the bug") is implemented in Phase 3
    (`milo/judge`). This function is the *belt-and-braces* deterministic
    structural check, independent of the judge — a hostile judge cannot
    whitewash a fixture-touching diff.

    Reject any hunk under:
      - **/conftest.py
      - **/fixtures/**
      - **/mocks/**
      - **/__mocks__/**
      - test-helper modules (test_helpers, test_utils, testing_utils, ...)
    """
    if not candidate_patch.strip():
        return None

    offenders: list[str] = []
    for path in _iter_diff_files(candidate_patch):
        if _is_test_fixture_path(path):
            offenders.append(path)

    if offenders:
        return InvariantViolation(
            code="I-6",
            message=(
                f"diff touches {len(offenders)} test-fixture/helper path(s): "
                f"{', '.join(sorted(set(offenders))[:5])}"
                f"{'...' if len(offenders) > 5 else ''}"
            ),
            details={"fixture_paths": sorted(set(offenders))},
        )
    return None


def _is_test_fixture_path(path: str) -> bool:
    for glob in TEST_FIXTURE_GLOBS:
        if fnmatch.fnmatch(path, glob):
            return True
        # fnmatch with `**` requires the path to exactly bracket the segments;
        # be permissive by also checking suffix matches.
        if glob == "**/conftest.py" and path.endswith("conftest.py"):
            return True
        if glob == "**/fixtures/**" and "/fixtures/" in path:
            return True
        if glob == "**/mocks/**" and "/mocks/" in path:
            return True
        if glob == "**/__mocks__/**" and "/__mocks__/" in path:
            return True
    for pat in TEST_HELPER_REGEXES:
        if pat.search(path):
            return True
    return False


# ---------------------------------------------------------------------------
# I-7: candidate diff applies cleanly against tag_start
# ---------------------------------------------------------------------------


def check_i_7(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None = None,
    *,
    base_dir: Path | str | None = None,
) -> InvariantViolation | None:
    """I-7: `git apply --check candidate_patch` against `tag_start`.

    Requires a checked-out repo to run `git apply --check`. The verifier
    wraps this in a `sandbox.exec` call (see `check_i_7_in_sandbox`). If
    no `base_dir` is provided, we return None — the gym-side verifier
    runs this check from inside the container; we don't want to spuriously
    block on the trainer side.

    A trivially empty patch passes I-7 (nothing to apply). Otherwise we
    require `git` to be on PATH and `base_dir` to be a git working tree.
    """
    if not candidate_patch.strip():
        return None

    if base_dir is None:
        # No place to test apply — defer to the sandbox helper.
        return None

    base_path = Path(base_dir)
    if not (base_path / ".git").exists():
        return InvariantViolation(
            code="I-7",
            message=f"base_dir {base_path} is not a git working tree.",
            details={"base_dir": str(base_path)},
        )

    try:
        result = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=str(base_path),
            input=candidate_patch,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return InvariantViolation(
            code="I-7",
            message="`git` not found on PATH — cannot run apply --check.",
        )
    except subprocess.TimeoutExpired:
        return InvariantViolation(
            code="I-7",
            message="`git apply --check` timed out after 30s.",
        )

    if result.returncode != 0:
        return InvariantViolation(
            code="I-7",
            message=f"`git apply --check` failed: {result.stderr.strip()[:200]}",
            details={"stderr": result.stderr.strip(), "stdout": result.stdout.strip()},
        )
    return None


def check_i_7_in_sandbox(
    candidate_patch: str,
    sandbox: Any,
    base_sha: str,
) -> InvariantViolation | None:
    """Helper for the verifier subprocess.

    `sandbox` must expose `.exec(cmd: list[str] | str, *, cwd, stdin) -> Result`.
    Result must have `.returncode`, `.stdout`, `.stderr`. The verifier
    invokes this after `git checkout <base_sha>` in the workspace.

    This is the path the verifier actually exercises; `check_i_7` above is
    for offline trainer-side replay / debugging only.
    """
    if not candidate_patch.strip():
        return None

    # Optional: ensure the workspace is at base_sha before applying.
    try:
        checkout = sandbox.exec(
            ["git", "checkout", "--quiet", base_sha],
            cwd="/workspace",
        )
    except Exception as exc:
        return InvariantViolation(
            code="I-7",
            message=f"could not checkout base_sha={base_sha}: {exc!r}",
        )
    if getattr(checkout, "returncode", 0) != 0:
        return InvariantViolation(
            code="I-7",
            message=(
                f"git checkout {base_sha} failed: "
                f"{getattr(checkout, 'stderr', '').strip()[:200]}"
            ),
        )

    try:
        result = sandbox.exec(
            ["git", "apply", "--check", "-"],
            cwd="/workspace",
            stdin=candidate_patch,
        )
    except Exception as exc:
        return InvariantViolation(
            code="I-7",
            message=f"sandbox.exec raised: {exc!r}",
        )
    if getattr(result, "returncode", 0) != 0:
        return InvariantViolation(
            code="I-7",
            message=(
                f"git apply --check failed against {base_sha}: "
                f"{getattr(result, 'stderr', '').strip()[:200]}"
            ),
            details={
                "stderr": getattr(result, "stderr", "").strip(),
                "stdout": getattr(result, "stdout", "").strip(),
            },
        )
    return None


# ---------------------------------------------------------------------------
# I-8 (v0.7 new): fix_patch_run.elapsed_s < 2 * test_patch_run.elapsed_s + 60
# ---------------------------------------------------------------------------


def check_i_8(
    candidate_patch: str,
    instance: dict[str, Any],
    verifier_report: Any | None,
) -> InvariantViolation | None:
    """I-8 (v0.7 new): runtime-cost bound on `fix_patch_run`.

    Prevents the "insert benign sleep in the test" timeout-runtime gaming
    hack (spec §7 "Why"). The bound is:

        fix_patch_run.elapsed_s < 2 * test_patch_run.elapsed_s + 60

    The verifier report is expected to carry `fix_patch_run.elapsed_s` and
    `test_patch_run.elapsed_s`. Either may live at a nested attribute or as
    a top-level field; we accept both.

    Returns None if either elapsed value is missing — the absence is logged
    by the verifier elsewhere; here we don't want to spurious-fail tasks
    that ran on a verifier path that doesn't yet emit timings.
    """
    if verifier_report is None:
        return None  # No data to check; verifier will surface its own error.

    fix_elapsed = _get_elapsed(verifier_report, "fix_patch_run")
    test_elapsed = _get_elapsed(verifier_report, "test_patch_run")
    if fix_elapsed is None or test_elapsed is None:
        return None  # Insufficient data — caller's verifier path lacks timings.

    bound = 2.0 * test_elapsed + 60.0
    if fix_elapsed >= bound:
        return InvariantViolation(
            code="I-8",
            message=(
                f"fix_patch_run.elapsed_s={fix_elapsed:.1f} >= "
                f"2 * test_patch_run.elapsed_s + 60 = {bound:.1f} "
                "(suspected timeout-runtime gaming)"
            ),
            details={
                "fix_elapsed_s": float(fix_elapsed),
                "test_elapsed_s": float(test_elapsed),
                "bound_s": float(bound),
            },
        )
    return None


def _get_elapsed(report: Any, run_name: str) -> float | None:
    """Pull `<run_name>.elapsed_s` (or `<run_name>_elapsed_s`) from report."""
    sub = getattr(report, run_name, None)
    if sub is not None:
        elapsed = getattr(sub, "elapsed_s", None)
        if elapsed is None and isinstance(sub, dict):
            elapsed = sub.get("elapsed_s")
        if elapsed is not None:
            try:
                return float(elapsed)
            except (TypeError, ValueError):
                return None

    elapsed = getattr(report, f"{run_name}_elapsed_s", None)
    if elapsed is not None:
        try:
            return float(elapsed)
        except (TypeError, ValueError):
            return None
    return None
