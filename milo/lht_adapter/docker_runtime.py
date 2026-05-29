"""Per-rollout Docker sandbox for milo-bench LHT instances.

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §1.3 — a context manager that opens
a fresh container per rollout, exposes a ``Sandbox`` interface (``exec``,
``read_file``, ``write_file``, ``git_diff``, ``apply_patch``, ``close``) and
mirrors the contract of ``mini_swe_agent``'s ``get_sb_environment`` helper so we
inherit its Docker / podman / singularity backend selection without duplicating
the logic. The image name is derived via :mod:`milo.lht_adapter.image_naming`.
"""

from __future__ import annotations

import logging
import shlex
import tempfile
import textwrap
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from milo.lht_adapter.image_naming import get_image_name


__all__ = ["Sandbox", "SandboxError", "ExecResult", "milo_sandbox"]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    """Structured result from a single shell exec inside the sandbox.

    Mirrors the dict shape returned by ``mini_swe_agent``'s ``Environment.execute``
    (``{"returncode": int, "output": str}``) but adds elapsed time and an
    explicit ``timed_out`` flag so callers don't have to grep stderr.
    """

    returncode: int
    output: str
    elapsed_s: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "returncode": self.returncode,
            "output": self.output,
            "elapsed_s": self.elapsed_s,
            "timed_out": self.timed_out,
        }


class SandboxError(RuntimeError):
    """Raised when the sandbox backend is unavailable or the container is in a
    bad state. Distinct from a non-zero exit code (which is a normal
    ``ExecResult`` with ``returncode != 0``)."""


# ---------------------------------------------------------------------------
# Sandbox wrapper
# ---------------------------------------------------------------------------


class Sandbox:
    """Thin facade over a ``minisweagent.environments.Environment`` (or any
    duck-typed equivalent exposing an ``execute(cmd, *, cwd=None, timeout=None)``
    method returning ``{"returncode": int, "output": str}``).

    The wrapper centralizes:
      - timeouts (we pass them through and detect the 124 exit code)
      - ``read_file`` / ``write_file`` via cat + heredoc (no host-side fs
        access; everything is in-container)
      - ``git_diff`` against the current HEAD (cheap; the spec §6.4 step
        record carries this as ``working_tree_patch``)
      - ``apply_patch`` via ``git apply`` (matches I-7 invariant check
        prerequisites; the Phase 6 invariant code re-runs the check itself)

    Callers MUST use the ``milo_sandbox`` context manager; constructing this
    directly works but bypasses the per-rollout cleanup we want.
    """

    def __init__(
        self,
        backend_env: Any,
        instance: Dict[str, Any],
        workdir: str = "/testbed",
        default_timeout_s: int = 300,
    ) -> None:
        self._env = backend_env
        self.instance = instance
        self.workdir = workdir
        self.default_timeout_s = default_timeout_s
        self._closed = False

    # ------------------------------------------------------------------
    # Core shell exec
    # ------------------------------------------------------------------

    def exec(
        self,
        cmd: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
    ) -> ExecResult:
        """Run ``cmd`` (a bash one-liner or multi-line string) inside the
        container. Maps to the underlying backend's ``execute``.

        Args:
            cmd: shell command. Multi-line commands work; the backend wraps in
                ``bash -lc`` (mini_swe_agent convention).
            timeout: seconds; defaults to ``self.default_timeout_s``. On
                timeout the result has ``returncode=124`` and ``timed_out=True``
                (matches the gym spec §4.2.5 convention).
            cwd: override the working directory for this exec (defaults to the
                container's ``cwd``, typically ``/testbed`` for mini_swe_agent).

        Returns:
            ``ExecResult`` (never raises on a non-zero exit code; raises
            :class:`SandboxError` only if the backend itself errors out).
        """
        if self._closed:
            raise SandboxError("Sandbox.exec called after close()")

        timeout = timeout if timeout is not None else self.default_timeout_s

        # Prepend cd if cwd override provided. The mini_swe_agent backend
        # doesn't support a per-call cwd kwarg uniformly, so we synthesize it.
        if cwd:
            cmd = f"cd {shlex.quote(cwd)} && {cmd}"

        import time

        t0 = time.monotonic()
        try:
            # The minisweagent Environment.execute signature is
            # ``execute(cmd: str, timeout: int = None) -> dict``.
            kwargs: Dict[str, Any] = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            raw = self._env.execute(cmd, **kwargs)
        except Exception as e:  # pragma: no cover - depends on backend
            raise SandboxError(f"backend exec failed: {e}") from e
        elapsed = time.monotonic() - t0

        rc = int(raw.get("returncode", -1))
        output = raw.get("output", "") or ""
        timed_out = rc == 124 or "<timeout" in output.lower()

        return ExecResult(returncode=rc, output=output, elapsed_s=elapsed, timed_out=timed_out)

    # ------------------------------------------------------------------
    # File I/O (in-container)
    # ------------------------------------------------------------------

    def read_file(self, path: str, max_bytes: int = 10 * 1024 * 1024) -> str:
        """Read a file from inside the container.

        Uses ``head -c`` to bound the read; large files are truncated and a
        marker appended. Returns the file contents as a string (UTF-8, errors
        replaced).
        """
        # ``head -c N`` is the POSIX-safe truncating read; combine with file
        # existence check so callers can distinguish empty file from missing
        # file via the ExecResult.
        cmd = (
            f"if [ -f {shlex.quote(path)} ]; then "
            f"head -c {int(max_bytes)} {shlex.quote(path)}; "
            f"else echo '__MILO_FILE_NOT_FOUND__' >&2 && exit 2; fi"
        )
        r = self.exec(cmd)
        if r.returncode == 2 and "__MILO_FILE_NOT_FOUND__" in r.output:
            raise FileNotFoundError(path)
        return r.output

    def write_file(self, path: str, content: str, mode: str = "0644") -> ExecResult:
        """Write ``content`` to ``path`` inside the container via heredoc.

        The heredoc delimiter embeds a uuid to avoid collisions with content
        text (same pattern ``mini_swe_utils.evaluate_trajectory`` uses for
        patch application).
        """
        delim = f"MILO_W_{uuid.uuid4().hex}"
        # ``cat`` to a temp path then mv keeps the write atomic from the
        # observer's POV. ``install -m`` sets the mode in one syscall.
        tmp = f"/tmp/.milo_write_{uuid.uuid4().hex}"
        cmd = textwrap.dedent(
            f"""\
            mkdir -p $(dirname {shlex.quote(path)})
            cat <<'{delim}' > {shlex.quote(tmp)}
            {content}
            {delim}
            install -m {mode} {shlex.quote(tmp)} {shlex.quote(path)}
            rm -f {shlex.quote(tmp)}
            """
        )
        return self.exec(cmd)

    # ------------------------------------------------------------------
    # Git helpers (working-tree diff + patch apply)
    # ------------------------------------------------------------------

    def git_diff(self, against: str = "HEAD") -> str:
        """Return the git diff of the working tree against ``against`` (default
        ``HEAD``). Empty string if no changes.

        Used by the generator to populate ``step_record.working_tree_patch``
        per spec §6.4 and to materialize the terminal-time candidate patch fed
        to the verifier (spec §4.4.1).
        """
        r = self.exec(
            f"cd {shlex.quote(self.workdir)} && git --no-pager add -N . >/dev/null 2>&1 && "
            f"git --no-pager diff {shlex.quote(against)}",
            timeout=60,
        )
        # Some images don't have git or the workdir isn't a repo — return ""
        # rather than raising. The verifier sees an empty patch and produces
        # R_terminal=0, which is the right behavior.
        if not r.ok:
            return ""
        return r.output

    def apply_patch(self, diff: str, check_only: bool = False) -> ExecResult:
        """Apply a unified diff via ``git apply``.

        Mirrors what ``mini_swe_utils.evaluate_trajectory`` does, plus a
        ``check_only`` flag so the Phase 6 invariant code can call
        ``git apply --check`` cheaply (I-7).

        Returns the ``ExecResult`` of the ``git apply`` call; non-zero
        returncode means the patch did not apply (the caller decides what to
        do with that).
        """
        if not diff.strip():
            # Empty patch is a no-op success.
            return ExecResult(returncode=0, output="", elapsed_s=0.0, timed_out=False)
        delim = f"PATCH_{uuid.uuid4().hex}"
        flag = "--check" if check_only else ""
        cmd = (
            f"cd {shlex.quote(self.workdir)} && "
            f"git apply {flag} <<'{delim}'\n{diff}\n{delim}"
        )
        return self.exec(cmd, timeout=120)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Tear down the container. Idempotent; multiple calls are safe."""
        if self._closed:
            return
        self._closed = True
        # mini_swe_agent's docker/podman Environment exposes a ``cleanup``
        # method on some backends and just relies on container TTL on others.
        # Try the conventional names; swallow errors (we're already on the
        # rollout teardown path and shouldn't mask the original failure).
        for attr in ("cleanup", "close", "stop", "__del__"):
            fn = getattr(self._env, attr, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception as e:  # pragma: no cover
                    logger.debug("Sandbox.close: %s.%s raised %s; trying next", type(self._env).__name__, attr, e)


# ---------------------------------------------------------------------------
# Context manager — the public entrypoint
# ---------------------------------------------------------------------------


def _build_minisweagent_env(
    instance: Dict[str, Any],
    image: str,
    cwd: str,
    timeout: int,
    pull_timeout: int,
    cpu: int,
    mem_gb: int,
    disk_gb: int,
    executable: str,
) -> Any:
    """Construct the underlying mini_swe_agent Environment.

    Centralized here so the dependency import is lazy (tests that don't need
    a real sandbox can import the module without minisweagent installed). If
    the import fails, raise :class:`SandboxError` with a clear message.
    """
    try:
        # mini_swe_agent ships a factory ``get_environment`` keyed by
        # ``environment_class``. We use the same Docker/podman knobs the
        # ``examples/train/mini_swe_agent/swebench.yaml`` config uses so our
        # sandbox behaves identically to mini_swe_agent's reference setup.
        from minisweagent.environments import get_environment  # type: ignore
    except ImportError as e:
        raise SandboxError(
            "minisweagent is required for milo_sandbox. Install via "
            "`uv sync --extra miniswe` or pin the dep in your env."
        ) from e

    env_config = {
        "environment_class": "docker",
        "image": image,
        "cwd": cwd,
        "timeout": timeout,
        "pull_timeout": pull_timeout,
        "executable": executable,
        # Resource limits — mini_swe_agent accepts these as docker run flags
        # in newer versions; older versions silently ignore unknown keys,
        # which is the desired fallback.
        "cpu": cpu,
        "mem_gb": mem_gb,
        "disk_gb": disk_gb,
        # Tame pagers + progress bars so output stays parseable.
        "env": {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        },
    }
    return get_environment(env_config)


@contextmanager
def milo_sandbox(
    instance: Dict[str, Any],
    cpu: int = 4,
    mem_gb: int = 16,
    disk_gb: int = 40,
    workdir: str = "/testbed",
    timeout_s: int = 300,
    pull_timeout_s: int = 1200,
    executable: str = "docker",
    image_override: Optional[str] = None,
) -> Iterator[Sandbox]:
    """Open a one-shot sandbox for a milo-bench instance.

    Args:
        instance: milo-bench task dict (must carry at minimum ``org``,
            ``repo``, ``number`` for image-name derivation, or an
            ``image_name`` override; see :mod:`milo.lht_adapter.image_naming`).
        cpu: vCPU cap (spec §5.1 default 4).
        mem_gb: RAM cap (spec §5.1 default 16).
        disk_gb: ephemeral disk cap (spec §5.1 default 40).
        workdir: in-container working dir for execs (default ``/testbed`` —
            matches Multi-SWE-bench image convention).
        timeout_s: per-exec default timeout in seconds.
        pull_timeout_s: docker pull timeout (large for the first pull).
        executable: ``docker``, ``podman``, ``singularity``. Defaults to
            ``docker``; integrators on a rootless cluster will override.
        image_override: if set, skip image-name derivation and use this
            literal image (handy for offline tests).

    Yields:
        :class:`Sandbox` instance. The container is torn down on context exit
        even if the body raises.

    Raises:
        SandboxError: if the backend isn't available or the container fails
            to start.
    """
    # 1. Resolve image name.
    image = image_override or instance.get("image_name") or get_image_name(instance)
    logger.info("milo_sandbox: opening container for %s (image=%s)", instance.get("instance_id"), image)

    # 2. Build the backend env.
    backend = _build_minisweagent_env(
        instance=instance,
        image=image,
        cwd=workdir,
        timeout=timeout_s,
        pull_timeout=pull_timeout_s,
        cpu=cpu,
        mem_gb=mem_gb,
        disk_gb=disk_gb,
        executable=executable,
    )

    sandbox = Sandbox(
        backend_env=backend,
        instance=instance,
        workdir=workdir,
        default_timeout_s=timeout_s,
    )
    try:
        yield sandbox
    finally:
        sandbox.close()


# ---------------------------------------------------------------------------
# Convenience: a no-op sandbox for tests that want the interface but not the
# backend. Construct with ``Sandbox(backend_env=FakeBackend(), ...)`` to mock.
# ---------------------------------------------------------------------------


class _NullBackend:
    """In-process backend that just echoes commands. Useful for unit tests.

    Not part of the public API — tests that need this construct it directly:

        s = Sandbox(_NullBackend(), instance={"instance_id": "x"})
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, cmd: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        self.calls.append(cmd)
        return {"returncode": 0, "output": f"(null-backend stub) ran: {cmd[:80]}\n"}

    def cleanup(self) -> None:
        return None


@contextmanager
def null_sandbox(instance: Dict[str, Any], workdir: str = "/testbed") -> Iterator[Sandbox]:
    """Dependency-free sandbox for tests that don't need a real container."""
    backend = _NullBackend()
    sb = Sandbox(backend, instance=instance, workdir=workdir)
    try:
        yield sb
    finally:
        sb.close()


def _write_host_diff_file(diff: str) -> Path:
    """Utility for the converter / verifier: stash a diff on the host fs.

    Not used by Sandbox itself (everything is in-container) — exposed for the
    Phase 5 logger which dumps the candidate patch to the per-rollout log dir
    on the host side.
    """
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False)
    fd.write(diff)
    fd.close()
    return Path(fd.name)
