"""Hot-reload watcher for the policy vLLM server.

Implements ``RL_GYM_SPEC.md`` v0.7 §18.3 and ``IMPLEMENTATION_PLAN.md`` v0.4
§13.1.

The reload protocol (from the spec)::

    1. trainer writes <scratch>/<step>.safetensors
    2. trainer writes <scratch>/READY sentinel (atomic)
    3. serving process detects sentinel, calls vLLM's reload endpoint
    4. serving process deletes the sentinel

v0.7 design choice (plan §14): **full-weight reload only**. vLLM's runtime
``LoadLoRA`` endpoint remains an advanced override but is not the default —
the LoRA-vs-full-FT bake-off (plan §19.1.6) may produce a full-FT winner, in
which case there is no LoRA delta to load anyway, so the safe path is to
publish merged weights and reload.

The watcher is sync + polling-based on purpose: vLLM's runtime weight-load
endpoint already serialises, and the polling cadence (default 5 s) is far
slower than any plausible filesystem-event burst. This avoids a dependency on
``watchdog`` or ``inotify`` (which are not available on every OS the team's
laptops run).

The HTTP call to vLLM uses the stdlib ``urllib`` — no ``requests``
dependency — so the watcher can run inside the bare serving container.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

__all__ = [
    "HotReloadWatcher",
    "ReloadEvent",
    "DEFAULT_SENTINEL_NAME",
    "DEFAULT_RELOAD_ENDPOINT",
]

logger = logging.getLogger(__name__)


DEFAULT_SENTINEL_NAME = "READY"
"""Trainer-side convention; matches plan §13.1 example."""

DEFAULT_RELOAD_ENDPOINT = "/v1/load_weights"
"""vLLM exposes runtime weight reload at this path on recent versions. Older
versions used ``/v1/reload``; override via constructor if running an older
build."""

# A weights file produced by the trainer-side merge step is expected to be
# named ``<step>.safetensors`` (plan §13.1). The integer ``step`` is captured
# in :class:`ReloadEvent.step` for logging + idempotency.
_WEIGHTS_FILENAME_RE = re.compile(r"^(\d+)\.safetensors$")


@dataclass(slots=True)
class ReloadEvent:
    """Recorded each time the watcher hands off to vLLM.

    Persisted (in-memory) on the :class:`HotReloadWatcher` so tests + the
    observability backend can introspect the last N reloads.
    """

    sentinel_path: Path
    weights_path: Path | None
    step: int | None
    detected_at_ts: float
    reload_endpoint: str
    succeeded: bool
    duration_s: float = 0.0
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class HotReloadWatcher:
    """Polls a scratch directory for ``READY`` sentinels and reloads vLLM.

    Parameters
    ----------
    scratch_dir:
        The trainer/serving shared filesystem path (typically NFS). The
        trainer writes the merged ``<step>.safetensors`` here, then the
        sentinel.
    server_url:
        Base URL of the policy vLLM server (e.g. ``"http://127.0.0.1:8000"``).
    sentinel_name:
        Defaults to ``READY``. Override only for tests / advanced setups.
    reload_endpoint:
        Endpoint path on ``server_url``. Default matches recent vLLM
        ``/v1/load_weights`` builds.
    poll_interval_s:
        Seconds between scratch-dir scans. Default ``5.0``.
    max_history:
        Bounded history of :class:`ReloadEvent` entries kept in memory.
    http_post:
        Override for tests — accepts ``(url, payload_dict) -> int (status)``.
        Defaults to a stdlib ``urllib``-backed POST.

    Lifecycle
    ---------
    Construct, then either:

    * call :meth:`run_forever` (blocks; use from a serving sidecar process), or
    * call :meth:`poll_once` from your own event loop (used in tests).

    On each detected sentinel:

    1. Resolve the matching ``<step>.safetensors`` next to the sentinel; if
       multiple weight files exist, pick the highest-numbered. If no weight
       file exists, log an error and leave the sentinel in place (so the
       trainer can investigate and re-publish).
    2. POST to ``{server_url}{reload_endpoint}`` with a JSON body of
       ``{"weights_path": "...", "step": N}``.
    3. On success, **delete** the sentinel. On failure, leave it so the next
       poll cycle retries (the trainer also writes a wall-clock timestamp
       inside the sentinel so we can detect deadlocked retries).
    """

    def __init__(
        self,
        scratch_dir: Path | str,
        server_url: str,
        *,
        sentinel_name: str = DEFAULT_SENTINEL_NAME,
        reload_endpoint: str = DEFAULT_RELOAD_ENDPOINT,
        poll_interval_s: float = 5.0,
        max_history: int = 64,
        http_post: Callable[[str, dict[str, Any]], int] | None = None,
    ) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.server_url = server_url.rstrip("/")
        self.sentinel_name = sentinel_name
        self.reload_endpoint = reload_endpoint
        self.poll_interval_s = float(poll_interval_s)
        self.max_history = max(1, int(max_history))
        self._http_post = http_post or _default_http_post
        self._history: list[ReloadEvent] = []
        self._stop = False

    # ----- introspection ------------------------------------------------

    @property
    def history(self) -> list[ReloadEvent]:
        """All :class:`ReloadEvent` entries seen so far (bounded)."""
        return list(self._history)

    def stop(self) -> None:
        """Cooperative shutdown for :meth:`run_forever`."""
        self._stop = True

    # ----- core polling --------------------------------------------------

    def _record(self, event: ReloadEvent) -> None:
        self._history.append(event)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    def _resolve_weights(self, sentinel: Path) -> tuple[Path | None, int | None]:
        """Look for a sibling ``<step>.safetensors`` file."""
        # If the sentinel itself contains a payload, prefer it.
        try:
            text = sentinel.read_text().strip()
        except OSError:
            text = ""
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("weights_path"), str):
                p = Path(payload["weights_path"])
                step = payload.get("step")
                return p, (int(step) if isinstance(step, int) else None)

        best: tuple[Path, int] | None = None
        for child in sentinel.parent.iterdir():
            m = _WEIGHTS_FILENAME_RE.match(child.name)
            if not m:
                continue
            step = int(m.group(1))
            if best is None or step > best[1]:
                best = (child, step)
        if best is None:
            return None, None
        return best

    def _do_reload(self, weights_path: Path, step: int | None) -> tuple[bool, str | None]:
        """POST the reload request, returning ``(success, error_msg)``."""
        url = self.server_url + self.reload_endpoint
        payload = {"weights_path": str(weights_path), "step": step}
        try:
            status = self._http_post(url, payload)
        except Exception as e:  # noqa: BLE001 — broad on purpose; we want all transport errors
            return False, str(e)
        if 200 <= status < 300:
            return True, None
        return False, f"HTTP {status} from {url}"

    def poll_once(self) -> ReloadEvent | None:
        """Scan ``scratch_dir`` once. Returns the event handled, or ``None``."""
        if not self.scratch_dir.is_dir():
            logger.debug("scratch dir %s does not exist yet — skipping", self.scratch_dir)
            return None

        sentinel = self.scratch_dir / self.sentinel_name
        if not sentinel.is_file():
            return None

        weights_path, step = self._resolve_weights(sentinel)
        detected_at = time.time()

        if weights_path is None:
            event = ReloadEvent(
                sentinel_path=sentinel,
                weights_path=None,
                step=None,
                detected_at_ts=detected_at,
                reload_endpoint=self.reload_endpoint,
                succeeded=False,
                error="no weights file found alongside sentinel",
            )
            self._record(event)
            logger.error(event.error)
            return event

        t0 = time.time()
        ok, err = self._do_reload(weights_path, step)
        t1 = time.time()
        event = ReloadEvent(
            sentinel_path=sentinel,
            weights_path=weights_path,
            step=step,
            detected_at_ts=detected_at,
            reload_endpoint=self.reload_endpoint,
            succeeded=ok,
            duration_s=t1 - t0,
            error=err,
        )
        if ok:
            try:
                sentinel.unlink()
            except OSError as e:
                # The reload itself succeeded — log but don't fail the event.
                logger.warning("failed to remove sentinel %s: %s", sentinel, e)
                event.extras["sentinel_unlink_error"] = str(e)
        else:
            logger.error("hot-reload failed: %s", err)
        self._record(event)
        return event

    def run_forever(self) -> None:
        """Block, polling every ``poll_interval_s`` until :meth:`stop`."""
        self._stop = False
        while not self._stop:
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001
                logger.exception("watcher iteration crashed; continuing")
            # Use a small sleep step so :meth:`stop` is responsive.
            slept = 0.0
            while slept < self.poll_interval_s and not self._stop:
                step = min(0.25, self.poll_interval_s - slept)
                time.sleep(step)
                slept += step


# ---------------------------------------------------------------------------
# Default HTTP transport (stdlib only)
# ---------------------------------------------------------------------------


def _default_http_post(url: str, payload: dict[str, Any]) -> int:
    """POST ``payload`` as JSON to ``url``; return the HTTP status code."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    timeout = float(os.environ.get("MILO_RELOAD_HTTP_TIMEOUT_S", "120"))
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — internal control-plane URL
            return int(getattr(resp, "status", 200))
    except URLError as e:
        # Surface as an HTTP-ish 599 so the caller's success check fails.
        logger.warning("urlopen failed: %s", e)
        return 599
