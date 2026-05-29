"""milo.serving — vLLM serving configuration + hot-reload watcher (Phase 13).

Implements ``IMPLEMENTATION_PLAN.md`` v0.4 §13 / ``RL_GYM_SPEC.md`` v0.7 §18.

Public surface:

* :mod:`milo.serving.vllm_config` — ``MILO_VLLM_DEFAULTS`` dict + the
  ``make_vllm_args(model_path, **overrides)`` CLI-args helper.
* :mod:`milo.serving.hot_reload` — :class:`HotReloadWatcher` for the
  ``<step>.safetensors`` + ``READY`` sentinel handshake (spec §18.3).

v0.7 note: the **reference** vLLM server has been retired in favour of an
offline logprobs cache (see :mod:`milo.algos.reference_cache`). Only the
*policy* server runs live during RL — the watcher and config helpers here all
target that single server.
"""

from __future__ import annotations

__all__ = [
    "MILO_VLLM_DEFAULTS",
    "make_vllm_args",
    "HotReloadWatcher",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in ("MILO_VLLM_DEFAULTS", "make_vllm_args"):
        from milo.serving.vllm_config import MILO_VLLM_DEFAULTS, make_vllm_args

        return {"MILO_VLLM_DEFAULTS": MILO_VLLM_DEFAULTS, "make_vllm_args": make_vllm_args}[name]
    if name == "HotReloadWatcher":
        from milo.serving.hot_reload import HotReloadWatcher

        return HotReloadWatcher
    raise AttributeError(name)
