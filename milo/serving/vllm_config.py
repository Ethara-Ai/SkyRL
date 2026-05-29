"""vLLM configuration defaults + CLI-args helper.

Implements ``RL_GYM_SPEC.md`` v0.7 §18.2 (vLLM defaults table) and
``IMPLEMENTATION_PLAN.md`` v0.4 §13.1.

The dict :data:`MILO_VLLM_DEFAULTS` is the shipped reference recipe. Every
field maps 1:1 to a vLLM CLI flag, which keeps the chain of custody simple:

* dict → CLI args (:func:`make_vllm_args`)
* CLI args → ``vllm.entrypoints.openai.api_server`` argparse
* parsed args → vLLM engine kwargs

Numbers come straight from the spec table; the comments explain the why so
overrides are made with eyes open.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "MILO_VLLM_DEFAULTS",
    "make_vllm_args",
    "DEFAULT_POLICY_MODEL",
]


DEFAULT_POLICY_MODEL = os.environ.get(
    "MILO_POLICY_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
)
"""Default policy model identifier. Overridable via ``MILO_POLICY_MODEL``."""


MILO_VLLM_DEFAULTS: dict[str, Any] = {
    # ---------- Sharding / model -----------------------------------------
    # tensor_parallel: 4 — matches 4 H100s per policy server (spec §18.2).
    "tensor_parallel": 4,
    # max_model_len: 65_536 — ~2× observed median rollout context (spec §18.2).
    "max_model_len": 65_536,
    # dtype: bfloat16 — no quantization; KL needs identical numerics (spec §18.2).
    "dtype": "bfloat16",
    # ---------- Memory / scheduling --------------------------------------
    # gpu_memory_utilization: 0.92 — aggressive but stable for 32B BF16. Drop to
    # 0.85 if CUDA OOMs under bursty load (spec §18.2).
    "gpu_memory_utilization": 0.92,
    # prefix_caching: True — group rollouts share prefixes; saves ~70% prefill
    # cost (spec §18.2). Critical.
    "prefix_caching": True,
    # disable_log_requests: True — rollout logs live in the gym's log dir (§5.5),
    # not vLLM's (spec §18.2).
    "disable_log_requests": True,
    # enforce_eager: False — CUDA graphs on for throughput (spec §18.2).
    "enforce_eager": False,
    # ---------- Auxiliary --------------------------------------------------
    # OpenAI-compat endpoint port the gym's policy adapter dials.
    "port": 8000,
    "host": "0.0.0.0",
    # If set, override the model name in OpenAI API responses (defaults to
    # the value of ``--model``).
    "served_model_name": None,
    # Required for some Qwen variants; off-by-default would block reference.
    "trust_remote_code": True,
    # GiB of CPU swap for vLLM's KV cache spillover.
    "swap_space": 16,
    # Concurrent generation slots: 64-way rollout × 2 to absorb bursts.
    "max_num_seqs": 128,
    # Per-engine-step token budget. Tuned for the 32K context workload.
    "max_num_batched_tokens": 16_384,
    # Long prefills interleave with decodes — important for 30K+ rollouts.
    "enable_chunked_prefill": True,
    # Reproducibility seed; recorded in the manifest (spec §21.2).
    "seed": 42,
}


# Mapping: defaults-dict key → vLLM CLI flag name (only where it differs).
_CLI_FLAG_MAP = {
    "tensor_parallel": "tensor-parallel-size",
    "max_model_len": "max-model-len",
    "gpu_memory_utilization": "gpu-memory-utilization",
    "prefix_caching": ("enable-prefix-caching", "no-enable-prefix-caching"),
    "disable_log_requests": "disable-log-requests",
    "enforce_eager": ("enforce-eager", None),
    "served_model_name": "served-model-name",
    "trust_remote_code": ("trust-remote-code", None),
    "swap_space": "swap-space",
    "max_num_seqs": "max-num-seqs",
    "max_num_batched_tokens": "max-num-batched-tokens",
    "enable_chunked_prefill": ("enable-chunked-prefill", None),
    "dtype": "dtype",
    "port": "port",
    "host": "host",
    "seed": "seed",
}


def _bool_flag(name_pair: tuple[str, str | None], value: bool) -> list[str]:
    """Render a boolean flag.

    ``name_pair = (on_flag, off_flag)``. If ``off_flag`` is None and the value
    is False we simply omit the flag; if both are provided we use the explicit
    on/off forms (some vLLM versions need ``--no-enable-prefix-caching`` to
    disable).
    """
    on, off = name_pair
    if value:
        return [f"--{on}"]
    if off is None:
        return []
    return [f"--{off}"]


def make_vllm_args(model_path: str | None = None, /, **overrides: Any) -> list[str]:
    """Render the vLLM ``api_server`` CLI args list from defaults + overrides.

    Example::

        args = make_vllm_args("/ckpts/sft_v1", gpu_memory_utilization=0.88)
        subprocess.Popen(["python", "-m", "vllm.entrypoints.openai.api_server", *args])

    The first positional argument is the policy model path / HF id. It maps
    to vLLM's required ``--model`` flag. If unset we fall back to
    :data:`DEFAULT_POLICY_MODEL` (env-driven per IMPLEMENTATION_PLAN §22).

    Unknown ``overrides`` are passed through as ``--<key-with-dashes>=<value>``
    for forward compatibility with future vLLM flags (and for the smattering
    of less-common knobs we do not enumerate above).
    """
    model = model_path or DEFAULT_POLICY_MODEL
    args: list[str] = ["--model", model]

    merged: dict[str, Any] = {**MILO_VLLM_DEFAULTS, **overrides}

    for key, value in merged.items():
        if value is None:
            continue
        flag = _CLI_FLAG_MAP.get(key)
        if isinstance(flag, tuple):
            args.extend(_bool_flag(flag, bool(value)))
            continue
        if flag is None:
            flag = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag}")
            # else: omit (vLLM CLI convention)
            continue
        args.append(f"--{flag}")
        args.append(str(value))
    return args
