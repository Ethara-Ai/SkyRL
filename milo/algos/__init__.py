"""Phase 14 — RL algorithm helpers (GRPO loss wrapper, Schulman k3 KL, offline reference logprobs cache).

These are thin wrappers around SkyRL's existing GRPO/FSDP/LoRA workers
in `skyrl/backends/skyrl_train/`. We do not re-implement GRPO; we add the
loss-decomposition logging that the spec §20.2 dashboards expect, the
deterministic Schulman k3 KL estimator, and the v0.7 offline reference
logprobs cache that replaces the 4-H100 live reference vLLM server.
"""
