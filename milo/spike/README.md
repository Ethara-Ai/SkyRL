# Phase 0.5 spike — execution guide

Implements `IMPLEMENTATION_PLAN.md` v0.4 §0.5. **No other phase starts until this returns green.**

## What this spike answers

> *Can we get a single milo-bench instance from raw jsonl, through an SFT-warmstarted
> Qwen2.5-Coder policy, through one GRPO update, with a verifier-grounded reward,
> end-to-end on SkyRL?*

If yes → unblock Phase 1. If no → categorize the failure (see decision tree in `SPIKE_REPORT.md`)
and decide whether to fix forward, re-cost, or escalate to CEO.

## Prerequisites

| Requirement | How to satisfy | Lead time |
|---|---|---|
| SkyRL repo cloned, `uv` working | Already done | — |
| `--extra fsdp --extra miniswe` deps install cleanly | `uv sync --extra fsdp --extra miniswe` | 15 min |
| Docker (or podman) available | `docker --version` | — |
| **Spike Docker image** for `locustio/locust@1b9738c3` | See "Image options" below | hours to days |
| **GPU** (4× preferred, 1× minimum for tiny model) | Hardware reservation | (deferred per user) |
| Optional: W&B API key | `export WANDB_API_KEY=...` | minutes |

## Files

| File | Purpose |
|---|---|
| `preprocess_one_milo.py` | Convert one milo jsonl → parquet in mini_swe_agent format |
| `milo_swebench.yaml` | mini-swe-agent config (system prompt, env, step_limit) |
| `run_spike.sh` | End-to-end driver (3 stages: preprocess → rollout → 1 GRPO step) |
| `SPIKE_REPORT.md` | Template; populated by `run_spike.sh` with pass/fail checkboxes |

## Image options for `MILO_SPIKE_IMAGE_NAME`

This is the single biggest blocker. The spike instance (`locustio__locust-1541`) needs
a Docker image with the locust repo checked out at `tag_start` (commit `1b9738c3`) plus
its Python deps. Three options, fastest to slowest:

1. **Pull from ECR if the milo-bench team has already built it.** Check:
   `aws ecr describe-images --repository-name mswebench --region ap-south-1 --image-ids imageTag=locustio_m_locust:1541 --profile <your-profile>`.
   Then set `MILO_SPIKE_IMAGE_NAME=<account>.dkr.ecr.ap-south-1.amazonaws.com/mswebench:locustio_m_locust-1541`
   and `docker login` to ECR.

2. **Build locally from `benchmarks/multiswebench/build_images.py`** (in the freya repo).
   Slow — multi-GB image, dep resolution, but reproducible.

3. **Use a stand-in `python:3.11-slim` for harness validation only** (the default in
   `run_spike.sh`). This proves the *pipeline* works but `evaluate_trajectory` will fail
   because the locust repo isn't present in the container. Useful to validate
   stages 1–2 (preprocess + rollout dispatch) before sinking ECR auth time.

   Mark the spike "PARTIAL PASS" if this option is chosen; the verifier check
   only completes meaningfully with options 1 or 2.

## Running it (full spike)

```bash
# minimum env
export MILO_SPIKE_IMAGE_NAME='<your-image-name>'
export WANDB_API_KEY='<optional, omit to use console logger>'

# defaults: locustio__locust-1541, Qwen2.5-Coder-1.5B-Instruct, 1 GPU
bash milo/spike/run_spike.sh
```

## Running on smaller hardware

```bash
# CPU-only preprocess sanity check (no GPU needed)
MILO_SPIKE_MODE=preprocess bash milo/spike/run_spike.sh

# Single GPU, tiny model (sufficient for spike validation)
MILO_SPIKE_NUM_GPUS=1 \
MILO_SPIKE_MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct \
MILO_SPIKE_NUM_INFERENCE_ENGINES=1 \
MILO_SPIKE_TP_SIZE=1 \
bash milo/spike/run_spike.sh
```

## Running on larger hardware (matches plan §19 reference run)

```bash
# 8 GPUs, single node, 32B model
MILO_SPIKE_NUM_GPUS=8 \
MILO_SPIKE_NNODES=1 \
MILO_SPIKE_NUM_INFERENCE_ENGINES=4 \
MILO_SPIKE_TP_SIZE=2 \
MILO_SPIKE_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
bash milo/spike/run_spike.sh
```

## Adaptability for client infra

Every infra knob is an env var. The spike runs on:

- **1 GPU** with Qwen2.5-Coder-0.5B (fastest validation; ~$20 of compute on a spot H100)
- **8 GPUs** with Qwen2.5-Coder-32B (full reference-scale validation; ~$200)
- **podman instead of docker**: `MILO_SWE_EXECUTABLE=podman`
- **K8s instead of single-node**: not yet wired; see Phase 18 in the plan
- **Different cloud / no AWS**: image name is an env var; checkpoint dir is an env var; nothing AWS-specific in the spike code

## Pass / fail interpretation

After `run_spike.sh` returns, manually check the four pass criteria in `SPIKE_REPORT.md`.
Tick the checkboxes; the report is yours to update.

- **All four ticked** → spike is green. Update `SPIKE_REPORT.md` status to `PASS`, hand to
  CTO/CEO, unblock Phase 1.
- **One or more unticked** → use the decision tree at the bottom of `SPIKE_REPORT.md` to
  classify the failure and pick a recovery path.

## Known limitations of this scaffold

1. The `eval_script` synthesized by `preprocess_one_milo.py` is a Phase-0.5-grade
   approximation. The real verifier (Phase 2, in `milo/verifier/three_run.py`) implements
   the 3-Docker-run pattern with proper invariant guards (I-1..I-8). For the spike, the
   eval is "apply test_patch + model patch, run F2P tests, expect them to pass."
2. The script uses `examples.train.mini_swe_agent.main_mini_swe` directly. Phase 1 will
   replace this with `milo.lht_adapter.main_milo` so the agent loop is Milo-aware
   (multi-PR bundles, OpenHands-shaped traces, our reward decomposition).
3. The reward computed during the spike is binary (pass/fail of F2P tests). The composite
   reward (`R_total = R_terminal + α·Σ R_delta + β·R_rubric + γ·Σ R_tir`) ships in Phase 4.
4. No invariant checks. Phase 6 adds those.

All four limitations are deliberate — the spike answers "can the pipeline run end-to-end?"
not "does the full system work as specified?" The latter is the rest of the plan.
