#!/usr/bin/env bash
# Phase 0.5 end-to-end spike — IMPLEMENTATION_PLAN.md v0.4 §0.5.
#
# Three engineers, three days, ONE milo-bench instance, end-to-end through
# preprocess → rollout → verifier → one GRPO step.
#
# Pass criteria (all four must hold):
#   1. Rollout terminates in <30 min wall clock with a non-error reward.
#   2. Verifier returns a well-formed report ({passed_count, failed_count, ...}).
#   3. GRPO step completes without NaN loss; gradient norm is finite.
#   4. W&B run shows non-zero train/loss_pg and train/loss_kl on at least one batch.
#
# Fail criteria (any one):
#   - Docker pull / container start fails on the spike instance.
#   - Verifier returns empty/malformed report.
#   - mini_swe_agent's get_sb_environment doesn't work against the chosen image
#     without modification (a real risk — mini_swe_agent was built for SWE-Bench,
#     not Multi-SWE-bench).
#   - GRPO step OOMs or NaNs on the spike GPU count.
#
# Adaptability: every infra knob is an environment variable. The default values
# assume a single-node 4×H100 box, but the script will run on smaller hardware
# (with smaller models) if the relevant env vars are overridden. See README.md.

set -euo pipefail
cd "$(dirname "$0")/../.."   # cd to SkyRL repo root

# ---------------- Configurable knobs ----------------

: "${MILO_SPIKE_INSTANCE_JSONL:=/Users/piyush/github/freya/milo-bench/dataset/locustio__locust-1541.jsonl}"
: "${MILO_SPIKE_OUTPUT_DIR:=$PWD/milo/data/spike_v1}"
: "${MILO_SPIKE_IMAGE_NAME:=docker.io/library/python:3.11-slim}"   # placeholder; see BLOCKERS.md
: "${MILO_SPIKE_MODEL:=Qwen/Qwen2.5-Coder-1.5B-Instruct}"          # small for fast spike
: "${MILO_SPIKE_NUM_GPUS:=1}"
: "${MILO_SPIKE_NNODES:=1}"
: "${MILO_SPIKE_NUM_INFERENCE_ENGINES:=1}"
: "${MILO_SPIKE_TP_SIZE:=1}"
: "${MILO_SPIKE_LOGGER:=console}"                                    # use 'wandb' if W&B is configured
: "${MILO_SPIKE_TRAJ_DIR:=$PWD/milo/data/spike_v1/trajectories}"
: "${MILO_SPIKE_CKPT_DIR:=$PWD/milo/data/spike_v1/ckpts}"
: "${MILO_SWE_EXECUTABLE:=docker}"                                    # docker | podman
: "${MILO_LITELLM_MODEL_NAME:=hosted_vllm/$MILO_SPIKE_MODEL}"          # litellm route to local vLLM

: "${MILO_SPIKE_MODE:=all}"      # preprocess | rollout-only | grpo | all
: "${MILO_SPIKE_REPORT:=$PWD/milo/spike/SPIKE_REPORT.md}"

mkdir -p "$MILO_SPIKE_OUTPUT_DIR" "$MILO_SPIKE_TRAJ_DIR" "$MILO_SPIKE_CKPT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------- Stage 1: preprocess ----------------

stage_preprocess() {
  log "Stage 1/3 — preprocess milo jsonl → parquet"
  uv run --isolated --extra dev python -m milo.spike.preprocess_one_milo \
      --milo-jsonl "$MILO_SPIKE_INSTANCE_JSONL" \
      --output-dir "$MILO_SPIKE_OUTPUT_DIR" \
      --image-name "$MILO_SPIKE_IMAGE_NAME"
}

# ---------------- Stage 2: rollout-only smoke ----------------
# Drives mini_swe_agent's generator against a stub inference endpoint (vLLM serving the small model).
# Goal: prove get_sb_environment can stand up the container, run the agent, and call evaluate_trajectory.

stage_rollout_only() {
  log "Stage 2/3 — rollout-only smoke (no training, just generator + verifier)"
  # The smallest invocation is `main_generate` not `main_base` — generates rollouts without an update step.
  # We reuse mini_swe_agent's generator class via a one-off CLI override.

  uv run --isolated --extra fsdp --extra miniswe \
         --env-file examples/train/mini_swe_agent/.env.miniswe \
         -m skyrl.train.entrypoints.main_generate \
      data.val_data="['$MILO_SPIKE_OUTPUT_DIR/validation.parquet']" \
      trainer.policy.model.path="$MILO_SPIKE_MODEL" \
      generator.inference_engine.num_engines="$MILO_SPIKE_NUM_INFERENCE_ENGINES" \
      generator.inference_engine.tensor_parallel_size="$MILO_SPIKE_TP_SIZE" \
      generator.sampling_params.max_generate_length=4096 \
      generator.sampling_params.temperature=0.7 \
      generator.batched=true \
      trainer.placement.colocate_all=true \
      trainer.strategy=fsdp \
      trainer.placement.policy_num_gpus_per_node="$MILO_SPIKE_NUM_GPUS" \
      trainer.placement.policy_num_nodes="$MILO_SPIKE_NNODES" \
      generator.use_conversation_multi_turn=true \
      generator.max_turns=30 \
      trainer.logger="$MILO_SPIKE_LOGGER" \
      +generator.miniswe_config_path="$PWD/milo/spike/milo_swebench.yaml" \
      +generator.miniswe_traj_dir="$MILO_SPIKE_TRAJ_DIR" \
      2>&1 | tee "$MILO_SPIKE_OUTPUT_DIR/rollout.log"
}

# ---------------- Stage 3: one GRPO step ----------------

stage_grpo_one_step() {
  log "Stage 3/3 — one GRPO step against the smallest viable batch"

  uv run --isolated --extra fsdp --extra miniswe \
         --env-file examples/train/mini_swe_agent/.env.miniswe \
         -m examples.train.mini_swe_agent.main_mini_swe \
      data.train_data="['$MILO_SPIKE_OUTPUT_DIR/train.parquet']" \
      data.val_data="['$MILO_SPIKE_OUTPUT_DIR/validation.parquet']" \
      trainer.algorithm.advantage_estimator="grpo" \
      trainer.policy.model.path="$MILO_SPIKE_MODEL" \
      trainer.placement.colocate_all=true \
      trainer.strategy=fsdp \
      trainer.placement.policy_num_gpus_per_node="$MILO_SPIKE_NUM_GPUS" \
      trainer.placement.ref_num_gpus_per_node="$MILO_SPIKE_NUM_GPUS" \
      trainer.placement.policy_num_nodes="$MILO_SPIKE_NNODES" \
      trainer.placement.ref_num_nodes="$MILO_SPIKE_NNODES" \
      generator.inference_engine.num_engines="$MILO_SPIKE_NUM_INFERENCE_ENGINES" \
      generator.inference_engine.tensor_parallel_size="$MILO_SPIKE_TP_SIZE" \
      trainer.epochs=1 \
      trainer.eval_batch_size=1 \
      trainer.eval_before_train=false \
      trainer.eval_interval=99999 \
      trainer.update_epochs_per_batch=1 \
      trainer.train_batch_size=1 \
      trainer.policy_mini_batch_size=1 \
      trainer.micro_forward_batch_size_per_gpu=1 \
      trainer.micro_train_batch_size_per_gpu=1 \
      trainer.ckpt_interval=99999 \
      trainer.max_prompt_length=4096 \
      generator.sampling_params.max_generate_length=4096 \
      generator.n_samples_per_prompt=2 \
      generator.use_conversation_multi_turn=true \
      generator.max_turns=30 \
      generator.batched=true \
      trainer.logger="$MILO_SPIKE_LOGGER" \
      trainer.ckpt_path="$MILO_SPIKE_CKPT_DIR" \
      +generator.miniswe_config_path="$PWD/milo/spike/milo_swebench.yaml" \
      +generator.miniswe_traj_dir="$MILO_SPIKE_TRAJ_DIR" \
      2>&1 | tee "$MILO_SPIKE_OUTPUT_DIR/grpo_step.log"
}

# ---------------- Report ----------------

write_report() {
  local mode_status="$1"
  cat > "$MILO_SPIKE_REPORT" <<EOF
# Phase 0.5 Spike Report

**Generated:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")
**Mode:** $MILO_SPIKE_MODE
**Status:** $mode_status

## Configuration

- Instance: \`$MILO_SPIKE_INSTANCE_JSONL\`
- Image: \`$MILO_SPIKE_IMAGE_NAME\`
- Model: \`$MILO_SPIKE_MODEL\`
- GPUs: $MILO_SPIKE_NUM_GPUS × $MILO_SPIKE_NNODES nodes
- vLLM engines: $MILO_SPIKE_NUM_INFERENCE_ENGINES (TP=$MILO_SPIKE_TP_SIZE)
- Logger: $MILO_SPIKE_LOGGER

## Pass criteria (all four must hold)

- [ ] **(1) Rollout terminates in <30 min wall clock with a non-error reward.**
      Check \`$MILO_SPIKE_OUTPUT_DIR/rollout.log\` for the final reward line.
- [ ] **(2) Verifier returns a well-formed report.**
      Check \`$MILO_SPIKE_TRAJ_DIR/\` for the per-rollout report json.
- [ ] **(3) GRPO step completes without NaN loss; gradient norm is finite.**
      Check \`$MILO_SPIKE_OUTPUT_DIR/grpo_step.log\` for \`train/loss_pg\` and
      \`train/grad_norm\` lines.
- [ ] **(4) W&B (or console logger) shows non-zero train/loss_pg and train/loss_kl
      on at least one batch.**

## Outputs

- Preprocess preview: \`$MILO_SPIKE_OUTPUT_DIR/preview.json\`
- Rollout log: \`$MILO_SPIKE_OUTPUT_DIR/rollout.log\`
- GRPO log: \`$MILO_SPIKE_OUTPUT_DIR/grpo_step.log\`
- Trajectories: \`$MILO_SPIKE_TRAJ_DIR/\`
- Checkpoints: \`$MILO_SPIKE_CKPT_DIR/\`

## Failure-mode → next-step decision tree

Per \`IMPLEMENTATION_PLAN.md\` v0.4 §28.8:

| Failure mode | Recovery |
|---|---|
| Docker harness mismatch | Add 1-week "harness adaptation" mini-phase before Phase 1. |
| Verifier broken on Multi-SWE-bench schema | Fix forward; Phase 2 absorbs extra work (1.5 → 2.5 wk). |
| Compute / OOM on small GPU | Re-cost; sequence length may need reduction. Trigger Phase 0.7 "compute re-baseline." |
| Fundamental incompatibility | Escalate to CEO. Foundation pivot adds ~8 weeks; project-existential. |

## Manual checklist after the script returns

1. Inspect \`rollout.log\` for the final \`reward\` and \`response_length\` numbers.
2. Inspect \`grpo_step.log\` for \`train/loss_pg\`, \`train/loss_kl\`, \`train/grad_norm\`.
3. If any of the four pass-criteria checkboxes is unchecked, classify by the table
   above and update this report manually with the chosen recovery path.
4. Hand the report to CTO at end of Day 3 of the spike.
EOF
  log "Wrote report → $MILO_SPIKE_REPORT"
}

# ---------------- Driver ----------------

main() {
  log "Spike mode: $MILO_SPIKE_MODE"
  log "Output dir: $MILO_SPIKE_OUTPUT_DIR"
  case "$MILO_SPIKE_MODE" in
    preprocess)    stage_preprocess; write_report "PREPROCESS_ONLY (manually check stages 2-3)"; ;;
    rollout-only)  stage_preprocess; stage_rollout_only; write_report "ROLLOUT_ONLY (manually check stage 3)"; ;;
    grpo)          stage_preprocess; stage_grpo_one_step; write_report "GRPO_ONLY (skipped standalone rollout)"; ;;
    all)           stage_preprocess; stage_rollout_only; stage_grpo_one_step; write_report "ALL_STAGES_EXECUTED (verify checkboxes manually)"; ;;
    *)             echo "Unknown MILO_SPIKE_MODE=$MILO_SPIKE_MODE"; exit 2; ;;
  esac
  log "Spike script complete. See $MILO_SPIKE_REPORT."
}

main "$@"
