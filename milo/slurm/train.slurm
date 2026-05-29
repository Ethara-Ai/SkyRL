#!/bin/bash
#SBATCH --job-name=milo-rl
#SBATCH --output=logs/milo-rl-%j.out
#SBATCH --error=logs/milo-rl-%j.err
#SBATCH --time=120:00:00         # 5 days; long jobs requeue cleanly (plan §28.2)
#SBATCH --nodes=${MILO_NNODES:-2}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:${MILO_GPUS_PER_NODE:-8}
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail

# Phase 14 + 19.2 — RL reference run.
#
# Defaults match the v0.4 §19.2 envelope:
#   - 16 H100s (2 nodes × 8 GPUs)
#   - β_KL = 0.01 → 0.005 over 4800 steps (v0.4 default — NOT 0.04)
#   - GRPO with milo_grpo policy loss (decomposition logging)
#   - Reference logprobs from offline S3 cache (v0.7 §18 — no live ref server)
#   - LoRA OR full FT per the bake-off winner (set by MILO_POLICY_METHOD)

: "${MILO_RL_CONFIG:?MILO_RL_CONFIG is required}"
: "${MILO_DATA_DIR:?MILO_DATA_DIR is required}"
: "${MILO_SFT_CKPT:?MILO_SFT_CKPT is required}"
: "${MILO_CKPT_DIR:?MILO_CKPT_DIR is required}"
: "${MILO_NNODES:=2}"
: "${MILO_GPUS_PER_NODE:=8}"
: "${MILO_BASE_MODEL:=Qwen/Qwen2.5-Coder-32B-Instruct}"
: "${MILO_POLICY_METHOD:=lora}"         # lora | full_ft (Phase 19.1.6 bake-off winner)
: "${MILO_LOGGER:=wandb}"
: "${MILO_BETA_KL_INITIAL:=0.01}"       # v0.4 default
: "${MILO_BETA_KL_FINAL:=0.005}"
: "${MILO_BETA_KL_ANNEAL_STEPS:=4800}"

cd "${MILO_REPO_ROOT:-$(pwd)}"

# Ensure the offline reference logprobs cache exists.
if [[ ! -f "$MILO_REF_CACHE" ]]; then
  echo "[$(date)] Building offline reference logprobs cache → $MILO_REF_CACHE"
  uv run --isolated --extra fsdp --extra miniswe python -m milo.algos.reference_cache_build \
      --sft-checkpoint "$MILO_SFT_CKPT" \
      --prompts "$MILO_DATA_DIR/train_prompts.jsonl" \
      --out "$MILO_REF_CACHE"
fi

echo "[$(date)] Starting Milo RL run"
echo "  config=$MILO_RL_CONFIG  policy_method=$MILO_POLICY_METHOD"
echo "  β_KL: $MILO_BETA_KL_INITIAL → $MILO_BETA_KL_FINAL over $MILO_BETA_KL_ANNEAL_STEPS steps"

srun uv run --isolated --extra fsdp --extra miniswe -m milo.lht_adapter.main_milo \
    data.train_data="['$MILO_DATA_DIR/train.parquet']" \
    data.val_data="['$MILO_DATA_DIR/validation.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.policy_loss_type=milo_grpo \
    trainer.policy.model.path="$MILO_BASE_MODEL" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node="$MILO_GPUS_PER_NODE" \
    trainer.placement.policy_num_nodes="$MILO_NNODES" \
    +trainer.algorithm.beta_kl_initial="$MILO_BETA_KL_INITIAL" \
    +trainer.algorithm.beta_kl_final="$MILO_BETA_KL_FINAL" \
    +trainer.algorithm.beta_kl_anneal_steps="$MILO_BETA_KL_ANNEAL_STEPS" \
    +trainer.policy.method="$MILO_POLICY_METHOD" \
    trainer.ckpt_path="$MILO_CKPT_DIR" \
    trainer.ckpt_interval=50 \
    trainer.logger="$MILO_LOGGER"

echo "[$(date)] RL slurm job complete"
